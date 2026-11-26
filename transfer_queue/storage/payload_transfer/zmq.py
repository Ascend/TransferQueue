# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapter for the existing SimpleStorage multipart protocol."""

from __future__ import annotations

import errno
from threading import Event, Lock

import zmq

from transfer_queue.storage.payload_transfer.base import (
    Operation,
    PayloadPeerInfo,
    PayloadReply,
    PayloadTransfer,
    RequestHeader,
    StorageHandler,
)
from transfer_queue.storage.payload_transfer.errors import (
    StorageRejected,
    TransferError,
    TransferTimeout,
)
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQMessageDecodeError,
    ZMQRequestType,
    ZMQServerInfo,
    ZMQSocketPool,
    frame_nbytes,
)

# Preserves the existing ZMQ GET_ERROR message convention across rolling upgrades.
KEY_NOT_FOUND_MARKER = "TQKeyNotFound"

# libzmq handles TCP connect/reconnect asynchronously in its C++ I/O thread (peer drops surface as
# zmq.Again -> TransferTimeout). Python-side ZMQError only merits a retry on transient local
# resource exhaustion or interrupted syscalls; EINVAL/ETERM/EPROTONOSUPPORT fail immediately.
_TRANSIENT_ZMQ_ERRNOS = frozenset(
    {
        errno.EAGAIN,
        errno.EINTR,
        errno.EMFILE,
        errno.ENFILE,
        errno.ENOBUFS,
        errno.ENOMEM,
    }
)

# Maps each logical Operation to its ZMQ (request, response, error) message types.
_WIRE_TYPES = {
    Operation.PUT: (ZMQRequestType.PUT_DATA, ZMQRequestType.PUT_DATA_RESPONSE, ZMQRequestType.PUT_ERROR),
    Operation.GET: (ZMQRequestType.GET_DATA, ZMQRequestType.GET_DATA_RESPONSE, ZMQRequestType.GET_ERROR),
    Operation.UPDATE: (ZMQRequestType.UPDATE_DATA, ZMQRequestType.UPDATE_DATA_RESPONSE, ZMQRequestType.UPDATE_ERROR),
}
_OPERATIONS_BY_REQUEST = {req_type: op for op, (req_type, _, _) in _WIRE_TYPES.items()}


class ZmqPayloadTransfer(PayloadTransfer):
    """PayloadTransfer adapter over ZMQ multipart frames and pooled DEALER sockets."""

    @staticmethod
    def can_handle(request_type: ZMQRequestType) -> bool:
        """Return whether ``request_type`` is a ZMQ payload operation handled by this adapter."""
        return request_type in _OPERATIONS_BY_REQUEST

    def __init__(self, *, owner_id: str, pool: ZMQSocketPool | None = None):
        self.owner_id = owner_id
        self.pool = pool
        self._handle: StorageHandler | None = None
        self._closed = Event()
        self._handler_lock = Lock()

    def listen(self, *, handle: StorageHandler, address=None) -> PayloadPeerInfo:
        """Attach ``handle`` to the host unit's shared ZMQ listener described by ``address``."""
        self._handle = handle
        return PayloadPeerInfo("zmq", address)

    async def request(self, peer_info, header, *, send=None):
        """Lease a pooled DEALER socket for ``peer_info``, send ``header``, and decode its reply."""
        if self._closed.is_set() or self.pool is None:
            raise TransferError("ZMQ payload client is unavailable")
        if not isinstance(peer_info, PayloadPeerInfo) or peer_info.backend != "zmq":
            raise TransferError("invalid ZMQ payload peer")
        address = peer_info.address
        if not isinstance(address, ZMQServerInfo):
            raise TransferError("invalid ZMQ payload peer")
        if header.operation == Operation.GET:
            if send is not None:
                raise ValueError("GET does not accept send field data")
        elif send is None:
            raise ValueError(f"{header.operation.value} requires send field data")

        request_type, response_type, error_type = _WIRE_TYPES[header.operation]
        wire_bytes = 0
        reply = None
        try:
            async with self.pool.alease(address) as socket:
                # Exit the lease normally if closed while queued so the untouched socket stays pooled.
                if not self._closed.is_set():
                    # Serialize only after acquiring a socket permit so queued callers keep zero-copy views.
                    body = dict(header.params)
                    if send is not None:
                        body["data"] = send
                    request = ZMQMessage.create(
                        request_type=request_type,
                        sender_id=self.owner_id,
                        receiver_id=address.id,
                        body=body,
                    )
                    frames = request.serialize()
                    if send is not None:
                        wire_bytes = sum(frame_nbytes(frame) or 0 for frame in frames)
                    await socket.send_multipart(frames, copy=False)
                    del frames
                    raw_reply = await socket.recv_multipart(copy=False)
                    if header.operation == Operation.GET:
                        wire_bytes = sum(frame_nbytes(frame) or 0 for frame in raw_reply)
                    reply = ZMQMessage.deserialize(raw_reply)
                    del raw_reply
        except zmq.error.Again as exc:
            raise TransferTimeout("ZMQ storage request timed out", wire_bytes=wire_bytes) from exc
        except zmq.ZMQError as exc:
            raise TransferError(
                f"ZMQ storage request failed: {type(exc).__name__}: {exc}",
                transient=exc.errno in _TRANSIENT_ZMQ_ERRNOS,
            ) from exc
        except ZMQMessageDecodeError as exc:
            raise TransferError(f"ZMQ storage request failed: {type(exc).__name__}: {exc}") from exc

        if reply is None:
            raise TransferError("ZMQ payload client is unavailable")
        if reply.request_type not in (response_type, error_type):
            reason = f"unexpected response {reply.request_type}"
            if isinstance(reply.body, dict) and (detail := reply.body.get("message")):
                reason += f": {detail}"
            raise TransferError(reason)
        reply_meta = dict(reply.body)
        payload = None
        if reply.request_type == response_type and header.operation == Operation.GET:
            payload = reply_meta.pop("data", None)
            if not isinstance(payload, dict):
                raise TransferError("GET response must contain field data")

        # A valid storage rejection completes the exchange and leaves the socket reusable in the pool.
        if reply.request_type == error_type:
            message = reply_meta.get("message", "storage request rejected")
            prefix = f"[{KEY_NOT_FOUND_MARKER}] "
            missing_key = header.operation == Operation.GET and message.startswith(prefix)
            if missing_key:
                message = message.removeprefix(prefix)
            raise StorageRejected(message, missing_key=missing_key)
        return PayloadReply(meta=reply_meta, payload=payload, wire_bytes=wire_bytes)

    def handle_request(self, message):
        """Admit storage work at worker dispatch. Frames still queued at shutdown may time out."""
        operation = _OPERATIONS_BY_REQUEST[message.request_type]
        _, response_type, error_type = _WIRE_TYPES[operation]
        params = dict(message.body)
        data = params.pop("data", None)
        try:
            with self._handler_lock:
                if self._closed.is_set():
                    raise StorageRejected("storage unit is shutting down")
                if self._handle is None:
                    raise StorageRejected("storage payload handler is not registered")
                reply = self._handle(RequestHeader(operation, params), data)
            result = dict(reply.meta)
            if reply.payload is not None:
                result["data"] = reply.payload
            reply_type = response_type
        except StorageRejected as exc:
            marker = f"[{KEY_NOT_FOUND_MARKER}] " if exc.missing_key else ""
            result = {"message": f"{marker}{exc}"}
            reply_type = error_type
        return ZMQMessage.create(request_type=reply_type, sender_id=self.owner_id, body=result)

    def shutdown(self):
        """Stop admitting new payload work and wait for any active handler call to finish.

        Both the server-side ROUTER socket (owned by ``SimpleStorageUnit``) and the
        client-side ``ZMQSocketPool`` (shared with administrative RPCs such as ``CLEAR_DATA``)
        are closed by their respective owners after ``shutdown()`` returns.
        """
        self._closed.set()
        with self._handler_lock:
            self._handle = None
