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

from threading import Event, Lock

import zmq

from transfer_queue.storage.payload_transfer.base import (
    DeliveryState,
    Operation,
    PayloadPeerInfo,
    PayloadReply,
    PayloadTransfer,
    RequestHeader,
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
        self._closed = Event()
        self._handler_lock = Lock()

    def listen(self, *, handle, address=None) -> PayloadPeerInfo:
        """Bind the local storage handler and return ZMQ server bootstrap metadata."""
        self._handle = handle
        return PayloadPeerInfo("zmq", address)

    async def request(self, peer_info, header, *, send=None, recv=None):
        """Lease a pooled DEALER socket for ``peer_info``, send ``header``, and decode its reply."""
        if self._closed.is_set() or self.pool is None:
            raise TransferError("ZMQ payload client is unavailable", transient=False, delivery=DeliveryState.UNSENT)
        if not isinstance(peer_info, PayloadPeerInfo) or peer_info.backend != "zmq":
            raise TransferError("invalid ZMQ payload peer", transient=False, delivery=DeliveryState.UNSENT)
        address = peer_info.address
        if not isinstance(address, ZMQServerInfo):
            raise TransferError("invalid ZMQ payload peer", transient=False, delivery=DeliveryState.UNSENT)
        if recv is not None:
            raise NotImplementedError("ZMQ receive destinations are not implemented")
        if header.operation == Operation.GET:
            if send is not None:
                raise ValueError("GET does not accept send field data")
        elif send is None:
            raise ValueError(f"{header.operation.value} requires send field data")

        request_type, response_type, error_type = _WIRE_TYPES[header.operation]
        wire_bytes = 0
        delivery = DeliveryState.UNSENT
        try:
            async with self.pool.alease(address) as socket:
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
                delivery = DeliveryState.IN_DOUBT
                await socket.send_multipart(frames, copy=False)
                del frames
                reply = ZMQMessage.deserialize(await socket.recv_multipart(copy=False))
                if reply.request_type not in (response_type, error_type):
                    detail = reply.body.get("message") if isinstance(reply.body, dict) else None
                    reason = f"unexpected response {reply.request_type}" + (f": {detail}" if detail else "")
                    rejected_at_ingress = reply.request_type == ZMQRequestType.PUT_GET_OPERATION_ERROR or (
                        reply.request_type == ZMQRequestType.PUT_GET_ERROR
                        and isinstance(detail, str)
                        and detail.startswith("undecodable request:")
                    )
                    raise TransferError(
                        reason,
                        transient=False,
                        delivery=DeliveryState.REJECTED if rejected_at_ingress else DeliveryState.IN_DOUBT,
                    )
                reply_meta = dict(reply.body)
                payload = None
                if reply.request_type == response_type and header.operation == Operation.GET:
                    payload = reply_meta.pop("data", None)
                    if not isinstance(payload, dict):
                        raise TransferError(
                            "GET response must contain field data",
                            transient=False,
                            delivery=DeliveryState.IN_DOUBT,
                        )
        except zmq.error.Again as exc:
            raise TransferTimeout("ZMQ storage request timed out", wire_bytes=wire_bytes, delivery=delivery) from exc
        except zmq.ZMQError as exc:
            raise TransferError(
                f"ZMQ storage request failed: {type(exc).__name__}: {exc}",
                transient=exc.errno != zmq.ETERM,
                delivery=delivery,
            ) from exc
        except ZMQMessageDecodeError as exc:
            raise TransferError(
                f"ZMQ storage request failed: {type(exc).__name__}: {exc}",
                transient=False,
                delivery=delivery,
            ) from exc

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

        The borrowed ``ZMQSocketPool`` is shared with administrative RPCs (e.g. ``CLEAR_DATA``)
        and is closed by ``AsyncSimpleStorageManager.close()`` rather than here.
        """
        self._closed.set()
        with self._handler_lock:
            self._handle = None
