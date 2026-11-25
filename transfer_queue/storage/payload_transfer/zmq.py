# Copyright 2026 The TransferQueue Team
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
    Operation,
    PayloadTransfer,
    ReplyHeader,
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

# Maps each logical Operation to its ZMQ (request, response, error) message types.
_WIRE_TYPES = {
    Operation.PUT: (ZMQRequestType.PUT_DATA, ZMQRequestType.PUT_DATA_RESPONSE, ZMQRequestType.PUT_ERROR),
    Operation.GET: (ZMQRequestType.GET_DATA, ZMQRequestType.GET_DATA_RESPONSE, ZMQRequestType.GET_ERROR),
    Operation.UPDATE: (ZMQRequestType.UPDATE_DATA, ZMQRequestType.UPDATE_DATA_RESPONSE, ZMQRequestType.UPDATE_ERROR),
}
_OPERATIONS_BY_REQUEST = {req_type: op for op, (req_type, _, _) in _WIRE_TYPES.items()}


def is_zmq_payload_request(request_type: ZMQRequestType) -> bool:
    """Return whether ``request_type`` is a ZMQ data-plane payload operation."""
    return request_type in _OPERATIONS_BY_REQUEST


class ZmqPayloadTransfer(PayloadTransfer):
    """PayloadTransfer adapter over ZMQ multipart frames and pooled DEALER sockets."""

    def __init__(self, *, owner_id: str, pool: ZMQSocketPool | None = None):
        self.owner_id = owner_id
        self.pool = pool
        self._closed = Event()
        self._handler_lock = Lock()

    def listen(self, *, handle, address=None):
        """Bind the local storage handler and return ZMQ server bootstrap metadata."""
        self._handle = handle
        return {"backend": "zmq", "address": address}

    async def connect(self, peer_info):
        """Validate a ZMQ peer descriptor and return its ``ZMQServerInfo`` handle."""
        if peer_info.get("backend") != "zmq" or not isinstance(peer_info.get("address"), ZMQServerInfo):
            raise TransferError("invalid ZMQ payload peer")
        return peer_info["address"]

    async def request(self, connection, header, *, send=None, recv=None):
        """Send a multipart request over a pooled ZMQ socket and decode its reply."""
        if self._closed.is_set() or self.pool is None:
            raise TransferError("ZMQ payload client is unavailable")
        if recv is not None:
            raise NotImplementedError("ZMQ receive destinations are not implemented")
        if header.operation == Operation.GET:
            if send is not None:
                raise ValueError("GET does not accept send field data")
        elif send is None:
            raise ValueError(f"{header.operation.value} requires send field data")

        request_type, response_type, error_type = _WIRE_TYPES[header.operation]
        body = dict(header.body)
        if send is not None:
            body["data"] = send
        request = ZMQMessage.create(
            request_type=request_type,
            sender_id=self.owner_id,
            receiver_id=connection.id,
            body=body,
        )
        frames = request.serialize()
        wire_bytes = sum(frame_nbytes(frame) or 0 for frame in frames) if send is not None else 0
        try:
            async with self.pool.alease(connection) as socket:
                await socket.send_multipart(frames, copy=False)
                reply = ZMQMessage.deserialize(await socket.recv_multipart(copy=False))
                if reply.request_type not in (response_type, error_type):
                    reason = f"unexpected response {reply.request_type}"
                    if isinstance(reply.body, dict) and (detail := reply.body.get("message")):
                        reason += f": {detail}"
                    raise TransferError(reason)
                body = dict(reply.body)
                data = None
                if reply.request_type == response_type and header.operation == Operation.GET:
                    data = body.pop("data", None)
                    if not isinstance(data, dict):
                        raise TransferError("GET response must contain field data")
        except zmq.error.Again as exc:
            raise TransferTimeout("ZMQ storage request timed out", wire_bytes=wire_bytes) from exc
        except (zmq.ZMQError, ZMQMessageDecodeError) as exc:
            raise TransferError(f"ZMQ storage request failed: {type(exc).__name__}: {exc}") from exc

        # A valid storage rejection completes the exchange and leaves the socket reusable.
        if reply.request_type == error_type:
            message = body.get("message", "storage request rejected")
            missing_key = header.operation == Operation.GET and bool(body.get("missing_key"))
            raise StorageRejected(message, missing_key=missing_key)
        return ReplyHeader(body, wire_bytes), data

    def handle_request(self, message):
        """Admit storage work at worker dispatch. Frames still queued at shutdown may time out."""
        operation = _OPERATIONS_BY_REQUEST[message.request_type]
        _, response_type, error_type = _WIRE_TYPES[operation]
        body = dict(message.body)
        data = body.pop("data", None)
        try:
            with self._handler_lock:
                if self._closed.is_set():
                    raise StorageRejected("storage unit is shutting down")
                result, payload = self._handle(RequestHeader(operation, body), data)
            if payload is not None:
                result = {**result, "data": payload}
            reply_type = response_type
        except StorageRejected as exc:
            result = {"message": str(exc), "missing_key": exc.missing_key}
            reply_type = error_type
        return ZMQMessage.create(request_type=reply_type, sender_id=self.owner_id, body=result)

    def shutdown(self):
        """Mark the transport closed and wait for any active handler call to finish."""
        self._closed.set()
        # The worker may still be accessing storage when admission closes.
        # Release its callback only after it has returned.
        with self._handler_lock:
            self._handle = None
