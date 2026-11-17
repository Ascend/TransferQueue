"""Adapter for the existing SimpleStorage multipart protocol."""

from __future__ import annotations

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
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo, ZMQSocketPool, frame_nbytes

KEY_NOT_FOUND_MARKER = "TQKeyNotFound"

REQUEST_TYPES = {
    Operation.PUT: ZMQRequestType.PUT_DATA,
    Operation.GET: ZMQRequestType.GET_DATA,
    Operation.UPDATE: ZMQRequestType.UPDATE_DATA,
}
RESPONSE_TYPES = {
    Operation.PUT: ZMQRequestType.PUT_DATA_RESPONSE,
    Operation.GET: ZMQRequestType.GET_DATA_RESPONSE,
    Operation.UPDATE: ZMQRequestType.UPDATE_DATA_RESPONSE,
}
ERROR_TYPES = {
    Operation.PUT: ZMQRequestType.PUT_ERROR,
    Operation.GET: ZMQRequestType.GET_ERROR,
    Operation.UPDATE: ZMQRequestType.UPDATE_ERROR,
}


class ZmqPayloadTransfer(PayloadTransfer):
    def __init__(self, *, owner_id: str, pool: ZMQSocketPool | None = None):
        self.owner_id = owner_id
        self.pool = pool
        self._closed = False

    def listen(self, *, handle, address=None):
        self._handle = handle
        return {"backend": "zmq", "address": address}

    async def connect(self, peer_info):
        if peer_info.get("backend") != "zmq" or not isinstance(peer_info.get("address"), ZMQServerInfo):
            raise TransferError("invalid ZMQ payload peer")
        return peer_info["address"]

    async def request(self, connection, header, *, send=None, recv=()):
        if self._closed or self.pool is None:
            raise TransferError("ZMQ payload client is unavailable")
        if recv:
            raise NotImplementedError("ZMQ receive destinations are not implemented")
        if header.operation == Operation.GET:
            if send is not None or recv is not None:
                raise TransferError("GET requires send=None and recv=None")
        elif send is None or recv != ():
            raise TransferError("PUT and UPDATE require input data and recv=()")
        body = dict(header.body)
        if send is not None:
            body["data"] = send
        request = ZMQMessage.create(
            request_type=REQUEST_TYPES[header.operation],
            sender_id=self.owner_id,
            receiver_id=connection.id,
            body=body,
        )
        wire_bytes = 0
        try:
            async with self.pool.alease(connection) as socket:
                frames = request.serialize()
                if send is not None:
                    wire_bytes = sum(frame_nbytes(frame) or 0 for frame in frames)
                await socket.send_multipart(frames, copy=False)
                reply = ZMQMessage.deserialize(await socket.recv_multipart(copy=False))
        except zmq.error.Again as exc:
            raise TransferTimeout("ZMQ storage request timed out", wire_bytes=wire_bytes) from exc

        if reply.request_type != RESPONSE_TYPES[header.operation]:
            message = reply.body.get("message", f"unexpected response {reply.request_type}")
            raise StorageRejected(message, missing_key=KEY_NOT_FOUND_MARKER in message)
        body = dict(reply.body)
        if header.operation == Operation.GET:
            data = body.pop("data", None)
            if not isinstance(data, dict):
                raise TransferError("GET response must contain field data")
            return ReplyHeader(body, wire_bytes), data
        return ReplyHeader(body, wire_bytes), None

    def handle_legacy(self, message):
        """Translate the legacy wire message at the storage boundary."""
        operation = Operation(message.request_type.name)
        body = dict(message.body)
        data = body.pop("data", None)
        try:
            result, payload = self._handle(RequestHeader(operation, body), data)
            if payload is not None:
                result = {**result, "data": payload}
            response_type = RESPONSE_TYPES[operation]
        except StorageRejected as exc:
            marker = f"[{KEY_NOT_FOUND_MARKER}] " if exc.missing_key else ""
            result = {"message": f"{marker}{exc}"}
            response_type = ERROR_TYPES[operation]
        return ZMQMessage.create(request_type=response_type, sender_id=self.owner_id, body=result)

    def shutdown(self):
        # The manager also uses this pool for administrative requests.
        self._closed = True
