"""NIXL notifications carry requests. Receivers pull published snapshots with READ."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from uuid import uuid4

from transfer_queue.storage.payload_transfer.base import (
    Operation,
    PayloadTransfer,
    ReplyHeader,
    RequestHeader,
    StorageRejected,
    TransferError,
    TransferTimeout,
    WriteOutcomeUnknown,
    decode_spans,
    encode_spans,
    place_reply,
    validate_request,
)
from transfer_queue.storage.payload_transfer.nixl_ucx_runtime import NixlRuntime, ReadAccessUnknown
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.serial_utils import calc_packed_size, decode, encode, pack_into, unpack_from

logger = get_logger(__name__)
MAX_CONTROL_BYTES = 1024**2
PROTOCOL_VERSION = 1


class NixlPayloadTransfer(PayloadTransfer):
    poll_interval_ms = 1

    def __init__(self, *, ucx_env=None, timeout=200):
        self._timeout = timeout
        self._runtime = NixlRuntime(ucx_env, timeout=min(timeout, 180))
        self._endpoint = self._runtime.endpoint()
        self._handle = None
        self._lock = threading.RLock()
        self._closed = False
        self._connections = {}
        self._waiters: dict[str, tuple[str, Future]] = {}
        self._exports: dict[tuple[str, str], int] = {}
        self._active_reads: set[tuple[str, str]] = set()
        self._reads = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tq-nixl-read")

    def listen(self, *, handle, address=None):
        self._handle = handle
        return {"backend": "nixl-ucx", "version": PROTOCOL_VERSION, **self._endpoint}

    async def connect(self, peer_info):
        if (
            peer_info.get("backend") != "nixl-ucx"
            or peer_info.get("version") != PROTOCOL_VERSION
            or not isinstance(peer_info.get("agent"), str)
            or not isinstance(peer_info.get("metadata"), bytes)
        ):
            raise TransferError("invalid NIXL payload peer or protocol version")
        with self._lock:
            if self._closed:
                raise TransferError("NIXL payload transport is closed")
            self._runtime.connect(peer_info)
            self._connections[peer_info["agent"]] = peer_info
        return peer_info

    @staticmethod
    def _pack_control(message):
        frames = encode({"version": PROTOCOL_VERSION, **message})
        size = calc_packed_size(frames)
        if size > MAX_CONTROL_BYTES:
            raise TransferError("NIXL control notification exceeds 1 MiB")
        packed = bytearray(size)
        pack_into(packed, frames)
        return bytes(packed)

    def _publish(self, peer, request_id, spans):
        key = (peer, request_id)
        if key in self._exports:
            raise TransferError("duplicate NIXL buffer offer")
        size = calc_packed_size(spans)
        offer = self._runtime.publish(size, lambda buffer: pack_into(buffer, spans))
        self._exports[key] = offer["address"]
        return offer

    def _release_export(self, peer, request_id):
        address = self._exports.pop((peer, request_id), None)
        if address is not None:
            self._runtime.release_export(address)

    def _release_response(self, peer, request_id):
        try:
            self._runtime.send_notification(peer, self._pack_control({"kind": "release", "id": request_id}))
        except Exception:
            # The exporter retains the snapshot when release cannot be delivered.
            logger.exception("failed to release NIXL response %s", request_id)

    def _read_response(self, peer, request_id, offer):
        safe_to_release = True
        try:
            return tuple(unpack_from(self._runtime.read(peer, offer)))
        except ReadAccessUnknown:
            safe_to_release = False
            raise
        finally:
            if safe_to_release:
                self._release_response(peer, request_id)
                with self._lock:
                    self._active_reads.discard((peer["agent"], request_id))

    async def request(self, connection, header, *, send=(), recv=()):
        validate_request(header, send, recv)
        writing = header.operation in (Operation.PUT, Operation.UPDATE)
        peer = connection["agent"]
        request_id = uuid4().hex
        reply = Future()
        response = None
        with self._lock:
            if self._closed:
                raise TransferError("NIXL payload transport is closed")
            offer = self._publish(peer, request_id, send) if send else None
            try:
                control = self._pack_control(
                    {
                        "kind": "request",
                        "id": request_id,
                        "peer": self._endpoint,
                        "op": header.operation.value,
                        "body": header.body,
                        "offer": offer,
                    }
                )
            except BaseException:
                self._release_export(peer, request_id)
                raise
            self._waiters[request_id] = (peer, reply)

        wire_bytes = len(control) + (offer["size"] if offer else 0)
        try:
            try:
                self._runtime.send_notification(connection, control)
                deadline = time.monotonic() + self._timeout
                while not reply.done():
                    self._poll_replies()
                    if not reply.done():
                        if time.monotonic() >= deadline:
                            raise TransferTimeout("NIXL storage request timed out", wire_bytes=wire_bytes)
                        await asyncio.sleep(0.001)
                response = reply.result()
            except Exception as exc:
                if writing:
                    raise WriteOutcomeUnknown(f"NIXL write {request_id} outcome unknown: {exc}") from exc
                raise
        finally:
            with self._lock:
                self._waiters.pop(request_id, None)
                if response is None and reply.done() and reply.exception() is None and reply.result().get("offer"):
                    self._release_response(connection, request_id)

        error = response.get("error")
        if error is not None:
            if response.get("rejected"):
                raise StorageRejected(error, missing_key=response.get("missing_key", False))
            if writing and response.get("outcome_unknown"):
                raise WriteOutcomeUnknown(error)
            raise TransferError(error)

        offer = response.get("offer")
        if writing and (not response.get("input_done") or offer is not None):
            raise WriteOutcomeUnknown("invalid NIXL write acknowledgement")
        if not writing and offer is None:
            raise TransferError("NIXL GET response is missing its buffer offer")
        if offer is None:
            spans = ()
        else:
            with self._lock:
                if self._closed:
                    raise TransferError("NIXL payload transport is closed")
                self._active_reads.add((peer, request_id))
                pending = asyncio.wrap_future(self._reads.submit(self._read_response, connection, request_id, offer))
            try:
                spans = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # The READ owns private memory and must finish before releasing the remote offer.
                pending.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
                raise
        return ReplyHeader(response["body"], wire_bytes), place_reply(spans, recv)

    def _messages(self):
        for sender, messages in self._runtime.poll_notifications().items():
            for packed in messages:
                try:
                    if len(packed) > MAX_CONTROL_BYTES:
                        raise TransferError("oversized NIXL control notification")
                    message = decode(unpack_from(packed))
                    if message["version"] != PROTOCOL_VERSION:
                        raise TransferError("unsupported NIXL notification protocol")
                    request_id = message["id"]
                    if not isinstance(request_id, str) or len(request_id) != 32:
                        raise TransferError("invalid NIXL request id")
                except Exception:
                    logger.exception("invalid NIXL notification from %s", sender)
                    continue
                yield sender, request_id, message

    def _poll_replies(self):
        with self._lock:
            if self._closed:
                return
            for sender, request_id, message in self._messages():
                if message.get("kind") != "reply" or sender not in self._connections:
                    continue
                if message.get("input_done") is True:
                    self._release_export(sender, request_id)
                pending = self._waiters.get(request_id)
                if pending is not None and pending[0] == sender:
                    if not pending[1].done():
                        pending[1].set_result(message)
                elif message.get("offer") is not None and (sender, request_id) not in self._active_reads:
                    self._release_response(self._connections[sender], request_id)

    def poll_requests(self):
        """Run on the existing storage worker, serial with administrative operations."""
        for sender, request_id, message in self._messages():
            try:
                if message.get("kind") == "release":
                    self._release_export(sender, request_id)
                elif message.get("kind") == "request":
                    self._serve(sender, request_id, message)
            except Exception:
                logger.exception("NIXL request %s from %s failed", request_id, sender)

    def _serve(self, sender, request_id, message):
        peer = message["peer"]
        if peer["agent"] != sender:
            raise TransferError("NIXL sender does not match its endpoint")
        self._runtime.connect(peer)
        reply = {"kind": "reply", "id": request_id, "input_done": True}
        try:
            operation = Operation(message["op"])
            offer = message.get("offer")
            if (operation == Operation.GET) != (offer is None):
                raise TransferError("invalid NIXL input buffer offer")
            data = None
            if offer is not None:
                reply["input_done"] = False
                try:
                    packed = self._runtime.read(peer, offer)
                except ReadAccessUnknown:
                    raise
                except Exception:
                    reply["input_done"] = True
                    raise
                reply["input_done"] = True
                data = decode_spans(tuple(unpack_from(packed)))
            result, payload = self._handle(RequestHeader(operation, message["body"]), data)
            reply["body"] = result
            if payload is not None:
                reply["offer"] = self._publish(sender, request_id, encode_spans(payload))
        except StorageRejected as exc:
            reply.update(error=str(exc)[:4096], rejected=True, missing_key=exc.missing_key)
        except Exception as exc:
            logger.exception("NIXL storage request %s failed", request_id)
            reply["error"] = str(exc)[:4096]
        try:
            packed = self._pack_control(reply)
        except Exception:
            self._release_export(sender, request_id)
            # A write may already be accepted even if its result cannot be encoded.
            packed = self._pack_control(
                {
                    "kind": "reply",
                    "id": request_id,
                    "input_done": reply["input_done"],
                    "error": "NIXL storage reply cannot be encoded",
                    "outcome_unknown": message.get("op") != Operation.GET.value,
                }
            )
        self._runtime.send_notification(peer, packed)

    def shutdown(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for _, waiter in self._waiters.values():
                if not waiter.done():
                    waiter.set_exception(TransferError("NIXL payload transport closed"))
        self._reads.shutdown(wait=True)
        self._runtime.close()
        self._exports.clear()
        self._connections.clear()
