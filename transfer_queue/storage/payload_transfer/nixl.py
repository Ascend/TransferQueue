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

"""NIXL H2H payload transfer strategy for SimpleStorage."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

import zmq
import zmq.asyncio

from transfer_queue.storage.payload_transfer.base import (
    DeferredResponse,
    PayloadTransfer,
    PayloadTransferError,
    PayloadTransferTimeout,
)
from transfer_queue.storage.payload_transfer.nixl_ucx_runtime import (
    DEFAULT_NIXL_RECEIVE_BUFFER_CACHE_MB,
    NixlError,
    NixlRuntime,
    NixlWriteNotStarted,
)
from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.serial_utils import decode, encode
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo

logger = get_logger(__name__)
TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))
TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT", 200))


@dataclass(frozen=True)
class PayloadDescriptor:
    """Description of one encoded payload."""

    transfer_id: str
    payload_bytes: int
    frame_sizes: tuple[int, ...]

    @staticmethod
    def validate_transfer_id(transfer_id: str) -> None:
        if not transfer_id or len(transfer_id) > 128:
            raise PayloadTransferError("transfer_id must contain 1 to 128 characters")

    def validate(self) -> None:
        self.validate_transfer_id(self.transfer_id)
        if self.payload_bytes < 0:
            raise PayloadTransferError(f"negative payload length for {self.transfer_id}")
        if any(size < 0 for size in self.frame_sizes):
            raise PayloadTransferError(f"negative frame length for {self.transfer_id}")
        frame_bytes = sum(self.frame_sizes)
        if frame_bytes != self.payload_bytes:
            raise PayloadTransferError(
                f"payload length mismatch for {self.transfer_id}: expected {frame_bytes}, got {self.payload_bytes}"
            )

    def to_dict(self) -> dict[str, int | str | list[int]]:
        return {
            "transfer_id": self.transfer_id,
            "payload_bytes": self.payload_bytes,
            "frame_sizes": list(self.frame_sizes),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PayloadDescriptor:
        descriptor = cls(
            transfer_id=str(value["transfer_id"]),
            payload_bytes=int(value["payload_bytes"]),
            frame_sizes=tuple(int(size) for size in value["frame_sizes"]),
        )
        descriptor.validate()
        return descriptor


@dataclass
class TransferEndpoint:
    """Bootstrap metadata for a NIXL endpoint."""

    transport: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"transport": self.transport, "data": self.data}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TransferEndpoint:
        return cls(transport=str(value["transport"]), data=dict(value["data"]))


@dataclass
class ReceiveToken:
    """Transport-owned metadata returned after preparing a receive."""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReceiveToken:
        return cls(data=dict(value["data"]))


@dataclass
class _PendingPut:
    descriptor: PayloadDescriptor
    sender_id: str
    global_indexes: tuple[int, ...]
    data_parser: Any


@dataclass
class _PendingGet:
    descriptor: PayloadDescriptor
    sender_id: str
    frames: tuple[bytes | bytearray | memoryview, ...]


class NixlPayloadTransfer(PayloadTransfer):
    """Own the NIXL payload protocol and its transfer lifecycle."""

    transport = "nixl-ucx"

    def __init__(
        self,
        ucx_env_vars: dict[str, object] | None = None,
        peer_infos: Mapping[str, object] | None = None,
        control_peer_infos: Mapping[str, ZMQServerInfo] | None = None,
        *,
        receive_buffer_cache_mb: int = DEFAULT_NIXL_RECEIVE_BUFFER_CACHE_MB,
    ):
        self._peer_infos = dict(peer_infos or {})
        if control_peer_infos and set(control_peer_infos) != set(self._peer_infos):
            raise RuntimeError("SimpleStorage payload transfer endpoints are missing")
        try:
            self._runtime = NixlRuntime(ucx_env_vars, receive_buffer_cache_mb=receive_buffer_cache_mb)
        except NixlError:
            raise
        except Exception as exc:
            raise NixlError(f"failed to create NIXL runtime: {exc}") from exc
        self._pending_puts: dict[str, _PendingPut] = {}
        self._pending_gets: dict[str, _PendingGet] = {}
        self._failed_get_targets: set[str] = set()

    @property
    def diagnostics(self) -> dict[str, float | int]:
        return {**self._runtime.diagnostics, "failed_get_targets": len(self._failed_get_targets)}

    def bootstrap_info(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint().to_dict()}

    def endpoint(self) -> TransferEndpoint:
        return TransferEndpoint(transport=self.transport, data={"agent_name": self._runtime.agent_name})

    async def put(
        self,
        *,
        control_socket: zmq.asyncio.Socket,
        sender_id: str,
        target_id: str,
        global_indexes: list[int],
        data: dict[str, Any],
        data_parser: Callable[[Any], Any] | None,
    ) -> int:
        frames = tuple(encode(data))
        frame_sizes = tuple(memoryview(frame).nbytes for frame in frames)
        descriptor = PayloadDescriptor(transfer_id=uuid4().hex, payload_bytes=sum(frame_sizes), frame_sizes=frame_sizes)
        remote_may_be_prepared = False
        commit_attempted = False
        send_future: Future[None] | None = None
        try:
            prepare = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_PREPARE,
                sender_id=sender_id,
                receiver_id=target_id,
                body={
                    "global_indexes": global_indexes,
                    "descriptor": descriptor.to_dict(),
                    "data_parser": data_parser,
                },
            )
            remote_may_be_prepared = True
            await control_socket.send_multipart(prepare.serialize(), copy=False)
            ready = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            self._expect(ready, ZMQRequestType.PUT_DATA_READY, target_id)
            if PayloadDescriptor.from_dict(ready.body["descriptor"]) != descriptor:
                raise RuntimeError(f"PUT descriptor changed by storage unit {target_id}")
            token = ReceiveToken.from_dict(ready.body["receive_token"])
            endpoint = self._peer_endpoint(target_id)
            send_future = self.send(endpoint, token, descriptor, frames)
            await asyncio.wrap_future(send_future)

            commit = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_COMMIT,
                sender_id=sender_id,
                receiver_id=target_id,
                body={"transfer_id": descriptor.transfer_id},
            )
            commit_attempted = True
            await control_socket.send_multipart(commit.serialize(), copy=False)
            response = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            self._expect(response, ZMQRequestType.PUT_DATA_RESPONSE, target_id)
            return descriptor.payload_bytes
        except BaseException as exc:
            cancel_confirmed = False
            if remote_may_be_prepared and not commit_attempted:
                safe_to_cancel = True
                if send_future is not None and not send_future.cancel():
                    # Keep the transaction socket alive until a running WRITE finishes.
                    try:
                        await asyncio.wrap_future(send_future)
                    except NixlWriteNotStarted:
                        pass
                    except Exception:
                        safe_to_cancel = False
                if safe_to_cancel:
                    cancel_confirmed = await self._cancel(
                        sender_id, target_id, ZMQRequestType.PUT_DATA_CANCEL, descriptor.transfer_id, control_socket
                    )
            if isinstance(exc, zmq.error.Again):
                if commit_attempted:
                    raise PayloadTransferError(
                        f"NIXL PUT commit outcome is unknown for storage unit {target_id}; automatic retry disabled"
                    ) from exc
                if not cancel_confirmed:
                    raise PayloadTransferError(
                        f"NIXL PUT prepare timed out at storage unit {target_id} without confirmed cancellation; "
                        "automatic retry disabled"
                    ) from exc
                raise PayloadTransferTimeout(
                    f"ZMQ recv timeout during put to storage unit {target_id}",
                    wire_bytes=descriptor.payload_bytes,
                ) from exc
            if isinstance(exc, zmq.ZMQError):
                raise PayloadTransferError(
                    f"Error in put to storage unit {target_id}: {type(exc).__name__}: {exc}"
                ) from exc
            raise

    async def get(
        self,
        *,
        control_socket: zmq.asyncio.Socket,
        sender_id: str,
        target_id: str,
        global_indexes: list[int],
        fields: list[str],
    ) -> dict[str, Any]:
        if target_id in self._failed_get_targets:
            raise NixlError(f"NIXL GET session for {target_id!r} has failed; recreate the payload transfer")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT
        transfer_id = uuid4().hex
        remote_prepared = False
        receive_prepared = False
        commit_attempted = False
        response: ZMQMessage | None = None
        response_future: asyncio.Future | None = None
        try:
            prepare = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_PREPARE,
                sender_id=sender_id,
                receiver_id=target_id,
                body={"global_indexes": global_indexes, "fields": fields, "transfer_id": transfer_id},
            )
            remote_prepared = True
            await control_socket.send_multipart(prepare.serialize(), copy=False)
            ready = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            self._expect(ready, ZMQRequestType.GET_DATA_READY, target_id)
            descriptor = PayloadDescriptor.from_dict(ready.body["descriptor"])
            if descriptor.transfer_id != transfer_id:
                raise RuntimeError(f"GET descriptor identity changed by storage unit {target_id}")
            if target_id in self._failed_get_targets:
                raise NixlError(f"NIXL GET session for {target_id!r} has failed; recreate the payload transfer")
            token = self.prepare_receive(descriptor)
            receive_prepared = True
            commit = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_COMMIT,
                sender_id=sender_id,
                receiver_id=target_id,
                body={
                    "transfer_id": descriptor.transfer_id,
                    "receiver_endpoint": TransferEndpoint(
                        self.transport, {"agent_name": token.data["agent_name"]}
                    ).to_dict(),
                    "receive_token": token.to_dict(),
                },
            )
            commit_attempted = True
            await control_socket.send_multipart(commit.serialize(), copy=False)
            response_future = asyncio.ensure_future(control_socket.recv_multipart(copy=False))
            response = ZMQMessage.deserialize(await asyncio.shield(response_future))
            self._expect(response, ZMQRequestType.GET_DATA_RESPONSE, target_id)
            if response.body.get("transfer_id") != transfer_id:
                raise RuntimeError(f"GET completion identity changed by storage unit {target_id}")
            frames = self.receive(descriptor)
            return decode(list(frames))
        except BaseException as exc:
            cancel_confirmed = False
            if receive_prepared:
                if commit_attempted and isinstance(exc, asyncio.CancelledError):
                    # Keep the original receive alive: cancelling the caller does not stop a remote WRITE.
                    try:
                        if response_future is None:
                            response_future = asyncio.ensure_future(control_socket.recv_multipart(copy=False))
                        frames = await asyncio.wait_for(asyncio.shield(response_future), max(0, deadline - loop.time()))
                        response = ZMQMessage.deserialize(list(frames))
                    except (Exception, asyncio.CancelledError):
                        pass
                safe_to_release = not commit_attempted or (
                    response is not None
                    and response.sender_id == target_id
                    and response.body.get("transfer_id") == transfer_id
                    and (
                        response.request_type == ZMQRequestType.GET_DATA_RESPONSE
                        or (
                            response.request_type == ZMQRequestType.PUT_GET_ERROR
                            and response.body.get("write_not_started") is True
                        )
                    )
                )
                if safe_to_release:
                    self.cancel_receive(transfer_id)
                else:
                    self._failed_get_targets.add(target_id)
                    self.quarantine_receive(transfer_id)
            if remote_prepared and not commit_attempted:
                cancel_confirmed = await self._cancel(
                    sender_id, target_id, ZMQRequestType.GET_DATA_CANCEL, transfer_id, control_socket
                )
            if isinstance(exc, zmq.error.Again):
                if commit_attempted:
                    raise PayloadTransferError(
                        f"NIXL GET commit outcome is unknown for storage unit {target_id}; automatic retry disabled"
                    ) from exc
                if not cancel_confirmed:
                    raise PayloadTransferError(
                        f"NIXL GET prepare timed out at storage unit {target_id} without confirmed cancellation; "
                        "automatic retry disabled"
                    ) from exc
                raise PayloadTransferTimeout(f"ZMQ recv timeout during get from storage unit {target_id}") from exc
            if isinstance(exc, zmq.ZMQError):
                raise PayloadTransferError(
                    f"Error getting data from storage unit {target_id}: {type(exc).__name__}: {exc}"
                ) from exc
            raise
        finally:
            if response_future is not None:
                response_future.cancel()

    def handle_request(
        self,
        request: ZMQMessage,
        *,
        storage_id: str,
        load_data: Callable[..., dict[str, Any]],
        store_data: Callable[..., None],
    ) -> ZMQMessage | DeferredResponse | None:
        if request.request_type == ZMQRequestType.PUT_DATA_PREPARE:
            return self._handle_put_prepare(request, storage_id)
        if request.request_type == ZMQRequestType.PUT_DATA_COMMIT:
            return self._handle_put_commit(request, storage_id, store_data)
        if request.request_type == ZMQRequestType.PUT_DATA_CANCEL:
            return self._handle_put_cancel(request, storage_id)
        if request.request_type == ZMQRequestType.GET_DATA_PREPARE:
            return self._handle_get_prepare(request, storage_id, load_data)
        if request.request_type == ZMQRequestType.GET_DATA_COMMIT:
            return self._handle_get_commit(request, storage_id)
        if request.request_type == ZMQRequestType.GET_DATA_CANCEL:
            return self._handle_get_cancel(request, storage_id)
        return None

    def _handle_put_prepare(self, request: ZMQMessage, storage_id: str) -> ZMQMessage:
        descriptor = None
        prepared = False
        try:
            descriptor = PayloadDescriptor.from_dict(request.body["descriptor"])
            if descriptor.transfer_id in self._pending_puts:
                raise RuntimeError(f"duplicate PUT transfer_id: {descriptor.transfer_id}")
            token = self.prepare_receive(descriptor)
            prepared = True
            self._pending_puts[descriptor.transfer_id] = _PendingPut(
                descriptor,
                request.sender_id,
                tuple(request.body["global_indexes"]),
                request.body.get("data_parser"),
            )
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_READY,
                sender_id=storage_id,
                body={"descriptor": descriptor.to_dict(), "receive_token": token.to_dict()},
            )
        except Exception as exc:
            if descriptor is not None and prepared:
                self._pending_puts.pop(descriptor.transfer_id, None)
                self.cancel_receive(descriptor.transfer_id)
            return self._error(storage_id, "PUT prepare", exc)

    def _handle_put_commit(self, request: ZMQMessage, storage_id: str, store_data: Callable[..., None]) -> ZMQMessage:
        transfer_id = request.body["transfer_id"]
        try:
            pending = self._pending_puts.get(transfer_id)
            if pending is None:
                raise RuntimeError(f"unknown or expired PUT transfer_id: {transfer_id}")
            if pending.sender_id != request.sender_id:
                raise RuntimeError(f"PUT transfer {transfer_id} belongs to another sender")
            self._pending_puts.pop(transfer_id)
            try:
                frames = self.receive(pending.descriptor)
            except Exception:
                self.cancel_receive(transfer_id)
                raise
            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{storage_id}] PUT commit"
            ):
                store_data(list(pending.global_indexes), decode(list(frames)), pending.data_parser)
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE,
                sender_id=storage_id,
                body={"transfer_id": transfer_id},
            )
        except Exception as exc:
            return self._error(storage_id, "PUT commit", exc)

    def _handle_put_cancel(self, request: ZMQMessage, storage_id: str) -> ZMQMessage:
        transfer_id = request.body["transfer_id"]
        pending = self._pending_puts.get(transfer_id)
        if pending is not None and pending.sender_id != request.sender_id:
            return self._error(
                storage_id,
                "PUT cancel",
                RuntimeError(f"PUT transfer {transfer_id} belongs to another sender"),
            )
        self._pending_puts.pop(transfer_id, None)
        self.cancel_receive(transfer_id)
        return ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA_RESPONSE,
            sender_id=storage_id,
            body={"transfer_id": transfer_id},
        )

    def _handle_get_prepare(
        self,
        request: ZMQMessage,
        storage_id: str,
        load_data: Callable[..., dict[str, Any]],
    ) -> ZMQMessage:
        try:
            transfer_id = str(request.body["transfer_id"])
            PayloadDescriptor.validate_transfer_id(transfer_id)
            if transfer_id in self._pending_gets:
                raise RuntimeError(f"duplicate GET transfer_id: {transfer_id}")
            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{storage_id}] GET prepare"
            ):
                frames = tuple(encode(load_data(request.body["fields"], request.body["global_indexes"])))
            frame_sizes = tuple(memoryview(frame).nbytes for frame in frames)
            descriptor = PayloadDescriptor(
                transfer_id=transfer_id, payload_bytes=sum(frame_sizes), frame_sizes=frame_sizes
            )
            self._pending_gets[transfer_id] = _PendingGet(descriptor, request.sender_id, frames)
            return ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_READY,
                sender_id=storage_id,
                body={"descriptor": descriptor.to_dict()},
            )
        except Exception as exc:
            return self._error(storage_id, "GET prepare", exc)

    def _handle_get_commit(self, request: ZMQMessage, storage_id: str) -> DeferredResponse | ZMQMessage:
        transfer_id = request.body["transfer_id"]
        write_not_started = False
        try:
            pending = self._pending_gets.get(transfer_id)
            if pending is None:
                raise RuntimeError(f"unknown or expired GET transfer_id: {transfer_id}")
            if pending.sender_id != request.sender_id:
                raise RuntimeError(f"GET transfer {transfer_id} belongs to another sender")
            self._pending_gets.pop(transfer_id)
            write_not_started = True
            endpoint = TransferEndpoint.from_dict(request.body["receiver_endpoint"])
            token = ReceiveToken.from_dict(request.body["receive_token"])
            send_future = self.send(endpoint, token, pending.descriptor, pending.frames)
            write_not_started = False
            response_future: Future[ZMQMessage] = Future()

            def complete_response(completed: Future[None]) -> None:
                try:
                    completed.result()
                    response = ZMQMessage.create(
                        request_type=ZMQRequestType.GET_DATA_RESPONSE,
                        sender_id=storage_id,
                        body={"transfer_id": transfer_id},
                    )
                except Exception as exc:
                    response = self._error(
                        storage_id,
                        "GET commit",
                        exc,
                        transfer_id=transfer_id,
                        write_not_started=completed.cancelled() or isinstance(exc, NixlWriteNotStarted),
                    )
                response_future.set_result(response)

            send_future.add_done_callback(complete_response)
            return DeferredResponse(response_future)
        except Exception as exc:
            return self._error(
                storage_id, "GET commit", exc, transfer_id=transfer_id, write_not_started=write_not_started
            )

    def _handle_get_cancel(self, request: ZMQMessage, storage_id: str) -> ZMQMessage:
        transfer_id = request.body["transfer_id"]
        pending = self._pending_gets.get(transfer_id)
        if pending is not None and pending.sender_id != request.sender_id:
            return self._error(
                storage_id,
                "GET cancel",
                RuntimeError(f"GET transfer {transfer_id} belongs to another sender"),
            )
        self._pending_gets.pop(transfer_id, None)
        return ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA_RESPONSE,
            sender_id=storage_id,
            body={"transfer_id": transfer_id},
        )

    def prepare_receive(self, descriptor: PayloadDescriptor) -> ReceiveToken:
        return ReceiveToken(data=self._runtime.prepare_receive(descriptor))

    def send(
        self,
        endpoint: TransferEndpoint,
        token: ReceiveToken,
        descriptor: PayloadDescriptor,
        frames: tuple[bytes | bytearray | memoryview, ...] | list[bytes | bytearray | memoryview],
    ) -> Future[None]:
        if endpoint.transport != self.transport:
            raise PayloadTransferError(f"NIXL cannot use endpoint for {endpoint.transport!r}")
        if token.data.get("agent_name") != endpoint.data.get("agent_name"):
            raise PayloadTransferError("NIXL endpoint and receive token agent names differ")
        return self._runtime.send(endpoint.data, token.data, descriptor, tuple(frames))

    def receive(self, descriptor: PayloadDescriptor) -> tuple[memoryview, ...]:
        return self._runtime.receive(descriptor)

    def cancel_receive(self, transfer_id: str) -> None:
        self._runtime.cancel_receive(transfer_id)

    def quarantine_receive(self, transfer_id: str) -> None:
        self._runtime.quarantine_receive(transfer_id)

    def close(self) -> None:
        self._pending_puts.clear()
        self._pending_gets.clear()
        self._runtime.close()

    def _peer_endpoint(self, target_id: str) -> TransferEndpoint:
        try:
            peer_info = self._peer_infos[target_id]
            if not isinstance(peer_info, Mapping):
                raise TypeError("invalid NIXL peer info")
            return TransferEndpoint.from_dict(peer_info["endpoint"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"NIXL endpoint is missing for storage unit {target_id}") from exc

    async def _cancel(
        self,
        sender_id: str,
        target_id: str,
        request_type: ZMQRequestType,
        transfer_id: str,
        control_socket: zmq.asyncio.Socket,
    ) -> bool:
        """Keep PREPARE/CANCEL ordered and confirm cleanup on the transaction's original pipe."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT, 10)
        try:
            cancel = ZMQMessage.create(
                request_type=request_type,
                sender_id=sender_id,
                receiver_id=target_id,
                body={"transfer_id": transfer_id},
            )
            await asyncio.wait_for(
                control_socket.send_multipart(cancel.serialize(), copy=False), deadline - loop.time()
            )
            expected = (
                ZMQRequestType.PUT_DATA_RESPONSE
                if request_type == ZMQRequestType.PUT_DATA_CANCEL
                else ZMQRequestType.GET_DATA_RESPONSE
            )
            while True:
                # A late READY or PREPARE error can precede the CANCEL response.
                frames = await asyncio.wait_for(control_socket.recv_multipart(copy=False), deadline - loop.time())
                response = ZMQMessage.deserialize(frames)
                if (
                    response.request_type == expected
                    and response.sender_id == target_id
                    and response.body.get("transfer_id") == transfer_id
                ):
                    return True
        except Exception as exc:
            logger.warning("failed to cancel %s transfer %s at %s: %s", request_type.value, transfer_id, target_id, exc)
            return False

    @staticmethod
    def _expect(response: ZMQMessage, expected: ZMQRequestType, target_id: str) -> None:
        if response.sender_id != target_id:
            raise PayloadTransferError(f"expected response from storage unit {target_id}, got {response.sender_id}")
        if response.request_type != expected:
            raise RuntimeError(
                f"storage unit {target_id} returned {response.request_type}: "
                f"{response.body.get('message', 'unknown error')}"
            )

    @staticmethod
    def _error(
        storage_id: str,
        operation: str,
        error: Exception,
        *,
        transfer_id: str | None = None,
        write_not_started: bool = False,
    ) -> ZMQMessage:
        body: dict[str, Any] = {"message": f"{operation} failed in storage unit {storage_id}: {error}"}
        if transfer_id is not None:
            body.update(transfer_id=transfer_id, write_not_started=write_not_started)
        return ZMQMessage.create(
            request_type=ZMQRequestType.PUT_GET_ERROR,
            sender_id=storage_id,
            body=body,
        )
