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

import zmq.asyncio

from transfer_queue.storage.payload_transfer.base import PayloadTransfer, PayloadTransferError
from transfer_queue.storage.payload_transfer.nixl_ucx_runtime import NixlError, NixlRuntime
from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.serial_utils import calc_packed_size, decode, encode, unpack_from
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    format_zmq_address,
)

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
        packed_size = 4 + 8 * len(self.frame_sizes) + sum(self.frame_sizes)
        if packed_size != self.payload_bytes:
            raise PayloadTransferError(
                f"packed payload length mismatch for {self.transfer_id}: "
                f"expected {packed_size}, got {self.payload_bytes}"
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
    ):
        self._peer_infos = dict(peer_infos or {})
        self._control_peer_infos = dict(control_peer_infos or {})
        if self._control_peer_infos and set(self._control_peer_infos) != set(self._peer_infos):
            raise RuntimeError("SimpleStorage payload transfer endpoints are missing")
        try:
            self._runtime = NixlRuntime(ucx_env_vars)
        except NixlError:
            raise
        except Exception as exc:
            raise NixlError(f"failed to create NIXL runtime: {exc}") from exc
        self._pending_puts: dict[str, _PendingPut] = {}
        self._pending_gets: dict[str, _PendingGet] = {}

    def bootstrap_info(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint().to_dict()}

    def endpoint(self) -> TransferEndpoint:
        return TransferEndpoint(
            transport=self.transport,
            data={
                "agent_name": self._runtime.agent_name,
                "agent_metadata": self._runtime.endpoint_metadata(),
            },
        )

    async def put(
        self,
        *,
        control_socket: zmq.asyncio.Socket,
        sender_id: str,
        target_id: str,
        global_indexes: list[int],
        data: dict[str, Any],
        data_parser: Callable[[Any], Any] | None,
    ) -> None:
        frames = tuple(encode(data))
        descriptor = PayloadDescriptor(
            transfer_id=uuid4().hex,
            payload_bytes=calc_packed_size(frames),
            frame_sizes=tuple(memoryview(frame).nbytes for frame in frames),
        )
        descriptor.validate()
        remote_may_be_prepared = True
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
            await control_socket.send_multipart(prepare.serialize(), copy=False)
            ready = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            self._expect(ready, ZMQRequestType.PUT_DATA_READY, target_id)
            if PayloadDescriptor.from_dict(ready.body["descriptor"]) != descriptor:
                raise RuntimeError(f"PUT descriptor changed by storage unit {target_id}")
            token = ReceiveToken.from_dict(ready.body["receive_token"])
            endpoint = self._peer_endpoint(target_id)
            await asyncio.wrap_future(self.send(endpoint, token, descriptor, frames))

            commit = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_COMMIT,
                sender_id=sender_id,
                receiver_id=target_id,
                body={"transfer_id": descriptor.transfer_id},
            )
            await control_socket.send_multipart(commit.serialize(), copy=False)
            response = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            self._expect(response, ZMQRequestType.PUT_DATA_RESPONSE, target_id)
            remote_may_be_prepared = False
        except BaseException:
            if remote_may_be_prepared:
                await self._cancel(sender_id, target_id, ZMQRequestType.PUT_DATA_CANCEL, descriptor.transfer_id)
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
        transfer_id = uuid4().hex
        remote_prepared = False
        receive_prepared = False
        descriptor = None
        try:
            prepare = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_PREPARE,
                sender_id=sender_id,
                receiver_id=target_id,
                body={"global_indexes": global_indexes, "fields": fields, "transfer_id": transfer_id},
            )
            await control_socket.send_multipart(prepare.serialize(), copy=False)
            remote_prepared = True
            ready = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            self._expect(ready, ZMQRequestType.GET_DATA_READY, target_id)
            descriptor = PayloadDescriptor.from_dict(ready.body["descriptor"])
            if descriptor.transfer_id != transfer_id:
                raise RuntimeError(f"GET descriptor identity changed by storage unit {target_id}")
            token = self.prepare_receive(descriptor)
            receive_prepared = True
            commit = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_COMMIT,
                sender_id=sender_id,
                receiver_id=target_id,
                body={
                    "transfer_id": descriptor.transfer_id,
                    "receiver_endpoint": self.endpoint().to_dict(),
                    "receive_token": token.to_dict(),
                },
            )
            await control_socket.send_multipart(commit.serialize(), copy=False)
            response = ZMQMessage.deserialize(await control_socket.recv_multipart(copy=False))
            self._expect(response, ZMQRequestType.GET_DATA_RESPONSE, target_id)
            remote_prepared = False
            payload = await asyncio.wrap_future(self.receive(descriptor))
            return decode(unpack_from(payload))
        except BaseException:
            if receive_prepared and descriptor is not None:
                self.cancel_receive(descriptor.transfer_id)
            if remote_prepared:
                await self._cancel(sender_id, target_id, ZMQRequestType.GET_DATA_CANCEL, transfer_id)
            raise

    def handle_request(
        self,
        request: ZMQMessage,
        *,
        storage_id: str,
        load_data: Callable[..., dict[str, Any]],
        store_data: Callable[..., None],
    ) -> ZMQMessage | None:
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
        owns_receive = False
        try:
            pending = self._pending_puts.get(transfer_id)
            if pending is None:
                raise RuntimeError(f"unknown or expired PUT transfer_id: {transfer_id}")
            if pending.sender_id != request.sender_id:
                raise RuntimeError(f"PUT transfer {transfer_id} belongs to another sender")
            self._pending_puts.pop(transfer_id)
            owns_receive = True
            payload = self.receive(pending.descriptor).result()
            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{storage_id}] PUT commit"
            ):
                store_data(list(pending.global_indexes), decode(unpack_from(payload)), pending.data_parser)
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE,
                sender_id=storage_id,
                body={"transfer_id": transfer_id},
            )
        except Exception as exc:
            if owns_receive:
                self.cancel_receive(transfer_id)
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
            descriptor = PayloadDescriptor(
                transfer_id=transfer_id,
                payload_bytes=calc_packed_size(frames),
                frame_sizes=tuple(memoryview(frame).nbytes for frame in frames),
            )
            descriptor.validate()
            self._pending_gets[transfer_id] = _PendingGet(descriptor, request.sender_id, frames)
            return ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_READY,
                sender_id=storage_id,
                body={"descriptor": descriptor.to_dict()},
            )
        except Exception as exc:
            return self._error(storage_id, "GET prepare", exc)

    def _handle_get_commit(self, request: ZMQMessage, storage_id: str) -> ZMQMessage:
        transfer_id = request.body["transfer_id"]
        try:
            pending = self._pending_gets.get(transfer_id)
            if pending is None:
                raise RuntimeError(f"unknown or expired GET transfer_id: {transfer_id}")
            if pending.sender_id != request.sender_id:
                raise RuntimeError(f"GET transfer {transfer_id} belongs to another sender")
            self._pending_gets.pop(transfer_id)
            endpoint = TransferEndpoint.from_dict(request.body["receiver_endpoint"])
            token = ReceiveToken.from_dict(request.body["receive_token"])
            self.send(endpoint, token, pending.descriptor, pending.frames).result()
            return ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_RESPONSE,
                sender_id=storage_id,
                body={"transfer_id": transfer_id},
            )
        except Exception as exc:
            return self._error(storage_id, "GET commit", exc)

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
        self._validate_metadata(endpoint, token, descriptor)
        return self._runtime.send(endpoint.data, token.data, descriptor, tuple(frames))

    def receive(self, descriptor: PayloadDescriptor) -> Future[memoryview]:
        return self._runtime.receive(descriptor)

    def cancel_receive(self, transfer_id: str) -> None:
        self._runtime.cancel_receive(transfer_id)

    def close(self) -> None:
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
    ) -> None:
        """Send cancellation on a fresh DEALER to isolate late responses."""
        cancel_context = zmq.asyncio.Context()
        cancel_socket = None
        try:
            server_info = self._control_peer_infos[target_id]
            cancel_socket = create_zmq_socket(
                cancel_context,
                zmq.DEALER,
                server_info.ip,
                identity=(f"{sender_id}_cancel_{target_id}_{uuid4().hex[:8]}").encode(),
            )
            timeout_ms = min(TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT, 10) * 1000
            cancel_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            cancel_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            cancel_socket.connect(format_zmq_address(server_info.ip, server_info.ports["put_get_socket"]))
            cancel = ZMQMessage.create(
                request_type=request_type,
                sender_id=sender_id,
                receiver_id=target_id,
                body={"transfer_id": transfer_id},
            )
            await cancel_socket.send_multipart(cancel.serialize(), copy=False)
            response = ZMQMessage.deserialize(await cancel_socket.recv_multipart(copy=False))
            expected = (
                ZMQRequestType.PUT_DATA_RESPONSE
                if request_type == ZMQRequestType.PUT_DATA_CANCEL
                else ZMQRequestType.GET_DATA_RESPONSE
            )
            self._expect(response, expected, target_id)
        except Exception as exc:
            logger.warning("failed to cancel %s transfer %s at %s: %s", request_type.value, transfer_id, target_id, exc)
        finally:
            if cancel_socket is not None:
                cancel_socket.close(linger=0)
            cancel_context.term()

    @staticmethod
    def _expect(response: ZMQMessage, expected: ZMQRequestType, target_id: str) -> None:
        if response.request_type != expected:
            raise RuntimeError(
                f"storage unit {target_id} returned {response.request_type}: "
                f"{response.body.get('message', 'unknown error')}"
            )

    @staticmethod
    def _error(storage_id: str, operation: str, error: Exception) -> ZMQMessage:
        return ZMQMessage.create(
            request_type=ZMQRequestType.PUT_GET_ERROR,
            sender_id=storage_id,
            body={"message": f"{operation} failed in storage unit {storage_id}: {error}"},
        )

    @staticmethod
    def _validate_metadata(endpoint: TransferEndpoint, token: ReceiveToken, descriptor: PayloadDescriptor) -> None:
        descriptor.validate()
        if endpoint.transport != "nixl-ucx":
            raise PayloadTransferError(f"NIXL cannot use endpoint for {endpoint.transport!r}")
        if token.data.get("agent_name") != endpoint.data.get("agent_name"):
            raise PayloadTransferError("NIXL endpoint and receive token agent names differ")
        if not isinstance(token.data.get("frame_remote_descs"), bytes):
            raise PayloadTransferError("NIXL receive token is missing frame descriptors")
        if not isinstance(token.data.get("agent_metadata"), bytes):
            raise PayloadTransferError("NIXL receive token is missing agent metadata")
