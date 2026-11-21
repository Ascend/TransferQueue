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


"""Exercise the field-data contract over the existing ZMQ multipart path."""

import asyncio
import errno
import gc
import weakref
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import torch
import zmq.asyncio

from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager, StorageUnitTimeout
from transfer_queue.storage.payload_transfer.base import (
    Operation,
    RequestHeader,
    StorageRejected,
    TransferError,
    TransferTimeout,
)
from transfer_queue.storage.payload_transfer.zmq import (
    ERROR_TYPES,
    KEY_NOT_FOUND_MARKER,
    RESPONSE_TYPES,
    ZmqPayloadTransfer,
)
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import (
    STORAGE_MANAGER_IDENTITY_PREFIX,
    ZMQMessage,
    ZMQMessageDecodeError,
    ZMQRequestType,
    ZMQServerInfo,
    ZMQSocketPool,
)


@pytest.mark.parametrize("metrics_enabled", [False, True])
def test_storage_observability_without_zmq_dispatch(monkeypatch, metrics_enabled):
    from transfer_queue.metrics import TQMetricsExporter

    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    heavy = MagicMock()
    monkeypatch.setattr("transfer_queue.storage.simple_storage.log_heavy_operation", heavy)
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    if metrics_enabled:
        unit._metrics = TQMetricsExporter(role="storage")
    try:
        # A non-ZMQ backend invokes exactly this listen callback, without the worker loop.
        handle = unit.payload_transfer._handle
        handle(RequestHeader(Operation.PUT, {"global_indexes": [0]}), {"value": [torch.tensor([1])]})
        _, data = handle(RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["value"]}), None)
        torch.testing.assert_close(data["value"][0], torch.tensor([1]))
        with pytest.raises(StorageRejected):
            handle(RequestHeader(Operation.GET, {"global_indexes": [99], "fields": ["value"]}), None)
        assert unit._requests_arrived == 3
        assert unit._arrivals_by_op == {"PUT_DATA": 1, "GET_DATA": 2}
        assert heavy.call_count == 2
        assert all(call.args[3] is None and "phase=storage" in call.args[4] for call in heavy.call_args_list)
        if metrics_enabled:
            assert unit._metrics.request_total.labels(op_type="GET_DATA")._value.get() == 2
            assert unit._metrics.request_errors_total.labels(op_type="GET_DATA")._value.get() == 1
            hist = unit._metrics.request_duration.labels(op_type="GET_DATA")
            assert sum(bucket.get() for bucket in hist._buckets) == 2
        else:
            assert unit._perf_monitor.success_counts["GET_DATA"] == 2
    finally:
        unit.shutdown()


def test_storage_listen_callback_does_not_retain_owner(monkeypatch):
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    # Stop threads first, then check the finalizer's retained transport and callback.
    unit.shutdown()
    transfer, ref = unit.payload_transfer, weakref.ref(unit)
    del unit
    gc.collect()
    assert ref() is None
    assert transfer is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["send_timeout", "receive_timeout", "disconnect", "decode_error", "cancel", "receive_error"],
)
@pytest.mark.parametrize("operation", list(Operation))
async def test_payload_failure_discards_socket_and_releases_permit(monkeypatch, failure, operation):
    peer = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    failed = MagicMock(closed=False)
    failed.send_multipart = AsyncMock()
    failed.recv_multipart = AsyncMock()
    error = {
        "send_timeout": zmq.Again(),
        "receive_timeout": zmq.Again(),
        "disconnect": zmq.ZMQError(zmq.ENOTSOCK),
        "decode_error": ZMQMessageDecodeError("leading frame is empty"),
        "cancel": asyncio.CancelledError(),
        "receive_error": ValueError("invalid response"),
    }[failure]
    if failure == "send_timeout":
        failed.send_multipart.side_effect = error
    elif failure == "decode_error":
        failed.recv_multipart.return_value = [b""]
    else:
        failed.recv_multipart.side_effect = error
    healthy = MagicMock(closed=False)
    healthy.send_multipart = AsyncMock()
    healthy.recv_multipart = AsyncMock(
        return_value=ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA_RESPONSE,
            sender_id="unit",
            body={"data": {"text": ["next"]}},
        ).serialize()
    )
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    connect = MagicMock(side_effect=[failed, healthy])
    monkeypatch.setattr(pool, "_connect", connect)
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    header = RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["text"]})
    try:
        expected = type(error)
        if isinstance(error, zmq.Again):
            expected = TransferTimeout
        elif isinstance(error, (zmq.ZMQError, ZMQMessageDecodeError)):
            expected = TransferError
        with pytest.raises(expected) as caught:
            await client.request(
                peer,
                RequestHeader(operation, header.body),
                send=None if operation == Operation.GET else {"text": ["input"]},
                recv=None if operation == Operation.GET else (),
            )
        if expected is TransferTimeout:
            assert isinstance(caught.value.__cause__, zmq.Again)
            assert (caught.value.wire_bytes > 0) == (operation != Operation.GET)
        elif expected is TransferError:
            assert isinstance(caught.value.__cause__, type(error))
        failed.close.assert_called_once_with(linger=0)
        # A single permit forces recovery to exercise both socket eviction and permit return.
        for _ in range(2):
            _, data = await asyncio.wait_for(client.request(peer, header, recv=None), timeout=2)
            assert data == {"text": ["next"]}
        assert connect.call_count == 2
        healthy.close.assert_not_called()
    finally:
        client.shutdown()
        pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_number", [errno.EMFILE, zmq.ENOTSOCK])
async def test_socket_creation_error_is_wrapped_and_releases_permit(monkeypatch, error_number):
    peer = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    error = zmq.ZMQError(error_number)
    socket = MagicMock(closed=False)
    socket.send_multipart = AsyncMock()
    socket.recv_multipart = AsyncMock(return_value=ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA_RESPONSE, sender_id="unit", body={"data": {"text": ["next"]}},
    ).serialize())
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    connect = MagicMock(side_effect=[error, socket])
    monkeypatch.setattr(pool, "_connect", connect)
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    header = RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["text"]})
    try:
        with pytest.raises(TransferError) as caught:
            await client.request(peer, header, recv=None)
        assert type(caught.value) is TransferError
        assert caught.value.__cause__ is error
        _, data = await asyncio.wait_for(client.request(peer, header, recv=None), timeout=2)
        assert data == {"text": ["next"]}
        assert connect.call_count == 2
    finally:
        client.shutdown()
        pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", list(Operation))
@pytest.mark.parametrize("response_kind", ["other_success", "other_error", "generic_error", "operation_error"])
async def test_unexpected_response_discards_socket(monkeypatch, operation, response_kind):
    other = Operation.PUT if operation == Operation.GET else Operation.GET
    response_type = {
        "other_success": RESPONSE_TYPES[other],
        "other_error": ERROR_TYPES[other],
        "generic_error": ZMQRequestType.PUT_GET_ERROR,
        "operation_error": ZMQRequestType.PUT_GET_OPERATION_ERROR,
    }[response_kind]
    await _check_protocol_failure(
        monkeypatch, operation, response_type, {"data": {"text": ["wrong"]}}, "unexpected response",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{}, {"data": None}, {"data": ["not field data"]}])
async def test_malformed_get_response_discards_socket(monkeypatch, body):
    await _check_protocol_failure(
        monkeypatch, Operation.GET, ZMQRequestType.GET_DATA_RESPONSE, body, "GET response must contain field data",
    )


async def _check_protocol_failure(monkeypatch, operation, response_type, body, match):
    peer = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    stale = ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA_RESPONSE, sender_id="unit", body={"data": {"text": ["key-10"]}},
    ).serialize()
    failed = MagicMock(closed=False)
    failed.send_multipart = AsyncMock()
    failed.recv_multipart = AsyncMock(side_effect=[
        ZMQMessage.create(request_type=response_type, sender_id="unit", body=body).serialize(), stale,
    ])
    healthy = MagicMock(closed=False)
    healthy.send_multipart = AsyncMock()
    healthy.recv_multipart = AsyncMock(return_value=ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA_RESPONSE, sender_id="unit", body={"data": {"text": ["key-99"]}},
    ).serialize())
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    connect = MagicMock(side_effect=[failed, healthy])
    monkeypatch.setattr(pool, "_connect", connect)
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    try:
        with pytest.raises(TransferError, match=match):
            await client.request(
                peer, RequestHeader(operation, {"global_indexes": [10], "fields": ["text"]}),
                send=None if operation == Operation.GET else {"text": ["input"]},
                recv=None if operation == Operation.GET else (),
            )
        failed.close.assert_called_once_with(linger=0)
        for _ in range(2):
            _, data = await asyncio.wait_for(client.request(
                peer, RequestHeader(Operation.GET, {"global_indexes": [99], "fields": ["text"]}), recv=None,
            ), timeout=2)
            assert data == {"text": ["key-99"]}
        assert failed.recv_multipart.await_count == 1
        assert connect.call_count == 2
        healthy.close.assert_not_called()
    finally:
        client.shutdown()
        pool.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", list(Operation))
async def test_storage_rejection_keeps_socket_reusable(monkeypatch, operation):
    peer = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    message = f"[{KEY_NOT_FOUND_MARKER}] missing" if operation == Operation.GET else "write rejected"
    socket = MagicMock(closed=False)
    socket.send_multipart = AsyncMock()
    socket.recv_multipart = AsyncMock(side_effect=[
        ZMQMessage.create(request_type=ERROR_TYPES[operation], sender_id="unit", body={"message": message}).serialize(),
        ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA_RESPONSE, sender_id="unit", body={"data": {"text": ["next"]}},
        ).serialize(),
    ])
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    connect = MagicMock(return_value=socket)
    monkeypatch.setattr(pool, "_connect", connect)
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    try:
        with pytest.raises(StorageRejected) as caught:
            await client.request(
                peer, RequestHeader(operation, {"global_indexes": [0], "fields": ["text"]}),
                send=None if operation == Operation.GET else {"text": ["input"]},
                recv=None if operation == Operation.GET else (),
            )
        assert str(caught.value) == message
        assert caught.value.missing_key == (operation == Operation.GET)
        socket.close.assert_not_called()
        _, data = await asyncio.wait_for(client.request(
            peer, RequestHeader(Operation.GET, {"global_indexes": [1], "fields": ["text"]}), recv=None,
        ), timeout=2)
        assert data == {"text": ["next"]}
        assert connect.call_count == 1
        socket.close.assert_not_called()
    finally:
        client.shutdown()
        pool.close()


@pytest.mark.asyncio
async def test_update_lost_reply_is_not_replayed_through_payload_adapter(monkeypatch):
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    unit.storage_data.put_data({"value": [torch.tensor([1])]}, [0])
    peer = unit.get_zmq_server_info()
    socket = MagicMock(closed=False)
    applied = []

    async def send(frames, **kwargs):
        applied.append(unit.payload_transfer.handle_request(ZMQMessage.deserialize(frames)))

    socket.send_multipart = AsyncMock(side_effect=send)
    socket.recv_multipart = AsyncMock(side_effect=zmq.Again())
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    monkeypatch.setattr(pool, "_connect", MagicMock(return_value=socket))
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_manager_id = "manager"
    manager.storage_unit_infos = {peer.id: peer}
    manager.config = {}
    manager._payload_connections = {}
    manager._payload_connect_locks = {}
    manager.payload_transfer = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    manager.close = lambda: None
    merge = MagicMock(side_effect=lambda old, new: torch.cat([old, new]))
    try:
        with pytest.raises(StorageUnitTimeout):
            await manager._update_to_single_storage_unit(
                [0],
                {"value": [torch.tensor([2])]},
                merge,
                {"value": {"held": [True], "dtype": torch.int64, "is_non_tensor": False}},
                peer.id,
            )
        assert len(applied) == 1
        assert applied[0].request_type == ZMQRequestType.UPDATE_DATA_RESPONSE
        torch.testing.assert_close(unit.storage_data.get_data(["value"], [0])["value"][0], torch.tensor([1, 2]))
        socket.send_multipart.assert_awaited_once()
        socket.close.assert_called_once_with(linger=0)
    finally:
        manager.payload_transfer.shutdown()
        pool.close()
        unit.shutdown()


@pytest.mark.asyncio
async def test_zmq_field_data_roundtrip(monkeypatch):
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    context = zmq.asyncio.Context()
    pool = ZMQSocketPool(context, f"{STORAGE_MANAGER_IDENTITY_PREFIX}{uuid4().hex[:8]}", "put_get_socket", timeout=3)
    client = ZmqPayloadTransfer(owner_id="test-manager", pool=pool)
    try:
        connection = await client.connect(unit.get_payload_peer_info())
        put = RequestHeader(Operation.PUT, {"global_indexes": [0]})
        _, ack = await client.request(connection, put, send={"value": [torch.tensor([1, 2])], "text": ["first"]})
        assert ack is None

        update = RequestHeader(
            Operation.UPDATE,
            {
                "global_indexes": [0],
                "merge_fn": lambda old, new: torch.cat([old, new]),
                "merge_plan": {"value": {"held": [True], "dtype": torch.int64, "is_non_tensor": False}},
            },
        )
        reply, ack = await client.request(connection, update, send={"value": [torch.tensor([3])]})
        assert ack is None
        assert tuple(reply.body["stored_shapes"]["value"]["shapes"][0]) == (3,)

        get = RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["value", "text"]})
        _, data = await client.request(connection, get, recv=None)
        torch.testing.assert_close(data["value"][0], torch.tensor([1, 2, 3]))
        assert data["text"] == ["first"]

        with pytest.raises(NotImplementedError, match="receive destinations"):
            await client.request(connection, get, recv=(memoryview(bytearray(24)),))
        assert unit._arrivals_by_op["GET_DATA"] == 1
    finally:
        client.shutdown()
        pool.close()
        context.term()
        unit.shutdown()
