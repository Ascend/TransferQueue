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

"""End-to-end contract, concurrency, and fault-isolation tests for ZmqPayloadTransfer."""

import asyncio
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Event
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import torch
import zmq.asyncio

from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager, StorageUnitTimeout
from transfer_queue.storage.payload_transfer import (
    Operation,
    RequestHeader,
    StorageRejected,
    TransferError,
    TransferTimeout,
    ZmqPayloadTransfer,
)
from transfer_queue.storage.simple_storage import (
    SimpleStorageUnit,
    StorageUnitData,
    handle_storage_request,
)
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import (
    STORAGE_MANAGER_IDENTITY_PREFIX,
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    ZMQSocketPool,
)


@asynccontextmanager
async def _running_zmq_transfer(monkeypatch):
    """Start one local SimpleStorageUnit and a connected ZmqPayloadTransfer client."""
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    context = zmq.asyncio.Context()
    pool = ZMQSocketPool(context, f"{STORAGE_MANAGER_IDENTITY_PREFIX}{uuid4().hex[:8]}", "put_get_socket", timeout=3)
    client = ZmqPayloadTransfer(owner_id="test-manager", pool=pool)
    try:
        connection = await client.connect(unit.get_payload_peer_info())
        yield client, connection, unit
    finally:
        client.shutdown()
        pool.close()
        context.term()
        unit.shutdown()


@pytest.mark.asyncio
async def test_payload_roundtrip_dense_nested_and_object_fields(monkeypatch):
    async with _running_zmq_transfer(monkeypatch) as (client, connection, unit):
        put = RequestHeader(Operation.PUT, {"global_indexes": [10, 2]})
        payload = {
            "dense": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            "ragged": [torch.tensor([10, 20]), torch.tensor([30, 40, 50])],
            "text": ["first", "second"],
        }
        _, ack = await client.request(connection, put, send=payload)
        assert ack is None

        update = RequestHeader(
            Operation.UPDATE,
            {
                "global_indexes": [2],
                "merge_fn": lambda old, new: torch.cat([old, new]),
                "merge_plan": {"ragged": {"held": [True], "dtype": torch.int64, "is_non_tensor": False}},
            },
        )
        reply, ack = await client.request(connection, update, send={"ragged": [torch.tensor([60])]})
        assert ack is None
        assert tuple(reply.body["stored_shapes"]["ragged"]["shapes"][0]) == (4,)

        # Read in reversed sample order and verify earlier tensor views survive later writes.
        get = RequestHeader(Operation.GET, {"global_indexes": [2, 10], "fields": ["dense", "ragged", "text"]})
        _, data = await client.request(connection, get)
        held_view = data["ragged"][0][1:3]
        torch.testing.assert_close(
            data["dense"], [torch.tensor([4, 5, 6], dtype=torch.int32), torch.tensor([1, 2, 3], dtype=torch.int32)]
        )
        torch.testing.assert_close(data["ragged"][0], torch.tensor([30, 40, 50, 60]))
        torch.testing.assert_close(data["ragged"][1], torch.tensor([10, 20]))
        assert data["text"] == ["second", "first"]

        await client.request(
            connection,
            RequestHeader(Operation.PUT, {"global_indexes": [2]}),
            send={"ragged": [torch.tensor([99])]},
        )
        torch.testing.assert_close(held_view, torch.tensor([40, 50]))

        for recv_target in ((), (memoryview(bytearray(24)),)):
            with pytest.raises(NotImplementedError, match="receive destinations"):
                await client.request(connection, get, recv=recv_target)
        assert unit._arrivals_by_op["GET_DATA"] == 1


@pytest.mark.asyncio
async def test_payload_parsers_and_missing_key_preserve_connection_health(monkeypatch):
    async with _running_zmq_transfer(monkeypatch) as (client, connection, unit):

        def scale_put(fields):
            fields["tokens"] = [t * 2 for t in fields["tokens"]]
            return fields

        await client.request(
            connection,
            RequestHeader(Operation.PUT, {"global_indexes": [0, 1], "data_parser": scale_put}),
            send={"tokens": [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]},
        )

        def slice_get(fields, limits):
            fields["tokens"] = [t[:n] for t, n in zip(fields["tokens"], limits, strict=True)]
            return fields

        _, parsed = await client.request(
            connection,
            RequestHeader(
                Operation.GET,
                {"global_indexes": [1, 0], "fields": ["tokens"], "data_parser": slice_get, "parser_args": [2, 1]},
            ),
        )
        torch.testing.assert_close(parsed["tokens"][0], torch.tensor([8, 10]))
        torch.testing.assert_close(parsed["tokens"][1], torch.tensor([2]))

        # A failing user parser reports its exception class without poisoning the connection.
        def broken_parser(_fields):
            raise AssertionError()

        with pytest.raises(StorageRejected, match="AssertionError"):
            await client.request(
                connection,
                RequestHeader(Operation.PUT, {"global_indexes": [3], "data_parser": broken_parser}),
                send={"tokens": [torch.tensor([1])]},
            )

        # Clearing a sample turns subsequent GETs into missing_key rejections while keeping the socket usable.
        unit.storage_data.clear([1])
        with pytest.raises(StorageRejected) as missing:
            await client.request(
                connection, RequestHeader(Operation.GET, {"global_indexes": [1], "fields": ["tokens"]})
            )
        assert missing.value.missing_key is True

        _, surviving = await client.request(
            connection, RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["tokens"]})
        )
        torch.testing.assert_close(surviving["tokens"][0], torch.tensor([2, 4, 6]))


@pytest.mark.parametrize("error", [AssertionError(), KeyError("k")])
def test_parser_rejection_preserves_exception_type(error):
    parser = MagicMock(side_effect=error)
    header = RequestHeader(Operation.PUT, {"global_indexes": [0], "data_parser": parser})
    with pytest.raises(StorageRejected, match=type(error).__name__):
        handle_storage_request("unit", StorageUnitData(), header, {"text": ["value"]})


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
        # A non-ZMQ backend invokes this listen callback directly without the ZMQ worker loop.
        handle = unit.payload_transfer._handle
        handle(RequestHeader(Operation.PUT, {"global_indexes": [0]}), {"value": [torch.tensor([1])]})
        _, data = handle(RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["value"]}), None)
        torch.testing.assert_close(data["value"][0], torch.tensor([1]))
        with pytest.raises(StorageRejected):
            handle(RequestHeader(Operation.GET, {"global_indexes": [99], "fields": ["value"]}), None)
        assert unit._requests_arrived == 3
        assert unit._arrivals_by_op == {"PUT_DATA": 1, "GET_DATA": 2}
        heavy.assert_not_called()
        if metrics_enabled:
            assert unit._metrics.request_total.labels(op_type="GET_DATA")._value.get() == 2
            assert unit._metrics.request_errors_total.labels(op_type="GET_DATA")._value.get() == 1
            hist = unit._metrics.request_duration.labels(op_type="GET_DATA")
            assert sum(bucket.get() for bucket in hist._buckets) == 2
        else:
            assert unit._perf_monitor.success_counts["GET_DATA"] == 2
    finally:
        unit.shutdown()


def test_storage_handler_allows_concurrent_admin_while_parser_runs(monkeypatch):
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    try:
        handle = unit.payload_transfer._handle
        handle(
            RequestHeader(Operation.PUT, {"global_indexes": [0, 1]}),
            {"value": [torch.tensor([1]), torch.tensor([2])]},
        )

        parser_entered, release_parser = Event(), Event()

        def slow_get_parser(data, _args):
            parser_entered.set()
            assert release_parser.wait(5)
            return {"value": [v + 10 for v in data["value"]]}

        with ThreadPoolExecutor(max_workers=4) as executor:
            slow_get = executor.submit(
                handle,
                RequestHeader(
                    Operation.GET, {"global_indexes": [0], "fields": ["value"], "data_parser": slow_get_parser}
                ),
                None,
            )
            assert parser_entered.wait(5)

            # While the GET parser runs outside _storage_lock, concurrent PUT, CLEAR, and GET_METRICS do not block.
            concurrent_put = executor.submit(
                handle,
                RequestHeader(Operation.PUT, {"global_indexes": [2]}),
                {"value": [torch.tensor([3])]},
            )
            concurrent_clear = executor.submit(
                unit._handle_clear,
                ZMQMessage.create(
                    request_type=ZMQRequestType.CLEAR_DATA, sender_id="manager", body={"global_indexes": [1]}
                ),
            )
            concurrent_metrics = executor.submit(unit._handle_get_metrics)

            concurrent_put.result(timeout=2)
            assert concurrent_clear.result(timeout=2).request_type == ZMQRequestType.CLEAR_DATA_RESPONSE
            metrics_reply = concurrent_metrics.result(timeout=2)
            assert metrics_reply.body["active_keys"] == 2

            release_parser.set()
            _, parsed = slow_get.result(timeout=2)
            torch.testing.assert_close(parsed["value"][0], torch.tensor([11]))
    finally:
        unit.shutdown()


def test_unit_drains_dispatched_work_before_stopping_worker():
    transfer = ZmqPayloadTransfer(owner_id="unit")
    entered, release, stop_worker = Event(), Event(), Event()
    storage = MagicMock()

    def handle(header, data):
        entered.set()
        assert release.wait(5)
        return {}, {"text": ["done"]}

    transfer.listen(handle=handle)
    request = ZMQMessage.create(request_type=ZMQRequestType.GET_DATA, sender_id="manager", body={})
    unit_class = SimpleStorageUnit.__ray_metadata__.modified_class
    with ThreadPoolExecutor(max_workers=3) as threads:
        first = threads.submit(transfer.handle_request, request)
        assert entered.wait(5)
        shutdown = threads.submit(
            unit_class._shutdown_resources,
            stop_worker,
            None,
            None,
            None,
            None,
            payload_transfer=transfer,
            storage_data=storage,
        )
        try:
            assert transfer._closed.wait(5)
            assert not stop_worker.is_set()
            storage.close.assert_not_called()
            late = threads.submit(transfer.handle_request, request)
        finally:
            release.set()
        assert first.result(timeout=5).body["data"] == {"text": ["done"]}
        assert late.result(timeout=5).request_type == ZMQRequestType.GET_ERROR
        shutdown.result(timeout=5)

    assert stop_worker.is_set()
    storage.close.assert_called_once_with()
    assert transfer.handle_request(request).request_type == ZMQRequestType.GET_ERROR


def test_storage_listen_callback_does_not_retain_owner(monkeypatch):
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    transfer = unit.payload_transfer
    handle = transfer._handle
    unit.shutdown()
    ref = weakref.ref(unit)
    del unit
    gc.collect()
    assert ref() is None
    assert handle is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "disconnect", "worker_error", "cancel"])
async def test_transport_fault_discards_socket_and_releases_permit(monkeypatch, failure):
    peer = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    failed = MagicMock(closed=False)
    failed.send_multipart = AsyncMock()
    if failure == "worker_error":
        failed.recv_multipart = AsyncMock(
            return_value=ZMQMessage.create(
                request_type=ZMQRequestType.PUT_GET_ERROR,
                sender_id="unit",
                body={"message": "failed to serialize response"},
            ).serialize()
        )
        expected = TransferError
    else:
        err = {
            "timeout": zmq.Again(),
            "disconnect": zmq.ZMQError(zmq.ENOTSOCK),
            "cancel": asyncio.CancelledError(),
        }[failure]
        failed.recv_multipart = AsyncMock(side_effect=err)
        expected = (
            TransferTimeout
            if failure == "timeout"
            else (TransferError if failure == "disconnect" else asyncio.CancelledError)
        )

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
    try:
        with pytest.raises(expected) as caught:
            await client.request(
                peer,
                RequestHeader(Operation.PUT, {"global_indexes": [0]}),
                send={"text": ["input"]},
            )
        if failure == "timeout":
            assert caught.value.wire_bytes > 0
        elif failure == "worker_error":
            assert "failed to serialize response" in str(caught.value)
        failed.close.assert_called_once_with(linger=0)

        # A single permit verifies both poisoned-socket eviction and permit release.
        _, data = await asyncio.wait_for(
            client.request(peer, RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["text"]})),
            timeout=2,
        )
        assert data == {"text": ["next"]}
        assert connect.call_count == 2
        healthy.close.assert_not_called()
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
    manager._payload_connecting = {}
    manager._payload_closed = False
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
async def test_request_validates_send_arguments_locally():
    peer = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    pool = MagicMock()
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    with pytest.raises(ValueError, match="does not accept send"):
        await client.request(peer, RequestHeader(Operation.GET, {"fields": ["x"]}), send={"x": [1]})
    for write_op in (Operation.PUT, Operation.UPDATE):
        with pytest.raises(ValueError, match="requires send"):
            await client.request(peer, RequestHeader(write_op, {"global_indexes": [0]}))
    pool.alease.assert_not_called()
