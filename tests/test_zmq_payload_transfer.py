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
import errno
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Event
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import torch
import zmq.asyncio
from omegaconf import OmegaConf

from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager, StorageUnitTimeout
from transfer_queue.storage.payload_transfer import (
    Operation,
    PayloadPeerInfo,
    PayloadReply,
    RequestHeader,
    StorageRejected,
    TransferError,
    ZmqPayloadTransfer,
)
from transfer_queue.storage.payload_transfer.zmq import KEY_NOT_FOUND_MARKER
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
    create_zmq_socket,
    format_zmq_address,
    get_free_port,
)


@asynccontextmanager
async def _running_zmq_transfer(monkeypatch):
    """Start one local SimpleStorageUnit and a ZmqPayloadTransfer client sharing a ZMQSocketPool."""
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    context = zmq.asyncio.Context()
    pool = ZMQSocketPool(context, f"{STORAGE_MANAGER_IDENTITY_PREFIX}{uuid4().hex[:8]}", "put_get_socket", timeout=3)
    client = ZmqPayloadTransfer(owner_id="test-manager", pool=pool)
    try:
        yield client, unit.get_payload_peer_info(), unit
    finally:
        client.shutdown()
        pool.close()
        context.term()
        unit.shutdown()


def test_payload_peer_info_is_opaque_in_omegaconf_and_compact_in_logs():
    """PayloadPeerInfo avoids OmegaConf ${...} interpolation and summarizes binary descriptors."""
    conf = OmegaConf.create({}, flags={"allow_objects": True})
    raw_descriptor = b"\x02\x00\x1f\xa4" * 1024
    peers = {
        "su_str": PayloadPeerInfo("custom", "endpoint-${unclosed"),
        "su_bin": PayloadPeerInfo("nixl", raw_descriptor),
    }
    conf.payload_peers = peers

    assert conf.payload_peers["su_str"] == PayloadPeerInfo("custom", "endpoint-${unclosed")
    assert conf.payload_peers["su_bin"].address == raw_descriptor
    rendered = f"TransferQueue config: {conf}"
    assert "<4096 bytes>" in rendered
    assert len(rendered) < 256


@pytest.mark.asyncio
async def test_payload_roundtrip_dense_nested_and_object_fields(monkeypatch):
    async with _running_zmq_transfer(monkeypatch) as (client, peer_info, unit):
        put = RequestHeader(Operation.PUT, {"global_indexes": [10, 2]})
        payload = {
            "dense": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
            "ragged": [torch.tensor([10, 20]), torch.tensor([30, 40, 50])],
            "text": ["first", "second"],
        }
        put_reply = await client.request(peer_info, put, send=payload)
        assert put_reply.payload is None

        update = RequestHeader(
            Operation.UPDATE,
            {
                "global_indexes": [2],
                "merge_fn": lambda old, new: torch.cat([old, new]),
                "merge_plan": {"ragged": {"held": [True], "dtype": torch.int64, "is_non_tensor": False}},
            },
        )
        update_reply = await client.request(peer_info, update, send={"ragged": [torch.tensor([60])]})
        assert update_reply.payload is None
        assert tuple(update_reply.meta["stored_shapes"]["ragged"]["shapes"][0]) == (4,)

        # Read in reversed sample order and verify earlier tensor views survive later writes.
        get = RequestHeader(Operation.GET, {"global_indexes": [2, 10], "fields": ["dense", "ragged", "text"]})
        get_reply = await client.request(peer_info, get)
        data = get_reply.payload
        assert data is not None
        held_view = data["ragged"][0][1:3]
        torch.testing.assert_close(
            data["dense"],
            [torch.tensor([4, 5, 6], dtype=torch.int32), torch.tensor([1, 2, 3], dtype=torch.int32)],
        )
        torch.testing.assert_close(data["ragged"][0], torch.tensor([30, 40, 50, 60]))
        torch.testing.assert_close(data["ragged"][1], torch.tensor([10, 20]))
        assert data["text"] == ["second", "first"]

        await client.request(
            peer_info,
            RequestHeader(Operation.PUT, {"global_indexes": [2]}),
            send={"ragged": [torch.tensor([99])]},
        )
        torch.testing.assert_close(held_view, torch.tensor([40, 50]))

        for recv_target in ((), (memoryview(bytearray(24)),)):
            with pytest.raises(NotImplementedError, match="receive destinations"):
                await client.request(peer_info, get, recv=recv_target)
        assert unit._arrivals_by_op["GET_DATA"] == 1


@pytest.mark.asyncio
async def test_payload_parsers_and_missing_key_preserve_connection_health(monkeypatch):
    async with _running_zmq_transfer(monkeypatch) as (client, peer_info, unit):

        def scale_put(fields):
            fields["tokens"] = [t * 2 for t in fields["tokens"]]
            return fields

        await client.request(
            peer_info,
            RequestHeader(Operation.PUT, {"global_indexes": [0, 1], "data_parser": scale_put}),
            send={"tokens": [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]},
        )

        def slice_get(fields, limits):
            fields["tokens"] = [t[:n] for t, n in zip(fields["tokens"], limits, strict=True)]
            return fields

        parsed = (
            await client.request(
                peer_info,
                RequestHeader(
                    Operation.GET,
                    {"global_indexes": [1, 0], "fields": ["tokens"], "data_parser": slice_get, "parser_args": [2, 1]},
                ),
            )
        ).payload
        assert parsed is not None
        torch.testing.assert_close(parsed["tokens"][0], torch.tensor([8, 10]))
        torch.testing.assert_close(parsed["tokens"][1], torch.tensor([2]))

        def broken_parser(_fields):
            raise AssertionError()

        with pytest.raises(StorageRejected, match="AssertionError"):
            await client.request(
                peer_info,
                RequestHeader(Operation.PUT, {"global_indexes": [3], "data_parser": broken_parser}),
                send={"tokens": [torch.tensor([1])]},
            )

        unit.storage_data.clear([1])
        with pytest.raises(StorageRejected) as missing:
            await client.request(peer_info, RequestHeader(Operation.GET, {"global_indexes": [1], "fields": ["tokens"]}))
        assert missing.value.missing_key is True

        surviving = (
            await client.request(peer_info, RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["tokens"]}))
        ).payload
        assert surviving is not None
        torch.testing.assert_close(surviving["tokens"][0], torch.tensor([2, 4, 6]))


@pytest.mark.asyncio
async def test_cancelled_request_isolates_late_reply_from_next_request():
    """Cancelling an in-flight ZMQ request discards its socket so its late reply never reaches the next caller."""
    context = zmq.asyncio.Context()
    ip = "127.0.0.1"
    port = get_free_port(ip=ip)
    router = create_zmq_socket(context, zmq.ROUTER, ip)
    router.bind(format_zmq_address(ip, port))
    info = ZMQServerInfo(role=Role.STORAGE, id="unit", ip=ip, ports={"put_get_socket": port})
    peer_info = PayloadPeerInfo("zmq", info)
    pool = ZMQSocketPool(context, f"{STORAGE_MANAGER_IDENTITY_PREFIX}cancel", "put_get_socket", timeout=3, maxsize=1)
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)

    first_arrived = asyncio.Event()
    allow_late_reply = asyncio.Event()

    async def serve_two():
        frames = await router.recv_multipart(copy=False)
        first_identity = frames[0]
        first_arrived.set()
        await allow_late_reply.wait()
        stale = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA_RESPONSE,
            sender_id="unit",
            body={"data": {"text": ["stale-late-reply"]}},
        ).serialize()
        await router.send_multipart([first_identity, *stale], copy=False)

        second_frames = await router.recv_multipart(copy=False)
        second_identity = second_frames[0]
        fresh = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA_RESPONSE,
            sender_id="unit",
            body={"data": {"text": ["fresh-reply"]}},
        ).serialize()
        await router.send_multipart([second_identity, *fresh], copy=False)

    server = asyncio.create_task(serve_two())
    try:
        first_task = asyncio.create_task(
            client.request(peer_info, RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["text"]}))
        )
        await asyncio.wait_for(first_arrived.wait(), timeout=2)
        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task

        allow_late_reply.set()
        reply = await asyncio.wait_for(
            client.request(peer_info, RequestHeader(Operation.GET, {"global_indexes": [1], "fields": ["text"]})),
            timeout=2,
        )
        assert reply.payload == {"text": ["fresh-reply"]}
        await asyncio.wait_for(server, timeout=2)
    finally:
        server.cancel()
        client.shutdown()
        pool.close()
        router.close(linger=0)
        context.term()


@pytest.mark.asyncio
async def test_queued_puts_do_not_serialize_before_socket_lease(monkeypatch):
    """Queued writers wait on the pool semaphore before materializing contiguous copies."""
    info = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    peer_info = PayloadPeerInfo("zmq", info)
    send_entered, release_send = asyncio.Event(), asyncio.Event()
    serialized_count = 0
    original_serialize = ZMQMessage.serialize

    def counting_serialize(msg):
        nonlocal serialized_count
        serialized_count += 1
        return original_serialize(msg)

    async def blocked_send(_frames, **_kwargs):
        send_entered.set()
        await release_send.wait()

    socket = MagicMock(closed=False)
    socket.send_multipart = AsyncMock(side_effect=blocked_send)
    socket.recv_multipart = AsyncMock(
        return_value=ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA_RESPONSE, sender_id="unit", body={}
        ).serialize()
    )
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    monkeypatch.setattr(pool, "_connect", MagicMock(return_value=socket))
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)

    batch_tensor = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    strided_view = batch_tensor[::2]
    assert not strided_view.is_contiguous()

    try:
        with patch.object(ZMQMessage, "serialize", counting_serialize):
            tasks = [
                asyncio.create_task(
                    client.request(
                        peer_info,
                        RequestHeader(Operation.PUT, {"global_indexes": [i]}),
                        send={"x": strided_view},
                    )
                )
                for i in range(4)
            ]
            await asyncio.wait_for(send_entered.wait(), timeout=2)
            await asyncio.sleep(0.01)
            assert serialized_count == 1
            release_send.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
            assert serialized_count == 4
    finally:
        client.shutdown()
        pool.close()


@pytest.mark.asyncio
async def test_missing_key_wire_compatibility_both_directions(monkeypatch):
    """Preserve [TQKeyNotFound] wire compatibility between old and new clients/storage units."""
    info = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    peer_info = PayloadPeerInfo("zmq", info)

    storage = StorageUnitData()
    storage.put_data({"x": [torch.tensor([1])]}, [99])
    server_transfer = ZmqPayloadTransfer(owner_id="unit")
    server_transfer.listen(handle=lambda hdr, data: handle_storage_request("unit", storage, hdr, data))
    raw_reply = server_transfer.handle_request(
        ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id="old-client",
            body={"fields": ["x"], "global_indexes": [0]},
        )
    )
    assert raw_reply.request_type == ZMQRequestType.GET_ERROR
    assert raw_reply.body["message"].startswith(f"[{KEY_NOT_FOUND_MARKER}] ")

    legacy_reply = ZMQMessage.create(
        request_type=ZMQRequestType.GET_ERROR,
        sender_id="unit",
        body={"message": f"[{KEY_NOT_FOUND_MARKER}] Failed to get data from storage unit id #unit, detail: key 0"},
    ).serialize()
    socket = MagicMock(closed=False)
    socket.send_multipart = AsyncMock()
    socket.recv_multipart = AsyncMock(return_value=legacy_reply)
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    monkeypatch.setattr(pool, "_connect", MagicMock(return_value=socket))
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    try:
        with pytest.raises(StorageRejected) as caught:
            await client.request(peer_info, RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["x"]}))
        assert caught.value.missing_key is True
        assert KEY_NOT_FOUND_MARKER not in str(caught.value)
    finally:
        client.shutdown()
        pool.close()


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
        handle = unit.payload_transfer._handle
        handle(RequestHeader(Operation.PUT, {"global_indexes": [0]}), {"value": [torch.tensor([1])]})
        reply = handle(RequestHeader(Operation.GET, {"global_indexes": [0], "fields": ["value"]}), None)
        assert reply.payload is not None
        torch.testing.assert_close(reply.payload["value"][0], torch.tensor([1]))
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
            concurrent_put.result(timeout=2)
            assert concurrent_clear.result(timeout=2).request_type == ZMQRequestType.CLEAR_DATA_RESPONSE
            metrics_reply = executor.submit(unit._handle_get_metrics).result(timeout=2)
            assert metrics_reply.body["active_keys"] == 2

            release_parser.set()
            parsed = slow_get.result(timeout=2).payload
            assert parsed is not None
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
        return PayloadReply(payload={"text": ["done"]})

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
async def test_socket_open_failure_recovers_on_next_request(monkeypatch):
    info = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    peer_info = PayloadPeerInfo("zmq", info)
    healthy = MagicMock(closed=False)
    healthy.send_multipart = AsyncMock()
    healthy.recv_multipart = AsyncMock(
        return_value=ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA_RESPONSE,
            sender_id="unit",
            body={},
        ).serialize()
    )
    pool = ZMQSocketPool(MagicMock(), "manager", "put_get_socket", maxsize=1)
    monkeypatch.setattr(pool, "_connect", MagicMock(side_effect=[zmq.ZMQError(errno.EMFILE), healthy]))
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    try:
        with pytest.raises(TransferError):
            await client.request(
                peer_info,
                RequestHeader(Operation.PUT, {"global_indexes": [0]}),
                send={"text": ["input"]},
            )

        reply = await asyncio.wait_for(
            client.request(peer_info, RequestHeader(Operation.PUT, {"global_indexes": [0]}), send={"text": ["input"]}),
            timeout=2,
        )
        assert reply.payload is None
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
    manager._payload_peers = {peer.id: unit.get_payload_peer_info()}
    manager._payload_closed = False
    manager.payload_transfer = ZmqPayloadTransfer(owner_id="manager", pool=pool)
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
    info = ZMQServerInfo(role=Role.STORAGE, id="unit", ip="127.0.0.1", ports={"put_get_socket": 5555})
    peer_info = PayloadPeerInfo("zmq", info)
    pool = MagicMock()
    client = ZmqPayloadTransfer(owner_id="manager", pool=pool)
    with pytest.raises(ValueError, match="does not accept send"):
        await client.request(peer_info, RequestHeader(Operation.GET, {"fields": ["x"]}), send={"x": [1]})
    for write_op in (Operation.PUT, Operation.UPDATE):
        with pytest.raises(ValueError, match="requires send"):
            await client.request(peer_info, RequestHeader(write_op, {"global_indexes": [0]}))
    pool.alease.assert_not_called()
