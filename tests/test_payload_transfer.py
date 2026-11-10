"""Local ZMQ and simulated NIXL binding checks for the shared request contract."""

import asyncio
import ctypes
import json
import sys
import threading
from collections import defaultdict
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
import zmq.asyncio

from transfer_queue.storage.payload_transfer import create_payload_transfer
from transfer_queue.storage.payload_transfer.base import (
    Operation,
    RequestHeader,
    StorageRejected,
    TransferError,
    WriteOutcomeUnknown,
    decode_spans,
    encode_spans,
)
from transfer_queue.storage.payload_transfer.nixl_ucx_runtime import NixlRuntime, ReadAccessUnknown
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.serial_utils import decode, unpack_from
from transfer_queue.utils.zmq_utils import STORAGE_MANAGER_IDENTITY_PREFIX, ZMQSocketPool


@pytest.fixture
def native_bus(monkeypatch):
    """Exercise the real runtime against a binding that checks registrations."""
    agents = {}
    bus = SimpleNamespace(lock=threading.Lock(), drop=None, deferred=False, reads=0, read_started=threading.Event())

    class Agent:
        backends = {"UCX": 1}

        def __init__(self, name, config):
            self.name = name
            self.regions = {}
            self.notifications = defaultdict(list)
            self.peers = {}
            agents[name] = self

        def get_agent_metadata(self):
            return json.dumps([self.name, []]).encode()

        def get_partial_agent_metadata(self, registration, **kwargs):
            return json.dumps([self.name, registration]).encode()

        def add_remote_agent(self, metadata):
            peer, regions = json.loads(metadata)
            self.peers.setdefault(peer, {}).update((addr, size) for addr, size, *_ in regions)
            return peer

        def remove_remote_agent(self, peer):
            self.peers.pop(peer)

        def send_notif(self, peer, packed):
            if bus.drop is not None and bus.drop(self.name, peer, packed):
                return
            with bus.lock:
                agents[peer].notifications[self.name].append(packed)

        def get_new_notifs(self, **kwargs):
            with bus.lock:
                messages = dict(self.notifications)
                self.notifications.clear()
            return messages

        def register_memory(self, regions, **kwargs):
            for addr, size, *_ in regions:
                assert addr not in self.regions
                self.regions[addr] = size
            return regions

        def deregister_memory(self, regions, **kwargs):
            for addr, *_ in regions:
                self.regions.pop(addr)

        def get_xfer_descs(self, regions, **kwargs):
            return regions

        def initialize_xfer(self, operation, local, remote, peer, **kwargs):
            assert operation == "READ"
            return local[0], remote[0], peer

        def transfer(self, handle):
            if bus.deferred:
                bus.read_started.set()
                return "PROC"
            (dst, size, _), (src, remote_size, _), peer = handle
            assert remote_size == size <= self.peers[peer][src]
            assert size <= self.regions[dst]
            assert size <= agents[peer].regions[src]
            ctypes.memmove(dst, src, size)
            bus.reads += 1
            return "DONE"

        def check_xfer_state(self, handle):
            return self.transfer(handle)

        def release_xfer_handle(self, handle):
            if bus.deferred:
                raise RuntimeError("READ still in flight")

    monkeypatch.setitem(sys.modules, "nixl", SimpleNamespace(nixl_agent=Agent, nixl_agent_config=dict))
    bus.agents = agents
    return bus


@pytest.fixture
def storage_pair(monkeypatch, native_bus):
    monkeypatch.setattr("transfer_queue.storage.simple_storage.get_node_ip_address", lambda: "127.0.0.1")
    created = []

    def create(backend, timeout=3):
        config = {"backend": backend}
        unit = SimpleStorageUnit.__ray_metadata__.modified_class({"payload_transfer": config})
        ctx = zmq.asyncio.Context()
        pool = ZMQSocketPool(ctx, f"{STORAGE_MANAGER_IDENTITY_PREFIX}{uuid4().hex[:8]}", "put_get_socket", timeout=3)
        client = create_payload_transfer(config, owner_id="test-manager", pool=pool, timeout=timeout)
        created.append((unit, client, pool, ctx))
        return unit, client

    yield create
    for unit, client, pool, ctx in reversed(created):
        client.shutdown()
        unit.shutdown()
        pool.close()
        ctx.term()


def header(op, **body):
    return RequestHeader(op, {"global_indexes": [0], **body})


def update_header():
    return header(
        Operation.UPDATE,
        merge_fn=lambda old, new: torch.cat([old, new]),
        merge_plan={"value": {"held": [True], "dtype": torch.int64, "is_non_tensor": False}},
    )


@pytest.mark.parametrize("backend", ["zmq", "nixl-ucx"])
def test_payload_contract(storage_pair, native_bus, backend):
    unit, client = storage_pair(backend)
    if backend == "nixl-ucx":
        # The client must not need a SimpleStorage ZMQ payload lease.
        assert not hasattr(client, "pool")

    async def run():
        peer = await client.connect(unit.get_payload_transfer_info())
        await client.request(
            peer,
            header(Operation.PUT, data_parser=lambda data: {"value": [data["value"][0] + 1]}),
            send=encode_spans({"value": [torch.arange(8)]}),
        )
        reply, _ = await client.request(peer, update_header(), send=encode_spans({"value": [torch.tensor([9])]}))
        assert tuple(reply.body["stored_shapes"]["value"]["shapes"][0]) == (9,)

        request = header(
            Operation.GET,
            fields=["value"],
            parser_args=[10],
            data_parser=lambda data, args: {"value": [data["value"][0] + args[0]]},
        )
        _, initial = await client.request(peer, request, recv=None)
        expected = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18, 19])
        torch.testing.assert_close(decode_spans(initial)["value"][0], expected)
        destinations = tuple(memoryview(bytearray(span.nbytes)) for span in initial)
        _, result = await client.request(peer, request, recv=destinations)
        assert result is destinations
        torch.testing.assert_close(decode_spans(result)["value"][0], expected)

        too_short = (memoryview(bytearray(b"x")),)
        with pytest.raises(TransferError, match="layout"):
            await client.request(peer, request, recv=too_short)
        assert bytes(too_short[0]) == b"x"

        with pytest.raises(StorageRejected, match="must not change dict keys"):
            await client.request(
                peer, header(Operation.GET, fields=["value"], data_parser=lambda *_: {"bad": [1]}), recv=None
            )
        results = await asyncio.gather(*(client.request(peer, request, recv=None) for _ in range(5)))
        for _, spans in results:
            torch.testing.assert_close(decode_spans(spans)["value"][0], expected)
        torch.testing.assert_close(decode_spans(initial)["value"][0], expected)

    asyncio.run(run())
    assert unit._arrivals_by_op["UPDATE_DATA"] == 1
    if backend == "nixl-ucx":
        assert native_bus.reads == 10
        assert not client._exports


def test_nixl_lost_update_reply_holds_source_and_never_replays(storage_pair, native_bus):
    unit, client = storage_pair("nixl-ucx", timeout=0.3)
    dropped = []

    def drop(sender, peer, packed):
        envelope = decode(unpack_from(packed))
        if envelope["kind"] == "reply" and envelope.get("body", {}).get("stored_shapes"):
            dropped.append((sender, peer, packed))
            return True
        return False

    native_bus.drop = drop

    async def run():
        peer = await client.connect(unit.get_payload_transfer_info())
        await client.request(peer, header(Operation.PUT), send=encode_spans({"value": [torch.tensor([1])]}))
        with pytest.raises(WriteOutcomeUnknown):
            await client.request(peer, update_header(), send=encode_spans({"value": [torch.tensor([2])]}))
        assert len(client._exports) == 1
        sender, target, packed = dropped[0]
        native_bus.drop = None
        native_bus.agents[sender].send_notif(target, packed)
        _, result = await client.request(peer, header(Operation.GET, fields=["value"]), recv=None)
        torch.testing.assert_close(decode_spans(result)["value"][0], torch.tensor([1, 2]))
        assert not client._exports

    asyncio.run(run())
    assert unit._arrivals_by_op["UPDATE_DATA"] == 1
    assert len(dropped) == 1


def test_native_read_keeps_registration_when_release_fails(native_bus):
    source = NixlRuntime()
    destination = NixlRuntime(timeout=0.01)
    offer = source.publish(3, lambda buf: buf.__setitem__(slice(0, 3), b"abc"))
    try:
        native_bus.deferred = True
        with pytest.raises(ReadAccessUnknown):
            destination.read(source.endpoint(), offer)
        assert len(native_bus.agents[destination.name].regions) == 1
        with pytest.raises(TransferError, match="unresolved READ"):
            destination.read(source.endpoint(), offer)
    finally:
        native_bus.deferred = False
        destination.close()
        source.close()


def test_cancelled_get_keeps_offer_until_read_stops(storage_pair, native_bus):
    unit, client = storage_pair("nixl-ucx")

    async def run():
        peer = await client.connect(unit.get_payload_transfer_info())
        await client.request(peer, header(Operation.PUT), send=encode_spans({"value": [torch.tensor([1])]}))
        native_bus.deferred = True
        get = asyncio.create_task(client.request(peer, header(Operation.GET, fields=["value"]), recv=None))
        try:
            assert await asyncio.to_thread(native_bus.read_started.wait, 2)
            get.cancel()
            with pytest.raises(asyncio.CancelledError):
                await get
            assert len(unit.payload_transfer._exports) == 1
            assert len(client._active_reads) == 1
        finally:
            native_bus.deferred = False
        deadline = asyncio.get_running_loop().time() + 2
        while unit.payload_transfer._exports:
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)

    asyncio.run(run())


def test_manager_routes_nixl_requests_without_a_zmq_lease(storage_pair, monkeypatch):
    from transfer_queue.storage.managers.base import StorageManager
    from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager

    unit, _ = storage_pair("nixl-ucx")
    context = zmq.asyncio.Context()

    def initialize_base(self, controller_info, config, **kwargs):
        self.zmq_context = context
        self.storage_manager_id = "test-manager"

    monkeypatch.setattr(StorageManager, "__init__", initialize_base)
    monkeypatch.setattr(StorageManager, "close", lambda self: None)
    info = unit.get_zmq_server_info()
    manager = AsyncSimpleStorageManager(
        None,
        {
            "zmq_info": info,
            "payload_transfer": {"backend": "nixl-ucx"},
            "payload_transfer_infos": {info.id: unit.get_payload_transfer_info()},
        },
    )

    def reject_lease(*args):
        pytest.fail("NIXL data request borrowed a ZMQ socket")

    monkeypatch.setattr(manager.storage_rpc_pool, "alease", reject_lease)

    async def run():
        await manager._put_to_single_storage_unit([0], {"value": [torch.tensor([5])]}, info.id)
        _, values = await manager._get_from_single_storage_unit([0], ["value"], info.id)
        torch.testing.assert_close(values["value"][0], torch.tensor([5]))

    try:
        asyncio.run(run())
    finally:
        manager.close()
        context.term()
