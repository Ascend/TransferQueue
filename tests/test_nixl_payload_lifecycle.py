# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team

"""Focused tests for the frame-native NIXL payload lifecycle."""

from __future__ import annotations

import asyncio
import gc
import socket
import sys
import threading
import time
import types
from concurrent.futures import Future
from queue import SimpleQueue
from typing import Any, Callable

import numpy as np
import pytest
import torch
import zmq

from transfer_queue.storage.payload_transfer import DeferredResponse
from transfer_queue.storage.payload_transfer.nixl import (
    NixlPayloadTransfer,
    PayloadDescriptor,
    ReceiveToken,
    TransferEndpoint,
    _PendingGet,
)
from transfer_queue.storage.payload_transfer.nixl_ucx_runtime import (
    NixlError,
    NixlRuntime,
    _frame_regions,
    _FrameSource,
    _RegisteredReceiveBuffer,
)
from transfer_queue.storage.simple_storage import _drain_deferred_responses, _queue_deferred_response
from transfer_queue.utils.serial_utils import decode, encode
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


class _FakeAgent:
    def __init__(self) -> None:
        self.backends = {"UCX": object()}
        self.register_calls: list[Any] = []
        self.deregister_calls: list[Any] = []
        self.add_remote_calls: list[bytes] = []
        self.remove_remote_calls: list[str] = []
        self.transfer_calls: list[Any] = []
        self.metadata_names: dict[bytes, str] = {}
        self.status_by_remote: dict[str, str] = {}
        self.release_error = False
        self.deregister_error = False
        self._registration_id = 0
        self._handle_id = 0

    def get_agent_metadata(self) -> bytes:
        return f"metadata-{len(self.register_calls)}".encode()

    def register_memory(self, regions: Any, **kwargs: Any) -> tuple[str, int]:
        self.register_calls.append(regions)
        self._registration_id += 1
        return ("registration", self._registration_id)

    def deregister_memory(self, registration: Any, **kwargs: Any) -> None:
        if self.deregister_error:
            raise RuntimeError("deregister failed")
        self.deregister_calls.append(registration)

    def get_xfer_descs(self, regions: Any, **kwargs: Any) -> Any:
        return list(regions)

    def get_serialized_descs(self, descs: Any) -> bytes:
        return repr(descs).encode()

    def deserialize_descs(self, descs: bytes) -> bytes:
        return descs

    def add_remote_agent(self, metadata: bytes) -> str:
        self.add_remote_calls.append(metadata)
        return self.metadata_names[metadata]

    def remove_remote_agent(self, remote_name: str) -> None:
        self.remove_remote_calls.append(remote_name)

    def initialize_xfer(self, operation: str, local: Any, remote: Any, remote_name: str, **kwargs: Any) -> Any:
        self._handle_id += 1
        return (self._handle_id, remote_name, local, remote)

    def transfer(self, handle: Any) -> str:
        self.transfer_calls.append(handle)
        return self.status_by_remote.get(handle[1], "DONE")

    def check_xfer_state(self, handle: Any) -> str:
        return self.status_by_remote.get(handle[1], "DONE")

    def release_xfer_handle(self, handle: Any) -> None:
        if self.release_error:
            raise RuntimeError("release failed")


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch):
    agents: list[_FakeAgent] = []
    module = types.ModuleType("nixl")

    def make_agent(*args: Any, **kwargs: Any) -> _FakeAgent:
        agent = _FakeAgent()
        agents.append(agent)
        return agent

    module.nixl_agent = make_agent
    module.nixl_agent_config = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "nixl", module)
    instance = NixlRuntime()
    yield instance, agents[0]
    instance.close()


def _descriptor(transfer_id: str, *sizes: int) -> PayloadDescriptor:
    return PayloadDescriptor(transfer_id, sum(sizes), tuple(sizes))


def _token(remote_name: str, metadata: bytes, descriptor: PayloadDescriptor) -> dict[str, Any]:
    return {
        "agent_name": remote_name,
        "agent_metadata": metadata,
        "frame_remote_descs": b"remote-descs",
        "payload_bytes": descriptor.payload_bytes,
    }


def test_frame_native_layout_preserves_empty_frames_and_large_offsets():
    sizes = (3 * 1024**3, 0, 3 * 1024**3, 2 * 1024**3)
    descriptor = _descriptor("large", *sizes)
    descriptor.validate()

    address = 0x1000
    regions = _frame_regions(address, descriptor.frame_sizes)

    assert descriptor.payload_bytes == 8 * 1024**3
    assert regions == [
        (address, 3 * 1024**3, 0),
        (address + 3 * 1024**3, 3 * 1024**3, 0),
        (address + 6 * 1024**3, 2 * 1024**3, 0),
    ]


def test_all_empty_frames_skip_registration_and_write(runtime):
    instance, agent = runtime
    descriptor = _descriptor("empty", 0, 0)

    token = instance.prepare_receive(descriptor)
    frames = instance.receive(descriptor).result()
    sent = instance.send({"agent_name": "peer"}, token, descriptor, (b"", b""))

    assert [bytes(frame) for frame in frames] == [b"", b""]
    assert sent.result() is None
    assert agent.register_calls == []
    assert agent.transfer_calls == []


def test_receive_frames_decode_without_copy_and_lease_until_last_tensor(runtime):
    instance, _ = runtime
    source = {"value": torch.arange(8, dtype=torch.int32)}
    encoded = tuple(encode(source))
    descriptor = _descriptor("receive", *(memoryview(frame).nbytes for frame in encoded))

    instance.prepare_receive(descriptor)
    scratch = instance._receives[descriptor.transfer_id]
    assert scratch is not None
    offset = 0
    for frame in encoded:
        view = memoryview(frame)
        scratch.buffer[offset : offset + view.nbytes] = view
        offset += view.nbytes

    frames = instance.receive(descriptor).result()
    decoded = decode(list(frames))
    detached = decoded["value"].detach()
    del frames, decoded
    gc.collect()

    assert instance._idle_receive_buffers == []
    assert torch.equal(detached, source["value"])

    del detached
    gc.collect()
    assert instance._idle_receive_buffers == [scratch]


def test_receive_pool_uses_first_usable_buffer_without_deregistering(runtime):
    instance, agent = runtime
    first = _descriptor("first", 16)
    second = _descriptor("second", 8)

    instance.prepare_receive(first)
    first_buffer = instance._receives["first"]
    instance.cancel_receive("first")
    instance.prepare_receive(second)

    assert instance._receives["second"] is first_buffer
    assert len(agent.register_calls) == 1
    assert agent.deregister_calls == []


def test_sources_reuse_leased_mr_and_external_registration_is_transfer_scoped(runtime):
    instance, agent = runtime
    descriptor = _descriptor("lease", 8)
    instance.prepare_receive(descriptor)
    leased_frame = instance.receive(descriptor).result()[0]

    reused = instance._acquire_frame_source(leased_frame)
    writable = bytearray(b"writable")
    direct = instance._acquire_frame_source(writable)
    readonly = instance._acquire_frame_source(b"readonly")
    non_contiguous = instance._acquire_frame_source(memoryview(np.arange(8, dtype=np.uint8))[::2])
    direct_again = instance._acquire_frame_source(writable)

    assert reused.registration is None
    assert isinstance(direct.owner, memoryview)
    assert isinstance(readonly.owner, bytearray)
    assert isinstance(non_contiguous.owner, bytearray)
    assert isinstance(direct_again.owner, bytearray)
    assert direct_again.address != direct.address
    assert direct.registration != direct_again.registration
    assert len(agent.register_calls) == 5  # one receive MR plus four transfer-local registrations


def test_full_metadata_reused_then_replaced_in_peer_executor(runtime):
    instance, agent = runtime
    descriptor = _descriptor("send", 1)
    agent.metadata_names.update({b"m1": "peer", b"m2": "peer"})

    instance.send({}, _token("peer", b"m1", descriptor), descriptor, (bytearray(b"a"),)).result()
    instance.send({}, _token("peer", b"m1", descriptor), descriptor, (bytearray(b"b"),)).result()
    instance.send({}, _token("peer", b"m2", descriptor), descriptor, (bytearray(b"c"),)).result()

    assert agent.add_remote_calls == [b"m1", b"m2"]
    assert agent.remove_remote_calls == ["peer"]


def test_same_peer_serializes_while_different_peer_runs_concurrently(runtime, monkeypatch: pytest.MonkeyPatch):
    instance, _ = runtime
    descriptor = _descriptor("concurrency", 1)
    first_started = threading.Event()
    other_started = threading.Event()
    release = threading.Event()
    starts: list[str] = []

    def blocking_send(remote_name: str, *args: Any) -> None:
        starts.append(remote_name)
        (first_started if remote_name == "peer-a" else other_started).set()
        release.wait(timeout=2)

    monkeypatch.setattr(instance, "_send_scatter", blocking_send)
    first = instance.send({}, _token("peer-a", b"a", descriptor), descriptor, (bytearray(b"a"),))
    assert first_started.wait(timeout=1)
    queued = instance.send({}, _token("peer-a", b"a", descriptor), descriptor, (bytearray(b"b"),))
    other = instance.send({}, _token("peer-b", b"b", descriptor), descriptor, (bytearray(b"c"),))

    assert other_started.wait(timeout=1)
    assert starts.count("peer-a") == 1
    release.set()
    first.result(timeout=1)
    queued.result(timeout=1)
    other.result(timeout=1)
    assert starts.count("peer-a") == 2


def test_completed_write_cleanup_failure_keeps_success_and_fails_peer(runtime):
    instance, agent = runtime
    descriptor = _descriptor("cleanup", 1)
    agent.metadata_names[b"metadata"] = "peer"
    agent.release_error = True

    completed = instance.send({}, _token("peer", b"metadata", descriptor), descriptor, (bytearray(b"x"),))

    assert completed.result(timeout=1) is None
    assert instance._retained_resources
    with pytest.raises(NixlError, match="has failed"):
        instance.send({}, _token("peer", b"metadata", descriptor), descriptor, (bytearray(b"y"),))


def test_failed_write_retains_resources_and_other_peer_still_works(runtime):
    instance, agent = runtime
    descriptor = _descriptor("failure", 1)
    agent.metadata_names.update({b"bad": "bad-peer", b"good": "good-peer"})
    agent.status_by_remote["bad-peer"] = "ERR"

    with pytest.raises(NixlError, match="status"):
        instance.send({}, _token("bad-peer", b"bad", descriptor), descriptor, (bytearray(b"x"),)).result()

    assert instance._retained_resources
    assert instance.send({}, _token("good-peer", b"good", descriptor), descriptor, (bytearray(b"y"),)).result() is None


def test_failed_future_does_not_retain_handle_or_sources(runtime):
    instance, agent = runtime
    descriptor = _descriptor("failure-traceback", 1)
    agent.metadata_names[b"metadata"] = "peer"
    agent.status_by_remote["peer"] = "ERR"

    future = instance.send({}, _token("peer", b"metadata", descriptor), descriptor, (bytearray(b"x"),))
    error = future.exception(timeout=1)

    assert isinstance(error, NixlError)
    traceback = error.__traceback__
    while traceback is not None and traceback.tb_frame.f_code.co_name != "_send_scatter":
        traceback = traceback.tb_next
    assert traceback is not None
    assert traceback.tb_frame.f_locals["handle"] is None
    assert traceback.tb_frame.f_locals["sources"] == []


def test_close_interrupts_active_polling(runtime):
    instance, agent = runtime
    descriptor = _descriptor("closing", 1)
    agent.metadata_names[b"metadata"] = "peer"
    agent.status_by_remote["peer"] = "PROC"
    future = instance.send({}, _token("peer", b"metadata", descriptor), descriptor, (bytearray(b"x"),))

    deadline = time.monotonic() + 1
    while not agent.transfer_calls and time.monotonic() < deadline:
        time.sleep(0.001)
    started = time.monotonic()
    instance.close()

    assert time.monotonic() - started < 1
    with pytest.raises(NixlError, match="closing"):
        future.result()


def test_close_tears_down_agent_before_retained_owners():
    events: list[str] = []

    class TrackingAgent:
        def __del__(self) -> None:
            events.append("agent")

    class AgentHandle:
        def __init__(self, agent: TrackingAgent, release_lease: Callable[[], None]) -> None:
            self.agent = agent
            self.release_lease = release_lease

        def __del__(self) -> None:
            events.append("handle")
            self.release_lease()

    class TrackingBuffer(bytearray):
        def __init__(self, name: str) -> None:
            super().__init__(b"x")
            self.name = name

        def __del__(self) -> None:
            events.append(self.name)

    instance = object.__new__(NixlRuntime)
    instance._lock = threading.RLock()
    instance._closing = threading.Event()
    instance._closed = False
    instance._peer_executors = {}
    instance._registered_sources = {}
    instance._receives = {}
    instance._idle_receive_buffers = []
    instance._leased_receive_buffers = {}
    instance._remote_metadata = {}
    instance._registered_bytes = 2
    instance._quarantined_bytes = 1

    agent = TrackingAgent()
    source = _FrameSource(TrackingBuffer("source"), None, 0, 1)
    receiver = _RegisteredReceiveBuffer(TrackingBuffer("receiver"), object(), 0)
    leased_receiver = _RegisteredReceiveBuffer(TrackingBuffer("leased_receiver"), object(), 1)
    lease_key = id(leased_receiver)
    handle = AgentHandle(agent, lambda: instance._release_lease(lease_key))
    instance._agent = agent
    instance._retained_resources = [(handle, [source])]
    instance._quarantined_receive_buffers = [receiver]
    instance._leased_receive_buffers[lease_key] = leased_receiver
    del agent, handle, source, receiver, leased_receiver

    instance.close()
    gc.collect()

    assert events.index("handle") < events.index("agent")
    assert events.index("agent") < events.index("source")
    assert events.index("agent") < events.index("receiver")
    assert events.index("agent") < events.index("leased_receiver")


def test_quarantine_never_returns_exposed_receiver_to_pool(runtime):
    instance, _ = runtime
    descriptor = _descriptor("unsafe", 32)
    instance.prepare_receive(descriptor)
    scratch = instance._receives["unsafe"]

    instance.quarantine_receive("unsafe")

    assert instance._idle_receive_buffers == []
    assert instance._quarantined_receive_buffers == [scratch]
    assert instance.diagnostics["quarantined_bytes"] == 32


def test_get_commit_returns_deferred_response_and_maps_completion():
    transfer = object.__new__(NixlPayloadTransfer)
    descriptor = _descriptor("get", 1)
    transfer._pending_gets = {"get": _PendingGet(descriptor, "manager", (bytearray(b"x"),))}
    send_future: Future[None] = Future()
    transfer.send = lambda *args, **kwargs: send_future
    request = ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA_COMMIT,
        sender_id="manager",
        body={
            "transfer_id": "get",
            "receiver_endpoint": {"transport": "nixl-ucx", "data": {}},
            "receive_token": {"data": {}},
        },
    )

    response = transfer._handle_get_commit(request, "storage")

    assert isinstance(response, DeferredResponse)
    assert not response.future.done()
    send_future.set_result(None)
    assert response.future.result().request_type == ZMQRequestType.GET_DATA_RESPONSE


def test_get_commit_maps_transfer_failure_to_normal_error_response():
    transfer = object.__new__(NixlPayloadTransfer)
    descriptor = _descriptor("failed-get", 1)
    transfer._pending_gets = {"failed-get": _PendingGet(descriptor, "manager", (bytearray(b"x"),))}
    send_future: Future[None] = Future()
    transfer.send = lambda *args, **kwargs: send_future
    request = ZMQMessage.create(
        request_type=ZMQRequestType.GET_DATA_COMMIT,
        sender_id="manager",
        body={
            "transfer_id": "failed-get",
            "receiver_endpoint": {"transport": "nixl-ucx", "data": {}},
            "receive_token": {"data": {}},
        },
    )

    response = transfer._handle_get_commit(request, "storage")
    send_future.set_exception(NixlError("write failed"))

    assert isinstance(response, DeferredResponse)
    assert response.future.result().request_type == ZMQRequestType.PUT_GET_ERROR


class _ProtocolSocket:
    def __init__(self, *, fail_commit: bool = False, fail_ready: bool = False) -> None:
        self.fail_commit = fail_commit
        self.fail_ready = fail_ready
        self.last_request: ZMQMessage | None = None

    async def send_multipart(self, frames: Any, **kwargs: Any) -> None:
        request = ZMQMessage.deserialize(frames)
        if self.fail_commit and request.request_type == ZMQRequestType.GET_DATA_COMMIT:
            raise RuntimeError("commit send failed")
        self.last_request = request

    async def recv_multipart(self, **kwargs: Any) -> Any:
        assert self.last_request is not None
        request = self.last_request
        if request.request_type == ZMQRequestType.PUT_DATA_PREPARE:
            if self.fail_ready:
                raise RuntimeError("ready receive failed")
            response = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_READY,
                sender_id="storage",
                body={
                    "descriptor": request.body["descriptor"],
                    "receive_token": {"data": {}},
                },
            )
        elif request.request_type == ZMQRequestType.GET_DATA_PREPARE:
            descriptor = _descriptor(str(request.body["transfer_id"]), 1)
            response = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_READY,
                sender_id="storage",
                body={"descriptor": descriptor.to_dict()},
            )
        else:
            raise AssertionError(f"unexpected receive after {request.request_type}")
        return response.serialize()


def test_put_does_not_cancel_after_send_is_attempted():
    transfer = object.__new__(NixlPayloadTransfer)
    failed_send: Future[None] = Future()
    failed_send.set_exception(NixlError("write failed"))
    transfer._peer_endpoint = lambda target_id: TransferEndpoint("nixl-ucx", {})
    transfer.send = lambda *args, **kwargs: failed_send
    cancellations: list[Any] = []

    async def cancel(*args: Any) -> None:
        cancellations.append(args)

    transfer._cancel = cancel
    with pytest.raises(NixlError, match="write failed"):
        asyncio.run(
            transfer.put(
                control_socket=_ProtocolSocket(),
                sender_id="manager",
                target_id="storage",
                global_indexes=[1],
                data={"value": [1]},
                data_parser=None,
            )
        )

    assert cancellations == []


def test_put_cancels_prepared_receiver_before_send_is_attempted():
    transfer = object.__new__(NixlPayloadTransfer)

    def missing_endpoint(target_id: str) -> TransferEndpoint:
        raise RuntimeError("endpoint unavailable")

    transfer._peer_endpoint = missing_endpoint
    cancellations: list[Any] = []

    async def cancel(*args: Any) -> None:
        cancellations.append(args)

    transfer._cancel = cancel
    with pytest.raises(RuntimeError, match="endpoint unavailable"):
        asyncio.run(
            transfer.put(
                control_socket=_ProtocolSocket(),
                sender_id="manager",
                target_id="storage",
                global_indexes=[1],
                data={"value": [1]},
                data_parser=None,
            )
        )

    assert cancellations[0][2] == ZMQRequestType.PUT_DATA_CANCEL


def test_put_cancels_when_send_fails_before_submission():
    transfer = object.__new__(NixlPayloadTransfer)
    transfer._peer_endpoint = lambda target_id: TransferEndpoint("nixl-ucx", {})

    def failed_send(*args: Any, **kwargs: Any) -> Future[None]:
        raise NixlError("metadata invalid")

    transfer.send = failed_send
    cancellations: list[Any] = []

    async def cancel(*args: Any) -> None:
        cancellations.append(args)

    transfer._cancel = cancel
    with pytest.raises(NixlError, match="metadata invalid"):
        asyncio.run(
            transfer.put(
                control_socket=_ProtocolSocket(),
                sender_id="manager",
                target_id="storage",
                global_indexes=[1],
                data={"value": [1]},
                data_parser=None,
            )
        )

    assert cancellations[0][2] == ZMQRequestType.PUT_DATA_CANCEL


def test_put_attempts_cancel_when_ready_is_lost():
    transfer = object.__new__(NixlPayloadTransfer)
    cancellations: list[Any] = []

    async def cancel(*args: Any) -> None:
        cancellations.append(args)

    transfer._cancel = cancel
    with pytest.raises(RuntimeError, match="ready receive failed"):
        asyncio.run(
            transfer.put(
                control_socket=_ProtocolSocket(fail_ready=True),
                sender_id="manager",
                target_id="storage",
                global_indexes=[1],
                data={"value": [1]},
                data_parser=None,
            )
        )

    assert cancellations[0][2] == ZMQRequestType.PUT_DATA_CANCEL


def test_get_cancels_remote_source_when_local_receive_prepare_fails():
    transfer = object.__new__(NixlPayloadTransfer)

    def failed_prepare(descriptor: PayloadDescriptor) -> ReceiveToken:
        raise NixlError("registration failed")

    transfer.prepare_receive = failed_prepare
    cancellations: list[Any] = []

    async def cancel(*args: Any) -> None:
        cancellations.append(args)

    transfer._cancel = cancel
    with pytest.raises(NixlError, match="registration failed"):
        asyncio.run(
            transfer.get(
                control_socket=_ProtocolSocket(),
                sender_id="manager",
                target_id="storage",
                global_indexes=[1],
                fields=["value"],
            )
        )

    assert cancellations[0][2] == ZMQRequestType.GET_DATA_CANCEL


def test_get_commit_attempt_failure_quarantines_without_cancel():
    transfer = object.__new__(NixlPayloadTransfer)
    transfer.prepare_receive = lambda descriptor: ReceiveToken({"agent_name": "manager"})
    quarantined: list[str] = []
    cancellations: list[Any] = []
    transfer.quarantine_receive = quarantined.append
    transfer.cancel_receive = lambda transfer_id: pytest.fail("exposed receiver was cancelled")

    async def cancel(*args: Any) -> None:
        cancellations.append(args)

    transfer._cancel = cancel
    with pytest.raises(RuntimeError, match="commit send failed"):
        asyncio.run(
            transfer.get(
                control_socket=_ProtocolSocket(fail_commit=True),
                sender_id="manager",
                target_id="storage",
                global_indexes=[1],
                fields=["value"],
            )
        )

    assert len(quarantined) == 1
    assert cancellations == []


class _CapturingSocket:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def send_multipart(self, message: Any, **kwargs: Any) -> None:
        self.messages.append(message)


def test_deferred_response_copies_identity_and_worker_sends_once():
    future: Future[ZMQMessage] = Future()
    response = DeferredResponse(future)
    completions: SimpleQueue[tuple[bytes, Future[ZMQMessage]]] = SimpleQueue()
    reader, writer = socket.socketpair()
    reader.settimeout(1)
    poller = zmq.Poller()
    poller.register(reader.fileno(), zmq.POLLIN)
    shutdown = threading.Event()
    identity = bytearray(b"client-a")
    worker = _CapturingSocket()
    try:
        _queue_deferred_response(identity, response, completions, writer, shutdown)
        identity[:] = b"client-b"
        future.set_result(
            ZMQMessage.create(request_type=ZMQRequestType.GET_DATA_RESPONSE, sender_id="storage", body={})
        )
        assert dict(poller.poll(1000))[reader.fileno()] == zmq.POLLIN
        assert reader.recv(1) == b"\0"
        _drain_deferred_responses(completions, worker, "storage")
        _drain_deferred_responses(completions, worker, "storage")
    finally:
        reader.close()
        writer.close()

    assert len(worker.messages) == 1
    assert worker.messages[0][0] == b"client-a"
