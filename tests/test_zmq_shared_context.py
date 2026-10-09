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

"""Regression tests for the shared long-lived ZMQ context in with_zmq_socket.

with_zmq_socket used to create and term() a context per RPC call, which churned libzmq
signaler file descriptors and crashed the process under concurrency (Bad file descriptor
-> SIGABRT). It now reuses the owner's context and leases sockets from a shared pool, so
these tests assert every concurrent call sees the SAME context, alive until close().
Pool-specific behavior is covered in test_zmq_socket_pool.py.
"""

import asyncio
from contextlib import contextmanager
from threading import Thread
from unittest.mock import patch

import pytest
import zmq

import transfer_queue.utils.zmq_utils as zmq_utils
from transfer_queue.client import AsyncTransferQueueClient, TransferQueueClient
from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers.base import StorageManager
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo


class _EchoController:
    """Minimal in-process ROUTER controller that answers GET_META requests."""

    def __init__(self, controller_id="controller_0"):
        self.controller_id = controller_id
        self.context = zmq.Context()
        self.request_socket = self.context.socket(zmq.ROUTER)
        self.request_port = self.request_socket.bind_to_random_port("tcp://127.0.0.1")
        self.zmq_server_info = ZMQServerInfo(
            role=Role.CONTROLLER,
            id=controller_id,
            ip="127.0.0.1",
            ports={"request_handle_socket": self.request_port},
        )
        self.running = True
        self.request_thread = Thread(target=self._handle_requests, daemon=True)
        self.request_thread.start()

    def _handle_requests(self):
        poller = zmq.Poller()
        poller.register(self.request_socket, zmq.POLLIN)
        while self.running:
            try:
                socks = dict(poller.poll(100))
                if self.request_socket not in socks:
                    continue
                messages = self.request_socket.recv_multipart(copy=False)
                identity = messages.pop(0)
                request_msg = ZMQMessage.deserialize(messages)

                batch_size = request_msg.body.get("batch_size", 1)
                data_fields = request_msg.body.get("data_fields", [])
                field_schema = {
                    name: {"dtype": None, "shape": None, "is_nested": False, "is_non_tensor": False}
                    for name in data_fields
                }
                metadata = BatchMeta(
                    global_indexes=list(range(batch_size)),
                    partition_ids=["0"] * batch_size,
                    field_schema=field_schema,
                )
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_META_RESPONSE,
                    sender_id=self.controller_id,
                    receiver_id=request_msg.sender_id,
                    body={"metadata": metadata},
                )
                self.request_socket.send_multipart([identity, *response_msg.serialize()])
            except zmq.Again:
                continue
            except Exception as e:  # pragma: no cover - surfaced via test failure
                print(f"_EchoController ERROR: {e}")

    def stop(self):
        self.running = False
        self.request_thread.join(timeout=2.0)
        self.request_socket.close(linger=0)
        self.context.term()


@pytest.fixture
def echo_controller():
    controller = _EchoController()
    yield controller
    controller.stop()


@pytest.mark.asyncio
async def test_shared_context_reused_across_concurrent_calls(echo_controller, monkeypatch):
    """Many concurrent decorated RPCs must all reuse the client's single context.

    This is the core regression guard: pre-fix, each call created and term()ed its own
    context, which is what corrupted libzmq's signaler FDs under concurrency.
    """
    client = AsyncTransferQueueClient(
        client_id="client_shared_ctx",
        controller_info=echo_controller.zmq_server_info,
    )

    # Record the context object handed to every socket creation inside the decorator.
    seen_contexts = []
    original_create = zmq_utils.create_zmq_socket

    def _spy_create(ctx, *args, **kwargs):
        seen_contexts.append(ctx)
        return original_create(ctx, *args, **kwargs)

    # The decorator resolves create_zmq_socket from the zmq_utils module globals.
    monkeypatch.setattr(zmq_utils, "create_zmq_socket", _spy_create)

    num_calls = 200
    coros = [
        client.async_get_meta(data_fields=["tokens", "labels"], batch_size=2, partition_id="0")
        for _ in range(num_calls)
    ]
    # wait_for guards against the hang failure mode.
    results = await asyncio.wait_for(asyncio.gather(*coros), timeout=60)

    assert len(results) == num_calls
    assert all(isinstance(meta, BatchMeta) for meta in results)

    # Every socket must come from the SAME context, and it must be the client's. The count
    # is bounded by the pool's cap rather than by the call count, since concurrency past it
    # waits for a socket instead of opening one.
    assert seen_contexts, "no socket was created at all"
    assert all(ctx is client.zmq_context for ctx in seen_contexts)
    # The shared context must NOT have been terminated by any call.
    assert not client.zmq_context.closed

    client.close()


def test_client_context_has_fixed_io_thread_pool(echo_controller):
    client = AsyncTransferQueueClient(
        client_id="client_fixed_context_pool",
        controller_info=echo_controller.zmq_server_info,
        zmq_io_threads=4,
    )

    assert client.zmq_context.get(zmq.IO_THREADS) == 4

    client.close()


def test_client_rejects_invalid_context_pool_size(echo_controller):
    with pytest.raises(ValueError, match="at least 1"):
        AsyncTransferQueueClient(
            client_id="client_invalid_context_pool",
            controller_info=echo_controller.zmq_server_info,
            zmq_io_threads=0,
        )


@contextmanager
def _no_context_left_open():
    """Assert every ZMQ context built inside the block is closed by the time it exits.

    __init__ arms the finalizer only on success, so an invalid argument must either be
    rejected before the context is built or destroy it on the way out -- otherwise the
    context and its native I/O threads leak for the process lifetime.
    """
    created = []
    real_context = zmq.asyncio.Context

    def _spy(*args, **kwargs):
        ctx = real_context(*args, **kwargs)
        created.append(ctx)
        return ctx

    with patch("zmq.asyncio.Context", side_effect=_spy):
        try:
            yield created
        finally:
            leaked = [ctx for ctx in created if not ctx.closed]
            for ctx in leaked:
                ctx.destroy(linger=0)
            assert not leaked, f"{len(leaked)} ZMQ context(s) left open on the failure path"


def test_client_rejects_invalid_socket_pool_size(echo_controller):
    """A bad TQ_SOCKET_POOL_SIZE must name the variable, not silently disable reuse.

    Below 1 nothing is ever parked, so every request pays a fresh connect while the client
    still looks pooled.
    """
    for bad in (-1, 0):
        with patch("transfer_queue.client.TQ_SOCKET_POOL_SIZE", bad):
            with _no_context_left_open() as created:
                with pytest.raises(ValueError, match="TQ_SOCKET_POOL_SIZE must be at least 1"):
                    AsyncTransferQueueClient(
                        client_id="client_invalid_socket_pool",
                        controller_info=echo_controller.zmq_server_info,
                    )
                assert created == [], "checkable without a context, so none should be built"


def test_invalid_max_sockets_does_not_leak_a_context(echo_controller):
    """No invalid max-sockets input may leave the context or its I/O threads behind."""
    # Rejectable without a live context.
    with patch("transfer_queue.client.TQ_CLIENT_ZMQ_MAX_SOCKETS", "not-a-number"):
        with _no_context_left_open() as created:
            with pytest.raises(ValueError, match="TQ_CLIENT_ZMQ_MAX_SOCKETS must be an integer"):
                AsyncTransferQueueClient(
                    client_id="client_garbage_max_sockets",
                    controller_info=echo_controller.zmq_server_info,
                )
            assert created == [], "parsing needs no context, so none should be built"

    for bad in (0, -5):
        with _no_context_left_open() as created:
            with pytest.raises(ValueError, match="at least 1"):
                AsyncTransferQueueClient(
                    client_id="client_low_max_sockets",
                    controller_info=echo_controller.zmq_server_info,
                    zmq_max_sockets=bad,
                )
            assert created == [], "the lower bound needs no context, so none should be built"

    # Above ZMQ_SOCKET_LIMIT: this one genuinely needs a live context, so it must be
    # destroyed rather than hoisted.
    with _no_context_left_open() as created:
        with pytest.raises(ValueError, match="ZMQ_SOCKET_LIMIT"):
            AsyncTransferQueueClient(
                client_id="client_huge_max_sockets",
                controller_info=echo_controller.zmq_server_info,
                zmq_max_sockets=10**9,
            )
        assert len(created) == 1, "the limit check requires a built context"


def test_simple_storage_borrows_client_context(echo_controller):
    client = AsyncTransferQueueClient(
        client_id="client_simple_storage_context",
        controller_info=echo_controller.zmq_server_info,
    )
    config = {"zmq_info": {}}

    with patch("transfer_queue.client.StorageManagerFactory.create") as create_manager:
        client.initialize_storage_manager("SimpleStorage", config)

    create_manager.assert_called_once_with(
        "SimpleStorage",
        controller_info=echo_controller.zmq_server_info,
        config=config,
        zmq_context=client.zmq_context,
    )

    client.close()


def test_each_scenario_gets_its_own_pool(echo_controller):
    """Controller RPC, storage RPC and notify must each hold a separate pool.

    They share one context but never one pool: each dials a different peer or socket name,
    so a shared pool reused nothing while letting one scenario's sockets be swept by
    another's. Separate pools keep each scenario's failures to itself.
    """
    client = AsyncTransferQueueClient(
        client_id="client_scenario_pools",
        controller_info=echo_controller.zmq_server_info,
    )

    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager(
            echo_controller.zmq_server_info,
            {"zmq_info": {"storage_0": echo_controller.zmq_server_info}},
            zmq_context=client.zmq_context,
        )

    pools = [client.controller_rpc_pool, manager.storage_rpc_pool, manager.notify_pool]
    assert len({id(pool) for pool in pools}) == 3, "scenarios must not share a pool"

    manager.close()
    client.close()


@pytest.mark.parametrize(
    "pool_size, units, warns",
    [
        (8, 4, False),  # 32 idle sockets against a 64-socket budget
        (8, 32, True),  # 256 would exceed it
    ],
)
def test_warns_when_pool_size_times_units_exceeds_the_context(echo_controller, caplog, pool_size, units, warns):
    """The per-address cap multiplies by storage-unit count; the context ceiling does not.

    At a few thousand units the product passes ZMQ_MAX_SOCKETS, where opening a socket
    fails outright, so the mismatch is worth naming at construction rather than at a lease.
    """
    client = AsyncTransferQueueClient(
        client_id="client_pool_budget",
        controller_info=echo_controller.zmq_server_info,
        zmq_max_sockets=64,
    )
    zmq_info = {
        f"storage_{i}": ZMQServerInfo(
            role=Role.STORAGE,
            id=f"storage_{i}",
            ip="127.0.0.1",
            ports={"put_get_socket": 5600 + i},
        )
        for i in range(units)
    }

    with (
        patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller"),
        patch("transfer_queue.storage.managers.simple_storage_manager.TQ_SOCKET_POOL_SIZE", pool_size),
        caplog.at_level("WARNING"),
    ):
        manager = AsyncSimpleStorageManager(
            echo_controller.zmq_server_info,
            {"zmq_info": zmq_info},
            zmq_context=client.zmq_context,
        )

    assert ("above this context's ZMQ_MAX_SOCKETS" in caplog.text) is warns

    manager.close()
    client.close()


def test_close_after_failed_handshake_still_releases_the_context(echo_controller):
    """A manager whose base constructor raised must still tear down on close().

    storage_rpc_pool is assigned after super().__init__(), so a handshake timeout leaves it
    unset while the context and its native I/O threads are already allocated. __del__ calls
    close() regardless, so dereferencing the pool there aborts the base teardown and leaks
    the context.
    """
    built = []

    def _fail(self):
        built.append(self)
        raise TimeoutError("handshake failed")

    with patch.object(StorageManager, "_connect_to_controller", _fail):
        with pytest.raises(TimeoutError):
            AsyncSimpleStorageManager(
                echo_controller.zmq_server_info,
                {"zmq_info": {"storage_0": echo_controller.zmq_server_info}},
            )

    manager = built[0]
    assert not hasattr(manager, "storage_rpc_pool"), "the test no longer exercises the partial-build path"

    manager.close()
    assert manager.zmq_context.closed, "the context outlived a failed construction"


def test_simple_storage_does_not_destroy_borrowed_context(echo_controller):
    client = AsyncTransferQueueClient(
        client_id="client_borrowed_context_lifecycle",
        controller_info=echo_controller.zmq_server_info,
    )

    with patch("transfer_queue.storage.managers.base.StorageManager._connect_to_controller"):
        manager = AsyncSimpleStorageManager(
            echo_controller.zmq_server_info,
            {"zmq_info": {"storage_0": echo_controller.zmq_server_info}},
            zmq_context=client.zmq_context,
        )

    assert manager.zmq_context is client.zmq_context
    assert not manager._owns_zmq_context

    manager.close()
    assert not client.zmq_context.closed

    client.close()
    assert client.zmq_context.closed


@pytest.mark.asyncio
async def test_close_destroys_context(echo_controller):
    """close() must terminate the shared context exactly once (no leak, no hang)."""
    client = AsyncTransferQueueClient(
        client_id="client_close_ctx",
        controller_info=echo_controller.zmq_server_info,
    )
    assert not client.zmq_context.closed

    # A normal call before shutdown leaves the context alive.
    await asyncio.wait_for(
        client.async_get_meta(data_fields=["tokens"], batch_size=1, partition_id="0"),
        timeout=30,
    )
    assert not client.zmq_context.closed

    client.close()
    assert client.zmq_context.closed


def test_non_numeric_max_sockets_env_var_names_the_variable(echo_controller):
    """A typo'd value must say which env var is wrong, not raise a bare int() error."""
    with patch("transfer_queue.client.TQ_CLIENT_ZMQ_MAX_SOCKETS", "not-a-number"):
        with pytest.raises(ValueError, match="TQ_CLIENT_ZMQ_MAX_SOCKETS must be an integer"):
            AsyncTransferQueueClient(
                client_id="client_max_sockets_garbage",
                controller_info=echo_controller.zmq_server_info,
            )


def test_close_skips_destroy_while_loop_thread_alive(echo_controller):
    """destroy() is not thread-safe, so a stuck loop thread must veto it.

    TransferQueueClient.close() joins its loop thread with a timeout that only warns on
    expiry. Falling through to destroy() with that thread still holding sockets is the
    documented hazard, so the context is leaked instead.
    """
    client = TransferQueueClient(
        client_id="client_stuck_thread",
        controller_info=echo_controller.zmq_server_info,
    )
    context = client.zmq_context

    # Simulate a loop thread that outlived its join timeout.
    with patch.object(client._thread, "is_alive", return_value=True):
        assert client._can_destroy_zmq_context() is False
        client.close()

    assert not context.closed, "context must be leaked, not destroyed unsafely"

    # With the thread genuinely gone, the veto lifts.
    assert client._can_destroy_zmq_context() is True
    context.destroy(linger=0)


def _make_borrowing_manager(client=None):
    """A minimal manager borrowing *client*'s context, like SimpleStorage does.

    With no client it creates its own context. Either way it builds its own notify pool.
    """

    class Borrower(StorageManager):
        def _connect_to_controller(self):
            pass

        async def put_data(self, *args, **kwargs):
            return None

        async def get_data(self, *args, **kwargs):
            return None

        async def clear_data(self, *args, **kwargs):
            return None

    return Borrower(None, {}) if client is None else Borrower(None, {}, zmq_context=client.zmq_context)


def test_stuck_notify_thread_vetoes_destroy_of_borrowed_context(echo_controller):
    """A borrowing manager's stuck notify thread must veto the owner's destroy().

    The manager has no destroy() of its own to skip, so staying silent would let the client
    destroy a context whose sockets that thread may still hold. It must be asked first.
    """
    client = AsyncTransferQueueClient(
        client_id="client_notify_thread_veto",
        controller_info=echo_controller.zmq_server_info,
    )
    client.storage_manager = _make_borrowing_manager(client)
    context = client.zmq_context

    with patch.object(client.storage_manager._notify_thread, "is_alive", return_value=True):
        assert client._can_destroy_zmq_context() is False
        client.close()
        assert not context.closed, "context must be leaked while the notify thread lives"

    # Veto lifts once the thread is genuinely gone.
    assert client._can_destroy_zmq_context() is True
    context.destroy(linger=0)


def test_healthy_notify_thread_does_not_block_destroy(echo_controller):
    """The veto must not over-trigger: a clean manager shutdown still destroys."""
    client = AsyncTransferQueueClient(
        client_id="client_notify_thread_clean",
        controller_info=echo_controller.zmq_server_info,
    )
    client.storage_manager = _make_borrowing_manager(client)

    client.close()
    assert client.zmq_context.closed


def test_manager_with_own_context_does_not_veto(echo_controller):
    """A manager holding its own context has no say in the client's teardown."""
    client = AsyncTransferQueueClient(
        client_id="client_independent_manager",
        controller_info=echo_controller.zmq_server_info,
    )
    client.storage_manager = _make_borrowing_manager()  # creates its own context
    assert client.storage_manager.zmq_context is not client.zmq_context

    # Even a stuck notify thread on an unrelated context must not block the client.
    with patch.object(client.storage_manager._notify_thread, "is_alive", return_value=True):
        assert client._can_destroy_zmq_context() is True

    client.storage_manager.zmq_context.destroy(linger=0)
    client.close()
    assert client.zmq_context.closed
