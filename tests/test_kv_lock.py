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

import asyncio
import gc
import importlib
import threading
import time
from types import SimpleNamespace

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq
from transfer_queue import async_kv_local_lock, async_kv_local_locked, kv_local_lock, kv_local_locked

# `transfer_queue.kv_lock` resolves to the re-exported function, so fetch the module itself.
kvl = importlib.import_module("transfer_queue.kv_lock")
P = "p"
# Bound every wait so that a lock regression fails the test instead of hanging it.
TIMEOUT = 10.0
E2E_TIMEOUT = 120.0


@pytest.fixture(autouse=True)
def table_is_empty_afterwards():
    yield
    assert kvl._table == {}


def n_waiters(key):
    with kvl._state_lock:
        return len(kvl._table.get((P, key), ()))


def wait_until(predicate):
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.001)


async def async_wait_until(predicate):
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.001)


def start_holder(keys, release):
    """Hold `keys` on a new thread until `release` is set; return once they are held."""
    entered = threading.Event()

    def run():
        with kv_local_lock(keys, P, timeout=TIMEOUT):
            entered.set()
            release.wait(TIMEOUT)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(TIMEOUT)
    return thread


def test_threads_and_tasks_on_several_loops_serialize_read_modify_write():
    counter = [0]

    def thread_worker():
        for _ in range(100):
            with kv_local_lock("c", P, timeout=TIMEOUT):
                value = counter[0]
                time.sleep(0)
                counter[0] = value + 1

    async def loop_main():
        async def task_worker():
            for _ in range(100):
                async with async_kv_local_lock("c", P, timeout=TIMEOUT):
                    value = counter[0]
                    await asyncio.sleep(0)
                    counter[0] = value + 1

        await asyncio.gather(*(task_worker() for _ in range(4)))

    threads = [threading.Thread(target=thread_worker) for _ in range(4)]
    threads += [threading.Thread(target=asyncio.run, args=(loop_main(),)) for _ in range(2)]
    for thread in threads:
        thread.start()
    asyncio.run(loop_main())
    for thread in threads:
        thread.join(TIMEOUT)
        assert not thread.is_alive()
    assert counter[0] == 4 * 100 + 3 * 4 * 100


def test_thread_and_task_wake_each_other():
    order = []

    def thread_body():
        with kv_local_lock("k", P, timeout=TIMEOUT):
            order.append("thread")
            wait_until(lambda: n_waiters("k") == 1)

    async def main():
        thread = threading.Thread(target=thread_body)
        async with async_kv_local_lock("k", P, timeout=TIMEOUT):
            thread.start()
            wait_until(lambda: n_waiters("k") == 1)
            order.append("task")
        async with async_kv_local_lock("k", P, timeout=TIMEOUT):
            order.append("task again")
        thread.join(TIMEOUT)
        assert not thread.is_alive()

    asyncio.run(main())
    assert order == ["task", "thread", "task again"]


def test_waiters_are_served_fifo():
    order, release = [], threading.Event()
    holder = start_holder("k", release)

    def waiter(i):
        with kv_local_lock("k", P, timeout=TIMEOUT):
            order.append(i)

    threads = [threading.Thread(target=waiter, args=(i,)) for i in range(5)]
    for i, thread in enumerate(threads):
        thread.start()
        wait_until(lambda n=i + 1: n_waiters("k") == n)
    release.set()
    for thread in [holder, *threads]:
        thread.join(TIMEOUT)
        assert not thread.is_alive()
    assert order == list(range(5))


def test_release_hands_off_directly_so_a_newcomer_cannot_barge():
    outcome = []

    def holder_then_newcomer():
        with kv_local_lock("k", P, timeout=TIMEOUT):
            wait_until(lambda: n_waiters("k") == 1)
        try:
            with kv_local_lock("k", P, timeout=0):
                outcome.append("barged")
        except TimeoutError:
            outcome.append("queued behind the woken waiter")

    thread = threading.Thread(target=holder_then_newcomer)
    thread.start()
    wait_until(lambda: (P, "k") in kvl._table)
    with kv_local_lock("k", P, timeout=TIMEOUT):
        thread.join(TIMEOUT)
        assert not thread.is_alive()
    assert outcome == ["queued behind the woken waiter"]


def test_timeout_raises_and_leaves_no_waiter():
    release = threading.Event()
    holder = start_holder("k", release)

    async def take():
        async with async_kv_local_lock("k", P, timeout=0.05):
            pass

    with pytest.raises(TimeoutError):
        with kv_local_lock("k", P, timeout=0.05):
            pass
    with pytest.raises(TimeoutError):
        asyncio.run(take())
    assert n_waiters("k") == 0
    release.set()
    holder.join(TIMEOUT)
    assert not holder.is_alive()


def test_timeout_is_a_total_deadline_and_partial_acquisition_is_released():
    release_a, release_b = threading.Event(), threading.Event()
    holders = [start_holder("a", release_a), start_holder("b", release_b)]
    threading.Timer(0.3, release_a.set).start()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        with kv_local_lock(["b", "a"], P, timeout=0.5):
            pass
    assert time.monotonic() - start < 0.7  # a per-key budget would take 0.3 + 0.5 s
    assert (P, "a") not in kvl._table
    release_b.set()
    for holder in holders:
        holder.join(TIMEOUT)
        assert not holder.is_alive()


def test_grant_racing_a_timeout_counts_as_acquired(monkeypatch):
    release = threading.Event()
    holder = start_holder("k", release)

    class GrantedThenTimedOut(threading.Event):
        def wait(self, timeout=None):
            release.set()
            holder.join(TIMEOUT)  # the holder hands the key to this waiter...
            return False  # ...but the wait still reports a timeout

    monkeypatch.setattr(kvl, "threading", SimpleNamespace(Event=GrantedThenTimedOut))
    with kv_local_lock("k", P, timeout=1):
        assert (P, "k") in kvl._table and n_waiters("k") == 0


def test_cancellation_never_strands_the_key():
    release, order = threading.Event(), []
    holder = start_holder("k", release)

    async def take(tag):
        async with async_kv_local_lock("k", P, timeout=TIMEOUT):
            order.append(tag)

    async def main():
        waiting = asyncio.create_task(take("cancelled while waiting"))
        await async_wait_until(lambda: n_waiters("k") == 1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiting, TIMEOUT)
        assert n_waiters("k") == 0

        granted, nxt = asyncio.create_task(take("cancelled after grant")), asyncio.create_task(take("next"))
        await async_wait_until(lambda: n_waiters("k") == 2)
        release.set()
        holder.join(TIMEOUT)  # blocks this loop, so `granted` is cancelled before it can run
        granted.cancel()
        await asyncio.wait_for(asyncio.gather(granted, nxt, return_exceptions=True), TIMEOUT)
        assert granted.cancelled()

    asyncio.run(main())
    assert order == ["next"]


def test_waiter_on_a_closed_loop_is_skipped():
    release, taken = threading.Event(), []
    holder = start_holder("k", release)

    async def take():
        async with async_kv_local_lock("k", P, timeout=TIMEOUT):
            taken.append("abandoned task")

    def thread_waiter():
        with kv_local_lock("k", P, timeout=TIMEOUT):
            taken.append("thread")

    loop = asyncio.new_event_loop()
    loop.create_task(take())
    loop.run_until_complete(async_wait_until(lambda: n_waiters("k") == 1))
    loop.close()  # leaves the waiting task pending on a closed loop
    thread = threading.Thread(target=thread_waiter)
    thread.start()
    wait_until(lambda: n_waiters("k") == 2)
    release.set()
    for t in (holder, thread):
        t.join(TIMEOUT)
        assert not t.is_alive()
    assert taken == ["thread"]
    gc.collect()  # finalizing the abandoned task must leave the table alone


def test_invalid_and_nested_use_is_rejected():
    with pytest.raises(ValueError):
        with kv_local_lock([], P):
            pass
    with kv_local_lock(["a", "a", "b"], P):
        assert set(kvl._table) == {(P, "a"), (P, "b")}
        for keys in ("a", "c"):
            with pytest.raises(RuntimeError, match="Nested"):
                with kv_local_lock(keys, P, timeout=TIMEOUT):
                    pass

    async def main():
        with pytest.raises(RuntimeError, match="async_kv_local_lock"):
            with kv_local_lock("a", P):
                pass
        async with async_kv_local_lock("a", P):
            with pytest.raises(RuntimeError, match="Nested"):
                async with async_kv_local_lock("b", P):
                    pass

            async def child():
                async with async_kv_local_lock("c", P):
                    pass

            with pytest.raises(RuntimeError, match="Nested"):
                await asyncio.create_task(child())

    asyncio.run(main())


def is_held_elsewhere(keys):
    """Whether another thread's kv_local_lock on `keys` times out."""
    outcome = []

    def run():
        try:
            with kv_local_lock(keys, P, timeout=0):
                outcome.append(False)
        except TimeoutError:
            outcome.append(True)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(TIMEOUT)
    assert not thread.is_alive()
    return outcome == [True]


def test_locked_wrapper_passes_arguments_through_and_holds_the_lock_during_the_call():
    def fn(keys, partition_id, **kwargs):
        assert is_held_elsewhere(keys)
        return keys, partition_id, kwargs

    assert kv_local_locked(fn, "a", P, lock_timeout=TIMEOUT, x=1) == ("a", P, {"x": 1})
    assert kv_local_locked(fn, ["b", "a"], P) == (["b", "a"], P, {})
    assert not is_held_elsewhere(["a", "b"])

    def boom(keys, partition_id):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        kv_local_locked(boom, "a", P)
    assert not is_held_elsewhere("a")


def test_async_locked_wrapper_holds_the_lock_and_releases_on_error():
    async def fn(keys, partition_id, **kwargs):
        assert is_held_elsewhere(keys)
        return keys, partition_id, kwargs

    async def boom(keys, partition_id):
        raise ValueError("boom")

    async def main():
        assert await async_kv_local_locked(fn, ["a", "b"], P, lock_timeout=TIMEOUT, x=1) == (["a", "b"], P, {"x": 1})
        with pytest.raises(ValueError, match="boom"):
            await async_kv_local_locked(boom, "a", P)

    asyncio.run(main())
    assert not is_held_elsewhere(["a", "b"])


def test_locked_wrapper_lock_timeout_skips_the_call():
    calls, release = [], threading.Event()
    holder = start_holder("k", release)

    async def async_fn(keys, partition_id):
        calls.append(keys)

    with pytest.raises(TimeoutError):
        kv_local_locked(lambda keys, partition_id: calls.append(keys), ["j", "k"], P, lock_timeout=0.05)
    with pytest.raises(TimeoutError):
        asyncio.run(async_kv_local_locked(async_fn, "k", P, lock_timeout=0.05))
    assert calls == [] and n_waiters("k") == 0
    release.set()
    holder.join(TIMEOUT)
    assert not holder.is_alive()


def test_locked_wrappers_reject_the_wrong_kind_of_function():
    async def async_fn(keys, partition_id):
        pass

    with pytest.raises(TypeError, match="async_kv_local_locked"):
        kv_local_locked(async_fn, "a", P)
    with pytest.raises(TypeError, match="coroutine function"):
        asyncio.run(async_kv_local_locked(lambda keys, partition_id: None, "a", P))


@pytest.fixture
def simple_storage_tq():
    if not ray.is_initialized():
        ray.init(namespace="TestKVLock")
    backend = {"storage_backend": "SimpleStorage", "SimpleStorage": {"total_storage_size": 200}}
    tq.init(OmegaConf.create({"controller": {"polling_mode": True}, "backend": backend}))
    yield
    tq.close()
    ray.shutdown()


def test_kv_local_lock_serializes_simple_storage_read_modify_write(simple_storage_tq):
    tq.kv_put("counter", "lock_e2e", fields={"v": torch.tensor([0])})  # also creates the client on this thread

    def increment(key, partition_id):
        value = tq.kv_batch_get(key, partition_id)["v"]
        tq.kv_put(key, partition_id, fields={"v": value[0] + 1})

    def context_worker():
        for _ in range(50):
            with tq.kv_local_lock("counter", "lock_e2e", timeout=E2E_TIMEOUT):
                increment("counter", "lock_e2e")

    def wrapper_worker():
        for _ in range(50):
            tq.kv_local_locked(increment, "counter", "lock_e2e", lock_timeout=E2E_TIMEOUT)

    threads = [threading.Thread(target=worker) for worker in [context_worker, wrapper_worker] * 4]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(E2E_TIMEOUT)
        assert not thread.is_alive()
    assert tq.kv_batch_get("counter", "lock_e2e")["v"][0].item() == 400

    fields = TensorDict({"v": torch.tensor([[1], [2]])}, batch_size=2)
    tq.kv_local_locked(tq.kv_batch_put, ["a", "b"], "lock_e2e", fields=fields)
    data = tq.kv_local_locked(tq.kv_batch_get, ["a", "b"], "lock_e2e", select_fields="v")
    assert [row.tolist() for row in data["v"]] == [[1], [2]]  # rows may come back as a nested tensor
    data = asyncio.run(tq.async_kv_local_locked(tq.async_kv_batch_get, "b", "lock_e2e", lock_timeout=E2E_TIMEOUT))
    assert [row.tolist() for row in data["v"]] == [[2]]
    tq.kv_clear(["counter", "a", "b"], "lock_e2e")
