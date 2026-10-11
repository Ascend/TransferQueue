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
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from transfer_queue import async_kv_local_lock, kv_local_lock

# `transfer_queue.kv_lock` resolves to the re-exported function, so fetch the module itself.
kvl = importlib.import_module("transfer_queue.kv_lock")
P = "p"
# Bound every wait so that a lock regression fails the test instead of hanging it.
TIMEOUT = 10.0


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
            later = asyncio.create_task(child())
        await later  # starts after the block exited, so it no longer counts as nested

    asyncio.run(main())


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_forked_child_drops_the_parents_locks_leases_and_managers(monkeypatch):
    monkeypatch.setattr(kvl, "_registry", threading.Condition())
    monkeypatch.setattr(kvl, "_leases", {"parent-token": object()})
    monkeypatch.setattr(kvl, "_renewer_running", True)
    monkeypatch.setattr(kvl, "_managers", {0: object()})
    release, registry_held = threading.Event(), threading.Event()

    def hold_registry():
        with kvl._registry:
            registry_held.set()
            release.wait(TIMEOUT)

    registry_holder = threading.Thread(target=hold_registry)
    registry_holder.start()
    assert registry_held.wait(TIMEOUT)
    key_holder = start_holder("a", release)
    try:
        pid = os.fork()
        if pid == 0:
            ok = False
            try:
                with kv_local_lock("a", P, timeout=1):
                    ok = kvl._leases == {} and kvl._managers == {} and not kvl._renewer_running
                    ok = ok and kvl._registry.acquire(blocking=False)
            finally:
                os._exit(0 if ok else 1)
        deadline = time.monotonic() + TIMEOUT
        while (status := os.waitpid(pid, os.WNOHANG))[0] == 0:
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                pytest.fail("forked child hung")
            time.sleep(0.01)
        assert os.waitstatus_to_exitcode(status[1]) == 0
    finally:
        release.set()
        registry_holder.join()
        key_holder.join()


def test_global_lock_shards_are_pinned_and_spread_keys():
    # Pinned rather than hash(): every process, whatever its PYTHONHASHSEED, must pick the same actor.
    assert [kvl._shard((P, f"key{i}"), 8) for i in range(4)] == [0, 6, 4, 2]
    shards = [kvl._shard((P, f"key{i}"), 8) for i in range(800)]
    assert all(60 < shards.count(shard) < 140 for shard in range(8))


def test_global_lock_poll_window_fits_the_lease_and_the_timeout():
    assert kvl._poll_window(None, 30) == kvl._POLL_S
    assert kvl._poll_window(None, 0.6) == pytest.approx(0.2)  # a renewal waits at most a third of the lease
    assert kvl._poll_window(time.monotonic() + 0.05, 30) <= 0.05
