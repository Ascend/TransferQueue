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
import importlib
import threading
import time
from unittest.mock import MagicMock

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


A, B, C = "01" * 28, "02" * 28, "03" * 28  # Ray checks node IDs are hex


@pytest.fixture
def fake_lock_actors(monkeypatch):
    """Three alive nodes and one dead one; yields the mocked lock actor class instead of starting actors."""
    nodes = [{"NodeID": node, "Alive": True} for node in (C, A, B)] + [{"NodeID": "04" * 28, "Alive": False}]
    monkeypatch.setattr(kvl.ray, "nodes", lambda: nodes)
    manager_cls = MagicMock()
    monkeypatch.setattr(kvl, "TransferQueueLockManager", manager_cls)
    monkeypatch.setattr(kvl, "_owned", [])
    monkeypatch.setenv("TQ_ENABLE_KV_GLOBAL_LOCK", "1")
    monkeypatch.setenv("TQ_KV_GLOBAL_LOCK_MAX_WAITING_ACQUIRES", "2")
    yield manager_cls


@pytest.mark.parametrize(
    "num_shards, expected",
    [(None, [A, B, C, A, B, C, A, B]), ("3", [A, B, C])],  # None: unset, so the default 8
)
def test_lock_actors_are_spread_evenly_over_the_alive_nodes(fake_lock_actors, monkeypatch, num_shards, expected):
    if num_shards is not None:
        monkeypatch.setenv("TQ_KV_GLOBAL_LOCK_NUM_SHARDS", num_shards)
    else:
        monkeypatch.delenv("TQ_KV_GLOBAL_LOCK_NUM_SHARDS", raising=False)

    kvl._start_lock_managers(kvl._lock_settings())

    strategies = [call.kwargs["scheduling_strategy"] for call in fake_lock_actors.options.call_args_list]
    assert [s.node_id for s in strategies] == expected  # per-node counts differ by at most 1
    assert not any(s.soft for s in strategies)
    fake_lock_actors.options.return_value.remote.assert_called_with(len(expected), 2)


@pytest.mark.parametrize("name", ["TQ_KV_GLOBAL_LOCK_NUM_SHARDS", "TQ_KV_GLOBAL_LOCK_MAX_WAITING_ACQUIRES"])
@pytest.mark.parametrize("value", ["0", "-1", "abc", "true", "2.5"])
def test_invalid_lock_settings_raise_before_any_actor_starts(fake_lock_actors, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"{name} must be an integer >= 1"):
        kvl._lock_settings()
    fake_lock_actors.options.assert_not_called()


def test_lock_settings_are_ignored_while_the_lock_is_off(monkeypatch):
    monkeypatch.delenv("TQ_ENABLE_KV_GLOBAL_LOCK", raising=False)
    monkeypatch.setenv("TQ_KV_GLOBAL_LOCK_NUM_SHARDS", "0")
    assert kvl._lock_settings() is None
