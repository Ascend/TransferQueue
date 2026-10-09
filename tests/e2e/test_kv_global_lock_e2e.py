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

"""kv_global_lock across Ray actors: exclusion, expiry, withdrawal, lost leases, and close() ownership."""

import asyncio
import importlib
import threading
import time

import pytest
import ray
import torch
from omegaconf import OmegaConf

import transfer_queue as tq
from transfer_queue.lock_manager import TransferQueueLockManager

kvl = importlib.import_module("transfer_queue.kv_lock")
P = "global_lock_e2e"
# Bound every wait so that a lock regression fails the test instead of hanging it.
TIMEOUT_S = 60
CONF = {
    "controller": {"polling_mode": True},
    "backend": {"storage_backend": "SimpleStorage", "SimpleStorage": {"total_storage_size": 100}},
    "lock": {"enabled": True},
}


@ray.remote
class Worker:
    def __init__(self):
        tq.init()

    def increment(self, n, keys="counter"):
        for _ in range(n):
            with tq.kv_global_lock(keys, P, timeout=TIMEOUT_S):
                value = int(tq.kv_batch_get("counter", P)["v"][0])
                time.sleep(0.001)
                tq.kv_put("counter", P, fields={"v": torch.tensor([value + 1])})

    def cycle(self, keys, n):
        for _ in range(n):
            with tq.kv_global_lock(keys, P, timeout=TIMEOUT_S):
                pass

    def hold(self, key, lease_s=30):
        self.held = tq.kv_global_lock(key, P, timeout=TIMEOUT_S, lease_s=lease_s)
        self.held.__enter__()

    def release(self):
        self.held.__exit__(None, None, None)

    def close(self):
        tq.close()


@pytest.fixture(scope="module", autouse=True)
def tq_session():
    if not ray.is_initialized():
        ray.init(namespace="TestKVGlobalLockE2E")
    tq.init(OmegaConf.create(CONF))
    yield
    tq.close()
    ray.shutdown()


def wait_until(predicate):
    deadline = time.monotonic() + TIMEOUT_S
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.05)


def holders():
    # Exit sends its release without waiting for it, so wait for holders and waiters to drain.
    return {h["key"] for h in tq.kv_lock_list(P)["holders"]}


def shard_of(key):
    return kvl._shard((P, key), kvl._shard_count())


def manager(key):
    return ray.get_actor(f"TransferQueueLockManager_{shard_of(key)}", namespace="transfer_queue")


def keys_on_two_shards(prefix):
    """Return two keys on different lock actors, the one on the lower shard first."""
    keys = sorted((f"{prefix}{i}" for i in range(50)), key=shard_of)
    assert shard_of(keys[0]) < shard_of(keys[-1])
    return keys[0], keys[-1]


def counter():
    return int(tq.kv_batch_get("counter", P)["v"][0])


def test_actors_serialize_read_modify_write():
    tq.kv_put("counter", P, fields={"v": torch.tensor([0])})
    workers = [Worker.remote() for _ in range(3)]
    ray.get([w.increment.remote(20) for w in workers], timeout=TIMEOUT_S)
    assert counter() == 60
    tq.kv_clear("counter", P)


def test_opposite_key_orders_on_two_shards_do_not_deadlock():
    a, b = keys_on_two_shards("ab")
    tq.kv_put("counter", P, fields={"v": torch.tensor([0])})
    first, second = Worker.remote(), Worker.remote()
    ray.get([first.increment.remote(30, [a, b]), second.increment.remote(30, [b, a])], timeout=TIMEOUT_S)
    assert counter() == 60
    tq.kv_clear("counter", P)
    wait_until(lambda: holders() == set())


def test_multi_shard_timeout_releases_the_earlier_shard():
    early, late = keys_on_two_shards("mt")
    worker = Worker.remote()
    ray.get(worker.hold.remote(late), timeout=TIMEOUT_S)
    with pytest.raises(TimeoutError):
        with tq.kv_global_lock([early, late], P, timeout=0.5):
            pass
    wait_until(lambda: holders() == {late})
    wait_until(lambda: tq.kv_lock_list(P)["waiters"] == 0)
    with tq.kv_global_lock(early, P, timeout=5):  # a leaked grant would hold it for its 30 s lease
        pass
    ray.get(worker.release.remote(), timeout=TIMEOUT_S)
    wait_until(lambda: tq.kv_lock_list(P) == {"holders": [], "waiters": 0})


def test_lease_lost_on_one_shard_raises():
    early, late = keys_on_two_shards("ll")
    with pytest.raises(tq.LockLostError):
        with tq.kv_global_lock([early, late], P, lease_s=1) as lease:
            assert holders() == {early, late} and len(lease.shards) == 2
            ray.get(manager(late).release.remote(lease.token), timeout=TIMEOUT_S)
            wait_until(lambda: lease.lost)
            with pytest.raises(tq.LockLostError):
                lease.check()
    wait_until(lambda: holders() == set())


@ray.remote
def lock_without_init(keys):
    with tq.kv_global_lock(keys, P, timeout=TIMEOUT_S) as lease:
        return list(lease.shards)


def test_process_without_init_learns_the_shard_count():
    a, b = keys_on_two_shards("ni")
    assert ray.get(lock_without_init.remote([a, b]), timeout=TIMEOUT_S) == [shard_of(a), shard_of(b)]
    wait_until(lambda: holders() == set())


def test_timeout_leaves_no_holder_or_waiter():
    worker = Worker.remote()
    ray.get(worker.hold.remote("t"), timeout=TIMEOUT_S)
    with pytest.raises(TimeoutError):
        with tq.kv_global_lock("t", P, timeout=0.3):
            pass
    wait_until(lambda: tq.kv_lock_list(P)["waiters"] == 0)
    ray.get(worker.release.remote(), timeout=TIMEOUT_S)
    wait_until(lambda: tq.kv_lock_list(P) == {"holders": [], "waiters": 0})


def test_cancelled_waiter_is_withdrawn_and_never_granted():
    worker = Worker.remote()
    ray.get(worker.hold.remote("c"), timeout=TIMEOUT_S)

    async def wait_for_lock():
        async with tq.async_kv_global_lock("c", P):
            pass

    async def main():
        task = asyncio.create_task(wait_for_lock())
        deadline = time.monotonic() + TIMEOUT_S
        while tq.kv_lock_list(P)["waiters"] != 1:
            assert time.monotonic() < deadline, "the task never started waiting"
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, TIMEOUT_S)

    asyncio.run(main())
    wait_until(lambda: tq.kv_lock_list(P)["waiters"] == 0)
    ray.get(worker.release.remote(), timeout=TIMEOUT_S)
    # A ghost grant to the cancelled waiter would hold "c" for its whole 30 s lease.
    with tq.kv_global_lock("c", P, timeout=5):
        pass


def test_waiting_on_a_later_shard_keeps_the_earlier_keys_held():
    early, late = keys_on_two_shards("lw")
    worker = Worker.remote()
    ray.get(worker.hold.remote(late), timeout=TIMEOUT_S)
    entered = []

    def take_both():
        with tq.kv_global_lock([early, late], P, timeout=TIMEOUT_S, lease_s=2) as lease:
            entered.append(lease)

    thread = threading.Thread(target=take_both)
    thread.start()
    wait_until(lambda: holders() == {early, late})
    time.sleep(3)  # longer than our lease: unrenewed, the grant on `early` would have expired
    assert holders() == {early, late}
    ray.get(worker.release.remote(), timeout=TIMEOUT_S)
    thread.join(TIMEOUT_S)
    assert len(entered) == 1  # entered without LockLostError
    wait_until(lambda: holders() == set())


def test_release_before_acquire_withdraws_it():
    lock_manager = manager("w")
    ray.get(lock_manager.release.remote("early"), timeout=TIMEOUT_S)
    assert ray.get(lock_manager.acquire.remote([(P, "w")], "early", None, 5, {}, 8), timeout=TIMEOUT_S) is None
    assert holders() == set()


@pytest.fixture
def own_manager():
    """A lock actor of its own, so these tests control every request it sees."""
    lock_manager = TransferQueueLockManager.remote(1)
    yield lock_manager
    ray.kill(lock_manager)


def acquire(lock_manager, keys, token, timeout=None, lease_s=30, num_shards=1):
    return lock_manager.acquire.remote([(P, key) for key in keys], token, timeout, lease_s, {}, num_shards)


def n_waiters(lock_manager):
    return ray.get(lock_manager.list_locks.remote(P), timeout=TIMEOUT_S)["waiters"]


def test_acquire_rejects_a_caller_hashing_over_another_shard_count(own_manager):
    # e.g. a process that attached before TransferQueue restarted with a new lock.num_shards
    with pytest.raises(ValueError, match="lock.num_shards is 1"):
        ray.get(acquire(own_manager, ["a"], "stale", num_shards=8), timeout=TIMEOUT_S)
    assert ray.get(acquire(own_manager, ["a"], "fresh", timeout=0), timeout=TIMEOUT_S) is not None


def test_waiters_are_granted_in_arrival_order(own_manager):
    ray.get(acquire(own_manager, ["a"], "holder"), timeout=TIMEOUT_S)
    both = acquire(own_manager, ["a", "b"], "both")
    wait_until(lambda: n_waiters(own_manager) == 1)
    # b is free, but granting it past the older [a, b] request would let single-key traffic starve it.
    assert ray.get(acquire(own_manager, ["b"], "newcomer", timeout=0), timeout=TIMEOUT_S) is None
    ray.get(own_manager.release.remote("newcomer"), timeout=TIMEOUT_S)  # as kv_global_lock does on timeout
    only_b = acquire(own_manager, ["b"], "only_b")
    wait_until(lambda: n_waiters(own_manager) == 2)
    ray.get(own_manager.release.remote("holder"), timeout=TIMEOUT_S)
    assert ray.get(both, timeout=TIMEOUT_S) is not None
    assert n_waiters(own_manager) == 1
    ray.get(own_manager.release.remote("both"), timeout=TIMEOUT_S)
    assert ray.get(only_b, timeout=TIMEOUT_S) is not None


def test_a_timed_out_call_keeps_its_place_until_withdrawn(own_manager):
    ray.get(acquire(own_manager, ["a"], "holder"), timeout=TIMEOUT_S)
    assert ray.get(acquire(own_manager, ["a"], "first", timeout=0.1), timeout=TIMEOUT_S) is None
    second = acquire(own_manager, ["a"], "second")
    wait_until(lambda: n_waiters(own_manager) == 2)
    ray.get(own_manager.release.remote("holder"), timeout=TIMEOUT_S)
    assert ray.get(own_manager.wait.remote("first", TIMEOUT_S), timeout=TIMEOUT_S) > 0
    assert n_waiters(own_manager) == 1
    ray.get(own_manager.release.remote("first"), timeout=TIMEOUT_S)
    assert ray.get(second, timeout=TIMEOUT_S) is not None


def test_a_waiter_that_withdraws_lets_the_next_one_through(own_manager):
    ray.get(acquire(own_manager, ["a"], "holder"), timeout=TIMEOUT_S)
    assert ray.get(acquire(own_manager, ["a", "b"], "both", timeout=0.1), timeout=TIMEOUT_S) is None
    only_b = acquire(own_manager, ["b"], "only_b")
    wait_until(lambda: n_waiters(own_manager) == 2)
    ray.get(own_manager.release.remote("both"), timeout=TIMEOUT_S)
    assert ray.get(only_b, timeout=TIMEOUT_S) is not None


def test_expired_leases_pass_the_key_down_the_queue(own_manager):
    # Nobody releases: each grant has to come from the actor expiring the lease ahead of it.
    ray.get(acquire(own_manager, ["a"], "dead", lease_s=0.5), timeout=TIMEOUT_S)
    first = acquire(own_manager, ["a"], "first", lease_s=0.5)
    wait_until(lambda: n_waiters(own_manager) == 1)
    second = acquire(own_manager, ["a"], "second")
    assert ray.get(second, timeout=TIMEOUT_S) is not None
    assert ray.get(first, timeout=TIMEOUT_S) is not None


def test_killed_holder_frees_the_key_after_its_lease():
    worker = Worker.remote()
    ray.get(worker.hold.remote("k", lease_s=1.5), timeout=TIMEOUT_S)
    ray.kill(worker)
    start = time.monotonic()
    with tq.kv_global_lock("k", P, timeout=TIMEOUT_S):
        assert time.monotonic() - start < 10


def test_lease_renews_and_lost_lease_raises():
    with tq.kv_global_lock("r", P, lease_s=2) as lease:
        time.sleep(3)  # outlives one lease only through renewal
        lease.check()

    with pytest.raises(tq.LockLostError):
        with tq.kv_global_lock("r", P, lease_s=1) as lease:
            with kvl._registry:  # stalls the renewer past the lease
                time.sleep(1.5)
            with pytest.raises(tq.LockLostError):
                lease.check()

    with pytest.raises(ValueError, match="body"):  # never masked by LockLostError
        with tq.kv_global_lock("r", P, lease_s=1) as lease:
            ray.get(manager("r").release.remote(lease.token), timeout=TIMEOUT_S)  # the server drops it
            wait_until(lambda: lease.lost)
            raise ValueError("body")
    wait_until(lambda: holders() == set())


def test_nesting_and_ordering_rules():
    with pytest.raises(ValueError):
        with tq.kv_global_lock([], P):
            pass
    with tq.kv_global_lock("n", P):
        with pytest.raises(RuntimeError, match="Nested kv_global_lock"):
            with tq.kv_global_lock("m", P):
                pass
        with tq.kv_local_lock("n", P):  # global, then local is allowed
            pass
    with tq.kv_local_lock("n", P):
        with pytest.raises(RuntimeError, match="before kv_local_lock"):
            with tq.kv_global_lock("n", P):
                pass

    async def main():
        with pytest.raises(RuntimeError, match="async_kv_global_lock"):
            with tq.kv_global_lock("n", P):
                pass
        async with tq.async_kv_global_lock("n", P):
            with pytest.raises(RuntimeError, match="Nested kv_global_lock"):
                async with tq.async_kv_global_lock("m", P):
                    pass

    asyncio.run(main())
    wait_until(lambda: holders() == set())


def actors_named(prefix):
    return [a["name"] for a in ray.util.list_named_actors(all_namespaces=True) if a["name"].startswith(prefix)]


def test_non_owner_close_releases_its_locks_and_keeps_the_managers():
    worker = Worker.remote()
    ray.get(worker.hold.remote("o"), timeout=TIMEOUT_S)
    ray.get(worker.close.remote(), timeout=TIMEOUT_S)
    wait_until(lambda: holders() == set())
    with tq.kv_global_lock("o", P, timeout=5):
        pass
    assert len(actors_named("TransferQueueLockManager_")) == kvl._shard_count() == 8


def test_owner_close_kills_the_managers_and_fails_waiters():
    """Tears TransferQueue down for this module; only the single-shard test runs after it."""
    held = tq.kv_global_lock("z", P)
    held.__enter__()
    worker = Worker.remote()
    waiting = worker.cycle.remote(["z"], 1)
    wait_until(lambda: tq.kv_lock_list(P)["waiters"] == 1)

    tq.close()
    with pytest.raises(RuntimeError, match="TransferQueueLockManager actor is gone"):
        ray.get(waiting, timeout=TIMEOUT_S)
    with pytest.raises(tq.LockLostError):
        held.__exit__(None, None, None)

    wait_until(lambda: actors_named("TransferQueueLockManager_") == [])
    with pytest.raises(RuntimeError, match="lock.enabled"):
        tq.kv_lock_list()


def test_one_lock_shard_still_works():
    wait_until(lambda: actors_named("TransferQueue") == [])
    tq.init(OmegaConf.create({**CONF, "lock": {"enabled": True, "num_shards": 1}}))
    assert actors_named("TransferQueueLockManager_") == ["TransferQueueLockManager_0"]
    with tq.kv_global_lock(["a", "b"], P, timeout=5) as lease:
        assert holders() == {"a", "b"} and list(lease.shards) == [0]
    wait_until(lambda: holders() == set())
