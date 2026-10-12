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

"""kv_global_lock across Ray actors: exclusion, timeouts, queue order, leases, and close()."""

import asyncio
import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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

    def increment(self, n, keys):
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

    def wait_in_thread(self, keys):
        """Wait for ``keys`` on another thread, so ``close()`` can run meanwhile; ``outcome()`` reports the end."""

        def wait():
            try:
                with tq.kv_global_lock(keys, P, timeout=TIMEOUT_S):
                    self.ended = "entered"
            except Exception as e:
                self.ended = type(e).__name__

        self.ended = None
        threading.Thread(target=wait, daemon=True).start()

    def outcome(self):
        return self.ended

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


def test_a_late_reply_is_awaited_instead_of_sent_again(monkeypatch):
    lock_manager = manager("late")
    ray.get(lock_manager.acquire.remote([(P, "late")], "holder", 0, 30, {}, 8), timeout=TIMEOUT_S)
    real_send, sends = kvl._send_acquire, []

    def send_acquire(lease, shard, window):
        # The actor keeps the call past the caller's wait, as a call queued behind others would be.
        sends.append(window)
        return real_send(lease, shard, TIMEOUT_S)

    monkeypatch.setattr(kvl, "_send_acquire", send_acquire)
    with pytest.raises(TimeoutError):  # 0.5 s polls, each awaited for up to 1.5 s
        with tq.kv_global_lock("late", P, timeout=2.5, lease_s=1.5):
            pass
    assert len(sends) == 1
    ray.get(lock_manager.release.remote("holder"), timeout=TIMEOUT_S)
    wait_until(lambda: tq.kv_lock_list(P) == {"holders": [], "waiters": 0})


def test_acquire_carries_the_shard_count_its_keys_were_hashed_with(monkeypatch):
    n = kvl._shard_count()
    key = next(k for k in map(str, range(50)) if kvl._shard((P, k), n + 1) < n)
    monkeypatch.setattr(kvl, "_num_shards", n + 1)  # hashed before a restart with another lock.num_shards
    lease = kvl._new_lease(key, P, 30)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="restarted"):
        ray.get(kvl._send_acquire(lease, next(iter(lease._shards)), 0), timeout=TIMEOUT_S)
    kvl._release_lease(lease)


@pytest.fixture
def own_manager():
    """A lock actor of its own, so these tests control every request it sees.

    Sized as tq.init() sizes one for ``lock.max_concurrency: 2``.
    """
    lock_manager = TransferQueueLockManager.options(max_concurrency=2 + kvl._RESERVED_SLOTS).remote(1, 2)
    yield lock_manager
    ray.kill(lock_manager)


def acquire(lock_manager, keys, token, timeout=None, lease_s=30, num_shards=1):
    return lock_manager.acquire.remote([(P, key) for key in keys], token, timeout, lease_s, {}, num_shards)


def n_waiters(lock_manager):
    return ray.get(lock_manager.list_locks.remote(P), timeout=TIMEOUT_S)["waiters"]


def test_waiters_are_granted_in_arrival_order(own_manager):
    ray.get(acquire(own_manager, ["a"], "holder"), timeout=TIMEOUT_S)
    both = acquire(own_manager, ["a", "b"], "both")
    wait_until(lambda: n_waiters(own_manager) == 1)
    # b is free, but granting it past the older [a, b] request would let single-key traffic starve it.
    assert ray.get(acquire(own_manager, ["b"], "newcomer", timeout=0), timeout=TIMEOUT_S) is None
    ray.get(own_manager.release.remote("newcomer"), timeout=TIMEOUT_S)  # as kv_global_lock does on timeout
    # A call Ray runs after the withdrawal must not queue it again.
    assert ray.get(acquire(own_manager, ["b"], "newcomer", timeout=0), timeout=TIMEOUT_S) is None
    assert n_waiters(own_manager) == 1
    only_b = acquire(own_manager, ["b"], "only_b")
    wait_until(lambda: n_waiters(own_manager) == 2)
    ray.get(own_manager.release.remote("holder"), timeout=TIMEOUT_S)
    assert ray.get(both, timeout=TIMEOUT_S) is not None
    assert n_waiters(own_manager) == 1
    ray.get(own_manager.release.remote("both"), timeout=TIMEOUT_S)
    assert ray.get(only_b, timeout=TIMEOUT_S) is not None


def test_a_grant_that_expires_before_the_caller_sees_it_queues_afresh(own_manager):
    ray.get(acquire(own_manager, ["a"], "holder"), timeout=TIMEOUT_S)
    assert ray.get(acquire(own_manager, ["a"], "slow", timeout=0, lease_s=0.3), timeout=TIMEOUT_S) is None
    ray.get(own_manager.release.remote("holder"), timeout=TIMEOUT_S)  # grants "slow" between its calls
    wait_until(lambda: n_waiters(own_manager) == 0)
    time.sleep(0.6)  # the grant expires unseen
    assert ray.get(acquire(own_manager, ["a"], "other"), timeout=TIMEOUT_S) is not None
    # Calling again must not spin on an unknown token: it queues behind the current holder.
    assert ray.get(acquire(own_manager, ["a"], "slow", timeout=0.1), timeout=TIMEOUT_S) is None
    assert n_waiters(own_manager) == 1
    ray.get(own_manager.release.remote("other"), timeout=TIMEOUT_S)
    assert ray.get(acquire(own_manager, ["a"], "slow"), timeout=TIMEOUT_S) > 0


def test_a_waiter_that_stopped_polling_is_skipped(own_manager):
    ray.get(acquire(own_manager, ["a"], "holder"), timeout=TIMEOUT_S)
    # Queued once, then its caller died: granting it would idle the key for its whole lease.
    assert ray.get(acquire(own_manager, ["a"], "dead", timeout=0, lease_s=TIMEOUT_S), timeout=TIMEOUT_S) is None
    live = acquire(own_manager, ["a"], "live")
    wait_until(lambda: n_waiters(own_manager) == 2)
    # Date its last call back past the abandonment threshold instead of sleeping through it.
    backdate = own_manager.__ray_call__.remote(lambda self: self._waiting["dead"].update(seen=float("-inf")))
    ray.get(backdate, timeout=TIMEOUT_S)
    ray.get(own_manager.release.remote("holder"), timeout=TIMEOUT_S)
    assert ray.get(live, timeout=5) is not None
    assert n_waiters(own_manager) == 0


def test_renewal_and_release_run_while_waiters_fill_the_actor(own_manager):
    ray.get(acquire(own_manager, ["a"], "holder"), timeout=TIMEOUT_S)
    first = acquire(own_manager, ["a"], "first")
    wait_until(lambda: n_waiters(own_manager) == 1)  # an async actor may run two calls out of order
    second = acquire(own_manager, ["a"], "second")
    wait_until(lambda: n_waiters(own_manager) == 2)
    # Both waiting slots are taken: a third waiter is answered at once and keeps its place.
    assert ray.get(acquire(own_manager, ["a"], "third"), timeout=1) is None
    assert ray.get(own_manager.renew_many.remote(["holder"]), timeout=1) == {"holder": True}
    for token, call in (("holder", first), ("first", second)):
        ray.get(own_manager.release.remote(token), timeout=1)
        assert ray.get(call, timeout=TIMEOUT_S) is not None
    ray.get(own_manager.release.remote("second"), timeout=1)
    assert ray.get(acquire(own_manager, ["a"], "third"), timeout=1) > 0


def test_killed_holder_frees_the_key_after_its_lease():
    worker = Worker.remote()
    ray.get(worker.hold.remote("k", lease_s=1.5), timeout=TIMEOUT_S)
    ray.kill(worker)
    start = time.monotonic()
    with tq.kv_global_lock("k", P, timeout=TIMEOUT_S):
        assert time.monotonic() - start < 10


def test_lease_renews_and_lost_lease_raises():
    with tq.kv_global_lock("r", P, lease_s=2) as lease:
        assert isinstance(lease, tq.GlobalLease) and lease.names == [(P, "r")]
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
            ray.get(manager("r").release.remote(lease._token), timeout=TIMEOUT_S)  # the server drops it
            wait_until(lambda: lease._lost)
            raise ValueError("body")
    wait_until(lambda: holders() == set())


@ray.remote(num_cpus=0)
class Unresponsive:
    """Stands in for a lock actor on a partitioned node: renewals never answer."""

    async def renew_many(self, tokens):
        await asyncio.sleep(3600)


def test_an_unresponsive_shard_costs_no_lease_on_another(monkeypatch):
    healthy, hung = keys_on_two_shards("ur")
    other = tq.kv_global_lock(hung, P, timeout=TIMEOUT_S, lease_s=2)
    with ThreadPoolExecutor(1) as pool:  # a lock of its own needs a thread of its own
        other_lease = pool.submit(other.__enter__).result(TIMEOUT_S)
    unresponsive = Unresponsive.remote()
    with tq.kv_global_lock(healthy, P, timeout=TIMEOUT_S, lease_s=2) as lease:
        monkeypatch.setitem(kvl._managers, shard_of(hung), unresponsive)
        time.sleep(3)  # past both leases
        lease.check()
        with pytest.raises(tq.LockLostError):  # unanswered, it runs out once its deadline passes
            other_lease.check()
    with pytest.raises(tq.LockLostError):
        other.__exit__(None, None, None)
    monkeypatch.undo()
    ray.kill(unresponsive)
    wait_until(lambda: holders() == set())


def actors_named(prefix):
    return [a["name"] for a in ray.util.list_named_actors(all_namespaces=True) if a["name"].startswith(prefix)]


def test_attached_close_gives_up_its_leases_and_leaves_tq_running():
    early, late = keys_on_two_shards("ac")
    holder, attached = Worker.remote(), Worker.remote()
    ray.get(holder.hold.remote(late), timeout=TIMEOUT_S)
    ray.get(attached.wait_in_thread.remote([early, late]), timeout=TIMEOUT_S)
    wait_until(lambda: holders() == {early, late} and tq.kv_lock_list(P)["waiters"] == 1)

    ray.get(attached.close.remote(), timeout=TIMEOUT_S)
    # The waiter must stop, not queue its lost lease again behind the holder of `late`.
    wait_until(lambda: ray.get(attached.outcome.remote(), timeout=TIMEOUT_S) == "LockLostError")
    wait_until(lambda: tq.kv_lock_list(P)["waiters"] == 0 and holders() == {late})
    assert len(actors_named("TransferQueueLockManager_")) == kvl._shard_count()
    assert actors_named("TransferQueueController") == ["TransferQueueController"]
    ray.get(holder.release.remote(), timeout=TIMEOUT_S)
    wait_until(lambda: tq.kv_lock_list(P) == {"holders": [], "waiters": 0})


def test_owner_close_kills_the_managers_and_fails_waiters():
    """Tears TransferQueue down for this module, so it runs last."""
    held = tq.kv_global_lock("z", P)
    held.__enter__()
    worker = Worker.remote()
    waiting = worker.cycle.remote(["z"], 1)
    wait_until(lambda: tq.kv_lock_list(P)["waiters"] == 1)

    tq.close()
    wait_until(lambda: not kvl._renewer_running)  # even though the block has not exited
    with pytest.raises(RuntimeError, match="TransferQueueLockManager actor is gone"):
        ray.get(waiting, timeout=TIMEOUT_S)
    with pytest.raises(tq.LockLostError):
        held.__exit__(None, None, None)

    wait_until(lambda: actors_named("TransferQueue") == [])  # the controller and storage go too
    with pytest.raises(RuntimeError, match="lock.enabled"):
        tq.kv_lock_list()
