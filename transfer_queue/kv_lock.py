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

"""Advisory locks on TransferQueue KV keys.

``kv_local_lock`` and ``async_kv_local_lock`` hold an exclusive mutex per
``(partition_id, key)``, shared by every thread and asyncio task of this process. They
are advisory: KV calls never check them. They do not exclude other processes or Ray
actors. Waiters on a key are served in FIFO order. Nesting is rejected; pass all keys
to a single call.

``kv_global_lock`` and ``async_kv_global_lock`` do the same across the Ray cluster through
``TransferQueueLockManager`` actors that split keys by hash, under a lease that renews
automatically. Take a global lock before a local one, never inside it.
"""

import asyncio
import os
import threading
import time
import uuid
import zlib
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

import ray
from ray.exceptions import RayActorError

# Not asyncio.Lock: it is bound to one event loop and is not thread-safe, while a process
# may run several loops plus plain threads. A key is held exactly while it is in _table,
# which maps it to its FIFO of waiters. _state_lock guards every change and is never held
# while waiting.
_state_lock = threading.Lock()
_table: dict[tuple[str, str], deque["_Waiter"]] = {}
# Whether this thread or task holds (or is acquiring) a local lock. A future global lock
# reads it to enforce the "global, then local" order.
_holding: ContextVar[bool] = ContextVar("_holding_kv_lock", default=False)


class _Waiter:
    def __init__(self, wake):
        self.wake = wake  # raises RuntimeError if the waiter's event loop is closed
        self.granted = False


def _take_or_queue(name: tuple[str, str], waiter: _Waiter) -> bool:
    with _state_lock:
        if name in _table:
            _table[name].append(waiter)
            return False
        _table[name] = deque()
        return True


def _release(name: tuple[str, str]) -> None:
    """Hand the key to its first live waiter, or free it. Call under _state_lock."""
    # Direct handoff keeps strict FIFO: no newcomer can take the key between this
    # release and the woken waiter running.
    waiters = _table[name]
    while waiters:
        waiter = waiters.popleft()
        try:
            waiter.wake()
        except RuntimeError:  # its event loop is closed, so nobody is left to own the key
            continue
        waiter.granted = True
        return
    del _table[name]


def _settle(name: tuple[str, str], waiter: _Waiter, keep_grant: bool) -> bool:
    """Resolve a wait that ended without a wake-up; return whether the caller owns the key."""
    # A grant can race the timeout or cancellation, so `granted` decides, not the
    # Event/Future. A raced grant is kept on timeout but passed on when the caller is
    # leaving, so a caller that has left never owns the key.
    with _state_lock:
        if not waiter.granted:
            if waiter in _table.get(name, ()):  # _release already dropped it if its loop closed
                _table[name].remove(waiter)
            return False
        if not keep_grant:
            _release(name)
        return keep_grant


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _acquire(name: tuple[str, str], deadline: float | None) -> None:
    event = threading.Event()
    waiter = _Waiter(event.set)
    if _take_or_queue(name, waiter):
        return
    try:
        if not event.wait(_remaining(deadline)):
            raise TimeoutError(f"Timed out waiting for kv_local_lock on {name}")
    except BaseException as e:  # timeout or KeyboardInterrupt
        if not _settle(name, waiter, keep_grant=isinstance(e, TimeoutError)):
            raise


async def _acquire_async(name: tuple[str, str], deadline: float | None) -> None:
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    waiter = _Waiter(lambda: loop.call_soon_threadsafe(future.set_result, None))
    if _take_or_queue(name, waiter):
        return
    try:
        await asyncio.wait([future], timeout=_remaining(deadline))
        if not future.done():
            raise TimeoutError(f"Timed out waiting for async_kv_local_lock on {name}")
    except BaseException as e:  # timeout or cancellation
        if not _settle(name, waiter, keep_grant=isinstance(e, TimeoutError)):
            raise


def _lock_names(
    keys: str | list[str], partition_id: str, held: ContextVar[bool] = _holding, kind: str = "kv_local_lock"
) -> list[tuple[str, str]]:
    keys = [keys] if isinstance(keys, str) else keys
    if not keys:
        raise ValueError(f"{kind} needs at least one key")
    if held.get():
        raise RuntimeError(f"Nested {kind} is not supported; pass all keys to a single call")
    # A fixed order keeps two overlapping multi-key calls from deadlocking.
    return [(partition_id, key) for key in sorted(set(keys))]


def _release_all(names: list[tuple[str, str]]) -> None:
    with _state_lock:
        for name in reversed(names):
            _release(name)


def _reject_running_loop(kind: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(f"{kind} would block the running event loop; use async_{kind} instead")


@contextmanager
def kv_local_lock(keys: str | list[str], partition_id: str, timeout: float | None = None):
    """Hold an exclusive process-local lock on ``keys`` in ``partition_id``.

    Blocks the calling thread, so use ``async_kv_local_lock`` inside a running event loop.
    Raises ``TimeoutError`` if not all keys are acquired within ``timeout`` seconds
    (``None`` waits forever) and ``RuntimeError`` if a local lock is already held.
    """
    _reject_running_loop("kv_local_lock")
    names, acquired = _lock_names(keys, partition_id), []
    deadline = None if timeout is None else time.monotonic() + timeout
    _holding.set(True)
    try:
        for name in names:
            _acquire(name, deadline)
            acquired.append(name)
        yield
    finally:
        _release_all(acquired)
        _holding.set(False)


@asynccontextmanager
async def async_kv_local_lock(keys: str | list[str], partition_id: str, timeout: float | None = None):
    """Async version of ``kv_local_lock``. Tasks created inside the block cannot take a local lock."""
    names, acquired = _lock_names(keys, partition_id), []
    deadline = None if timeout is None else time.monotonic() + timeout
    _holding.set(True)
    try:
        for name in names:
            await _acquire_async(name, deadline)
            acquired.append(name)
        yield
    finally:
        _release_all(acquired)
        _holding.set(False)


_holding_global: ContextVar[bool] = ContextVar("_holding_kv_global_lock", default=False)
# This process's leases, renewed in batches by one daemon thread that runs while any exist.
_registry = threading.Condition()
_leases: dict[str, "GlobalLease"] = {}
_renewer_running = False
_num_shards: int | None = None  # controller.num_lock_shards, asked of shard 0 on first use
_managers: dict = {}  # shard -> TransferQueueLockManager handle, looked up on first use
_NOT_INITIALIZED = "TransferQueueLockManager not found; call tq.init() first"
_MANAGER_GONE = "The TransferQueueLockManager actor is gone; did the process that ran tq.init() call tq.close()?"


class LockLostError(RuntimeError):
    """A kv_global_lock lease ran out or failed to renew, so another holder may own the keys."""


class GlobalLease:
    """Yielded by ``kv_global_lock``; ``check()`` raises ``LockLostError`` once the lease is lost."""

    def __init__(self, names: list[tuple[str, str]], shards: dict[int, list[tuple[str, str]]], lease_s: float):
        self.names, self.shards, self.lease_s = names, shards, lease_s
        self.token = f"{os.getpid()}:{uuid.uuid4().hex}"
        self.deadline = float("inf")  # local monotonic time the lease is known to last until, once granted
        self.lost = False
        self.pending: list[int] = []  # shards that may hold or await this token, so exit must release them

    def check(self) -> None:
        """Raise ``LockLostError`` if the lease was lost; call it right before writes in long sections."""
        if self.lost or time.monotonic() >= self.deadline:
            raise LockLostError(f"kv_global_lock lease on {self.names} was lost")


def _shard_count() -> int:
    global _num_shards
    if _num_shards is None:
        try:
            _num_shards = ray.get(_lock_manager(0).num_shards.remote())
        except RayActorError as e:
            raise RuntimeError(_MANAGER_GONE) from e
    return _num_shards


def _shard(name: tuple[str, str], num_shards: int) -> int:
    # crc32, not hash(): str hashes are salted per process, and every process must send a
    # key to the same lock actor.
    return zlib.crc32(f"{name[0]}\0{name[1]}".encode()) % num_shards


def _lock_manager(shard: int):
    if shard not in _managers:
        try:
            _managers[shard] = ray.get_actor(f"TransferQueueLockManager_{shard}", namespace="transfer_queue")
        except ValueError:
            raise RuntimeError(_NOT_INITIALIZED) from None
    return _managers[shard]


def _new_lease(keys: str | list[str], partition_id: str, lease_s: float) -> GlobalLease:
    names = _lock_names(keys, partition_id, _holding_global, "kv_global_lock")
    if _holding.get():
        raise RuntimeError("Take kv_global_lock before kv_local_lock, not while holding one")
    if lease_s <= 0:
        raise ValueError("lease_s must be positive")
    num_shards = _shard_count()
    shards: dict[int, list[tuple[str, str]]] = {}
    for name in names:
        shards.setdefault(_shard(name, num_shards), []).append(name)
    return GlobalLease(names, dict(sorted(shards.items())), lease_s)


def _send_acquire(lease: GlobalLease, shard: int, deadline: float | None):
    manager = _lock_manager(shard)
    holder = {"node_ip": ray.util.get_node_ip_address(), "pid": os.getpid(), "thread": threading.current_thread().name}
    lease.pending.append(shard)
    return manager.acquire.remote(lease.shards[shard], lease.token, _remaining(deadline), lease.lease_s, holder)


def _take_grant(lease: GlobalLease, shard: int, waited: float | None, sent: float) -> None:
    if waited is None:
        lease.pending.remove(shard)
        raise TimeoutError(f"Timed out waiting for kv_global_lock on {lease.names}")
    # The grant happened at least `waited` after `sent`, so this never outlives the shard's lease.
    lease.deadline = min(lease.deadline, sent + waited + lease.lease_s)


def _start_lease(lease: GlobalLease) -> None:
    global _renewer_running
    with _registry:
        _leases[lease.token] = lease
        _registry.notify()  # a shorter lease_s shortens the renewal period
        if not _renewer_running:
            _renewer_running = True
            threading.Thread(target=_renew_loop, name="kv_global_lock_renewer", daemon=True).start()


def _release_lease(lease: GlobalLease) -> None:
    with _registry:
        _leases.pop(lease.token, None)
    shards, lease.pending = lease.pending, []
    for shard in shards:
        # Releasing by token also withdraws an acquire that is still waiting or in flight, so
        # it is never granted to a caller that left; ray.cancel would miss a grant already made.
        try:
            _managers[shard].release.remote(lease.token)
        except Exception:  # close() dropped the handle, or the actor is gone
            pass


def _renew_loop() -> None:
    global _renewer_running
    last = time.monotonic()
    while True:
        with _registry:
            while True:
                if not _leases:
                    _renewer_running = False
                    return
                period = min(lease.lease_s for lease in _leases.values()) / 3
                remaining = last + period - time.monotonic()
                if remaining <= 0:
                    break
                _registry.wait(remaining)
            batch = dict(_leases)
        by_shard: dict[int, list[str]] = {}
        for token, lease in batch.items():
            for shard in lease.shards:
                by_shard.setdefault(shard, []).append(token)
        # Count from the send time: the server extends the lease from when the call arrives,
        # which is later, so the local deadline never outlives the server's.
        last = time.monotonic()
        calls = {}
        renewed: set[tuple[int, str]] = set()
        for shard, tokens in by_shard.items():
            try:
                calls[shard] = _managers[shard].renew_many.remote(tokens)
            except Exception:  # close() dropped the handle
                pass
        for shard, call in calls.items():
            try:
                alive = ray.get(call, timeout=max(0.0, last + 3 * period - time.monotonic()))
            except Exception:  # a dead lock manager or a timeout
                continue
            renewed.update((shard, token) for token, ok in alive.items() if ok)
        # A lease is lost as soon as any one of its shards fails to renew it.
        for token, lease in batch.items():
            if all((shard, token) in renewed for shard in lease.shards):
                lease.deadline = last + lease.lease_s
            else:
                lease.lost = True


@contextmanager
def kv_global_lock(keys: str | list[str], partition_id: str, timeout: float | None = None, lease_s: float = 30):
    """Hold an exclusive cluster-wide lock on ``keys`` in ``partition_id``; yield a ``GlobalLease``.

    The locks live in the ``controller.num_lock_shards`` actors that ``tq.init()`` creates,
    each key in the one its hash picks, and are held under a ``lease_s`` lease that a
    background thread renews. Each actor grants in arrival order: a request waits behind
    every earlier one that shares a key with it. Keys on several actors are taken one actor
    at a time, each all at once, so keys already granted stay held while a later actor's
    keys are awaited.
    Raises ``TimeoutError`` if the keys are not all granted within ``timeout`` seconds
    (``None`` waits forever), and ``RuntimeError`` when nested, taken while holding a
    ``kv_local_lock``, or called with a running event loop. Exit always releases the lock,
    then raises ``LockLostError`` if the lease was lost, unless the body already raised:
    an exception from the body is never masked.
    """
    _reject_running_loop("kv_global_lock")
    lease = _new_lease(keys, partition_id, lease_s)
    deadline = None if timeout is None else time.monotonic() + timeout
    _holding_global.set(True)
    try:
        # Every caller visits shards in ascending order, so no two callers can each hold a
        # shard the other awaits: multi-shard requests cannot deadlock.
        for shard in lease.shards:
            sent = time.monotonic()
            try:
                waited = ray.get(_send_acquire(lease, shard, deadline))
            except RayActorError as e:
                raise RuntimeError(_MANAGER_GONE) from e
            _take_grant(lease, shard, waited, sent)
        _start_lease(lease)
        yield lease
        lease.check()
    finally:
        _release_lease(lease)
        _holding_global.set(False)


@asynccontextmanager
async def async_kv_global_lock(
    keys: str | list[str], partition_id: str, timeout: float | None = None, lease_s: float = 30
):
    """Async version of ``kv_global_lock``. Tasks created inside the block cannot take a global lock."""
    lease = _new_lease(keys, partition_id, lease_s)
    deadline = None if timeout is None else time.monotonic() + timeout
    _holding_global.set(True)
    try:
        for shard in lease.shards:
            sent = time.monotonic()
            try:
                waited = await _send_acquire(lease, shard, deadline)
            except RayActorError as e:
                raise RuntimeError(_MANAGER_GONE) from e
            _take_grant(lease, shard, waited, sent)
        _start_lease(lease)
        yield lease
        lease.check()
    finally:
        _release_lease(lease)
        _holding_global.set(False)


def kv_lock_list(partition_id: str | None = None) -> dict:
    """Return ``{"holders": [...], "waiters": n}`` for live global locks, optionally in one partition.

    Each holder has ``partition_id``, ``key``, ``holder`` (node IP, pid, thread), ``held_s`` and
    ``lease_remaining_s``.
    """
    replies = ray.get([_lock_manager(shard).list_locks.remote(partition_id) for shard in range(_shard_count())])
    return {"holders": [h for r in replies for h in r["holders"]], "waiters": sum(r["waiters"] for r in replies)}


def _detach_lock_managers(release: bool) -> None:
    """Called by ``tq.close()``: give up this process's global leases and forget the actor handles."""
    global _num_shards, _managers
    with _registry:
        leases = list(_leases.values())
    for lease in leases:
        lease.lost = True
        if release:
            _release_lease(lease)
    _num_shards, _managers = None, {}


def _reset_after_fork() -> None:
    # The child inherits keys, leases and Ray handles that belong to threads and connections it lacks.
    global _state_lock, _table, _registry, _leases, _renewer_running, _managers
    _state_lock, _table = threading.Lock(), {}
    _registry, _leases, _renewer_running, _managers = threading.Condition(), {}, False, {}


os.register_at_fork(after_in_child=_reset_after_fork)
