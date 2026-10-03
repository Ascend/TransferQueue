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

"""Advisory, process-local locks on TransferQueue KV keys.

``kv_local_lock`` and ``async_kv_local_lock`` hold an exclusive mutex per
``(partition_id, key)``, shared by every thread and asyncio task of this process. They
are advisory: KV calls never check them. They do not exclude other processes or Ray
actors. Waiters on a key are served in FIFO order. Nesting is rejected; pass all keys
to a single call.
"""

import asyncio
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar

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


def _lock_names(keys: str | list[str], partition_id: str) -> list[tuple[str, str]]:
    keys = [keys] if isinstance(keys, str) else keys
    if not keys:
        raise ValueError("kv_local_lock needs at least one key")
    if _holding.get():
        raise RuntimeError("Nested kv_local_lock is not supported; pass all keys to a single call")
    # A fixed order keeps two overlapping multi-key calls from deadlocking.
    return [(partition_id, key) for key in sorted(set(keys))]


def _release_all(names: list[tuple[str, str]]) -> None:
    with _state_lock:
        for name in reversed(names):
            _release(name)


@contextmanager
def kv_local_lock(keys: str | list[str], partition_id: str, timeout: float | None = None):
    """Hold an exclusive process-local lock on ``keys`` in ``partition_id``.

    Blocks the calling thread, so use ``async_kv_local_lock`` inside a running event loop.
    Raises ``TimeoutError`` if not all keys are acquired within ``timeout`` seconds
    (``None`` waits forever) and ``RuntimeError`` if a local lock is already held.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("kv_local_lock would block the running event loop; use async_kv_local_lock instead")
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


def _reset_after_fork() -> None:
    # The child inherits a snapshot that may show keys held by threads it does not have.
    global _state_lock, _table
    _state_lock, _table = threading.Lock(), {}


os.register_at_fork(after_in_child=_reset_after_fork)
