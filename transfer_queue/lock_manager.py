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

"""Server side of ``kv_global_lock``: async Ray actors, each holding leased locks on the keys that hash to it."""

import asyncio
import heapq
import time
from collections import deque

import ray

# A withdrawal whose acquire never arrives (e.g. the caller died first) is dropped after this.
_WITHDRAWN_TTL_S = 300


# A waiting acquire call holds a concurrency slot for at most the caller's poll
# window, so even if waiters take every slot, renew and release get one within that window.
@ray.remote(num_cpus=0, max_concurrency=10_000)
class TransferQueueLockManager:
    """Exclusive leased locks on ``(partition_id, key)``. Runs on one event loop, so no thread locks."""

    def __init__(self, num_shards: int):
        # Kept so that a process that never ran tq.init() can learn how many shards to hash over.
        self._num_shards = num_shards
        self._holder: dict[tuple[str, str], str] = {}  # name -> token
        self._leases: dict[str, dict] = {}  # token -> names, lease_s, expires_at, granted_at, info
        # Min-heap of (expires_at, token). Release and renewal leave stale entries behind, which
        # _expire skips, so neither has to search the heap.
        self._expiries: list[tuple[float, str]] = []
        # Fires _expire at the earliest expiry, so a dead holder's keys pass on even while every
        # waiter sleeps until it is granted.
        self._timer: asyncio.TimerHandle | None = None
        self._timer_at = float("inf")
        # Waiting tokens per key in arrival order. A request is granted only when all its keys are
        # free and it heads every one of their queues: no newcomer can barge, and single-key
        # requests cannot starve an older multi-key one. All queues share one arrival order, so the
        # oldest waiter heads all of its queues and the wait can never form a cycle.
        self._queues: dict[tuple[str, str], deque[str]] = {}
        self._waiting: dict[str, dict] = {}  # token -> names, lease_s, info, done (set on grant or withdrawal)
        self._withdrawn: dict[str, float] = {}  # token -> when its tombstone expires

    def _grant(self, token: str, names: list[tuple[str, str]], lease_s: float, info: dict) -> None:
        now = time.monotonic()
        for name in names:
            self._holder[name] = token
        self._leases[token] = dict(names=names, lease_s=lease_s, expires_at=now + lease_s, granted_at=now, info=info)
        heapq.heappush(self._expiries, (now + lease_s, token))
        if now + lease_s < self._timer_at:
            if self._timer is not None:
                self._timer.cancel()
            self._timer_at = now + lease_s
            self._timer = asyncio.get_running_loop().call_later(lease_s, self._expire)

    def _leave_queues(self, token: str) -> dict:
        waiter = self._waiting.pop(token)
        for name in waiter["names"]:
            self._queues[name].remove(token)
            if not self._queues[name]:
                del self._queues[name]
        return waiter

    def _hand_off(self, names: list[tuple[str, str]]) -> None:
        """Grant each free name to the first waiter in its queue, if that waiter now heads all of its queues."""
        for name in names:
            if name in self._holder or name not in self._queues:
                continue
            token = self._queues[name][0]
            wanted = self._waiting[token]["names"]
            if all(n not in self._holder and self._queues[n][0] == token for n in wanted):
                waiter = self._leave_queues(token)
                self._grant(token, wanted, waiter["lease_s"], waiter["info"])
                waiter["done"].set()

    def _free(self, token: str) -> bool:
        lease = self._leases.pop(token, None)
        if lease is None:
            return False
        for name in lease["names"]:
            del self._holder[name]
        self._hand_off(lease["names"])
        return True

    def _expire(self) -> None:
        now = time.monotonic()
        while self._expiries and self._expiries[0][0] <= now:
            expires_at, token = heapq.heappop(self._expiries)
            lease = self._leases.get(token)
            if lease is not None and lease["expires_at"] == expires_at:
                self._free(token)
        # Re-arm for the next entry; a stale one only makes the timer fire early.
        self._timer, self._timer_at = None, float("inf")
        if self._expiries:
            self._timer_at = self._expiries[0][0]
            self._timer = asyncio.get_running_loop().call_later(max(0.0, self._timer_at - now), self._expire)

    async def acquire(self, names, token, timeout, lease_s, holder_info, num_shards) -> float | None:
        """Grant all ``names`` to ``token`` at once; return the seconds left on its lease.

        Returns None if still queued after ``timeout``. Calling again with the same token keeps
        its place in the queues; only ``release`` withdraws it. A token that is neither queued
        nor held (its grant expired before the caller saw it) queues afresh.
        """
        # A caller that hashed over another shard count may send a key to the wrong shard,
        # where nothing excludes the callers that sent it to the right one.
        if num_shards != self._num_shards:
            raise ValueError(
                f"kv_global_lock hashed keys over {num_shards} lock shards, but lock.num_shards is "
                f"{self._num_shards}; TransferQueue restarted, so call tq.close() and tq.init() again"
            )
        if token in self._withdrawn:  # kept until it expires: a retried call may still arrive
            return None
        if token not in self._leases and token not in self._waiting:
            if not any(name in self._holder or name in self._queues for name in names):
                self._grant(token, names, lease_s, holder_info)
            else:
                self._waiting[token] = dict(names=names, lease_s=lease_s, info=holder_info, done=asyncio.Event())
                for name in names:
                    self._queues.setdefault(name, deque()).append(token)
        waiter = self._waiting.get(token)
        if waiter is not None:
            try:
                await asyncio.wait_for(waiter["done"].wait(), timeout)
            except asyncio.TimeoutError:
                pass
        lease = self._leases.get(token)
        return None if lease is None else lease["expires_at"] - time.monotonic()

    def num_shards(self) -> int:
        """Return the global shard count recorded by this manager."""
        return self._num_shards

    def renew_many(self, tokens: list[str]) -> dict[str, bool]:
        """Extend each live lease by its own ``lease_s``; an expired or released one stays lost."""
        now = time.monotonic()
        alive = {}
        for token in tokens:
            lease = self._leases.get(token)
            alive[token] = lease is not None and lease["expires_at"] > now
            if lease is not None and alive[token]:
                lease["expires_at"] = now + lease["lease_s"]
                heapq.heappush(self._expiries, (lease["expires_at"], token))
        return alive

    async def release(self, token: str) -> None:
        """Free ``token``'s locks, or withdraw its acquire if it is still waiting or has not arrived."""
        if self._free(token):
            return
        if token in self._waiting:
            waiter = self._leave_queues(token)
            self._hand_off(waiter["names"])
            waiter["done"].set()  # its pending call finds no lease and returns None
        else:
            now = time.monotonic()
            # One TTL for all, so insertion order is expiry order: expired tombstones lead.
            while self._withdrawn:
                oldest, until = next(iter(self._withdrawn.items()))
                if until > now:
                    break
                del self._withdrawn[oldest]
            self._withdrawn[token] = now + _WITHDRAWN_TTL_S

    def list_locks(self, partition_id: str | None = None) -> dict:
        """Return current holders and the number of waiters, optionally for one partition."""
        now = time.monotonic()
        holders = []
        for (pid, key), token in self._holder.items():
            lease = self._leases[token]
            if partition_id in (None, pid) and lease["expires_at"] > now:
                holders.append(
                    dict(
                        partition_id=pid,
                        key=key,
                        holder=lease["info"],
                        held_s=now - lease["granted_at"],
                        lease_remaining_s=lease["expires_at"] - now,
                    )
                )
        waiters = sum(partition_id in (None, waiter["names"][0][0]) for waiter in self._waiting.values())
        return {"holders": holders, "waiters": waiters}
