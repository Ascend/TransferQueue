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

"""Async Ray actor that holds leased global-shard or process-family KV locks."""

import asyncio
import heapq
import time

import ray

# A withdrawal whose acquire never arrives (e.g. the caller died first) is dropped after this.
_WITHDRAWN_TTL_S = 300


# Every waiting acquire occupies a concurrency slot, so the limit sits far above Ray's async
# default of 1000 to keep renew and release from queueing behind waiters.
@ray.remote(num_cpus=0, max_concurrency=10_000)
class TransferQueueLockManager:
    """Exclusive leased locks on ``(partition_id, key)`` managed on one event loop."""

    def __init__(self, num_shards: int):
        # Kept so that a process that never ran tq.init() can learn how many shards to hash over.
        self._num_shards = num_shards
        self._holder: dict[tuple[str, str], str] = {}  # name -> token
        self._leases: dict[str, dict] = {}  # token -> names, lease_s, expires_at, granted_at, info
        # Min-heap of (expires_at, token). Release and renewal leave stale entries behind, which
        # _expire skips, so neither has to search the heap.
        self._expiries: list[tuple[float, str]] = []
        self._fences: dict[tuple[str, str], int] = {}  # never reset, so a key's fence only grows
        self._waiting: dict[str, tuple[list[tuple[str, str]], asyncio.Event]] = {}  # token -> names, wakeup
        # name -> wakeups of its waiters, as an insertion-ordered dict so they wake oldest first
        self._wakeups: dict[tuple[str, str], dict[asyncio.Event, None]] = {}
        self._withdrawn: dict[str, float] = {}  # token -> when its tombstone expires

    def _free(self, token: str) -> bool:
        lease = self._leases.pop(token, None)
        if lease is None:
            return False
        for name in lease["names"]:
            del self._holder[name]
            for wakeup in self._wakeups.get(name, ()):
                wakeup.set()
        return True

    def _expire(self, now: float) -> None:
        while self._expiries and self._expiries[0][0] <= now:
            expires_at, token = heapq.heappop(self._expiries)
            lease = self._leases.get(token)
            if lease is not None and lease["expires_at"] == expires_at:
                self._free(token)

    async def acquire(self, names, token, timeout, lease_s, holder_info):
        """Grant all ``names`` at once; return ``({key: fence}, waited_s)``, or None on timeout or withdrawal."""
        start = time.monotonic()
        deadline = None if timeout is None else start + timeout
        # Freeing a name sets the wakeups of its waiters only, so a release costs O(its waiters)
        # instead of waking every waiter on the actor.
        wakeup = asyncio.Event()
        self._waiting[token] = (names, wakeup)
        for name in names:
            self._wakeups.setdefault(name, {})[wakeup] = None
        try:
            while True:
                now = time.monotonic()
                self._expire(now)
                if self._withdrawn.pop(token, None) is not None:
                    return None
                # All-or-nothing: a waiter never holds part of its keys, so overlapping
                # requests cannot deadlock whatever order their keys come in.
                busy = [self._leases[self._holder[name]]["expires_at"] for name in names if name in self._holder]
                if not busy:
                    for name in names:
                        self._holder[name] = token
                        self._fences[name] = self._fences.get(name, 0) + 1
                    self._leases[token] = dict(
                        names=names, lease_s=lease_s, expires_at=now + lease_s, granted_at=now, info=holder_info
                    )
                    heapq.heappush(self._expiries, (now + lease_s, token))
                    return {key: self._fences[(pid, key)] for pid, key in names}, now - start
                if deadline is not None and now >= deadline:
                    return None
                # Nothing runs between this clear and the wait, so no wakeup can be missed. Waking
                # at the earliest blocking expiry lets _expire free it without a background task.
                wakeup.clear()
                wake_at = min(busy) if deadline is None else min(*busy, deadline)
                try:
                    await asyncio.wait_for(wakeup.wait(), wake_at - now)
                except asyncio.TimeoutError:
                    pass
        finally:
            del self._waiting[token]
            for name in names:
                del self._wakeups[name][wakeup]
                if not self._wakeups[name]:
                    del self._wakeups[name]

    def num_shards(self) -> int:
        """Return the global shard count recorded by this manager."""
        return self._num_shards

    def renew_many(self, tokens: list[str]) -> dict[str, bool]:
        """Extend each live lease by its own ``lease_s``; an expired or released one stays lost."""
        now = time.monotonic()
        alive = {}
        for token in tokens:
            lease = self._leases.get(token)
            is_alive = lease is not None and lease["expires_at"] > now
            alive[token] = is_alive
            if is_alive:
                assert lease is not None
                lease["expires_at"] = now + lease["lease_s"]
                heapq.heappush(self._expiries, (lease["expires_at"], token))
        return alive

    async def release(self, token: str) -> None:
        """Free ``token``'s locks, or withdraw its acquire if it is still waiting or has not arrived."""
        if not self._free(token):
            now = time.monotonic()
            self._withdrawn = {t: until for t, until in self._withdrawn.items() if until > now}
            self._withdrawn[token] = now + _WITHDRAWN_TTL_S
            if token in self._waiting:
                self._waiting[token][1].set()

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
        waiters = sum(partition_id in (None, names[0][0]) for names, _ in self._waiting.values())
        return {"holders": holders, "waiters": waiters}
