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

"""Client-side connection lifecycle and handshake sharing over a PayloadTransfer backend."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from functools import partial
from threading import Lock
from typing import Any

from omegaconf import DictConfig, OmegaConf

from transfer_queue.storage.payload_transfer.base import (
    PayloadReply,
    PayloadTransfer,
    RequestHeader,
    StorageRejected,
    TransferError,
    TransferTimeout,
)


class PayloadClient:
    """Manage per-loop peer handshakes, connection caching/eviction, and shutdown."""

    def __init__(
        self,
        transfer: PayloadTransfer,
        peers: Mapping[str, Any],
        *,
        timeout: float,
    ) -> None:
        self._transfer = transfer
        # Backend peer metadata is opaque; avoid OmegaConf interpolation on arbitrary strings.
        self._peers: dict[str, dict[str, Any]] = {
            target: OmegaConf.to_container(peer, resolve=False) if isinstance(peer, DictConfig) else peer  # type: ignore[misc]
            for target, peer in peers.items()
        }
        self._timeout = timeout
        self._connections: dict[tuple[asyncio.AbstractEventLoop, str], tuple[Any]] = {}
        self._connecting: dict[tuple[asyncio.AbstractEventLoop, str], asyncio.Future[tuple[Any]]] = {}
        self._closed = False
        self._shutdown_lock = Lock()
        self._shutdown_started = False

    @property
    def transfer(self) -> PayloadTransfer:
        """Underlying PayloadTransfer backend."""
        return self._transfer

    @property
    def closed(self) -> bool:
        """Whether ``close()`` has been called."""
        return self._closed

    async def request(
        self,
        target: str,
        header: RequestHeader,
        *,
        send: dict[str, Any] | None = None,
        recv: tuple[memoryview, ...] | None = None,
    ) -> PayloadReply:
        """Acquire or establish the loop-local connection to ``target`` and execute ``header``."""
        key = (asyncio.get_running_loop(), target)
        entry = None
        try:
            entry = await self._get_connection(key)
            return await self._transfer.request(entry[0], header, send=send, recv=recv)
        except StorageRejected:
            raise
        except BaseException:
            # A late failure must not evict a newer replacement connection for the same peer.
            if self._connections.get(key) is entry:
                self._connections.pop(key, None)
            raise

    async def _get_connection(self, key: tuple[asyncio.AbstractEventLoop, str]) -> tuple[Any]:
        """Share one handshake and its resulting connection per owning loop and target."""
        if self._closed:
            raise TransferError("payload transport is closed")
        entry = self._connections.get(key)
        if entry is not None:
            return entry
        # Closed loops cannot reuse their handles; sweep them across asyncio.run calls.
        for old_key in list(self._connections):
            if old_key[0].is_closed():
                self._connections.pop(old_key, None)
        task = self._connecting.get(key)
        if task is None:
            task = asyncio.create_task(self._connect(key))
            self._connecting[key] = task
            task.add_done_callback(partial(self._on_connect_done, key))
        return await asyncio.shield(task)

    async def _connect(self, key: tuple[asyncio.AbstractEventLoop, str]) -> tuple[Any]:
        """Run one bounded handshake with the target peer."""
        if self._closed:
            raise TransferError("payload transport is closed")
        _, target = key
        peer = self._peers[target]
        try:
            connection = await asyncio.wait_for(self._transfer.connect(peer), timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            raise TransferTimeout("payload connection timed out") from exc
        if self._closed:
            raise TransferError("payload transport closed during connection")
        # Wrap in a fresh 1-tuple to distinguish reconnects even if a backend returns the same handle.
        entry = (connection,)
        self._connections[key] = entry
        return entry

    def _on_connect_done(self, key: tuple[asyncio.AbstractEventLoop, str], task: asyncio.Future[tuple[Any]]) -> None:
        """Remove a finished handshake and release the backend if shutdown was waiting on it."""
        if self._connecting.get(key) is task:
            self._connecting.pop(key, None)
        # Retrieve failures even when every waiter cancelled its shielded await.
        if not task.cancelled():
            task.exception()
        if self._closed:
            self._shutdown_if_idle()

    def _shutdown_if_idle(self) -> None:
        """Shut down the underlying transport once after all running handshakes finish."""
        with self._shutdown_lock:
            if self._connecting or self._shutdown_started:
                return
            self._shutdown_started = True
        self._transfer.shutdown()

    def close(self) -> None:
        """Stop admitting requests, cancel pending handshakes, and shut down the transport."""
        self._closed = True
        self._connections.clear()
        for key, task in self._connecting.copy().items():
            loop = task.get_loop()
            if not task.done():
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    if not loop.is_closed():
                        raise
                if loop.is_running():
                    continue
            elif not task.cancelled():
                task.exception()
            # A stopped or closed loop may never run completion callbacks; drop it here so shutdown proceeds.
            self._connecting.pop(key, None)
        self._shutdown_if_idle()
