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

"""Transport contract for one SimpleStorage payload request."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeAlias


class Operation(str, Enum):
    """Logical data-plane storage operation."""

    PUT = "PUT"
    GET = "GET"
    UPDATE = "UPDATE"


class TransferError(RuntimeError):
    """Transport or protocol failure."""


class TransferTimeout(TransferError):
    """Transport request or handshake timed out."""

    def __init__(self, message: str, *, wire_bytes: int = 0):
        super().__init__(message)
        self.wire_bytes = wire_bytes


class StorageRejected(RuntimeError):
    """Storage unit received the request and rejected it."""

    def __init__(self, message: str, *, missing_key: bool = False):
        super().__init__(message)
        self.missing_key = missing_key


@dataclass(frozen=True)
class RequestHeader:
    """Control parameters for one logical storage operation.

    ``params`` may carry arbitrary serializable Python objects (such as ``data_parser``,
    ``merge_fn``, ``parser_args``, and per-sample metadata) and scales with sample count;
    backends must not assume a tiny fixed control-frame size ceiling.
    """

    operation: Operation
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PayloadReply:
    """Result of one storage payload operation."""

    meta: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] | None = None
    wire_bytes: int = 0


StorageHandler: TypeAlias = Callable[[RequestHeader, dict[str, Any] | None], PayloadReply]


class PayloadTransfer(ABC):
    """Pluggable SimpleStorage data-plane transport owning both server and client endpoints."""

    # ==================== Server-side (called by SimpleStorageUnit) ====================

    @abstractmethod
    def listen(self, *, handle: StorageHandler, address: Any = None) -> dict[str, Any]:
        """Start the server endpoint, bind ``handle``, and return peer discovery metadata.

        ``SimpleStorageUnit`` synchronizes storage state access internally, so transport
        threads may invoke ``handle`` directly once request field data is ready. Backends
        should avoid blocking on network I/O while inside ``handle``.
        """

    # ==================== Client-side (called by PayloadClient) ====================

    @abstractmethod
    async def connect(self, peer_info: dict[str, Any]) -> Any:
        """Handshake with ``peer_info`` and return a connection usable on its owning event loop.

        A connection must support concurrent ``request`` calls on that loop, and a failed or
        cancelled ``connect`` must release any partial resources.
        """

    @abstractmethod
    async def request(
        self,
        connection: Any,
        header: RequestHeader,
        *,
        send: dict[str, Any] | None = None,
        recv: tuple[memoryview, ...] | None = None,
    ) -> PayloadReply:
        """Send one storage request over ``connection`` and await its ``PayloadReply``.

        Write path (``PUT`` / ``UPDATE``):
            Pass unencoded field values in ``send`` and leave ``recv=None``. Returns
            ``PayloadReply(meta=..., payload=None)`` after the StorageUnit commits the write.
            Once returned, the backend no longer reads ``send`` and source buffers may be reused.

        Read path (``GET``, ``send=None``):
            - ``recv=None`` (backend-allocated): backend allocates receive buffers and returns
              decoded field values in ``reply.payload``.
            - ``recv=(span0, ...)`` (pre-allocated): caller provides writable ``memoryview`` slices
              (retaining their backing memory); backend validates layout bounds, fills ``recv``
              in place, and returns ``PayloadReply(meta=..., payload=None)`` once readable.

        Errors and in-flight memory:
            - ``StorageRejected``: StorageUnit rejected a delivered request; ``connection`` stays usable.
            - ``TransferTimeout`` / ``TransferError``: transport or protocol failure. On any failure or
              cancellation, the backend must retain any ``send``/``recv`` memory still accessed in flight.
        """

    # ==================== Shared Lifecycle ====================

    @abstractmethod
    def shutdown(self) -> None:
        """Stop admitting requests, drain or terminate local work, and release resources.

        Called synchronously by ``SimpleStorageUnit`` or ``PayloadClient``. On return,
        ``handle`` must not remain active or be invoked later; retain buffers still
        referenced by outstanding native operations until safe to reclaim.
        """
