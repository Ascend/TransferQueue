# Copyright 2026 The TransferQueue Team
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
from dataclasses import dataclass
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


FieldData: TypeAlias = dict[str, Any]
HeaderBody: TypeAlias = dict[str, Any]
# Caller-allocated destination byte ranges that retain their memory owners.
Spans: TypeAlias = tuple[memoryview, ...]
PeerInfo: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class RequestHeader:
    """Control metadata for one logical storage operation.

    ``body`` may carry arbitrary serializable Python objects (such as ``data_parser``,
    ``merge_fn``, ``parser_args``, and per-sample metadata) and scales with sample count;
    backends must not assume a tiny fixed control-frame size ceiling.
    """

    operation: Operation
    body: HeaderBody


@dataclass(frozen=True)
class ReplyHeader:
    """Control metadata returned after storage completes the operation."""

    body: HeaderBody
    wire_bytes: int = 0


# Local StorageUnit callback: (request_header, write_field_data) -> (reply_header_body, read_field_data)
StorageHandler: TypeAlias = Callable[[RequestHeader, FieldData | None], tuple[HeaderBody, FieldData | None]]


class PayloadTransfer(ABC):
    """Pluggable SimpleStorage data-plane transport owning both server and client endpoints."""

    # ==================== Server-side (called by SimpleStorageUnit) ====================

    @abstractmethod
    def listen(self, *, handle: StorageHandler, address: Any = None) -> PeerInfo:
        """Start the server endpoint, bind the local storage handler, and return ``PeerInfo``.

        ``SimpleStorageUnit`` synchronizes storage state access internally, so transport
        threads may invoke ``handle`` directly once request field data is ready. Backends
        should avoid blocking on network I/O while inside ``handle``.
        """

    # ==================== Client-side (called by AsyncSimpleStorageManager) ====================

    @abstractmethod
    async def connect(self, peer_info: PeerInfo) -> Any:
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
        send: FieldData | None = None,
        recv: Spans | None = None,
    ) -> tuple[ReplyHeader, FieldData | None]:
        """Execute one request and return once sources are reusable and results are readable.

        Args:
            connection: Handle returned by ``connect``.
            header: Operation kind and control metadata.
            send: Unencoded field values for write-like requests, or ``None`` when the
                request carries no field payload.
            recv: Optional caller-allocated destination byte ranges for direct placement:
                - ``None`` (default): if the server returns field payload, the backend
                  allocates receive memory and returns decoded ``FieldData``; if the server
                  returns no field payload, returns ``(ReplyHeader, None)``.
                - ``(span0, ...)``: write response payload directly into the caller-owned
                  spans and return ``(ReplyHeader, None)``.

        Raises:
            StorageRejected: Storage rejected a delivered request; ``connection`` stays usable.
            TransferTimeout: The request timed out on the transport.
            TransferError: Transport or protocol failure. On any transport failure or
                cancellation, retain memory still accessible to in-flight transport work.
        """

    # ==================== Shared Lifecycle ====================

    @abstractmethod
    def shutdown(self) -> None:
        """Stop admitting requests, drain or terminate local work, and release resources.

        Called synchronously by ``SimpleStorageUnit`` or ``AsyncSimpleStorageManager``. On
        return, ``handle`` must not remain active or be invoked later; retain buffers still
        referenced by outstanding native operations until safe to reclaim.
        """
