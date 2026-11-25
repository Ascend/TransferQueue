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


class PayloadPeerInfo:
    """Opaque peer discovery descriptor for one StorageUnit's payload endpoint.

    Kept as a plain class rather than a ``dict`` or ``dataclass`` so OmegaConf
    (with ``allow_objects=True``) stores instances as atomic ``AnyNode`` objects
    without ``${...}`` interpolation or dumping binary descriptors into logs.
    """

    def __init__(self, backend: str, address: Any = None):
        self.backend = backend
        self.address = address

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PayloadPeerInfo):
            return NotImplemented
        return self.backend == other.backend and self.address == other.address

    def __repr__(self) -> str:
        if isinstance(self.address, bytes | bytearray | memoryview):
            addr_repr = f"<{len(self.address)} bytes>"
        else:
            text = repr(self.address) if isinstance(self.address, str) else str(self.address)
            addr_repr = text if len(text) <= 96 else f"{text[:93]}..."
        return f"PayloadPeerInfo(backend={self.backend!r}, address={addr_repr})"

    __str__ = __repr__


class DeliveryState(str, Enum):
    """Whether a failed payload request may have been executed by the StorageUnit."""

    UNSENT = "unsent"
    REJECTED = "rejected"
    IN_DOUBT = "in_doubt"


class TransferError(RuntimeError):
    """Transport, protocol, or peer-configuration failure.

    ``transient`` indicates whether retrying on a fresh connection could recover.
    ``delivery`` distinguishes failures before sending (``UNSENT``), ingress rejections
    before storage execution (``REJECTED``), and post-send failures whose server-side
    execution is uncertain (``IN_DOUBT``).
    """

    def __init__(
        self,
        message: str,
        *,
        transient: bool = False,
        delivery: DeliveryState = DeliveryState.IN_DOUBT,
    ):
        super().__init__(message)
        self.transient = transient
        self.delivery = delivery


class TransferTimeout(TransferError):
    """Transport request timed out."""

    def __init__(
        self,
        message: str,
        *,
        wire_bytes: int = 0,
        delivery: DeliveryState = DeliveryState.IN_DOUBT,
    ):
        super().__init__(message, transient=True, delivery=delivery)
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
    """Pluggable SimpleStorage data-plane transport.

    Each backend owns its own peer connection establishment, reuse, fault isolation,
    and teardown. When a failure occurs on a multiplexed endpoint or remote agent, the
    backend decides whether only that request failed or the peer handle must be retired
    after draining in-flight operations and retaining accessible memory.
    """

    # ==================== Server-side (called by SimpleStorageUnit) ====================

    @abstractmethod
    def listen(self, *, handle: StorageHandler, address: Any = None) -> PayloadPeerInfo:
        """Start the server endpoint, bind ``handle``, and return peer discovery metadata.

        ``SimpleStorageUnit`` synchronizes storage state access internally, so transport
        threads may invoke ``handle`` directly once request field data is ready. Backends
        should avoid blocking on network I/O while inside ``handle``.
        """

    # ==================== Client-side (called by AsyncSimpleStorageManager) ====================

    @abstractmethod
    async def request(
        self,
        peer_info: PayloadPeerInfo,
        header: RequestHeader,
        *,
        send: dict[str, Any] | None = None,
        recv: tuple[memoryview, ...] | None = None,
    ) -> PayloadReply:
        """Send one storage request to ``peer_info`` and await its ``PayloadReply``.

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
            - ``StorageRejected``: StorageUnit rejected a delivered request; peer state stays usable.
            - ``TransferTimeout`` / ``TransferError``: set ``transient=True`` only when a retry over
              a healthy connection may succeed, and set ``delivery`` to ``UNSENT`` / ``REJECTED`` /
              ``IN_DOUBT`` so callers never replay in-doubt writes over concurrent updates. Retain any
              ``send``/``recv`` memory still accessed by in-flight native operations until safe.
        """

    # ==================== Shared Lifecycle ====================

    @abstractmethod
    def shutdown(self) -> None:
        """Stop admitting new requests, drain or terminate local work, and release owned resources.

        Called synchronously by ``SimpleStorageUnit`` or ``AsyncSimpleStorageManager``. On return,
        ``handle`` must not remain active or be invoked later; retain buffers still referenced by
        outstanding native operations until safe to reclaim.
        """
