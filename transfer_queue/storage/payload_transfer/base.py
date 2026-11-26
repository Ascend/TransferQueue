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
from typing import Any, Callable, TypeAlias

from transfer_queue.utils.enum_utils import ExplicitEnum


class Operation(ExplicitEnum):
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
        """Bind ``handle`` to a server endpoint and return peer discovery metadata.

        A backend may either start its own dedicated listener or attach ``handle`` to a
        host-owned listener supplied via ``address`` (for example when control-plane RPCs
        share the same port). ``SimpleStorageUnit`` synchronizes storage state internally,
        so transport threads may invoke ``handle`` directly once request field data is ready.
        """

    # ==================== Client-side (called by AsyncSimpleStorageManager) ====================

    @abstractmethod
    async def request(
        self,
        peer_info: PayloadPeerInfo,
        header: RequestHeader,
        *,
        send: dict[str, Any] | None = None,
    ) -> PayloadReply:
        """Send one storage request to ``peer_info`` and await its ``PayloadReply``.

        Write path (``PUT`` / ``UPDATE``):
            Pass unencoded field values in ``send``. Returns ``PayloadReply(meta=..., payload=None)``
            after the StorageUnit commits the write. Once returned, the backend no longer reads
            ``send`` and source buffers may be reused.

        Read path (``GET``, ``send=None``):
            Backend allocates receive buffers and returns decoded field values in ``reply.payload``.

        Errors and in-flight memory:
            - ``StorageRejected``: StorageUnit rejected a delivered request; peer state stays usable.
            - ``TransferTimeout`` / ``TransferError``: set ``transient=True`` only when retrying on
              a healthy connection may succeed. Retain any ``send`` memory still accessed by
              in-flight native operations until safe.
        """

    # ==================== Shared Lifecycle ====================

    @abstractmethod
    def shutdown(self) -> None:
        """Stop admitting new requests and release resources created by this adapter.

        Called synchronously by ``SimpleStorageUnit`` or ``AsyncSimpleStorageManager``:
        - Server-side (``SimpleStorageUnit``): stop admitting new requests and wait for any
          active ``handle`` invocation to finish so ``handle`` is never called after return.
        - Client-side (``AsyncSimpleStorageManager``): matching ``AsyncTransferQueueClient.close()``,
          callers must finish or cancel in-flight ``request()`` coroutines first; requests queued
          behind internal concurrency gates must not be sent after ``shutdown()`` returns, and any
          buffers still referenced by cancelled native operations must be retained until safe.
        - Host-supplied listeners (passed via ``address``) and shared socket pools remain owned
          and closed by their creator rather than by ``shutdown()``.
        """
