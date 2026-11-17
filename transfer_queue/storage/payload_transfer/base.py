"""Transport contract for one SimpleStorage payload request."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Operation(str, Enum):
    PUT = "PUT_DATA"
    GET = "GET_DATA"
    UPDATE = "UPDATE_DATA"


class TransferError(RuntimeError):
    """Transport or protocol failure."""


class TransferTimeout(TransferError):
    def __init__(self, message: str, *, wire_bytes: int = 0):
        super().__init__(message)
        self.wire_bytes = wire_bytes


class StorageRejected(RuntimeError):
    def __init__(self, message: str, *, missing_key: bool = False):
        super().__init__(message)
        self.missing_key = missing_key


@dataclass(frozen=True)
class RequestHeader:
    operation: Operation
    body: dict[str, Any]


@dataclass(frozen=True)
class ReplyHeader:
    body: dict[str, Any]
    wire_bytes: int = 0


# Reserved destinations describe logical output ranges and retain their owners.
Spans = tuple[memoryview, ...]
PeerInfo = dict[str, Any]
StorageHandler = Callable[[RequestHeader, dict[str, Any] | None], tuple[dict[str, Any], dict[str, Any] | None]]


class PayloadTransfer(ABC):
    """The selected backend owns the logical request through storage acknowledgement."""

    @abstractmethod
    def listen(self, *, handle: StorageHandler, address: Any = None) -> PeerInfo:
        """Bind a local storage handler and return the bootstrap information."""

    @abstractmethod
    async def connect(self, peer_info: PeerInfo) -> Any:
        """Resolve an opaque connection usable by this backend."""

    @abstractmethod
    async def request(
        self, connection: Any, header: RequestHeader, *, send: dict[str, Any] | None = None, recv: Spans | None = ()
    ) -> tuple[ReplyHeader, dict[str, Any] | None]:
        """Return once response bytes are readable and request sources are reusable.

        Send carries unencoded field values. With recv=None, return decoded fields
        owning their backing memory. With recv=(), return only an acknowledgement.
        A nonempty recv is reserved for writing into caller-owned output ranges.
        On failure, retain any memory still accessible to the transport.
        """

    async def close(self) -> None:
        await asyncio.to_thread(self.shutdown)

    @abstractmethod
    def shutdown(self) -> None:
        """Drain local work and close transport resources for synchronous TQ owners."""
