"""Transport contract for one SimpleStorage payload request."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class Operation(str, Enum):
    PUT = "PUT"
    GET = "GET"
    UPDATE = "UPDATE"


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
        """Bind a local storage handler and return the bootstrap information.

        Invoke handle only on the StorageUnit worker, serially with administrative
        operations. Transport threads must hand requests to that worker rather than
        invoke handle directly; binding the handler does not provide this dispatch.
        """

    @abstractmethod
    async def connect(self, peer_info: PeerInfo) -> Any:
        """Handshake and return a connection supporting concurrent requests on its owning loop.

        The manager shares pending handshakes and caches connections per loop and peer.
        Cancelling one caller does not cancel the shared handshake. A failed or cancelled
        connect must release partial resources. The backend owns connection resources
        until shutdown, including handles evicted by the manager after request failure.
        """

    @abstractmethod
    async def request(
        self, connection: Any, header: RequestHeader, *, send: dict[str, Any] | None = None, recv: Spans | None = ()
    ) -> tuple[ReplyHeader, dict[str, Any] | None]:
        """Return once response bytes are readable and request sources are reusable.

        Send carries unencoded field values. With recv=None, return decoded fields
        owning their backing memory. With recv=(), return only an acknowledgement.
        A nonempty recv is reserved for writing into caller-owned output ranges.
        On failure, retain any memory still accessible to the transport.
        A transport failure or cancellation evicts the manager's cached connection;
        existing requests may finish, and a subsequent connect must be safe to use.
        StorageRejected completes the exchange without invalidating the connection.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Drain local work and close transport resources for synchronous TQ owners.

        The StorageUnit owner and backend must stop admitting requests and finish or
        reject accepted storage work before stopping the worker. On return, no storage
        callback may remain active or be invoked later; retain resources until safe.
        """
