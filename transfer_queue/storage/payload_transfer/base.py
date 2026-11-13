"""Transport contract for one SimpleStorage payload request."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from transfer_queue.utils.serial_utils import decode, encode


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


class WriteOutcomeUnknown(TransferError):
    """The peer may already have applied the write. Do not replay it."""


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


# Each byte view retains its backing object. Spans are ordered codec frames.
Spans = tuple[memoryview, ...]
PeerInfo = dict[str, Any]
StorageHandler = Callable[[RequestHeader, dict[str, Any] | None], tuple[dict[str, Any], dict[str, Any] | None]]


def encode_spans(data: dict[str, Any]) -> Spans:
    return tuple(memoryview(frame).cast("B") for frame in encode(data))


def decode_spans(spans: Spans) -> dict[str, Any]:
    return decode(list(spans))


def validate_destinations(recv: Spans | None) -> None:
    if recv is not None and any(
        span.readonly or span.format != "B" or span.ndim != 1 or not span.c_contiguous for span in recv
    ):
        raise TransferError("receive spans must be writable contiguous byte views")


def validate_request(header: RequestHeader, send: Spans, recv: Spans | None) -> None:
    writing = header.operation in (Operation.PUT, Operation.UPDATE)
    if bool(send) != writing:
        raise TransferError("PUT and UPDATE require input spans. GET takes no input spans.")
    if writing and recv != ():
        raise TransferError("PUT and UPDATE expect an acknowledgement without data")
    validate_destinations(recv)


def place_reply(spans: Spans, recv: Spans | None) -> Spans:
    if recv is None:
        return spans
    validate_destinations(recv)
    if tuple(span.nbytes for span in spans) != tuple(span.nbytes for span in recv):
        raise TransferError("response span layout does not match the supplied receive spans")
    for source, destination in zip(spans, recv, strict=True):
        destination[:] = source
    return recv


class PayloadTransfer(ABC):
    """The selected backend owns the logical request through storage acknowledgement."""

    poll_interval_ms: int | None = None

    def poll_requests(self) -> None:
        """Make progress on the unit's existing worker if the backend needs polling."""
        return None

    @abstractmethod
    def listen(self, *, handle: StorageHandler, address: Any = None) -> PeerInfo:
        """Bind a local storage handler and return the bootstrap information."""

    @abstractmethod
    async def connect(self, peer_info: PeerInfo) -> Any:
        """Resolve an opaque connection usable by this backend."""

    @abstractmethod
    async def request(
        self, connection: Any, header: RequestHeader, *, send: Spans = (), recv: Spans | None = ()
    ) -> tuple[ReplyHeader, Spans]:
        """Return once response bytes are readable and request sources are reusable.

        None asks the backend to supply response memory. On failure, any memory
        still accessible to the transport must remain owned until access stops.
        """

    async def close(self) -> None:
        await asyncio.to_thread(self.shutdown)

    @abstractmethod
    def shutdown(self) -> None:
        """Drain local work and close transport resources for synchronous TQ owners."""
