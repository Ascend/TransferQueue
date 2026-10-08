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

import asyncio
import itertools
import os
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeAlias
from uuid import uuid4

import psutil
import ray
import zmq
import zmq.asyncio

from transfer_queue.utils.enum_utils import ExplicitEnum, Role
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.serial_utils import decode, encode

logger = get_logger(__name__)

# Identity prefixes of the peers allowed to reach a storage unit. The storage proxy drops
# anything else, so an identity built without these prefixes is silently unreachable.
STORAGE_MANAGER_IDENTITY_PREFIX = "TQ_STORAGE_"
METRICS_COLLECTOR_IDENTITY_PREFIX = "metrics_collector_"
STORAGE_CLIENT_IDENTITY_PREFIXES = (
    STORAGE_MANAGER_IDENTITY_PREFIX.encode(),
    METRICS_COLLECTOR_IDENTITY_PREFIX.encode(),
)

# Cap for every pool; see ZMQSocketPool for what it bounds. Small because it multiplies by
# peer count, and one socket per peer already avoids the repeated handshake.
TQ_SOCKET_POOL_SIZE = int(os.environ.get("TQ_SOCKET_POOL_SIZE", 4))

bytestr: TypeAlias = bytes | bytearray | memoryview


class ZMQRequestType(ExplicitEnum):
    """
    Enumerate all available request types in TransferQueue.
    """

    # HANDSHAKE
    HANDSHAKE = "HANDSHAKE"  # TransferQueueStorageUnit -> TransferQueueController
    HANDSHAKE_ACK = "HANDSHAKE_ACK"  # TransferQueueController  -> TransferQueueStorageUnit

    # GENERIC
    REQUEST_ERROR = "REQUEST_ERROR"  # TransferQueueController -> requester, when a request cannot be served

    # DATA_OPERATION
    GET_DATA = "GET"
    PUT_DATA = "PUT"
    GET_DATA_RESPONSE = "GET_DATA_RESPONSE"
    PUT_DATA_RESPONSE = "PUT_DATA_RESPONSE"
    CLEAR_DATA = "CLEAR_DATA"
    CLEAR_DATA_RESPONSE = "CLEAR_DATA_RESPONSE"

    PUT_GET_OPERATION_ERROR = "PUT_GET_OPERATION_ERROR"
    PUT_GET_ERROR = "PUT_GET_ERROR"
    PUT_ERROR = "PUT_ERROR"
    GET_ERROR = "GET_ERROR"
    CLEAR_DATA_ERROR = "CLEAR_DATA_ERROR"

    # META_OPERATION
    GET_META = "GET_META"
    GET_META_RESPONSE = "GET_META_RESPONSE"
    GET_PARTITION_META = "GET_PARTITION_META"
    GET_PARTITION_META_RESPONSE = "GET_PARTITION_META_RESPONSE"
    SET_CUSTOM_META = "SET_CUSTOM_META"
    SET_CUSTOM_META_RESPONSE = "SET_CUSTOM_META_RESPONSE"
    MARK_CLEARING = "MARK_CLEARING"
    MARK_CLEARING_RESPONSE = "MARK_CLEARING_RESPONSE"
    CLEAR_META = "CLEAR_META"
    CLEAR_META_RESPONSE = "CLEAR_META_RESPONSE"
    CLEAR_PARTITION = "CLEAR_PARTITION"
    CLEAR_PARTITION_RESPONSE = "CLEAR_PARTITION_RESPONSE"

    # GET_CONSUMPTION
    GET_CONSUMPTION = "GET_CONSUMPTION"
    CONSUMPTION_RESPONSE = "CONSUMPTION_RESPONSE"
    RESET_CONSUMPTION = "RESET_CONSUMPTION"
    RESET_CONSUMPTION_RESPONSE = "RESET_CONSUMPTION_RESPONSE"

    # GET_PRODUCTION
    GET_PRODUCTION = "GET_PRODUCTION"
    PRODUCTION_RESPONSE = "PRODUCTION_RESPONSE"

    # LIST_PARTITIONS
    GET_LIST_PARTITIONS = "GET_LIST_PARTITIONS"
    LIST_PARTITIONS_RESPONSE = "LIST_PARTITIONS_RESPONSE"

    # NOTIFY_DATA_UPDATE
    NOTIFY_DATA_UPDATE = "NOTIFY_DATA_UPDATE"
    NOTIFY_DATA_UPDATE_ACK = "NOTIFY_DATA_UPDATE_ACK"
    NOTIFY_DATA_UPDATE_ERROR = "NOTIFY_DATA_UPDATE_ERROR"

    # KV_INTERFACE
    KV_RETRIEVE_META = "KV_RETRIEVE_META"
    KV_RETRIEVE_META_RESPONSE = "KV_RETRIEVE_META_RESPONSE"
    KV_RETRIEVE_KEYS = "KV_RETRIEVE_KEYS"
    KV_RETRIEVE_KEYS_RESPONSE = "KV_RETRIEVE_KEYS_RESPONSE"
    KV_LIST = "KV_LIST"
    KV_LIST_RESPONSE = "KV_LIST_RESPONSE"

    # METRICS
    GET_METRICS = "GET_METRICS"
    METRICS_RESPONSE = "METRICS_RESPONSE"

    # CHECKPOINT
    SAVE_CONTROLLER_CHECKPOINT = "SAVE_CONTROLLER_CHECKPOINT"
    SAVE_CONTROLLER_CHECKPOINT_RESPONSE = "SAVE_CONTROLLER_CHECKPOINT_RESPONSE"
    SAVE_STORAGE_CHECKPOINT = "SAVE_STORAGE_CHECKPOINT"
    SAVE_STORAGE_CHECKPOINT_RESPONSE = "SAVE_STORAGE_CHECKPOINT_RESPONSE"
    LOAD_CONTROLLER_CHECKPOINT = "LOAD_CONTROLLER_CHECKPOINT"
    LOAD_CONTROLLER_CHECKPOINT_RESPONSE = "LOAD_CONTROLLER_CHECKPOINT_RESPONSE"
    LOAD_STORAGE_CHECKPOINT = "LOAD_STORAGE_CHECKPOINT"
    LOAD_STORAGE_CHECKPOINT_RESPONSE = "LOAD_STORAGE_CHECKPOINT_RESPONSE"

    # SELECTIVE DATA DUMP
    DESCRIBE_ROWS_BY_KEY = "DESCRIBE_ROWS_BY_KEY"
    DESCRIBE_ROWS_BY_KEY_RESPONSE = "DESCRIBE_ROWS_BY_KEY_RESPONSE"
    DUMP_ROWS = "DUMP_ROWS"
    DUMP_ROWS_RESPONSE = "DUMP_ROWS_RESPONSE"
    LOAD_ROWS = "LOAD_ROWS"
    LOAD_ROWS_RESPONSE = "LOAD_ROWS_RESPONSE"
    VALIDATE_DUMP_SCHEMA = "VALIDATE_DUMP_SCHEMA"
    VALIDATE_DUMP_SCHEMA_RESPONSE = "VALIDATE_DUMP_SCHEMA_RESPONSE"


class ZMQServerInfo:
    """
    TransferQueue server info class.
    """

    def __init__(self, role: Role, id: str, ip: str, ports: dict[str, int]):
        self.role = role
        self.id = id
        self.ip = ip
        self.ports = ports

    def to_addr(self, port_name: str) -> str:
        """Convert zmq port name to address string."""
        return format_zmq_address(self.ip, self.ports[port_name])

    def to_dict(self):
        """Convert ZMQServerInfo to dict."""
        return {
            "role": self.role,
            "id": self.id,
            "ip": self.ip,
            "ports": self.ports,
        }

    def __str__(self) -> str:
        return f"ZMQSocketInfo(role={self.role}, id={self.id}, ip={self.ip}, ports={self.ports})"


class ZMQMessageDecodeError(ValueError):
    """Raised when a received multipart message cannot be decoded into a ZMQMessage."""


def frame_nbytes(frame: Any) -> int | None:
    """Byte length of a single ZMQ frame, or None if the object exposes no buffer."""
    try:
        return memoryview(frame).nbytes
    except TypeError:
        return None


def describe_frames(frames: Sequence[Any], max_reported: int = 32) -> str:
    """Summarize a multipart message's frame layout: frame count and per-frame byte sizes.

    ``encode()`` emits a fixed layout (msgpack header in frame 0, one buffer per
    tensor/ndarray after it), so the frame count and sizes are what distinguish a
    genuinely corrupt payload from shifted multipart boundaries.
    """
    sizes = [frame_nbytes(frame) for frame in frames[:max_reported]]
    shown = ", ".join("?" if size is None else str(size) for size in sizes)
    ellipsis = ", ..." if len(frames) > max_reported else ""
    return f"num_frames={len(frames)}, frame_sizes=[{shown}{ellipsis}]"


@dataclass
class ZMQMessage:
    """
    ZMQMessage class for TransferQueue communication.
    """

    request_type: ZMQRequestType
    sender_id: str
    receiver_id: str | None
    body: dict[str, Any]
    request_id: str
    timestamp: float

    @classmethod
    def create(
        cls,
        request_type: ZMQRequestType,
        sender_id: str,
        body: dict[str, Any],
        receiver_id: str | None = None,
    ) -> "ZMQMessage":
        """Create ZMQMessage."""
        return cls(
            request_type=request_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            body=body,
            request_id=str(uuid4().hex[:8]),
            timestamp=time.time(),
        )

    def serialize(self) -> list:
        """Serialize using zero-copy msgpack; falls back to pickle for unsupported types."""
        msg_dict = {
            "request_type": self.request_type.value,  # Enum -> str for msgpack
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "body": self.body,
        }
        return encode(msg_dict)

    @classmethod
    def deserialize(cls, frames: list) -> "ZMQMessage":
        """Deserialize: choose decoding path based on the first frame marker (zero-copy or pickle fallback)."""
        if not frames:
            raise ValueError("Empty frames received")

        # Frame 0 is always the msgpack header; buffers for tensors/ndarrays follow it. A
        # zero-length frame 0 therefore means the multipart boundaries have shifted rather
        # than that the payload itself is bad, and decoding it would only report a
        # misleading msgpack error. Report the frame layout instead.
        if frame_nbytes(frames[0]) == 0:
            raise ZMQMessageDecodeError(f"leading frame is empty; {describe_frames(frames)}")

        try:
            result = decode(frames)
        except Exception as e:
            raise ZMQMessageDecodeError(f"{type(e).__name__}: {e}; {describe_frames(frames)}") from e

        return cls(
            request_type=ZMQRequestType(result["request_type"]),
            sender_id=result["sender_id"],
            receiver_id=result["receiver_id"],
            body=result["body"],
            request_id=result["request_id"],
            timestamp=result["timestamp"],
        )


def is_ipv6_address(ip: str) -> bool:
    """Check if the given IP address is an IPv6 address."""
    try:
        socket.inet_pton(socket.AF_INET6, ip)
        return True
    except OSError:
        return False


def format_zmq_address(ip: str, port: int) -> str:
    """
    Format IP and port for ZMQ binding/connecting.

    For IPv6 addresses, ZMQ requires the address to be wrapped in brackets:
    - IPv6: tcp://[::1]:port
    - IPv4: tcp://1.2.3.4:port

    Args:
        ip: IP address (IPv4 or IPv6)
        port: Port number

    Returns:
        Formatted ZMQ address string
    """
    if is_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


def get_node_ip_address() -> str:
    """A wrapper around Ray's get_node_ip_address().

    This function intentionally returns a raw IPv4/IPv6 address WITHOUT brackets.
    """

    return ray.util.get_node_ip_address().strip("[]")


def get_free_port(ip: str) -> int:
    """Get free port of the host.

    Args:
        ip: IP address to detect IPv6 and enable IPV6 socket option
    """
    is_ipv6 = is_ipv6_address(ip)
    family = socket.AF_INET6 if is_ipv6 else socket.AF_INET

    with socket.socket(family, socket.SOCK_STREAM) as sock:
        if is_ipv6:
            # Try to allow dual-stack if the platform supports it.
            try:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except (OSError, AttributeError):
                # Some platforms don't support IPV6_V6ONLY or this option;
                # in that case just ignore and use the default behavior.
                pass

        bind_host = "::" if is_ipv6 else ""
        sock.bind((bind_host, 0))
        return sock.getsockname()[1]


def create_zmq_socket(
    ctx: zmq.Context,
    socket_type: Any,
    ip: str,
    identity: bytestr | None = None,
) -> zmq.Socket:
    """Create ZMQ socket.

    Args:
        ctx: ZMQ context
        socket_type: ZMQ socket type
        ip: IP address to detect IPv6 and enable IPV6 socket option
        identity: Optional socket identity
    """
    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Enable IPv6 if the IP address is IPv6
    if is_ipv6_address(ip):
        socket.setsockopt(zmq.IPV6, 1)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5GB in bytes
    else:
        buf_size = -1  # Use system default buffer size

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)
    return socket


def _lease_owner() -> Any:
    """The running event loop, or the current thread outside one.

    A ZMQ socket is safe on neither a second thread nor a second event loop, so leases
    never cross either. pyzmq silently rebinds an async socket to whatever loop it next
    sees, and one bound to a *closed* loop is the "Bad file descriptor / SIGABRT" failure
    that per-call context churn used to cause. Returning the thread outside a loop lets
    synchronous callers (the metrics collector) share this pool unchanged.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return threading.current_thread()


def _owner_finished(owner: Any) -> bool:
    """Whether a lease owner can no longer serve its sockets."""
    if isinstance(owner, threading.Thread):
        return not owner.is_alive()
    return owner.is_closed()


# Addresses whose permit the current task already holds. A ContextVar because asyncio
# copies the context per task, so siblings under one gather do not see each other's.
_held_permits: ContextVar[frozenset[str]] = ContextVar("_held_permits", default=frozenset())


class ZMQSocketPool:
    """Lends connected DEALER sockets for one request scenario, reusing them across calls.

    One pool serves one kind of request -- controller RPC, storage RPC, notify, metrics --
    so ``socket_name`` and ``timeout`` are fixed here and every socket it holds is
    interchangeable but for the address it is connected to. Separate pools per scenario also
    keep them from interfering: a pool shared across scenarios reused nothing anyway, since
    scenarios differ by owning loop, socket name, or timeout.

    A lease is exclusive. Responses do not echo their request's ``request_id`` (see
    ``ZMQMessage.create``), so a reply is matched to its request only by arrival order:
    two concurrent users of one socket would read each other's replies. Each caller
    therefore gets a socket to itself and returns it only after a clean send/recv.

    Reuse avoids a TCP + ZMTP handshake per request and stops the peer's ROUTER from
    accreting a fresh identity every call.
    """

    def __init__(
        self,
        ctx: zmq.Context,
        owner_id: str,
        socket_name: str,
        *,
        timeout: int | None = None,
        maxsize: int | None = None,
    ):
        """
        Args:
            ctx: Long-lived context to open sockets on, sync or async. The pool borrows it
                and never terminates it; the owner outlives the pool.
            owner_id: Identity prefix for pooled sockets, for readable peer-side logs.
            socket_name: Port key in ``ZMQServerInfo.ports`` that every lease dials.
            timeout: Send/recv timeout in seconds applied to every socket, or None for none.
            maxsize: Idle sockets kept per bucket, at least 1, defaulting to
                ``TQ_SOCKET_POOL_SIZE``. A soft cap: a burst beyond it still gets sockets,
                and the excess is closed on return rather than made to wait. Counted per
                (owner, address), not per pool, so a pool dialling N peers may hold
                N*maxsize idle sockets.
        """
        # Resolved here rather than in the signature so the env var stays patchable; a
        # default bound at import froze whatever the environment held when this module loaded.
        maxsize = TQ_SOCKET_POOL_SIZE if maxsize is None else maxsize
        if maxsize < 1:
            # Below 1 nothing is ever parked, so every request pays a fresh connect while
            # still looking pooled. Reject it rather than silently disable reuse.
            raise ValueError(f"ZMQ socket pool size must be at least 1, got {maxsize}")
        self._ctx = ctx
        self._owner_id = owner_id
        self._socket_name = socket_name
        self._timeout = timeout
        self._maxsize = maxsize
        # Keyed by lease owner (see _lease_owner), then by address rather than peer id: a peer
        # restarted under the same id at a new address must not get a socket wired to the old.
        self._idle: dict[Any, dict[str, list[zmq.Socket]]] = {}
        self._lock = threading.Lock()
        # One semaphore per (owner, address), keyed like _idle because a semaphore belongs
        # to the loop that awaits it. Used by alease only; see there.
        self._permits: dict[Any, dict[str, asyncio.Semaphore]] = {}
        # A ROUTER silently drops a second peer claiming an identity it already has, and
        # owner_id alone repeats across nodes because client ids are pid-derived.
        self._identity_prefix = f"{owner_id}_{uuid4().hex[:8]}"
        self._counter = itertools.count()

    @contextmanager
    def lease(self, peer: ZMQServerInfo) -> Iterator[zmq.Socket]:
        """Yield a socket connected to ``peer``, returning it to the pool only on success.

        A plain (non-async) contextmanager on purpose: ``with`` still sees exceptions and
        ``CancelledError`` raised across ``await``s in its body, so this one definition
        serves both async and synchronous callers.

        Concurrency beyond ``maxsize`` opens extra sockets here rather than waiting; async
        callers that want the cap enforced use ``alease``.
        """
        address = self._address(peer)
        sock = self._take(address) or self._connect(peer, address)
        try:
            yield sock
        except BaseException:
            # Poisoned: the request may already be on the wire, so its reply could still
            # arrive and the next lessee would read it as its own. Cancellation counts too.
            sock.close(linger=0)
            raise
        else:
            self._release(address, sock)

    @asynccontextmanager
    async def alease(self, peer: ZMQServerInfo) -> AsyncIterator[zmq.Socket]:
        """Like ``lease``, but waits for a socket instead of opening one past ``maxsize``.

        This bounds sockets in flight at ``maxsize`` per (owner, address), so the context's
        socket budget follows configuration rather than peak concurrency.

        A permit is held for the whole body, so a task that leases inside another lease can
        wait on a permit it -- or a sibling mid-cycle -- already holds, and that hangs with
        no timeout. Any nesting therefore raises, including across pools: an RPC's reply is
        what releases its permit, so a second RPC belongs after the first returns. Notify
        already works this way, running once the puts it reports have completed.
        """
        address = self._address(peer)
        held = _held_permits.get()
        if held:
            raise RuntimeError(
                f"Lease on {address} from a task already holding {sorted(held)}. A lease "
                f"keeps its permit until its body ends, so nesting one inside another can "
                f"wait on a permit that is already held and hang. Complete the outer "
                f"request first, then start this one."
            )
        token = _held_permits.set(held | {address})
        permit = self._permit(address)
        try:
            async with permit:
                with self.lease(peer) as sock:
                    yield sock
        finally:
            _held_permits.reset(token)

    def _address(self, peer: ZMQServerInfo) -> str:
        port = peer.ports.get(self._socket_name)
        if port is None:
            raise RuntimeError(f"Socket '{self._socket_name}' not configured for server '{peer.id}'")
        return format_zmq_address(peer.ip, port)

    def _permit(self, address: str) -> asyncio.Semaphore:
        """The permit gating this (owner, address), created on first use by that owner."""
        with self._lock:
            return self._permits.setdefault(_lease_owner(), {}).setdefault(address, asyncio.Semaphore(self._maxsize))

    def _owner_buckets(self) -> dict[str, list[zmq.Socket]]:
        """Buckets for the current lease owner, first evicting any owner that has finished.

        Callers must hold ``self._lock``. A closed loop's sockets must not linger: a pooled
        async socket keeps its loop referenced, so they would never be collected, and reusing
        one is the SIGABRT hazard in _lease_owner. Sweeping here needs no background thread,
        and the loop count stays tiny (one per client, plus one per notify thread).
        """
        for owner in [o for o in self._idle if _owner_finished(o)]:
            self._close_all(self._idle.pop(owner))
            self._permits.pop(owner, None)
        return self._idle.setdefault(_lease_owner(), {})

    def _take(self, address: str) -> zmq.Socket | None:
        """Pop a live idle socket for *address*, discarding any found closed."""
        with self._lock:
            bucket = self._owner_buckets().get(address)
            while bucket:
                sock = bucket.pop()
                if not sock.closed:
                    return sock
        return None

    def _release(self, address: str, sock: zmq.Socket) -> None:
        """Return a socket, closing it if already closed or its bucket is full."""
        with self._lock:
            # A caller may close the socket itself without raising, as the notify path does
            # to discard a possibly-late ACK.
            if not sock.closed:
                bucket = self._owner_buckets().setdefault(address, [])
                if len(bucket) < self._maxsize:
                    bucket.append(sock)
                    return
        sock.close(linger=0)

    def _connect(self, peer: ZMQServerInfo, address: str) -> zmq.Socket:
        """Open and connect a new DEALER socket to *address*."""
        identity = f"{self._identity_prefix}_to_{peer.id}_{next(self._counter)}".encode()
        sock = create_zmq_socket(self._ctx, zmq.DEALER, peer.ip, identity=identity)
        try:
            if self._timeout is not None:
                sock.setsockopt(zmq.RCVTIMEO, self._timeout * 1000)
                sock.setsockopt(zmq.SNDTIMEO, self._timeout * 1000)
            sock.connect(address)
        except BaseException:
            # Nothing owns the socket until it is handed to a lease, so close it here or it
            # leaks. connect() raises on a malformed endpoint or a terminating context.
            sock.close(linger=0)
            raise
        return sock

    def close(self) -> None:
        """Close every idle socket. Safe to call twice, and after the context is gone.

        The pool stays usable afterwards: a lease still outstanding returns its socket to a
        fresh bucket. Callers destroy the context right after, so nothing is reused.
        """
        with self._lock:
            owned = list(self._idle.values())
            self._idle = {}
        for buckets in owned:
            self._close_all(buckets)

    def _close_all(self, buckets: dict[str, list[zmq.Socket]]) -> None:
        """Close every socket in *buckets*, tolerating an already-destroyed context."""
        for sock in itertools.chain.from_iterable(buckets.values()):
            try:
                if not sock.closed:
                    sock.close(linger=0)
            except Exception as e:
                logger.debug(f"[{self._owner_id}]: Error closing pooled socket: {e}")


def with_zmq_socket(
    *,
    get_peer: Callable[[Any, str | None], ZMQServerInfo],
    get_pool: Callable[[Any], ZMQSocketPool],
    resolve_target: Callable[[tuple, dict], str | None] | None = None,
):
    """Create a reusable async decorator that injects a pooled request socket.

    Lifecycle: resolve peer -> lease a socket from ``self``'s pool -> inject as the
    ``socket`` kwarg -> return it to the pool if the call succeeded, else discard it.

    The socket name and timeout come from the pool, which serves one request scenario.

    Args:
        get_peer: Callable that returns ``ZMQServerInfo`` for the target.
            For single-target scenarios, ignore the target parameter.
            Example: ``lambda self, target: self.server_info``
            Example: ``lambda self, target: self.storage_unit_infos[target]``
        get_pool: Callable that returns the pool for this scenario.
            Example: ``lambda self: self.controller_rpc_pool``
        resolve_target: Optional callable that extracts target identifier from
            function arguments. Receives (args, kwargs) and returns target name.
            Example: ``lambda args, kwargs: kwargs.get("target_storage_unit")``
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            target_name: str | None = None
            if resolve_target is not None:
                target_name = resolve_target(args, kwargs)

            server_info = get_peer(self, target_name)
            if server_info is None:
                raise RuntimeError(f"get_peer returned None for target '{target_name}'")

            pool = get_pool(self)
            if pool is None:
                raise RuntimeError("get_pool returned None")

            async with pool.alease(server_info) as sock:
                kwargs["socket"] = sock
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def process_zmq_server_info(handlers: dict[Any, Any] | Any):
    """Extract ZMQ server information from handler objects.

    Args:
        handlers: Dictionary of handler objects (controllers, storage managers or storage units),
                  or a single handler object

    Returns:
        If handlers is a dictionary: Dictionary mapping handler names to their ZMQ server information
        If handlers is a single object: ZMQ server information for that object

    Examples:
        >>> # Single handler
        >>> controller = TransferQueueController.remote(...)
        >>> info = process_zmq_server_info(controller)
        >>>
        >>> # Multiple handlers
        >>> handlers = {"storage_0": storage_0, "storage_1": storage_1}
        >>> info_dict = process_zmq_server_info(handlers)"""
    if not isinstance(handlers, dict):
        return ray.get(handlers.get_zmq_server_info.remote())  # type: ignore[union-attr, attr-defined]
    else:
        server_info = {}
        for name, handler in handlers.items():
            server_info[name] = ray.get(handler.get_zmq_server_info.remote())  # type: ignore[union-attr, attr-defined]
        return server_info
