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

import socket
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeAlias
from uuid import uuid4

import psutil
import ray
import zmq
import zmq.asyncio
from ray.util import get_node_ip_address

from transfer_queue.utils.enum_utils import ExplicitEnum, Role
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.serial_utils import decode, encode

logger = get_logger(__name__)


bytestr: TypeAlias = bytes | bytearray | memoryview


class ZMQRequestType(ExplicitEnum):
    """
    Enumerate all available request types in TransferQueue.
    """

    # HANDSHAKE
    HANDSHAKE = "HANDSHAKE"  # TransferQueueStorageUnit -> TransferQueueController
    HANDSHAKE_ACK = "HANDSHAKE_ACK"  # TransferQueueController  -> TransferQueueStorageUnit

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

        result = decode(frames)
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


def get_node_ip_address_raw() -> str:
    """A wrapper around Ray's get_node_ip_address().

    This function intentionally returns a raw IPv4/IPv6 address WITHOUT brackets.
    """

    return get_node_ip_address().strip("[]")


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

        sock.bind(("", 0))
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


def with_zmq_socket(
    socket_name: str,
    *,
    get_identity: Callable[[Any], str],
    get_peer: Callable[[Any, str | None], ZMQServerInfo],
    resolve_target: Callable[[tuple, dict], str | None] | None = None,
    timeout: int | None = None,
):
    """Create a reusable async decorator for request sockets.

    This decorator encapsulates the common socket lifecycle used by both
    client-side and storage-manager-side request paths:
    create context/socket -> connect -> inject socket -> close/term.

    Args:
        socket_name: Socket port key in ``ZMQServerInfo.ports``.
        get_identity: Callable that extracts owner identity from ``self``.
            Example: ``lambda self: self.client_id``
        get_peer: Callable that returns ``ZMQServerInfo`` for the target.
            For single-target scenarios, ignore the target parameter.
            Example: ``lambda self, target: self.server_info``
            Example: ``lambda self, target: self.storage_unit_infos[target]``
        resolve_target: Optional callable that extracts target identifier from
            function arguments. Receives (args, kwargs) and returns target name.
            Example: ``lambda args, kwargs: kwargs.get("target_storage_unit")``
        timeout: Optional timeout (seconds) for both send/recv operations.
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            owner_id = get_identity(self)
            if owner_id is None:
                raise RuntimeError("get_identity returned None")

            target_name: str | None = None
            if resolve_target is not None:
                target_name = resolve_target(args, kwargs)

            server_info = get_peer(self, target_name)
            if server_info is None:
                raise RuntimeError(f"get_peer returned None for target '{target_name}'")

            port = server_info.ports.get(socket_name)
            if port is None:
                raise RuntimeError(f"Socket '{socket_name}' not configured for server '{server_info.id}'")

            context = zmq.asyncio.Context()
            sock = None
            try:
                address = format_zmq_address(server_info.ip, port)
                identity = f"{owner_id}_to_{server_info.id}_{uuid4().hex[:8]}".encode()
                sock = create_zmq_socket(context, zmq.DEALER, server_info.ip, identity=identity)
                sock.connect(address)
                if timeout is not None:
                    sock.setsockopt(zmq.RCVTIMEO, timeout * 1000)
                    sock.setsockopt(zmq.SNDTIMEO, timeout * 1000)
                kwargs["socket"] = sock
                return await func(self, *args, **kwargs)
            finally:
                if sock is not None:
                    try:
                        if not sock.closed:
                            sock.close(linger=-1)
                    finally:
                        context.term()
                else:
                    context.term()

        return wrapper

    return decorator


def process_zmq_server_info(
    handlers: dict[Any, Any] | Any,
):
    """Extract ZMQ server information from handler objects.

    Args:
        handlers: Dictionary of handler objects (controllers, storage managers, or storage units),
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
    # Handle single handler object case
    if not isinstance(handlers, dict):
        return ray.get(handlers.get_zmq_server_info.remote())  # type: ignore[union-attr, attr-defined]
    else:
        # Handle dictionary case
        server_info = {}
        for name, handler in handlers.items():
            server_info[name] = ray.get(handler.get_zmq_server_info.remote())  # type: ignore[union-attr, attr-defined]
        return server_info
