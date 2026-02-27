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

import logging
import os
import pickle
import socket
import time
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias
from uuid import uuid4

import psutil
import ray
import zmq

from transfer_queue.utils.enum_utils import ExplicitEnum, TransferQueueRole
from transfer_queue.utils.serial_utils import _decoder, _encoder

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


# 0xC1 is permanently reserved (invalid) in msgpack spec — safe to use as pickle fallback sentinel.
_PICKLE_FALLBACK_SENTINEL = b"\xc1\xfe\xed"

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
    KV_RETRIEVE_KEYS = "KV_RETRIEVE_KEYS"
    KV_RETRIEVE_KEYS_RESPONSE = "KV_RETRIEVE_KEYS_RESPONSE"
    KV_LIST = "KV_LIST"
    KV_LIST_RESPONSE = "KV_LIST_RESPONSE"


class ZMQServerInfo:
    """
    TransferQueue server info class.
    """

    def __init__(self, role: TransferQueueRole, id: str, ip: str, ports: dict[str, str]):
        self.role = role
        self.id = id
        self.ip = ip
        self.ports = ports

    def to_addr(self, port_name: str) -> str:
        """Convert zmq port name to address string."""
        return f"tcp://{self.ip}:{self.ports[port_name]}"

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
        receiver_id: Optional[str] = None,
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
        try:
            return list(_encoder.encode(msg_dict))
        except (TypeError, ValueError) as e:
            # Pickle fallback is a normal degradation path (e.g. body contains torch.dtype objects).
            # Log at INFO so operators are aware but not alarmed; use WARNING only for unexpected errors.
            logger.info(
                "ZMQMessage.serialize: msgpack encoding unsupported type (%s), using pickle fallback.",
                type(e).__name__,
            )
            return [_PICKLE_FALLBACK_SENTINEL, pickle.dumps(self)]

    @classmethod
    def deserialize(cls, frames: list) -> "ZMQMessage":
        """Deserialize: choose decoding path based on the first frame marker (zero-copy or pickle fallback)."""
        if not frames:
            raise ValueError("Empty frames received")

        # pickle fallback path: serialize() sets frame[0] to _PICKLE_FALLBACK_SENTINEL on failure.
        if len(frames) >= 2 and frames[0] == _PICKLE_FALLBACK_SENTINEL:
            return pickle.loads(frames[1])

        msg_dict = _decoder.decode(frames)
        return cls(
            request_type=ZMQRequestType(msg_dict["request_type"]),
            sender_id=msg_dict["sender_id"],
            receiver_id=msg_dict["receiver_id"],
            body=msg_dict["body"],
            request_id=msg_dict["request_id"],
            timestamp=msg_dict["timestamp"],
        )


def get_free_port() -> str:
    """Get free port of the host."""
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def create_zmq_socket(
    ctx: zmq.Context,
    socket_type: Any,
    identity: Optional[bytestr] = None,
) -> zmq.Socket:
    """Create ZMQ socket."""
    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

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


def process_zmq_server_info(
    handlers: dict[Any, Any] | Any,
):  # noqa: UP007
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
