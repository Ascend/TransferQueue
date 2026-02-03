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
import zmq

from transfer_queue.utils.common import (
    get_env_bool,
)
from transfer_queue.utils.enum_utils import ExplicitEnum, TransferQueueRole
from transfer_queue.utils.serial_utils import _decoder, _encoder

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


bytestr: TypeAlias = bytes | bytearray | memoryview

TQ_ZERO_COPY_SERIALIZATION = get_env_bool("TQ_ZERO_COPY_SERIALIZATION", default=False)


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
        """
        Serialize message using unified MsgpackEncoder or pickle.
        Returns: list[bytestr] - [msgpack_header, *tensor_buffers] or [bytes]
        """
        if TQ_ZERO_COPY_SERIALIZATION:
            msg_dict = {
                "request_type": self.request_type.value,  # Enum -> str for msgpack
                "sender_id": self.sender_id,
                "receiver_id": self.receiver_id,
                "request_id": self.request_id,
                "timestamp": self.timestamp,
                "body": self.body,
            }
            return list(_encoder.encode(msg_dict))
        else:
            return [pickle.dumps(self)]

    @classmethod
    def deserialize(cls, frames: list) -> "ZMQMessage":
        """
        Deserialize message using unified MsgpackDecoder or pickle.
        """
        if not frames:
            raise ValueError("Empty frames received")

        if TQ_ZERO_COPY_SERIALIZATION:
            msg_dict = _decoder.decode(frames)
            return cls(
                request_type=ZMQRequestType(msg_dict["request_type"]),
                sender_id=msg_dict["sender_id"],
                receiver_id=msg_dict["receiver_id"],
                body=msg_dict["body"],
                request_id=msg_dict["request_id"],
                timestamp=msg_dict["timestamp"],
            )
        else:
            return pickle.loads(frames[0])


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
