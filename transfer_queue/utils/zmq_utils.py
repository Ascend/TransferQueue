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
from dataclasses import dataclass, field
from typing import Any, Optional, TypeAlias
from uuid import uuid4

import psutil
import zmq

from transfer_queue.utils.serial_utils import _pack_data, _unpack_data
from transfer_queue.utils.utils import (
    ExplicitEnum,
    TransferQueueRole,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


bytestr: TypeAlias = bytes | bytearray | memoryview


class ZMQRequestType(ExplicitEnum):
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
    CLEAR_META = "CLEAR_META"
    CLEAR_META_RESPONSE = "CLEAR_META_RESPONSE"
    CLEAR_PARTITION = "CLEAR_PARTITION"
    CLEAR_PARTITION_RESPONSE = "CLEAR_PARTITION_RESPONSE"

    # CHECK_CONSUMPTION
    CHECK_CONSUMPTION = "CHECK_CONSUMPTION"
    CONSUMPTION_RESPONSE = "CONSUMPTION_RESPONSE"

    # CHECK_PRODUCTION
    CHECK_PRODUCTION = "CHECK_PRODUCTION"
    PRODUCTION_RESPONSE = "PRODUCTION_RESPONSE"

    # LIST_PARTITIONS
    GET_LIST_PARTITIONS = "GET_LIST_PARTITIONS"
    LIST_PARTITIONS_RESPONSE = "LIST_PARTITIONS_RESPONSE"

    # NOTIFY_DATA_UPDATE
    NOTIFY_DATA_UPDATE = "NOTIFY_DATA_UPDATE"
    NOTIFY_DATA_UPDATE_ACK = "NOTIFY_DATA_UPDATE_ACK"
    NOTIFY_DATA_UPDATE_ERROR = "NOTIFY_DATA_UPDATE_ERROR"


class ZMQServerInfo:
    def __init__(self, role: TransferQueueRole, id: str, ip: str, ports: dict[str, str]):
        self.role = role
        self.id = id
        self.ip = ip
        self.ports = ports

    def to_addr(self, port_name: str) -> str:
        return f"tcp://{self.ip}:{self.ports[port_name]}"

    def to_dict(self):
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
    request_type: ZMQRequestType
    sender_id: str
    receiver_id: str | None
    body: dict[str, Any]
    request_id: str
    timestamp: float

    # 仅在序列化/反序列化过程中使用的临时容器，不需要在构造时传入
    _buffers: list = field(default_factory=list, repr=False)

    @classmethod
    def create(
        cls,
        request_type: ZMQRequestType,
        sender_id: str,
        body: dict[str, Any],
        receiver_id: Optional[str] = None,
    ) -> "ZMQMessage":
        return cls(
            request_type=request_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
            body=body,
            request_id=str(uuid4().hex[:8]),
            timestamp=time.time(),
        )

    def serialize(self) -> list[bytestr]:
        """
        将消息序列化为 ZMQ Multipart 帧列表。
        Frame 0: Pickled Header (包含结构树和非 Tensor 数据)
        Frame 1...N: Raw Tensor Buffers
        """
        self._buffers = []

        # 1. 提取所有 Tensor 到 self._buffers，并将 body 转换为轻量级结构
        packed_body = _pack_data(self.body, self._buffers)

        # 2. 构建 Header
        header = {
            "request_type": self.request_type,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "body_structure": packed_body,  # 仅包含结构和元数据
        }

        # 3. 序列化 Header
        header_bytes = pickle.dumps(header)

        # 4. 组装最终发送列表：[Header, Buffer_0, Buffer_1, ...]
        return [header_bytes, *self._buffers]

    @classmethod
    def deserialize(cls, frames: list[bytestr]) -> "ZMQMessage":
        """
        从 ZMQ Multipart 帧列表反序列化消息。
        """
        if not frames:
            raise ValueError("Empty frames received")

        # 1. 解析 Header
        header_bytes = frames[0]
        header = pickle.loads(header_bytes)

        # 2. 获取数据帧 (Frames 1...N)
        raw_buffers = frames[1:]

        # 3. 递归重构 Body (Zero-Copy)
        body_structure = header["body_structure"]
        restored_body = _unpack_data(body_structure, raw_buffers)

        # 4. 构建对象
        msg = cls(
            request_type=header["request_type"],
            sender_id=header["sender_id"],
            receiver_id=header["receiver_id"],
            body=restored_body,
            request_id=header["request_id"],
            timestamp=header["timestamp"],
        )
        return msg


def get_free_port() -> str:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def create_zmq_socket(
    ctx: zmq.Context,
    socket_type: Any,
    identity: Optional[bytestr] = None,
) -> zmq.Socket:
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
