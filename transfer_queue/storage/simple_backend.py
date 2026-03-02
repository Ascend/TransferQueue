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

import dataclasses
import logging
import os
from dataclasses import dataclass
from threading import Thread
from typing import Any
from uuid import uuid4

import ray
import zmq
from ray.util import get_node_ip_address

from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.enum_utils import TransferQueueRole
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo, create_zmq_socket, get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler (for Ray Actor subprocess)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))  # in seconds
TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))


class StorageUnitData:
    """Storage unit for managing 2D data structure (samples × fields).

    Uses dict-based storage keyed by global_index (gi) instead of pre-allocated list.
    This allows O(1) insert/delete without index translation and avoids capacity bloat.

    Data Structure Example:
        field_data = {
            "field_name1": {gi0: item1, gi3: item2, ...},
            "field_name2": {gi0: item3, gi3: item4, ...},
        }
    """

    def __init__(self, storage_size: int):
        # field_name -> {gi: data} nested dict
        self.field_data: dict[str, dict] = {}
        # Capacity upper bound (not pre-allocated list length)
        self.storage_size = storage_size

    def get_data(self, fields: list[str], global_indexes: list) -> dict[str, list]:
        """Get data by global index keys.

        Args:
            fields: Field names used for getting data.
            global_indexes: Global indexes used as dict keys.

        Returns:
            dict with field names as keys, corresponding data list as values.
        """
        result: dict[str, list] = {}
        for field in fields:
            if field not in self.field_data:
                raise ValueError(
                    f"StorageUnitData get_data: field '{field}' not found. Available: {list(self.field_data.keys())}"
                )
            try:
                result[field] = [self.field_data[field][k] for k in global_indexes]
            except KeyError as e:
                raise KeyError(f"StorageUnitData get_data: key {e} not found in field '{field}'") from e
        return result

    def put_data(self, field_data: dict[str, Any], global_indexes: list) -> None:
        """Put data into storage.

        Args:
            field_data: Dict with field names as keys, data list as values.
            global_indexes: Global indexes to use as dict keys.
        """
        # Capacity is enforced per unique sample key, not counted per-field
        existing_keys: set = set()
        for fd in self.field_data.values():
            existing_keys.update(fd.keys())
        new_global_keys = [k for k in global_indexes if k not in existing_keys]
        if len(existing_keys) + len(new_global_keys) > self.storage_size:
            raise ValueError(
                f"Storage capacity exceeded: {len(existing_keys)} existing + "
                f"{len(new_global_keys)} new > {self.storage_size}"
            )
        for f, values in field_data.items():
            if f not in self.field_data:
                self.field_data[f] = {}
            for key, val in zip(global_indexes, values, strict=False):
                self.field_data[f][key] = val

    def clear(self, keys: list[int]) -> None:
        """Remove data at given global index keys, immediately freeing memory.

        Args:
            keys: Global indexes to remove.
        """
        for f in self.field_data:
            for key in keys:
                self.field_data[f].pop(key, None)


@ray.remote(num_cpus=1)
class SimpleStorageUnit:
    """A storage unit that provides distributed data storage functionality.

    This class represents a storage unit that can store data in a 2D structure
    (samples × data fields) and provides ZMQ-based communication for put/get/clear operations.

    Note: We use Ray decorator (@ray.remote) only for initialization purposes.
    We do NOT use Ray's .remote() call capabilities - the storage unit runs
    as a standalone process with its own ZMQ server socket.

    Attributes:
        storage_unit_id: Unique identifier for this storage unit.
        storage_unit_size: Maximum number of elements that can be stored.
        storage_data: Internal StorageUnitData instance for data management.
        zmq_server_info: ZMQ connection information for clients.
    """

    def __init__(self, storage_unit_size: int):
        """Initialize a SimpleStorageUnit with the specified size.

        Args:
            storage_unit_size: Maximum number of elements that can be stored in this storage unit.
        """
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4().hex[:8]}"
        self.storage_unit_size = storage_unit_size

        self.storage_data = StorageUnitData(self.storage_unit_size)

        self._init_zmq_socket()
        self._start_process_put_get()

    def _init_zmq_socket(self) -> None:
        """
        Initialize ZMQ socket connections between storage unit and controller/clients:
        - put_get_socket:
            Handle put/get requests from clients.
        """
        self.zmq_context = zmq.Context()
        self.put_get_socket = create_zmq_socket(self.zmq_context, zmq.ROUTER)
        self._node_ip = get_node_ip_address()

        while True:
            try:
                self._put_get_socket_port = get_free_port()
                self.put_get_socket.bind(f"tcp://{self._node_ip}:{self._put_get_socket_port}")
                break
            except zmq.ZMQError:
                logger.warning(f"[{self.storage_unit_id}]: Try to bind ZMQ sockets failed, retrying...")
                continue

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.STORAGE,
            id=str(self.storage_unit_id),
            ip=self._node_ip,
            ports={"put_get_socket": self._put_get_socket_port},
        )

    def _start_process_put_get(self) -> None:
        """Create a daemon thread and start put/get process."""
        self.process_put_get_thread = Thread(
            target=self._process_put_get, name=f"StorageUnitProcessPutGetThread-{self.storage_unit_id}", daemon=True
        )
        self.process_put_get_thread.start()

    def _process_put_get(self) -> None:
        """Process put_get_socket request."""
        poller = zmq.Poller()
        poller.register(self.put_get_socket, zmq.POLLIN)

        logger.info(f"[{self.storage_unit_id}]: start processing put/get requests...")

        perf_monitor = IntervalPerfMonitor(caller_name=self.storage_unit_id)

        while True:
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT * 1000))

            if self.put_get_socket in socks:
                messages = self.put_get_socket.recv_multipart()
                identity = messages.pop(0)
                serialized_msg = messages
                request_msg = ZMQMessage.deserialize(serialized_msg)
                operation = request_msg.request_type
                try:
                    logger.debug(f"[{self.storage_unit_id}]: receive operation: {operation}, message: {request_msg}")

                    if operation == ZMQRequestType.PUT_DATA:
                        with perf_monitor.measure(op_type="PUT_DATA"):
                            response_msg = self._handle_put(request_msg)
                    elif operation == ZMQRequestType.GET_DATA:
                        with perf_monitor.measure(op_type="GET_DATA"):
                            response_msg = self._handle_get(request_msg)
                    elif operation == ZMQRequestType.CLEAR_DATA:
                        with perf_monitor.measure(op_type="CLEAR_DATA"):
                            response_msg = self._handle_clear(request_msg)
                    else:
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.PUT_GET_OPERATION_ERROR,  # type: ignore[arg-type]
                            sender_id=self.storage_unit_id,
                            body={
                                "message": f"Storage unit id #{self.storage_unit_id} "
                                f"receive invalid operation: {operation}."
                            },
                        )
                except Exception as e:
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.PUT_GET_ERROR,  # type: ignore[arg-type]
                        sender_id=self.storage_unit_id,
                        body={
                            "message": f"Storage unit id #{self.storage_unit_id} occur error in processing "
                            f"put/get/clear request, detail error message: {str(e)}."
                        },
                    )

                self.put_get_socket.send_multipart([identity, *response_msg.serialize()], copy=False)

    def _handle_put(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle put request, add or update data into storage unit.

        Args:
            data_parts: ZMQMessage from client.

        Returns:
            Put data success response ZMQMessage.
        """
        try:
            global_indexes = data_parts.body["global_indexes"]
            field_data = data_parts.body["data"]  # field_data should be a TensorDict.
            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_put"
            ):
                self.storage_data.put_data(field_data, global_indexes)

            # After put operation finish, send a message to the client
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={},
            )

            return response_msg
        except Exception as e:
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"Failed to put data into storage unit id "
                    f"#{self.storage_unit_id}, detail error message: {str(e)}"
                },
            )

    def _handle_get(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle get request, return data from storage unit.

        Args:
            data_parts: ZMQMessage from client.

        Returns:
            Get data success response ZMQMessage, containing target data.
        """
        try:
            fields = data_parts.body["fields"]
            global_indexes = data_parts.body["global_indexes"]

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_get"
            ):
                result_data = self.storage_data.get_data(fields, global_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "data": result_data,
                },
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"Failed to get data from storage unit id #{self.storage_unit_id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def _handle_clear(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle clear request, clear data in storage unit according to given global_indexes.

        Args:
            data_parts: ZMQMessage from client, including target global_indexes.

        Returns:
            Clear data success response ZMQMessage.
        """
        try:
            global_indexes = data_parts.body["global_indexes"]

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_clear"
            ):
                self.storage_data.clear(global_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"Clear data in storage unit id #{self.storage_unit_id} successfully."},
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"Failed to clear data in storage unit id #{self.storage_unit_id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get the ZMQ server information for this storage unit.

        Returns:
            ZMQServerInfo containing connection details for this storage unit.
        """
        return self.zmq_server_info


@dataclass
class StorageMetaGroup:
    """Group of metadata for a specific storage unit."""

    storage_id: str
    global_indexes: list[int] = dataclasses.field(default_factory=list)
    partition_ids: list[str] = dataclasses.field(default_factory=list)
    batch_indexes: list[int] = dataclasses.field(default_factory=list)  # Original TensorDict positions
    field_names: list[str] = dataclasses.field(default_factory=list)  # Field names from BatchMeta

    def add_meta(self, global_index: int, partition_id: str, batch_index: int | None = None):
        """Add metadata to the group.

        Args:
            global_index: Global unique index across all storage
            partition_id: Partition identifier
            batch_index: Original position in input TensorDict (optional)
        """
        self.global_indexes.append(global_index)
        self.partition_ids.append(partition_id)
        if batch_index is not None:
            self.batch_indexes.append(batch_index)

    def get_global_indexes(self) -> list[int]:
        """Get all global indexes from stored samples"""
        return self.global_indexes

    def get_storage_keys(self) -> list[int]:
        """Return global indexes used as storage dict keys."""
        return self.global_indexes

    def get_batch_indexes(self) -> list[int]:
        """Get original TensorDict position indexes for _filter_storage_data."""
        return self.batch_indexes

    def get_field_names(self) -> list[str]:
        """Get all field names for this storage group."""
        return self.field_names

    @property
    def size(self) -> int:
        """Number of samples in this storage meta group"""
        return len(self.global_indexes)

    @property
    def is_empty(self) -> bool:
        """Check if this storage meta group is empty"""
        return len(self.global_indexes) == 0

    def __len__(self) -> int:
        """Number of samples in this storage meta group"""
        return self.size

    def __bool__(self) -> bool:
        """Truthiness based on whether group has samples"""
        return not self.is_empty

    def __str__(self) -> str:
        return f"StorageMetaGroup(storage_id='{self.storage_id}', size={self.size})"
