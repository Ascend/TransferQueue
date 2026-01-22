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
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

import torch
import zmq
from tensordict import NonTensorStack, TensorDict
from torch import Tensor

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.clients.factory import StorageClientFactory
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo, create_zmq_socket

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

# ZMQ timeouts (in seconds) and retry configurations
TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))
TQ_STORAGE_HANDSHAKE_TIMEOUT = int(os.environ.get("TQ_STORAGE_HANDSHAKE_TIMEOUT", 30))
TQ_STORAGE_HANDSHAKE_RETRY_INTERVAL = int(os.environ.get("TQ_STORAGE_HANDSHAKE_RETRY_INTERVAL", 1))
TQ_STORAGE_HANDSHAKE_MAX_RETRIES = int(os.environ.get("TQ_STORAGE_HANDSHAKE_MAX_RETRIES", 3))
TQ_DATA_UPDATE_RESPONSE_TIMEOUT = int(os.environ.get("TQ_DATA_UPDATE_RESPONSE_TIMEOUT", 30))


class TransferQueueStorageManager(ABC):
    """Base class for storage layer. It defines the interface for data operations and
    generally provides handshake & notification capabilities."""

    def __init__(self, config: dict[str, Any]):
        self.storage_manager_id = f"TQ_STORAGE_{uuid4().hex[:8]}"
        self.config = config
        self.controller_info = config.get("controller_info", None)  # type: ZMQServerInfo

        self.data_status_update_socket = None
        self.controller_handshake_socket = None

        self.zmq_context = None
        self._connect_to_controller()

    def _connect_to_controller(self) -> None:
        """Initialize ZMQ sockets between storage unit and controller for handshake."""
        if not isinstance(self.controller_info, ZMQServerInfo):
            raise ValueError(f"controller_info should be ZMQServerInfo, but got {type(self.controller_info)}")

        try:
            # create zmq context
            self.zmq_context = zmq.Context()

            # create zmq sockets for handshake and data status update
            self.controller_handshake_socket = create_zmq_socket(
                self.zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_manager_id}-controller_handshake_socket-{uuid4().hex[:8]}".encode(),
            )
            self.data_status_update_socket = create_zmq_socket(
                self.zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_manager_id}-data_status_update_socket-{uuid4().hex[:8]}".encode(),
            )
            self.data_status_update_socket.connect(self.controller_info.to_addr("data_status_update_socket"))

            # do handshake with controller
            self._do_handshake_with_controller()

        except Exception as e:
            logger.error(f"Failed to connect to controller: {e}")
            raise

    def _do_handshake_with_controller(self) -> None:
        """Handshake with controller to establish connection with retransmission mechanism."""
        is_connected: bool = False
        pending_connection: bool = True
        handshake_retries: int = 0

        # Create zmq poller for handshake confirmation between controller and storage manager
        poller = zmq.Poller()

        self.controller_handshake_socket.connect(self.controller_info.to_addr("handshake_socket"))
        logger.debug(
            f"[{self.storage_manager_id}]: Handshake connection from storage manager id #{self.storage_manager_id} "
            f"to controller id #{self.controller_info.id} establish successfully."
        )
        poller.register(self.controller_handshake_socket, zmq.POLLIN)

        # Initial handshake request send
        self._send_handshake_requests()

        start_time = time.time()
        last_retry_time = time.time()

        while (
            not is_connected  # Only one controller to connect to
            and time.time() - start_time < TQ_STORAGE_HANDSHAKE_TIMEOUT
        ):
            # Check for timeout and retransmission
            current_time = time.time()
            if pending_connection:
                if (
                    current_time - last_retry_time >= TQ_STORAGE_HANDSHAKE_RETRY_INTERVAL
                    and handshake_retries < TQ_STORAGE_HANDSHAKE_MAX_RETRIES
                ):
                    logger.warning(
                        f"[{self.storage_manager_id}]: Retransmitting handshake "
                        f"to controller {self.controller_info.id}, "
                        f"attempt {handshake_retries + 1}/{TQ_STORAGE_HANDSHAKE_MAX_RETRIES}"
                    )
                    self._send_handshake_requests()
                    last_retry_time = current_time
                    handshake_retries += 1
                elif handshake_retries >= TQ_STORAGE_HANDSHAKE_MAX_RETRIES:
                    raise TimeoutError(
                        f"[{self.storage_manager_id}]: Handshake with controller {self.controller_info.id} "
                        f"({self.controller_info.ip}) failed after "
                        f"{TQ_STORAGE_HANDSHAKE_MAX_RETRIES} attempts."
                    )

            # Use shorter poll timeout for more responsive retry timing
            # while maintaining overall handshake timeout behavior
            poll_timeout = min(TQ_STORAGE_POLLER_TIMEOUT * 1000, 500)  # Max 500ms
            socks = dict(poller.poll(poll_timeout))

            if (socks.get(self.controller_handshake_socket, 0) & zmq.POLLIN) and pending_connection:
                try:
                    response_msg = ZMQMessage.deserialize(self.controller_handshake_socket.recv_multipart())

                    if response_msg.request_type == ZMQRequestType.HANDSHAKE_ACK:
                        is_connected = True
                        pending_connection = False
                        logger.debug(
                            f"[{self.storage_manager_id}]: Get handshake ACK response from "
                            f"controller id #{str(response_msg.sender_id)} to storage manager id "
                            f"#{self.storage_manager_id} successfully."
                        )
                except Exception as e:
                    logger.warning(
                        f"[{self.storage_manager_id}]: Error receiving handshake "
                        f"response from {self.controller_info.id}: {e}"
                    )

    def _send_handshake_requests(self) -> None:
        """Send handshake request to controller."""
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.HANDSHAKE,
            sender_id=self.storage_manager_id,
            body={
                "storage_manager_id": self.storage_manager_id,
                "storage_manager_type": self.__class__.__name__,
            },
        ).serialize()
        self.controller_handshake_socket.send_multipart(request_msg)
        logger.debug(
            f"[{self.storage_manager_id}]: Send handshake request from storage manager id "
            f"{self.storage_manager_id} to controller id #{self.controller_info.id} successfully."
        )

    async def notify_data_update(
        self,
        partition_id: str,
        fields: list[str],
        global_indexes: list[int],
        dtypes: dict[int, dict[str, Any]],
        shapes: dict[int, dict[str, Any]],
        custom_meta: dict[int, dict[str, Any]] = None,
    ) -> None:
        """
        Notify controller that new data is ready.

        Args:
            partition_id: Current data partition id.
            fields: Data update related fields.
            global_indexes: Data update related global_indexes.
            dtypes: Per-field dtypes for each field, in {global_index: {field: dtype}} format.
            shapes: Per-field shapes for each field, in {global_index: {field: shape}} format.
            custom_meta: Per-field custom_meta for each field, in {global_index: {field: custom_meta}} format.
        """
        # Create zmq poller for notifying data update information

        if not self.controller_info:
            logger.warning(f"No controller connected for storage manager {self.storage_manager_id}")
            return

        # Create zmq poller for notifying data update information
        poller = zmq.Poller()
        # Note: data_status_update_socket is already connected during initialization

        try:
            poller.register(self.data_status_update_socket, zmq.POLLIN)

            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.NOTIFY_DATA_UPDATE,
                sender_id=self.storage_manager_id,
                body={
                    "partition_id": partition_id,
                    "fields": fields,
                    "global_indexes": global_indexes,
                    "dtypes": dtypes,
                    "shapes": shapes,
                    "custom_meta": custom_meta,
                },
            ).serialize()

            self.data_status_update_socket.send_multipart(request_msg)
            logger.debug(
                f"[{self.storage_manager_id}]: Send data status update request "
                f"from storage manager id #{self.storage_manager_id} "
                f"to controller id #{self.controller_info.id} successfully."
            )
        except Exception as e:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR,
                sender_id=self.storage_manager_id,
                body={
                    "message": f"Failed to notify data status update information from "
                    f"storage manager id #{self.storage_manager_id}, "
                    f"detail error message: {str(e)}"
                },
            ).serialize()

            self.data_status_update_socket.send_multipart(request_msg)

        # Make sure controller successfully receives data status update information.
        response_received: bool = False
        start_time = time.time()

        while (
            not response_received  # Only one controller to get response from
            and time.time() - start_time < TQ_DATA_UPDATE_RESPONSE_TIMEOUT
        ):
            socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT * 1000))

            if self.data_status_update_socket in socks:
                response_msg = ZMQMessage.deserialize(self.data_status_update_socket.recv_multipart())

                if response_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ACK:
                    response_received = True
                    logger.debug(
                        f"[{self.storage_manager_id}]: Get data status update ACK response "
                        f"from controller id #{response_msg.sender_id} "
                        f"to storage manager id #{self.storage_manager_id} successfully."
                    )

        if not response_received:
            logger.error(
                f"[{self.storage_manager_id}]: Storage manager id #{self.storage_manager_id} "
                f"did not receive data status update ACK response from controller."
            )

    @abstractmethod
    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        raise NotImplementedError("Subclasses must implement put_data")

    @abstractmethod
    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        raise NotImplementedError("Subclasses must implement get_data")

    @abstractmethod
    async def clear_data(self, metadata: BatchMeta) -> None:
        raise NotImplementedError("Subclasses must implement clear_data")

    def close(self) -> None:
        """Close all ZMQ sockets and context to prevent resource leaks."""
        for sock in (self.controller_handshake_socket, self.data_status_update_socket):
            try:
                if sock and not sock.closed:
                    sock.close(linger=0)
            except Exception as e:
                logger.error(f"[{self.storage_manager_id}]: Error closing socket {sock}: {str(e)}")

        try:
            if self.zmq_context:
                self.zmq_context.term()
        except Exception as e:
            logger.error(f"[{self.storage_manager_id}]: Error terminating zmq_context: {str(e)}")

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        try:
            self.close()
        except Exception as e:
            logger.error(f"[{self.storage_manager_id}]: Exception during __del__: {str(e)}")


from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory  # noqa: E402


@TransferQueueStorageManagerFactory.register("KVStorageManager")
class KVStorageManager(TransferQueueStorageManager):
    """
    A storage manager that uses a key-value (KV) backend (e.g., YuanRong) to store and retrieve tensor data.
    It maps structured metadata (BatchMeta) to flat lists of keys and values for efficient KV operations.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the KVStorageManager with configuration.
        """
        client_name = config.get("client_name", None)
        if client_name is None:
            raise ValueError("Missing client_name in config")
        super().__init__(config)
        self.storage_client = StorageClientFactory.create(client_name, config)

    @staticmethod
    def _generate_keys(field_names: list[str], global_indexes: list[int]) -> list[str]:
        """
        Generate KV keys in the format 'global_index@field_name' for all sample-field pairs.
        Keys are generated in sorted order by field name first, then by global index,
        ensuring consistent ordering for batched operations.

        Args:
            field_names : list of field names.
            global_indexes : list of global indexes.
        Returns:
            list[str]: List of keys, e.g., ['0@field_a', '1@field_a', '0@field_b', ...]
        """
        return [f"{index}@{field}" for field, index in itertools.product(sorted(field_names), global_indexes)]

    @staticmethod
    def _generate_values(data: TensorDict) -> list[Tensor]:
        """
        Extract and flatten tensor values from a TensorDict in field-major order.
        Values are ordered by sorted field names, then by row (sample) order within each field.
        This matches the key order generated by `_generate_keys`.

        Args:
            data (TensorDict): Input data where keys are field names and values are tensors.
        Returns:
            list[Tensor]: Flattened list of tensors, e.g.,
                          [data[field_a][0], data[field_a][1], data[field_a][2], ..., data[field_b][0], ...]
        """
        return [row_data for field in sorted(data.keys()) for row_data in data[field]]

    @staticmethod
    def _merge_tensors_to_tensordict(metadata: BatchMeta, values: list[Tensor]) -> TensorDict:
        """
        Reconstruct a TensorDict from a list of values using metadata.
        The values list is assumed to be in the same order as keys generated by `_generate_keys`.
        According to field names and global indexes in metadata, this method can determine
        which dict key and which row this tensor belongs to. Then it reshapes the flat tensors list
        back into a structured TensorDict .

        Args:
            metadata (BatchMeta): Metadata containing global indexes and field names.
            values (list[Tensor]): List of tensors in field-major order.
        Returns:
            TensorDict: Reconstructed tensor dictionary with batch size equal to number of samples.
        """
        global_indexes = metadata.global_indexes
        field_names = sorted(metadata.field_names)
        expected_length = len(global_indexes) * len(field_names)
        if len(values) != expected_length:
            raise ValueError(f"Length of values ({len(values)}) does not match expected ({expected_length})")

        if len(values) == 0:
            return TensorDict({}, batch_size=len(global_indexes))

        grouped_data: dict[str, list[Tensor]] = {field: [] for field in field_names}

        # Group values by field_name
        # TODO: Performance optimize
        value_idx = 0
        for field in field_names:
            for _ in range(len(global_indexes)):
                grouped_data[field].append(values[value_idx])
                value_idx += 1

        # Stack or nest tensors per field
        # TODO: These codes about data merging will serve as a general function
        merged_data = {}
        for field, data_list in grouped_data.items():
            if all(isinstance(item, torch.Tensor) for item in data_list):
                try:
                    merged_data[field] = torch.stack(data_list)
                except RuntimeError:
                    try:
                        # Fallback to nested tensor if shapes are irregular
                        merged_data[field] = torch.nested.as_nested_tensor(data_list, layout=torch.jagged)
                    except Exception:
                        merged_data[field] = NonTensorStack(*data_list)
            else:
                merged_data[field] = NonTensorStack(*data_list)

        return TensorDict(merged_data, batch_size=len(global_indexes))

    @staticmethod
    def _get_shape_type_custom_meta_list(metadata: BatchMeta):
        """
        Extract the expected shape, dtype, and custom meta for each field-sample pair in metadata.
        The order matches the key/value order: sorted by field name, then by global index.

        Args:
            metadata (BatchMeta): Metadata containing sample and field information.
        Returns:
            tuple[list[torch.Size], list[torch.dtype], list[Any]]: the shape list, dtype list and
            custom meta list for each tensor to be retrieved.
        """
        shapes = []
        dtypes = []
        custom_meta_list = []
        all_custom_meta = metadata.get_all_custom_meta()
        for field_name in sorted(metadata.field_names):
            for index in range(len(metadata)):
                field = metadata.samples[index].get_field_by_name(field_name)
                shapes.append(field.shape)
                dtypes.append(field.dtype)
                global_index = metadata.global_indexes[index]
                custom_meta_list.append(all_custom_meta.get(global_index, {}).get(field_name, None))
        return shapes, dtypes, custom_meta_list

    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        """
        Store tensor data in the backend storage and notify the controller.

        Serializes the input tensors, stores them using the storage client,
        extracts per-sample dtype and shape information, and sends a notification
        to the controller that new data is available.
        """
        if not metadata.field_names:
            logger.warning("Attempted to put data, but metadata contains no fields.")
            return

        # For each field, extract dtype and shape for each sample
        num_samples = len(metadata.global_indexes)
        if num_samples == 0:
            return

        keys = self._generate_keys(data.keys(), metadata.global_indexes)
        values = self._generate_values(data)
        loop = asyncio.get_event_loop()
        custom_meta = await loop.run_in_executor(None, self.storage_client.put, keys, values)

        per_field_dtypes = {}
        per_field_shapes = {}

        # Initialize the data structure for each global index
        for global_idx in metadata.global_indexes:
            per_field_dtypes[global_idx] = {}
            per_field_shapes[global_idx] = {}

        for field_name, field_data in data.items():
            for i in range(num_samples):
                data_item = field_data[i]
                global_idx = metadata.global_indexes[i]
                per_field_dtypes[global_idx][field_name] = (
                    getattr(data_item, "dtype", None) if isinstance(data_item, Tensor) else None
                )
                per_field_shapes[global_idx][field_name] = (
                    getattr(data_item, "shape", None) if isinstance(data_item, Tensor) else None
                )

        # Prepare per-field custom_meta if available
        per_field_custom_meta = {}
        if custom_meta:
            if len(custom_meta) != len(keys):
                raise ValueError(f"Length of custom_meta ({len(custom_meta)}) does not match expected ({len(keys)})")
            # custom meta is a flat list aligned with keys/values
            # Use itertools.product to eliminate nested loops
            for global_idx in metadata.global_indexes:
                per_field_custom_meta[global_idx] = {}

            # TODO(tianyi): the order of custom meta is coupled with keys/values
            # if _generate_keys or _generate_values changes, this will break
            for (field_name, global_idx), meta_value in zip(
                itertools.product(sorted(metadata.field_names), metadata.global_indexes),
                custom_meta,
                strict=False,
            ):
                per_field_custom_meta[global_idx][field_name] = meta_value
            metadata.update_custom_meta(per_field_custom_meta)

        # Get current data partition id
        # Note: Currently we only support putting to & getting data from a single data partition simultaneously,
        # but in the future we may support putting to & getting data from multiple data partitions concurrently.
        partition_id = metadata.samples[0].partition_id
        # notify controller that new data is ready
        await self.notify_data_update(
            partition_id,
            list(data.keys()),
            metadata.global_indexes,
            per_field_dtypes,
            per_field_shapes,
            per_field_custom_meta,
        )

    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        """
        Retrieve tensor data from the backend storage.

        Fetches tensors using the provided metadata, reconstructs them with the
        correct shapes and dtypes, and merge them as a TensorDict according to metadata.
        """
        if not metadata.field_names:
            logger.warning("Attempted to get data, but metadata contains no fields.")
            return TensorDict({}, batch_size=len(metadata))
        keys = self._generate_keys(metadata.field_names, metadata.global_indexes)
        shapes, dtypes, custom_meta = self._get_shape_type_custom_meta_list(metadata)
        values = self.storage_client.get(keys=keys, shapes=shapes, dtypes=dtypes, custom_meta=custom_meta)
        return self._merge_tensors_to_tensordict(metadata, values)

    async def clear_data(self, metadata: BatchMeta) -> None:
        """Remove stored data associated with the given metadata."""
        if not metadata.field_names:
            logger.warning("Attempted to clear data, but metadata contains no fields.")
            return
        keys = self._generate_keys(metadata.field_names, metadata.global_indexes)
        self.storage_client.clear(keys=keys)
