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
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
import zmq
import zmq.asyncio
from omegaconf import DictConfig
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

LIMIT_THREADS_PER_MANAGER_IN_DRIVER = 8
LIMIT_THREADS_PER_MANAGER_IN_RAY_ACTOR = 4


class TransferQueueStorageManager(ABC):
    """Base class for storage layer. It defines the interface for data operations and
    generally provides handshake & notification capabilities."""

    def __init__(self, controller_info: ZMQServerInfo, config: DictConfig):
        self.storage_manager_id = f"TQ_STORAGE_{uuid4().hex[:8]}"
        self.config = config
        self.controller_info = controller_info

        # Handshake socket is sync (used only during initialization)
        self.controller_handshake_socket: Optional[zmq.Socket] = None

        self.zmq_context: Optional[zmq.asyncio.Context] = None
        self._connect_to_controller()

    def _connect_to_controller(self) -> None:
        """Initialize ZMQ sockets between storage unit and controller for handshake."""
        if not isinstance(self.controller_info, ZMQServerInfo):
            raise ValueError(f"controller_info should be ZMQServerInfo, but got {type(self.controller_info)}")

        try:
            # Create a synchronous context for handshake (blocking operation)
            sync_zmq_context = zmq.Context()

            # create zmq socket for handshake (sync, for initial connection)
            self.controller_handshake_socket = create_zmq_socket(
                sync_zmq_context,
                zmq.DEALER,
                identity=f"{self.storage_manager_id}-controller_handshake_socket-{uuid4().hex[:8]}".encode(),
            )

            # do handshake with controller using sync socket
            self._do_handshake_with_controller()

            # close the sync handshake socket and context after handshake
            if self.controller_handshake_socket and not self.controller_handshake_socket.closed:
                self.controller_handshake_socket.close(linger=0)
                self.controller_handshake_socket = None
            sync_zmq_context.term()

            # create async context for data status update
            self.zmq_context = zmq.asyncio.Context()

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

        assert self.controller_handshake_socket is not None, "controller_handshake_socket is not properly initialized"
        self.controller_handshake_socket.connect(self.controller_info.to_addr("handshake_socket"))
        logger.debug(
            f"[{self.storage_manager_id}]: Handshake connection from storage manager id #{self.storage_manager_id} "
            f"to controller id #{self.controller_info.id} establish successfully."
        )
        poller.register(self.controller_handshake_socket, zmq.POLLIN)

        self._send_handshake_requests()

        start_time = time.time()
        last_retry_time = time.time()

        while (
            not is_connected  # Only one controller to connect to
            and time.time() - start_time < TQ_STORAGE_HANDSHAKE_TIMEOUT
        ):
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
        assert self.controller_handshake_socket is not None, "controller_handshake_socket is not properly initialized"
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.HANDSHAKE,  # type: ignore[arg-type]
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
        custom_backend_meta: Optional[dict[int, dict[str, Any]]] = None,
    ) -> None:
        """
        Notify controller that new data is ready.

        Args:
            partition_id: Current data partition id.
            fields: Data update related fields.
            global_indexes: Data update related global_indexes.
            dtypes: Per-field dtypes for each field, in {global_index: {field: dtype}} format.
            shapes: Per-field shapes for each field, in {global_index: {field: shape}} format.
            custom_backend_meta: Per-field custom_meta for each sample, in {global_index: {field: custom_meta}} format.
        """

        if not self.controller_info:
            logger.warning(f"No controller connected for storage manager {self.storage_manager_id}")
            return

        # create dynamic socket
        identity = f"{self.storage_manager_id}-data_update-{uuid4().hex[:8]}".encode()
        sock = create_zmq_socket(self.zmq_context, zmq.DEALER, identity=identity)

        try:
            sock.connect(self.controller_info.to_addr("data_status_update_socket"))

            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.NOTIFY_DATA_UPDATE,  # type: ignore[arg-type]
                sender_id=self.storage_manager_id,
                body={
                    "partition_id": partition_id,
                    "fields": fields,
                    "global_indexes": global_indexes,
                    "dtypes": dtypes,
                    "shapes": shapes,
                    "custom_backend_meta": custom_backend_meta,
                },
            ).serialize()

            await sock.send_multipart(request_msg)
            logger.debug(
                f"[{self.storage_manager_id}]: Send data status update request "
                f"from storage manager id #{self.storage_manager_id} "
                f"to controller id #{self.controller_info.id} successfully."
            )

            response_received = False
            timeout = TQ_DATA_UPDATE_RESPONSE_TIMEOUT

            while not response_received and timeout > 0:
                try:
                    poll_interval = min(TQ_STORAGE_POLLER_TIMEOUT, timeout)
                    messages = await asyncio.wait_for(sock.recv_multipart(), timeout=poll_interval)
                    response_msg = ZMQMessage.deserialize(messages)

                    if response_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE_ACK:
                        response_received = True
                        logger.debug(
                            f"[{self.storage_manager_id}]: Get data status update ACK response "
                            f"from controller id #{response_msg.sender_id} successfully."
                        )
                except asyncio.TimeoutError:
                    timeout -= poll_interval
                except Exception as e:
                    logger.warning(f"[{self.storage_manager_id}]: Error receiving response: {e}")
                    break

            if not response_received:
                logger.error(f"[{self.storage_manager_id}]: Did not receive data status update ACK.")

        except Exception as e:
            logger.error(f"[{self.storage_manager_id}]: Error during notify_data_update: {e}")
            try:
                error_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ERROR,  # type: ignore[arg-type]
                    sender_id=self.storage_manager_id,
                    body={"message": f"Failed to notify: {str(e)}"},
                ).serialize()
                await sock.send_multipart(error_msg)
            except Exception:
                pass
        finally:
            try:
                if not sock.closed:
                    sock.close(linger=-1)
            except Exception:
                pass

    @abstractmethod
    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        """
        Put data into the storage backend.

        Args:
            data: Data to be put into the storage.
            metadata: BatchMeta of the corresponding data.
        """
        raise NotImplementedError("Subclasses must implement put_data")

    @abstractmethod
    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        """
        Get data from the storage backend.

        Args:
            metadata: BatchMeta of the data to be retrieved from the storage.

        Returns:
            TensorDict containing the data retrieved from the storage.
        """
        raise NotImplementedError("Subclasses must implement get_data")

    @abstractmethod
    async def clear_data(self, metadata: BatchMeta) -> None:
        """
        Clear data from the storage backend.

        Args:
            metadata: BatchMeta of the data to be cleared from the storage.
        """
        raise NotImplementedError("Subclasses must implement clear_data")

    def close(self) -> None:
        """Close all ZMQ sockets and context to prevent resource leaks."""
        # Close handshake socket if it exists
        if self.controller_handshake_socket:
            try:
                if not self.controller_handshake_socket.closed:
                    self.controller_handshake_socket.close(linger=0)
            except Exception as e:
                logger.error(f"[{self.storage_manager_id}]: Error closing controller_handshake_socket: {str(e)}")

        if self.zmq_context:
            self.zmq_context.term()

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

    def __init__(self, controller_info: ZMQServerInfo, config: dict[str, Any]):
        """
        Initialize the KVStorageManager with configuration.
        """
        client_name = config.get("client_name", None)
        if client_name is None:
            raise ValueError("Missing client_name in config")
        super().__init__(controller_info, config)
        self.storage_client = StorageClientFactory.create(client_name, config)
        self._multi_threads_executor: Optional[ThreadPoolExecutor] = None
        self._executor_finalizer = weakref.finalize(self, self._shutdown_executor, self._multi_threads_executor)

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
        sorted_fields = sorted(field_names)
        keys_suffixes = ["@" + f for f in sorted_fields]
        keys_prefixes = [f"{i}" for i in global_indexes]
        return [pfx + sfx for sfx, pfx in itertools.product(keys_suffixes, keys_prefixes)]

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
        results: list[Tensor] = []
        for field in sorted(data.keys()):
            field_data = data[field]
            if isinstance(field_data, Tensor) and field_data.is_nested:
                results.extend(field_data.unbind())
            else:
                results.extend(field_data)
        return results

    @staticmethod
    def _shutdown_executor(thread_executor: Optional[ThreadPoolExecutor]) -> None:
        """
        A static method to ensure no strong reference to 'self' is held within the
        finalizer's callback, enabling proper garbage collection.
        """
        if thread_executor:
            thread_executor.shutdown(wait=False)

    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazy Creating multi-thread executor for speeding up '_merge_tensors_to_tensordict'"""
        if self._multi_threads_executor is None:
            ray_context = ray.get_runtime_context()
            is_in_ray_actor_or_task = ray_context.get_actor_id() is not None or ray_context.get_task_id() is not None

            if is_in_ray_actor_or_task:
                # In ray actor:
                ray_assigned_cpus = ray_context.get_assigned_resources().get("CPU", 1)
                # num_threads must be 2 at least.
                num_threads = max(2, int(ray_assigned_cpus))
                num_threads = min(num_threads, LIMIT_THREADS_PER_MANAGER_IN_RAY_ACTOR)
            else:
                # In Driver:
                # num_threads must be 2 at least.
                num_threads = max(2, os.cpu_count() or 2)
                num_threads = min(num_threads, LIMIT_THREADS_PER_MANAGER_IN_DRIVER)

            self._num_threads = num_threads
            self._multi_threads_executor = ThreadPoolExecutor(
                max_workers=self._num_threads, thread_name_prefix="KVStorageManager"
            )

        assert self._multi_threads_executor is not None
        return self._multi_threads_executor

    def _merge_tensors_to_tensordict(self, metadata: BatchMeta, values: list[Tensor]) -> TensorDict:
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
        num_samples = len(metadata.global_indexes)
        field_names = sorted(metadata.field_names)
        num_fields = len(field_names)
        expected_length = num_samples * num_fields
        if len(values) != expected_length:
            raise ValueError(f"Length of values ({len(values)}) does not match expected ({expected_length})")

        if not values:
            return TensorDict({}, batch_size=num_samples)

        def process_field(field_idx: int):
            """
            for each field:
            1. compute chunk (Each chunk is a slice of the values list
                and All data in the chunk belong to the same field of tensordict.)
            2. if first or last value of chunk is not tensor, use NonTensorStack
            3. if the first and the last has the same shape, try torch.stack
            4. if failed, try as_nested_tensor
            5. if failed, finally use NonTensorStack

            note: we use first value and last value to Estimate the situation of the entire chunk.
            """
            field = field_names[field_idx]
            chunk = values[field_idx * num_samples : (field_idx + 1) * num_samples]
            if not chunk:
                return field, None
            first_value, last_value = chunk[0], chunk[-1]

            if not (isinstance(first_value, torch.Tensor) and isinstance(last_value, torch.Tensor)):
                return field, NonTensorStack(*chunk)

            if first_value.shape == last_value.shape:
                try:
                    return field, torch.stack(chunk)
                except (RuntimeError, TypeError):
                    pass

            try:
                return field, torch.nested.as_nested_tensor(chunk, layout=torch.jagged)
            except (RuntimeError, TypeError):
                return field, NonTensorStack(*chunk)

        executor = self._get_executor()
        use_multi_threads = num_fields > 1 and executor is not None
        if use_multi_threads:
            # Prioritize processing fields with larger tensor sizes to improve parallel efficiency
            field_sizes = []
            for i in range(num_fields):
                _first_value = values[i * num_samples]
                if isinstance(_first_value, torch.Tensor):
                    size = _first_value.nelement() * _first_value.element_size()
                else:
                    size = 0
                field_sizes.append(size)
            indexed_tasks = sorted(range(num_fields), key=lambda i: field_sizes[i], reverse=True)
            results = list(executor.map(process_field, indexed_tasks))
        else:
            results = [process_field(i) for i in range(num_fields)]

        merged_data = {field: data for field, data in results if data is not None}
        return TensorDict(merged_data, batch_size=num_samples)

    @staticmethod
    def _get_shape_type_custom_backend_meta_list(metadata: BatchMeta):
        """
        Extract the expected shape, dtype, and custom_backend_meta for each field-sample pair in metadata.
        The order matches the key/value order: sorted by field name, then by global index.

        Args:
            metadata (BatchMeta): Metadata containing sample and field information.
        Returns:
            tuple[list[torch.Size], list[torch.dtype], list[Any]]: the shape list, dtype list and
            custom meta list for each tensor to be retrieved.
        """
        shapes = []
        dtypes = []
        custom_backend_meta_list = []
        num_samples = len(metadata)

        for field_name in sorted(metadata.field_names):
            field_meta = metadata.field_schema.get(field_name, {})
            field_shape = field_meta.get("shape")
            field_dtype = field_meta.get("dtype")
            per_sample_shapes = field_meta.get("per_sample_shapes")

            for index in range(num_samples):
                if per_sample_shapes is not None:
                    shapes.append(per_sample_shapes[index])
                else:
                    shapes.append(field_shape)
                dtypes.append(field_dtype)
                custom_backend_meta_list.append(metadata._custom_backend_meta[index].get(field_name, None))
        return shapes, dtypes, custom_backend_meta_list

    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        """
        Store tensor data in the backend storage and notify the controller.
        """
        if not metadata.field_names:
            logger.warning("Attempted to put data, but metadata contains no fields.")
            return

        num_samples = len(metadata.global_indexes)
        if num_samples == 0:
            return

        keys = self._generate_keys(data.keys(), metadata.global_indexes)
        values = self._generate_values(data)
        loop = asyncio.get_event_loop()

        custom_backend_meta = await loop.run_in_executor(None, self.storage_client.put, keys, values)

        # O(F): Extract field-level schema by sampling the first item
        field_schema: dict[str, dict[str, Any]] = {}
        for field_name, field_data in data.items():
            first_item = field_data[0] if len(field_data) > 0 else None

            is_nested = isinstance(field_data, torch.Tensor) and field_data.is_nested

            is_non_tensor = not isinstance(first_item, Tensor) if first_item is not None else False

            field_meta = {
                "dtype": getattr(first_item, "dtype", type(first_item) if first_item is not None else None),
                "shape": getattr(first_item, "shape", None)
                if not is_nested and isinstance(first_item, Tensor)
                else None,
                "is_nested": is_nested,
                "is_non_tensor": is_non_tensor,
            }

            if is_nested:
                field_meta["per_sample_shapes"] = [tuple(t.shape) for t in field_data.unbind()]

            field_schema[field_name] = field_meta

        per_field_custom_backend_meta: dict[int, dict[str, Any]] = {}
        if custom_backend_meta:
            if len(custom_backend_meta) != len(keys):
                raise ValueError(
                    f"Length of custom_backend_meta ({len(custom_backend_meta)}) does not match expected ({len(keys)})"
                )
            # Build gi→position index for O(1) lookup
            gi_to_pos = {gi: i for i, gi in enumerate(metadata.global_indexes)}

            for global_idx in metadata.global_indexes:
                per_field_custom_backend_meta[global_idx] = {}

            # FIXME(tianyi): the order of custom backend meta is coupled with keys/values
            # FIXME: if put_data is called to partially update/add new fields, the current
            #       implementation will cause custom_backend_meta losses or mismatch!
            for (field_name, global_idx), meta_value in zip(
                itertools.product(sorted(metadata.field_names), metadata.global_indexes),
                custom_backend_meta,
                strict=True,
            ):
                per_field_custom_backend_meta[global_idx][field_name] = meta_value
                # Also update columnar _custom_backend_meta
                metadata._custom_backend_meta[gi_to_pos[global_idx]][field_name] = meta_value

        # Get current data partition id
        partition_id = metadata.partition_ids[0]

        # Build per-sample per-field dtypes and shapes for controller notification
        per_field_dtypes: dict[int, dict[str, Any]] = {}
        per_field_shapes: dict[int, dict[str, Any]] = {}
        for field_name, field_data in data.items():
            is_nested = isinstance(field_data, torch.Tensor) and field_data.is_nested

            if is_nested:
                # Nested tensor: each sample may have a different shape and dtype, iterate per-sample
                unbound = field_data.unbind()
                for i, global_idx in enumerate(metadata.global_indexes):
                    if global_idx not in per_field_dtypes:
                        per_field_dtypes[global_idx] = {}
                        per_field_shapes[global_idx] = {}
                    per_field_dtypes[global_idx][field_name] = unbound[i].dtype
                    per_field_shapes[global_idx][field_name] = tuple(unbound[i].shape)
            else:
                # Non-nested tensor: all samples share the same dtype and shape, bulk-fill
                first_item = field_data[0] if len(field_data) > 0 else None
                field_dtype = getattr(first_item, "dtype", type(first_item) if first_item is not None else None)
                field_shape = getattr(first_item, "shape", None) if isinstance(first_item, Tensor) else None
                for global_idx in metadata.global_indexes:
                    if global_idx not in per_field_dtypes:
                        per_field_dtypes[global_idx] = {}
                        per_field_shapes[global_idx] = {}
                    per_field_dtypes[global_idx][field_name] = field_dtype
                    per_field_shapes[global_idx][field_name] = field_shape

        await self.notify_data_update(
            partition_id,
            list(data.keys()),
            metadata.global_indexes,
            per_field_dtypes,
            per_field_shapes,
            per_field_custom_backend_meta,
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
        shapes, dtypes, custom_backend_meta = self._get_shape_type_custom_backend_meta_list(metadata)
        values = self.storage_client.get(
            keys=keys, shapes=shapes, dtypes=dtypes, custom_backend_meta=custom_backend_meta
        )
        return self._merge_tensors_to_tensordict(metadata, values)

    async def clear_data(self, metadata: BatchMeta) -> None:
        """Remove stored data associated with the given metadata."""
        if not metadata.field_names:
            logger.warning("Attempted to clear data, but metadata contains no fields.")
            return
        keys = self._generate_keys(metadata.field_names, metadata.global_indexes)
        _, _, custom_meta = self._get_shape_type_custom_backend_meta_list(metadata)
        self.storage_client.clear(keys=keys, custom_backend_meta=custom_meta)
