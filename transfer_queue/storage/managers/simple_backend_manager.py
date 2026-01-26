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
import logging
import os
from collections.abc import Mapping
from functools import wraps
from operator import itemgetter
from typing import Any, Callable
from uuid import uuid4

import torch
import zmq
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers.base import TransferQueueStorageManager
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory
from transfer_queue.storage.simple_backend import StorageMetaGroup
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo, create_zmq_socket

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

TQ_SIMPLE_STORAGE_MANAGER_RECV_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_MANAGER_RECV_TIMEOUT", 200))  # seconds
TQ_SIMPLE_STORAGE_MANAGER_SEND_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_MANAGER_SEND_TIMEOUT", 200))  # seconds


@TransferQueueStorageManagerFactory.register("AsyncSimpleStorageManager")
class AsyncSimpleStorageManager(TransferQueueStorageManager):
    """Asynchronous storage manager that handles multiple storage units.

    This manager provides async put/get/clear operations across multiple SimpleStorageUnit
    instances using ZMQ communication and dynamic socket management.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        self.config = config
        server_infos: ZMQServerInfo | dict[str, ZMQServerInfo] | None = config.get("storage_unit_infos", None)

        if server_infos is None:
            raise ValueError("AsyncSimpleStorageManager requires non-empty 'storage_unit_infos' in config.")

        self.storage_unit_infos = self._register_servers(server_infos)
        self._build_storage_mapping_functions()

    def _register_servers(self, server_infos: "ZMQServerInfo | dict[Any, ZMQServerInfo]"):
        """Register and validate server information.

        Args:
            server_infos: ZMQServerInfo | dict[Any, ZMQServerInfo])
                ZMQServerInfo or dict of server infos to register.

        Returns:
            Dictionary with server IDs as keys and ZMQServerInfo objects as values.

        Raises:
            ValueError: If server_infos format is invalid.
        """
        server_infos_transform = {}

        if isinstance(server_infos, ZMQServerInfo):
            server_infos_transform[server_infos.id] = server_infos
        elif isinstance(server_infos, Mapping):
            for k, v in server_infos.items():
                if not isinstance(v, ZMQServerInfo):
                    raise ValueError(f"Invalid server info for key {k}: {v}")
                server_infos_transform[v.id] = v
        else:
            raise ValueError(f"Invalid server infos: {server_infos}")

        return server_infos_transform

    def _build_storage_mapping_functions(self):
        """Build mapping functions for global index to storage unit and local index.

        Creates round-robin mapping functions to distribute data across storage units.
        """
        self.global_index_storage_unit_mapping = lambda x: list(self.storage_unit_infos.keys())[
            x % len(self.storage_unit_infos)
        ]
        self.global_index_local_index_mapping = lambda x: x // len(self.storage_unit_infos)

    # TODO (TQStorage): Provide a general dynamic socket function for both Client & Storage @huazhong.
    @staticmethod
    def dynamic_storage_manager_socket(socket_name: str):
        """Decorator to auto-manage ZMQ sockets for Controller/Storage servers (create -> connect -> inject -> close).

        Args:
            socket_name (str): Port name (from server config) to use for ZMQ connection (e.g., "data_req_port").

        Decorated Function Rules:
            1. Must be an async class method (needs `self`).
            2. `self` requires:
            - `storage_unit_infos: storage unit infos (ZMQServerInfo | dict[Any, ZMQServerInfo]).
            3. Specify target server via:
            - `target_storage_unit` arg.
            4. Receives ZMQ socket via `socket` keyword arg (injected by decorator).
        """

        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                server_key = kwargs.get("target_storage_unit")
                if server_key is None:
                    for arg in args:
                        if isinstance(arg, str) and arg in self.storage_unit_infos.keys():
                            server_key = arg
                            break

                server_info = self.storage_unit_infos.get(server_key)

                if not server_info:
                    raise RuntimeError(f"Server {server_key} not found in registered servers")

                context = zmq.asyncio.Context()
                address = f"tcp://{server_info.ip}:{server_info.ports.get(socket_name)}"
                identity = f"{self.storage_manager_id}_to_{server_info.id}_{uuid4().hex[:8]}".encode()
                sock = create_zmq_socket(context, zmq.DEALER, identity=identity)

                try:
                    sock.connect(address)
                    # Timeouts to avoid indefinite await on recv/send
                    sock.setsockopt(zmq.RCVTIMEO, TQ_SIMPLE_STORAGE_MANAGER_RECV_TIMEOUT * 1000)
                    sock.setsockopt(zmq.SNDTIMEO, TQ_SIMPLE_STORAGE_MANAGER_SEND_TIMEOUT * 1000)
                    logger.debug(
                        f"[{self.storage_manager_id}]: Connected to StorageUnit {server_info.id} at {address} "
                        f"with identity {identity.decode()}"
                    )

                    kwargs["socket"] = sock
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"[{self.storage_manager_id}]: Error in socket operation with StorageUnit {server_info.id}: {e}"
                    )
                    raise
                finally:
                    try:
                        if not sock.closed:
                            sock.close(linger=-1)
                    except Exception as e:
                        logger.warning(
                            f"[{self.storage_manager_id}]: Error closing socket to StorageUnit {server_info.id}: {e}"
                        )

                    context.term()

            return wrapper

        return decorator

    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        """
        Send data to remote StorageUnit based on metadata.

        Args:
            data: TensorDict containing the data to store.
            metadata: BatchMeta containing storage location information.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive put_data request, putting {metadata.size} samples.")

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(
            metadata, self.global_index_storage_unit_mapping, self.global_index_local_index_mapping
        )

        # send data to each storage unit
        tasks = [
            self._put_to_single_storage_unit(
                meta_group.get_local_indexes(), _filter_storage_data(meta_group, data), target_storage_unit=storage_id
            )
            for storage_id, meta_group in storage_meta_groups.items()
        ]
        await asyncio.gather(*tasks)

        # Gather per-field dtype and shape information for each field
        # global_indexes, local_indexes, and field_data correspond one-to-one
        per_field_dtypes: dict[int, dict[str, Any]] = {}
        per_field_shapes: dict[int, dict[str, Any]] = {}

        # Initialize the data structure for each global index
        for global_idx in metadata.global_indexes:
            per_field_dtypes[global_idx] = {}
            per_field_shapes[global_idx] = {}

        # For each field, extract dtype and shape for each sample
        for field in data.keys():
            for i, data_item in enumerate(data[field]):
                global_idx = metadata.global_indexes[i]
                per_field_dtypes[global_idx][field] = data_item.dtype if hasattr(data_item, "dtype") else None
                per_field_shapes[global_idx][field] = data_item.shape if hasattr(data_item, "shape") else None

        # Get current data partition id
        # Note: Currently we only support putting to & getting data from a single data partition simultaneously,
        # but in the future we may support putting to & getting data from multiple data partitions concurrently.
        partition_id = metadata.samples[0].partition_id

        # notify controller that new data is ready
        await self.notify_data_update(
            partition_id, list(data.keys()), metadata.global_indexes, per_field_dtypes, per_field_shapes
        )

    @dynamic_storage_manager_socket(socket_name="put_get_socket")
    async def _put_to_single_storage_unit(
        self,
        local_indexes: list[int],
        storage_data: dict[str, Any],
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        """
        Send data to a specific storage unit.
        """

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"local_indexes": local_indexes, "data": storage_data},
        )

        try:
            data = request_msg.serialize()
            await socket.send_multipart(data, copy=False)
            messages = await socket.recv_multipart()
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type != ZMQRequestType.PUT_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to put data to storage unit {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error in put to storage unit {target_storage_unit}: {str(e)}") from e

    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        """
        Retrieve data from remote StorageUnit based on metadata.

        Args:
            metadata: BatchMeta that contains metadata for data retrieval.

        Returns:
            TensorDict containing the retrieved data.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive get_data request, getting {metadata.size} samples.")

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(
            metadata, self.global_index_storage_unit_mapping, self.global_index_local_index_mapping
        )

        # retrive data
        tasks = [
            self._get_from_single_storage_unit(meta_group, target_storage_unit=storage_id)
            for storage_id, meta_group in storage_meta_groups.items()
        ]

        results = await asyncio.gather(*tasks)

        # post-process data segments to generate a batch of data
        merged_data: dict[int, dict[str, torch.Tensor]] = {}
        for global_indexes, fields, data_from_single_storage_unit, messages in results:
            field_getter = itemgetter(*fields)
            field_values = field_getter(data_from_single_storage_unit)

            if len(fields) == 1:
                extracted_data = {fields[0]: field_values}
            else:
                extracted_data = dict(zip(fields, field_values, strict=False))

            for idx, global_idx in enumerate(global_indexes):
                if global_idx not in merged_data:
                    merged_data[global_idx] = {}
                merged_data[global_idx].update({field: extracted_data[field][idx] for field in fields})

        ordered_data: dict[str, list[torch.Tensor]] = {}
        for field in metadata.field_names:
            ordered_data[field] = [merged_data[global_idx][field] for global_idx in metadata.global_indexes]

        # In the final packing stage we intentionally perform a memory copy through torch.stack and as_nested_tensor.
        # This detaches the received tensors from the original zero‑copy buffers,
        # gives them their own lifetime, and ensures the resulting tensors are writable.
        tensor_data = {
            field: (
                torch.stack(v)
                if v
                and all(isinstance(item, torch.Tensor) for item in v)
                and all(item.shape == v[0].shape for item in v)
                else (
                    torch.nested.as_nested_tensor(v, layout=torch.jagged)
                    if v and all(isinstance(item, torch.Tensor) for item in v)
                    else NonTensorStack(*v)
                )
            )
            for field, v in ordered_data.items()
        }

        return TensorDict(tensor_data, batch_size=len(metadata))

    @dynamic_storage_manager_socket(socket_name="put_get_socket")
    async def _get_from_single_storage_unit(
        self, storage_meta_group: StorageMetaGroup, target_storage_unit: str, socket: zmq.Socket = None
    ):
        global_indexes = storage_meta_group.get_global_indexes()
        local_indexes = storage_meta_group.get_local_indexes()
        fields = storage_meta_group.get_field_names()

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"local_indexes": local_indexes, "fields": fields},
        )
        try:
            await socket.send_multipart(request_msg.serialize())
            messages = await socket.recv_multipart()
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type == ZMQRequestType.GET_DATA_RESPONSE:
                # Return data and index information from this storage unit
                # We need to return messages to get_data() since the zero-copy deserialization directly points to the
                # memory of messages object.
                storage_unit_data = response_msg.body["data"]
                return global_indexes, fields, storage_unit_data, messages
            else:
                raise RuntimeError(
                    f"Failed to get data from storage unit {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error getting data from storage unit {target_storage_unit}: {str(e)}") from e

    async def clear_data(self, metadata: BatchMeta) -> None:
        """Clear data in remote StorageUnit.

        Args:
            metadata: BatchMeta that contains metadata for data clearing.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive clear_data request, clearing {metadata.size} samples.")

        # group samples by storage unit
        storage_meta_groups = build_storage_meta_groups(
            metadata, self.global_index_storage_unit_mapping, self.global_index_local_index_mapping
        )

        # clear data
        tasks = [
            self._clear_single_storage_unit(meta_group.get_local_indexes(), target_storage_unit=storage_id)
            for storage_id, meta_group in storage_meta_groups.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[{self.storage_manager_id}]: Error in clear operation task {i}: {result}")

    @dynamic_storage_manager_socket(socket_name="put_get_socket")
    async def _clear_single_storage_unit(self, local_indexes, target_storage_unit=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA,
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"local_indexes": local_indexes},
            )

            await socket.send_multipart(request_msg.serialize())
            messages = await socket.recv_multipart()
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type != ZMQRequestType.CLEAR_DATA_RESPONSE:
                raise RuntimeError(
                    f"Failed to clear storage {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"[{self.storage_manager_id}]: Error clearing storage unit {target_storage_unit}: {str(e)}")
            raise

    def get_zmq_server_info(self) -> dict[str, ZMQServerInfo]:
        """Get ZMQ server information for all storage units.

        Returns:
            Dictionary mapping storage unit IDs to their ZMQServerInfo.
        """
        return self.storage_unit_infos

    def close(self) -> None:
        """Close all ZMQ sockets and context to prevent resource leaks."""
        super().close()


def _filter_storage_data(storage_meta_group: StorageMetaGroup, data: TensorDict) -> dict[str, Any]:
    """Filter batch-aligned data from a TensorDict using batch indexes from a StorageMetaGroup.
    This helper extracts a subset of items from each field in ``data`` according to the
    batch indexes stored in ``storage_meta_group``. The same indexes are applied to every
    field in the input ``TensorDict`` so that the returned samples remain aligned across
    fields.

    Args:
        storage_meta_group: A :class:`StorageMetaGroup` instance that provides
            a sequence of batch indexes via :meth:`get_batch_indexes`. Each index
            refers to a position along the batch dimension of the tensors stored
            in ``data``.
        data: A :class:`tensordict.TensorDict` containing batched data fields. All
            fields are expected to be indexable by the batch indexes returned by
            ``storage_meta_group.get_batch_indexes()``.
    Returns:
        dict[str, Any]: A dictionary mapping each field name in ``data`` to a list
            of items selected at the requested batch indexes. The order of items in
            each list matches the order of ``storage_meta_group.get_batch_indexes()``.
    """

    # We use dict here instead of TensorDict to avoid unnecessary TensorDict overhead
    results: dict[str, Any] = {}
    batch_indexes = storage_meta_group.get_batch_indexes()

    if not batch_indexes:
        return results

    for fname in data.keys():
        result = itemgetter(*batch_indexes)(data[fname])
        if not isinstance(result, tuple):
            result = (result,)
        results[fname] = list(result)

    return results


def build_storage_meta_groups(
    batch_meta: BatchMeta,
    global_index_storage_unit_mapping: Callable,
    global_index_local_index_mapping: Callable,
) -> dict[str, StorageMetaGroup]:
    """Build storage meta groups from batch metadata for distributed storage.

    This function is the starting point of the data distribution workflow. It analyzes
    BatchMeta containing SampleMeta objects (originating from client requests) and
    groups them by target storage unit based on their global_index.

    Key Data Flow:
    1. BatchMeta contains SampleMeta objects with batch_index (original TensorDict position)
    2. Each SampleMeta is assigned to a storage unit using global_index mapping
    3. Local storage positions are calculated for each sample
    4. Results in StorageMetaGroup objects ready for transfer operations

    Args:
        batch_meta: BatchMeta containing SampleMeta objects from client request.
            Each SampleMeta has:
            - batch_index: Position in original TensorDict (0-based)
            - global_index: Global unique identifier across all storage
        global_index_storage_unit_mapping: Function to map global_index to storage_unit_id.
            Example: lambda x: storage_unit_ids[x % num_storage_units] (round-robin distribution)
        global_index_local_index_mapping: Function to map global_index to local_index.
            Example: lambda x: x // num_storage_units (local position within storage unit)

    Returns:
        Dictionary mapping storage_unit_id to StorageMetaGroup, where each group contains:
        - storage_id: Target storage unit identifier
        - sample_metas: List of SampleMeta objects assigned to this unit
        - local_indexes: List of storage positions for each sample

    Example:
        >>> # Input: BatchMeta with samples at global_indexes [10, 11, 12]
        >>> # 3 storage units available: storage_0, storage_1, storage_2
        >>> batch_meta = BatchMeta(samples=[
        ...     SampleMeta(batch_index=0, global_index=10),  # Original position 0
        ...     SampleMeta(batch_index=1, global_index=11),  # Original position 1
        ...     SampleMeta(batch_index=2, global_index=12)   # Original position 2
        ... ])
        >>> groups = build_storage_meta_groups(
        ...     batch_meta,
        ...     lambda x: f"storage_{x % 3}",  # 10->storage_1, 11->storage_2, 12->storage_0
        ...     lambda x: x // 3               # 10->3, 11->3, 12->4
        ... )
        >>> groups["storage_1"].sample_metas[0].batch_index  # 0 - original TensorDict position
        >>> groups["storage_1"].sample_metas[0].local_index  # 3 - storage position

    Note:
        This function preserves the crucial batch_index information that links each
        SampleMeta back to its original position in the client's TensorDict.
        This batch_index is later used by _add_field_data() to extract
        the correct data items for storage.
    """
    storage_meta_groups: dict[str, StorageMetaGroup] = {}

    for sample in batch_meta.samples:
        storage_id = global_index_storage_unit_mapping(sample.global_index)
        local_index = global_index_local_index_mapping(sample.global_index)
        if storage_id not in storage_meta_groups:
            storage_meta_groups[storage_id] = StorageMetaGroup(storage_id=storage_id)

        # Use add_sample_meta to store SampleMeta references directly
        storage_meta_groups[storage_id].add_sample_meta(sample, local_index)

    return storage_meta_groups
