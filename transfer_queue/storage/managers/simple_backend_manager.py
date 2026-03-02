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
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import wraps
from operator import itemgetter
from typing import Any, Callable
from uuid import uuid4

import torch
import zmq
from omegaconf import DictConfig
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers.base import TransferQueueStorageManager
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory
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


@TransferQueueStorageManagerFactory.register("SimpleStorage")
class AsyncSimpleStorageManager(TransferQueueStorageManager):
    """Asynchronous storage manager that handles multiple storage units.

    This manager provides async put/get/clear operations across multiple SimpleStorageUnit
    instances using ZMQ communication and dynamic socket management.
    """

    def __init__(self, controller_info: ZMQServerInfo, config: DictConfig):
        super().__init__(controller_info, config)

        self.config = config
        server_infos: ZMQServerInfo | dict[str, ZMQServerInfo] | None = config.get("zmq_info", None)

        if server_infos is None:
            server_infos = config.get("storage_unit_infos", None)
            if server_infos is not None:
                warnings.warn(
                    "The config entry `storage_unit_infos` will be deprecated in 0.1.7, please use `zmq_info` instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )

        if server_infos is None:
            raise ValueError("AsyncSimpleStorageManager requires non-empty 'zmq_info' in config.")

        self.storage_unit_infos = self._register_servers(server_infos)

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

        Optimized version using TensorDict slicing and unified async processing.
        Complexity: O(F) for schema extraction + O(S) for data distribution.

        Args:
            data: TensorDict containing the data to store.
            metadata: BatchMeta containing storage location information.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive put_data request, putting {metadata.size} samples.")

        storage_unit_keys = list(self.storage_unit_infos.keys())
        num_units = len(storage_unit_keys)
        batch_size = metadata.size

        if batch_size == 0:
            return

        chunk_size = (batch_size + num_units - 1) // num_units

        field_schema = self._extract_field_schema(data)

        su_to_global_indexes: dict[str, list[int]] = {}
        tasks = []
        for unit_idx, storage_id in enumerate(storage_unit_keys):
            start = unit_idx * chunk_size
            end = min((unit_idx + 1) * chunk_size, batch_size)
            if start >= batch_size or start >= end:
                continue
            gi_slice = metadata.global_indexes[start:end]
            su_to_global_indexes[storage_id] = list(gi_slice)
            tasks.append(
                self._prepare_and_send_to_unit(
                    storage_id=storage_id,
                    start=start,
                    end=end,
                    data=data,
                    metadata=metadata,
                )
            )

        await asyncio.gather(*tasks)

        partition_id = metadata.partition_ids[0]
        dtypes_for_notify = {
            gi: {fname: fmeta.get("dtype") for fname, fmeta in field_schema.items()} for gi in metadata.global_indexes
        }
        shapes_for_notify = {
            gi: {fname: fmeta.get("shape") for fname, fmeta in field_schema.items()} for gi in metadata.global_indexes
        }
        backend_meta = {
            gi: {"_su_id": storage_id} for storage_id, gi_list in su_to_global_indexes.items() for gi in gi_list
        }
        await self.notify_data_update(
            partition_id,
            list(data.keys()),
            metadata.global_indexes,
            dtypes_for_notify,
            shapes_for_notify,
            custom_backend_meta=backend_meta,
        )

    async def _prepare_and_send_to_unit(
        self,
        storage_id: str,
        start: int,
        end: int,
        data: TensorDict,
        metadata: BatchMeta,
    ) -> None:
        """Prepare data slice and send to a single storage unit.

        All operations use O(1) slicing.
        """
        global_indexes = list(metadata.global_indexes[start:end])

        storage_data = {}
        for fname in data.keys():
            field_data = data[fname]
            if isinstance(field_data, torch.Tensor) and field_data.is_nested:
                # CPU NestedTensor does not support slicing; unbind first then index
                unbound = field_data.unbind()
                storage_data[fname] = unbound[start:end]
            else:
                storage_data[fname] = field_data[start:end]

        await self._put_to_single_storage_unit(global_indexes, storage_data, target_storage_unit=storage_id)

    def _extract_field_schema(self, data: TensorDict) -> dict[str, dict[str, Any]]:
        """Extract field-level schema from TensorDict. O(F) complexity."""
        field_schema: dict[str, dict[str, Any]] = {}

        for field_name in data.keys():
            field_data = data[field_name]

            # NestedTensor does not support len()/indexing; check is_nested then unbind
            is_tensor = isinstance(field_data, torch.Tensor)
            is_nested = is_tensor and field_data.is_nested

            if is_nested:
                unbound = field_data.unbind()
                first_item = unbound[0] if unbound else None
            elif is_tensor:
                first_item = field_data[0] if field_data.shape[0] > 0 else None
            else:
                first_item = field_data[0] if len(field_data) > 0 else None

            is_non_tensor = not isinstance(first_item, torch.Tensor) if first_item is not None else False

            field_meta = {
                "dtype": getattr(first_item, "dtype", type(first_item) if first_item is not None else None),
                "shape": getattr(first_item, "shape", None) if is_tensor and not is_nested else None,
                "is_nested": is_nested,
                "is_non_tensor": is_non_tensor,
            }

            if is_nested:
                field_meta["per_sample_shapes"] = [tuple(t.shape) for t in unbound]

            field_schema[field_name] = field_meta

        return field_schema

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
            request_type=ZMQRequestType.PUT_DATA,  # type: ignore[arg-type]
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

        Routes to each SU using the _su_id recorded in metadata._custom_backend_meta
        at put time. No re-computation of block allocation.

        Args:
            metadata: BatchMeta that contains metadata for data retrieval.

        Returns:
            TensorDict containing the retrieved data.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive get_data request, getting {metadata.size} samples.")

        if metadata.size == 0:
            return TensorDict({}, batch_size=0)

        groups: dict[str, list[int]] = defaultdict(list)
        for i, gi in enumerate(metadata.global_indexes):
            backend_meta = metadata._custom_backend_meta[i]
            if not backend_meta or "_su_id" not in backend_meta:
                raise RuntimeError(
                    f"get_data: missing _su_id for global_index {gi} in _custom_backend_meta. "
                    f"Make sure put_data was called before get_data."
                )
            groups[backend_meta["_su_id"]].append(gi)

        tasks = [
            self._get_from_single_storage_unit(gi_list, metadata.field_names, target_storage_unit=su_id)
            for su_id, gi_list in groups.items()
        ]
        results = await asyncio.gather(*tasks)

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
        # This detaches the received tensors from the original zero-copy buffers,
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
        self,
        gi_list: list[int],
        fields: list[str],
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        """Get data from a single SU by gi keys."""
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,  # type: ignore[arg-type]
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"local_indexes": gi_list, "fields": fields},
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
                return gi_list, fields, storage_unit_data, messages
            else:
                raise RuntimeError(
                    f"Failed to get data from storage unit {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except Exception as e:
            raise RuntimeError(f"Error getting data from storage unit {target_storage_unit}: {str(e)}") from e

    async def clear_data(self, metadata: BatchMeta) -> None:
        """Clear data in remote StorageUnit.

        Routes to each SU using the _su_id recorded in metadata._custom_backend_meta.

        Args:
            metadata: BatchMeta that contains metadata for data clearing.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive clear_data request, clearing {metadata.size} samples.")

        if metadata.size == 0:
            return

        groups: dict[str, list[int]] = defaultdict(list)
        for i, gi in enumerate(metadata.global_indexes):
            backend_meta = metadata._custom_backend_meta[i]
            if not backend_meta or "_su_id" not in backend_meta:
                raise RuntimeError(
                    f"clear_data: missing _su_id for global_index {gi} in _custom_backend_meta. "
                    f"Make sure put_data was called before clear_data."
                )
            groups[backend_meta["_su_id"]].append(gi)

        tasks = [
            self._clear_single_storage_unit(gi_list, target_storage_unit=su_id) for su_id, gi_list in groups.items()
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
