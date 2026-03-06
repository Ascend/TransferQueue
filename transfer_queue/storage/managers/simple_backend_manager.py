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
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    format_zmq_address,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT", 200))  # seconds


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
    def dynamic_storage_manager_socket(socket_name: str, timeout: int):
        """Decorator to auto-manage ZMQ sockets for Controller/Storage servers (create -> connect -> inject -> close).

        Args:
            socket_name (str): Port name (from server config) to use for ZMQ connection (e.g., "data_req_port").
            timeout (float): Timeout in seconds for ZMQ connection (in seconds).

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
                address = format_zmq_address(server_info.ip, server_info.ports.get(socket_name))
                identity = f"{self.storage_manager_id}_to_{server_info.id}_{uuid4().hex[:8]}".encode()
                sock = create_zmq_socket(context, zmq.DEALER, server_info.ip, identity)

                try:
                    sock.connect(address)
                    # Timeouts to avoid indefinite await on recv/send
                    sock.setsockopt(zmq.RCVTIMEO, timeout * 1000)
                    sock.setsockopt(zmq.SNDTIMEO, timeout * 1000)
                    logger.debug(
                        f"[{self.storage_manager_id}]: Connected to StorageUnit {server_info.id} at {address} "
                        f"with identity {identity.decode()}"
                    )

                    kwargs["socket"] = sock
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"[{self.storage_manager_id}]: Error in socket operation with "
                        f"StorageUnit {server_info.id} at {address}: "
                        f"{type(e).__name__}: {e}"
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

    def _group_by_hash(self, global_indexes: list[int]) -> dict[str, list[int]]:
        """Group samples by global_idx % num_su, return {storage_id: [global_indexes]}.

        Routing depends solely on global_idx, independent of batch_size, key ordering,
        or number of calls. The same global_idx always routes to the same SU across
        put/get/clear operations.

        NOTE: Dynamic SU scaling requires a data migration mechanism (not yet supported).
        """
        storage_unit_keys = list(self.storage_unit_infos.keys())
        num_units = len(storage_unit_keys)
        groups: dict[str, list[int]] = defaultdict(list)
        for global_idx in global_indexes:
            groups[storage_unit_keys[global_idx % num_units]].append(global_idx)
        return dict(groups)

    async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None:
        """
        Send data to remote StorageUnit based on metadata.

        Routes each sample to its target SU using global_idx % num_su (hash routing).
        Complexity: O(F) for schema extraction + O(S) for data distribution.

        Args:
            data: TensorDict containing the data to store.
            metadata: BatchMeta containing storage location information.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive put_data request, putting {metadata.size} samples.")

        batch_size = metadata.size

        if batch_size == 0:
            return

        field_schema = self._extract_field_schema(data)

        storage_unit_to_global_indexes = self._group_by_hash(metadata.global_indexes)
        # Build global_idx -> batch position mapping for non-contiguous slicing
        gi_to_pos = {gi: pos for pos, gi in enumerate(metadata.global_indexes)}
        tasks = [
            self._prepare_and_send_to_unit_by_positions(
                storage_id=su_id,
                positions=[gi_to_pos[gi] for gi in gi_list],
                data=data,
                metadata=metadata,
            )
            for su_id, gi_list in storage_unit_to_global_indexes.items()
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: put_data failed. "
                f"partition_id={metadata.partition_ids[0]}, "
                f"num_samples={metadata.size}, "
                f"storage_units={list(storage_unit_to_global_indexes.keys())}, "
                f"error={type(e).__name__}: {e}"
            )
            raise

        partition_id = metadata.partition_ids[0]
        dtypes_for_notify = {
            global_index: {field_name: field_meta.get("dtype") for field_name, field_meta in field_schema.items()}
            for global_index in metadata.global_indexes
        }
        shapes_for_notify = {
            global_index: {field_name: field_meta.get("shape") for field_name, field_meta in field_schema.items()}
            for global_index in metadata.global_indexes
        }
        await self.notify_data_update(
            partition_id,
            list(data.keys()),
            metadata.global_indexes,
            dtypes_for_notify,
            shapes_for_notify,
        )

    async def _prepare_and_send_to_unit_by_positions(
        self,
        storage_id,
        positions,
        data,
        metadata,
    ) -> None:
        """Slice data by non-contiguous positions and send to the specified SU."""
        global_indexes = [metadata.global_indexes[pos] for pos in positions]
        storage_data = {}
        for field_name in data.keys():
            field_data = data[field_name]
            if isinstance(field_data, torch.Tensor) and field_data.is_nested:
                unbound = field_data.unbind()
                storage_data[field_name] = [unbound[pos] for pos in positions]
            elif isinstance(field_data, NonTensorStack):
                items = field_data.tolist()
                storage_data[field_name] = NonTensorStack(*[items[pos] for pos in positions])
            elif isinstance(field_data, list):
                storage_data[field_name] = [field_data[pos] for pos in positions]
            else:
                # torch.Tensor (non-nested) and numpy arrays support fancy indexing
                storage_data[field_name] = field_data[positions]
        await self._put_to_single_storage_unit(global_indexes, storage_data, target_storage_unit=storage_id)

    @dynamic_storage_manager_socket(socket_name="put_get_socket", timeout=TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT)
    async def _put_to_single_storage_unit(
        self,
        global_indexes: list[int],
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
            body={"global_indexes": global_indexes, "data": storage_data},
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
        except zmq.error.Again as e:
            timeout_sec = TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT
            logger.error(
                f"[{self.storage_manager_id}]: ZMQ recv timeout ({timeout_sec}s) "
                f"during put to storage unit {target_storage_unit}. "
                f"The storage unit may be overloaded or crashed."
            )
            raise RuntimeError(
                f"ZMQ recv timeout ({timeout_sec}s) during put to storage unit {target_storage_unit}"
            ) from e
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: Unexpected error during put to storage unit "
                f"{target_storage_unit}: {type(e).__name__}: {e}"
            )
            raise RuntimeError(f"Error in put to storage unit {target_storage_unit}: {type(e).__name__}: {e}") from e

    async def get_data(self, metadata: BatchMeta) -> TensorDict:
        """
        Retrieve data from remote StorageUnit based on metadata.

        Routes to each SU using global_idx % num_su (hash routing).

        Args:
            metadata: BatchMeta that contains metadata for data retrieval.

        Returns:
            TensorDict containing the retrieved data.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive get_data request, getting {metadata.size} samples.")

        if metadata.size == 0:
            return TensorDict({}, batch_size=0)

        storage_unit_groups = self._group_by_hash(metadata.global_indexes)

        tasks = [
            self._get_from_single_storage_unit(global_indexes, metadata.field_names, target_storage_unit=su_id)
            for su_id, global_indexes in storage_unit_groups.items()
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: get_data failed. "
                f"partition_id={metadata.partition_ids[0]}, "
                f"num_samples={metadata.size}, "
                f"storage_units={list(storage_unit_groups.keys())}, "
                f"error={type(e).__name__}: {e}"
            )
            raise

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

    @dynamic_storage_manager_socket(socket_name="put_get_socket", timeout=TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT)
    async def _get_from_single_storage_unit(
        self,
        global_indexes: list[int],
        fields: list[str],
        target_storage_unit: str,
        socket: zmq.Socket = None,
    ):
        """Get data from a single SU by global index keys."""
        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.GET_DATA,  # type: ignore[arg-type]
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"global_indexes": global_indexes, "fields": fields},
        )
        try:
            await socket.send_multipart(request_msg.serialize())
            messages = await socket.recv_multipart()
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type == ZMQRequestType.GET_DATA_RESPONSE:
                storage_unit_data = response_msg.body["data"]
                return global_indexes, fields, storage_unit_data, messages
            else:
                raise RuntimeError(
                    f"Failed to get data from storage unit {target_storage_unit}: "
                    f"{response_msg.body.get('message', 'Unknown error')}"
                )
        except zmq.error.Again as e:
            timeout_sec = TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT
            logger.error(
                f"[{self.storage_manager_id}]: ZMQ recv timeout ({timeout_sec}s) "
                f"from storage unit {target_storage_unit}. "
                f"The storage unit may be overloaded or crashed."
            )
            raise RuntimeError(f"ZMQ recv timeout ({timeout_sec}s) from storage unit {target_storage_unit}") from e
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: Unexpected error from storage unit "
                f"{target_storage_unit}: {type(e).__name__}: {e}"
            )
            raise RuntimeError(
                f"Error getting data from storage unit {target_storage_unit}: {type(e).__name__}: {e}"
            ) from e

    async def clear_data(self, metadata: BatchMeta) -> None:
        """Clear data in remote StorageUnit.

        Routes to each SU using global_idx % num_su (hash routing).

        Args:
            metadata: BatchMeta that contains metadata for data clearing.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive clear_data request, clearing {metadata.size} samples.")

        if metadata.size == 0:
            return

        storage_unit_groups = self._group_by_hash(metadata.global_indexes)

        tasks = [
            self._clear_single_storage_unit(global_indexes, target_storage_unit=su_id)
            for su_id, global_indexes in storage_unit_groups.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[{self.storage_manager_id}]: Error in clear operation task {i}: {result}")

    @dynamic_storage_manager_socket(socket_name="put_get_socket", timeout=TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT)
    async def _clear_single_storage_unit(self, global_indexes, target_storage_unit=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA,
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"global_indexes": global_indexes},
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
