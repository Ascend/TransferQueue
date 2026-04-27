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
import os
import warnings
from collections import defaultdict
from collections.abc import Mapping
from operator import itemgetter
from typing import Any, Callable, NamedTuple

import torch
import zmq
from omegaconf import DictConfig
from tensordict import NonTensorStack, TensorDict

from transfer_queue.metadata import BatchMeta, extract_field_schema
from transfer_queue.storage.managers.base import TransferQueueStorageManager
from transfer_queue.storage.managers.factory import TransferQueueStorageManagerFactory
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    with_zmq_socket,
)

logger = get_logger(__name__)

TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT = int(os.environ.get("TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT", 200))  # seconds

# Pre-bound decorator for storage-unit socket operations.
with_storage_unit_socket = with_zmq_socket(
    "put_get_socket",
    get_identity=lambda self: self.storage_manager_id,
    get_peer=lambda self, target: self.storage_unit_infos[target],
    resolve_target=lambda args, kwargs: kwargs.get("target_storage_unit"),
    timeout=TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT,
)


class RoutingGroup(NamedTuple):
    """Routing result for a single storage unit."""

    global_indexes: list[int]  # global indexes routed to this SU
    batch_positions: list[int]  # corresponding positions in the original batch


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

    def _group_by_hash(self, global_indexes: list[int]) -> dict[str, RoutingGroup]:
        """Group samples by global_idx % num_su, return {storage_id: RoutingGroup}.

        Routing depends solely on global_idx, independent of batch_size, key ordering,
        or number of calls. The same global_idx always routes to the same SU across
        put/get/clear operations.

        NOTE: Dynamic SU scaling requires a data migration mechanism (not yet supported).
        """
        storage_unit_keys = list(self.storage_unit_infos.keys())
        num_units = len(storage_unit_keys)
        gi_lists: dict[str, list[int]] = defaultdict(list)
        pos_lists: dict[str, list[int]] = defaultdict(list)
        for pos, global_idx in enumerate(global_indexes):
            key = storage_unit_keys[global_idx % num_units]
            gi_lists[key].append(global_idx)
            pos_lists[key].append(pos)
        return {key: RoutingGroup(gi_lists[key], pos_lists[key]) for key in gi_lists}

    @staticmethod
    def _select_by_positions(field_data, positions: list[int]):
        """Slice a single field's data by non-contiguous batch positions.

        This method optimizes selection to minimize memory overhead and network fragmentation:
        - Nested tensors: Unbinds into a list of views (end-to-end zero-copy).
        - Regular tensors (step == 1): Returns a contiguous slice (end-to-end zero-copy).
        - Regular tensors (step > 1): Returns a strided view (shares storage). Note that
          downstream serialization will force a `.contiguous()` copy, but slicing is still
          faster than `index_select` and the peak memory period is reduced.
        - Regular tensors (irregular): Falls back to `index_select` to assemble a single
          contiguous tensor, preventing excessive ZMQ multipart frames.
        - NonTensorStack: tolist → select → re-wrap.
        - List: Direct index selection via `itemgetter`.
        - Numpy arrays / Others: Advanced indexing (memory copy).
        """

        n = len(positions)
        if n == 0:
            raise ValueError("No positions specified for selection.")

        # --- Handle PyTorch Tensors ---
        if isinstance(field_data, torch.Tensor):
            if field_data.is_nested:
                # Nested tensors cannot be directly sliced into a single tensor view.
                # Unbinding and selecting returns a list of individual views (zero-copy),
                # which is acceptable for nested structures.
                unbound = field_data.unbind()
                getter = itemgetter(*positions) if len(positions) > 1 else lambda seq: (seq[positions[0]],)
                selected = getter(unbound)
                return list(selected)
            else:
                # --- Smart Slicing for Regular Tensors ---
                # Goal: Return a single underlying memory view (zero-copy) to avoid both
                # memory allocation overhead and downstream ZMQ frame fragmentation.

                # Case 1: Single element selection (returns a single-row view)
                if n == 1:
                    # Single element is natively contiguous
                    return field_data[positions[0] : positions[0] + 1]

                # Case 2: Check if positions form a constant-stride sequence
                step = positions[1] - positions[0]
                is_constant_stride = True
                for i in range(2, n):
                    if positions[i] - positions[i - 1] != step:
                        is_constant_stride = False
                        break

                # If perfectly regular (e.g., [0, 2, 4]), use Python slicing to get a view
                if is_constant_stride and step > 0:
                    # Note:
                    # A strided slice (step > 1) creates a non-contiguous view.
                    # While it shares storage here, the downstream MsgpackEncoder will force
                    # a .contiguous() copy before extracting the buffer. However, this pure
                    # Python slicing is still more efficient than falling back to index_select,
                    # and it reduces memory peak period.
                    return field_data[positions[0] : positions[-1] + 1 : step]

                # Case 3: Fallback for irregular indices (Typically this will not happen!)
                # We intentionally accept a memory copy here to assemble a single contiguous
                # tensor. Returning a list of individual views for irregular indices would
                # generate excessive multipart ZMQ frames, severely degrading network performance.
                else:
                    idx_tensor = torch.tensor(positions, device=field_data.device)
                    return torch.index_select(field_data, dim=0, index=idx_tensor)

        # --- Handle Non-Tensor Types ---
        elif isinstance(field_data, NonTensorStack):
            items = field_data.tolist()
            getter = itemgetter(*positions) if len(positions) > 1 else lambda seq: (seq[positions[0]],)
            selected = getter(items)
            return NonTensorStack(*selected)
        elif isinstance(field_data, list):
            getter = itemgetter(*positions) if len(positions) > 1 else lambda seq: (seq[positions[0]],)
            selected = getter(field_data)
            return list(selected)
        else:
            return field_data[positions]

    async def put_data(
        self, data: TensorDict, metadata: BatchMeta, data_parser: Callable[[Any], Any] | None = None
    ) -> None:
        """
        Send data to remote StorageUnit based on metadata.

        Routes each sample to its target SU using global_idx % num_su (hash routing).
        Complexity: O(F) for schema extraction + O(S) for data distribution.

        Args:
            data: TensorDict containing the data to store.
            metadata: BatchMeta containing storage location information.
            data_parser: Optional callable to parse reference data (e.g., URLs) into real
                         content. The input is a plain dict (not TensorDict) mapping
                         field_name -> batched values. For a regular tensor column the
                         value is a batched tensor; for nested tensors (jagged or strided)
                         and NonTensorStack columns the values are extracted into a list.
                         It must modify values in-place based on the original keys; do not
                         add or remove keys. The number of elements per column must also
                         remain unchanged. Do not change the inner order of values within
                         each column. Executed distributedly on each SimpleStorageUnit.
        """

        logger.debug(f"[{self.storage_manager_id}]: receive put_data request, putting {metadata.size} samples.")

        batch_size = metadata.size

        if batch_size == 0:
            return

        field_schema = extract_field_schema(data)

        routing = self._group_by_hash(metadata.global_indexes)
        tasks = [
            self._put_to_single_storage_unit(
                group.global_indexes,
                {f: self._select_by_positions(data[f], group.batch_positions) for f in data.keys()},
                target_storage_unit=su_id,
                data_parser=data_parser,
            )
            for su_id, group in routing.items()
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: put_data failed. "
                f"partition_id={metadata.partition_ids[0]}, "
                f"num_samples={metadata.size}, "
                f"storage_units={list(routing.keys())}, "
                f"error={type(e).__name__}: {e}"
            )
            raise

        partition_id = metadata.partition_ids[0]
        await self.notify_data_update(
            partition_id,
            metadata.global_indexes,
            field_schema,
        )

    @with_storage_unit_socket
    async def _put_to_single_storage_unit(
        self,
        global_indexes: list[int],
        storage_data: dict[str, Any],
        target_storage_unit: str,
        data_parser: Callable[[Any], Any] | None = None,
        socket: zmq.Socket = None,
    ):
        """
        Send data to a specific storage unit.
        """

        request_msg = ZMQMessage.create(
            request_type=ZMQRequestType.PUT_DATA,  # type: ignore[arg-type]
            sender_id=self.storage_manager_id,
            receiver_id=target_storage_unit,
            body={"global_indexes": global_indexes, "data": storage_data, "data_parser": data_parser},
        )

        try:
            data = request_msg.serialize()
            await socket.send_multipart(data, copy=False)
            messages = await socket.recv_multipart(copy=False)
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

    @staticmethod
    def _pack_field_values(values: list) -> torch.Tensor | NonTensorStack:
        """
        Pack a list of per-sample values into a batched container.

        For pure tensor lists (no None), this performs a memory copy via stacking
        or nested tensor creation. Mixed types, non-tensor values, or lists
        containing None placeholders are grouped into a ``NonTensorStack``.

        Args:
            values: List of per-sample values to pack. May contain None for
                unfilled batch positions.

        Returns:
            A stacked ``torch.Tensor`` (or nested tensor) when all values are
            tensors, otherwise a ``NonTensorStack``.

        Raises:
            ValueError: If *values* is empty.
        """
        if not values:
            raise ValueError("_pack_field_values received empty values list; caller should filter empty batches")
        non_none = [v for v in values if v is not None]
        if non_none and all(isinstance(v, torch.Tensor) for v in non_none):
            if len(non_none) == len(values):
                # Pure tensor list — try stacking / nested tensor
                if all(v.shape == values[0].shape for v in values):
                    return torch.stack(values)
                try:
                    return torch.nested.as_nested_tensor(values, layout=torch.jagged)
                except (RuntimeError, TypeError) as e:
                    logger.warning(
                        f"Failed to pack nested tensor with jagged layout. "
                        f"Falling back to strided layout. Detailed error: {e}"
                    )
                    return torch.nested.as_nested_tensor(values, layout=torch.strided)
            # Mixed tensor + None — cannot stack, fall through to NonTensorStack
        return NonTensorStack(*values)

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

        routing = self._group_by_hash(metadata.global_indexes)

        tasks = [
            self._get_from_single_storage_unit(group.global_indexes, metadata.field_names, target_storage_unit=su_id)
            for su_id, group in routing.items()
        ]
        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(
                f"[{self.storage_manager_id}]: get_data failed. "
                f"partition_id={metadata.partition_ids[0]}, "
                f"num_samples={metadata.size}, "
                f"storage_units={list(routing.keys())}, "
                f"error={type(e).__name__}: {e}"
            )
            raise

        # Scatter results directly to batch positions — no intermediate per-sample dict
        n = len(metadata.global_indexes)
        ordered_data: dict[str, list] = {field: [None] * n for field in metadata.field_names}

        for (su_id, group), (fields, su_data) in zip(routing.items(), results, strict=True):
            for field in fields:
                for i, pos in enumerate(group.batch_positions):
                    ordered_data[field][pos] = su_data[field][i]

        tensor_data = {field: self._pack_field_values(v) for field, v in ordered_data.items()}

        return TensorDict(tensor_data, batch_size=len(metadata))

    @with_storage_unit_socket
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
            messages = await socket.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(messages)

            if response_msg.request_type == ZMQRequestType.GET_DATA_RESPONSE:
                storage_unit_data = response_msg.body["data"]
                return fields, storage_unit_data
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

        routing = self._group_by_hash(metadata.global_indexes)

        tasks = [
            self._clear_single_storage_unit(group.global_indexes, target_storage_unit=su_id)
            for su_id, group in routing.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"[{self.storage_manager_id}]: Error in clear operation task {i}: {result}")

    @with_storage_unit_socket
    async def _clear_single_storage_unit(self, global_indexes, target_storage_unit=None, socket=None):
        try:
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA,
                sender_id=self.storage_manager_id,
                receiver_id=target_storage_unit,
                body={"global_indexes": global_indexes},
            )

            await socket.send_multipart(request_msg.serialize())
            messages = await socket.recv_multipart(copy=False)
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
