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

import copy
import os
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import groupby
from operator import itemgetter
from threading import Thread
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import numpy as np
import ray
import torch
import zmq
from omegaconf import DictConfig
from torch import Tensor

from transfer_queue.metadata import (
    BatchMeta,
)
from transfer_queue.sampler import BaseSampler, SequentialSampler
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    format_zmq_address,
    get_free_port,
    get_node_ip_address,
)

if TYPE_CHECKING:
    from transfer_queue.metrics import TQMetricsExporter

logger = get_logger(__name__)

TQ_CONTROLLER_GET_METADATA_TIMEOUT = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_TIMEOUT", 1))
TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL", 5))
# Above libzmq's default of 1023: the metrics exporter borrows this context to query
# storage units, so the budget covers a fleet rather than the two ROUTERs bound here.
TQ_CONTROLLER_ZMQ_MAX_SOCKETS = int(os.environ.get("TQ_CONTROLLER_ZMQ_MAX_SOCKETS", 4096))

# Holds payloads that keys of other partitions read through kv_link, e.g. a prompt video
# shared by its n responses. Its rows are balanced over storage units like any partition's.
PUBLIC_PARTITION = "__public__"

# Sample pre-allocation for StreamingDataLoader compatibility.
# By pre-allocating sample indices (typically global_batch_size), consumers can accurately
# determine consumption status even before producers have generated the samples.


class PartitionIndexManager:
    """
    Manages the mapping relationship between partitions and global indexes,
    responsible for index allocation and reuse.
    """

    def __init__(self):
        # Records the set of global_indexes used by each partition
        self.partition_to_indexes = defaultdict(set)

        # Global index counter for allocating new indexes
        self.global_index_counter = 0

        # Track all active indexes
        self.allocated_indexes = set()

        # SimpleStorage stores global_idx on unit global_idx % num_storage_units, so picking the
        # index picks the unit. Reusable indexes are kept FIFO per unit.
        self.set_num_storage_units(1, [])

    @property
    def reusable_indexes(self) -> list[int]:
        return [idx for pool in self.reusable_by_unit for idx in pool]

    def set_num_storage_units(self, num_storage_units: int, reusable_indexes: list[int] | None = None) -> None:
        """Regroup the reusable indexes and per-partition unit loads for ``num_storage_units`` units."""
        reusable = self.reusable_indexes if reusable_indexes is None else reusable_indexes
        self.num_storage_units = num_storage_units
        self.reusable_by_unit: list[deque[int]] = [deque() for _ in range(num_storage_units)]
        for idx in reusable:
            self.reusable_by_unit[idx % num_storage_units].append(idx)
        # partition_id -> rows of the partition on each unit
        self.unit_loads: dict[str, list[int]] = {}
        for partition_id, indexes in self.partition_to_indexes.items():
            loads = self.unit_loads[partition_id] = [0] * num_storage_units
            for idx in indexes:
                loads[idx % num_storage_units] += 1

    def allocate_indexes(self, partition_id, count=1) -> list[int]:
        """
        Allocate global_indexes for the specified partition, each on the unit where the
        partition has the fewest rows, so every partition stays balanced across units.
        Prioritizes obtaining from reusable pool, allocates new indexes when insufficient.

        Args:
            partition_id: Partition ID
            count: Number of indexes needed

        Returns:
            list: List of allocated global_indexes
        """
        if count <= 0:
            raise ValueError(f"Number of indexes needed must be larger than 0, but got {count}")
        units = self.num_storage_units
        loads = self.unit_loads.setdefault(partition_id, [0] * units)
        indexes = []
        for _ in range(count):
            # Among the least-loaded units, prefer reusing an index, then the counter's next one.
            unit = min(
                range(units),
                key=lambda u: (loads[u], not self.reusable_by_unit[u], u != self.global_index_counter % units),
            )
            if self.reusable_by_unit[unit]:
                idx = self.reusable_by_unit[unit].popleft()
            else:
                # Indexes skipped to reach this unit wait in their own unit's pool.
                while self.global_index_counter % units != unit:
                    self.reusable_by_unit[self.global_index_counter % units].append(self.global_index_counter)
                    self.global_index_counter += 1
                idx = self.global_index_counter
                self.global_index_counter += 1
            loads[unit] += 1
            indexes.append(idx)

        self.allocated_indexes.update(indexes)
        # Record partition-index relationship
        self.partition_to_indexes[partition_id].update(indexes)

        return indexes

    def _reuse(self, partition_id, indexes) -> None:
        loads = self.unit_loads.get(partition_id)
        for idx in indexes:
            self.reusable_by_unit[idx % self.num_storage_units].append(idx)
            if loads is not None:
                loads[idx % self.num_storage_units] -= 1

    def release_partition(self, partition_id) -> list[int]:
        """
        Release all global_indexes of the specified partition, adding them to reusable pool.

        Args:
            partition_id: Partition ID

        Returns:
            list: List of released global_indexes
        """
        if partition_id in self.partition_to_indexes:
            indexes = self.partition_to_indexes.pop(partition_id)

            # Add released indexes to reusable pool
            self._reuse(partition_id, indexes)
            self.unit_loads.pop(partition_id, None)

            # Remove these indexes from allocated_indexes
            for idx in indexes:
                self.allocated_indexes.discard(idx)

            return list(indexes)
        return []

    def release_indexes(self, partition_id: str, indexes_to_release: list[int]):
        """
        Release specific global_indexes for a partition, adding them to reusable pool.

        Args:
            partition_id: Partition ID
            indexes_to_release: List of specific indexes to release
        """
        if partition_id not in self.partition_to_indexes:
            return []

        partition_indexes = self.partition_to_indexes[partition_id]

        if not set(indexes_to_release).issubset(partition_indexes):
            raise ValueError("Some indexes to release do not belong to the specified partition.")

        partition_indexes.difference_update(indexes_to_release)
        self._reuse(partition_id, indexes_to_release)
        self.allocated_indexes.difference_update(indexes_to_release)

        # If partition has no more indexes, remove it from the mapping
        if not partition_indexes:
            self.partition_to_indexes.pop(partition_id, None)
            self.unit_loads.pop(partition_id, None)

    def get_indexes_for_partition(self, partition_id) -> list[int]:
        """
        Get all global_indexes for the specified partition.

        Args:
            partition_id: Partition ID

        Returns:
            list: List of global_indexes for this partition
        """
        return list(self.partition_to_indexes.get(partition_id, set()).copy())


@dataclass
class FieldMeta:
    """
    Single source of truth for one field's metadata in a partition.

    Field-level attributes (dtype/shape/is_nested/is_non_tensor) are shared across all samples, O(1) storage.
    Sample-level attributes (per_sample_shapes) are only needed for nested tensors,
    indexed by global_idx, O(B_nested) storage.
    """

    global_indexes: set[int] = field(default_factory=set)
    dtype: Any | None = None
    shape: tuple | None = None  # None when is_nested=True
    is_nested: bool | None = None
    is_non_tensor: bool | None = None

    per_sample_shapes: dict[int, tuple] = field(default_factory=dict)  # {global_idx: shape}

    # TODO: FieldMeta needs to be refactored to prevent these complicated and fragile logics
    def update(self, incoming: dict[str, Any], incoming_global_indexes: list[int]) -> None:
        """Update this field's metadata from an incoming schema dict.

        Encapsulates dtype consistency check, shape conflict detection,
        and automatic is_nested inference.

        Args:
            incoming: Schema dict with optional keys:
                      global_indexes, dtype, shape, is_nested, is_non_tensor, per_sample_shapes
            incoming_global_indexes: global indexes of the input meta
        Raises:
            ValueError: If incoming dtype conflicts with existing dtype.
        """
        # dtype consistency check
        new_dtype = incoming.get("dtype")
        if new_dtype is not None:
            if self.dtype is None:
                self.dtype = new_dtype
            elif self.dtype != new_dtype:
                raise ValueError(
                    f"dtype mismatch: existing={self.dtype}, incoming={new_dtype}. "
                    f"All batches for the same field must have the same dtype."
                )

        new_is_nested = incoming.get("is_nested")
        new_is_non_tensor = incoming.get("is_non_tensor")

        if new_is_nested:
            new_per_sample_shapes = incoming.get("per_sample_shapes", None)
            if new_per_sample_shapes is None:
                raise ValueError("Receiving a nested field without 'per_sample_shapes'!")
            if self.is_nested is not None and not self.is_nested:
                # new input is nested, but original is regular tensor.
                # We need to write old shape into per_sample_shapes
                assert self.shape is not None
                for gi in self.global_indexes:
                    self.per_sample_shapes[gi] = self.shape
                self.is_nested = True
                self.shape = None

            # Update newly provided per_sample_shapes
            self.per_sample_shapes.update(new_per_sample_shapes)

        else:
            if not new_is_non_tensor:
                # newly input is regular tensor
                new_shape = incoming.get("shape", None)
                if new_shape is None:
                    raise ValueError("Receiving a regular tensor without 'shape'!")
                if self.is_nested:
                    # we need to update incoming shape into per_sample_shapes
                    for gi in incoming_global_indexes:
                        self.per_sample_shapes[gi] = new_shape
                else:
                    if self.is_non_tensor is not None and not self.is_non_tensor:
                        # original data is also regular tensor
                        assert self.shape is not None
                        if self.shape != new_shape:
                            for gi in self.global_indexes:
                                self.per_sample_shapes[gi] = self.shape
                            for gi in incoming_global_indexes:
                                self.per_sample_shapes[gi] = new_shape

                            self.shape = None
                            self.is_nested = True

        self.global_indexes.update(incoming_global_indexes)

    def remove_samples(self, indexes: list[int]):
        """Remove sample-level data for the given indexes."""
        for idx in indexes:
            self.per_sample_shapes.pop(idx, None)
            self.global_indexes.discard(idx)

        # After removing samples, check if we can update is_nested and shape
        if len(self.global_indexes) == 0:
            # If no samples remain, fully reset field-level metadata.
            self.is_nested = None
            self.is_non_tensor = None
            self.shape = None
            self.dtype = None
            self.per_sample_shapes.clear()
        else:
            if self.is_nested:
                # Check if all remaining shapes are the same
                remaining_shapes = set(
                    tuple(shape) if isinstance(shape, list) else shape for shape in self.per_sample_shapes.values()
                )
                if len(remaining_shapes) == 1:
                    # All remaining samples have the same shape - update to non-nested
                    self.is_nested = False
                    self.shape = next(iter(remaining_shapes))
                    # Clear per-sample shapes since we are no longer nested
                    self.per_sample_shapes.clear()

    def to_batch_schema(self, batch_global_indexes: list[int]) -> dict[str, Any]:
        """Export as a BatchMeta.field_schema-compatible dict for generate_batch_meta."""
        schema = {
            "dtype": self.dtype,
            "shape": self.shape,
            "is_nested": self.is_nested,
            "is_non_tensor": self.is_non_tensor,
        }
        if self.is_nested and self.per_sample_shapes:
            schema["per_sample_shapes"] = [self.per_sample_shapes.get(gi) for gi in batch_global_indexes]

        return schema


@dataclass
class DataPartitionStatus:
    """
    Robust status information for a data partition with dynamic expansion support.

    This class tracks the production and consumption status of data within a specific
    partition (e.g., "train@global_batch_0", "inference@kv_cache_1") with full support
    for dynamic row and column expansion.
    """

    partition_id: str
    created_at: float = field(default_factory=time.time)

    # Production status tensor - dynamically expandable
    # Values: 0 = not produced, 1 = ready for consumption
    TQ_PRE_ALLOC_SAMPLE_NUM = int(os.environ.get("TQ_PRE_ALLOC_SAMPLE_NUM", 1))

    production_status: Tensor = field(
        default_factory=lambda: torch.zeros(DataPartitionStatus.TQ_PRE_ALLOC_SAMPLE_NUM, 1, dtype=torch.int8)
    )

    # Consumption status per task - task_name -> consumption_tensor
    # Each tensor tracks which samples have been consumed by that task
    consumption_status: dict[str, Tensor] = field(default_factory=dict)

    # Global indexes
    global_indexes: set[int] = field(
        default_factory=set
    )  # set of global indexes that have been added to this partition

    pre_allocated_global_indexes: set[int] = field(
        default_factory=set
    )  # set of global indexes that pre-allocated, but not active in this partition

    # Metadata
    field_name_mapping: dict[str, int] = field(default_factory=dict)  # field_name -> column_index
    # O(F) columnar field metadata: field_name -> FieldMeta
    field_metadata: dict[str, FieldMeta] = field(default_factory=dict)
    field_custom_backend_meta: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )  # global_idx -> {field: custom_backend_meta}
    # User-defined metadata that may not apply to field level
    custom_meta: dict[int, dict[str, Any]] = field(default_factory=dict)  # global_idx -> {}

    # User-defined Keys
    keys_mapping: dict[str, int] = field(default_factory=dict)  # key -> global_idx
    revert_keys_mapping: dict[int, str] = field(default_factory=dict)  # global_idx -> key

    # Linked fields hold no payload of their own and read a PUBLIC_PARTITION row's instead
    links: dict[int, dict[str, int]] = field(default_factory=dict)  # global_idx -> {field: public global_idx}
    linked_by: dict[int, set[int]] = field(default_factory=dict)  # public global_idx -> global_idxs linking to it

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Controller checkpoints pickle partitions; ones saved before kv_link have no links.
        self.__dict__.update({"links": {}, "linked_by": {}, **state})

    # Dynamic configuration - these are computed from the current state
    @property
    def total_samples_num(self) -> int:
        """Current number of samples in the partition."""
        return len(self.global_indexes)

    @property
    def total_fields_num(self) -> int:
        """Current number of fields (columns) in the partition."""
        return len(self.field_name_mapping)

    @property
    def allocated_fields_num(self) -> int:
        """Current number of allocated columns in the tensor."""
        return self.production_status.shape[1]

    @property
    def allocated_samples_num(self) -> int:
        """Current number of allocated rows in the tensor."""
        return self.production_status.shape[0]

    # ==================== Index Pre-Allocation Methods ====================
    def register_pre_allocated_indexes(self, allocated_indexes: list[int]):
        """
        Register pre-allocated sample indexes to this partition.

        These indexes are reserved before actual data production, allowing consumers
        to see the expected total sample count via get_consumption_status even when
        producers haven't generated all samples yet.

        Args:
            allocated_indexes: List of global indexes to pre-allocate
        """

        if len(allocated_indexes) < 1:
            logger.info("Trying to pre-allocate global_indexes with empty list!")
            return

        self.pre_allocated_global_indexes.update(allocated_indexes)

        # Expand the state matrices
        max_sample_idx = max(allocated_indexes)
        required_samples = max_sample_idx + 1

        self.ensure_samples_capacity(required_samples)

        logger.debug(f"Pre-allocated indexes in {self.partition_id}: {allocated_indexes}")

    def activate_pre_allocated_indexes(self, sample_num: int) -> list[int]:
        """
        Activate and retrieve pre-allocated indexes for use in data insertion.

        This method consumes pre-allocated indexes and marks them as active in global_indexes.
        If pre-allocated indexes are insufficient, returns all available ones.

        Args:
            sample_num: Number of indexes needed

        Returns:
            List of retrieved global indexes
        """
        available_indexes = len(self.pre_allocated_global_indexes)

        if available_indexes < sample_num:
            global_index_to_allocate = list(self.pre_allocated_global_indexes)
            logger.debug(
                f"Not enough pre-allocated indexes in partition {self.partition_id}. "
                f"Returning {available_indexes} of {sample_num} requested."
            )
        else:
            global_index_to_allocate = list(sorted(self.pre_allocated_global_indexes))[:sample_num]

        self.global_indexes.update(global_index_to_allocate)
        self.pre_allocated_global_indexes.difference_update(set(global_index_to_allocate))

        return global_index_to_allocate

    # ==================== Dynamic Expansion Methods ====================
    def ensure_samples_capacity(self, required_samples: int) -> None:
        """
        Ensure the production status tensor has enough rows for the required samples.

        Args:
            required_samples: Minimum number of samples needed
        """

        current_sample_space = self.allocated_samples_num
        if required_samples > current_sample_space:
            # Expand rows
            expansion_needed = required_samples - current_sample_space
            new_samples = current_sample_space + expansion_needed
            new_fields = self.production_status.shape[1]

            expanded_tensor = torch.zeros(new_samples, new_fields, dtype=torch.int8)
            expanded_tensor[:current_sample_space, :] = self.production_status
            self.production_status = expanded_tensor

            # Update consumption tensors for all tasks
            for task_name, consumption_tensor in self.consumption_status.items():
                expanded_consumption = torch.zeros(new_samples, dtype=torch.int8)
                expanded_consumption[:current_sample_space] = consumption_tensor
                self.consumption_status[task_name] = expanded_consumption

            logger.debug(f"Expanded partition {self.partition_id} from {current_sample_space} to {new_samples} samples")

    def ensure_fields_capacity(self, required_fields: int) -> None:
        """
        Ensure the production status tensor has enough columns for the required fields.

        Args:
            required_fields: Minimum number of fields needed
        """

        current_fields = self.production_status.shape[1]
        if required_fields > current_fields:
            # Expand columns
            expansion_needed = required_fields - current_fields
            new_fields = current_fields + expansion_needed
            new_samples = self.production_status.shape[0]

            expanded_tensor = torch.zeros(new_samples, new_fields, dtype=torch.int8)
            expanded_tensor[:, :current_fields] = self.production_status
            self.production_status = expanded_tensor

            logger.debug(f"Expanded partition {self.partition_id} from {current_fields} to {new_fields} fields")

    # ==================== Production Status Interface ====================
    def update_production_status(
        self,
        global_indices: list[int],
        field_names: list[str],
        field_schema: dict[str, dict[str, Any]],
        custom_backend_meta: dict[int, dict[str, Any]] | None = None,
        ready: bool = True,
    ) -> bool:
        """
        Update production status for specific samples and fields.
        Handles dynamic expansion of both samples and fields.

        Note: field_names is derived from field_schema.keys() internally.
        The parameter is kept for backward compatibility but ignored;
        callers should ensure field_schema contains all intended fields.

        Args:
            global_indices: List of sample indices to update
            field_names: List of field names (ignored; derived from field_schema.keys())
            field_schema: Columnar field schema {field_name: {dtype, shape, is_nested, ...}}
            custom_backend_meta: Optional per-sample per-field
                custom metadata provided by storage backend
            ready: Whether consumers may see these cells. False records the stored
                value (field metadata) but hides it until a finalizing kv_update.

        Returns:
            True if update was successful, False on error

        Raises:
            ValueError: If a field's dtype conflicts with the partition's; metadata and readiness are unchanged.
        """
        # A dtype conflict must leave controller metadata and readiness unchanged.
        # Payloads are already stored, so this check cannot roll back the write.
        for name, incoming in field_schema.items():
            existing = self.field_metadata.get(name)
            dtype = incoming.get("dtype")
            if existing is not None and None not in (existing.dtype, dtype) and existing.dtype != dtype:
                raise ValueError(f"Field {name!r} dtype mismatch: {existing.dtype} != {dtype}")

        try:
            # Derive field_names from field_schema to guarantee consistency
            field_names = list(field_schema.keys())

            # Determine required capacity
            max_sample_idx = max(global_indices) if global_indices else -1
            required_samples = max_sample_idx + 1

            # Ensure we have enough rows
            self.ensure_samples_capacity(required_samples)

            # Register new fields if needed
            new_fields = [f for f in field_names if f not in self.field_name_mapping]
            if new_fields:
                # Add new fields to mapping
                for f in new_fields:
                    self.field_name_mapping[f] = len(self.field_name_mapping)

                required_fields = len(self.field_name_mapping)
                self.ensure_fields_capacity(required_fields)

            # Update production status
            if self.production_status is not None and global_indices and field_names:
                field_indices = [self.field_name_mapping.get(f) for f in field_names]
                self.production_status[torch.tensor(global_indices)[:, None], torch.tensor(field_indices)] = int(ready)

            # Update field metadata
            self._update_field_metadata(global_indices, field_schema, custom_backend_meta)

            # Save these global_indexes
            self.global_indexes.update(global_indices)

            # A written value replaces the link it lands on
            self._drop_links(global_indices, field_names)

            return True

        except Exception as e:
            logger.error(f"Error updating production status for partition {self.partition_id}: {e}")
            return False

    def validate_field_schema(self, field_schema: dict[str, dict[str, Any]]) -> None:
        """Reject incompatible field types without changing metadata or readiness."""
        for name, incoming in field_schema.items():
            existing = self.field_metadata.get(name)
            if existing is None:
                continue
            if bool(existing.is_non_tensor) != bool(incoming.get("is_non_tensor", False)):
                raise ValueError(f"Field {name!r} tensor/non-tensor type mismatch")
            dtype = incoming.get("dtype")
            if dtype is not None and existing.dtype is not None and dtype != existing.dtype:
                raise ValueError(f"Field {name!r} dtype mismatch: {existing.dtype} != {dtype}")

    def _update_field_metadata(
        self,
        global_indexes: list[int],
        field_schema: dict[str, dict[str, Any]],
        custom_backend_meta: dict[int, dict[str, Any]] | None = None,
    ):
        """Update field metadata from columnar field_schema."""
        if not global_indexes:
            return

        for field_name, meta in field_schema.items():
            if field_name not in self.field_metadata:
                self.field_metadata[field_name] = FieldMeta(
                    global_indexes=set(global_indexes),
                    dtype=meta.get("dtype"),
                    shape=meta.get("shape"),
                    is_nested=meta.get("is_nested", False),
                    is_non_tensor=meta.get("is_non_tensor", False),
                    per_sample_shapes=meta.get("per_sample_shapes", {}),
                )
            else:
                self.field_metadata[field_name].update(meta, global_indexes)

        # custom_backend_meta remains row-oriented storage
        if custom_backend_meta:
            for global_idx, per_field_meta in custom_backend_meta.items():
                if global_idx not in self.field_custom_backend_meta:
                    self.field_custom_backend_meta[global_idx] = {}
                self.field_custom_backend_meta[global_idx].update(per_field_meta)

    def mark_consumed(self, task_name: str, global_indices: list[int]):
        """
        Mark specific samples as consumed by a task.

        Args:
            task_name: Name of the consumer task
            global_indices: List of sample indices to mark as consumed

        """
        try:
            _, consumption_status = self.get_consumption_status(task_name, mask=False)

            if consumption_status.numel() > 0 and global_indices:
                consumption_status[global_indices] = 1
        except Exception as e:
            logger.error(
                f"Error marking samples consumed for partition {self.partition_id}, task {task_name}: {e}. "
                f"Target global_indices {global_indices}, but current consumption_status has "
                f"shape {consumption_status.shape}"
            )

    # ==================== Consumption Status Interface ====================
    def get_consumption_status(self, task_name: str, mask: bool = False) -> tuple[Tensor, Tensor]:
        """
        Get or create consumption status for a specific task.
        Handles dynamic expansion when new samples are added.

        Args:
            task_name: Name of the consumer task
            mask: Whether to return only the status for current partition samples

        Returns:
            Tuple of:
            - Partition global index tensor
            - Consumption status tensor for the specified task. 1 for consumed, 0 for not consumed.
        """

        if task_name not in self.consumption_status:
            if self.production_status is not None:
                self.consumption_status[task_name] = torch.zeros(self.allocated_samples_num, dtype=torch.int8)
            else:
                self.consumption_status[task_name] = torch.zeros(0, dtype=torch.int8)

        # Get consumption status for requested task
        partition_global_index = torch.tensor(
            sorted(self.global_indexes | self.pre_allocated_global_indexes), dtype=torch.long
        )

        if mask:
            if partition_global_index.numel() == 0:
                empty_status = self.consumption_status[task_name].new_zeros(0)
                return partition_global_index, empty_status
            self.ensure_samples_capacity(max(partition_global_index) + 1)
            consumption_status = self.consumption_status[task_name][partition_global_index]
        else:
            consumption_status = self.consumption_status[task_name]
        return partition_global_index, consumption_status

    def reset_consumption(self, task_name: str | None = None):
        """
        Reset consumption status for a specific task or all tasks.

        This allows the same data to be re-consumed without clearing the actual data.
        Useful for debugging scenarios where the same rollout data needs to be
        trained multiple times.
        Args:
            task_name: Name of the task to reset consumption for.
                      If None, resets consumption status for all tasks.
        """
        if task_name is not None:
            # Reset specific task
            if task_name in self.consumption_status:
                self.consumption_status[task_name].zero_()
                logger.debug(f"Reset consumption status for task '{task_name}' in partition {self.partition_id}")
        else:
            # Reset all tasks
            for name, status_tensor in self.consumption_status.items():
                status_tensor.zero_()
            logger.debug(f"Reset consumption status for all tasks in partition {self.partition_id}")

    # ==================== Production Status Interface ====================
    def get_production_status_for_fields(
        self, field_names: list[str], mask: bool = False
    ) -> tuple[Tensor | None, Tensor | None]:
        """
        Check if all samples for specified fields are fully produced and ready.

        Args:
            field_names: List of field names to check production status for
            mask: Whether to return only the status for current partition samples

        Returns:
            Tuple of:
            - Partition global index tensor
            - Production status tensor for the specified task. 1 for ready, 0 for not ready.
        """
        if field_names is None or len(field_names) == 0:
            return None, None

        # Check if all requested fields are registered
        for field_name in field_names:
            if field_name not in self.field_name_mapping:
                return None, None

        # Create column mask for requested fields
        col_mask = torch.zeros(self.allocated_fields_num, dtype=torch.bool)
        field_indices = [self.field_name_mapping[field] for field in field_names]
        if field_indices:
            col_mask[field_indices] = True

        production_status = self.production_status[:, col_mask]

        partition_global_index = torch.tensor(
            sorted(self.global_indexes | self.pre_allocated_global_indexes), dtype=torch.long
        )

        if mask:
            production_status = production_status[partition_global_index]

        return partition_global_index, production_status

    # ==================== Data Scanning and Query Methods ====================
    def scan_data_status(self, field_names: list[str], task_name: str) -> list[int]:
        """
        Scan data status to find samples ready for consumption.
        This replaces the original _scan_data_status functionality.

        Args:
            field_names: List of required field names
            task_name: Name of the consumer task

        Returns:
            List of sample indices that are ready for consumption
        """

        # Check if all requested fields are registered
        for field_name in field_names:
            if field_name not in self.field_name_mapping:
                return []

        row_mask = torch.ones(self.allocated_samples_num, dtype=torch.bool)

        # Apply consumption filter (exclude already consumed samples)
        _, consumption_status = self.get_consumption_status(task_name, mask=False)
        if consumption_status is not None:
            unconsumed_mask = consumption_status == 0
            row_mask &= unconsumed_mask

        # Create column mask for requested fields
        col_mask = torch.zeros(self.allocated_fields_num, dtype=torch.bool)
        field_indices = [self.field_name_mapping[field] for field in field_names]
        if field_indices:
            col_mask[field_indices] = True

        # Filter production status by masks
        relevant_status = self.production_status[row_mask][:, col_mask]

        # Check if all required fields are ready for each sample
        all_fields_ready = torch.all(relevant_status, dim=1)
        ready_indices_in_filtered = torch.nonzero(all_fields_ready, as_tuple=False).flatten()

        # Map back to original sample indices
        all_indices = torch.where(row_mask)[0]
        ready_sample_indices = all_indices[ready_indices_in_filtered].tolist()

        return ready_sample_indices

    # ==================== Metadata Methods ====================
    def get_field_schema(
        self, field_names: list[str], batch_global_indexes: list[int] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return field_schema from the FieldMeta store."""
        gi = batch_global_indexes or []
        return {f: self.field_metadata[f].to_batch_schema(gi) for f in field_names if f in self.field_metadata}

    def get_field_custom_backend_meta(
        self, global_indices: list[int], field_names: list[str]
    ) -> dict[int, dict[str, Any]]:
        """
        Get custom_backend_meta for multiple samples and fields.

        This method retrieves backend-specific metadata stored at per-sample per-field level.
        The returned dictionary maps global_index to a dictionary of field_name to metadata.

        Args:
            global_indices: List of global sample indices to retrieve metadata for
            field_names: List of field names to filter by. Only metadata for these
                        fields will be included in the result.

        Returns:
            Dictionary mapping global_index to field-name-to-metadata mapping.
            Only includes indices that have custom_backend_meta set.

        Example:
            >>> partition.get_field_custom_backend_meta([0, 1], ["field_a", "field_b"])
            {0: {'field_a': {'meta1': 'xxx'}, 'field_b': {'meta1': 'xxx'}}, 1: {...}}
        """

        wanted = set(field_names)
        return {
            idx: {f: v for f, v in self.field_custom_backend_meta[idx].items() if f in wanted}
            for idx in global_indices
            if idx in self.field_custom_backend_meta
        }

    def get_custom_meta(self, global_indices: list[int]) -> dict[int, dict]:
        """
        Get custom_meta for multiple samples.

        This method retrieves user-defined per-sample metadata.

        Args:
            global_indices: List of global sample indices to retrieve metadata for

        Returns:
            Dictionary mapping global_index to custom metadata dict.
            Only includes indices that have custom_meta set.

        Example:
            >>> partition.get_custom_meta([0, 2])
            {0: {'score': 0.9}, 2: {'label': 'positive'}}
        """
        return {idx: self.custom_meta[idx] for idx in global_indices if idx in self.custom_meta}

    def set_custom_meta(self, custom_meta: dict[int, dict]) -> None:
        """
        Set custom_meta for multiple samples.

        This method sets or updates user-defined per-sample metadata.

        Args:
            custom_meta: Dictionary mapping global_index to custom metadata dict.
                        Existing entries will be overwritten.
        """

        self.custom_meta.update(custom_meta)

    # ==================== Statistics and Monitoring ====================
    def get_statistics(self) -> dict[str, Any]:
        """Get detailed statistics for this partition."""
        stats = {
            "partition_id": self.partition_id,
            "created_at": self.created_at,
            "total_samples_num": self.total_samples_num,
            "total_fields_num": self.total_fields_num,
            "allocated_samples_num": self.allocated_samples_num,
            "allocated_fields_num": self.allocated_fields_num,
            "registered_tasks": list(self.consumption_status.keys()),
        }

        if self.production_status is not None:
            produced_samples = torch.any(self.production_status == 1, dim=1).sum().item()
            stats["produced_samples"] = produced_samples
            stats["production_progress"] = (
                produced_samples / self.total_samples_num if self.total_samples_num > 0 else 0
            )

            # Field-wise production statistics
            field_stats = {}
            for field_name, field_idx in self.field_name_mapping.items():
                field_produced = (self.production_status[:, field_idx] == 1).sum().item()
                field_stats[field_name] = {
                    "produced_samples": field_produced,
                    "production_progress": (
                        field_produced / self.total_samples_num if self.total_samples_num > 0 else 0
                    ),
                }
            stats["field_statistics"] = field_stats

        # Consumption statistics per task
        consumption_stats = {}
        for task_name, consumption_tensor in self.consumption_status.items():
            consumed_samples = (consumption_tensor == 1).sum().item()
            consumption_stats[task_name] = {
                "consumed_samples": consumed_samples,
                "consumption_progress": (
                    consumed_samples / self.total_samples_num if self.total_samples_num > 0 else 0
                ),
            }
        stats["consumption_statistics"] = consumption_stats

        return stats

    # ==================== Serialization ====================
    def to_snapshot(self):
        """
        Get a snapshot of partition status information.

        Returns:
            DataPartitionStatus object
        """

        cls = self.__class__
        snapshot = cls.__new__(cls)

        for name, value in self.__dict__.items():
            if isinstance(value, Tensor):
                new_val = value.clone().detach()
            else:
                new_val = copy.deepcopy(value)

            setattr(snapshot, name, new_val)
        return snapshot

    def clear_data(self, indexes_to_release: list[int], clear_consumption: bool = True):
        """Clear all production and optionally consumption data for given global_indexes."""
        try:
            if self.production_status is not None:
                self.production_status[indexes_to_release, :] = 0

            if clear_consumption:
                for consumption_tensor in self.consumption_status.values():
                    consumption_tensor[indexes_to_release] = 0

            self.global_indexes.difference_update(indexes_to_release)

            empty_fields = []
            for field_name, field_meta in self.field_metadata.items():
                field_meta.remove_samples(indexes_to_release)
                if len(field_meta.global_indexes) == 0:
                    empty_fields.append(field_name)
            if len(self.global_indexes) == 0:
                # clear the whole field_meta if the whole partition is empty
                self.field_metadata.clear()
            else:
                # only clear empty fields
                for field_name in empty_fields:
                    self.field_metadata.pop(field_name)
            self._drop_links(indexes_to_release)
            for idx in indexes_to_release:
                self.field_custom_backend_meta.pop(idx, None)
                self.custom_meta.pop(idx, None)

                if idx in self.revert_keys_mapping:
                    self.keys_mapping.pop(self.revert_keys_mapping[idx], None)
                    self.revert_keys_mapping.pop(idx, None)

        except Exception as e:
            logger.error(
                f"Error clearing data for partition {self.partition_id}: {e}. "
                f"Attempted to clear global_indexes: {indexes_to_release}"
            )

    def clear_fields(self, global_indexes: list[int], field_names: list[str]):
        """Mark fields as never written for these samples; keys, tags, consumption and indexes stay."""
        for field_name in field_names:
            col = self.field_name_mapping.get(field_name)
            if col is not None:
                self.production_status[global_indexes, col] = 0
            field_meta = self.field_metadata.get(field_name)
            if field_meta is not None:
                field_meta.remove_samples(global_indexes)
                if not field_meta.global_indexes:
                    self.field_metadata.pop(field_name)
        for idx in global_indexes:
            backend_meta = self.field_custom_backend_meta.get(idx, {})
            for field_name in field_names:
                backend_meta.pop(field_name, None)
        self._drop_links(global_indexes, field_names)

    # ==================== Field Links ====================
    def link_fields(
        self,
        global_indexes: list[int],
        public: "DataPartitionStatus",
        source: int,
        field_names: list[str],
        ready: bool = True,
    ) -> None:
        """Mark ``field_names`` of these rows as read from row ``source`` of the public partition.

        The rows take the source row's dtype and shape, and ``ready`` as their readiness.
        """
        schema = {}
        for name in field_names:
            meta = public.field_metadata[name]
            shape = meta.per_sample_shapes.get(source) if meta.is_nested else meta.shape
            schema[name] = {
                "dtype": meta.dtype,
                "shape": shape,
                "is_nested": False,
                "is_non_tensor": meta.is_non_tensor,
            }
        # Before recording links: update_production_status drops links on the cells it marks.
        if not self.update_production_status(global_indexes, field_names, schema, ready=ready):
            raise RuntimeError(f"failed to link fields {field_names} in partition {self.partition_id}")
        for idx in global_indexes:
            row_links = self.links.setdefault(idx, {})
            for name in field_names:
                row_links[name] = source
        self.linked_by.setdefault(source, set()).update(global_indexes)

    def linked_sources(self, sources: list[int], field_names: list[str] | None = None) -> set[int]:
        """The public rows among ``sources`` that rows here link ``field_names`` (all if None) to."""
        return {
            source
            for source in sources
            for idx in self.linked_by.get(source, ())
            if field_names is None or any(self.links[idx].get(name) == source for name in field_names)
        }

    def _drop_links(self, global_indexes: list[int], field_names: list[str] | None = None) -> None:
        """Drop the links of these rows on ``field_names`` (all fields if None)."""
        for idx in global_indexes:
            row_links = self.links.get(idx)
            if row_links is None:
                continue
            for name in list(row_links) if field_names is None else field_names:
                source = row_links.pop(name, None)
                if source is not None and source not in row_links.values():
                    self.linked_by[source].discard(idx)
                    if not self.linked_by[source]:
                        del self.linked_by[source]
            if not row_links:
                del self.links[idx]

    def kv_retrieve_indexes(self, keys: list[str]) -> list[int | None]:
        """Translate the user-specified keys to global_indexes"""
        global_indexes = [self.keys_mapping.get(k, None) for k in keys]
        return global_indexes

    def kv_retrieve_keys(self, global_indexes: list[int]) -> list[str | None]:
        """Translate the global_indexes to keys"""
        keys = [self.revert_keys_mapping.get(idx, None) for idx in global_indexes]
        return keys


@ray.remote(num_cpus=1)
class TransferQueueController:
    """
    TransferQueue Controller with partition-based data management.

    This refactored controller manages data through dynamic partitions instead of
    fixed global batches. Each partition represents a logical data container
    (e.g., "train@global_batch_0", "inference@kv_cache_1") that can be created
    on-demand and managed independently.

    Key improvements:
    - Dynamic partition creation on-demand
    - No dependency on training-specific parameters (global_batch_size, etc.)
    - Support for diverse use cases (KV cache migration, model resharding, etc.)
    - Flexible data organization through partition-based addressing
    """

    def __init__(
        self,
        sampler: BaseSampler | type[BaseSampler] = SequentialSampler,
        polling_mode: bool = False,
    ) -> None:
        """Initialize the TransferQueue Controller.

        Args:
            sampler: Sampler instance or sampler class to use for data sampling.
                    - If a BaseSampler instance is provided, it will be used directly
                    - If a BaseSampler subclass is provided, it will be instantiated
                    - Defaults to SequentialSampler for simple sequential sampling
                    - Example: sampler=GRPOGroupNSampler() (instance)
                    - Example: sampler=SequentialSampler (class)
            polling_mode: Whether to use polling mode for TransferQueue controller.
                    - If False, the controller will raise an error when no enough data is available.
                    - If True, the controller will return an empty BatchMeta when no enough data is available.
                               The user side is responsible for handling this empty case (retrying later).
        """
        if isinstance(sampler, BaseSampler):
            self.sampler = sampler
        elif isinstance(sampler, type) and issubclass(sampler, BaseSampler):
            self.sampler = sampler()
        else:
            raise TypeError(
                f"sampler {getattr(sampler, '__name__', repr(sampler))} must be an instance or subclass of BaseSampler"
            )

        self.controller_id = f"TQ_CONTROLLER_{uuid4().hex[:8]}"
        self.polling_mode = polling_mode
        self.tq_config = None  # global config for TransferQueue system

        # Initialize ZMQ sockets for communication
        self._init_zmq_socket()

        # Partition management
        self.partitions: dict[str, DataPartitionStatus] = {}  # partition_id -> DataPartitionStatus

        # Partition-GlobalIndex management
        self.index_manager = PartitionIndexManager()  # partition_id -> global_indexes

        # Connected storage managers tracking
        self._connected_storage_managers: set[str] = set()

        # Prometheus metrics exporter (lazy-initialized when enabled)
        self._metrics: TQMetricsExporter | None = None
        self._metrics_endpoint: str = ""

        # Start background processing threads
        self._start_process_handshake()
        self._start_process_request()

        logger.info(f"TransferQueue Controller {self.controller_id} initialized")

    # ==================== Partition Management API ====================
    def create_partition(self, partition_id: str) -> bool:
        """
        Create a new data partition with pre-allocated sample indexes.

        Partitions dynamically expand as needed. Additionally, TQ_PRE_ALLOC_SAMPLE_NUM
        indexes are pre-allocated to allow consumers to determine consumption status
        before all samples are produced.

        Args:
            partition_id: Unique identifier for the partition (e.g., "train@global_batch_0")

        Returns:
            True if partition was created successfully, False if it already exists
        """
        TQ_PRE_ALLOC_SAMPLE_NUM = int(os.environ.get("TQ_PRE_ALLOC_SAMPLE_NUM", 1))

        if partition_id in self.partitions:
            logger.warning(f"Partition {partition_id} already exists")
            return False

        self.partitions[partition_id] = DataPartitionStatus(partition_id=partition_id)

        # Pre-allocate global indexes for consumer consumption tracking
        global_indexes = self.index_manager.allocate_indexes(partition_id, count=TQ_PRE_ALLOC_SAMPLE_NUM)
        self.partitions[partition_id].register_pre_allocated_indexes(global_indexes)

        logger.info(f"Created partition {partition_id} with {TQ_PRE_ALLOC_SAMPLE_NUM} pre-allocated indexes")
        return True

    def _get_partition(self, partition_id: str) -> DataPartitionStatus | None:
        """
        Get partition status information.

        Args:
            partition_id: ID of the partition to retrieve

        Returns:
            DataPartitionStatus object if partition exists, None otherwise
        """
        return self.partitions.get(partition_id)

    def get_partition_snapshot(self, partition_id: str) -> DataPartitionStatus | None:
        """
        Get a copy of partition status information.

        Args:
            partition_id: ID of the partition to retrieve

        Returns:
            DataPartitionStatus object if partition exists, None otherwise
        """

        partition = self._get_partition(partition_id)

        if partition is None:
            return None

        return partition.to_snapshot()

    def list_partitions(self) -> list[str]:
        """
        List all available partition IDs.

        Returns:
            List of partition IDs
        """
        return list(self.partitions.keys())

    # ==================== Partition Index Management API ====================
    def get_partition_index_range(self, partition_id: str) -> list[int]:
        """
        Get all indexes for a specific partition.

        Args:
            partition_id: Partition identifier

        Returns:
            List of indexes allocated to the partition
        """
        # Note: This includes the pre-allocated global_indexes for the partition.
        # i.e., partition.global_indexes + partition.pre_allocated_global_indexes
        return self.index_manager.get_indexes_for_partition(partition_id)

    # ==================== Data Production API ====================
    def update_production_status(
        self,
        partition_id: str,
        global_indexes: list[int],
        field_schema: dict[str, dict[str, Any]],
        custom_backend_meta: dict[int, dict[str, Any]] | None = None,
        ready: bool = True,
    ) -> bool:
        """
        Update production status for specific samples and fields in a partition.
        Delegates to the partition's own update_production_status method.

        Args:
            partition_id: ID of the partition
            global_indexes: List of sample indices to update
            field_schema: Columnar field schema {field_name: {dtype, shape, is_nested, ...}}
            custom_backend_meta: Optional custom backend metadata
            ready: False keeps the written cells hidden from consumers (unfinalized kv_update)

        Returns:
            True if update was successful, False otherwise

        Raises:
            ValueError: If a field's dtype conflicts with the partition's; metadata and readiness are unchanged.
        """
        field_names = list(field_schema.keys())
        partition = self._get_partition(partition_id)
        if not partition:
            logger.error(f"Partition {partition_id} not found")
            return False

        success = partition.update_production_status(
            global_indexes, field_names, field_schema, custom_backend_meta, ready=ready
        )
        if success and partition_id == PUBLIC_PARTITION:
            # Linking rows read the rewritten payload, so they take on its readiness (an
            # unfinalized kv_update stays hidden behind links too) and its shape.
            for other in self.partitions.values():
                for source in set(global_indexes) & other.linked_by.keys():
                    for idx in list(other.linked_by[source]):
                        names = [name for name in field_names if other.links[idx].get(name) == source]
                        if names:
                            other.link_fields([idx], partition, source, names, ready=ready)
        if success:
            logger.debug(
                f"[{self.controller_id}]: Updated production status for partition {partition_id}: "
                f"samples={global_indexes}, fields={field_names}, ready={ready}"
            )
        return success

    # ==================== Data Consumption API ====================
    def get_consumption_status(self, partition_id: str, task_name: str) -> tuple[Tensor | None, Tensor | None]:
        """
        Get or create consumption status for a specific task and partition.
        Delegates to the partition's own method.

        Args:
            partition_id: ID of the partition
            task_name: Name of the consumer task

        Returns:
            Tuple of:
            - Partition global index tensor
            - Consumption status tensor for the specified task. 1 for consumed, 0 for not consumed.
        """
        partition = self._get_partition(partition_id)
        if not partition:
            return None, None

        return partition.get_consumption_status(task_name, mask=True)

    def get_production_status(self, partition_id: str, data_fields: list[str]) -> tuple[Tensor | None, Tensor | None]:
        """
        Check if all samples for specified fields are fully produced in a partition.

        Args:
            partition_id: ID of the partition
            data_fields: List of field names to check production status for

        Returns:
            Tuple of:
            - Partition global index tensor
            - Production status tensor for the specified task. 1 for ready, 0 for not ready.
        """
        partition = self._get_partition(partition_id)
        if not partition:
            return None, None

        return partition.get_production_status_for_fields(data_fields, mask=True)

    def set_custom_meta(self, partition_custom_meta: dict[str, dict[int, dict]]) -> None:
        """
        Set custom_meta for samples in partitions.

        This method allows setting per-sample custom metadata (custom_meta) for samples
        identified by their global indexes within specific partitions. Custom metadata
        is stored per-sample and can be retrieved along with BatchMeta in subsequent
        get_meta calls.

        Args:
            partition_custom_meta: Dictionary mapping partition_id to custom metadata dict.
                                  Format: {partition_id: {global_index: {metadata_key: metadata_value}}}
                                  - partition_id: ID of the partition
                                  - global_index: Global index of the sample
                                  - metadata_key/value: User-defined metadata key-value pairs

        Example:
            >>> # Set custom metadata for samples in different partitions
            >>> controller.set_custom_meta({
            ...     "train_0": {
            ...         0: {"score": 0.9, "label": "positive"},
            ...         1: {"score": 0.8, "label": "negative"}
            ...     },
            ...     "train_1": {
            ...         10: {"score": 0.95, "label": "positive"}
            ...     }
            ... })
        """

        for partition_id, custom_meta in partition_custom_meta.items():
            partition = self._get_partition(partition_id)
            if partition:
                partition.set_custom_meta(custom_meta)
            else:
                logger.warning(
                    f"set_custom_meta: partition {partition_id}' not found; "
                    f"custom_metadata for this partition will be ignored"
                )

    def get_metadata(
        self,
        data_fields: list[str],
        partition_id: str,
        mode: str = "fetch",
        task_name: str | None = None,
        batch_size: int | None = None,
        sampling_config: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> BatchMeta:
        """
        Retrieve metadata with support for three modes.

        Args:
            data_fields: List of field names to include in metadata
            partition_id: Partition id for which to retrieve metadata
            mode: Operation mode - 'insert', 'fetch', or 'force_fetch'
                - mode="insert": Create metadata for new samples (for data insertion)
                - mode="fetch": Get metadata from ready samples using the configured sampler
                - mode="force_fetch": Get metadata for unconsumed samples without sampling
                                      (excludes already consumed samples)
            task_name: Name of the consumer task (required for fetch modes)
            batch_size: Number of samples to retrieve
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            BatchMeta object containing the requested metadata

        Raises:
            TimeoutError: If waiting for sufficient data times out in fetch mode
        """

        if mode == "insert":
            if partition_id not in self.partitions:
                self.create_partition(partition_id)

            partition = self._get_partition(partition_id)
            if data_fields is None:
                raise RuntimeError("Must provide data_fields for inserting new data")

            # This is called during put_data call without providing metadata.
            # try to use pre-allocated global index first

            if batch_size is None:
                raise ValueError("must provide batch_size for inserting new data")

            assert partition is not None
            batch_global_indexes = partition.activate_pre_allocated_indexes(batch_size)

            if len(batch_global_indexes) < batch_size:
                new_global_indexes = self.index_manager.allocate_indexes(
                    partition_id, count=(batch_size - len(batch_global_indexes))
                )
                batch_global_indexes.extend(new_global_indexes)

            # register global_indexes in partition
            partition.global_indexes.update(batch_global_indexes)

            return self.generate_batch_meta(partition_id, batch_global_indexes, data_fields, mode)

        if mode == "fetch":
            assert task_name is not None
            # Find ready samples within current data partition and package into BatchMeta when reading

            if batch_size is None:
                raise ValueError("must provide batch_size in fetch mode")

            start_time = time.time()
            while True:
                # ready_for_consume_indexes: samples where all required fields are produced
                # (production status is ready) and not yet consumed
                ready_for_consume_indexes = self.scan_data_status(partition_id, data_fields, task_name)

                if len(ready_for_consume_indexes) < batch_size:
                    if self.polling_mode:
                        # Return cached result if available
                        if self.sampler.has_cached_result(partition_id, task_name, sampling_config):
                            break
                        else:
                            logger.debug(
                                f"[{self.controller_id}]: Not enough data for task {task_name} in "
                                f"partition {partition_id}. Required: {batch_size}, "
                                f"Available: {len(ready_for_consume_indexes)}."
                                f" Returning None due to polling mode."
                            )
                            return BatchMeta.empty()
                    else:
                        logger.warning(
                            f"[{self.controller_id}]: Insufficient data for task {task_name}. Required: {batch_size} "
                            f"samples with fields {data_fields} in partition {partition_id}, but only have "
                            f"{len(ready_for_consume_indexes)} samples meeting the criteria. "
                            f"Retrying in {TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
                        )
                        time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)
                    if time.time() - start_time > TQ_CONTROLLER_GET_METADATA_TIMEOUT:
                        raise TimeoutError(
                            f"Timeout while waiting for sufficient data for task {task_name}. "
                            f"Required: {batch_size}, Available: {len(ready_for_consume_indexes)}"
                        )
                else:
                    break

            batch_global_indexes, consumed_indexes = self.sampler(
                ready_for_consume_indexes,
                batch_size,
                partition=self._get_partition(partition_id),
                **(sampling_config or {}),
                **kwargs,
            )

            # Check if we got valid results from the sampler.
            # Some samplers (e.g. SeqlenBalancedSampler) may return variable-size
            # batches per DP rank, so we only check for empty results.
            if len(batch_global_indexes) == 0:
                if self.polling_mode:
                    return BatchMeta.empty()
                raise RuntimeError(
                    f"Sampler returned no samples. Please check the sampler logic. "
                    f"Expected: {batch_size}, before sampling: {len(ready_for_consume_indexes)}, "
                    f"after sampling: {len(batch_global_indexes)}"
                )

            # Mark samples as consumed if in fetch mode
            if consumed_indexes:
                partition = self.partitions[partition_id]
                partition.mark_consumed(task_name, consumed_indexes)

        elif mode == "force_fetch":
            batch_global_indexes = self.index_manager.get_indexes_for_partition(partition_id)
            consumed_indexes = []

        # Package into metadata
        metadata = self.generate_batch_meta(partition_id, batch_global_indexes, data_fields, mode)

        return metadata

    def scan_data_status(
        self,
        partition_id: str,
        data_fields: list[str],
        task_name: str,
    ) -> list[int]:
        """
        Find samples that are ready for consumption in a specific partition.
        Delegates scanning functionality to the partition's own method.

        Args:
            partition_id: ID of the partition
            data_fields: List of required field names
            task_name: Name of the consumer task

        Returns:
            List of global indices that are ready for consumption
        """

        partition = self._get_partition(partition_id)
        if not partition:
            return []

        # Use partition's own scanning method
        ready_sample_indices = partition.scan_data_status(data_fields, task_name)

        return ready_sample_indices

    # ==================== Metadata Generation API ====================
    def generate_batch_meta(
        self,
        partition_id: str,
        batch_global_indexes: list[int],
        data_fields: list[str],
        mode: str = "fetch",
    ) -> BatchMeta:
        """
        Generate BatchMeta for specific samples in a partition.

        O(F) optimized version that uses field_schema instead of per-sample metadata.

        This function is responsible only for metadata generation and does not
        modify consumption state. State management is handled by the calling function.

        Args:
            partition_id: ID of the partition
            batch_global_indexes: List of sample indices to include in the batch
            data_fields: List of field names to include
            mode: Operation mode - 'fetch', 'insert', or 'force_fetch'

        Returns:
            BatchMeta object containing sample metadata

        Raises:
            ValueError: If partition doesn't exist or invalid mode
        """
        partition = self._get_partition(partition_id)
        if not partition:
            raise ValueError(f"Partition {partition_id} not found")

        if mode not in ["fetch", "insert", "force_fetch"]:
            raise ValueError(f"Invalid mode: {mode}")

        batch_size = len(batch_global_indexes)

        field_schema = partition.get_field_schema(data_fields, batch_global_indexes)

        # In insert mode, create placeholder schema for unregistered fields so that
        # metadata.field_names is complete and update_production_status() can recognize them.
        if mode == "insert":
            for field_name in data_fields:
                if field_name not in field_schema:
                    field_schema[field_name] = {
                        "dtype": None,
                        "shape": None,
                        "is_nested": False,
                        "is_non_tensor": False,
                    }

        if mode == "fetch":
            production_status = np.ones(batch_size, dtype=np.int8)
        elif mode == "insert":
            production_status = np.zeros(batch_size, dtype=np.int8)
        else:  # force_fetch
            production_status = np.zeros(batch_size, dtype=np.int8)
            if partition.production_status is not None and data_fields:
                field_indices = [
                    partition.field_name_mapping.get(field_name)
                    for field_name in data_fields
                    if field_name in partition.field_name_mapping
                ]
                if field_indices:
                    rows = np.fromiter(batch_global_indexes, dtype=np.int64, count=batch_size)
                    in_range = rows < partition.production_status.shape[0]
                    if in_range.any():
                        row_idx = torch.from_numpy(rows[in_range])
                        col_idx = torch.as_tensor(field_indices, dtype=torch.long)
                        ready = (
                            (partition.production_status[row_idx[:, None], col_idx] == 1)
                            .all(dim=1)
                            .to(torch.int8)
                            .numpy()
                        )
                        production_status[in_range] = ready

        custom_meta_dict = partition.get_custom_meta(batch_global_indexes)
        custom_backend_meta = partition.get_field_custom_backend_meta(batch_global_indexes, data_fields)

        # Convert controller dict[int, dict] → BatchMeta list[dict] (aligned with batch_global_indexes)
        custom_meta_list = [custom_meta_dict.get(global_index, {}) for global_index in batch_global_indexes]
        custom_backend_meta_list = [custom_backend_meta.get(global_index, {}) for global_index in batch_global_indexes]
        links_list = None
        if partition.links:
            wanted = set(data_fields)
            links_list = [
                {name: source for name, source in partition.links.get(global_index, {}).items() if name in wanted}
                for global_index in batch_global_indexes
            ]

        batch_meta = BatchMeta(
            global_indexes=batch_global_indexes,
            partition_ids=[partition_id] * batch_size,
            field_schema=field_schema,
            production_status=production_status,
            custom_meta=custom_meta_list,
            _custom_backend_meta=custom_backend_meta_list,
            _links=links_list,
        )
        return batch_meta

    def mark_clearing(self, global_indexes: list[int], partition_ids: list[str]) -> None:
        """Mark samples as pending deletion by zeroing their production status.

        This prevents consumers from reading data that is about to be deleted from
        storage, closing the window between storage clear and controller clear.

        Args:
            global_indexes: global indexes to mark as pending deletion
            partition_ids: corresponding partition IDs for each global index
        """
        if global_indexes is None or partition_ids is None:
            raise ValueError("global_indexes and partition_ids cannot be None")

        if len(global_indexes) != len(partition_ids):
            raise ValueError(
                f"global_indexes and partition_ids must have the same length, "
                f"got {len(global_indexes)} and {len(partition_ids)}"
            )

        combined = list(zip(partition_ids, global_indexes, strict=True))
        combined.sort(key=itemgetter(0))

        # Validate first so a refused clear marks nothing.
        self._check_public_not_linked([idx for partition_id, idx in combined if partition_id == PUBLIC_PARTITION])

        for partition_id, group in groupby(combined, key=itemgetter(0)):
            partition = self._get_partition(partition_id)
            if not partition:
                logger.info(
                    f"[{self.controller_id}]: Trying to mark clearing in a non-existent partition {partition_id}. "
                    f"Skipping operation for this partition."
                )
                continue
            indexes = [idx for _, idx in group]
            existing = list(set(indexes) & partition.global_indexes)
            if existing and partition.production_status is not None:
                partition.production_status[existing, :] = 0

    def _check_public_not_linked(self, global_indexes: list[int], field_names: list[str] | None = None) -> None:
        """Refuse to clear or rewrite public payloads that keys of other partitions still link to."""
        linked = set().union(*(p.linked_sources(global_indexes, field_names) for p in self.partitions.values()))
        if linked:
            public = self.partitions[PUBLIC_PARTITION]
            keys = sorted(str(public.revert_keys_mapping.get(idx, idx)) for idx in linked)
            raise ValueError(
                f"keys {keys} in partition {PUBLIC_PARTITION!r} are linked by keys of other partitions; "
                "clear those keys first"
            )

    def _unlink_released_public(self, global_indexes: list[int]) -> None:
        """Mark fields linked to released public rows as never written.

        mark_clearing refuses linked rows, but a put during the clear makes a row ready
        again and lets kv_link through; such a link would point at a reusable index.
        """
        for other in self.partitions.values():
            for source in set(global_indexes) & other.linked_by.keys():
                for idx in list(other.linked_by[source]):
                    other.clear_fields([idx], [name for name, src in other.links[idx].items() if src == source])

    def clear_partition(self, partition_id: str, clear_consumption: bool = True):
        """
        Clear data for a specific partition (delete the whole partition).

        Args:
            partition_id: ID of the partition to clear
            clear_consumption: Whether to also clear consumption status
        """

        logger.debug(f"[{self.controller_id}]: Clearing metadata in partition {partition_id}.")

        partition = self._get_partition(partition_id)
        if not partition:
            logger.warning(
                f"[{self.controller_id}]: Trying to clear a non-existent partition {partition_id}. No action taken."
            )
            return

        global_indexes_range = list(self.index_manager.get_indexes_for_partition(partition_id))
        partition.clear_data(global_indexes_range, clear_consumption)
        if partition_id == PUBLIC_PARTITION:
            self._unlink_released_public(global_indexes_range)
        self.index_manager.release_partition(partition_id)
        self.partitions.pop(partition_id)
        self.sampler.clear_cache(partition_id)

    def reset_consumption(self, partition_id: str, task_name: str | None = None):
        """
        Reset consumption status for a partition without clearing the actual data.

        This allows the same data to be re-consumed, useful for debugging scenarios
        where the same rollout data needs to be trained multiple times.
        Args:
            partition_id: ID of the partition to reset consumption for
            task_name: Name of the task to reset. If None, resets all tasks.

        """
        logger.debug(f"[{self.controller_id}]: Resetting consumption for partition {partition_id}, task={task_name}")
        partition = self._get_partition(partition_id)
        if not partition:
            logger.warning(
                f"[{self.controller_id}]: Trying to reset consumption of a non-existent partition {partition_id}. "
                f"No action taken."
            )
            return
        partition.reset_consumption(task_name)

    def clear_meta(
        self,
        global_indexes: list[int],
        partition_ids: list[str],
        clear_consumption: bool = True,
    ):
        """
        Clear meta for individual samples (preserving the partition).

        Args:
            global_indexes: global_indexes to clear
            partition_ids: IDs of the partitions to clear
            clear_consumption: Whether to also clear consumption status
        """

        logger.debug(
            f"[{self.controller_id}]: Clearing meta with global_indexes {global_indexes} in partition {partition_ids}"
        )

        if global_indexes is None or partition_ids is None:
            raise ValueError("global_indexes and partition_ids cannot be None")

        if len(global_indexes) != len(partition_ids):
            raise ValueError(
                f"global_indexes and partition_ids must have the same length, "
                f"got {len(global_indexes)} and {len(partition_ids)}"
            )

        combined = list(zip(partition_ids, global_indexes, strict=True))
        combined.sort(key=itemgetter(0))

        for partition_id, group in groupby(combined, key=itemgetter(0)):
            partition = self._get_partition(partition_id)
            if not partition:
                logger.info(
                    f"[{self.controller_id}]: Trying to clear data in a non-existent partition {partition_id}. "
                    f"Skipping operation for this partition."
                )
                continue

            global_indexes_to_clear = [idx for _, idx in group]
            existing_global_indexes = partition.global_indexes
            non_existent_global_indexes = set(global_indexes_to_clear) - existing_global_indexes
            if non_existent_global_indexes:
                logger.info(
                    f"[{self.controller_id}]: Some global_indexes to be cleared do not exist in "
                    f"partition {partition_id}: {non_existent_global_indexes}. They will be ignored."
                )

            global_indexes_to_clear = list(set(global_indexes_to_clear) & existing_global_indexes)
            if not global_indexes_to_clear:
                logger.info(
                    f"[{self.controller_id}]: No existing global indexes to clear in partition {partition_id}. "
                    f"Skipping operation for this partition."
                )
                continue

            # Clear data from partition
            partition.clear_data(global_indexes_to_clear, clear_consumption)
            if partition_id == PUBLIC_PARTITION:
                self._unlink_released_public(global_indexes_to_clear)

            # Release the specific indexes from index manager
            self.index_manager.release_indexes(partition_id, global_indexes_to_clear)

    def kv_retrieve_meta(
        self,
        keys: list[str],
        partition_id: str,
        create: bool = False,
    ) -> BatchMeta:
        """
        Retrieve BatchMeta from the controller using a list of keys.
        For non-existing keys, we simply emit them in returned BatchMeta.

        Args:
            keys: List of keys to retrieve from the controller
            partition_id: Partition id to retrieve from the controller
            create: Whether to register new keys if not found.

        Returns:
            metadata: BatchMeta of the requested keys
        """

        logger.debug(f"[{self.controller_id}] Retrieve keys {keys} in partition {partition_id}")

        # Ensure partition exists
        partition = self._get_partition(partition_id)
        if partition is None:
            if not create:
                logger.warning(
                    f"[{self.controller_id}]: Partition {partition_id} not found. Returning empty BatchMeta."
                )
                return BatchMeta.empty()

            self.create_partition(partition_id)
            partition = self._get_partition(partition_id)

        assert partition is not None
        global_indexes = partition.kv_retrieve_indexes(keys)
        if create and partition_id == PUBLIC_PARTITION:
            # create=True is the KV write path (kv_put, kv_update, restore); a linked public row
            # stays read-only so linking keys never see their shared payload change.
            self._check_public_not_linked([idx for idx in global_indexes if idx is not None])

        none_indexes = [idx for idx, value in enumerate(global_indexes) if value is None]
        if len(none_indexes) > 0:
            if not create:
                logger.warning(
                    f"Keys {[keys[i] for i in none_indexes]} were not found in partition {partition_id}. "
                    f"They will be excluded from the retrieved BatchMeta."
                )
            else:
                # create non-exist keys
                batch_global_indexes = partition.activate_pre_allocated_indexes(len(none_indexes))

                if len(batch_global_indexes) < len(none_indexes):
                    new_global_indexes = self.index_manager.allocate_indexes(
                        partition_id, count=(len(none_indexes) - len(batch_global_indexes))
                    )
                    batch_global_indexes.extend(new_global_indexes)

                # register global_indexes in partition
                partition.global_indexes.update(batch_global_indexes)

                # register key-global_indexes mapping in partition
                for i in range(len(none_indexes)):
                    global_indexes[none_indexes[i]] = batch_global_indexes[i]
                    partition.keys_mapping[keys[none_indexes[i]]] = batch_global_indexes[i]
                    partition.revert_keys_mapping[batch_global_indexes[i]] = keys[none_indexes[i]]

                partition.ensure_samples_capacity(max(batch_global_indexes) + 1)

        verified_global_indexes = [idx for idx in global_indexes if idx is not None]

        # must fetch fields that the requested samples all have
        col_mask = partition.production_status[verified_global_indexes, :].sum(dim=0).reshape(-1) == len(
            verified_global_indexes
        )
        data_fields = []
        for field_name, col_idx in partition.field_name_mapping.items():
            if col_idx < len(col_mask) and col_mask[col_idx]:
                data_fields.append(field_name)

        return self.generate_batch_meta(partition_id, verified_global_indexes, data_fields, mode="force_fetch")

    def kv_prepare_update(
        self, keys: list[str], partition_id: str, fields: list[str]
    ) -> tuple[BatchMeta, dict[str, dict[str, Any]]]:
        """Register missing keys and tell kv_update, per field, which rows to merge into.

        Storage units may keep values the controller no longer tracks (e.g. after a failed
        clear), so ``held`` comes from ``FieldMeta.global_indexes``, which covers hidden cells.

        Returns:
            The keys' BatchMeta and ``{field: {"dtype", "is_non_tensor", "held"}}``, where
            ``held`` is aligned with the BatchMeta rows. The types are None for a new field.

        Raises:
            ValueError: A key links one of ``fields`` by kv_link; nothing is registered.
        """
        partition = self._get_partition(partition_id)
        if partition is not None:
            # A unit merges into the old value on the sample's own row; a linked value lives on the source's.
            linked = [
                key
                for key, idx in zip(keys, partition.kv_retrieve_indexes(keys), strict=True)
                if idx is not None and any(name in partition.links.get(idx, {}) for name in fields)
            ]
            if linked:
                raise ValueError(
                    f"kv_update cannot merge into fields linked by kv_link on keys {linked}; "
                    "kv_put a value to replace the link first"
                )
        metadata = self.kv_retrieve_meta(keys, partition_id, create=True)
        partition = self._get_partition(partition_id)
        assert partition is not None
        merge_plan = {}
        for name in fields:
            meta = partition.field_metadata.get(name)
            held = meta.global_indexes if meta is not None else set()
            merge_plan[name] = {
                "dtype": meta.dtype if meta is not None else None,
                "is_non_tensor": meta.is_non_tensor if meta is not None else None,
                "held": [idx in held for idx in metadata.global_indexes],
            }
        return metadata, merge_plan

    def kv_clear_fields(self, keys: list[str], partition_id: str, fields: list[str]) -> BatchMeta:
        """Clear fields of existing keys and return the cleared cells' metadata for storage deletion.

        Missing keys and fields are ignored. The metadata is captured before clearing so
        KV backends still receive the custom backend metadata needed to delete the values.
        """
        partition = self._get_partition(partition_id)
        if partition is None:
            return BatchMeta.empty()
        global_indexes = [idx for idx in partition.kv_retrieve_indexes(keys) if idx is not None]
        fields = [f for f in fields if f in partition.field_metadata]
        if not global_indexes or not fields:
            return BatchMeta.empty()
        if partition_id == PUBLIC_PARTITION:
            self._check_public_not_linked(global_indexes, fields)

        metadata = self.generate_batch_meta(partition_id, global_indexes, fields, mode="force_fetch")
        partition.clear_fields(global_indexes, fields)
        return metadata

    def kv_retrieve_keys(
        self,
        global_indexes: list[int],
        partition_id: str,
    ) -> list[str | None]:
        """
        Retrieve keys from the controller using a list of global_indexes.

        Args:
            global_indexes: List of global_indexes to retrieve keys from the controller
            partition_id: Partition id to retrieve from the controller

        Returns:
            metadata: BatchMeta of the requested keys
        """

        logger.debug(f"[{self.controller_id}]: Retrieve global_indexes {global_indexes} in partition {partition_id}")

        partition = self._get_partition(partition_id)

        if partition is None:
            logger.warning(f"Partition {partition_id} were not found in controller!")
            return []

        assert partition is not None
        keys = partition.kv_retrieve_keys(global_indexes)

        none_indexes = [idx for idx, value in enumerate(global_indexes) if value is None]
        if len(none_indexes) > 0:
            logger.error(
                f"Key for global_index {[keys[i] for i in none_indexes]} were not found in partition {partition_id}!"
            )
            return []

        return keys

    def kv_describe(self, keys: list[str], partition_id: str) -> dict[str, Any]:
        """Describe each key's row and the declared types of its fields, without payloads.

        Unlike ``kv_retrieve_meta``, which keeps only the fields every row has produced and
        drops missing keys, this reports each row's own field set and fails on a missing
        key. Avoids ``to_snapshot``: a handful of lookups per key, not a partition copy.

        Args:
            keys: Keys to describe, already deduplicated by the caller.
            partition_id: Partition that owns ``keys``.

        Returns:
            ``{"rows": {key: {"global_index", "fields", "unfinalized", "links", "tag"}}, "field_schema":
            {field: {"is_nested", "is_non_tensor"}}}``. ``fields`` lists the row's stored fields and is
            empty for a row with no produced field yet; ``unfinalized`` lists the stored fields not yet
            ready (written by ``kv_update(finalize=False)``); ``links`` maps each linked field to its
            key in ``PUBLIC_PARTITION``; ``field_schema`` covers the described rows' stored fields.

        Raises:
            KeyError: The partition or any key does not exist.
        """
        partition = self._get_partition(partition_id)
        if partition is None:
            raise KeyError(f"partition {partition_id!r} does not exist; existing partitions: {sorted(self.partitions)}")

        global_indexes = partition.kv_retrieve_indexes(keys)
        missing_keys = [key for key, index in zip(keys, global_indexes, strict=True) if index is None]
        if missing_keys:
            raise KeyError(f"keys not found in partition {partition_id!r}: {missing_keys}")

        public_keys = self.partitions[PUBLIC_PARTITION].revert_keys_mapping if partition.links else {}
        rows: dict[str, dict[str, Any]] = {}
        for key, global_index in zip(keys, cast(list[int], global_indexes), strict=True):
            links = partition.links.get(global_index, {})
            fields = sorted(
                field_name
                for field_name, field_meta in partition.field_metadata.items()
                if global_index in field_meta.global_indexes and field_name not in links
            )
            rows[key] = {
                "global_index": global_index,
                "fields": fields,
                "unfinalized": [
                    name
                    for name in fields
                    if not partition.production_status[global_index, partition.field_name_mapping[name]]
                ],
                "links": {name: public_keys[source] for name, source in links.items()},
                "tag": partition.custom_meta.get(global_index, {}),
            }
        # Only the declared types: row dtypes and shapes live with the storage units.
        field_schema = {
            name: {"is_nested": bool(meta.is_nested), "is_non_tensor": bool(meta.is_non_tensor)}
            for name, meta in partition.field_metadata.items()
            if any(name in row["fields"] for row in rows.values())
        }
        return {"rows": rows, "field_schema": field_schema}

    def kv_link(self, keys: list[str], partition_id: str, public_key: str, fields: list[str]) -> None:
        """Make ``fields`` of ``keys`` read the payload of ``public_key`` in ``PUBLIC_PARTITION``.

        Missing ``keys`` are created. ``public_key`` must already have produced ``fields``.

        Raises:
            KeyError: ``public_key`` does not exist.
            ValueError: ``partition_id`` is ``PUBLIC_PARTITION``, ``keys`` or ``fields`` is empty,
                ``public_key`` has not produced all ``fields``, or a key stores one of them itself.
        """
        if partition_id == PUBLIC_PARTITION:
            raise ValueError(f"keys of partition {PUBLIC_PARTITION!r} hold shared payloads and cannot link")
        if not keys:
            raise ValueError("keys must not be empty")
        public = self._get_partition(PUBLIC_PARTITION)
        if public is None or public_key not in public.keys_mapping:
            raise KeyError(f"key {public_key!r} not found in partition {PUBLIC_PARTITION!r}")
        source = public.keys_mapping[public_key]
        # Production status, not field metadata: mark_clearing zeroes it before storage deletes the payload.
        missing = [
            name
            for name in fields
            if name not in public.field_name_mapping
            or public.production_status[source, public.field_name_mapping[name]] != 1
        ]
        if not fields or missing:
            raise ValueError(f"public key {public_key!r} must have produced fields {missing or fields}")
        # Before creating keys, so an incompatible link leaves none behind.
        partition = self._get_partition(partition_id)
        if partition is not None:
            # Linking only redirects reads, so a value the key stores itself would stay in storage unread.
            stored = [
                key
                for key, idx in zip(keys, partition.kv_retrieve_indexes(keys), strict=True)
                if idx is not None
                and any(
                    name not in partition.links.get(idx, {})
                    and name in partition.field_metadata
                    and idx in partition.field_metadata[name].global_indexes
                    for name in fields
                )
            ]
            if stored:
                raise ValueError(
                    f"keys {stored[:20]} already store some of fields {fields}; "
                    "kv_clear those fields before linking them"
                )
            metas = {name: public.field_metadata[name] for name in fields}
            partition.validate_field_schema(
                {name: {"dtype": meta.dtype, "is_non_tensor": meta.is_non_tensor} for name, meta in metas.items()}
            )
        metadata = self.kv_retrieve_meta(keys, partition_id, create=True)
        partition = self._get_partition(partition_id)
        assert partition is not None
        partition.link_fields(metadata.global_indexes, public, source, fields)

    def mark_unlinked_public_clearing(self) -> tuple[BatchMeta, list[str]]:
        """Mark every ``PUBLIC_PARTITION`` key that no key links to as clearing.

        Selecting and marking in one controller call means a concurrent kv_link cannot link
        a row this clear deletes: kv_link refuses fields whose production status was zeroed.

        Returns:
            The marked rows' BatchMeta, for storage deletion and clear_meta, and their keys.
        """
        public = self._get_partition(PUBLIC_PARTITION)
        if public is None:
            return BatchMeta.empty(), []
        linked = set().union(*(p.linked_by.keys() for p in self.partitions.values()))
        keys = sorted(key for key, idx in public.keys_mapping.items() if idx not in linked)
        if not keys:
            return BatchMeta.empty(), []
        metadata = self.kv_retrieve_meta(keys, PUBLIC_PARTITION)
        self.mark_clearing(metadata.global_indexes, metadata.partition_ids)
        return metadata, keys

    def _init_zmq_socket(self):
        """Initialize ZMQ sockets for communication."""
        self.zmq_context = zmq.Context()
        # Before any socket is opened on it: libzmq applies MAX_SOCKETS at socket creation.
        socket_limit = self.zmq_context.get(zmq.SOCKET_LIMIT)
        self.zmq_context.set(zmq.MAX_SOCKETS, min(TQ_CONTROLLER_ZMQ_MAX_SOCKETS, socket_limit))
        self._node_ip = get_node_ip_address()

        while True:
            try:
                self._handshake_socket_port = get_free_port(ip=self._node_ip)
                self._request_handle_socket_port = get_free_port(ip=self._node_ip)

                self.handshake_socket = create_zmq_socket(
                    ctx=self.zmq_context,
                    socket_type=zmq.ROUTER,
                    ip=self._node_ip,
                )
                self.handshake_socket.bind(format_zmq_address(self._node_ip, self._handshake_socket_port))

                self.request_handle_socket = create_zmq_socket(
                    ctx=self.zmq_context,
                    socket_type=zmq.ROUTER,
                    ip=self._node_ip,
                )
                self.request_handle_socket.bind(format_zmq_address(self._node_ip, self._request_handle_socket_port))

                break
            except zmq.ZMQError:
                logger.warning(f"[{self.controller_id}]: Try to bind ZMQ sockets failed, retrying...")
                continue

        self.zmq_server_info = ZMQServerInfo(
            role=Role.CONTROLLER,
            id=self.controller_id,
            ip=self._node_ip,
            ports={
                "handshake_socket": self._handshake_socket_port,
                "request_handle_socket": self._request_handle_socket_port,
            },
        )

    def _wait_connection(self):
        """Wait for storage instances to complete handshake with retransmission support."""
        poller = zmq.Poller()
        poller.register(self.handshake_socket, zmq.POLLIN)

        logger.debug(f"Controller {self.controller_id} started waiting for storage connections...")

        while True:
            socks = dict(poller.poll(1000))

            if self.handshake_socket in socks:
                try:
                    messages = self.handshake_socket.recv_multipart(copy=False)
                    identity = messages.pop(0)
                    serialized_msg = messages
                    request_msg = ZMQMessage.deserialize(serialized_msg)

                    if request_msg.request_type == ZMQRequestType.HANDSHAKE:
                        storage_manager_id = request_msg.sender_id

                        # Always send ACK for HANDSHAKE
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.HANDSHAKE_ACK,
                            sender_id=self.controller_id,
                            body={},
                        ).serialize()
                        self.handshake_socket.send_multipart([identity, *response_msg])

                        # Track new connections
                        if storage_manager_id not in self._connected_storage_managers:
                            self._connected_storage_managers.add(storage_manager_id)
                            storage_manager_type = request_msg.body.get("storage_manager_type", "Unknown")
                            logger.debug(
                                f"[{self.controller_id}]: received handshake from "
                                f"storage manager {storage_manager_id} (type: {storage_manager_type}). "
                                f"Total connected: {len(self._connected_storage_managers)}"
                            )
                        else:
                            logger.debug(
                                f"[{self.controller_id}]: received duplicate handshake from "
                                f"storage manager {storage_manager_id}. Resending ACK."
                            )

                except Exception as e:
                    logger.error(f"[{self.controller_id}]: error processing handshake: {e}")

    def _start_process_handshake(self):
        """Start the handshake process thread."""
        self.wait_connection_thread = Thread(
            target=self._wait_connection,
            name="TransferQueueControllerWaitConnectionThread",
            daemon=True,
        )
        self.wait_connection_thread.start()

    def _start_process_request(self):
        """Start the request processing thread."""
        self.process_request_thread = Thread(
            target=self._process_request,
            name="TransferQueueControllerProcessRequestThread",
            daemon=True,
        )
        self.process_request_thread.start()

    def _process_request(self):
        """Keep the request loop running for the controller's lifetime.

        Anything that escapes the per-request guards below would otherwise terminate this
        thread and leave the controller permanently unable to answer requests. The ROUTER
        socket stays bound across a restart, so the loop simply resumes with the next
        queued request.
        """

        logger.info(f"[{self.controller_id}]: start processing requests...")

        perf_monitor = IntervalPerfMonitor(caller_name=self.controller_id)

        while True:
            try:
                self._process_request_loop(perf_monitor)
            except zmq.ContextTerminated:
                logger.info(f"[{self.controller_id}]: stopped processing requests (context terminated)")
                return
            except Exception as e:
                if self.request_handle_socket.closed:
                    logger.info(f"[{self.controller_id}]: stopped processing requests (socket closed)")
                    return
                logger.exception(f"[{self.controller_id}]: request loop raised {type(e).__name__}: {e}; restarting")
                # Keep a persistent failure from turning into a busy log flood.
                time.sleep(0.1)

    def _process_request_loop(self, perf_monitor: IntervalPerfMonitor) -> None:
        """Main request processing loop - adapted for partition-based operations.

        Every decoded request gets exactly one reply: either its handler's response or a
        REQUEST_ERROR explaining why it could not be served. Requesters block on recv
        until a reply arrives, so silently dropping a request would hang the caller.
        """

        while True:
            monitor = self._metrics if self._metrics is not None else perf_monitor

            messages = self.request_handle_socket.recv_multipart(copy=False)
            identity = messages.pop(0)
            serialized_msg = messages
            try:
                request_msg = ZMQMessage.deserialize(serialized_msg)
            except Exception as e:
                # ZMQMessageDecodeError carries the frame layout, which is what tells a
                # corrupt payload apart from shifted frame boundaries. The ROUTER identity
                # frame is prepended by the transport and survives such corruption, so the
                # requester can still get an error reply instead of hanging on recv.
                logger.error(
                    f"[{self.controller_id}]: undecodable request from "
                    f"identity={bytes(identity)!r}: {type(e).__name__}: {e}"
                )
                error_msg = self._make_error_response(
                    receiver_id=None,
                    message=f"undecodable request: {type(e).__name__}: {e}",
                )
                self.request_handle_socket.send_multipart([identity, *error_msg.serialize()])
                continue

            try:
                response_msg: ZMQMessage | None = self._handle_request(request_msg, monitor)
            except Exception as e:
                # A failing handler must not kill the loop (it used to terminate this
                # thread) nor hang the requester, so report the failure back.
                logger.exception(
                    f"[{self.controller_id}]: handler for request_type={request_msg.request_type} "
                    f"from sender={request_msg.sender_id} raised {type(e).__name__}: {e}"
                )
                response_msg = self._make_error_response(
                    receiver_id=request_msg.sender_id,
                    message=f"{request_msg.request_type} failed: {type(e).__name__}: {e}",
                )

            if response_msg is None:
                # No branch matched. Without this guard the reply below would send the
                # previous iteration's response to an unrelated requester.
                logger.error(
                    f"[{self.controller_id}]: no handler for request_type={request_msg.request_type} "
                    f"from sender={request_msg.sender_id}; request {request_msg.request_id}"
                )
                response_msg = self._make_error_response(
                    receiver_id=request_msg.sender_id,
                    message=f"no handler for request_type {request_msg.request_type}",
                )

            self.request_handle_socket.send_multipart([identity, *response_msg.serialize()])

    def _make_error_response(self, receiver_id: str | None, message: str) -> ZMQMessage:
        """Build a generic error reply so the requester raises instead of hanging on recv."""
        return ZMQMessage.create(
            request_type=ZMQRequestType.REQUEST_ERROR,
            sender_id=self.controller_id,
            receiver_id=receiver_id,
            body={"success": False, "message": message},
        )

    def _handle_request(self, request_msg: ZMQMessage, monitor: Any) -> ZMQMessage | None:
        """Build the response for a decoded request, or None if no handler matches.

        Whatever the matching handler raises propagates to the caller, which turns it
        into an error response.
        """
        handlers = {
            ZMQRequestType.GET_META: self._handle_get_meta_request,
            ZMQRequestType.NOTIFY_DATA_UPDATE: self._handle_notify_data_update_request,
            ZMQRequestType.GET_PARTITION_META: self._handle_get_partition_meta_request,
            ZMQRequestType.SET_CUSTOM_META: self._handle_set_custom_meta_request,
            ZMQRequestType.MARK_CLEARING: self._handle_mark_clearing_request,
            ZMQRequestType.CLEAR_META: self._handle_clear_meta_request,
            ZMQRequestType.CLEAR_PARTITION: self._handle_clear_partition_request,
            ZMQRequestType.GET_CONSUMPTION: self._handle_get_consumption_request,
            ZMQRequestType.RESET_CONSUMPTION: self._handle_reset_consumption_request,
            ZMQRequestType.GET_PRODUCTION: self._handle_get_production_request,
            ZMQRequestType.GET_LIST_PARTITIONS: self._handle_get_list_partitions_request,
            ZMQRequestType.KV_RETRIEVE_META: self._handle_kv_retrieve_meta_request,
            ZMQRequestType.KV_RETRIEVE_KEYS: self._handle_kv_retrieve_keys_request,
            ZMQRequestType.KV_LIST: self._handle_kv_list_request,
            ZMQRequestType.KV_DESCRIBE: self._handle_kv_describe_request,
            ZMQRequestType.KV_PREPARE_UPDATE: self._handle_kv_prepare_update_request,
            ZMQRequestType.VALIDATE_DUMP_SCHEMA: self._handle_validate_dump_schema_request,
            ZMQRequestType.KV_CLEAR_FIELDS: self._handle_kv_clear_fields_request,
            ZMQRequestType.KV_LINK: self._handle_kv_link_request,
            ZMQRequestType.MARK_UNLINKED_PUBLIC_CLEARING: self._handle_mark_unlinked_public_clearing_request,
            ZMQRequestType.SAVE_CONTROLLER_CHECKPOINT: self._handle_save_controller_checkpoint_request,
            ZMQRequestType.LOAD_CONTROLLER_CHECKPOINT: self._handle_load_controller_checkpoint_request,
        }
        handler = handlers.get(request_msg.request_type)
        if handler is None:
            return None

        checkpoint_requests = {
            ZMQRequestType.SAVE_CONTROLLER_CHECKPOINT,
            ZMQRequestType.LOAD_CONTROLLER_CHECKPOINT,
        }
        if request_msg.request_type in checkpoint_requests:
            return handler(request_msg)

        with monitor.measure(op_type=request_msg.request_type.value):
            return handler(request_msg)

    def _make_response(
        self,
        request_msg: ZMQMessage,
        response_type: ZMQRequestType,
        body: dict[str, Any],
    ) -> ZMQMessage:
        """Build a controller response addressed to the request sender."""
        return ZMQMessage.create(
            request_type=response_type,
            sender_id=self.controller_id,
            receiver_id=request_msg.sender_id,
            body=body,
        )

    def _handle_get_meta_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        metadata = self.get_metadata(
            data_fields=params["data_fields"],
            batch_size=params["batch_size"],
            partition_id=params["partition_id"],
            mode=params.get("mode", "fetch"),
            task_name=params.get("task_name"),
            sampling_config=params.get("sampling_config", {}),
        )
        return self._make_response(request_msg, ZMQRequestType.GET_META_RESPONSE, {"metadata": metadata})

    def _handle_notify_data_update_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        partition_id = params["partition_id"]
        global_indexes = params.get("global_indexes", [])
        if self._get_partition(partition_id) is None:
            return self._make_response(request_msg, ZMQRequestType.NOTIFY_DATA_UPDATE_ACK, {"success": False})

        if not self.update_production_status(
            partition_id=partition_id,
            global_indexes=global_indexes,
            field_schema=params.get("field_schema", {}),
            custom_backend_meta=params.get("custom_backend_meta", {}),
            ready=params.get("ready", True),
        ):
            raise RuntimeError(f"Failed to record data for partition {partition_id!r}; see the controller log")
        if self._metrics is not None:
            self._metrics.record_samples("NOTIFY_DATA_UPDATE", len(global_indexes))
        logger.debug(f"[{self.controller_id}]: Updated production status for partition {partition_id}")

        return self._make_response(request_msg, ZMQRequestType.NOTIFY_DATA_UPDATE_ACK, {"success": True})

    def _handle_get_partition_meta_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        partition_id = request_msg.body["partition_id"]
        partition = self._get_partition(partition_id)
        if partition is None:
            metadata = None
        else:
            metadata = self.get_metadata(
                data_fields=list(partition.field_name_mapping.keys()),
                partition_id=partition_id,
                mode="force_fetch",
            )
        return self._make_response(
            request_msg,
            ZMQRequestType.GET_PARTITION_META_RESPONSE,
            {"metadata": metadata},
        )

    def _handle_set_custom_meta_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        self.set_custom_meta(partition_custom_meta=request_msg.body["partition_custom_meta"])
        return self._make_response(
            request_msg,
            ZMQRequestType.SET_CUSTOM_META_RESPONSE,
            {"message": "Successfully set custom_meta"},
        )

    def _handle_mark_clearing_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        self.mark_clearing(params["global_indexes"], params["partition_ids"])
        return self._make_response(
            request_msg,
            ZMQRequestType.MARK_CLEARING_RESPONSE,
            {"message": "Mark clearing completed"},
        )

    def _handle_clear_meta_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        global_indexes = params["global_indexes"]
        self.clear_meta(global_indexes, params["partition_ids"])
        if self._metrics is not None:
            self._metrics.record_samples("CLEAR_META", len(global_indexes))
        return self._make_response(
            request_msg,
            ZMQRequestType.CLEAR_META_RESPONSE,
            {"message": f"Clear samples operation completed by controller {self.controller_id}"},
        )

    def _handle_clear_partition_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        partition_id = request_msg.body["partition_id"]
        self.clear_partition(partition_id)
        return self._make_response(
            request_msg,
            ZMQRequestType.CLEAR_PARTITION_RESPONSE,
            {"message": f"Clear partition operation completed by controller {self.controller_id}"},
        )

    def _handle_get_consumption_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        global_index, consumption_status = self.get_consumption_status(params["partition_id"], params["task_name"])
        sample_filter = params.get("sample_filter")  # TODO: DEPRECATED in future
        if sample_filter and consumption_status is not None:
            consumption_status = consumption_status[sample_filter]  # TODO: DEPRECATED in future

        return self._make_response(
            request_msg,
            ZMQRequestType.CONSUMPTION_RESPONSE,
            {
                "partition_id": params["partition_id"],
                "global_index": global_index,
                "consumption_status": consumption_status,
            },
        )

    def _handle_reset_consumption_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        partition_id = params["partition_id"]
        try:
            self.reset_consumption(partition_id, params.get("task_name"))
            success = True
            message = f"Consumption reset for partition {partition_id}"
        except Exception as e:
            success = False
            message = str(e)
        return self._make_response(
            request_msg,
            ZMQRequestType.RESET_CONSUMPTION_RESPONSE,
            {"partition_id": partition_id, "success": success, "message": message},
        )

    def _handle_get_production_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        global_index, production_status = self.get_production_status(params["partition_id"], params["data_fields"])
        return self._make_response(
            request_msg,
            ZMQRequestType.PRODUCTION_RESPONSE,
            {
                "partition_id": params["partition_id"],
                "global_index": global_index,
                "production_status": production_status,
            },
        )

    def _handle_get_list_partitions_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        return self._make_response(
            request_msg,
            ZMQRequestType.LIST_PARTITIONS_RESPONSE,
            {"partition_ids": self.list_partitions()},
        )

    def _handle_kv_retrieve_meta_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        metadata = self.kv_retrieve_meta(
            keys=params["keys"],
            partition_id=params["partition_id"],
            create=params["create"],
        )
        return self._make_response(
            request_msg,
            ZMQRequestType.KV_RETRIEVE_META_RESPONSE,
            {"metadata": metadata},
        )

    def _handle_kv_retrieve_keys_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        keys = self.kv_retrieve_keys(
            global_indexes=params["global_indexes"],
            partition_id=params["partition_id"],
        )
        return self._make_response(request_msg, ZMQRequestType.KV_RETRIEVE_KEYS_RESPONSE, {"keys": keys})

    def _handle_kv_describe_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        description = self.kv_describe(params["keys"], params["partition_id"])
        return self._make_response(request_msg, ZMQRequestType.KV_DESCRIBE_RESPONSE, description)

    def _handle_kv_prepare_update_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        metadata, merge_plan = self.kv_prepare_update(params["keys"], params["partition_id"], params["fields"])
        return self._make_response(
            request_msg, ZMQRequestType.KV_PREPARE_UPDATE_RESPONSE, {"metadata": metadata, "merge_plan": merge_plan}
        )

    def _handle_kv_link_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        self.kv_link(params["keys"], params["partition_id"], params["public_key"], params["fields"])
        return self._make_response(request_msg, ZMQRequestType.KV_LINK_RESPONSE, {})

    def _handle_mark_unlinked_public_clearing_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        metadata, keys = self.mark_unlinked_public_clearing()
        return self._make_response(
            request_msg, ZMQRequestType.MARK_UNLINKED_PUBLIC_CLEARING_RESPONSE, {"metadata": metadata, "keys": keys}
        )

    def _handle_kv_clear_fields_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        metadata = self.kv_clear_fields(
            keys=params["keys"], partition_id=params["partition_id"], fields=params["fields"]
        )
        return self._make_response(request_msg, ZMQRequestType.KV_CLEAR_FIELDS_RESPONSE, {"metadata": metadata})

    def _handle_kv_list_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        requested_partition_id = request_msg.body["partition_id"]
        partition_ids = list(self.partitions.keys()) if requested_partition_id is None else [requested_partition_id]
        message = "success"
        partition_info = {}
        for partition_id in partition_ids:
            partition = self._get_partition(partition_id)
            if partition is None:
                message = f"partition {partition_id} does not exist"
                continue
            partition_info[partition_id] = {
                key: partition.custom_meta.get(index, {}) for key, index in partition.keys_mapping.items()
            }

        return self._make_response(
            request_msg,
            ZMQRequestType.KV_LIST_RESPONSE,
            {"partition_info": partition_info, "message": message},
        )

    def _handle_save_controller_checkpoint_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        self.save_checkpoint(request_msg.body["path"])
        return self._make_response(
            request_msg,
            ZMQRequestType.SAVE_CONTROLLER_CHECKPOINT_RESPONSE,
            {"success": True},
        )

    def _handle_load_controller_checkpoint_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        self.load_checkpoint(request_msg.body["path"])
        return self._make_response(
            request_msg,
            ZMQRequestType.LOAD_CONTROLLER_CHECKPOINT_RESPONSE,
            {"success": True},
        )

    def _handle_validate_dump_schema_request(self, request_msg: ZMQMessage) -> ZMQMessage:
        params = request_msg.body
        partition = self._get_partition(params["partition_id"])
        if partition is not None:
            partition.validate_field_schema(params["field_schema"])
        return self._make_response(request_msg, ZMQRequestType.VALIDATE_DUMP_SCHEMA_RESPONSE, {"success": True})

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get ZMQ server connection information."""
        return self.zmq_server_info

    def store_config(self, conf: DictConfig) -> None:
        """Store the global config of TransferQueue."""
        self.tq_config = conf
        if conf.backend.storage_backend == "SimpleStorage":
            self.index_manager.set_num_storage_units(int(conf.backend.SimpleStorage.num_data_storage_units))

    def get_config(self) -> DictConfig:
        """Retrieve the global config of TransferQueue."""
        return self.tq_config

    def save_checkpoint(self, path: str) -> None:
        """Serialize controller state directly to a file.

        Writes in-process to avoid transmitting the payload back over the
        Ray object store.

        Args:
            path: Absolute path for the output .pkl file.

        Raises:
            Exception: If serialization or file I/O fails.
        """
        try:
            state = {
                "controller_id": self.controller_id,
                "partitions": {pid: p.to_snapshot() for pid, p in self.partitions.items()},
                "index_manager": {
                    "partition_to_indexes": dict(copy.deepcopy(self.index_manager.partition_to_indexes)),
                    "reusable_indexes": list(self.index_manager.reusable_indexes),
                    "global_index_counter": self.index_manager.global_index_counter,
                    "allocated_indexes": set(self.index_manager.allocated_indexes),
                },
                "sampler": self.sampler.save_checkpoint(),
            }
            with open(path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"[{self.controller_id}]: dumped to {path}")
        except Exception as e:
            raise RuntimeError(f"[{self.controller_id}]: save checkpoint failed: {e}") from e

    def load_checkpoint(self, path: str) -> None:
        """Restore controller state directly from a file.

        Args:
            path: Absolute path to a .pkl file previously written by save_checkpoint().

        Raises:
            Exception: If deserialization or file I/O fails.
        """
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)

            self.controller_id = state["controller_id"]
            self.partitions = state["partitions"]

            im = state["index_manager"]
            self.index_manager.partition_to_indexes = defaultdict(set, im["partition_to_indexes"])
            self.index_manager.global_index_counter = im["global_index_counter"]
            self.index_manager.allocated_indexes = im["allocated_indexes"]
            self.index_manager.set_num_storage_units(self.index_manager.num_storage_units, im["reusable_indexes"])

            self.sampler.load_checkpoint(state["sampler"])

            logger.info(f"[{self.controller_id}]: restored from {path}")
        except Exception as e:
            raise RuntimeError(f"[{self.controller_id}]: load checkpoint failed: {e}") from e

    def register_sampler(
        self,
        sampler: BaseSampler | type[BaseSampler] = SequentialSampler,
    ) -> None:
        """
        Register a sampler instance or subclass after the controller is initialized.

        Args:
            sampler: Sampler instance or sampler class to use for data sampling.
                    - If a BaseSampler instance is provided, it will be used directly
                    - If a BaseSampler subclass is provided, it will be instantiated
                    - Defaults to SequentialSampler for simple sequential sampling
                    - Example: sampler=GRPOGroupNSampler() (instance)
                    - Example: sampler=SequentialSampler (class)
        """
        if isinstance(sampler, BaseSampler):
            self.sampler = sampler
        elif isinstance(sampler, type) and issubclass(sampler, BaseSampler):
            self.sampler = sampler()
        else:
            raise TypeError(
                f"sampler {getattr(sampler, '__name__', repr(sampler))} must be an instance or subclass of BaseSampler"
            )

    # ==================== Metrics API ====================

    def _build_metrics_snapshot(self) -> dict:
        """Build a plain-dict snapshot of controller state for the metrics exporter.

        The snapshot contains only primitive / dict values — no references to
        live controller objects — so the metrics thread can read it safely.
        """
        partitions_data: dict = {}
        for pid, partition in list(self.partitions.items()):
            try:
                stats = partition.get_statistics()
                # Build per-task production statistics from registered tasks.
                # Always emit an "all" entry so production progress is visible
                # even before any consumer tasks are registered.
                production_statistics: dict = {}
                if "production_progress" in stats:
                    production_statistics["all"] = {
                        "production_progress": stats["production_progress"],
                    }
                for task_name in stats.get("registered_tasks", []):
                    production_statistics[task_name] = {
                        "production_progress": stats.get("production_progress", 0),
                    }
                partitions_data[pid] = {
                    "total_samples_num": stats["total_samples_num"],
                    "production_statistics": production_statistics,
                    "consumption_statistics": {
                        task_name: {"consumption_progress": cstats.get("consumption_progress", 0)}
                        for task_name, cstats in stats.get("consumption_statistics", {}).items()
                    },
                }
            except Exception:
                pass

        return {
            "partitions": partitions_data,
            "global_index_allocated": len(self.index_manager.allocated_indexes),
            "global_index_reusable": len(self.index_manager.reusable_indexes),
        }

    def _push_metrics_snapshot(self) -> None:
        """Push a fresh metrics snapshot to the exporter (called from controller threads)."""
        if self._metrics is None:
            return
        try:
            snapshot = self._build_metrics_snapshot()
            self._metrics.update_controller_snapshot(snapshot)
        except Exception as e:
            logger.debug(f"[{self.controller_id}]: Failed to push metrics snapshot: {e}")

    def start_metrics(self, port: int = 0) -> str:
        """Initialize and start the Prometheus metrics exporter.

        This creates a ``TQMetricsExporter``, starts an HTTP ``/metrics``
        endpoint and a background collection thread.  The method is safe
        to call multiple times -- subsequent calls return the existing endpoint.

        Args:
            port: HTTP port for the /metrics endpoint (0 = auto-assign).

        Returns:
            The metrics endpoint address in ``host:port`` format.
        """
        if self._metrics is not None:
            return self._metrics_endpoint
        from transfer_queue.metrics import TQMetricsExporter

        # Lend the controller's context rather than let the exporter build a second one.
        self._metrics = TQMetricsExporter(zmq_context=self.zmq_context)
        self._metrics_endpoint = self._metrics.start(node_ip=self._node_ip, port=port)
        self._metrics_snapshot_thread = Thread(
            target=self._metrics_snapshot_loop,
            name="TQMetricsSnapshotThread",
            daemon=True,
        )
        self._metrics_snapshot_thread.start()
        logger.info(f"[{self.controller_id}]: Prometheus metrics exporter started on {self._metrics_endpoint}")
        return self._metrics_endpoint

    def _metrics_snapshot_loop(self) -> None:
        """Periodically push a metrics snapshot to the exporter."""
        from transfer_queue.metrics import TQ_METRICS_COLLECT_INTERVAL

        while True:
            self._push_metrics_snapshot()
            time.sleep(TQ_METRICS_COLLECT_INTERVAL)

    def register_storage_units_for_metrics(self, storage_unit_infos: dict) -> None:
        """Register storage unit ZMQ endpoints so the metrics collector can query them.

        Args:
            storage_unit_infos: Mapping of storage unit names/IDs to ``ZMQServerInfo``.
        """
        if self._metrics is not None:
            self._metrics.register_storage_units(storage_unit_infos)
