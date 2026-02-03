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
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from operator import itemgetter
from threading import Lock, Thread
from typing import Any, Optional
from uuid import uuid4

import ray
import torch
import zmq
from ray.util import get_node_ip_address
from torch import Tensor

from transfer_queue.metadata import (
    BatchMeta,
    FieldMeta,
    SampleMeta,
)
from transfer_queue.sampler import BaseSampler, SequentialSampler
from transfer_queue.utils.enum_utils import ProductionStatus, TransferQueueRole
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    get_free_port,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler (for Ray Actor subprocess)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

TQ_CONTROLLER_GET_METADATA_TIMEOUT = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_TIMEOUT", 1))
TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL = int(os.environ.get("TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL", 5))

# Sample pre-allocation for StreamingDataLoader compatibility.
# By pre-allocating sample indices (typically global_batch_size), consumers can accurately
# determine consumption status even before producers have generated the samples.
TQ_PRE_ALLOC_SAMPLE_NUM = int(os.environ.get("TQ_PRE_ALLOC_SAMPLE_NUM", 1))


class PartitionIndexManager:
    """
    Manages the mapping relationship between partitions and global indexes,
    responsible for index allocation and reuse.
    """

    def __init__(self):
        # Records the set of global_indexes used by each partition
        self.partition_to_indexes = defaultdict(set)

        # Reusable global_index pool - stored using list
        self.reusable_indexes = []

        # Global index counter for allocating new indexes
        self.global_index_counter = 0

        # Track all active indexes
        self.allocated_indexes = set()

    def allocate_indexes(self, partition_id, count=1) -> list[int]:
        """
        Allocate global_indexes for the specified partition.
        Prioritizes obtaining from reusable pool, allocates new indexes when insufficient.

        Args:
            partition_id: Partition ID
            count: Number of indexes needed

        Returns:
            list: List of allocated global_indexes
        """
        if count <= 0:
            raise ValueError(f"Number of indexes needed must be larger than 0, but got {count}")
        indexes = []

        # Get indexes from reusable pool
        if self.reusable_indexes:
            # Calculate number of indexes needed from reusable pool
            num_reuse = min(count, len(self.reusable_indexes))

            # Use slice operation to get multiple elements at once (FIFO principle)
            indexes.extend(self.reusable_indexes[:num_reuse])
            del self.reusable_indexes[:num_reuse]

        # If reusable pool doesn't have enough indexes, allocate new ones
        if len(indexes) < count:
            # Ensure newly allocated indexes don't conflict with existing ones
            needed = count - len(indexes)
            # Batch allocate consecutive index ranges
            start_index = self.global_index_counter
            end_index = start_index + needed

            # Directly generate consecutive index list
            new_indexes = list(range(start_index, end_index))

            # Batch update status
            self.allocated_indexes.update(new_indexes)
            self.global_index_counter = end_index

            indexes.extend(new_indexes)

        # Record partition-index relationship
        self.partition_to_indexes[partition_id].update(indexes)

        return indexes

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
            self.reusable_indexes.extend(indexes)

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
        self.reusable_indexes.extend(indexes_to_release)
        self.allocated_indexes.difference_update(indexes_to_release)

        # If partition has no more indexes, remove it from the mapping
        if not partition_indexes:
            self.partition_to_indexes.pop(partition_id, None)

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

    production_status: Tensor = torch.zeros(TQ_PRE_ALLOC_SAMPLE_NUM, 1, dtype=torch.int8)

    # Consumption status per task - task_name -> consumption_tensor
    # Each tensor tracks which samples have been consumed by that task
    consumption_status: dict[str, Tensor] = field(default_factory=dict)

    # Sample metadata
    global_indexes: set[int] = field(
        default_factory=set
    )  # set of global indexes that have been added to this partition

    pre_allocated_global_indexes: set[int] = field(
        default_factory=set
    )  # set of global indexes that pre-allocated, but not active in this partition

    # Metadata
    field_name_mapping: dict[str, int] = field(default_factory=dict)  # field_name -> column_index
    field_dtypes: dict[int, dict[str, Any]] = field(default_factory=dict)  # global_idx -> {field: dtype}
    field_shapes: dict[int, dict[str, Any]] = field(default_factory=dict)  # global_idx -> {field: shape}
    field_custom_backend_meta: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )  # global_idx -> {field: custom_backend_meta}
    # User-defined metadata that may not apply to field level
    custom_meta: dict[int, dict[str, Any]] = field(default_factory=dict)  # global_idx -> {}

    # Threading lock for concurrency control; only for preventing mask operation error when expanding production_status.
    # No need to strictly lock for every read/write operation since freshness is not critical.
    data_status_lock: Lock = field(default_factory=Lock)

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

        with self.data_status_lock:
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
        dtypes: Optional[dict[int, dict[str, Any]]],
        shapes: Optional[dict[int, dict[str, Any]]],
        custom_backend_meta: Optional[dict[int, dict[str, Any]]] = None,
    ) -> bool:
        """
        Update production status for specific samples and fields.
        Handles dynamic expansion of both samples and fields.

        Args:
            global_indices: List of sample indices to update
            field_names: List of field names to mark as produced
            dtypes: Optional per-sample field dtype information
            shapes: Optional per-sample field shape information
            custom_backend_meta: Optional per-sample per-field
                custom metadata provided by storage backend

        Returns:
            True if update was successful, False on error
        """
        try:
            # Determine required capacity
            max_sample_idx = max(global_indices) if global_indices else -1
            required_samples = max_sample_idx + 1

            # Ensure we have enough rows
            with self.data_status_lock:
                self.ensure_samples_capacity(required_samples)

            # Register new fields if needed
            new_fields = [field for field in field_names if field not in self.field_name_mapping]
            if new_fields:
                # Add new fields to mapping
                for field in new_fields:
                    self.field_name_mapping[field] = len(self.field_name_mapping)

                required_fields = len(self.field_name_mapping)
                with self.data_status_lock:
                    self.ensure_fields_capacity(required_fields)

            # Update production status
            if self.production_status is not None and global_indices and field_names:
                field_indices = [self.field_name_mapping.get(field) for field in field_names]
                self.production_status[torch.tensor(global_indices)[:, None], torch.tensor(field_indices)] = 1

            # Update field metadata
            self._update_field_metadata(global_indices, dtypes, shapes, custom_backend_meta)

            # Save these global_indexes
            self.global_indexes.update(global_indices)

            return True

        except Exception as e:
            logger.error(f"Error updating production status for partition {self.partition_id}: {e}")
            return False

    def _update_field_metadata(
        self,
        global_indices: list[int],
        dtypes: Optional[dict[int, dict[str, Any]]],
        shapes: Optional[dict[int, dict[str, Any]]],
        custom_backend_meta: Optional[dict[int, dict[str, Any]]],
    ):
        """Update field dtype and shape metadata."""
        if not global_indices:
            return

        # Validate lengths only for provided mappings
        if dtypes and len(global_indices) != len(dtypes):
            raise ValueError(f"`global_indices` {len(global_indices)} and `dtypes` {len(dtypes)} length mismatch.")
        if shapes and len(global_indices) != len(shapes):
            raise ValueError(f"`global_indices` {len(global_indices)} and `shapes` {len(shapes)} length mismatch.")
        if custom_backend_meta and len(global_indices) != len(custom_backend_meta):
            raise ValueError(
                f"`global_indices` {len(global_indices)} and `custom_backend_meta` {len(custom_backend_meta)} "
                f"length mismatch."
            )

        # Extract values for each provided mapping; if a mapping is absent, use Nones
        if dtypes:
            dtype_value = itemgetter(*global_indices)(dtypes)
            if not isinstance(dtype_value, tuple):
                dtype_value = (dtype_value,)
        else:
            dtype_value = tuple([None] * len(global_indices))

        if shapes:
            shape_value = itemgetter(*global_indices)(shapes)
            if not isinstance(shape_value, tuple):
                shape_value = (shape_value,)
        else:
            shape_value = tuple([None] * len(global_indices))

        if custom_backend_meta:
            custom_backend_meta_value = itemgetter(*global_indices)(custom_backend_meta)
            if not isinstance(custom_backend_meta_value, tuple):
                custom_backend_meta_value = (custom_backend_meta_value,)
        else:
            custom_backend_meta_value = tuple([None] * len(global_indices))

        for i, global_idx in enumerate(global_indices):
            # Only create and update dtype mapping if a dtype value was provided
            if dtype_value[i] is not None:
                if global_idx not in self.field_dtypes:
                    self.field_dtypes[global_idx] = {}
                self.field_dtypes[global_idx].update(dtype_value[i])

            # Only create and update shape mapping if a shape value was provided
            if shape_value[i] is not None:
                if global_idx not in self.field_shapes:
                    self.field_shapes[global_idx] = {}
                self.field_shapes[global_idx].update(shape_value[i])

            # Only create and update custom_backend_meta mapping if a custom_backend_meta value was provided
            if custom_backend_meta_value[i] is not None:
                if global_idx not in self.field_custom_backend_meta:
                    self.field_custom_backend_meta[global_idx] = {}
                self.field_custom_backend_meta[global_idx].update(custom_backend_meta_value[i])

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
        consumption_status = self.consumption_status[task_name]

        partition_global_index = torch.tensor(
            sorted(self.global_indexes | self.pre_allocated_global_indexes), dtype=torch.long
        )

        if mask:
            consumption_status = consumption_status[partition_global_index]

        return partition_global_index, consumption_status

    def reset_consumption(self, task_name: Optional[str] = None):
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
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
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

        with self.data_status_lock:
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

    def get_field_dtype(self, global_index: int, field_name: str) -> Optional[Any]:
        """Get dtype for a specific sample and field."""
        return self.field_dtypes.get(global_index, {}).get(field_name)

    def get_field_shape(self, global_index: int, field_name: str) -> Optional[Any]:
        """Get shape for a specific sample and field."""
        return self.field_shapes.get(global_index, {}).get(field_name)

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
        return {
            idx: {f: v for f, v in self.field_custom_backend_meta[idx].items() if f in field_names}
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
            DataPartitionStatus object without threading.Lock()
        """

        def _perform_copy():
            cls = self.__class__
            snapshot = cls.__new__(cls)

            for name, value in self.__dict__.items():
                if name == "data_status_lock":
                    continue

                if isinstance(value, Tensor):
                    new_val = value.clone().detach()
                else:
                    new_val = copy.deepcopy(value)

                setattr(snapshot, name, new_val)
            return snapshot

        lock_obj = getattr(self, "data_status_lock", None)

        if lock_obj:
            with lock_obj:
                return _perform_copy()
        else:
            return _perform_copy()

    def clear_data(self, indexes_to_release: list[int], clear_consumption: bool = True):
        """Clear all production and optionally consumption data for given global_indexes."""
        try:
            if self.production_status is not None:
                self.production_status[indexes_to_release, :] = 0

            if clear_consumption:
                for consumption_tensor in self.consumption_status.values():
                    consumption_tensor[indexes_to_release] = 0

            self.global_indexes.difference_update(indexes_to_release)
            for idx in indexes_to_release:
                self.field_dtypes.pop(idx, None)
                self.field_shapes.pop(idx, None)
                self.field_custom_backend_meta.pop(idx, None)
                self.custom_meta.pop(idx, None)

        except Exception as e:
            logger.error(
                f"Error clearing data for partition {self.partition_id}: {e}. "
                f"Attempted to clear global_indexes: {indexes_to_release}"
            )


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

        # Initialize ZMQ sockets for communication
        self._init_zmq_socket()

        # Partition management
        self.partitions: dict[str, DataPartitionStatus] = {}  # partition_id -> DataPartitionStatus

        # Partition-GlobalIndex management
        self.index_manager = PartitionIndexManager()  # partition_id -> global_indexes

        # Connected storage managers tracking
        self._connected_storage_managers: set[str] = set()

        # Start background processing threads
        self._start_process_handshake()
        self._start_process_update_data_status()
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
        if partition_id in self.partitions:
            logger.warning(f"Partition {partition_id} already exists")
            return False

        self.partitions[partition_id] = DataPartitionStatus(partition_id=partition_id)

        # Pre-allocate global indexes for consumer consumption tracking
        global_indexes = self.index_manager.allocate_indexes(partition_id, count=TQ_PRE_ALLOC_SAMPLE_NUM)
        self.partitions[partition_id].register_pre_allocated_indexes(global_indexes)

        logger.info(f"Created partition {partition_id} with {TQ_PRE_ALLOC_SAMPLE_NUM} pre-allocated indexes")
        return True

    def _get_partition(self, partition_id: str) -> Optional[DataPartitionStatus]:
        """
        Get partition status information.

        Args:
            partition_id: ID of the partition to retrieve

        Returns:
            DataPartitionStatus object if partition exists, None otherwise
        """
        return self.partitions.get(partition_id)

    def get_partition_snapshot(self, partition_id: str) -> Optional[DataPartitionStatus]:
        """
        Get a copy of partition status information, without threading.Lock().

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

    def get_partition_index_range(self, partition: DataPartitionStatus) -> list[int]:
        """
        Get all indexes for a specific partition.

        Args:
            partition: Partition identifier

        Returns:
            List of indexes allocated to the partition
        """
        return self.index_manager.get_indexes_for_partition(partition)

    # ==================== Data Production API ====================

    # TODO: Modify dtypes & shapes to be required
    def update_production_status(
        self,
        partition_id: str,
        global_indexes: list[int],
        field_names: list[str],
        dtypes: Optional[dict[int, dict[str, Any]]],
        shapes: Optional[dict[int, dict[str, Any]]],
        custom_backend_meta: Optional[dict[int, dict[str, Any]]] = None,
    ) -> bool:
        """
        Update production status for specific samples and fields in a partition.
        Delegates to the partition's own update_production_status method.

        Args:
            partition_id: ID of the partition
            global_indexes: List of sample indices to update
            field_names: List of field names to mark as produced
            dtypes: Optional per-sample field dtype information
            shapes: Optional per-sample field shape information
            custom_backend_meta: Optional custom backend metadata

        Returns:
            True if update was successful, False otherwise
        """
        partition = self._get_partition(partition_id)
        if not partition:
            logger.error(f"Partition {partition_id} not found")
            return False

        success = partition.update_production_status(global_indexes, field_names, dtypes, shapes, custom_backend_meta)
        if success:
            logger.debug(
                f"[{self.controller_id}]: Updated production status for partition {partition_id}: "
                f"samples={global_indexes}, fields={field_names}"
            )
        return success

    # ==================== Data Consumption API ====================

    def get_consumption_status(self, partition_id: str, task_name: str) -> tuple[Optional[Tensor], Optional[Tensor]]:
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

    def get_production_status(
        self, partition_id: str, data_fields: list[str]
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
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
        sampling_config: Optional[dict[str, Any]] = None,
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
            if data_fields:
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

            else:
                # TODO: separate this "clear" related logic into a separated mode
                # clear metadata call passes empty data_fields
                batch_global_indexes = self.index_manager.get_indexes_for_partition(partition_id)
            return self.generate_batch_meta(partition_id, batch_global_indexes, data_fields, mode)

        assert task_name is not None
        if mode == "fetch":
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
                        logger.debug(
                            f"[{self.controller_id}]: Not enough data for task {task_name} in partition {partition_id}."
                            f" Required: {batch_size}, Available: {len(ready_for_consume_indexes)}."
                            f" Returning None due to polling mode."
                        )
                        return BatchMeta.empty()
                    if time.time() - start_time > TQ_CONTROLLER_GET_METADATA_TIMEOUT:
                        raise TimeoutError(
                            f"Timeout while waiting for sufficient data for task {task_name}. "
                            f"Required: {batch_size}, Available: {len(ready_for_consume_indexes)}"
                        )
                    logger.warning(
                        f"[{self.controller_id}]: Insufficient data for task {task_name}. Required: {batch_size} "
                        f"samples with fields {data_fields} in partition {partition_id}, but only have "
                        f"{len(ready_for_consume_indexes)} samples meeting the criteria. "
                        f"Retrying in {TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL}s..."
                    )
                    time.sleep(TQ_CONTROLLER_GET_METADATA_CHECK_INTERVAL)
                else:
                    break

            batch_global_indexes, consumed_indexes = self.sampler(
                ready_for_consume_indexes,
                batch_size,
                **(sampling_config or {}),
                **kwargs,
            )

            # Check if we got valid results from the sampler
            if len(batch_global_indexes) != batch_size:
                raise RuntimeError(
                    f"Sampler returned insufficient samples. Please check the sampler logic. "
                    f"Expected: {batch_size}, before sampling: {len(ready_for_consume_indexes)}, "
                    f"after sampling: {len(batch_global_indexes)}"
                )

        elif mode == "force_fetch":
            batch_global_indexes = self.index_manager.get_indexes_for_partition(partition_id)
            consumed_indexes = []

        # Package into metadata
        metadata = self.generate_batch_meta(partition_id, batch_global_indexes, data_fields, mode)

        # Mark samples as consumed if in fetch mode
        if mode == "fetch" and consumed_indexes:
            partition = self.partitions[partition_id]
            partition.mark_consumed(task_name, consumed_indexes)

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

        # Generate sample metadata
        samples = []
        for global_index in batch_global_indexes:
            fields = {}
            for field_name in data_fields:
                # Determine production status
                if mode == "fetch":
                    production_status = ProductionStatus.READY_FOR_CONSUME
                    dtype = partition.get_field_dtype(global_index, field_name)
                    shape = partition.get_field_shape(global_index, field_name)
                elif mode == "insert":
                    production_status = ProductionStatus.NOT_PRODUCED
                    dtype = None
                    shape = None
                elif mode == "force_fetch":
                    field_index = partition.field_name_mapping.get(field_name)
                    if (
                        field_index is not None
                        and partition.production_status is not None
                        and partition.production_status[global_index, field_index] == 1
                    ):
                        production_status = ProductionStatus.NOT_PRODUCED
                        dtype = partition.get_field_dtype(global_index, field_name)
                        shape = partition.get_field_shape(global_index, field_name)
                    else:
                        production_status = ProductionStatus.NOT_PRODUCED
                        dtype = None
                        shape = None

                fields[field_name] = FieldMeta(
                    name=field_name,
                    dtype=dtype,
                    shape=shape,
                    production_status=production_status,
                )

            sample = SampleMeta(
                partition_id=partition_id,
                global_index=global_index,
                fields=fields,
            )
            samples.append(sample)

        custom_meta = partition.get_custom_meta(batch_global_indexes)
        custom_backend_meta = partition.get_field_custom_backend_meta(batch_global_indexes, data_fields)

        batch_meta = BatchMeta(samples=samples)
        batch_meta.update_custom_meta(custom_meta)
        batch_meta._custom_backend_meta.update(custom_backend_meta)
        return batch_meta

    def clear_partition(self, partition_id: str, clear_consumption: bool = True):
        """
        Clear data for a specific partition (delete the whole partition).

        Args:
            partition_id: ID of the partition to clear
            clear_consumption: Whether to also clear consumption status
        """

        logger.debug(f"[{self.controller_id}]: clearing metadata in partition {partition_id}")

        partition = self._get_partition(partition_id)
        if not partition:
            raise ValueError(f"Partition {partition_id} not found")

        global_indexes_range = list(self.index_manager.get_indexes_for_partition(partition_id))
        partition.clear_data(global_indexes_range, clear_consumption)
        self.index_manager.release_partition(partition_id)
        self.partitions.pop(partition_id)
        self.sampler.clear_cache(partition_id)

    def reset_consumption(self, partition_id: str, task_name: Optional[str] = None):
        """
        Reset consumption status for a partition without clearing the actual data.

        This allows the same data to be re-consumed, useful for debugging scenarios
        where the same rollout data needs to be trained multiple times.
        Args:
            partition_id: ID of the partition to reset consumption for
            task_name: Name of the task to reset. If None, resets all tasks.
        Raises:
            ValueError: If partition not found
        """
        logger.debug(f"[{self.controller_id}]: Resetting consumption for partition {partition_id}, task={task_name}")
        partition = self._get_partition(partition_id)
        if not partition:
            raise ValueError(f"Partition {partition_id} not found")
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
                raise ValueError(f"Partition {partition_id} not found")

            global_indexes_to_clear = [idx for _, idx in group]
            if not set(global_indexes_to_clear).issubset(partition.global_indexes):
                raise ValueError(
                    f"Some global_indexes to clear do not exist in partition {partition_id}. "
                    f"Target: {global_indexes_to_clear}, Existing: {partition.global_indexes}"
                )

            # Clear data from partition
            partition.clear_data(global_indexes_to_clear, clear_consumption)

            # Release the specific indexes from index manager
            self.index_manager.release_indexes(partition_id, global_indexes_to_clear)

    def _init_zmq_socket(self):
        """Initialize ZMQ sockets for communication."""
        self.zmq_context = zmq.Context()
        self._node_ip = get_node_ip_address()

        while True:
            try:
                self._handshake_socket_port = get_free_port()
                self._request_handle_socket_port = get_free_port()
                self._data_status_update_socket_port = get_free_port()

                self.handshake_socket = create_zmq_socket(
                    ctx=self.zmq_context,
                    socket_type=zmq.ROUTER,
                )
                self.handshake_socket.bind(f"tcp://{self._node_ip}:{self._handshake_socket_port}")

                self.request_handle_socket = create_zmq_socket(
                    ctx=self.zmq_context,
                    socket_type=zmq.ROUTER,
                )
                self.request_handle_socket.bind(f"tcp://{self._node_ip}:{self._request_handle_socket_port}")

                self.data_status_update_socket = create_zmq_socket(
                    ctx=self.zmq_context,
                    socket_type=zmq.ROUTER,
                )
                self.data_status_update_socket.bind(f"tcp://{self._node_ip}:{self._data_status_update_socket_port}")

                break
            except zmq.ZMQError:
                logger.warning(f"[{self.controller_id}]: Try to bind ZMQ sockets failed, retrying...")
                continue

        self.zmq_server_info = ZMQServerInfo(
            role=TransferQueueRole.CONTROLLER,
            id=self.controller_id,
            ip=self._node_ip,
            ports={
                "handshake_socket": self._handshake_socket_port,
                "request_handle_socket": self._request_handle_socket_port,
                "data_status_update_socket": self._data_status_update_socket_port,
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
                    messages = self.handshake_socket.recv_multipart()
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

    def _start_process_update_data_status(self):
        """Start the data status update processing thread."""
        self.process_update_data_status_thread = Thread(
            target=self._update_data_status,
            name="TransferQueueControllerProcessUpdateDataStatusThread",
            daemon=True,
        )
        self.process_update_data_status_thread.start()

    def _start_process_request(self):
        """Start the request processing thread."""
        self.process_request_thread = Thread(
            target=self._process_request,
            name="TransferQueueControllerProcessRequestThread",
            daemon=True,
        )
        self.process_request_thread.start()

    def _process_request(self):
        """Main request processing loop - adapted for partition-based operations."""

        logger.info(f"[{self.controller_id}]: start processing requests...")

        perf_monitor = IntervalPerfMonitor(caller_name=self.controller_id)

        while True:
            messages = self.request_handle_socket.recv_multipart()
            identity = messages.pop(0)
            serialized_msg = messages
            request_msg = ZMQMessage.deserialize(serialized_msg)

            if request_msg.request_type == ZMQRequestType.GET_META:
                with perf_monitor.measure(op_type="GET_META"):
                    params = request_msg.body

                    metadata = self.get_metadata(
                        data_fields=params["data_fields"],
                        batch_size=params["batch_size"],
                        partition_id=params["partition_id"],
                        mode=params.get("mode", "fetch"),
                        task_name=params.get("task_name"),
                        sampling_config=params.get("sampling_config"),
                    )

                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.GET_META_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={"metadata": metadata},
                    )

            elif request_msg.request_type == ZMQRequestType.GET_PARTITION_META:
                with perf_monitor.measure(op_type="GET_PARTITION_META"):
                    params = request_msg.body
                    partition_id = params["partition_id"]

                    metadata = self.get_metadata(
                        data_fields=[],
                        partition_id=partition_id,
                        mode="insert",
                    )
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.GET_PARTITION_META_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={"metadata": metadata},
                    )
            elif request_msg.request_type == ZMQRequestType.SET_CUSTOM_META:
                with perf_monitor.measure(op_type="SET_CUSTOM_META"):
                    params = request_msg.body
                    partition_custom_meta = params["partition_custom_meta"]

                    self.set_custom_meta(partition_custom_meta=partition_custom_meta)

                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.SET_CUSTOM_META_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={"message": "Successfully set custom_meta"},
                    )

            elif request_msg.request_type == ZMQRequestType.CLEAR_META:
                with perf_monitor.measure(op_type="CLEAR_META"):
                    params = request_msg.body
                    global_indexes = params["global_indexes"]
                    partition_ids = params["partition_ids"]

                    self.clear_meta(global_indexes, partition_ids)

                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.CLEAR_META_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={"message": f"Clear samples operation completed by controller {self.controller_id}"},
                    )

            elif request_msg.request_type == ZMQRequestType.CLEAR_PARTITION:
                with perf_monitor.measure(op_type="CLEAR_PARTITION"):
                    params = request_msg.body
                    partition_id = params["partition_id"]

                    self.clear_partition(partition_id)
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.CLEAR_PARTITION_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={"message": f"Clear partition operation completed by controller {self.controller_id}"},
                    )

            elif request_msg.request_type == ZMQRequestType.GET_CONSUMPTION:
                with perf_monitor.measure(op_type="GET_CONSUMPTION"):
                    # Handle consumption status checks
                    params = request_msg.body

                    global_index, consumption_status = self.get_consumption_status(
                        params["partition_id"], params["task_name"]
                    )
                    sample_filter = params.get("sample_filter")  # TODO: DEPRECATED in future

                    if sample_filter and consumption_status is not None:
                        # TODO: DEPRECATED in future
                        consumption_status = consumption_status[sample_filter]

                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.CONSUMPTION_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={
                            "partition_id": params["partition_id"],
                            "global_index": global_index,
                            "consumption_status": consumption_status,
                        },
                    )

            elif request_msg.request_type == ZMQRequestType.RESET_CONSUMPTION:
                with perf_monitor.measure(op_type="RESET_CONSUMPTION"):
                    # Handle reset consumption status request
                    params = request_msg.body
                    partition_id = params["partition_id"]
                    task_name = params.get("task_name")  # Optional
                    try:
                        self.reset_consumption(partition_id, task_name)
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.RESET_CONSUMPTION_RESPONSE,
                            sender_id=self.controller_id,
                            receiver_id=request_msg.sender_id,
                            body={
                                "partition_id": partition_id,
                                "success": True,
                                "message": f"Consumption reset for partition {partition_id}",
                            },
                        )
                    except Exception as e:
                        response_msg = ZMQMessage.create(
                            request_type=ZMQRequestType.RESET_CONSUMPTION_RESPONSE,
                            sender_id=self.controller_id,
                            receiver_id=request_msg.sender_id,
                            body={
                                "partition_id": partition_id,
                                "success": False,
                                "message": str(e),
                            },
                        )

            elif request_msg.request_type == ZMQRequestType.GET_PRODUCTION:
                with perf_monitor.measure(op_type="GET_PRODUCTION"):
                    # Handle production status checks
                    params = request_msg.body

                    global_index, production_status = self.get_production_status(
                        params["partition_id"], params["data_fields"]
                    )

                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.PRODUCTION_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={
                            "partition_id": params["partition_id"],
                            "global_index": global_index,
                            "production_status": production_status,
                        },
                    )

            elif request_msg.request_type == ZMQRequestType.GET_LIST_PARTITIONS:
                with perf_monitor.measure(op_type="GET_LIST_PARTITIONS"):
                    # Handle list partitions request
                    partition_ids = self.list_partitions()
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.LIST_PARTITIONS_RESPONSE,
                        sender_id=self.controller_id,
                        receiver_id=request_msg.sender_id,
                        body={"partition_ids": partition_ids},
                    )

            self.request_handle_socket.send_multipart([identity, *response_msg.serialize()])

    def _update_data_status(self):
        """Process data status update messages from storage units - adapted for partitions."""
        logger.debug(f"[{self.controller_id}]: start receiving update_data_status requests...")

        perf_monitor = IntervalPerfMonitor(caller_name=self.controller_id)

        while True:
            messages = self.data_status_update_socket.recv_multipart()
            identity = messages.pop(0)
            serialized_msg = messages
            request_msg = ZMQMessage.deserialize(serialized_msg)

            if request_msg.request_type == ZMQRequestType.NOTIFY_DATA_UPDATE:
                with perf_monitor.measure(op_type="NOTIFY_DATA_UPDATE"):
                    message_data = request_msg.body
                    partition_id = message_data.get("partition_id")

                    # Update production status
                    success = self.update_production_status(
                        partition_id=partition_id,
                        global_indexes=message_data.get("global_indexes", []),
                        field_names=message_data.get("fields", []),
                        dtypes=message_data.get("dtypes", {}),
                        shapes=message_data.get("shapes", {}),
                        custom_backend_meta=message_data.get("custom_backend_meta", {}),
                    )

                    if success:
                        logger.debug(f"[{self.controller_id}]: Updated production status for partition {partition_id}")

                    # Send acknowledgment
                    response_msg = ZMQMessage.create(
                        request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
                        sender_id=self.controller_id,
                        body={
                            "controller_id": self.controller_id,
                            "partition_id": partition_id,
                            "success": success,
                        },
                    )
                    self.data_status_update_socket.send_multipart([identity, *response_msg.serialize()])

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get ZMQ server connection information."""
        return self.zmq_server_info
