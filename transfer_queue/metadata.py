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
import dataclasses
import itertools
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from transfer_queue.utils.enum_utils import ProductionStatus

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


# TODO: Add UT for metadata operations
@dataclass
class FieldMeta:
    """Records the metadata of a single data field (name, dtype, shape, etc.)."""

    name: str
    dtype: Optional[Any]  # Data type (e.g., torch.float32, numpy.float32)
    shape: Optional[Any]  # Data shape (e.g., torch.Size([3, 224, 224]), (3, 224, 224))
    production_status: ProductionStatus = ProductionStatus.NOT_PRODUCED

    def __str__(self) -> str:
        return (
            f"FieldMeta(name='{self.name}', dtype={self.dtype}, "
            f"shape={self.shape}, production_status={self.production_status})"
        )

    @property
    def is_ready(self) -> bool:
        """Check if this field is ready for consumption"""
        return self.production_status == ProductionStatus.READY_FOR_CONSUME

    @classmethod
    def from_dict(cls, data: dict) -> "FieldMeta":
        """Create FieldMeta from dictionary."""
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            shape=data["shape"],
            production_status=ProductionStatus(str(data["production_status"]))
            if isinstance(data["production_status"], int | str)
            else data["production_status"],
        )


@dataclass
class SampleMeta:
    """Records the metadata of a single data sample (stored as a row in the data system)."""

    partition_id: str  # Partition id, used for data versioning
    global_index: int  # Global row index, uniquely identifies a data sample
    fields: dict[str, FieldMeta]  # Fields of interest for this sample

    def __post_init__(self):
        """Initialize is_ready property based on field readiness"""
        # Check if all fields are ready and update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))

    def __str__(self) -> str:
        return f"SampleMeta(partition_id={self.partition_id}, global_index={self.global_index})"

    @property
    def field_names(self) -> list[str]:
        """Get list of field names for this sample"""
        return list(self.fields.keys())

    @property
    def batch_index(self) -> int:
        """Get the batch index of this sample (to be set by BatchMeta)"""
        return getattr(self, "_batch_index", -1)

    def get_field_by_name(self, name: str) -> Optional[FieldMeta]:
        """Get FieldMeta by field name"""
        return self.fields.get(name)

    def has_field(self, name: str) -> bool:
        """Check if this sample has a specific field"""
        return name in self.fields

    def is_field_ready(self, field_name: str) -> bool:
        """Check if a specific field is ready for consumption"""
        field = self.fields.get(field_name)
        return field.is_ready if field else False

    def add_fields(self, fields: dict[str, FieldMeta]) -> "SampleMeta":
        """
        Add new fields to this sample. New fields will be initialized with given dtype, shape
        and production_status (if provided). If not provided, default values (None, None, READY_FOR_CONSUME)
        will be used. This modifies the sample in-place to include the new fields.
        """
        self.fields = _union_fields(self.fields, fields)
        # Update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))
        return self

    def select_fields(self, field_names: list[str]) -> "SampleMeta":
        """
        Select specific fields from this sample.
        This will construct a new SampleMeta instance containing only the specified fields.

        Args:
            field_names (list[str]): List of field names to retain.

        Returns:
            SampleMeta: A new SampleMeta instance containing only the specified fields.
        """
        selected_fields = {name: self.fields[name] for name in field_names if name in self.fields}

        # construct new SampleMeta instance
        # TODO(tianyi): (maybe) move _custom_backend_meta and custom_meta to FieldMeta level?
        selected_sample_meta = SampleMeta(
            fields=selected_fields,
            partition_id=self.partition_id,
            global_index=self.global_index,
        )

        return selected_sample_meta

    def union(self, other: "SampleMeta", validate: bool = True) -> "SampleMeta":
        """
        Create a union of this sample's fields with another sample's fields.
        Assume both samples have the same global index. If fields overlap, the
        fields in this sample will be replaced by the other sample's fields.

        Args:
            other: Another SampleMeta to union with
            validate: Whether to validate union conditions

        Returns:
            New SampleMeta with unioned fields (None if validation fails)
        """
        if validate:
            if self.global_index != other.global_index:
                raise ValueError(
                    f"Error: Global indexes ({self.global_index} and {other.global_index}) do not match for union."
                )

        # Merge fields
        self.fields = _union_fields(self.fields, other.fields)

        # Update is_ready property
        object.__setattr__(self, "_is_ready", all(field.is_ready for field in self.fields.values()))
        return self

    @property
    def is_ready(self) -> bool:
        """Check if all fields in this sample are ready for consumption"""
        return getattr(self, "_is_ready", False)

    @property
    def production_status(self) -> dict[str, ProductionStatus]:
        """Get production status for all fields (backward compatibility)"""
        return {name: field.production_status for name, field in self.fields.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "SampleMeta":
        """Create SampleMeta from dictionary."""
        fields = {
            name: FieldMeta.from_dict(field_data) if isinstance(field_data, dict) else field_data
            for name, field_data in data["fields"].items()
        }
        return cls(
            partition_id=data["partition_id"],
            global_index=data["global_index"],
            fields=fields,
        )


@dataclass
class BatchMeta:
    """Records the metadata of a batch of data samples."""

    samples: list[SampleMeta]

    # external meta for non-sample level information
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)

    # user-defined meta for each sample
    custom_meta: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)

    # internal meta for different storage backends in per-sample per-field level
    _custom_backend_meta: dict[int, dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Initialize all computed properties during initialization"""
        self.samples = copy.deepcopy(self.samples)
        self.extra_info = copy.deepcopy(self.extra_info)

        # Basic properties
        object.__setattr__(self, "_size", len(self.samples))
        object.__setattr__(self, "_is_ready", all(sample.is_ready for sample in self.samples))

        # Pre-compute all list properties for better performance
        if self.samples:
            for idx, sample in enumerate(self.samples):
                object.__setattr__(sample, "_batch_index", idx)  # Ensure batch_index is set correctly

            object.__setattr__(self, "_global_indexes", [sample.global_index for sample in self.samples])

            # check if all samples have the same field names
            first_sample_field_names = sorted(self.samples[0].field_names)
            if not all(sorted(sample.field_names) == first_sample_field_names for sample in self.samples):
                raise ValueError("All samples in BatchMeta must have the same field_names.")
            object.__setattr__(self, "_field_names", first_sample_field_names)

            object.__setattr__(self, "_partition_ids", [sample.partition_id for sample in self.samples])

            # filter custom_meta and _custom_backend_meta
            self.custom_meta = copy.deepcopy(
                {k: self.custom_meta[k] for k in self.global_indexes if k in self.custom_meta}
            )
            self._custom_backend_meta = copy.deepcopy(
                {k: self._custom_backend_meta[k] for k in self.global_indexes if k in self._custom_backend_meta}
            )
        else:
            self.custom_meta = {}
            self._custom_backend_meta = {}
            object.__setattr__(self, "_global_indexes", [])
            object.__setattr__(self, "_field_names", [])
            object.__setattr__(self, "_partition_ids", [])

    @property
    def size(self) -> int:
        """Return the number of samples in this batch"""
        return getattr(self, "_size", 0)

    @property
    def global_indexes(self) -> list[int]:
        """Get all global indexes in this batch"""
        return getattr(self, "_global_indexes", [])

    @property
    def field_names(self) -> list[str]:
        """Get all unique field names in this batch"""
        return getattr(self, "_field_names", [])

    @property
    def is_ready(self) -> bool:
        """Check if all samples in this batch are ready for consumption"""
        # TODO: get ready status from controller realtime
        return getattr(self, "_is_ready", False)

    @property
    def partition_ids(self) -> list[str]:
        """Get partition ids for all samples in this batch as a list (one per sample)"""
        return getattr(self, "_partition_ids", [])

    def get_all_extra_info(self) -> dict[str, Any]:
        """Get all extra_info as a dictionary (deep copy for immutability).

        Returns:
            A deep copy of the extra_info dictionary
        """
        return copy.deepcopy(self.extra_info)

    def update_extra_info(self, info_dict: dict[str, Any]) -> None:
        """
        Update extra_info with multiple key-value pairs.

        This method updates the extra_info dictionary with the provided key-value pairs.
        Existing keys will be overwritten with new values.

        Args:
            info_dict: Dictionary of key-value pairs to add/update in extra_info
        """
        self.extra_info.update(info_dict)

    def clear_extra_info(self) -> None:
        """
        Clear all extra_info.

        This method removes all key-value pairs from the extra_info dictionary.
        """
        self.extra_info.clear()

    def set_custom_meta(self, global_index: int, meta_dict: dict[str, Any]) -> None:
        """
        Set custom_meta for a specific sample by global_index.

        Custom metadata is user-defined per-sample metadata that can be stored
        and retrieved along with the BatchMeta.

        Args:
            global_index: The global_index of the sample to set custom meta for
            meta_dict: Dictionary containing custom metadata for the sample

        Raises:
            ValueError: If the key is not in global_indexes
        """

        if global_index not in self.global_indexes:
            raise ValueError(f"key {global_index} not found in global_indexes {self.global_indexes}.")

        self.custom_meta[global_index] = copy.deepcopy(meta_dict)

    def get_all_custom_meta(self) -> dict[int, dict[str, Any]]:
        """
        Get all custom_meta as a dictionary.

        Returns:
            A deep copy of the custom_meta dictionary
        """
        return copy.deepcopy(self.custom_meta)

    def update_custom_meta(self, new_meta: dict[int, dict[str, Any]]):
        """
        Update custom_meta with a dictionary of new metadata.

        This method updates the custom_meta dictionary with the provided metadata.
        Existing keys will be overwritten with new values.

        Args:
            new_meta: Dictionary of new metadata

        Raises:
            ValueError: If any key in new_meta is not in global_indexes
        """

        if new_meta is None:
            return

        non_exist_global_indexes = set(new_meta.keys()) - set(self.global_indexes)
        if non_exist_global_indexes:
            raise ValueError(
                f"Trying to update custom_meta with non-exist global_indexes! {non_exist_global_indexes} "
                f"do not exist in this batch."
            )

        self.custom_meta.update(new_meta)

    def clear_custom_meta(self) -> None:
        """
        Clear all custom_meta.

        This method removes all entries from the custom_meta dictionary.
        """
        self.custom_meta.clear()

    def add_fields(self, tensor_dict: TensorDict, set_all_ready: bool = True) -> "BatchMeta":
        """
        Add new fields from a TensorDict to all samples in this batch.
        This modifies each sample in-place to include the new fields.

        Args:
            tensor_dict (TensorDict): The input TensorDict containing new fields.
            set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME. Default is True.
        """
        fields = _extract_field_metas(tensor_dict, set_all_ready)

        if fields:
            if len(self.samples) != len(fields):
                raise ValueError(f"add_fields length mismatch: samples={len(self.samples)} vs fields={len(fields)}")
        for idx, sample in enumerate(self.samples):
            sample.add_fields(fields=fields[idx])

        # Update batch-level fields cache
        if self.samples:
            object.__setattr__(self, "_field_names", sorted(self.samples[0].field_names))
            object.__setattr__(self, "_is_ready", all(sample.is_ready for sample in self.samples))
        return self

    def select_samples(self, indexes: list[int]) -> "BatchMeta":
        """
        Select specific samples from this batch.
        This will construct a new BatchMeta instance containing only the specified samples.

        Args:
            indexes (list[int]): List of indexes (relative to this batch, not global_indexes)
                to retain.

        Returns:
            BatchMeta: A new BatchMeta instance containing only the specified samples.
        """

        if any(i < 0 or i >= len(self.samples) for i in indexes):
            raise ValueError(f"Sample indices must be in range [0, {len(self.samples)})")

        selected_samples = [self.samples[i] for i in indexes]

        global_indexes = [self.global_indexes[i] for i in indexes]
        selected_custom_meta = {i: self.custom_meta[i] for i in global_indexes if i in self.custom_meta}
        selected_custom_backend_meta = {
            i: self._custom_backend_meta[i] for i in global_indexes if i in self._custom_backend_meta
        }

        # construct new BatchMeta instance
        selected_batch_meta = BatchMeta(
            samples=selected_samples,
            extra_info=self.extra_info,
            custom_meta=selected_custom_meta,
            _custom_backend_meta=selected_custom_backend_meta,
        )

        return selected_batch_meta

    def select_fields(self, field_names: list[str]) -> "BatchMeta":
        """
        Select specific fields from all samples in this batch.
        This will construct a new BatchMeta instance containing only the specified fields.

        Args:
            field_names (list[str]): List of field names to retain.

        Returns:
            BatchMeta: A new BatchMeta instance containing only the specified fields from all samples.
        """
        # select fields for each SampleMeta
        new_samples = [sample.select_fields(field_names=field_names) for sample in self.samples]

        # select fields in _custom_backend_meta
        selected_custom_backend_meta = {}
        for idx in self.global_indexes:
            if idx in self._custom_backend_meta:
                custom_backend_meta_idx = self._custom_backend_meta[idx]

                selected_custom_backend_meta[idx] = {
                    field: custom_backend_meta_idx[field] for field in field_names if field in custom_backend_meta_idx
                }

        # construct new BatchMeta instance
        new_batch_meta = BatchMeta(
            samples=new_samples,
            extra_info=self.extra_info,
            custom_meta=self.custom_meta,
            _custom_backend_meta=selected_custom_backend_meta,
        )

        return new_batch_meta

    def __len__(self) -> int:
        """Return the number of samples in this batch."""
        return len(self.samples)

    def __getitem__(self, item):
        if isinstance(item, int | np.integer):
            sample_meta = self.samples[item] if self.samples else []
            global_idx = self.global_indexes[item]

            if global_idx in self.custom_meta:
                custom_meta = {global_idx: self.custom_meta[global_idx]}
            else:
                custom_meta = {}

            if global_idx in self._custom_backend_meta:
                custom_backend_meta = {global_idx: self._custom_backend_meta[global_idx]}
            else:
                custom_backend_meta = {}

            return BatchMeta(
                samples=[sample_meta],
                extra_info=self.extra_info,
                custom_meta=custom_meta,
                _custom_backend_meta=custom_backend_meta,
            )
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported now!")

    def chunk(self, chunks: int) -> list["BatchMeta"]:
        """
        Split this batch into smaller chunks.

        Args:
            chunks: number of chunks

        Return:
            List of smaller BatchMeta chunks
        """
        chunk_list = []
        n = len(self.samples)

        if n < chunks:
            logger.warning(
                f"Chunk size {chunks} > number of samples in BatchMeta {n}, this will return some "
                f"empty BatchMeta chunks."
            )

        # Calculate the base size and remainder of each chunk
        base_size = n // chunks
        remainder = n % chunks

        start = 0
        for i in range(chunks):
            # Calculate the size of the current chunk(the first remainder chunk is 1 more than the base size)
            current_chunk_size = base_size + 1 if i < remainder else base_size
            end = start + current_chunk_size
            chunk_samples = self.samples[start:end]
            global_indexes = self.global_indexes[start:end]
            chunk_custom_meta = {i: self.custom_meta[i] for i in global_indexes if i in self.custom_meta}
            chunk_custom_backend_meta = {
                i: self._custom_backend_meta[i] for i in global_indexes if i in self._custom_backend_meta
            }
            chunk = BatchMeta(
                samples=chunk_samples,
                extra_info=self.extra_info,
                custom_meta=chunk_custom_meta,
                _custom_backend_meta=chunk_custom_backend_meta,
            )
            chunk_list.append(chunk)
            start = end
        return chunk_list

    def chunk_by_partition(
        self,
    ) -> list["BatchMeta"]:
        """
        Split this batch into smaller chunks according to partition_ids.

        Return:
            List of smaller BatchMeta chunks, each chunk has samples with identical partition_id
        """

        grouped_indexes = defaultdict(list)
        for partition_id, indexes in zip(self.partition_ids, range(self.size), strict=False):
            grouped_indexes[partition_id].append(indexes)

        chunk_list = [self.select_samples(idx) for idx in grouped_indexes.values()]

        return chunk_list

    @classmethod
    def concat(cls, data: list["BatchMeta"], validate: bool = True) -> "BatchMeta":
        """
        Concatenate multiple BatchMeta chunks into one large batch.

        Args:
            data: List of BatchMeta chunks to concatenate
            validate: Whether to validate concatenation conditions

        Returns:
            Concatenated BatchMeta

        Raises:
            ValueError: If validation fails (e.g., field names do not match)
        """
        if not data:
            logger.warning("Try to concat empty BatchMeta chunks. Returning empty BatchMeta.")
            return BatchMeta(samples=[], extra_info={}, custom_meta={}, _custom_backend_meta={})

        # skip empty chunks
        data = [chunk for chunk in data if chunk and len(chunk.samples) > 0]

        if len(data) == 0:
            logger.warning("No valid BatchMeta chunks to concatenate. Returning empty BatchMeta.")
            return BatchMeta(samples=[], extra_info={}, custom_meta={}, _custom_backend_meta={})

        if validate:
            base_fields = data[0].field_names

            for chunk in data:
                if chunk.field_names != base_fields:
                    raise ValueError("Error: Field names do not match for concatenation.")

        # Combine all samples
        all_samples = list(itertools.chain.from_iterable(chunk.samples for chunk in data))

        # Merge all extra_info dictionaries from the chunks
        merged_extra_info = dict()
        merged_custom_meta = dict()
        merged_custom_backend_meta = dict()

        values_by_key = defaultdict(list)
        for chunk in data:
            # For the sample-level custom_meta and field-level _custom_backend_meta, we directly update the dict.
            merged_custom_meta.update(chunk.custom_meta)
            merged_custom_backend_meta.update(chunk._custom_backend_meta)

            for key, value in chunk.extra_info.items():
                values_by_key[key].append(value)

        # For the batch-level extra_info, we concat the tensor/NonTensorStack/NonTensorData/list
        # objects to prevent information losses.
        for key, values in values_by_key.items():
            if all(isinstance(v, torch.Tensor) for v in values):
                try:
                    if all(v.dim() == 0 for v in values):
                        merged_extra_info[key] = torch.cat([v.unsqueeze(0) for v in values], dim=0)
                    else:
                        merged_extra_info[key] = torch.cat(values, dim=0)
                except RuntimeError as e:
                    logger.warning(
                        f"BatchMeta.concat try to use torch.cat(dim=0) to merge extra_info key '{key}'"
                        f" fails, with RuntimeError {e}. Falling back to use list."
                    )
                    merged_extra_info[key] = values
            elif all(isinstance(v, NonTensorStack | NonTensorData) for v in values):
                merged_extra_info[key] = torch.stack(values)
            elif all(isinstance(v, list) for v in values):
                merged_extra_info[key] = list(itertools.chain.from_iterable(values))
            else:
                merged_extra_info[key] = values[-1]

        return BatchMeta(
            samples=all_samples,
            extra_info=merged_extra_info,
            custom_meta=merged_custom_meta,
            _custom_backend_meta=merged_custom_backend_meta,
        )

    def union(self, other: "BatchMeta", validate: bool = True) -> Optional["BatchMeta"]:
        """
        Create a union of this batch's fields with another batch's fields.
        Assume both batches have the same global indices and matching partition_ids for all samples.
         If fields overlap, the fields in this batch will be replaced by the other batch's fields.

        Args:
            other: Another BatchMeta to union with
            validate: Whether to validate union conditions

        Returns:
            New BatchMeta with unioned fields

        Raises:
            ValueError: If validation fails (e.g., batch sizes or global indexes do not match)
        """
        if validate:
            if self.size != other.size:
                raise ValueError("Error: Batch sizes do not match for union.")

            self_global_indexes = sorted(self.global_indexes)
            other_global_indexes = sorted(other.global_indexes)
            if self_global_indexes != other_global_indexes:
                raise ValueError("Error: Global indexes do not match for union.")

            if self.partition_ids != other.partition_ids:
                raise ValueError("Error: Partition IDs do not match for union.")

        # Create a mapping from global_index to SampleMeta in the other batch
        other_sample_map = {sample.global_index: sample for sample in other.samples}

        # Merge samples
        merged_samples = []
        for sample in self.samples:
            if sample.global_index in other_sample_map:
                other_sample = other_sample_map[sample.global_index]
                merged_sample = sample.union(other_sample, validate=validate)
                merged_samples.append(merged_sample)
            else:
                merged_samples.append(sample)

        # Merge extra info dictionaries
        merged_extra_info = {**self.extra_info, **other.extra_info}

        # Merge custom_meta dictionaries
        merged_custom_meta = {**self.custom_meta, **other.custom_meta}

        # Merge custom_backend_meta dictionaries
        merged_custom_backend_meta = {}
        for idx in self.global_indexes:
            if idx in self._custom_backend_meta and idx in other._custom_backend_meta:
                merged_custom_backend_meta[idx] = {**self._custom_backend_meta[idx], **other._custom_backend_meta[idx]}
            elif idx in self._custom_backend_meta:
                merged_custom_backend_meta[idx] = {**self._custom_backend_meta[idx]}
            elif idx in other._custom_backend_meta:
                merged_custom_backend_meta[idx] = {**other._custom_backend_meta[idx]}

        return BatchMeta(
            samples=merged_samples,
            extra_info=merged_extra_info,
            custom_meta=merged_custom_meta,
            _custom_backend_meta=merged_custom_backend_meta,
        )

    def reorder(self, indexes: list[int]):
        """
        Reorder the SampleMeta in the BatchMeta according to the given indexes (must equal to the length of samples).

        The operation is performed in-place, modifying the current BatchMeta's SampleMeta order.

        To select a subset of samples or repeat specific samples, please use the non-inplace method select_samples().

        Args:
            indexes : list[int]
                A list of integers specifying the new order of SampleMeta. Each integer
                represents the current index of the SampleMeta in the BatchMeta.
        """

        if len(indexes) != self.size:
            raise ValueError(
                f"Attempted to reorder with indexes length {len(indexes)} that does not match samples length "
                f"{self.size}. Please use non-inplace method select_samples() instead if you want to "
                f"select a subset of samples or repeat specific samples."
            )

        if len(set(indexes)) != self.size:
            raise ValueError(
                f"Indexes={indexes} contain duplicates. Please use non-inplace method "
                f"select_samples() instead if you want to select a subset of samples or repeat specific samples."
            )

        if any(i < 0 or i >= len(self.samples) for i in indexes):
            raise ValueError(f"Reorder indexes must be in the range [0, {self.size}).")

        # Reorder the samples
        reordered_samples = [self.samples[i] for i in indexes]
        object.__setattr__(self, "samples", reordered_samples)

        # Update necessary attributes
        self._update_after_reorder()

    def _update_after_reorder(self) -> None:
        """Update related attributes specifically for the reorder operation"""
        # Update batch_index for each sample
        for idx, sample in enumerate(self.samples):
            object.__setattr__(sample, "_batch_index", idx)

        # Update cached index lists
        object.__setattr__(self, "_global_indexes", [sample.global_index for sample in self.samples])
        object.__setattr__(self, "_partition_ids", [sample.partition_id for sample in self.samples])

        # Note: No need to update _size, _field_names, _is_ready, etc., as these remain unchanged after reorder

    @classmethod
    def from_samples(
        cls, samples: SampleMeta | list[SampleMeta], extra_info: Optional[dict[str, Any]] = None
    ) -> "BatchMeta":
        """
        Create a BatchMeta from a single SampleMeta or a list of SampleMeta objects.

        Args:
            samples: A single SampleMeta or a list of SampleMeta objects
            extra_info: Optional additional information to store with the batch

        Returns:
            BatchMeta instance containing the provided sample(s)

        Example:
            >>> sample_meta = SampleMeta(...)
            >>> batch_meta = BatchMeta.from_samples(sample_meta)

            >>> sample_metas = [sample1, sample2, sample3]
            >>> batch_meta = BatchMeta.from_samples(sample_metas, extra_info={"source": "training"})
        """
        if extra_info is None:
            extra_info = {}

        if isinstance(samples, SampleMeta):
            samples = [samples]

        return cls(samples=samples, extra_info=extra_info)

    @classmethod
    def empty(cls, extra_info: Optional[dict[str, Any]] = None) -> "BatchMeta":
        """
        Create an empty BatchMeta with no samples.

        Args:
            extra_info: Optional additional information to store with the batch

        Returns:
            Empty BatchMeta instance

        Example:
            >>> empty_batch = BatchMeta.empty()
        """
        if extra_info is None:
            extra_info = {}
        return cls(samples=[], extra_info=extra_info, custom_meta={}, _custom_backend_meta={})

    def __str__(self):
        sample_strs = ", ".join(str(sample) for sample in self.samples)
        return (
            f"BatchMeta(size={self.size}, field_names={self.field_names}, is_ready={self.is_ready}, "
            f"samples=[{sample_strs}], extra_info={self.extra_info})"
        )

    @classmethod
    def from_dict(cls, data: dict) -> "BatchMeta":
        """Create BatchMeta from dictionary."""
        samples = [
            SampleMeta.from_dict(sample_data) if isinstance(sample_data, dict) else sample_data
            for sample_data in data["samples"]
        ]
        return cls(
            samples=samples,
            extra_info=data.get("extra_info", {}),
            custom_meta=data.get("custom_meta", {}),
            _custom_backend_meta=data.get("_custom_backend_meta", {}),
        )


def _union_fields(fields1: dict[str, FieldMeta], fields2: dict[str, FieldMeta]) -> dict[str, FieldMeta]:
    """Union two sample's fields. If fields overlap, the fields in fields1 will be replaced by fields2."""
    for name in fields2.keys():
        fields1[name] = fields2[name]
    return fields1


def _extract_field_metas(tensor_dict: TensorDict, set_all_ready: bool = True) -> list[dict[str, FieldMeta]]:
    """
    Extract field metas from a TensorDict. If data in tensor_dict does not have dtype or shape attribute,
    the corresponding dtype or shape will be set to None.

    Args:
        tensor_dict (TensorDict): The input TensorDict.
        set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME.
                              Otherwise, set to NOT_PRODUCED. Default is True.

    Returns:
        all_fields (list[dict[str, FieldMeta]]): A list of dictionaries containing field metadata.
    """
    batch_size = tensor_dict.batch_size[0]

    production_status = ProductionStatus.READY_FOR_CONSUME if set_all_ready else ProductionStatus.NOT_PRODUCED

    all_fields = [
        {
            name: FieldMeta(
                name=name,
                dtype=getattr(value, "dtype", None),
                shape=getattr(value, "shape", None),
                production_status=production_status,
            )
            for name, value in tensor_dict[idx].items()
        }
        for idx in range(batch_size)
    ]

    return all_fields
