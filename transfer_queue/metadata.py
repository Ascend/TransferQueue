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
from types import MappingProxyType
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extra_info_values_equal(a: Any, b: Any) -> bool:
    """Compare two extra_info values for equality.

    Handles torch.Tensor, np.ndarray specially to avoid ambiguous truth values.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    try:
        return a == b
    except Exception:
        return False


class _SampleView:
    """Lazy read-only view of a single sample row in a columnar BatchMeta.

    All returned dicts are ``MappingProxyType`` – attempts to mutate them
    raise ``TypeError``, making it obvious that this is a snapshot view.
    """

    __slots__ = ("_batch", "_idx")

    def __init__(self, batch: "BatchMeta", idx: int) -> None:
        self._batch = batch
        self._idx = idx

    @property
    def global_index(self) -> int:
        """Return the global sample index for this sample."""
        return self._batch.global_indexes[self._idx]

    @property
    def partition_id(self) -> str:
        """Return the partition ID for this sample."""
        return self._batch.partition_ids[self._idx]

    @property
    def production_status(self) -> int:
        """Return the production status for this sample."""
        return int(self._batch.production_status[self._idx])

    @property
    def custom_meta(self) -> "MappingProxyType[str, Any]":
        """Read-only view of per-sample custom metadata."""
        return MappingProxyType(self._batch.custom_meta[self._idx])

    @property
    def fields(self) -> "MappingProxyType[str, MappingProxyType]":
        """Read-only per-sample field schema.

        For nested-tensor fields the batch-level ``per_sample_shapes`` list is
        replaced by a single ``shape`` entry for *this* sample, so callers
        always see ``fields['x']['shape']`` as a tuple (not a list-of-tuples).
        """
        result: dict[str, MappingProxyType] = {}
        for name, meta in self._batch.field_schema.items():
            per_sample = meta.get("per_sample_shapes")
            if per_sample is not None:
                sample_meta = {k: v for k, v in meta.items() if k != "per_sample_shapes"}
                sample_meta["shape"] = per_sample[self._idx]
            else:
                sample_meta = dict(meta)
            result[name] = MappingProxyType(sample_meta)
        return MappingProxyType(result)

    def __repr__(self) -> str:
        return (
            f"_SampleView(global_index={self.global_index}, "
            f"partition_id={self.partition_id!r}, "
            f"production_status={self.production_status}, "
            f"fields={list(self._batch.field_schema.keys())})"
        )


class _SampleViewList:
    """Lazy indexable list returned by BatchMeta.samples.

    Supports: indexing (samples[i]), len(), and iteration.
    """

    __slots__ = ("_batch",)

    def __init__(self, batch: "BatchMeta") -> None:
        self._batch = batch

    def __len__(self) -> int:
        return len(self._batch.global_indexes)

    def __getitem__(self, idx: int) -> _SampleView:
        return _SampleView(self._batch, idx)

    def __iter__(self):
        return (_SampleView(self._batch, i) for i in range(len(self)))


def extract_field_schema(data: TensorDict) -> dict[str, dict[str, Any]]:
    """Extract field-level schema from TensorDict."""
    field_schema: dict[str, dict[str, Any]] = {}
    batch_size = data.batch_size[0]

    for field_name, value in data.items():
        is_tensor = isinstance(value, torch.Tensor)
        is_nested = is_tensor and value.is_nested

        first_item = None
        if is_nested:
            unbound = value.unbind()
            first_item = unbound[0] if unbound else None
        elif is_tensor:
            first_item = value[0] if value.shape[0] > 0 else None
        else:
            first_item = value[0] if len(value) > 0 else None

        # Determine is_non_tensor: when first_item is None (empty field), cannot determine type
        if first_item is None:
            is_non_tensor = None
        else:
            is_non_tensor = not is_tensor

        # Determine the shape of each sample (excluding batch dimension)
        # When TensorDict converts a Python list to tensor, the first dimension equals batch_size
        # We need to strip this batch dimension to get per-sample shape
        if isinstance(value, torch.Tensor) and not is_nested and value.shape[0] > 0:
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"Inconsistent batch dimension for field '{field_name}': "
                    f"expected batch_size[0]={batch_size}, got value.shape[0]={value.shape[0]}"
                )
            if len(value.shape) > 1:
                sample_shape = value.shape[1:]
            else:
                # When input is 1D tensor, manually set to torch.Size([1]).
                sample_shape = torch.Size([1])
        else:
            sample_shape = getattr(first_item, "shape", None) if first_item is not None else None

        field_meta = {
            "dtype": getattr(first_item, "dtype", type(first_item) if first_item is not None else None),
            "shape": sample_shape,
            "is_nested": is_nested,
            "is_non_tensor": is_non_tensor,
        }

        # For nested tensors, record per-sample shapes
        if is_nested:
            field_meta["per_sample_shapes"] = [tuple(t.shape) for t in value.unbind()]

        field_schema[field_name] = field_meta

    return field_schema


@dataclass
class BatchMeta:
    """Records the metadata of a batch of data samples with optimized field-level schema.

    This is the O(BxF) optimized version that stores field metadata at the field level
    instead of per-sample, reducing storage from O(B*F) to O(F).

    Attributes:
        global_indexes: List of global sample indices in this batch.
        partition_ids: List of partition IDs corresponding to each sample.
        field_schema: Field-level metadata {field_name: {dtype, shape, is_nested, is_non_tensor, per_sample_shapes}}.
        production_status: Vectorized production status, shape (B,) where B is batch size.
        extra_info: Additional batch-level information.
        custom_meta: Per-sample user-defined metadata, list aligned with global_indexes.
        _custom_backend_meta: Per-sample per-field storage backend metadata, list aligned with global_indexes.
    """

    global_indexes: list[int]
    partition_ids: list[str]
    # field-level metadata: {field_name: {dtype, shape, is_nested, is_non_tensor}}
    field_schema: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # vectorized production status matrix
    production_status: np.ndarray = dataclasses.field(default=None, repr=False)  # type: ignore[assignment]
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    # user-defined meta for each sample (sample-level), list aligned with global_indexes
    custom_meta: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # internal meta for different storage backends (per-sample per-field level), list aligned with global_indexes
    _custom_backend_meta: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """Initialize all computed properties during initialization"""
        self.global_indexes = list(self.global_indexes)
        self.partition_ids = list(self.partition_ids)
        self.field_schema = {k: dict(v) for k, v in self.field_schema.items()}
        self.extra_info = dict(self.extra_info)

        # Validation
        if len(self.global_indexes) != len(self.partition_ids):
            raise ValueError(
                f"Length mismatch: global_indexes has {len(self.global_indexes)}, "
                f"partition_ids has {len(self.partition_ids)}"
            )

        batch_size = len(self.global_indexes)

        if self.production_status is not None:
            self.production_status = np.array(self.production_status, dtype=np.int8, copy=True)

            if len(self.production_status) != batch_size:
                raise ValueError(f"production_status length {len(self.production_status)} != batch_size {batch_size}")
        else:
            # Default: all NOT_PRODUCED (including empty batches)
            self.production_status = np.zeros(batch_size, dtype=np.int8)

        for field_name, meta in self.field_schema.items():
            if meta.get("per_sample_shapes") is not None:
                if len(meta["per_sample_shapes"]) != batch_size:
                    raise ValueError(
                        f"Field '{field_name}' per_sample_shapes length {len(meta['per_sample_shapes'])} "
                        f"!= batch_size {batch_size}"
                    )

        self._size = batch_size
        self._field_names = sorted(self.field_schema.keys())

        is_ready = batch_size > 0 and bool(np.all(self.production_status == 1))
        self._is_ready = is_ready

        # Validate or initialize columnar custom_meta / _custom_backend_meta
        if not self.custom_meta:
            self.custom_meta = [{} for _ in range(batch_size)]
        else:
            self.custom_meta = [dict(d) for d in self.custom_meta]
            if len(self.custom_meta) != batch_size:
                raise ValueError(f"custom_meta length {len(self.custom_meta)} != batch_size {batch_size}")
        if not self._custom_backend_meta:
            self._custom_backend_meta = [{} for _ in range(batch_size)]
        else:
            self._custom_backend_meta = [dict(d) for d in self._custom_backend_meta]
            if len(self._custom_backend_meta) != batch_size:
                raise ValueError(
                    f"_custom_backend_meta length {len(self._custom_backend_meta)} != batch_size {batch_size}"
                )

    def __setstate__(self, state):
        """Restore instance from pickle/Ray deserialization.

        Python dataclass pickle skips __init__/__post_init__, so the
        .copy() guard for production_status is bypassed.  Ray Arrow
        zero-copy deserialization produces read-only numpy arrays.
        This method ensures writability after deserialization.
        """
        self.__dict__.update(state)
        if isinstance(self.production_status, np.ndarray) and not self.production_status.flags.writeable:
            self.production_status = self.production_status.copy()

    @property
    def size(self) -> int:
        """Return the number of samples in this batch"""
        return getattr(self, "_size", 0)

    @property
    def field_names(self) -> list[str]:
        """Get all unique field names in this batch"""
        return getattr(self, "_field_names", [])

    @property
    def samples(self) -> _SampleViewList:
        """Lazy per-sample view: supports samples[i].fields['a'], len(samples), for s in samples."""
        return _SampleViewList(self)

    @property
    def is_ready(self) -> bool:
        """Check if all samples in this batch are ready for consumption"""
        return getattr(self, "_is_ready", False)

    def get_dtypes(self, field_name: str) -> list:
        """Return a per-sample list of dtypes for the given field.

        Since dtype is uniform across all samples in a field, the returned list
        contains the same dtype repeated ``self.size`` times.

        Args:
            field_name: Name of the field to query.

        Returns:
            A list of length ``self.size`` where each element is the field's dtype.

        Raises:
            KeyError: If *field_name* is not present in ``field_schema``.
        """
        if field_name not in self.field_schema:
            raise KeyError(f"Field '{field_name}' not found in field_schema")
        dtype = self.field_schema[field_name].get("dtype")
        return [dtype] * self.size

    def get_shapes(self, field_name: str) -> list:
        """Return a per-sample list of shapes for the given field.

        For nested-tensor fields the shapes come from ``per_sample_shapes``.
        For regular (non-nested) fields the uniform ``shape`` is repeated
        ``self.size`` times so the caller always gets one entry per sample.

        Args:
            field_name: Name of the field to query.

        Returns:
            A list of length ``self.size`` where each element is a shape tuple.

        Raises:
            KeyError: If *field_name* is not present in ``field_schema``.
        """
        if field_name not in self.field_schema:
            raise KeyError(f"Field '{field_name}' not found in field_schema")
        meta = self.field_schema[field_name]
        per_sample = meta.get("per_sample_shapes")
        if per_sample is not None:
            return list(per_sample)
        return [meta.get("shape")] * self.size

    # ==================== Extra Info Methods ====================

    def get_extra_info(self, key: str, default: Any = None) -> Any:
        """Get extra info by key"""
        return self.extra_info.get(key, default)

    def set_extra_info(self, key: str, value: Any) -> None:
        """Set extra info by key"""
        self.extra_info[key] = value

    def get_all_extra_info(self) -> dict[str, Any]:
        """Get all extra_info as a dictionary (deep copy for immutability).

        Returns:
            A deep copy of the extra_info dictionary
        """
        return copy.deepcopy(self.extra_info)

    def update_extra_info(self, info_dict: dict[str, Any]) -> None:
        """Update extra_info with multiple key-value pairs.

        Args:
            info_dict: Dictionary of key-value pairs to add/update in extra_info
        """
        self.extra_info.update(info_dict)

    def remove_extra_info(self, key: str) -> Any:
        """Remove extra info by key and return its value"""
        return self.extra_info.pop(key, None)

    def clear_extra_info(self) -> None:
        """Clear all extra_info."""
        self.extra_info.clear()

    def has_extra_info(self, key: str) -> bool:
        """Check if extra info contains a specific key"""
        return key in self.extra_info

    # ==================== Custom Meta Methods (User Layer) ====================

    def get_all_custom_meta(self) -> list[dict[str, Any]]:
        """Get all custom_meta as a list of dictionary (one per sample, in global_indexes order).

        Returns:
            A deep copy of the custom_meta list
        """
        return copy.deepcopy(self.custom_meta)

    def update_custom_meta(self, custom_meta: list[dict[str, Any]]):
        """Update custom_meta with a list of dictionary of custom metadata.

        Args:
            custom_meta: list of custom_meta dictionary (one per sample, in global_indexes order)

        Raises:
            ValueError: If the length of custom_meta does not match the batch size
        """
        if custom_meta is None:
            return

        if len(custom_meta) != self.size:
            raise ValueError(
                f"The length of custom_meta list {len(custom_meta)} must match the batch size: {self.size}"
            )

        for i, meta in enumerate(custom_meta):
            self.custom_meta[i].update(meta)

    def clear_custom_meta(self) -> None:
        """Clear all custom_meta."""
        self.custom_meta = [{} for _ in range(self.size)]

    # ==================== Core BatchMeta Operations ====================

    def add_fields(self, tensor_dict: TensorDict, set_all_ready: bool = True) -> "BatchMeta":
        """Add new fields from a TensorDict to all samples in this batch.
        This modifies the batch in-place to include the new fields.

        Args:
            tensor_dict (TensorDict): The input TensorDict containing new fields.
            set_all_ready (bool): If True, set all production_status to READY_FOR_CONSUME. Default is True.
        """
        batch_size = tensor_dict.batch_size[0]
        if batch_size != self.size:
            raise ValueError(f"add_fields batch size mismatch: self.size={self.size} vs tensor_dict={batch_size}")

        field_schema = extract_field_schema(tensor_dict)

        for key, value in field_schema.items():
            self.field_schema[key] = value

        if set_all_ready:
            self.production_status[:] = 1

        self._field_names = sorted(self.field_schema.keys())

        self._is_ready = self.size > 0 and bool(np.all(self.production_status == 1))

        return self

    def select_samples(self, sample_indices: list[int]) -> "BatchMeta":
        """Select specific samples from this batch.
        This will construct a new BatchMeta instance containing only the specified samples.

        Args:
            sample_indices (list[int]): List of sample indices (relative to this batch) to retain.

        Returns:
            BatchMeta: A new BatchMeta instance containing only the specified samples.
        """
        if any(i < 0 or i >= self.size for i in sample_indices):
            raise ValueError(f"Sample indices must be in range [0, {self.size})")

        new_global_indexes = [self.global_indexes[i] for i in sample_indices]
        new_partition_ids = [self.partition_ids[i] for i in sample_indices]

        # Select production_status
        new_production_status = self.production_status[sample_indices]

        new_field_schema = {}
        for field_name, meta in self.field_schema.items():
            new_meta = copy.deepcopy(meta)
            if meta.get("per_sample_shapes") is not None:
                new_meta["per_sample_shapes"] = [meta["per_sample_shapes"][i] for i in sample_indices]
            new_field_schema[field_name] = new_meta

        new_custom_meta = [copy.deepcopy(self.custom_meta[i]) for i in sample_indices]

        new_custom_backend_meta = [copy.deepcopy(self._custom_backend_meta[i]) for i in sample_indices]

        return BatchMeta(
            global_indexes=new_global_indexes,
            partition_ids=new_partition_ids,
            field_schema=new_field_schema,
            production_status=new_production_status,
            extra_info=self.extra_info,
            custom_meta=new_custom_meta,
            _custom_backend_meta=new_custom_backend_meta,
        )

    def select_fields(self, field_names: list[str]) -> "BatchMeta":
        """Select specific fields from all samples in this batch.
        This will construct a new BatchMeta instance containing only the specified fields.

        Args:
            field_names (list[str]): List of field names to retain.

        Returns:
            BatchMeta: A new BatchMeta instance containing only the specified fields.
        """
        new_field_schema = {}
        for field_name in field_names:
            if field_name in self.field_schema:
                new_field_schema[field_name] = copy.deepcopy(self.field_schema[field_name])

        selected_custom_backend_meta = [
            {f: v for f, v in m.items() if f.startswith("_") or f in field_names} for m in self._custom_backend_meta
        ]

        return BatchMeta(
            global_indexes=self.global_indexes,
            partition_ids=self.partition_ids,
            field_schema=new_field_schema,
            production_status=self.production_status.copy(),
            extra_info=copy.deepcopy(self.extra_info),
            custom_meta=copy.deepcopy(self.custom_meta),
            _custom_backend_meta=selected_custom_backend_meta,
        )

    def __len__(self) -> int:
        """Return the number of samples in this batch."""
        return self.size

    def __getitem__(self, item) -> "BatchMeta":
        if isinstance(item, int | np.integer):
            if item < 0:
                item += self.size
            if item < 0 or item >= self.size:
                raise IndexError("BatchMeta index out of range")
            return self.select_samples([item])
        elif isinstance(item, slice):
            start, stop, step = item.indices(self.size)
            indices = list(range(start, stop, step))
            return self.select_samples(indices)
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported.")

    def chunk(self, chunks: int) -> list["BatchMeta"]:
        """Split this batch into smaller chunks.

        Args:
            chunks: number of chunks

        Return:
            List of smaller BatchMeta chunks
        """
        chunk_list = []
        n = self.size

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
            current_chunk_size = base_size + 1 if i < remainder else base_size
            end = start + current_chunk_size
            indices = list(range(start, end))
            chunk = self.select_samples(indices)
            chunk_list.append(chunk)
            start = end
        return chunk_list

    def chunk_by_partition(self) -> list["BatchMeta"]:
        """Split this batch into smaller chunks according to partition_ids.

        Return:
            List of smaller BatchMeta chunks, each chunk has samples with identical partition_id
        """
        grouped_indexes = defaultdict(list)
        for partition_id, indexes in zip(self.partition_ids, range(self.size), strict=True):
            grouped_indexes[partition_id].append(indexes)

        chunk_list = [self.select_samples(idx) for idx in grouped_indexes.values()]
        return chunk_list

    def union(self, other: "BatchMeta") -> "BatchMeta":
        """Return the union of this BatchMeta and another BatchMeta.
        Samples with global_indexes already present in this batch are ignored from the other batch.

        Args:
            other: The other BatchMeta to merge with.

        Returns:
            BatchMeta: A new merged BatchMeta.
        """
        if not other or other.size == 0:
            return self
        if self.size == 0:
            return other

        self_indexes = set(self.global_indexes)
        unique_indices_in_other = [i for i, idx in enumerate(other.global_indexes) if idx not in self_indexes]

        if not unique_indices_in_other:
            return self

        if len(unique_indices_in_other) == other.size:
            return BatchMeta.concat([self, other])

        other_unique = other.select_samples(unique_indices_in_other)
        return BatchMeta.concat([self, other_unique])

    @classmethod
    def concat(cls, data: list["BatchMeta"], validate: bool = True) -> "BatchMeta":
        """Concatenate multiple BatchMeta chunks into one large batch.

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
            return BatchMeta.empty()

        # skip empty chunks
        data = [chunk for chunk in data if chunk and chunk.size > 0]

        if len(data) == 0:
            logger.warning("No valid BatchMeta chunks to concatenate. Returning empty BatchMeta.")
            return BatchMeta.empty()

        if validate:
            base_fields = data[0].field_names

            for i, chunk in enumerate(data):
                if chunk.field_names != base_fields:
                    raise ValueError(
                        f"BatchMeta.concat: field_names mismatch at chunk[{i}]. "
                        f"Expected {base_fields}, got {chunk.field_names}. "
                        f"Extra in chunk: {set(chunk.field_names) - set(base_fields)}, "
                        f"Missing from chunk: {set(base_fields) - set(chunk.field_names)}"
                    )

            # Validate field_schema dtype consistency across chunks
            for field_name in base_fields:
                base_meta = data[0].field_schema.get(field_name, {})
                base_dtype = base_meta.get("dtype")
                for i, chunk in enumerate(data[1:], start=1):
                    chunk_meta = chunk.field_schema.get(field_name, {})
                    if chunk_meta.get("dtype") != base_dtype:
                        raise ValueError(
                            f"Field '{field_name}' dtype mismatch in concat: "
                            f"chunk[0]={base_dtype}, chunk[{i}]={chunk_meta.get('dtype')}"
                        )

        all_global_indexes = list(itertools.chain.from_iterable(chunk.global_indexes for chunk in data))
        all_partition_ids = list(itertools.chain.from_iterable(chunk.partition_ids for chunk in data))

        all_production_status = np.concatenate([chunk.production_status for chunk in data])

        all_field_schema: dict[str, dict[str, Any]] = {}
        first_chunk = data[0]
        for field_name, meta in first_chunk.field_schema.items():
            # Check if any chunk marks this field as nested
            any_nested = any(chunk.field_schema.get(field_name, {}).get("is_nested", False) for chunk in data)
            merged_is_nested = meta.get("is_nested", False) or any_nested

            all_field_schema[field_name] = {
                "dtype": meta.get("dtype"),
                "shape": None if merged_is_nested else meta.get("shape"),
                "is_nested": merged_is_nested,
                "is_non_tensor": meta.get("is_non_tensor", False),
            }

            # Build per_sample_shapes when the merged field is nested or any chunk already has them
            if merged_is_nested or any(
                chunk.field_schema.get(field_name, {}).get("per_sample_shapes") for chunk in data
            ):
                all_shapes = []
                for chunk in data:
                    chunk_meta = chunk.field_schema.get(field_name, {})
                    chunk_shapes = chunk_meta.get("per_sample_shapes")
                    if chunk_shapes:
                        all_shapes.extend(chunk_shapes)
                    else:
                        # Non-nested chunk: expand the uniform shape for each sample
                        uniform_shape = chunk_meta.get("shape")
                        all_shapes.extend([uniform_shape] * chunk.size)
                all_field_schema[field_name]["per_sample_shapes"] = all_shapes

        all_custom_meta: list[dict[str, Any]] = []
        all_custom_backend_meta: list[dict[str, Any]] = []
        for chunk in data:
            all_custom_meta.extend(chunk.custom_meta)
            all_custom_backend_meta.extend(chunk._custom_backend_meta)

        # Merge extra_info: batch-level metadata with equality check
        all_keys: set[str] = set()
        for chunk in data:
            all_keys.update(chunk.extra_info.keys())

        merged_extra_info = {}
        base_keys = set(data[0].extra_info.keys())

        # Warn if chunks have different key sets
        if any(set(chunk.extra_info.keys()) != base_keys for chunk in data[1:]):
            logger.warning("BatchMeta.concat: extra_info key sets differ across chunks, performing union of keys.")

        for key in all_keys:
            values = [chunk.extra_info[key] for chunk in data if key in chunk.extra_info]
            # Check all values are equal
            first = values[0]
            for i, v in enumerate(values[1:], start=1):
                if not _extra_info_values_equal(first, v):
                    raise ValueError(
                        f"BatchMeta.concat: extra_info key '{key}' has conflicting values "
                        f"across chunks and cannot be merged. "
                        f"chunk[0]={first!r}, chunk[{i}]={v!r}"
                    )
            merged_extra_info[key] = first

        return BatchMeta(
            global_indexes=all_global_indexes,
            partition_ids=all_partition_ids,
            field_schema=all_field_schema,
            production_status=all_production_status,
            extra_info=merged_extra_info,
            custom_meta=all_custom_meta,
            _custom_backend_meta=all_custom_backend_meta,
        )

    def reorder(self, indices: list[int]):
        """Reorder the samples in the BatchMeta according to the given indices.
        The operation is performed in-place.
        """
        if len(indices) != self.size:
            raise ValueError(f"Indices length {len(indices)} mismatch batch size {self.size}")

        if len(set(indices)) != self.size:
            raise ValueError("Indices contain duplicates")

        if any(i < 0 or i >= self.size for i in indices):
            raise ValueError(f"Reorder indices must be in range [0, {self.size})")

        self.global_indexes = [self.global_indexes[i] for i in indices]
        self.partition_ids = [self.partition_ids[i] for i in indices]

        self.production_status = self.production_status[indices]

        for field_name, meta in self.field_schema.items():
            if meta.get("per_sample_shapes") is not None:
                meta["per_sample_shapes"] = [meta["per_sample_shapes"][i] for i in indices]

        self.custom_meta = [self.custom_meta[i] for i in indices]
        self._custom_backend_meta = [self._custom_backend_meta[i] for i in indices]

    @classmethod
    def empty(cls, extra_info: Optional[dict[str, Any]] = None) -> "BatchMeta":
        """Create an empty BatchMeta with no samples.

        Args:
            extra_info: Optional additional information to store with the batch

        Returns:
            Empty BatchMeta instance

        Example:
            >>> empty_batch = BatchMeta.empty()
        """
        if extra_info is None:
            extra_info = {}
        return cls(
            global_indexes=[],
            partition_ids=[],
            field_schema={},
            production_status=None,
            extra_info=extra_info,
            custom_meta=[],
            _custom_backend_meta=[],
        )

    def __str__(self):
        return (
            f"BatchMeta(size={self.size}, field_names={self.field_names}, is_ready={self.is_ready}, "
            f"global_indexes={self.global_indexes}, extra_info={self.extra_info})"
        )


# ==================== KV Interface Metadata ====================
@dataclass
class KVBatchMeta:
    """Records the metadata for KV interface."""

    # keys of each sample
    keys: list[str] = dataclasses.field(default_factory=list)

    # sample-level tags
    tags: list[dict] = dataclasses.field(default_factory=list)

    # [optional] partition_id of this batch
    partition_id: Optional[str] = None

    # [optional] fields of each sample
    fields: Optional[list[str]] = None

    # [optional] external information for batch-level information
    extra_info: Optional[dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """Validate all the variables"""
        if len(self.keys) != len(self.tags):
            raise ValueError(f"keys and tags must have same length, but got {len(self.keys)} and {len(self.tags)}")
        if len(self.keys) != len(set(self.keys)):
            raise ValueError("Got duplicated keys.")
        if self.fields is not None:
            if len(self.fields) != len(set(self.fields)):
                raise ValueError("Got duplicated fields.")

        # deepcopy to prevent unexpected behavior after chunk/concat
        self.tags = copy.deepcopy(self.tags)
        self.extra_info = copy.deepcopy(self.extra_info)

        object.__setattr__(self, "_size", len(self.keys))

    @property
    def size(self) -> int:
        """Return the number of samples in this batch"""
        return getattr(self, "_size", 0)

    def __len__(self) -> int:
        """Return the number of samples in this batch."""
        return len(self.keys)

    def __str__(self):
        return f"KVBatchMeta(size={self.size}, field_names={self.fields}, extra_info={self.extra_info})"

    def select_keys(self, keys_to_select: list[str]) -> "KVBatchMeta":
        """Select specific keys from this batch.

        Args:
            keys_to_select (list[str]): List of keys to retain.

        Returns:
            KVBatchMeta: A new KVBatchMeta instance containing only the specified keys.

        Raises:
            ValueError: If duplicate keys exist in input param `keys_to_select`.
            RuntimeError: If `keys_to_select` contains keys that do not exist in this batch.
        """
        if len(set(keys_to_select)) != len(keys_to_select):
            raise ValueError("Contain duplicate keys.")

        non_exist_keys = set(keys_to_select) - set(self.keys)
        if len(non_exist_keys) > 0:
            raise RuntimeError(f"Keys {non_exist_keys} not found in current batch.")

        _keys_to_idx = {key: idx for idx, key in enumerate(self.keys)}

        loc_idx = [_keys_to_idx[k] for k in keys_to_select]
        tags = [self.tags[i] for i in loc_idx]

        return KVBatchMeta(
            keys=keys_to_select,
            tags=tags,
            partition_id=self.partition_id,
            fields=self.fields,
            extra_info=self.extra_info,
        )

    def reorder(self, indexes: list[int]):
        """Reorder the samples in this batch according to the specified indexes.

        The operation is performed in-place.

        Args:
            indexes : list[int]
                A list of integers specifying the new order of samples.

        Raises:
            ValueError: If the size of input `indexes` does not match with the batch size.
            ValueError: If duplicate indexes exist in input param `indexes`.
        """
        if len(indexes) != self.size:
            raise ValueError(
                f"Attempted to reorder with indexes length {len(indexes)} that does not match "
                f"the batch size {self.size}."
            )

        if len(set(indexes)) != len(indexes):
            raise ValueError("Contain duplicate indexes.")

        self.keys = [self.keys[i] for i in indexes]
        self.tags = [self.tags[i] for i in indexes]

    def chunk(self, chunks: int) -> list["KVBatchMeta"]:
        """Split this batch into smaller chunks.

        Args:
            chunks: number of chunks

        Return:
            List of smaller KVBatchMeta chunks
        """
        chunk_list = []
        if self.size < chunks:
            logger.warning(
                f"Chunk size {chunks} > number of samples in this batch {self.size}, this will return some "
                f"empty KVBatchMeta chunks."
            )

        # Calculate the base size and remainder of each chunk
        base_size = self.size // chunks
        remainder = self.size % chunks

        start = 0
        for i in range(chunks):
            current_chunk_size = base_size + 1 if i < remainder else base_size
            end = start + current_chunk_size
            chunk_keys = self.keys[start:end]
            chunk_tags = self.tags[start:end]

            chunk = KVBatchMeta(
                keys=chunk_keys,
                tags=chunk_tags,
                partition_id=self.partition_id,
                fields=self.fields,
                extra_info=self.extra_info,
            )
            chunk_list.append(chunk)
            start = end

        return chunk_list

    @classmethod
    def concat(cls, data: list["KVBatchMeta"]) -> "KVBatchMeta":
        """Concatenate multiple KVBatchMeta chunks into one large batch.

        Args:
            data: List of KVBatchMeta chunks to concatenate

        Returns:
            Concatenated KVBatchMeta

        Raises:
            ValueError: If validation fails (e.g., field names do not match)
        """
        if not data:
            logger.warning("Try to concat empty KVBatchMeta chunks. Returning empty KVBatchMeta.")
            return KVBatchMeta()

        # skip empty chunks
        data = [chunk for chunk in data if chunk and chunk.size > 0]

        if len(data) == 0:
            logger.warning("No valid KVBatchMeta chunks to concatenate. Returning empty KVBatchMeta.")
            return KVBatchMeta()

        base_fields = data[0].fields
        if base_fields is not None:
            base_fields_set = set(base_fields)
        else:
            base_fields_set = set()

        base_partition_id = data[0].partition_id

        all_keys = []
        all_tags = []

        for chunk in data:
            if chunk.fields is not None and set(chunk.fields) != base_fields_set:
                raise ValueError("Field names do not match for concatenation.")
            if chunk.partition_id != base_partition_id:
                raise ValueError("Partition do not match for concatenation.")

            all_keys.extend(chunk.keys)
            all_tags.extend(chunk.tags)

        # Merge extra_info with conflict detection
        all_extra_keys: set[str] = set()
        for chunk in data:
            if chunk.extra_info:
                all_extra_keys.update(chunk.extra_info.keys())

        all_extra_info = {}
        if all_extra_keys:
            base_info_keys = set(data[0].extra_info.keys()) if data[0].extra_info else set()
            for chunk in data[1:]:
                chunk_keys = set(chunk.extra_info.keys()) if chunk.extra_info else set()
                if chunk_keys != base_info_keys:
                    logger.warning(
                        "KVBatchMeta.concat: extra_info key sets differ across chunks, performing union of keys."
                    )
                    break

            for key in all_extra_keys:
                values = [chunk.extra_info[key] for chunk in data if chunk.extra_info and key in chunk.extra_info]
                first = values[0]
                for i, v in enumerate(values[1:], start=1):
                    if not _extra_info_values_equal(first, v):
                        raise ValueError(
                            f"KVBatchMeta.concat: extra_info key '{key}' has conflicting values across chunks."
                        )
                all_extra_info[key] = first

        return KVBatchMeta(
            keys=all_keys,
            tags=all_tags,
            partition_id=base_partition_id,
            fields=base_fields,
            extra_info=all_extra_info if all_extra_info else None,
        )
