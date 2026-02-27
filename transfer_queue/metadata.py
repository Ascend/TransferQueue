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


def _parse_dtype(dtype_str: str) -> Any:
    """Parse a dtype string produced by to_dict() back to a dtype object.

    Supports torch.dtype strings (e.g. "torch.float32") and numpy dtype
    strings (e.g. "float64", "int8").  Falls back to returning the raw
    string if parsing fails so that unknown types are not silently dropped.
    """
    if dtype_str is None:
        return None
    # torch.dtype: repr is "torch.<name>"
    if dtype_str.startswith("torch."):
        name = dtype_str[len("torch.") :]
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype):
            return dtype
    # numpy dtype
    try:
        return np.dtype(dtype_str)
    except TypeError:
        pass
    # Fallback: return as-is (e.g. plain Python type repr like "<class 'int'>")
    return dtype_str


class _SampleView:
    """Lazy read-only view of a single sample row in a columnar BatchMeta."""

    __slots__ = ("_batch", "_idx")

    def __init__(self, batch: "BatchMeta", idx: int) -> None:
        self._batch = batch
        self._idx = idx

    @property
    def fields(self) -> dict:
        """Read-only access to field_schema: batch.samples[i].fields['a'] -> field meta dict."""
        return self._batch.field_schema


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
    # O(F) field-level metadata: {field_name: {dtype, shape, is_nested, is_non_tensor}}
    field_schema: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    # O(B) vectorized production status; always np.ndarray after __post_init__ (never None)
    production_status: np.ndarray = dataclasses.field(default=None, repr=False)  # type: ignore[assignment]
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    # user-defined meta for each sample (sample-level), list aligned with global_indexes
    custom_meta: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    # internal meta for different storage backends (per-sample per-field level), list aligned with global_indexes
    _custom_backend_meta: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """Initialize all computed properties during initialization"""
        self.global_indexes = copy.deepcopy(self.global_indexes)
        self.partition_ids = copy.deepcopy(self.partition_ids)
        self.field_schema = copy.deepcopy(self.field_schema)
        self.extra_info = copy.deepcopy(self.extra_info)

        # Validation
        if len(self.global_indexes) != len(self.partition_ids):
            raise ValueError(
                f"Length mismatch: global_indexes has {len(self.global_indexes)}, "
                f"partition_ids has {len(self.partition_ids)}"
            )

        batch_size = len(self.global_indexes)

        if self.production_status is not None:
            if isinstance(self.production_status, np.ndarray):
                self.production_status = self.production_status.copy()
            elif isinstance(self.production_status, torch.Tensor):
                self.production_status = self.production_status.numpy().copy()
            elif isinstance(self.production_status, list):
                self.production_status = np.array(self.production_status, dtype=np.int8)

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
            self.custom_meta = copy.deepcopy(self.custom_meta)
            if len(self.custom_meta) != batch_size:
                raise ValueError(f"custom_meta length {len(self.custom_meta)} != batch_size {batch_size}")
        if not self._custom_backend_meta:
            self._custom_backend_meta = [{} for _ in range(batch_size)]
        else:
            self._custom_backend_meta = copy.deepcopy(self._custom_backend_meta)
            if len(self._custom_backend_meta) != batch_size:
                raise ValueError(
                    f"_custom_backend_meta length {len(self._custom_backend_meta)} != batch_size {batch_size}"
                )

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

        for name, value in tensor_dict.items():
            # Determine if this is a nested tensor
            is_nested = isinstance(value, torch.Tensor) and value.is_nested

            first_item = None
            if is_nested:
                unbound = value.unbind()
                first_item = unbound[0] if unbound else None
            else:
                first_item = value[0] if len(value) > 0 else None

            # Determine if this is non-tensor data.
            # When first_item is None (empty field), we cannot determine type—leave as None.
            is_non_tensor = not isinstance(first_item, torch.Tensor) if first_item is not None else None

            field_meta = {
                "dtype": getattr(first_item, "dtype", type(first_item) if first_item is not None else None),
                "shape": getattr(first_item, "shape", None) if not is_nested else None,
                "is_nested": is_nested,
                "is_non_tensor": is_non_tensor,
            }

            # For nested tensors, record per-sample shapes
            if is_nested:
                field_meta["per_sample_shapes"] = [tuple(t.shape) for t in value.unbind()]

            self.field_schema[name] = field_meta

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
        for fname, meta in self.field_schema.items():
            new_meta = copy.deepcopy(meta)
            if meta.get("per_sample_shapes") is not None:
                new_meta["per_sample_shapes"] = [meta["per_sample_shapes"][i] for i in sample_indices]
            new_field_schema[fname] = new_meta

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
        for fname in field_names:
            if fname in self.field_schema:
                new_field_schema[fname] = copy.deepcopy(self.field_schema[fname])

        selected_custom_backend_meta = [
            {f: v for f, v in m.items() if f.startswith("_") or f in field_names} for m in self._custom_backend_meta
        ]

        return BatchMeta(
            global_indexes=self.global_indexes,
            partition_ids=self.partition_ids,
            field_schema=new_field_schema,
            production_status=self.production_status.copy(),
            extra_info=self.extra_info,
            custom_meta=self.custom_meta,
            _custom_backend_meta=selected_custom_backend_meta,
        )

    def with_data_fields(self, field_names: list[str]) -> "BatchMeta":
        """Return a new BatchMeta with the given data fields, replacing the current field_schema.

        Unlike ``select_fields``, this method allows specifying field names that are not
        yet present in the current ``field_schema`` (e.g. fields added by a subsequent
        ``put`` call on a subset of samples).  Unknown fields are included in the new
        ``field_schema`` with an empty metadata dict so that ``get_data`` can retrieve
        them from the storage backend.

        Args:
            field_names (list[str]): List of field names to request. May include fields
                not present in the current ``field_schema``.

        Returns:
            BatchMeta: A new BatchMeta instance whose ``field_schema`` contains exactly
            the requested fields (existing metadata is preserved where available).
        """
        new_field_schema = {}
        for fname in field_names:
            if fname in self.field_schema:
                new_field_schema[fname] = copy.deepcopy(self.field_schema[fname])
            else:
                # Unknown field — include with empty schema so get_data can fetch it.
                new_field_schema[fname] = {}

        selected_custom_backend_meta = [
            {f: v for f, v in m.items() if f.startswith("_") or f in field_names} for m in self._custom_backend_meta
        ]

        return BatchMeta(
            global_indexes=self.global_indexes,
            partition_ids=self.partition_ids,
            field_schema=new_field_schema,
            production_status=self.production_status.copy(),
            extra_info=self.extra_info,
            custom_meta=self.custom_meta,
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

            for chunk in data:
                if chunk.field_names != base_fields:
                    raise ValueError("Error: Field names do not match for concatenation.")

            # Validate field_schema dtype and is_nested consistency across chunks
            for fname in base_fields:
                base_meta = data[0].field_schema.get(fname, {})
                base_dtype = base_meta.get("dtype")
                base_is_nested = base_meta.get("is_nested", False)
                for i, chunk in enumerate(data[1:], start=1):
                    chunk_meta = chunk.field_schema.get(fname, {})
                    if chunk_meta.get("dtype") != base_dtype:
                        raise ValueError(
                            f"Field '{fname}' dtype mismatch in concat: "
                            f"chunk[0]={base_dtype}, chunk[{i}]={chunk_meta.get('dtype')}"
                        )
                    if chunk_meta.get("is_nested", False) != base_is_nested:
                        raise ValueError(
                            f"Field '{fname}' is_nested mismatch in concat: "
                            f"chunk[0]={base_is_nested}, chunk[{i}]={chunk_meta.get('is_nested', False)}"
                        )

        all_global_indexes = list(itertools.chain.from_iterable(chunk.global_indexes for chunk in data))
        all_partition_ids = list(itertools.chain.from_iterable(chunk.partition_ids for chunk in data))

        all_production_status = np.concatenate([chunk.production_status for chunk in data])

        all_field_schema: dict[str, dict[str, Any]] = {}
        first_chunk = data[0]
        for fname, meta in first_chunk.field_schema.items():
            all_field_schema[fname] = {
                "dtype": meta.get("dtype"),
                "shape": meta.get("shape"),
                "is_nested": meta.get("is_nested", False),
                "is_non_tensor": meta.get("is_non_tensor", False),
            }
            if any(chunk.field_schema.get(fname, {}).get("per_sample_shapes") for chunk in data):
                all_shapes = []
                for chunk in data:
                    chunk_meta = chunk.field_schema.get(fname, {})
                    chunk_shapes = chunk_meta.get("per_sample_shapes")
                    if chunk_shapes:
                        all_shapes.extend(chunk_shapes)
                    else:
                        all_shapes.extend([None] * chunk.size)
                all_field_schema[fname]["per_sample_shapes"] = all_shapes

        all_custom_meta: list[dict[str, Any]] = []
        all_custom_backend_meta: list[dict[str, Any]] = []
        for chunk in data:
            all_custom_meta.extend(chunk.custom_meta)
            all_custom_backend_meta.extend(chunk._custom_backend_meta)

        # Merge all extra_info dictionaries from the chunks
        merged_extra_info = dict()

        values_by_key = defaultdict(list)
        for chunk in data:
            for key, value in chunk.extra_info.items():
                values_by_key[key].append(value)
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
                raise TypeError(
                    f"BatchMeta.concat: extra_info key '{key}' has type(s) "
                    f"{[type(v).__name__ for v in values]} which cannot be concatenated. "
                    f"Only list-like iterables support concat: list, torch.Tensor, "
                    f"NonTensorStack/NonTensorData. Scalar types (int, str, float, dict, etc.) "
                    f"have no defined merge strategy—handle '{key}' manually before calling concat."
                )

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

        for fname, meta in self.field_schema.items():
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

    def to_dict(self) -> dict:
        """Convert BatchMeta to dict for serialization.

        dtype is explicitly serialized as a string (e.g. "torch.float32", "float64") so
        that from_dict() can reconstruct it without relying on pickle to transparently
        round-trip torch.dtype / numpy.dtype objects.
        """
        serialized_schema = {}
        for fname, meta in self.field_schema.items():
            dtype = meta.get("dtype")
            serialized_schema[fname] = {
                "dtype": str(dtype) if dtype is not None else None,
                "shape": list(meta["shape"]) if meta.get("shape") else None,
                "is_nested": meta.get("is_nested", False),
                "is_non_tensor": meta.get("is_non_tensor", False),
            }
            if meta.get("per_sample_shapes") is not None:
                serialized_schema[fname]["per_sample_shapes"] = [list(s) for s in meta["per_sample_shapes"]]

        return {
            "global_indexes": self.global_indexes,
            "partition_ids": self.partition_ids,
            "field_schema": serialized_schema,
            "production_status": self.production_status.tolist() if self.production_status is not None else [],
            "extra_info": self.extra_info,
            "custom_meta": self.custom_meta,
            "_custom_backend_meta": self._custom_backend_meta,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchMeta":
        """Create BatchMeta from dictionary.

        dtype is stored as a string and decoded back to torch.dtype / numpy.dtype here.
        """
        field_schema = {}
        for fname, meta in data.get("field_schema", {}).items():
            dtype_str = meta.get("dtype")
            dtype = _parse_dtype(dtype_str) if dtype_str is not None else None
            field_schema[fname] = {
                "dtype": dtype,
                "shape": tuple(meta["shape"]) if meta.get("shape") else None,
                "is_nested": meta.get("is_nested", False),
                "is_non_tensor": meta.get("is_non_tensor", False),
            }
            if meta.get("per_sample_shapes") is not None:
                field_schema[fname]["per_sample_shapes"] = [tuple(s) for s in meta["per_sample_shapes"]]

        ps_data = data.get("production_status")
        production_status: np.ndarray = (
            np.array(ps_data, dtype=np.int8)
            if ps_data is not None
            else np.zeros(len(data["global_indexes"]), dtype=np.int8)
        )

        return cls(
            global_indexes=data["global_indexes"],
            partition_ids=data["partition_ids"],
            field_schema=field_schema,
            production_status=production_status,
            extra_info=data.get("extra_info", {}),
            custom_meta=data.get("custom_meta", []),
            _custom_backend_meta=data.get("_custom_backend_meta", []),
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
        all_extra_info = {}
        for chunk in data:
            if chunk.fields is not None and set(chunk.fields) != base_fields_set:
                raise ValueError("Field names do not match for concatenation.")
            if chunk.partition_id != base_partition_id:
                raise ValueError("Partition do not match for concatenation.")

            all_keys.extend(chunk.keys)
            all_tags.extend(chunk.tags)
            if chunk.extra_info is not None:
                all_extra_info.update(chunk.extra_info)

        return KVBatchMeta(
            keys=all_keys,
            tags=all_tags,
            partition_id=base_partition_id,
            fields=base_fields,
            extra_info=all_extra_info if all_extra_info else None,
        )
