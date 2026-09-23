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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pickle
import shutil
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import numpy as np
import psutil
import ray
import torch
import zmq

from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads, log_heavy_operation
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.zmq_utils import (
    STORAGE_CLIENT_IDENTITY_PREFIXES,
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    format_zmq_address,
    frame_nbytes,
    get_free_port,
    get_node_ip_address,
)

if TYPE_CHECKING:
    from transfer_queue.metrics import TQMetricsExporter
    from transfer_queue.utils.accept_probe import AcceptQueueProbe

logger = get_logger(__name__)

TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))  # in seconds
TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))
DEFAULT_SSD_OFFLOAD_THRESHOLD_BYTES = 1024 * 1024
DEFAULT_SSD_READ_THREADS = 32
DEFAULT_SSD_WRITE_THREADS = 8
SSD_OFFLOAD_DIRECTORY_NAME = "transfer_queue_ssd_offload"

_HYBRID_CHECKPOINT_FORMAT = "transfer_queue_hybrid_storage_v1"


@dataclass(frozen=True)
class SSDEncodedSample:
    """One sample represented in a form that can be written directly to SSD."""

    payload: memoryview
    codec: str
    dtype: str | np.dtype[Any] | None = None
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _SSDValueRef:
    """Internal reference to one SSD-backed value."""

    path: Path
    size_bytes: int
    codec: str
    dtype: str | np.dtype[Any] | None = None
    shape: tuple[int, ...] | None = None


# Marks a GET_ERROR reply as "the key is gone" so the caller can tell it apart from a real fault.
KEY_NOT_FOUND_MARKER = "TQKeyNotFound"

# Accept-queue depth for the client-facing ROUTER. A full queue loses connections silently.
TQ_STORAGE_ZMQ_BACKLOG = int(os.environ.get("TQ_STORAGE_ZMQ_BACKLOG", 4096))

# Accept-queue sampling period in seconds; 0 disables the probe. Keep sub-second.
TQ_ACCEPT_PROBE_INTERVAL = float(os.environ.get("TQ_ACCEPT_PROBE_INTERVAL", 0))


class StorageKeyNotFoundError(KeyError):
    """Raised when a requested global index is absent from a storage unit.

    Reads and ``clear`` are concurrent by design, so a key returned by ``kv_retrieve_meta`` can be
    cleared before ``get_data`` reaches the storage unit. Callers that tolerate that race catch this
    instead of matching on message text.
    """


class StorageUnitData:
    """Storage unit for managing 2D data structure (samples × fields).

    Uses dict-based storage keyed by global_index instead of pre-allocated list.
    This allows O(1) insert/delete without index translation and avoids capacity bloat.

    Data Structure Example:
        field_data = {
            "field_name1": {global_index_0: item1, global_index_3: item2, ...},
            "field_name2": {global_index_0: item3, global_index_3: item4, ...},
        }
    """

    def __init__(self, storage_size: int | None = None):
        # field_name -> {global_index: data} nested dict
        self.field_data: dict[str, dict] = {}
        # Capacity upper bound (None means unlimited)
        self.storage_size = storage_size
        # Track active global_index keys for O(1) capacity checks
        self._active_keys: set = set()

    @property
    def active_key_count(self) -> int:
        """Number of active keys currently stored."""
        return len(self._active_keys)

    @property
    def ssd_offload_enabled(self) -> bool:
        """Whether this storage data offloads values to SSD."""
        return False

    @property
    def ssd_active_values(self) -> int:
        """Number of active field values stored on SSD."""
        return 0

    @property
    def ssd_active_bytes(self) -> int:
        """Logical bytes held by active SSD-backed values."""
        return 0

    @property
    def ssd_fallback_values_total(self) -> int:
        """Values retained in memory because SSD encoding was unavailable."""
        return 0

    def get_data(self, fields: list[str], global_indexes: list) -> dict[str, list]:
        """Get data by global index keys.

        Args:
            fields: Field names used for getting data.
            global_indexes: Global indexes used as dict keys.

        Returns:
            dict with field names as keys, corresponding data list as values.
        """
        result: dict[str, list] = {}
        for field in fields:
            if field not in self.field_data:
                raise ValueError(
                    f"StorageUnitData get_data: field '{field}' not found. Available: {list(self.field_data.keys())}"
                )
            try:
                result[field] = [self.field_data[field][k] for k in global_indexes]
            except KeyError as e:
                raise StorageKeyNotFoundError(f"StorageUnitData get_data: key {e} not found in field '{field}'") from e
        return result

    def put_data(self, field_data: dict[str, Any], global_indexes: list) -> None:
        """Put data into storage.

        Args:
            field_data: Dict with field names as keys, data list as values.
            global_indexes: Global indexes to use as dict keys.
        """
        # Capacity is enforced per unique sample key, not counted per-field
        if self.storage_size is not None:
            new_global_keys = [k for k in global_indexes if k not in self._active_keys]
            if len(self._active_keys) + len(new_global_keys) > self.storage_size:
                raise ValueError(
                    f"Storage capacity exceeded: {len(self._active_keys)} existing + "
                    f"{len(new_global_keys)} new > {self.storage_size}"
                )
        for f, values in field_data.items():
            if len(values) != len(global_indexes):
                raise ValueError(
                    f"StorageUnitData put_data: field '{f}' values length {len(values)} "
                    f"!= global_indexes length {len(global_indexes)}, length mismatch"
                )
            if f not in self.field_data:
                self.field_data[f] = {}
            field_dict = self.field_data[f]
            for key, val in zip(global_indexes, values, strict=True):
                field_dict[key] = val
        self._active_keys.update(global_indexes)

    def clear(self, keys: list[int]) -> None:
        """Remove data at given global index keys, immediately freeing memory.

        Args:
            keys: Global indexes to remove.
        """
        for f in self.field_data:
            for key in keys:
                self.field_data[f].pop(key, None)
        self._active_keys -= set(keys)

    def save_checkpoint(self, path: str | Path, storage_unit_id: str) -> None:
        """Write in-memory storage state to a checkpoint."""
        state = {
            "storage_unit_id": storage_unit_id,
            "storage_unit_size": self.storage_size,
            "field_data": self.field_data,
            "active_keys": self._active_keys,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path: str | Path) -> int | None:
        """Replace in-memory storage state from a checkpoint."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        if state.get("format") == _HYBRID_CHECKPOINT_FORMAT:
            raise ValueError("An SSD checkpoint requires SSD offload to be enabled")

        checkpoint_size = state["storage_unit_size"]
        restored_field_data = state["field_data"]
        restored_active_keys = set(state["active_keys"])
        self.field_data = restored_field_data
        self._active_keys = restored_active_keys
        return checkpoint_size

    def close(self) -> None:
        """Release resources owned by this storage data."""
        return None


class SSDFileStore:
    """Own and read/write one SSD file per offloaded value."""

    def __init__(
        self,
        ssd_path: str,
        run_id: str,
        unit_id: str,
    ) -> None:
        self._closed = False
        configured_path = Path(ssd_path).resolve()
        if configured_path.exists() and not configured_path.is_dir():
            raise ValueError(f"SSD offload path is not a directory: {configured_path}")
        configured_path.mkdir(parents=True, exist_ok=True)
        self._ssd_root = configured_path / SSD_OFFLOAD_DIRECTORY_NAME
        self._ssd_root.mkdir(exist_ok=True)
        if self._ssd_root.is_symlink() or not self._ssd_root.is_dir():
            raise ValueError(f"SSD offload working path must be a directory, not a symlink: {self._ssd_root}")
        self._base_path = self._ssd_root / run_id / unit_id
        self._base_path.mkdir(parents=True)
        for prefix in range(256):
            (self._base_path / f"{prefix:02x}").mkdir()
        self._read_pool = ThreadPoolExecutor(
            max_workers=DEFAULT_SSD_READ_THREADS,
            thread_name_prefix="tq-ssd-read",
        )
        self._write_pool = ThreadPoolExecutor(
            max_workers=DEFAULT_SSD_WRITE_THREADS,
            thread_name_prefix="tq-ssd-write",
        )

    @staticmethod
    def _write_all_bytes(fd: int, payload: memoryview) -> None:
        """Write the entire payload to an open file descriptor, handling partial writes."""
        view = payload.cast("B")
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("SSDFileStore write returned no progress")
            view = view[written:]

    def _write_value_file(
        self,
        encoded_value: SSDEncodedSample,
    ) -> _SSDValueRef:
        """Atomically write one encoded value to a new SSD file and return its reference."""
        token = uuid4().hex
        directory = self._base_path / token[:2]
        temp_path = directory / f".tmp-{token}"
        final_path = directory / f"{token}.bin"
        try:
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                self._write_all_bytes(fd, encoded_value.payload)
            finally:
                os.close(fd)
            temp_path.rename(final_path)
        except Exception:
            for path in (temp_path, final_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return _SSDValueRef(
            path=final_path,
            size_bytes=encoded_value.payload.nbytes,
            codec=encoded_value.codec,
            dtype=encoded_value.dtype,
            shape=encoded_value.shape,
        )

    def write_values(
        self,
        encoded_values: list[SSDEncodedSample],
    ) -> list[_SSDValueRef]:
        """Write encoded values concurrently in input order and remove completed files if the batch fails."""
        refs: list[_SSDValueRef | None] = [None] * len(encoded_values)
        futures = {
            self._write_pool.submit(self._write_value_file, encoded_value): position
            for position, encoded_value in enumerate(encoded_values)
        }
        first_error: Exception | None = None
        for future, position in futures.items():
            try:
                refs[position] = future.result()
            except Exception as e:
                if first_error is None:
                    first_error = e
        if first_error is not None:
            for ref in refs:
                if ref is not None:
                    self.unlink(ref)
            raise first_error
        return [ref for ref in refs if ref is not None]

    def import_file(self, source: Path, metadata: dict[str, Any]) -> _SSDValueRef:
        """Copy one checkpoint blob into this store without materializing it."""
        token = uuid4().hex
        directory = self._base_path / token[:2]
        temp_path = directory / f".tmp-{token}"
        final_path = directory / f"{token}.bin"
        try:
            shutil.copyfile(source, temp_path)
            actual_size = temp_path.stat().st_size
            if actual_size != metadata["size_bytes"]:
                raise OSError(
                    f"SSD checkpoint blob has {actual_size} bytes, expected {metadata['size_bytes']}: {source}"
                )
            temp_path.rename(final_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return _SSDValueRef(
            path=final_path,
            size_bytes=metadata["size_bytes"],
            codec=metadata["codec"],
            dtype=metadata["dtype"],
            shape=metadata["shape"],
        )

    @staticmethod
    def unlink(entry: _SSDValueRef) -> None:
        """Delete one offloaded value, tolerating cleanup failures."""
        try:
            entry.path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to delete superseded SSD sample {entry.path}: {e}")

    def read_values(self, refs: list[_SSDValueRef]) -> list[bytes]:
        """Read SSD-backed values concurrently and return their payloads in input order."""
        return list(self._read_pool.map(self._read_value_file, refs))

    @staticmethod
    def _read_value_file(ref: _SSDValueRef) -> bytes:
        """Read one SSD value file and verify its size against the stored reference."""
        payload = ref.path.read_bytes()
        if len(payload) != ref.size_bytes:
            raise OSError(
                f"SSDFileStore short read from {ref.path}: expected {ref.size_bytes} bytes, got {len(payload)}"
            )
        return payload

    def close(self) -> None:
        """Stop I/O workers and delete the storage directory."""
        if self._closed:
            return
        self._closed = True
        self._write_pool.shutdown(wait=True)
        self._read_pool.shutdown(wait=True)
        shutil.rmtree(self._base_path, ignore_errors=True)
        try:
            self._base_path.parent.rmdir()
        except OSError:
            pass


class HybridStorageUnitData(StorageUnitData):
    """Store each value inline or as an SSD file reference in one field map."""

    def __init__(
        self,
        storage_size: int | None,
        ssd_path: str,
        run_id: str,
        unit_id: str,
        threshold_bytes: int = DEFAULT_SSD_OFFLOAD_THRESHOLD_BYTES,
    ) -> None:
        if threshold_bytes <= 0:
            raise ValueError("SSD offload threshold must be greater than zero")
        super().__init__(storage_size)
        self._ssd_store = SSDFileStore(ssd_path, run_id, unit_id)
        self._threshold = threshold_bytes
        self._ssd_active_values = 0
        self._ssd_active_bytes = 0
        self._ssd_fallback_values_total = 0

    @property
    def ssd_offload_enabled(self) -> bool:
        """Whether this storage data offloads values to SSD."""
        return True

    @property
    def ssd_active_values(self) -> int:
        """Number of active field values stored on SSD."""
        return self._ssd_active_values

    @property
    def ssd_active_bytes(self) -> int:
        """Logical bytes held by active SSD-backed values."""
        return self._ssd_active_bytes

    @property
    def ssd_fallback_values_total(self) -> int:
        """Values retained in memory because SSD encoding was unavailable."""
        return self._ssd_fallback_values_total

    @staticmethod
    def _sample_from_value(value: Any) -> SSDEncodedSample | None:
        if isinstance(value, torch.Tensor):
            if value.is_nested or value.is_sparse:
                return None
            try:
                tensor = value.detach()
                if tensor.device.type != "cpu":
                    tensor = tensor.cpu()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                payload = memoryview(tensor.flatten().view(torch.uint8).numpy()).cast("B")
            except (RuntimeError, TypeError, ValueError):
                return None
            return SSDEncodedSample(
                payload=payload,
                codec="tensor",
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor.shape),
            )
        if isinstance(value, np.ndarray) and not value.dtype.hasobject:
            try:
                array = value if value.flags["C_CONTIGUOUS"] else np.ascontiguousarray(value)
                payload = memoryview(array.view(np.uint8).ravel()).cast("B")
            except (TypeError, ValueError):
                return None
            return SSDEncodedSample(
                payload=payload,
                codec="numpy",
                dtype=array.dtype,
                shape=tuple(array.shape),
            )
        if isinstance(value, bytes):
            return SSDEncodedSample(payload=memoryview(value), codec="bytes")
        try:
            pickled_payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return None
        return SSDEncodedSample(payload=memoryview(pickled_payload), codec="pickle")

    @staticmethod
    def _decode_sample(raw: bytes, entry: _SSDValueRef) -> Any:
        if entry.codec == "tensor":
            if entry.dtype is None or entry.shape is None:
                raise ValueError("Tensor SSD entry is missing dtype or shape")
            dtype = getattr(torch, entry.dtype)
            return torch.frombuffer(raw, dtype=dtype).view(entry.shape)
        if entry.codec == "numpy":
            if entry.dtype is None or entry.shape is None:
                raise ValueError("NumPy SSD entry is missing dtype or shape")
            return np.frombuffer(raw, dtype=np.dtype(entry.dtype)).reshape(entry.shape)
        if entry.codec == "bytes":
            return raw
        if entry.codec == "pickle":
            return pickle.loads(raw)
        raise ValueError(f"Unsupported SSD codec: {entry.codec}")

    def _prepare_field_values(
        self,
        values: Any,
    ) -> tuple[list[Any], list[_SSDValueRef], int]:
        """Encode one field and write its offloaded values to the active SSD directory."""
        logical_samples = list(values.unbind()) if isinstance(values, torch.Tensor) else list(values)
        encoded_samples = [self._sample_from_value(sample) for sample in logical_samples]
        prepared_values: list[Any] = []
        ssd_positions: list[int] = []
        ssd_samples: list[SSDEncodedSample] = []
        fallback_values = 0
        for position, (value, encoded) in enumerate(zip(logical_samples, encoded_samples, strict=True)):
            if encoded is not None and encoded.payload.nbytes >= self._threshold:
                ssd_positions.append(position)
                ssd_samples.append(encoded)
                prepared_values.append(None)
            else:
                prepared_values.append(value)
                if encoded is None:
                    fallback_values += 1

        entries = self._ssd_store.write_values(ssd_samples)
        for position, entry in zip(ssd_positions, entries, strict=True):
            prepared_values[position] = entry
        return prepared_values, entries, fallback_values

    def put_data(
        self,
        field_data: dict[str, Any],
        global_indexes: list,
    ) -> None:
        """Store each sample in memory or SSD according to its encoded size."""
        if not global_indexes or not field_data:
            super().put_data(field_data, global_indexes)
            return

        unique_global_indexes = set(global_indexes)
        has_duplicate_indexes = len(unique_global_indexes) != len(global_indexes)

        for field, values in field_data.items():
            prepared_values, entries, fallback_values = self._prepare_field_values(values)

            stored_field = self.field_data.get(field, {})
            old_ssd_values = []
            for global_index in unique_global_indexes:
                old_value = stored_field.get(global_index)
                if isinstance(old_value, _SSDValueRef):
                    old_ssd_values.append(old_value)
            try:
                super().put_data({field: prepared_values}, global_indexes)
            except Exception:
                for entry in entries:
                    self._ssd_store.unlink(entry)
                raise

            obsolete_ssd_values = old_ssd_values
            if has_duplicate_indexes:
                retained_ssd_paths = set()
                for global_index in unique_global_indexes:
                    retained_value = self.field_data[field][global_index]
                    if isinstance(retained_value, _SSDValueRef):
                        retained_ssd_paths.add(retained_value.path)
                obsolete_ssd_values.extend(entry for entry in entries if entry.path not in retained_ssd_paths)

            self._ssd_active_values += len(entries) - len(obsolete_ssd_values)
            self._ssd_active_bytes += sum(entry.size_bytes for entry in entries) - sum(
                value.size_bytes for value in obsolete_ssd_values
            )
            self._ssd_fallback_values_total += fallback_values
            for obsolete_value in obsolete_ssd_values:
                self._ssd_store.unlink(obsolete_value)

    def get_data(self, fields: list[str], global_indexes: list) -> dict[str, list]:
        """Read mixed memory- and SSD-backed samples in request order."""
        result = super().get_data(fields, global_indexes)
        for values in result.values():
            ssd_values = [(position, value) for position, value in enumerate(values) if isinstance(value, _SSDValueRef)]
            raw_values = self._ssd_store.read_values([value for _, value in ssd_values])
            for (position, entry), raw in zip(ssd_values, raw_values, strict=True):
                values[position] = self._decode_sample(raw, entry)
        return result

    def clear(self, keys: list) -> None:
        """Remove values and unlink any files they reference."""
        ssd_values: list[_SSDValueRef] = []
        for values in self.field_data.values():
            for key in set(keys):
                value = values.get(key)
                if isinstance(value, _SSDValueRef):
                    ssd_values.append(value)
        super().clear(keys)
        self._ssd_active_values -= len(ssd_values)
        self._ssd_active_bytes -= sum(value.size_bytes for value in ssd_values)
        for value in ssd_values:
            self._ssd_store.unlink(value)

    def save_checkpoint(self, path: str | Path, storage_unit_id: str) -> None:
        """Write memory values to a manifest and copy SSD values beside it."""
        manifest_path = Path(path)
        blob_dir = Path(f"{manifest_path}.blobs")
        shutil.rmtree(blob_dir, ignore_errors=True)
        checkpoint_fields: dict[str, dict[int, Any]] = {}
        ssd_index: dict[str, dict[int, dict[str, Any]]] = {}
        try:
            for field, values in self.field_data.items():
                checkpoint_values = {}
                field_ssd_index = {}
                for global_index, value in values.items():
                    if not isinstance(value, _SSDValueRef):
                        checkpoint_values[global_index] = value
                        continue

                    blob_dir.mkdir(parents=True, exist_ok=True)
                    filename = value.path.name
                    destination = blob_dir / filename
                    shutil.copyfile(value.path, destination)
                    actual_size = destination.stat().st_size
                    if actual_size != value.size_bytes:
                        raise OSError(f"SSD value has {actual_size} bytes, expected {value.size_bytes}: {value.path}")
                    field_ssd_index[global_index] = {
                        "filename": filename,
                        "size_bytes": value.size_bytes,
                        "codec": value.codec,
                        "dtype": value.dtype,
                        "shape": value.shape,
                    }
                checkpoint_fields[field] = checkpoint_values
                if field_ssd_index:
                    ssd_index[field] = field_ssd_index

            state = {
                "format": _HYBRID_CHECKPOINT_FORMAT,
                "storage_unit_id": storage_unit_id,
                "storage_unit_size": self.storage_size,
                "field_data": checkpoint_fields,
                "ssd_index": ssd_index,
                "active_keys": set(self._active_keys),
            }
            with open(manifest_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            manifest_path.unlink(missing_ok=True)
            shutil.rmtree(blob_dir, ignore_errors=True)
            raise

    def load_checkpoint(self, path: str | Path) -> int | None:
        """Replace memory and SSD state after all checkpoint data is ready."""
        manifest_path = Path(path)
        with open(manifest_path, "rb") as f:
            state = pickle.load(f)
        checkpoint_format = state.get("format")
        if checkpoint_format not in (None, _HYBRID_CHECKPOINT_FORMAT):
            raise ValueError(f"Unsupported HybridStorageUnitData checkpoint format: {checkpoint_format}")
        checkpoint_size = state["storage_unit_size"]
        field_data = state["field_data"]
        restored_active_keys = set(state["active_keys"])
        old_ssd_values = [
            value for values in self.field_data.values() for value in values.values() if isinstance(value, _SSDValueRef)
        ]
        restored_ssd_values: list[_SSDValueRef] = []
        restored_fallback_values = 0

        try:
            if checkpoint_format == _HYBRID_CHECKPOINT_FORMAT:
                blob_dir = Path(f"{manifest_path}.blobs")
                restored_field_data = {field: dict(values) for field, values in field_data.items()}
                for field, entries in state["ssd_index"].items():
                    restored_values = restored_field_data.setdefault(field, {})
                    for global_index, metadata in entries.items():
                        filename = metadata["filename"]
                        if Path(filename).name != filename:
                            raise ValueError(f"Invalid SSD checkpoint blob name: {filename}")
                        restored_value = self._ssd_store.import_file(blob_dir / filename, metadata)
                        restored_ssd_values.append(restored_value)
                        restored_values[global_index] = restored_value
            else:
                restored_field_data = {}
                for field, values in field_data.items():
                    if not values:
                        restored_field_data[field] = {}
                        continue
                    indexes = list(values)
                    prepared_values, entries, fallback_values = self._prepare_field_values(
                        [values[key] for key in indexes]
                    )
                    restored_ssd_values.extend(entries)
                    restored_fallback_values += fallback_values
                    restored_field_data[field] = dict(zip(indexes, prepared_values, strict=True))
            restored_ssd_active_values = len(restored_ssd_values)
            restored_ssd_active_bytes = sum(value.size_bytes for value in restored_ssd_values)
        except Exception:
            for value in restored_ssd_values:
                self._ssd_store.unlink(value)
            raise

        self.field_data = restored_field_data
        self._active_keys = restored_active_keys
        self._ssd_active_values = restored_ssd_active_values
        self._ssd_active_bytes = restored_ssd_active_bytes
        self._ssd_fallback_values_total += restored_fallback_values
        for value in old_ssd_values:
            self._ssd_store.unlink(value)
        return checkpoint_size

    def close(self) -> None:
        """Release SSD resources owned by this hybrid store."""
        self._ssd_store.close()


@ray.remote(num_cpus=1)
class SimpleStorageUnit:
    """A storage unit that provides distributed data storage functionality.

    This class represents a storage unit that can store data in a 2D structure
    (samples, data_fields) and provides ZMQ-based communication for put/get/clear operations.

    Note: We use Ray decorator (@ray.remote) only for initialization purposes.
    We do NOT use Ray's .remote() call capabilities - the storage unit runs
    as a standalone process with its own ZMQ server socket.

    Attributes:
        storage_unit_id: Unique identifier for this storage unit.
        storage_unit_size: Maximum number of elements that can be stored.
        storage_data: Internal StorageUnitData instance for data management.
        zmq_server_info: ZMQ connection information for clients.
    """

    _requests_arrived = 0
    _arrivals_by_op: dict[str, int] = {}
    _accept_probe = None

    def __init__(self, config: dict[str, Any]):
        """Initialize a SimpleStorageUnit from the SimpleStorage config.

        Args:
            config: The ``backend.SimpleStorage`` configuration. Bootstrap adds one
                shared internal run ID.
        """
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4().hex[:8]}"
        total_storage_size = config.get("total_storage_size")
        num_data_storage_units = int(config.get("num_data_storage_units", 1))
        self.storage_unit_size = (
            math.ceil(total_storage_size / num_data_storage_units) if total_storage_size is not None else None
        )
        self.storage_data: StorageUnitData

        ssd_config = config.get("ssd_offload")
        if ssd_config is not None and ssd_config.get("enabled", False):
            ssd_path = ssd_config.get("path")
            if not ssd_path:
                raise ValueError("SimpleStorage SSD offload requires backend.SimpleStorage.ssd_offload.path")
            threshold_bytes = int(ssd_config.get("threshold_bytes", DEFAULT_SSD_OFFLOAD_THRESHOLD_BYTES))
            self.storage_data = HybridStorageUnitData(
                storage_size=self.storage_unit_size,
                ssd_path=str(ssd_path),
                run_id=str(config.get("_run_id") or uuid4().hex),
                unit_id=self.storage_unit_id,
                threshold_bytes=threshold_bytes,
            )
            logger.info(
                f"[{self.storage_unit_id}]: SSD offload enabled — "
                f"path={Path(ssd_path).resolve() / SSD_OFFLOAD_DIRECTORY_NAME}, "
                f"threshold={threshold_bytes} B/sample"
            )
        else:
            self.storage_data = StorageUnitData(self.storage_unit_size)

        self._requests_arrived = 0
        self._arrivals_by_op = {}

        # Internal communication address for proxy and workers
        self._inproc_addr = f"inproc://simple_storage_workers_{self.storage_unit_id}"

        # Shutdown event for graceful termination
        self._shutdown_event = Event()

        # Placeholder for zmq_context, proxy_thread and worker_threads
        self.zmq_context: zmq.Context | None = None
        self.put_get_socket: zmq.Socket | None = None
        self.proxy_thread: Thread | None = None
        self.worker_thread: Thread | None = None

        self._metrics: TQMetricsExporter | None = None

        self._init_zmq_socket()
        self._start_process_put_get()

        # Register finalizer for graceful cleanup when garbage collected
        self._finalizer = weakref.finalize(
            self,
            self._shutdown_resources,
            self._shutdown_event,
            self.worker_thread,
            self.proxy_thread,
            self.zmq_context,
            self.put_get_socket,
            self._accept_probe,
            self.worker_socket,
            self.storage_data,
        )

    def shutdown(self) -> None:
        """Stop request processing and release this storage unit's resources."""
        if self._finalizer.alive:
            self._finalizer()

    def _init_zmq_socket(self) -> None:
        """
        Initialize ZMQ socket connections between storage unit and controller/clients:
        - put_get_socket (ROUTER): Handle put/get requests from clients.
        - worker_socket (DEALER): Backend socket for worker communication.
        """
        self.zmq_context = zmq.Context()
        self._node_ip = get_node_ip_address()

        # Frontend: ROUTER for receiving client requests
        self.put_get_socket = create_zmq_socket(self.zmq_context, zmq.ROUTER, self._node_ip)
        self.put_get_socket.setsockopt(zmq.BACKLOG, TQ_STORAGE_ZMQ_BACKLOG)

        while True:
            try:
                self._put_get_socket_port = get_free_port(ip=self._node_ip)
                self.put_get_socket.bind(format_zmq_address(self._node_ip, self._put_get_socket_port))
                break
            except zmq.ZMQError:
                logger.warning(f"[{self.storage_unit_id}]: Try to bind ZMQ sockets failed, retrying...")
                continue

        if TQ_ACCEPT_PROBE_INTERVAL > 0:
            # Lazy: the probe shells out to ``ss`` on a timer, so keep it out of runs that
            # have not enabled it.
            from transfer_queue.utils.accept_probe import AcceptQueueProbe

            self._accept_probe = AcceptQueueProbe(
                port=self._put_get_socket_port,
                owner_id=str(self.storage_unit_id),
                interval_s=TQ_ACCEPT_PROBE_INTERVAL,
            )
            self._accept_probe.start()

        # Backend: DEALER for worker communication (connected via zmq.proxy)
        self.worker_socket = create_zmq_socket(self.zmq_context, zmq.DEALER, self._node_ip)
        self.worker_socket.bind(self._inproc_addr)

        self.zmq_server_info = ZMQServerInfo(
            role=Role.STORAGE,
            id=str(self.storage_unit_id),
            ip=self._node_ip,
            ports={"put_get_socket": self._put_get_socket_port},
        )

    def _start_process_put_get(self) -> None:
        """Start worker threads and ZMQ proxy for handling requests."""

        # Start worker thread
        self.worker_thread = Thread(
            target=self._worker_routine,
            name=f"StorageUnitWorkerThread-{self.storage_unit_id}",
            daemon=True,
        )
        self.worker_thread.start()

        time.sleep(0.5)  # make sure worker thread is ready before zmq.proxy forwarding messages

        # Start proxy thread (ROUTER <-> DEALER)
        self.proxy_thread = Thread(
            target=self._proxy_routine,
            name=f"StorageUnitProxyThread-{self.storage_unit_id}",
            daemon=True,
        )
        self.proxy_thread.start()

    def _proxy_routine(self) -> None:
        """ZMQ proxy for message forwarding between frontend ROUTER and backend DEALER."""
        logger.info(f"[{self.storage_unit_id}]: start ZMQ proxy...")
        assert self.put_get_socket is not None, "put_get_socket is not properly initialized"
        front, back = self.put_get_socket, self.worker_socket
        poller = zmq.Poller()
        poller.register(front, zmq.POLLIN)
        poller.register(back, zmq.POLLIN)
        try:
            # Forwarding by hand rather than via zmq.proxy() so a non-TransferQueue peer on
            # this exposed TCP port cannot reach the worker and terminate its request loop.
            while not self._shutdown_event.is_set():
                events = dict(poller.poll(1000))
                if front in events:
                    messages = front.recv_multipart(copy=False)
                    try:
                        identity = bytes(messages[0]) if messages else b""
                        if not identity.startswith(STORAGE_CLIENT_IDENTITY_PREFIXES):
                            logger.warning(
                                "[%s]: dropping request with unrecognized ZMQ identity",
                                self.storage_unit_id,
                            )
                            continue
                        back.send_multipart(messages, copy=False)
                    finally:
                        # This thread outlives each request, so drop forwarded payload frames while idle.
                        del messages

                if back in events:
                    front.send_multipart(back.recv_multipart(copy=False), copy=False)
        except zmq.ContextTerminated:
            logger.info(f"[{self.storage_unit_id}]: ZMQ Proxy stopped gracefully (Context Terminated)")
        except Exception as e:
            if self._shutdown_event.is_set():
                logger.info(f"[{self.storage_unit_id}]: ZMQ Proxy shutting down...")
            else:
                logger.error(f"[{self.storage_unit_id}]: ZMQ Proxy unexpected error: {e}")

    def _worker_routine(self) -> None:
        """Worker thread for processing requests."""

        worker_socket = create_zmq_socket(self.zmq_context, zmq.DEALER, self._node_ip)
        worker_socket.connect(self._inproc_addr)

        poller = zmq.Poller()
        poller.register(worker_socket, zmq.POLLIN)

        logger.info(f"[{self.storage_unit_id}]: worker thread started...")
        perf_monitor = IntervalPerfMonitor(caller_name=f"{self.storage_unit_id}")

        while not self._shutdown_event.is_set():
            monitor = self._metrics if self._metrics is not None else perf_monitor
            try:
                # The event cannot wake a ZMQ poll, so bound idle shutdown latency.
                socks = dict(poller.poll(min(TQ_STORAGE_POLLER_TIMEOUT, 1) * 1000))
            except zmq.error.ContextTerminated:
                # ZMQ context was terminated, exit gracefully
                logger.info(f"[{self.storage_unit_id}]: worker stopped gracefully (Context Terminated)")
                break
            except Exception as e:
                logger.warning(f"[{self.storage_unit_id}]: worker poll error: {e}")
                continue

            if self._shutdown_event.is_set():
                break

            if worker_socket in socks:
                # Keep request handling in a separate scope so payload references are
                # released before the worker waits for the next request.
                self._process_one_worker_request(worker_socket, monitor)

        logger.info(f"[{self.storage_unit_id}]: worker stopped.")
        poller.unregister(worker_socket)
        worker_socket.close(linger=0)

    def _process_one_worker_request(self, worker_socket: zmq.Socket, monitor: Any) -> None:
        """Process one storage request and send its response."""
        # Messages received from proxy: [identity, serialized_msg_frame1, ...]
        messages = worker_socket.recv_multipart(copy=False)
        identity = messages[0]
        serialized_msg = messages[1:]

        try:
            request_msg = ZMQMessage.deserialize(serialized_msg)
        except Exception as e:
            # The identity filter cannot cover this: an allowed peer can still send
            # frames that fail to decode, and decoding here used to kill the thread.
            logger.error(
                f"[{self.storage_unit_id}]: undecodable request from "
                f"identity={bytes(identity)!r}: {type(e).__name__}: {e}"
            )
            error_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_GET_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"undecodable request: {type(e).__name__}: {e}"},
            )
            worker_socket.send_multipart([identity] + error_msg.serialize(), copy=False)
            return

        operation = request_msg.request_type
        started = time.perf_counter()

        try:
            self._requests_arrived += 1
            self._arrivals_by_op[operation.name] = self._arrivals_by_op.get(operation.name, 0) + 1

            logger.debug(f"[{self.storage_unit_id}]: worker received operation: {operation}")

            if operation == ZMQRequestType.PUT_DATA:  # type: ignore[arg-type]
                with monitor.measure(op_type="PUT_DATA"):
                    response_msg = self._handle_put(request_msg)
            elif operation == ZMQRequestType.GET_DATA:  # type: ignore[arg-type]
                with monitor.measure(op_type="GET_DATA"):
                    response_msg = self._handle_get(request_msg)
            elif operation == ZMQRequestType.CLEAR_DATA:  # type: ignore[arg-type]
                with monitor.measure(op_type="CLEAR_DATA"):
                    response_msg = self._handle_clear(request_msg)
            elif operation == ZMQRequestType.GET_METRICS:  # type: ignore[arg-type]
                response_msg = self._handle_get_metrics()
            elif operation == ZMQRequestType.SAVE_STORAGE_CHECKPOINT:  # type: ignore[arg-type]
                response_msg = self._handle_save_checkpoint(request_msg)
            elif operation == ZMQRequestType.LOAD_STORAGE_CHECKPOINT:  # type: ignore[arg-type]
                response_msg = self._handle_load_checkpoint(request_msg)
            else:
                response_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.PUT_GET_OPERATION_ERROR,  # type: ignore[arg-type]
                    sender_id=self.storage_unit_id,
                    body={
                        "message": f"Storage unit id #{self.storage_unit_id} receive invalid operation: {operation}."
                    },
                )
        except Exception as e:
            logger.error(
                f"[{self.storage_unit_id}]: worker error during {operation} "
                f"from sender={request_msg.sender_id}: {type(e).__name__}: {e}"
            )
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_GET_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"{self.storage_unit_id}, worker encountered error "
                    f"during operation {operation}: {str(e)}."
                },
            )

        response_frames = response_msg.serialize()
        if operation == ZMQRequestType.GET_DATA:  # type: ignore[arg-type]
            # This end serializes the get response, so its frames give the true wire size.
            log_heavy_operation(
                self.storage_unit_id,
                "get",
                time.perf_counter() - started,
                sum(frame_nbytes(frame) or 0 for frame in response_frames),
                f"samples={len(request_msg.body.get('global_indexes', []))} "
                f"fields={list(request_msg.body.get('fields', []))}",
            )
        worker_socket.send_multipart([identity] + response_frames, copy=False)

    def _handle_put(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle put request, add or update data into storage unit.

        Args:
            data_parts: ZMQMessage from client.

        Returns:
            Put data success response ZMQMessage.
        """
        try:
            global_indexes = data_parts.body["global_indexes"]
            field_data = data_parts.body["data"]  # field_data should be a dict.
            data_parser = data_parts.body.get("data_parser", None)

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_put"
            ):
                if data_parser is not None:
                    if not callable(data_parser):
                        raise TypeError(f"data_parser must be callable, got {type(data_parser).__name__}")

                    original_keys = set(field_data.keys())
                    original_lengths = {}
                    for k, v in field_data.items():
                        if hasattr(v, "shape") and isinstance(v.shape, tuple | list) and len(v.shape) > 0:
                            original_lengths[k] = v.shape[0]
                        else:
                            try:
                                original_lengths[k] = len(v)
                            except Exception:
                                original_lengths[k] = None

                    field_data = data_parser(field_data)

                    if not isinstance(field_data, dict):
                        raise TypeError(f"data_parser must return a dict, got {type(field_data).__name__}")

                    new_keys = set(field_data.keys())
                    if new_keys != original_keys:
                        raise ValueError(
                            f"data_parser must not change dict keys. "
                            f"Original keys: {sorted(original_keys)}, got: {sorted(new_keys)}"
                        )

                    for k, v in field_data.items():
                        if hasattr(v, "shape") and isinstance(v.shape, tuple | list) and len(v.shape) > 0:
                            new_len = v.shape[0]
                        else:
                            try:
                                new_len = len(v)
                            except Exception:
                                new_len = None

                        orig_len = original_lengths[k]
                        if orig_len is not None and new_len is not None and orig_len != new_len:
                            raise ValueError(
                                f"data_parser changed the number of elements for key '{k}': "
                                f"expected {orig_len}, got {new_len}"
                            )
                self.storage_data.put_data(field_data, global_indexes)

            # After put operation finish, send a message to the client
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.PUT_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={},
            )

            return response_msg
        except Exception as e:
            return ZMQMessage.create(
                request_type=ZMQRequestType.PUT_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"Failed to put data into storage unit id "
                    f"#{self.storage_unit_id}, detail error message: {str(e)}"
                },
            )

    def _handle_get(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle get request, return data from storage unit.

        Args:
            data_parts: ZMQMessage from client.

        Returns:
            Get data success response ZMQMessage, containing target data.
        """
        try:
            fields = data_parts.body["fields"]
            global_indexes = data_parts.body["global_indexes"]

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_get"
            ):
                result_data = self.storage_data.get_data(fields, global_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "data": result_data,
                },
            )
        except Exception as e:
            key_not_found = isinstance(e, StorageKeyNotFoundError)
            log = logger.debug if key_not_found else logger.error
            log(
                f"[{self.storage_unit_id}]: _handle_get error, "
                f"fields={fields}, global_indexes={global_indexes}: {type(e).__name__}: {e}"
            )
            marker = f"[{KEY_NOT_FOUND_MARKER}] " if key_not_found else ""
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"{marker}Failed to get data from storage unit id #{self.storage_unit_id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def _handle_clear(self, data_parts: ZMQMessage) -> ZMQMessage:
        """
        Handle clear request, clear data in storage unit according to given global_indexes.

        Args:
            data_parts: ZMQMessage from client, including target global_indexes.

        Returns:
            Clear data success response ZMQMessage.
        """
        try:
            global_indexes = data_parts.body["global_indexes"]

            with limit_pytorch_auto_parallel_threads(
                target_num_threads=TQ_NUM_THREADS, info=f"[{self.storage_unit_id}] _handle_clear"
            ):
                self.storage_data.clear(global_indexes)

            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"message": f"Clear data in storage unit id #{self.storage_unit_id} successfully."},
            )
        except Exception as e:
            response_msg = ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_ERROR,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={
                    "message": f"Failed to clear data in storage unit id #{self.storage_unit_id}, "
                    f"detail error message: {str(e)}"
                },
            )
        return response_msg

    def _handle_get_metrics(self) -> ZMQMessage:
        """Handle GET_METRICS request by returning storage unit statistics.

        Returns:
            ZMQMessage containing storage unit ID, capacity, active keys,
            process RSS memory, and per-operation request stats.
        """
        try:
            process_rss = psutil.Process().memory_info().rss
        except Exception:
            process_rss = 0

        metrics = {
            "storage_unit_id": self.storage_unit_id,
            "capacity": self.storage_unit_size,
            "active_keys": self.storage_data.active_key_count,
            "process_rss_bytes": process_rss,
            "ssd_offload_enabled": int(self.storage_data.ssd_offload_enabled),
            "ssd_active_values": self.storage_data.ssd_active_values,
            "ssd_active_bytes": self.storage_data.ssd_active_bytes,
            "ssd_fallback_values_total": self.storage_data.ssd_fallback_values_total,
            # Counted on arrival; op_stats below only advances on completion.
            "requests_arrived": self._requests_arrived,
            "arrivals_by_op": dict(self._arrivals_by_op),
        }
        if self._accept_probe is not None:
            stats = self._accept_probe.stats
            metrics["accept_queue"] = {
                "backlog": stats.backlog,
                "peak_recv_q": stats.peak_recv_q,
                "peak_utilization": stats.peak_utilization,
                "sk_drops_delta": stats.sk_drops_delta,
                "listen_overflow_delta": stats.overflow_delta,
                "non_overflow_drop_delta": stats.non_overflow_drop_delta,
                "samples": stats.samples,
            }

        # Include per-operation stats if Prometheus metrics are enabled
        if self._metrics is not None:
            op_stats = {}
            for op_type in ("PUT_DATA", "GET_DATA", "CLEAR_DATA"):
                try:
                    hist = self._metrics.request_duration.labels(op_type=op_type)
                    counter = self._metrics.request_total.labels(op_type=op_type)
                    duration_sum = hist._sum.get()
                    # Build cumulative counts once, reuse for total and quantiles
                    cumulative_counts = self._cumulative_bucket_counts(hist)
                    duration_count = cumulative_counts[-1] if cumulative_counts else 0
                    op_stats[op_type] = {
                        "request_count": counter._value.get(),
                        "latency_avg": duration_sum / duration_count if duration_count > 0 else 0,
                        "latency_p50": self._quantile_from_cumulative(hist, cumulative_counts, 0.50),
                        "latency_p99": self._quantile_from_cumulative(hist, cumulative_counts, 0.99),
                    }
                except (AttributeError, TypeError, ZeroDivisionError) as e:
                    logger.debug(f"[{self.storage_unit_id}]: Failed to extract metrics for {op_type}: {e}")
            if op_stats:
                metrics["op_stats"] = op_stats

        return ZMQMessage.create(
            request_type=ZMQRequestType.METRICS_RESPONSE,  # type: ignore[arg-type]
            sender_id=self.storage_unit_id,
            body=metrics,
        )

    def _handle_save_checkpoint(self, data_parts) -> ZMQMessage:
        """Write storage unit data directly to its checkpoint path.

        Args:
            data_parts: ZMQMessage from client, containing ``path`` in body:
                absolute path for the output manifest. The caller must ensure
                this path is reachable from the node running this actor
                (shared filesystem required for multi-node setups).

        Returns:
            ZMQMessage with ``success=True`` on success, or ``success=False``
            and ``message`` containing the error string on failure.
        """
        path = data_parts.body["path"]
        try:
            self.storage_data.save_checkpoint(path, self.storage_unit_id)
            logger.info(f"[{self.storage_unit_id}]: saved checkpoint to {path}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.SAVE_STORAGE_CHECKPOINT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"success": True},
            )
        except Exception as e:
            logger.error(f"[{self.storage_unit_id}]: save checkpoint failed: {e}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.SAVE_STORAGE_CHECKPOINT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"success": False, "message": str(e)},
            )

    def _handle_load_checkpoint(self, data_parts) -> ZMQMessage:
        """Restore storage unit data directly from its checkpoint path.

        Args:
            data_parts: ZMQMessage from client, containing ``path`` in body:
                absolute path to a .pkl file previously written by
                ``_handle_save_checkpoint``. The caller must ensure this path
                is reachable from the node running this actor
                (shared filesystem required for multi-node setups).

        Returns:
            ZMQMessage with ``success=True`` on success, or ``success=False``
            and ``message`` containing the error string on failure.
        """
        path = data_parts.body["path"]
        try:
            previous_key_count = self.storage_data.active_key_count
            checkpoint_size = self.storage_data.load_checkpoint(path)

            if checkpoint_size != self.storage_unit_size:
                logger.warning(
                    f"[{self.storage_unit_id}]: storage_unit_size mismatch — "
                    f"checkpoint={checkpoint_size}, current={self.storage_unit_size}"
                )

            if previous_key_count:
                logger.warning(
                    f"[{self.storage_unit_id}]: overwriting {previous_key_count} "
                    f"existing keys with checkpoint data from {path}"
                )

            logger.info(
                f"[{self.storage_unit_id}]: loaded checkpoint from {path} — "
                f"{self.storage_data.active_key_count} keys, {len(self.storage_data.field_data)} fields"
            )
            return ZMQMessage.create(
                request_type=ZMQRequestType.LOAD_STORAGE_CHECKPOINT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"success": True},
            )

        except Exception as e:
            logger.error(f"[{self.storage_unit_id}]: load checkpoint failed: {e}")
            return ZMQMessage.create(
                request_type=ZMQRequestType.LOAD_STORAGE_CHECKPOINT_RESPONSE,  # type: ignore[arg-type]
                sender_id=self.storage_unit_id,
                body={"success": False, "message": str(e)},
            )

    @staticmethod
    def _cumulative_bucket_counts(hist) -> list[float]:
        """Build cumulative counts from a prometheus_client Histogram's non-cumulative buckets."""
        cumulative = 0.0
        counts = []
        for bucket in hist._buckets:
            cumulative += bucket.get()
            counts.append(cumulative)
        return counts

    @staticmethod
    def _quantile_from_cumulative(hist, cumulative_counts: list[float], q: float) -> float:
        """Estimate a quantile using pre-computed cumulative bucket counts.

        Uses linear interpolation matching Prometheus histogram_quantile() logic.
        """
        total = cumulative_counts[-1] if cumulative_counts else 0
        if total == 0:
            return 0.0
        target = q * total
        prev_bound = 0.0
        prev_cumulative = 0.0
        for bound, cum_count in zip(hist._upper_bounds, cumulative_counts, strict=False):
            if cum_count >= target:
                fraction = (
                    (target - prev_cumulative) / (cum_count - prev_cumulative) if cum_count > prev_cumulative else 0
                )
                return prev_bound + (bound - prev_bound) * fraction
            prev_bound = bound
            prev_cumulative = cum_count
        return prev_bound

    @staticmethod
    def _shutdown_resources(
        shutdown_event: Event,
        worker_thread: Thread | None,
        proxy_thread: Thread | None,
        zmq_context: zmq.Context | None,
        put_get_socket: zmq.Socket | None,
        accept_probe: "AcceptQueueProbe | None" = None,
        worker_socket: zmq.Socket | None = None,
        storage_data: StorageUnitData | None = None,
    ) -> None:
        """Clean up resources on garbage collection."""
        logger.info("Shutting down SimpleStorageUnit resources...")

        shutdown_event.set()

        # Before the ZMQ teardown: the probe runs on its own timer and would outlive the unit.
        if accept_probe is not None:
            accept_probe.stop()

        if worker_thread and worker_thread.is_alive():
            worker_thread.join()
        if proxy_thread and proxy_thread.is_alive():
            proxy_thread.join()

        if put_get_socket:
            put_get_socket.close(linger=0)
        if worker_socket:
            worker_socket.close(linger=0)
        if zmq_context:
            zmq_context.term()
        if storage_data is not None:
            try:
                storage_data.close()
            except Exception as e:
                logger.warning(f"Error closing storage data on shutdown: {e}")

        logger.info("SimpleStorageUnit resources shutdown complete.")

    def start_metrics(self, port: int = 0) -> str:
        """Initialize and start the Prometheus metrics exporter for this storage unit.

        When enabled, replaces ``IntervalPerfMonitor`` for request latency/throughput
        tracking with Prometheus counters and histograms.

        Args:
            port: HTTP port for the /metrics endpoint (0 = auto-assign).

        Returns:
            The metrics endpoint address in ``host:port`` format.
        """
        if self._metrics is not None:
            return self._metrics.endpoint
        from transfer_queue.metrics import TQMetricsExporter

        self._metrics = TQMetricsExporter(role="storage")
        endpoint = self._metrics.start(node_ip=self._node_ip, port=port)
        logger.info(f"[{self.storage_unit_id}]: Prometheus metrics exporter started on {endpoint}")
        return endpoint

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Get the ZMQ server information for this storage unit.

        Returns:
            ZMQServerInfo containing connection details for this storage unit.
        """
        return self.zmq_server_info
