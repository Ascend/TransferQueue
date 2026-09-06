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

import fcntl
import hashlib
import json
import os
import pickle
import re
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

from transfer_queue.utils.common import limit_pytorch_auto_parallel_threads
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.perf_utils import IntervalPerfMonitor
from transfer_queue.utils.serial_utils import DecodedBuffer, DecodeStorageInfo
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

TQ_STORAGE_POLLER_TIMEOUT = int(os.environ.get("TQ_STORAGE_POLLER_TIMEOUT", 5))  # in seconds
TQ_NUM_THREADS = int(os.environ.get("TQ_NUM_THREADS", 8))
TQ_SSD_READ_THREADS = int(os.environ.get("TQ_SSD_READ_THREADS", 32))
TQ_SSD_WRITE_THREADS = int(os.environ.get("TQ_SSD_WRITE_THREADS", 8))
DEFAULT_SSD_OFFLOAD_THRESHOLD_BYTES = 1024 * 1024

_SSD_OWNER_FORMAT = "transfer_queue_ssd_offload_v1"
_SSD_OWNER_FILE = "owner.json"
_SSD_OWNER_LOCK_FILE = "owner.lock"


@dataclass(frozen=True)
class SSDEncodedSample:
    """One sample represented in a form that can be written directly to SSD."""

    payload: memoryview
    codec: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SSDIndexEntry:
    """Location and reconstruction metadata for one SSD-backed sample."""

    path: Path
    length: int
    codec: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None


# Marks a GET_ERROR reply as "the key is gone" so the caller can tell it apart from a real fault.
KEY_NOT_FOUND_MARKER = "TQKeyNotFound"


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


def _field_to_filename(field: str) -> str:
    """Map an arbitrary field name to a filesystem-safe, collision-resistant stem."""
    safe = re.sub(r"[^\w.-]", "_", field).strip("._")[:64] or "field"
    digest = hashlib.sha256(field.encode()).hexdigest()[:16]
    return f"{safe}_{digest}"


def _validate_path_component(value: str, name: str) -> None:
    """Reject values that could escape or ambiguously address an owned directory."""
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{name} must be a non-empty filesystem path component")


def _cleanup_orphaned_ssd_units(ssd_root: Path) -> None:
    """Remove SSD unit directories whose owning process no longer holds its lock."""
    if not ssd_root.is_dir():
        return

    try:
        run_dirs = list(ssd_root.iterdir())
    except OSError as e:
        logger.warning(f"Failed to scan SSD offload root {ssd_root}: {e}")
        return

    for run_dir in run_dirs:
        if not run_dir.is_dir() or run_dir.is_symlink():
            continue
        try:
            unit_dirs = list(run_dir.iterdir())
        except OSError:
            continue
        for unit_dir in unit_dirs:
            if not unit_dir.is_dir() or unit_dir.is_symlink():
                continue

            owner_path = unit_dir / _SSD_OWNER_FILE
            lock_path = unit_dir / _SSD_OWNER_LOCK_FILE
            try:
                owner = json.loads(owner_path.read_text())
                if owner.get("format") != _SSD_OWNER_FORMAT:
                    continue
                lock_file = open(lock_path, "a+b")
            except (OSError, ValueError, TypeError):
                continue

            try:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                shutil.rmtree(unit_dir)
            except OSError as e:
                logger.warning(f"Failed to clean orphaned SSD offload directory {unit_dir}: {e}")
            finally:
                lock_file.close()

        try:
            run_dir.rmdir()
        except OSError:
            pass


class SSDFieldStore:
    """One-file-per-sample SSD storage with atomically published indexes."""

    def __init__(self, ssd_path: str, run_id: str, unit_id: str) -> None:
        _validate_path_component(run_id, "run_id")
        _validate_path_component(unit_id, "unit_id")
        self._ssd_root = Path(ssd_path).resolve()
        self._run_id = run_id
        self._unit_id = unit_id
        self._offset_index: dict[str, dict[int, SSDIndexEntry]] = {}
        self._active_keys: set[int] = set()
        self._closed = False
        self._owner_lock: Any = None

        if not self._ssd_root.parent.is_dir():
            raise ValueError(f"SSD offload parent directory does not exist: {self._ssd_root.parent}")
        self._ssd_root.mkdir(exist_ok=True)
        cleanup_lock_path = self._ssd_root / ".cleanup.lock"
        with open(cleanup_lock_path, "a+b") as cleanup_lock:
            fcntl.flock(cleanup_lock.fileno(), fcntl.LOCK_EX)
            _cleanup_orphaned_ssd_units(self._ssd_root)
            self._base_path = self._create_owned_directory()
        self._read_pool = ThreadPoolExecutor(
            max_workers=max(TQ_SSD_READ_THREADS, 1),
            thread_name_prefix="tq-ssd-read",
        )
        self._write_pool = ThreadPoolExecutor(
            max_workers=max(TQ_SSD_WRITE_THREADS, 1),
            thread_name_prefix="tq-ssd-write",
        )

    def _create_owned_directory(self) -> Path:
        run_dir = self._ssd_root / self._run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        base_path = run_dir / self._unit_id
        temp_path = run_dir / f".tmp-{self._unit_id}-{uuid4().hex}"
        temp_path.mkdir()

        lock_file = None
        try:
            lock_file = open(temp_path / _SSD_OWNER_LOCK_FILE, "a+b")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = {
                "format": _SSD_OWNER_FORMAT,
                "run_id": self._run_id,
                "unit_id": self._unit_id,
                "pid": os.getpid(),
            }
            (temp_path / _SSD_OWNER_FILE).write_text(json.dumps(owner))
            temp_path.rename(base_path)
            self._owner_lock = lock_file
            return base_path
        except Exception:
            if lock_file is not None:
                lock_file.close()
            shutil.rmtree(temp_path, ignore_errors=True)
            raise

    def _sample_directory(self, field: str, global_index: int) -> Path:
        digest = hashlib.sha256(f"{field}\0{global_index}".encode()).hexdigest()
        directory = self._base_path / _field_to_filename(field) / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _write_many(fd: int, payloads: list[memoryview]) -> None:
        """Write multiple payloads without concatenating them in host memory."""
        views = [payload.cast("B") for payload in payloads if payload.nbytes]
        configured_iov_max = os.sysconf("SC_IOV_MAX") if "SC_IOV_MAX" in os.sysconf_names else 1024
        iov_max = max(int(configured_iov_max), 1)
        while views:
            batch = views[:iov_max]
            written = os.writev(fd, batch)
            if written <= 0:
                raise OSError("SSDFieldStore writev returned no progress")

            consumed = 0
            while consumed < len(batch) and written >= batch[consumed].nbytes:
                written -= batch[consumed].nbytes
                consumed += 1
            views = views[consumed:]
            if written:
                views[0] = views[0][written:]

    @property
    def active_key_count(self) -> int:
        """Return the number of global indexes with SSD-backed samples."""
        return len(self._active_keys)

    def _write_sample(
        self,
        field: str,
        global_index: int,
        sample: SSDEncodedSample,
    ) -> SSDIndexEntry:
        directory = self._sample_directory(field, global_index)
        token = uuid4().hex
        temp_path = directory / f".tmp-{token}"
        final_path = directory / f"{token}.bin"
        try:
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                self._write_many(fd, [sample.payload])
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
        return SSDIndexEntry(
            path=final_path,
            length=sample.payload.nbytes,
            codec=sample.codec,
            dtype=sample.dtype,
            shape=sample.shape,
        )

    def prepare_encoded(
        self,
        encoded_fields: dict[str, dict[int, SSDEncodedSample]],
    ) -> dict[str, dict[int, SSDIndexEntry]]:
        """Write new sample files without publishing them to readers."""
        prepared: dict[str, dict[int, SSDIndexEntry]] = {}
        futures = {
            self._write_pool.submit(
                self._write_sample,
                field,
                global_index,
                sample,
            ): (field, global_index)
            for field, samples in encoded_fields.items()
            for global_index, sample in samples.items()
        }
        first_error: Exception | None = None
        for future, (field, global_index) in futures.items():
            try:
                entry = future.result()
                prepared.setdefault(field, {})[global_index] = entry
            except Exception as e:
                if first_error is None:
                    first_error = e
        if first_error is not None:
            self.discard_prepared(prepared)
            raise first_error
        return prepared

    @staticmethod
    def discard_prepared(prepared: dict[str, dict[int, SSDIndexEntry]]) -> None:
        """Delete prepared files after a PUT fails before publication."""
        for entries in prepared.values():
            for entry in entries.values():
                try:
                    entry.path.unlink(missing_ok=True)
                except OSError:
                    pass

    def commit_prepared(self, prepared: dict[str, dict[int, SSDIndexEntry]]) -> None:
        """Publish prepared files, then remove superseded sample files."""
        old_entries: list[SSDIndexEntry] = []
        for field, field_entries in prepared.items():
            index = self._offset_index.setdefault(field, {})
            for global_index, new_entry in field_entries.items():
                old_entry = index.get(global_index)
                index[global_index] = new_entry
                if old_entry is not None:
                    old_entries.append(old_entry)
                self._active_keys.add(global_index)

        for entry in old_entries:
            self._unlink_entry(entry)

    @staticmethod
    def _unlink_entry(entry: SSDIndexEntry) -> None:
        try:
            entry.path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning(f"Failed to delete superseded SSD sample {entry.path}: {e}")

    def remove(self, field: str, global_index: int) -> None:
        """Remove one SSD-backed field sample if it exists."""
        index = self._offset_index.get(field)
        if index is None:
            return
        entry = index.pop(global_index, None)
        if entry is not None:
            self._unlink_entry(entry)
        if not index:
            self._offset_index.pop(field, None)
        if not any(global_index in entries for entries in self._offset_index.values()):
            self._active_keys.discard(global_index)

    def get_data(self, fields: list[str], global_indexes: list) -> dict[str, list]:
        """Read per-sample values from their independently owned files."""
        result: dict[str, list] = {}
        for field in fields:
            if field not in self._offset_index:
                raise ValueError(
                    f"SSDFieldStore get_data: field '{field}' not found. Available: {list(self._offset_index.keys())}"
                )
            idx_map = self._offset_index[field]
            entries = []
            for gidx in global_indexes:
                if gidx not in idx_map:
                    raise KeyError(f"SSDFieldStore get_data: key {gidx} not found in field '{field}'")
                entries.append((field, gidx, idx_map[gidx]))
            result[field] = list(self._read_pool.map(self._read_entry, entries))
        return result

    @classmethod
    def _read_entry(cls, item: tuple[str, int, SSDIndexEntry]) -> Any:
        field, global_index, entry = item
        raw = entry.path.read_bytes()
        if len(raw) != entry.length:
            raise OSError(
                f"SSDFieldStore short read for field '{field}', key {global_index}: "
                f"expected {entry.length} bytes, got {len(raw)}"
            )
        return cls._decode_sample(raw, entry)

    @staticmethod
    def _decode_sample(raw: bytes, entry: SSDIndexEntry) -> Any:
        if entry.codec == "tensor":
            if entry.dtype is None or entry.shape is None:
                raise ValueError("Tensor SSD entry is missing dtype or shape")
            dtype = getattr(torch, entry.dtype)
            if not raw:
                return torch.empty(entry.shape, dtype=dtype)
            return torch.frombuffer(raw, dtype=dtype).view(entry.shape)
        if entry.codec == "numpy":
            if entry.dtype is None or entry.shape is None:
                raise ValueError("NumPy SSD entry is missing dtype or shape")
            if not raw:
                return np.empty(entry.shape, dtype=np.dtype(entry.dtype))
            return np.frombuffer(raw, dtype=np.dtype(entry.dtype)).reshape(entry.shape)
        if entry.codec == "bytes":
            return raw
        if entry.codec == "pickle":
            return pickle.loads(raw)
        raise ValueError(f"Unsupported SSD codec: {entry.codec}")

    def clear(self, keys: list) -> None:
        """Remove logical entries and delete their sample files."""
        for field, idx_map in list(self._offset_index.items()):
            for key in keys:
                extent = idx_map.pop(key, None)
                if extent is not None:
                    self._unlink_entry(extent)
            if not idx_map:
                self._offset_index.pop(field, None)
        self._active_keys -= set(keys)

    def get_state(self) -> dict[str, dict]:
        """Load all SSD data into memory; used by checkpoint serialisation."""
        state: dict[str, dict] = {}
        for field in self._offset_index:
            indexes = list(self._offset_index[field])
            values = self.get_data([field], indexes)[field]
            state[field] = dict(zip(indexes, values, strict=True))
        return state

    def close(self) -> None:
        """Stop I/O workers and delete the storage directory."""
        if self._closed:
            return
        self._closed = True
        self._write_pool.shutdown(wait=True)
        self._read_pool.shutdown(wait=True)
        cleanup_lock_path = self._ssd_root / ".cleanup.lock"
        with open(cleanup_lock_path, "a+b") as cleanup_lock:
            fcntl.flock(cleanup_lock.fileno(), fcntl.LOCK_EX)
            try:
                shutil.rmtree(self._base_path, ignore_errors=True)
            finally:
                if self._owner_lock is not None:
                    self._owner_lock.close()
                    self._owner_lock = None
                try:
                    self._base_path.parent.rmdir()
                except OSError:
                    pass


class HybridStorageUnitData:
    """Route fields to memory or SSD while preserving StorageUnitData semantics."""

    def __init__(
        self,
        storage_size: int | None,
        ssd_path: str,
        run_id: str,
        unit_id: str,
        threshold_bytes: int = DEFAULT_SSD_OFFLOAD_THRESHOLD_BYTES,
    ) -> None:
        self._mem_store = StorageUnitData(storage_size=None)
        self._ssd_store = SSDFieldStore(ssd_path, run_id, unit_id)
        self._threshold = threshold_bytes
        self._ssd_path = ssd_path
        self._run_id = run_id
        self._unit_id = unit_id
        self._storage_size = storage_size
        self._locations: dict[str, dict[int, str]] = {}
        self._active_keys: set[int] = set()

    @property
    def active_key_count(self) -> int:
        """Return the number of active global indexes across both tiers."""
        return len(self._active_keys)

    @staticmethod
    def _split_batched_buffer(
        decoded_buffer: DecodedBuffer,
        sample_count: int,
    ) -> list[SSDEncodedSample] | None:
        if (
            decoded_buffer.buffer is None
            or decoded_buffer.dtype is None
            or decoded_buffer.shape is None
            or not decoded_buffer.shape
            or decoded_buffer.shape[0] != sample_count
        ):
            return None

        payload = memoryview(decoded_buffer.buffer).cast("B")
        if payload.nbytes % sample_count:
            return None
        sample_bytes = payload.nbytes // sample_count
        codec = "tensor" if decoded_buffer.encoding == "tensor" else "numpy"
        return [
            SSDEncodedSample(
                payload=payload[position * sample_bytes : (position + 1) * sample_bytes],
                codec=codec,
                dtype=decoded_buffer.dtype,
                shape=decoded_buffer.shape[1:],
            )
            for position in range(sample_count)
        ]

    @staticmethod
    def _sample_from_decoded_buffer(
        decoded_buffer: DecodedBuffer,
    ) -> SSDEncodedSample | None:
        if (
            decoded_buffer.encoding not in {"tensor", "numpy"}
            or decoded_buffer.buffer is None
            or decoded_buffer.dtype is None
            or decoded_buffer.shape is None
        ):
            return None
        return SSDEncodedSample(
            payload=memoryview(decoded_buffer.buffer).cast("B"),
            codec=decoded_buffer.encoding,
            dtype=decoded_buffer.dtype,
            shape=decoded_buffer.shape,
        )

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
                dtype=str(array.dtype),
                shape=tuple(array.shape),
            )
        if isinstance(value, bytes):
            return SSDEncodedSample(payload=memoryview(value), codec="bytes")
        try:
            pickled_payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return None
        return SSDEncodedSample(payload=memoryview(pickled_payload), codec="pickle")

    def _build_encoded_samples(
        self,
        values: Any,
        sample_count: int,
        storage_info: DecodeStorageInfo | None,
    ) -> list[SSDEncodedSample] | None:
        decoded_buffer = storage_info.get_buffer(values) if storage_info is not None else None
        if decoded_buffer is not None:
            if decoded_buffer.encoding in {"tensor", "numpy"}:
                samples = self._split_batched_buffer(decoded_buffer, sample_count)
                if samples is not None:
                    return samples
            elif decoded_buffer.encoding == "nested_tensor" and len(decoded_buffer.children) == sample_count:
                nested_samples: list[SSDEncodedSample] = []
                for child in decoded_buffer.children:
                    sample = self._sample_from_decoded_buffer(child)
                    if sample is None:
                        break
                    nested_samples.append(sample)
                else:
                    return nested_samples

        if isinstance(values, torch.Tensor):
            if values.is_nested:
                value_list = list(values.unbind())
            else:
                encoded_batch = self._sample_from_value(values)
                if encoded_batch is None:
                    return None
                batch_buffer = DecodedBuffer(
                    encoding=encoded_batch.codec,
                    buffer=encoded_batch.payload,
                    dtype=encoded_batch.dtype,
                    shape=encoded_batch.shape,
                )
                return self._split_batched_buffer(batch_buffer, sample_count)
        elif isinstance(values, np.ndarray):
            encoded_batch = self._sample_from_value(values)
            if encoded_batch is None:
                return None
            batch_buffer = DecodedBuffer(
                encoding=encoded_batch.codec,
                buffer=encoded_batch.payload,
                dtype=encoded_batch.dtype,
                shape=encoded_batch.shape,
            )
            return self._split_batched_buffer(batch_buffer, sample_count)
        else:
            try:
                value_list = list(values)
            except TypeError:
                return None

        if len(value_list) != sample_count:
            return None

        samples = []
        for value in value_list:
            decoded_value = storage_info.get_buffer(value) if storage_info is not None else None
            sample = (
                self._sample_from_decoded_buffer(decoded_value)
                if decoded_value is not None
                else self._sample_from_value(value)
            )
            if sample is None:
                return None
            samples.append(sample)
        return samples

    @staticmethod
    def _values_as_samples(values: Any, sample_count: int) -> list[Any]:
        if isinstance(values, torch.Tensor):
            samples = list(values.unbind())
        elif isinstance(values, np.ndarray):
            samples = list(values)
        else:
            samples = list(values)
        if len(samples) != sample_count:
            raise ValueError(f"Expected {sample_count} samples, got {len(samples)}")
        return samples

    def _validate_put(self, field_data: dict[str, Any], global_indexes: list[int]) -> None:
        for field, values in field_data.items():
            if len(values) != len(global_indexes):
                raise ValueError(
                    f"HybridStorageUnitData put_data: field '{field}' values length {len(values)} "
                    f"!= global_indexes length {len(global_indexes)}, length mismatch"
                )
        if self._storage_size is not None:
            resulting_keys = self._active_keys | set(global_indexes)
            if len(resulting_keys) > self._storage_size:
                raise ValueError(
                    f"Storage capacity exceeded: {len(self._active_keys)} existing + "
                    f"{len(resulting_keys - self._active_keys)} new > {self._storage_size}"
                )

    def _snapshot_memory(
        self, memory_fields: dict[str, Any], global_indexes: list[int]
    ) -> tuple[set[int], dict[str, tuple[bool, dict[int, Any]]]]:
        active_keys = set(self._mem_store._active_keys)
        field_snapshot: dict[str, tuple[bool, dict[int, Any]]] = {}
        for field in memory_fields:
            existed = field in self._mem_store.field_data
            current = self._mem_store.field_data.get(field, {})
            field_snapshot[field] = (
                existed,
                {key: current[key] for key in global_indexes if key in current},
            )
        return active_keys, field_snapshot

    def _restore_memory(
        self,
        snapshot: tuple[set[int], dict[str, tuple[bool, dict[int, Any]]]],
        global_indexes: list[int],
    ) -> None:
        active_keys, fields = snapshot
        for field, (existed, old_values) in fields.items():
            current = self._mem_store.field_data.get(field)
            if current is None:
                continue
            for key in global_indexes:
                if key in old_values:
                    current[key] = old_values[key]
                else:
                    current.pop(key, None)
            if not existed:
                self._mem_store.field_data.pop(field, None)
        self._mem_store._active_keys = active_keys

    def put_data(
        self,
        field_data: dict[str, Any],
        global_indexes: list,
        storage_info: DecodeStorageInfo | None = None,
    ) -> None:
        """Store each sample in memory or SSD according to its encoded size."""
        self._validate_put(field_data, global_indexes)
        if not global_indexes:
            return

        memory_writes: dict[str, tuple[list[int], list[Any]]] = {}
        encoded_ssd_fields: dict[str, dict[int, SSDEncodedSample]] = {}
        pending_locations: dict[str, dict[int, str]] = {}
        for field, values in field_data.items():
            logical_samples = self._values_as_samples(values, len(global_indexes))
            encoded_samples = self._build_encoded_samples(
                values,
                len(global_indexes),
                storage_info,
            )
            field_locations: dict[int, str] = {}
            memory_indexes: list[int] = []
            memory_values: list[Any] = []
            ssd_values: dict[int, SSDEncodedSample] = {}
            for position, global_index in enumerate(global_indexes):
                encoded = encoded_samples[position] if encoded_samples is not None else None
                destination = "ssd" if encoded is not None and encoded.payload.nbytes >= self._threshold else "mem"
                field_locations[global_index] = destination
                if destination == "ssd":
                    assert encoded is not None
                    ssd_values[global_index] = encoded
                else:
                    memory_indexes.append(global_index)
                    memory_values.append(logical_samples[position])

            pending_locations[field] = field_locations
            if memory_indexes:
                memory_writes[field] = (memory_indexes, memory_values)
            if ssd_values:
                encoded_ssd_fields[field] = ssd_values

        memory_snapshot = self._snapshot_memory(field_data, global_indexes)
        prepared: dict[str, dict[int, SSDIndexEntry]] = {}
        try:
            if encoded_ssd_fields:
                prepared = self._ssd_store.prepare_encoded(encoded_ssd_fields)
            for field, (indexes, values) in memory_writes.items():
                self._mem_store.put_data({field: values}, indexes)
        except Exception:
            self._ssd_store.discard_prepared(prepared)
            self._restore_memory(memory_snapshot, global_indexes)
            raise

        self._ssd_store.commit_prepared(prepared)
        for field, locations in pending_locations.items():
            mem_field = self._mem_store.field_data.get(field)
            for global_index, destination in locations.items():
                if destination == "ssd":
                    if mem_field is not None:
                        mem_field.pop(global_index, None)
                else:
                    self._ssd_store.remove(field, global_index)
            self._locations.setdefault(field, {}).update(locations)
        self._active_keys.update(global_indexes)

    def get_data(self, fields: list[str], global_indexes: list) -> dict[str, list]:
        """Read mixed memory- and SSD-backed samples in request order."""
        result: dict[str, list] = {}
        for field in fields:
            if field not in self._locations:
                raise ValueError(
                    f"HybridStorageUnitData get_data: field '{field}' not found. Available: {list(self._locations)}"
                )
            field_locations = self._locations[field]
            ssd_indexes = [
                global_index for global_index in global_indexes if field_locations.get(global_index) == "ssd"
            ]
            ssd_values = {}
            if ssd_indexes:
                decoded_values = self._ssd_store.get_data([field], ssd_indexes)[field]
                ssd_values = dict(zip(ssd_indexes, decoded_values, strict=True))
            values = []
            for global_index in global_indexes:
                location = field_locations.get(global_index)
                if location == "ssd":
                    values.append(ssd_values[global_index])
                elif location == "mem":
                    try:
                        values.append(self._mem_store.field_data[field][global_index])
                    except KeyError as e:
                        raise KeyError(
                            f"HybridStorageUnitData get_data: key {global_index} not found in field '{field}'"
                        ) from e
                else:
                    raise KeyError(f"HybridStorageUnitData get_data: key {global_index} not found in field '{field}'")
            result[field] = values
        return result

    def clear(self, keys: list) -> None:
        """Clear the requested keys from both tiers and the location index."""
        self._mem_store.clear(keys)
        self._ssd_store.clear(keys)
        for field, locations in list(self._locations.items()):
            for key in keys:
                locations.pop(key, None)
            if not locations:
                self._locations.pop(field, None)
        self._active_keys -= set(keys)

    def get_state(self) -> tuple[dict, set]:
        """Return ``(field_data, active_keys)`` for checkpoint serialisation.

        SSD-backed fields are loaded into memory temporarily so the caller can
        write a single pickle file containing all data.
        """
        field_data: dict[str, dict] = {field: dict(fd) for field, fd in self._mem_store.field_data.items()}
        for field, values in self._ssd_store.get_state().items():
            field_data.setdefault(field, {}).update(values)
        return field_data, set(self._active_keys)

    def load_state(self, field_data: dict, active_keys: set) -> None:
        """Reset both stores and restore from checkpoint data.

        Each field is re-routed through the normal ``put_data`` path so the
        threshold-based routing decision is re-applied to the restored data.
        """
        if self._storage_size is not None and len(active_keys) > self._storage_size:
            raise ValueError(
                f"Checkpoint contains {len(active_keys)} active keys, exceeding storage capacity {self._storage_size}"
            )

        replacement = HybridStorageUnitData(
            storage_size=self._storage_size,
            threshold_bytes=self._threshold,
            ssd_path=self._ssd_path,
            run_id=self._run_id,
            unit_id=f"{self._unit_id}.restore.{uuid4().hex}",
        )
        try:
            for field, field_dict in field_data.items():
                if not field_dict:
                    continue
                indexes = sorted(field_dict)
                replacement.put_data({field: [field_dict[key] for key in indexes]}, indexes)
            replacement._active_keys = set(active_keys)
        except Exception:
            replacement.close()
            raise

        old_ssd_store = self._ssd_store
        self._mem_store = replacement._mem_store
        self._ssd_store = replacement._ssd_store
        self._locations = replacement._locations
        self._active_keys = replacement._active_keys
        old_ssd_store.close()

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

    def __init__(
        self,
        storage_unit_size: int | None = None,
        ssd_config=None,
        ssd_run_id: str | None = None,
    ):
        """Initialize a SimpleStorageUnit with the specified size.

        Args:
            storage_unit_size: Maximum number of elements that can be stored in this storage unit.
                If None, the storage unit has unlimited capacity.
            ssd_config: Optional OmegaConf DictConfig for SSD offload (the ``ssd_offload`` block
                from config.yaml).  When ``ssd_config.enabled`` is True a
                ``HybridStorageUnitData`` is used in place of the default in-memory
                ``StorageUnitData``.
            ssd_run_id: Internal run identifier used to isolate SSD directories.
        """
        self.storage_unit_id = f"TQ_STORAGE_UNIT_{uuid4().hex[:8]}"
        self.storage_unit_size = storage_unit_size
        self.storage_data: StorageUnitData | HybridStorageUnitData

        if ssd_config is not None and ssd_config.get("enabled", False):
            ssd_path = ssd_config.get("path")
            if not ssd_path:
                raise ValueError("SimpleStorage SSD offload requires backend.SimpleStorage.ssd_offload.path")
            self.storage_data = HybridStorageUnitData(
                storage_size=self.storage_unit_size,
                ssd_path=str(ssd_path),
                run_id=ssd_run_id or uuid4().hex,
                unit_id=self.storage_unit_id,
            )
            logger.info(
                f"[{self.storage_unit_id}]: SSD offload enabled — "
                f"path={ssd_path}, "
                f"threshold={DEFAULT_SSD_OFFLOAD_THRESHOLD_BYTES} B/sample"
            )
        else:
            self.storage_data = StorageUnitData(self.storage_unit_size)

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

        while True:
            try:
                self._put_get_socket_port = get_free_port(ip=self._node_ip)
                self.put_get_socket.bind(format_zmq_address(self._node_ip, self._put_get_socket_port))
                break
            except zmq.ZMQError:
                logger.warning(f"[{self.storage_unit_id}]: Try to bind ZMQ sockets failed, retrying...")
                continue

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
        try:
            zmq.proxy(self.put_get_socket, self.worker_socket)
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
                socks = dict(poller.poll(TQ_STORAGE_POLLER_TIMEOUT * 1000))
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
                # Messages received from proxy: [identity, serialized_msg_frame1, ...]
                messages = worker_socket.recv_multipart(copy=False)
                identity = messages[0]
                serialized_msg = messages[1:]

                request_msg, storage_info = ZMQMessage.deserialize_with_storage_info(serialized_msg)
                operation = request_msg.request_type

                try:
                    logger.debug(f"[{self.storage_unit_id}]: worker received operation: {operation}")

                    # Process request
                    if operation == ZMQRequestType.PUT_DATA:  # type: ignore[arg-type]
                        with monitor.measure(op_type="PUT_DATA"):
                            response_msg = self._handle_put(request_msg, storage_info)
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
                                "message": f"Storage unit id #{self.storage_unit_id} "
                                f"receive invalid operation: {operation}."
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

                # Send response back with identity for routing
                worker_socket.send_multipart([identity] + response_msg.serialize(), copy=False)

        logger.info(f"[{self.storage_unit_id}]: worker stopped.")
        poller.unregister(worker_socket)
        worker_socket.close(linger=0)

    def _handle_put(
        self,
        data_parts: ZMQMessage,
        storage_info: DecodeStorageInfo | None = None,
    ) -> ZMQMessage:
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
                if isinstance(self.storage_data, HybridStorageUnitData):
                    # A parser may replace or mutate decoded values, invalidating wire-buffer layout.
                    effective_storage_info = None if data_parser is not None else storage_info
                    self.storage_data.put_data(
                        field_data,
                        global_indexes,
                        storage_info=effective_storage_info,
                    )
                else:
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
        """Serialize storage unit data directly to a file.

        Args:
            data_parts: ZMQMessage from client, containing ``path`` in body:
                absolute path for the output .pkl file. The caller must ensure
                this path is reachable from the node running this actor
                (shared filesystem required for multi-node setups).

        Returns:
            ZMQMessage with ``success=True`` on success, or ``success=False``
            and ``message`` containing the error string on failure.
        """
        path = data_parts.body["path"]
        try:
            if isinstance(self.storage_data, HybridStorageUnitData):
                field_data, active_keys = self.storage_data.get_state()
            else:
                field_data = self.storage_data.field_data
                active_keys = self.storage_data._active_keys
            state = {
                "storage_unit_id": self.storage_unit_id,
                "storage_unit_size": self.storage_unit_size,
                "field_data": field_data,
                "active_keys": active_keys,
            }
            with open(path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
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
        """Restore storage unit data directly from a file.

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
            with open(path, "rb") as f:
                data = pickle.load(f)

            if data["storage_unit_size"] != self.storage_unit_size:
                logger.warning(
                    f"[{self.storage_unit_id}]: storage_unit_size mismatch — "
                    f"checkpoint={data['storage_unit_size']}, current={self.storage_unit_size}"
                )

            if isinstance(self.storage_data, HybridStorageUnitData):
                self.storage_data.load_state(data["field_data"], data["active_keys"])
            else:
                if self.storage_data._active_keys:
                    logger.warning(
                        f"[{self.storage_unit_id}]: overwriting {len(self.storage_data._active_keys)} "
                        f"existing keys with checkpoint data from {path}"
                    )
                self.storage_data.field_data.clear()
                self.storage_data._active_keys.clear()
                self.storage_data.field_data = data["field_data"]
                self.storage_data._active_keys = data["active_keys"]

            logger.info(
                f"[{self.storage_unit_id}]: loaded checkpoint from {path} — "
                f"{len(data['active_keys'])} keys, {len(data['field_data'])} fields"
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
        worker_socket: zmq.Socket | None,
        storage_data=None,
    ) -> None:
        """Clean up resources on garbage collection."""
        logger.info("Shutting down SimpleStorageUnit resources...")

        shutdown_event.set()
        try:
            if put_get_socket:
                put_get_socket.close(linger=0)
            if worker_socket:
                worker_socket.close(linger=0)
            if zmq_context:
                zmq_context.term()
        finally:
            if worker_thread and worker_thread.is_alive():
                worker_thread.join(timeout=5)
            if proxy_thread and proxy_thread.is_alive():
                proxy_thread.join(timeout=5)
            if storage_data is not None and hasattr(storage_data, "close"):
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
