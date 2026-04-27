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

import ctypes
import logging
import os
import pickle
import struct
import time
from typing import Any, Optional

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

MOONCAKE_STORE_IMPORTED: bool = True
try:
    from mooncake.store import MooncakeDistributedStore
except ImportError:
    MOONCAKE_STORE_IMPORTED = False

BATCH_SIZE_LIMIT: int = 500

# ---------------------------------------------------------------------------
# GDR (GPU Direct RDMA) helpers
# TensorMetadata layout must match the C++ struct in
# mooncake-integration/integration_utils.h:
#   struct TensorMetadata { int32_t dtype; int32_t ndim; int64_t shape[4]; };
# ---------------------------------------------------------------------------
TENSOR_METADATA_SIZE: int = 40  # 4 + 4 + 8*4 = 40 bytes
_TENSOR_METADATA_FMT: str = "<ii4q"  # little-endian: int32, int32, 4×int64

_TORCH_DTYPE_TO_MC_ENUM: dict[torch.dtype, int] = {
    torch.float32: 0,
    torch.float64: 1,
    torch.int8: 2,
    torch.uint8: 3,
    torch.int16: 4,
    torch.uint16: 5,
    torch.int32: 6,
    torch.uint32: 7,
    torch.int64: 8,
    torch.uint64: 9,
    torch.bool: 10,
    torch.float16: 11,
    torch.bfloat16: 12,
}
if hasattr(torch, "float8_e4m3fn"):
    _TORCH_DTYPE_TO_MC_ENUM[torch.float8_e4m3fn] = 13
if hasattr(torch, "float8_e5m2"):
    _TORCH_DTYPE_TO_MC_ENUM[torch.float8_e5m2] = 14

_MC_ENUM_TO_TORCH_DTYPE: dict[int, torch.dtype] = {v: k for k, v in _TORCH_DTYPE_TO_MC_ENUM.items()}


def _pack_tensor_metadata(dtype: torch.dtype, shape: tuple[int, ...]) -> bytes:
    """Pack a TensorMetadata struct matching the C++ layout."""
    dtype_enum = _TORCH_DTYPE_TO_MC_ENUM.get(dtype)
    if dtype_enum is None:
        raise ValueError(f"Unsupported dtype for GDR: {dtype}")
    ndim = len(shape)
    if ndim > 4:
        raise ValueError(f"TensorMetadata supports at most 4 dimensions, got {ndim}")
    shape_padded = list(shape) + [-1] * (4 - ndim)
    return struct.pack(_TENSOR_METADATA_FMT, dtype_enum, ndim, *shape_padded)


def _unpack_tensor_metadata(data: bytes) -> tuple[torch.dtype, tuple[int, ...]]:
    """Unpack a TensorMetadata struct."""
    dtype_enum, ndim, *shape_raw = struct.unpack(_TENSOR_METADATA_FMT, data)
    dtype = _MC_ENUM_TO_TORCH_DTYPE.get(dtype_enum)
    if dtype is None:
        raise ValueError(f"Unknown dtype enum: {dtype_enum}")
    shape = tuple(shape_raw[:ndim])
    return dtype, shape


# Pre-computed element sizes for all supported dtypes (avoids per-call tensor alloc)
_DTYPE_ELEMENT_SIZE: dict[torch.dtype, int] = {}
for _dt in _TORCH_DTYPE_TO_MC_ENUM:
    _DTYPE_ELEMENT_SIZE[_dt] = torch.tensor([], dtype=_dt).element_size()

_libcudart = None


def _cuda_d2d_copy(dst_ptr: int, src_ptr: int, num_bytes: int):
    """Synchronous device-to-device memcpy via cudart."""
    global _libcudart
    if _libcudart is None:
        _libcudart = ctypes.CDLL("libcudart.so")
    ret = _libcudart.cudaMemcpy(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        ctypes.c_size_t(num_bytes),
        ctypes.c_int(3),  # cudaMemcpyDeviceToDevice
    )
    if ret != 0:
        raise RuntimeError(f"cudaMemcpy D2D failed with error code {ret}")


@StorageClientFactory.register("MooncakeStoreClient")
class MooncakeStoreClient(TransferQueueStorageKVClient):
    """
    Storage client for MooncakeStore.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if not MOONCAKE_STORE_IMPORTED:
            raise ImportError("Mooncake Store not installed. Please install via: pip install mooncake-transfer-engine")

        # Required: Address of local host
        self.local_hostname = config.get("local_hostname", "")
        # Required: Address of the HTTP metadata server (e.g., "localhost:8080")
        self.metadata_server = config.get("metadata_server", None)
        # Required: Address of the master server RPC endpoint (e.g., "localhost:8081")
        self.master_server_address = config.get("master_server_address")

        self.global_segment_size = int(config.get("global_segment_size", 4096 * 1024 * 1024))
        self.local_buffer_size = int(config.get("local_buffer_size", 1024 * 1024 * 1024))
        self.protocol = config.get("protocol", "tcp")
        self.device_name = config.get("device_name", "")
        if self.device_name is None:
            self.device_name = ""

        if self.local_hostname is None or self.local_hostname == "":
            from transfer_queue.utils.zmq_utils import get_node_ip_address_raw

            ip = get_node_ip_address_raw()
            logger.info(f"Try to use Ray IP ({ip}) as local hostname for MooncakeStore.")
            self.local_hostname = ip

        if self.metadata_server is None or not isinstance(self.metadata_server, str):
            raise ValueError("Missing or invalid 'metadata_server' in config")
        if self.master_server_address is None or not isinstance(self.master_server_address, str):
            raise ValueError("Missing or invalid 'master_server_address' in config")

        if not self.metadata_server.startswith("http://") and not self.metadata_server.startswith("etcd://"):
            self.metadata_server = f"http://{self.metadata_server}"
        if not self.metadata_server.startswith("etcd://") and not self.metadata_server.endswith("/metadata"):
            self.metadata_server = self.metadata_server + "/metadata"

        if self.metadata_server is None:
            raise ValueError("Missing 'metadata_server' in config")
        if self.master_server_address is None:
            raise ValueError("Missing 'master_server_address' in config")

        self._store = MooncakeDistributedStore()
        ret = self._store.setup(
            self.local_hostname,
            self.metadata_server,
            self.global_segment_size,
            self.local_buffer_size,
            self.protocol,
            self.device_name,
            self.master_server_address,
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake store setup failed with error code: {ret}")

        # GDR (GPU Direct RDMA) configuration
        self._use_gdr = config.get("use_gdr", False)
        self._gdr_target_device = config.get("gdr_target_device", None)
        self._gdr_staging_buf = None
        self._gdr_staging_ptr = 0
        self._gdr_staging_size = 0
        self._gdr_meta_ctypes = None
        self._gdr_meta_ptr = 0
        self._gdr_meta_size = 0

        if self._use_gdr:
            gdr_buf_bytes = int(config.get("gdr_buffer_size", 1 * 1024 * 1024 * 1024))
            device = self._gdr_target_device or f"cuda:{torch.cuda.current_device()}"
            # GPU staging buffer — registered once, reused for every GDR transfer
            self._gdr_staging_buf = torch.empty(gdr_buf_bytes, dtype=torch.uint8, device=device)
            self._gdr_staging_ptr = self._gdr_staging_buf.data_ptr()
            self._gdr_staging_size = gdr_buf_bytes
            self._store.register_buffer(self._gdr_staging_ptr, self._gdr_staging_size)
            # CPU metadata buffer — enough for BATCH_SIZE_LIMIT tensors
            meta_capacity = TENSOR_METADATA_SIZE * BATCH_SIZE_LIMIT
            self._gdr_meta_buf = bytearray(meta_capacity)
            self._gdr_meta_ctypes = (ctypes.c_char * meta_capacity).from_buffer(self._gdr_meta_buf)
            self._gdr_meta_ptr = ctypes.addressof(self._gdr_meta_ctypes)
            self._gdr_meta_size = meta_capacity
            self._store.register_buffer(self._gdr_meta_ptr, self._gdr_meta_size)
            logger.info(
                f"GDR staging buffer: {gdr_buf_bytes / (1024**2):.0f}MB GPU + "
                f"{meta_capacity / 1024:.0f}KB CPU metadata on {device}"
            )

    def put(self, keys: list[str], values: list[Any]) -> Optional[list[Any]]:
        """Stores multiple key-value pairs to MooncakeStore.

        Args:
            keys (List[str]): List of unique string identifiers.
            values (List[Any]): List of values to store (tensors, scalars, dicts, etc.).
        """

        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        _t_classify0 = time.perf_counter()
        tensor_keys = []
        tensor_values = []
        gdr_tensor_keys = []
        gdr_tensor_values = []
        non_tensor_keys = []
        non_tensor_values = []

        for key, value in zip(keys, values, strict=True):
            if isinstance(value, torch.Tensor):
                tensor = value.contiguous()
                if self._use_gdr and tensor.device.type == "cuda" and tensor.numel() > 0:
                    gdr_tensor_keys.append(key)
                    gdr_tensor_values.append(tensor)
                else:
                    if tensor.device.type == "cuda":
                        tensor = tensor.cpu()
                    tensor_keys.append(key)
                    tensor_values.append(tensor)
            else:
                non_tensor_keys.append(key)
                non_tensor_values.append(pickle.dumps(value))
        _t_classify1 = time.perf_counter()

        if gdr_tensor_keys:
            self._batch_put_tensors_gdr(gdr_tensor_keys, gdr_tensor_values)

        if tensor_keys:
            self._batch_put_tensors(tensor_keys, tensor_values)

        if non_tensor_keys:
            self._batch_put_bytes(non_tensor_keys, non_tensor_values)

        _t_done = time.perf_counter()
        print(
            f"[TQ-TIMING] mooncake::put n={len(keys)}: "
            f"classify={_t_classify1-_t_classify0:.4f}s  total={_t_done-_t_classify0:.4f}s",
            flush=True,
        )
        return None

    def _batch_put_tensors(self, keys: list[str], tensors: list[Tensor]):
        _t_loop_total0 = time.perf_counter()
        _overhead_s = 0.0
        _rdma_s = 0.0
        _n_batches = 0
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            _t_oh0 = time.perf_counter()
            batch_keys = keys[i : i + BATCH_SIZE_LIMIT]
            batch_tensors = tensors[i : i + BATCH_SIZE_LIMIT]
            data_gb = sum(t.numel() * t.element_size() for t in batch_tensors) / (1024**3)
            _t_oh1 = time.perf_counter()

            _t0 = time.perf_counter()
            results = self._store.batch_put_tensor(batch_keys, batch_tensors)
            _t1 = time.perf_counter()
            print(
                f"[TQ-TIMING] mooncake::batch_put_tensor n={len(batch_keys)} "
                f"data={data_gb:.4f}GB  rdma={_t1-_t0:.4f}s  bw={data_gb*8/(_t1-_t0):.2f}Gb/s",
                flush=True,
            )

            _t_val0 = time.perf_counter()
            if not all(r == 0 for r in results):
                failed_indices = [j for j, r in enumerate(results) if r != 0]
                error_codes = [results[j] for j in failed_indices]
                raise RuntimeError(
                    f"batch_put_tensor failed for indices {failed_indices} with error codes: {error_codes}"
                )
            _t_val1 = time.perf_counter()

            _overhead_s += (_t_oh1 - _t_oh0) + (_t_val1 - _t_val0)
            _rdma_s += (_t1 - _t0)
            _n_batches += 1
        _t_loop_total1 = time.perf_counter()
        print(
            f"[TQ-TIMING] mooncake::put_loop n={len(keys)} batches={_n_batches}: "
            f"rdma_total={_rdma_s:.4f}s  py_overhead={_overhead_s:.4f}s  "
            f"loop_total={_t_loop_total1-_t_loop_total0:.4f}s",
            flush=True,
        )

    def _batch_put_tensors_gdr(self, keys: list[str], tensors: list[Tensor]):
        """Put GPU tensors via GDR using a pre-registered staging buffer.

        Tensors are D2D-copied into a contiguous GPU staging buffer that was
        registered with RDMA once at init.  This eliminates the per-tensor
        ``register_buffer`` / ``unregister_buffer`` overhead (~6 ms each for
        GPU memory through nvidia_peermem).
        """
        _t_loop_total0 = time.perf_counter()
        _rdma_s = 0.0
        _d2d_s = 0.0
        _n_batches = 0
        staging_ptr = self._gdr_staging_ptr
        staging_cap = self._gdr_staging_size
        meta_ptr = self._gdr_meta_ptr

        idx = 0
        while idx < len(keys):
            # Build sub-batch bounded by BATCH_SIZE_LIMIT *and* staging capacity
            batch_keys: list[str] = []
            batch_tensors: list[Tensor] = []
            used = 0
            while idx < len(keys) and len(batch_keys) < BATCH_SIZE_LIMIT:
                sz = tensors[idx].numel() * tensors[idx].element_size()
                if used + sz > staging_cap:
                    if not batch_keys:
                        raise RuntimeError(
                            f"Single tensor ({sz} bytes) exceeds GDR staging buffer "
                            f"({staging_cap} bytes). Increase gdr_buffer_size in config."
                        )
                    break
                batch_keys.append(keys[idx])
                batch_tensors.append(tensors[idx])
                used += sz
                idx += 1

            n = len(batch_keys)
            data_gb = used / (1024**3)

            # D2D copy each tensor into the staging buffer, fill metadata
            _t_d2d0 = time.perf_counter()
            all_buffer_ptrs: list[list[int]] = []
            all_sizes: list[list[int]] = []
            offset = 0
            for j, tensor in enumerate(batch_tensors):
                meta_offset = j * TENSOR_METADATA_SIZE
                self._gdr_meta_buf[meta_offset : meta_offset + TENSOR_METADATA_SIZE] = (
                    _pack_tensor_metadata(tensor.dtype, tuple(tensor.shape))
                )
                data_size = tensor.numel() * tensor.element_size()
                _cuda_d2d_copy(staging_ptr + offset, tensor.data_ptr(), data_size)

                all_buffer_ptrs.append([meta_ptr + meta_offset, staging_ptr + offset])
                all_sizes.append([TENSOR_METADATA_SIZE, data_size])
                offset += data_size
            _t_d2d1 = time.perf_counter()

            # RDMA transfer: metadata from CPU, tensor data from GPU staging buf
            _t0 = time.perf_counter()
            results = self._store.batch_put_from_multi_buffers(batch_keys, all_buffer_ptrs, all_sizes)
            _t1 = time.perf_counter()

            if not all(r == 0 for r in results):
                failed_indices = [j for j, r in enumerate(results) if r != 0]
                error_codes = [results[j] for j in failed_indices]
                raise RuntimeError(
                    f"GDR batch_put failed for indices {failed_indices} with error codes: {error_codes}"
                )

            print(
                f"[TQ-TIMING] mooncake::gdr_batch_put n={n} "
                f"data={data_gb:.4f}GB  d2d={_t_d2d1-_t_d2d0:.4f}s  "
                f"rdma={_t1-_t0:.4f}s  bw={data_gb*8/(_t1-_t0):.2f}Gb/s",
                flush=True,
            )
            _rdma_s += _t1 - _t0
            _d2d_s += _t_d2d1 - _t_d2d0
            _n_batches += 1

        _t_loop_total1 = time.perf_counter()
        print(
            f"[TQ-TIMING] mooncake::gdr_put_loop n={len(keys)} batches={_n_batches}: "
            f"rdma_total={_rdma_s:.4f}s  d2d_total={_d2d_s:.4f}s  "
            f"loop_total={_t_loop_total1-_t_loop_total0:.4f}s",
            flush=True,
        )

    def _batch_put_bytes(self, keys: list[str], values: list[bytes]):
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i : i + BATCH_SIZE_LIMIT]
            batch_values = values[i : i + BATCH_SIZE_LIMIT]

            ret = self._store.put_batch(batch_keys, batch_values)
            if ret != 0:
                raise RuntimeError(f"put_batch failed with error code: {ret}")

    def get(self, keys: list[str], shapes=None, dtypes=None, custom_backend_meta=None) -> list[Any]:
        """Get multiple key-value pairs from MooncakeStore.

        Args:
            keys (List[str]): Keys to fetch.
            shapes (List[List[int]]): Expected tensor shapes (use [] for scalars).
            dtypes (List[Optional[torch.dtype]]): Expected dtypes; use None for non-tensor data.
            custom_backend_meta (List[str], optional): ...

        Returns:
            List[Any]: Retrieved values in the same order as input keys.
        """

        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStoreClient needs shapes and dtypes")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        _t_classify0 = time.perf_counter()
        tensor_indices = []
        non_tensor_indices = []

        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)

        results = [None] * len(keys)
        _t_classify1 = time.perf_counter()

        if tensor_indices:
            _t_scatter0 = time.perf_counter()
            tensor_keys = [keys[i] for i in tensor_indices]
            tensor_shapes = [shapes[i] for i in tensor_indices]
            tensor_dtypes = [dtypes[i] for i in tensor_indices]
            _t_scatter1 = time.perf_counter()
            if self._use_gdr:
                tensor_results = self._batch_get_tensors_gdr(tensor_keys, tensor_shapes, tensor_dtypes)
            else:
                tensor_results = self._batch_get_tensors(tensor_keys, tensor_shapes, tensor_dtypes)
            _t_gather0 = time.perf_counter()
            # TODO: optimize these for loops
            for idx, tensor in zip(tensor_indices, tensor_results, strict=True):
                results[idx] = tensor
            _t_gather1 = time.perf_counter()

        if non_tensor_indices:
            non_tensor_keys = [keys[i] for i in non_tensor_indices]
            non_tensor_results = self._batch_get_bytes(non_tensor_keys)
            for idx, data in zip(non_tensor_indices, non_tensor_results, strict=True):
                results[idx] = pickle.loads(data)

        _t_done = time.perf_counter()
        _scatter = (_t_scatter1 - _t_scatter0) if tensor_indices else 0
        _gather = (_t_gather1 - _t_gather0) if tensor_indices else 0
        print(
            f"[TQ-TIMING] mooncake::get n={len(keys)}: "
            f"classify={_t_classify1-_t_classify0:.4f}s  scatter={_scatter:.4f}s  "
            f"gather={_gather:.4f}s  total={_t_done-_t_classify0:.4f}s",
            flush=True,
        )
        return results

    def _batch_get_tensors(self, keys: list[str], shapes: list, dtypes: list) -> list[Tensor]:
        tensors = [None] * len(keys)
        _t_loop_total0 = time.perf_counter()
        _overhead_s = 0.0
        _rdma_s = 0.0
        _n_batches = 0

        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            _t_oh0 = time.perf_counter()
            batch_keys = keys[i : i + BATCH_SIZE_LIMIT]
            batch_shapes = shapes[i : i + BATCH_SIZE_LIMIT]
            batch_dtypes = dtypes[i : i + BATCH_SIZE_LIMIT]
            dtype_sizes = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2, torch.int64: 8, torch.int32: 4}
            data_gb = sum(
                (torch.Size(s).numel() * dtype_sizes.get(d, 4))
                for s, d in zip(batch_shapes, batch_dtypes)
                if d is not None
            ) / (1024**3)
            _t_oh1 = time.perf_counter()

            _t0 = time.perf_counter()
            batch_results = self._store.batch_get_tensor(batch_keys)
            _t1 = time.perf_counter()
            print(
                f"[TQ-TIMING] mooncake::batch_get_tensor n={len(batch_keys)} "
                f"data={data_gb:.4f}GB  rdma={_t1-_t0:.4f}s  bw={data_gb*8/(_t1-_t0):.2f}Gb/s",
                flush=True,
            )

            _t_val0 = time.perf_counter()
            if len(batch_results) != len(batch_keys):
                raise RuntimeError(f"batch_get_tensor returned {len(batch_results)} items, expected {len(batch_keys)}")

            for j, (tensor, shape, dtype) in enumerate(zip(batch_results, batch_shapes, batch_dtypes, strict=True)):
                if tensor is None:
                    raise RuntimeError(f"batch_get_tensor returned None for key '{batch_keys[j]}'")
                if tensor.shape != torch.Size(shape):
                    raise RuntimeError(
                        f"Shape mismatch for key '{batch_keys[j]}': expected {shape}, got {tensor.shape}"
                    )
                if tensor.dtype != dtype:
                    raise RuntimeError(
                        f"Dtype mismatch for key '{batch_keys[j]}': expected {dtype}, got {tensor.dtype}"
                    )
                tensors[i + j] = tensor
            _t_val1 = time.perf_counter()

            _overhead_s += (_t_oh1 - _t_oh0) + (_t_val1 - _t_val0)
            _rdma_s += (_t1 - _t0)
            _n_batches += 1

        _t_loop_total1 = time.perf_counter()
        print(
            f"[TQ-TIMING] mooncake::get_loop n={len(keys)} batches={_n_batches}: "
            f"rdma_total={_rdma_s:.4f}s  py_overhead={_overhead_s:.4f}s  "
            f"loop_total={_t_loop_total1-_t_loop_total0:.4f}s",
            flush=True,
        )
        return tensors

    def _batch_get_tensors_gdr(self, keys: list[str], shapes: list, dtypes: list) -> list[Tensor]:
        """Get tensors into GPU memory via GDR using a pre-registered staging buffer.

        RDMA reads land in the staging buffer, then D2D-copied to individual
        output tensors.  Same registration-avoidance strategy as the PUT path.
        """
        device = self._gdr_target_device or f"cuda:{torch.cuda.current_device()}"
        tensors: list[Optional[Tensor]] = [None] * len(keys)
        _t_loop_total0 = time.perf_counter()
        _rdma_s = 0.0
        _d2d_s = 0.0
        _n_batches = 0
        staging_ptr = self._gdr_staging_ptr
        staging_cap = self._gdr_staging_size
        meta_ptr = self._gdr_meta_ptr

        idx = 0
        while idx < len(keys):
            batch_start = idx
            batch_keys: list[str] = []
            batch_shapes: list = []
            batch_dtypes: list = []
            used = 0
            while idx < len(keys) and len(batch_keys) < BATCH_SIZE_LIMIT:
                s, d = shapes[idx], dtypes[idx]
                sz = torch.Size(s).numel() * _DTYPE_ELEMENT_SIZE[d]
                if used + sz > staging_cap:
                    if not batch_keys:
                        raise RuntimeError(
                            f"Single tensor ({sz} bytes) exceeds GDR staging buffer "
                            f"({staging_cap} bytes). Increase gdr_buffer_size in config."
                        )
                    break
                batch_keys.append(keys[idx])
                batch_shapes.append(s)
                batch_dtypes.append(d)
                used += sz
                idx += 1

            n = len(batch_keys)
            data_gb = used / (1024**3)

            # Map each tensor's region in the staging buffer
            all_buffer_ptrs: list[list[int]] = []
            all_sizes: list[list[int]] = []
            tensor_regions: list[tuple[int, int]] = []  # (offset, data_size)
            offset = 0
            for j, (s, d) in enumerate(zip(batch_shapes, batch_dtypes)):
                meta_offset = j * TENSOR_METADATA_SIZE
                data_size = torch.Size(s).numel() * _DTYPE_ELEMENT_SIZE[d]
                all_buffer_ptrs.append([meta_ptr + meta_offset, staging_ptr + offset])
                all_sizes.append([TENSOR_METADATA_SIZE, data_size])
                tensor_regions.append((offset, data_size))
                offset += data_size

            # RDMA transfer: metadata → CPU, tensor data → GPU staging buffer
            _t0 = time.perf_counter()
            results = self._store.batch_get_into_multi_buffers(batch_keys, all_buffer_ptrs, all_sizes)
            _t1 = time.perf_counter()

            if any(r < 0 for r in results):
                failed_indices = [j for j, r in enumerate(results) if r < 0]
                error_codes = [results[j] for j in failed_indices]
                raise RuntimeError(
                    f"GDR batch_get failed for indices {failed_indices} with error codes: {error_codes}"
                )

            # D2D copy from staging buffer to individual output tensors
            _t_d2d0 = time.perf_counter()
            for j, (s, d) in enumerate(zip(batch_shapes, batch_dtypes)):
                t = torch.empty(s, dtype=d, device=device)
                buf_offset, data_size = tensor_regions[j]
                _cuda_d2d_copy(t.data_ptr(), staging_ptr + buf_offset, data_size)
                tensors[batch_start + j] = t
            _t_d2d1 = time.perf_counter()

            print(
                f"[TQ-TIMING] mooncake::gdr_batch_get n={n} "
                f"data={data_gb:.4f}GB  rdma={_t1-_t0:.4f}s  "
                f"d2d={_t_d2d1-_t_d2d0:.4f}s  bw={data_gb*8/(_t1-_t0):.2f}Gb/s",
                flush=True,
            )
            _rdma_s += _t1 - _t0
            _d2d_s += _t_d2d1 - _t_d2d0
            _n_batches += 1

        _t_loop_total1 = time.perf_counter()
        print(
            f"[TQ-TIMING] mooncake::gdr_get_loop n={len(keys)} batches={_n_batches}: "
            f"rdma_total={_rdma_s:.4f}s  d2d_total={_d2d_s:.4f}s  "
            f"loop_total={_t_loop_total1-_t_loop_total0:.4f}s",
            flush=True,
        )
        return tensors

    def _batch_get_bytes(self, keys: list[str]) -> list[bytes]:
        results = []
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i : i + BATCH_SIZE_LIMIT]
            batch_results = self._store.get_batch(batch_keys)
            if len(batch_results) != len(batch_keys):
                raise RuntimeError(f"get_batch returned {len(batch_results)} items, expected {len(batch_keys)}")
            results.extend(batch_results)
        return results

    def clear(self, keys: list[str], custom_backend_meta=None):
        """Deletes multiple keys from MooncakeStore.


        Args:
            keys (List[str]): List of keys to remove.
            custom_backend_meta (List[Any], optional): ...
        """
        global_indexes_patterns = {key.split("@")[0] + "@.*" for key in keys}
        for p in global_indexes_patterns:
            ret = self._store.remove_by_regex(p, force=True)
            if ret < 0:
                logger.warning(f"remove failed for key '{p}' with error code: {ret}")

        # FIXME: controller returned BatchMeta may have mismatched fields in some case, preventing
        #        key-value based backends to accurately clear all existing keys..
        # for key in keys:
        #     ret = self._store.remove(key)
        #     if not (ret == 0 or ret == -704):
        #         logger.warning(f"remove failed for key '{key}' with error code: {ret}")

    def close(self):
        """Closes MooncakeStore."""
        if self._store:
            if self._use_gdr:
                if self._gdr_staging_ptr:
                    self._store.unregister_buffer(self._gdr_staging_ptr)
                    self._gdr_staging_ptr = 0
                if self._gdr_meta_ptr:
                    self._store.unregister_buffer(self._gdr_meta_ptr)
                    self._gdr_meta_ptr = 0
                self._gdr_staging_buf = None
            self._store.close()
            self._store = None
