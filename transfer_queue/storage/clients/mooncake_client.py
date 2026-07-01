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

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, cast

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import StorageClientFactory, StorageKVClient
from transfer_queue.utils import serial_utils
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.mooncake_utils import (
    GdrStaging,
    _aligned_offsets,
    chunk_subkeys,
    split_by_bytes,
)
from transfer_queue.utils.tensor_utils import allocate_empty_tensors, get_nbytes, merge_contiguous_memory

logger = get_logger(__name__)

MOONCAKE_STORE_IMPORTED: bool = True
try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

except ImportError:
    MOONCAKE_STORE_IMPORTED = False

BATCH_SIZE_LIMIT: int = 400
MAX_BATCH_WORKER_THREADS = 4
MAX_SERIAL_WORKER_THREADS = 4
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0


@StorageClientFactory.register("MooncakeStoreClient")
class MooncakeStoreClient(StorageKVClient):
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

        self.use_gdr = bool(config.get("use_gdr", False))
        # gdr_staging_buffer_mb > 0: use persistent staging buffer (GDR path).
        # gdr_staging_buffer_mb = 0: fall back to CPU RDMA path even if use_gdr=True.
        self.gdr_staging_buffer_mb = int(config.get("gdr_staging_buffer_mb", 1024))
        buffer_bytes = self.gdr_staging_buffer_mb * 1024 * 1024
        # GdrStaging instance created eagerly but cudaMalloc is deferred to first use.
        # Skip GDR if CUDA context is not initialized in this process (e.g. CPU-only workers)
        gdr_eligible = self.use_gdr and buffer_bytes > 0 and torch.cuda.is_initialized()
        self._gdr_staging: GdrStaging | None = GdrStaging(buffer_bytes) if gdr_eligible else None

        if self.local_hostname is None or self.local_hostname == "":
            from transfer_queue.utils.zmq_utils import get_node_ip_address

            ip = get_node_ip_address()
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

        self.replica_config = ReplicateConfig()
        self.replica_config.with_hard_pin = True

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

    def put(self, keys: list[str], values: list[Any]) -> list[dict | None]:
        """Stores multiple key-value pairs to MooncakeStore.

        Args:
            keys (List[str]): List of unique string identifiers.
            values (List[Any]): List of values to store (tensors, scalars, dicts, etc.).

        Returns:
            Per-key metadata aligned with ``keys``. Tensor entries are ``None``;
            non-tensor entries carry ``{"packed_size": int}`` so the get-side
            can pre-allocate the receive buffer.
        """

        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        use_gdr_path = self.use_gdr and self._gdr_staging is not None

        tensor_keys: list[str] = []
        tensor_values: list[Tensor] = []
        non_tensor_keys: list[str] = []
        non_tensor_values: list[Any] = []

        for key, value in zip(keys, values, strict=True):
            if isinstance(value, torch.Tensor):
                tensor_keys.append(key)
                tensor_values.append(value)
            else:
                non_tensor_keys.append(key)
                non_tensor_values.append(value)

        gdr_meta: dict[str, dict | None] = {}
        if use_gdr_path and tensor_keys:
            gdr_meta = dict(zip(tensor_keys, self._put_tensors_gdr(tensor_keys, tensor_values), strict=False))

        tensor_futures: list[Future[None]] = []
        bytes_futures: list[Future[list[int]]] = []
        with ThreadPoolExecutor(max_workers=MAX_BATCH_WORKER_THREADS) as executor:
            if not use_gdr_path:
                for i in range(0, len(tensor_keys), BATCH_SIZE_LIMIT):
                    batch_keys = tensor_keys[i : i + BATCH_SIZE_LIMIT]
                    batch_tensors = tensor_values[i : i + BATCH_SIZE_LIMIT]
                    tensor_futures.append(executor.submit(self._put_tensors_thread_worker, batch_keys, batch_tensors))

            for i in range(0, len(non_tensor_keys), BATCH_SIZE_LIMIT):
                batch_keys = non_tensor_keys[i : i + BATCH_SIZE_LIMIT]
                batch_values = non_tensor_values[i : i + BATCH_SIZE_LIMIT]
                bytes_futures.append(executor.submit(self._put_bytes_thread_worker, batch_keys, batch_values))

            packed_sizes: list[int] = []
            for bf in bytes_futures:
                packed_sizes.extend(bf.result())

            for tf in tensor_futures:
                tf.result()

        # Walk keys/values once to scatter results back to original slots.
        sizes_iter = iter(packed_sizes)
        custom_backend_meta: list[dict | None] = [
            gdr_meta.get(key) if isinstance(value, torch.Tensor) else {"packed_size": next(sizes_iter)}
            for key, value in zip(keys, values, strict=False)
        ]

        return custom_backend_meta

    def _put_tensors_gdr(self, batch_keys: list[str], batch_tensors: list[Tensor]) -> list[dict | None]:
        """GDR tensor PUT path using the persistent pre-registered staging buffer.

        split_by_bytes() groups tensors so each group's aligned total fits within the
        staging buffer. Oversized tensors (nbytes > buffer_size) get a singleton group
        and are stored as :c{i} sub-keys. Normal groups are packed and upserted together.

        Returns per-key meta: None for normal tensors, {"n_chunks": n} for oversized
        tensors that were split into :c{i} sub-keys. clear() uses this to expand keys.
        """
        assert self._gdr_staging is not None
        self._gdr_staging.lazy_init(self._store)
        staging = self._gdr_staging
        buffer_size = staging.size

        # Caller guarantees all tensors are CUDA; make contiguous outside the lock.
        contiguous_tensors = [t.contiguous() for t in batch_tensors]
        nbytes = [t.nbytes for t in contiguous_tensors]
        groups = split_by_bytes(nbytes, buffer_size)

        meta: list[dict | None] = [None] * len(batch_keys)

        with staging.acquire():
            for idxs in groups:
                g_keys = [batch_keys[i] for i in idxs]
                g_tensors = [contiguous_tensors[i] for i in idxs]

                if len(idxs) == 1 and g_tensors[0].nbytes > buffer_size:
                    # Oversized tensor: split into :c{i} sub-keys.
                    tensor = g_tensors[0]
                    key = g_keys[0]
                    sub_keys = chunk_subkeys(key, tensor.nbytes, buffer_size)
                    memcpy_chunk = staging.memcpy_d2d_async if tensor.is_cuda else staging.memcpy_h2d_async
                    for i, sub_key in enumerate(sub_keys):
                        chunk_size = min(buffer_size, tensor.nbytes - i * buffer_size)
                        memcpy_chunk(staging.ptr, tensor.data_ptr() + i * buffer_size, chunk_size)
                        staging.synchronize()
                        self._batch_upsert_with_retry([sub_key], [staging.ptr], [chunk_size])
                    meta[idxs[0]] = {"n_chunks": len(sub_keys)}
                else:
                    # Normal group: aligned total fits in buffer; pack and upsert together.
                    sub_ptrs, sizes = staging.pack(g_tensors)
                    self._batch_upsert_with_retry(g_keys, sub_ptrs, sizes)

        return meta

    def _put_tensors_thread_worker(self, batch_keys: list[str], batch_tensors: list[Tensor]) -> None:
        """Worker thread for putting batch of cpu tensors to MooncakeStore."""

        batch_ptrs, batch_sizes, _ = self._preprocess_tensors_for_put(batch_tensors)
        batch_ptr_reduced, batch_sizes_reduced = merge_contiguous_memory(batch_ptrs, batch_sizes)
        self._register_all_buffers(batch_ptr_reduced, batch_sizes_reduced)
        try:
            self._batch_upsert_with_retry(batch_keys, batch_ptrs, batch_sizes)
        finally:
            self._unregister_all_buffers(batch_ptr_reduced)

    def _put_bytes_thread_worker(self, batch_keys: list[str], batch_values: list[Any]) -> list[int]:
        """Worker thread for putting batch of non-tensors to MooncakeStore."""

        # TODO: switch to a pre-registered buffer from MooncakeStore once such an API is available.
        region_ptrs: list[int] = []
        region_sizes: list[int] = []

        def alloc(sizes: list[int]) -> list[Tensor]:
            nonlocal region_ptrs, region_sizes
            # `batch_packed_sizes` are byte counts. With torch.uint8 (1 byte/element),
            # a 1-D shape of (N,) corresponds to exactly N bytes. We use
            # `allocate_empty_tensors` to get N uint8 views over a single contiguous,
            # register-able region. These are plain byte buffers, not real tensors;
            # consumers apply the actual dtype/shape interpretation when unpacking.
            dtypes = [torch.uint8] * len(sizes)
            shapes = [(s,) for s in sizes]
            buffers, _, region_ptrs, region_sizes = allocate_empty_tensors(dtypes, shapes)
            return buffers

        buffers, batch_sizes = serial_utils.batch_encode_into(
            batch_values, alloc, num_workers=MAX_SERIAL_WORKER_THREADS
        )
        batch_ptrs = [cast(Tensor, b).data_ptr() for b in buffers]

        self._register_all_buffers(region_ptrs, region_sizes)
        try:
            self._batch_upsert_with_retry(batch_keys, batch_ptrs, batch_sizes)
        finally:
            self._unregister_all_buffers(region_ptrs)

        return batch_sizes

    def get(
        self,
        keys: list[str],
        shapes: list[Any] | None = None,
        dtypes: list[Any] | None = None,
        custom_backend_meta: list[dict | None] | None = None,
    ) -> list[Any]:
        """Get multiple key-value pairs from MooncakeStore.

        Args:
            keys: Keys to fetch.
            shapes: Expected tensor shapes (use [] for scalars).
            dtypes: Expected dtypes; use None for non-tensor data.
            custom_backend_meta: Per-key dicts; non-tensor entries must carry
                ``{"packed_size": int}`` so the receive buffer can be sized.

        Returns:
            Retrieved values in the same order as input keys.
        """

        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStoreClient needs shapes and dtypes for zero-copy transfer.")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        use_gdr_path = self.use_gdr and self._gdr_staging is not None

        gpu_tensor_indices: list[int] = []
        gpu_tensor_keys: list[str] = []
        gpu_tensor_shapes: list[Any] = []
        gpu_tensor_dtypes: list[Any] = []
        cpu_tensor_indices: list[int] = []
        cpu_tensor_keys: list[str] = []
        cpu_tensor_shapes: list[Any] = []
        cpu_tensor_dtypes: list[Any] = []
        non_tensor_indices: list[int] = []
        non_tensor_keys: list[str] = []
        non_tensor_packed_sizes: list[int] = []

        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                if use_gdr_path:
                    gpu_tensor_indices.append(i)
                    gpu_tensor_keys.append(keys[i])
                    gpu_tensor_shapes.append(shapes[i])
                    gpu_tensor_dtypes.append(dtype)
                else:
                    cpu_tensor_indices.append(i)
                    cpu_tensor_keys.append(keys[i])
                    cpu_tensor_shapes.append(shapes[i])
                    cpu_tensor_dtypes.append(dtype)
            else:
                non_tensor_indices.append(i)
                non_tensor_keys.append(keys[i])

        if (gpu_tensor_indices and use_gdr_path) or non_tensor_indices:
            if custom_backend_meta is None or len(custom_backend_meta) != len(keys):
                raise ValueError(
                    "custom_backend_meta is required when GDR is enabled (for n_chunks) "
                    "or when any dtype is None (for packed_size)."
                )

        if non_tensor_indices:
            assert custom_backend_meta is not None
            for j in non_tensor_indices:
                meta = custom_backend_meta[j]
                assert meta is not None
                non_tensor_packed_sizes.append(meta["packed_size"])

        results = [None] * len(keys)

        if gpu_tensor_keys:
            assert custom_backend_meta is not None
            gpu_tensor_meta = [custom_backend_meta[i] for i in gpu_tensor_indices]
            retrieved, batch_idx = self._get_tensors_gdr(
                gpu_tensor_keys, gpu_tensor_shapes, gpu_tensor_dtypes, gpu_tensor_indices, gpu_tensor_meta
            )
            for idx, val in zip(batch_idx, retrieved, strict=True):
                results[idx] = val

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_BATCH_WORKER_THREADS) as executor:
            for i in range(0, len(cpu_tensor_indices), BATCH_SIZE_LIMIT):
                batch_keys = cpu_tensor_keys[i : i + BATCH_SIZE_LIMIT]
                batch_shapes = cpu_tensor_shapes[i : i + BATCH_SIZE_LIMIT]
                batch_dtypes = cpu_tensor_dtypes[i : i + BATCH_SIZE_LIMIT]
                batch_indexes = cpu_tensor_indices[i : i + BATCH_SIZE_LIMIT]
                futures.append(
                    executor.submit(
                        self._get_tensors_thread_worker, batch_keys, batch_shapes, batch_dtypes, batch_indexes
                    )
                )

            for i in range(0, len(non_tensor_indices), BATCH_SIZE_LIMIT):
                batch_keys = non_tensor_keys[i : i + BATCH_SIZE_LIMIT]
                batch_packed_sizes = non_tensor_packed_sizes[i : i + BATCH_SIZE_LIMIT]
                batch_indexes = non_tensor_indices[i : i + BATCH_SIZE_LIMIT]
                futures.append(
                    executor.submit(self._get_bytes_thread_worker, batch_keys, batch_packed_sizes, batch_indexes)
                )

            for future in as_completed(futures):
                retrieved_values, batch_indexes = future.result()
                for idx, val in zip(batch_indexes, retrieved_values, strict=True):
                    results[idx] = val

        return results

    def _get_tensors_thread_worker(
        self, batch_keys: list[str], batch_shapes: list[tuple], batch_dtypes: list[torch.dtype], indexes: list[int]
    ) -> tuple[list[Tensor], list[int]]:
        batch_nbytes = get_nbytes(batch_dtypes, batch_shapes)
        batch_buffer_tensors, batch_buffer_ptrs, region_ptrs, region_sizes = allocate_empty_tensors(
            batch_dtypes, batch_shapes
        )

        self._register_all_buffers(region_ptrs, region_sizes)
        try:
            self._batch_get_into_with_retry(batch_keys, batch_buffer_ptrs, batch_nbytes)
        finally:
            self._unregister_all_buffers(region_ptrs)

        return batch_buffer_tensors, indexes

    def _get_tensors_gdr(
        self,
        batch_keys: list[str],
        batch_shapes: list[tuple],
        batch_dtypes: list[torch.dtype],
        indexes: list[int],
        batch_meta: list[dict | None],
    ) -> tuple[list[Tensor], list[int]]:
        """GDR tensor GET path using the persistent pre-registered staging buffer.

        split_by_bytes() groups tensors so each group's aligned total fits within the
        staging buffer. Oversized singleton groups reassemble from :c{i} sub-keys.
        Normal groups use a single batch_get_into + unpack.

        NOTE: An alternative design is to skip the staging buffer entirely: for each group,
        cudaMalloc a fresh buffer, register it, RDMA GET directly into it, unregister it,
        then slice into tensors via torch.from_blob (eliminating the D2D copy and the staging
        buffer lock). However, all tensors in a group would share one underlying buffer via
        PyTorch's storage refcount — the buffer is freed only when the last tensor in the
        group is GC'd. A single long-lived tensor silently keeps the entire batch allocation
        alive, which is a hard-to-debug memory leak. We keep the D2D copy for now to give
        each returned tensor an independent PyTorch-managed lifetime.
        """
        assert self._gdr_staging is not None
        self._gdr_staging.lazy_init(self._store)
        staging = self._gdr_staging
        device = torch.device("cuda", torch.cuda.current_device())
        buffer_size = staging.size
        batch_nbytes = get_nbytes(batch_dtypes, batch_shapes)

        # Grouping happens outside the lock.
        groups = split_by_bytes(batch_nbytes, buffer_size)

        tensors: list[torch.Tensor] = [None] * len(batch_keys)  # type: ignore[list-item]

        with staging.acquire():
            for idxs in groups:
                g_keys = [batch_keys[i] for i in idxs]
                g_nbytes = [batch_nbytes[i] for i in idxs]
                g_dtypes = [batch_dtypes[i] for i in idxs]
                g_shapes = [batch_shapes[i] for i in idxs]

                if len(idxs) == 1 and g_nbytes[0] > buffer_size:
                    # Oversized tensor: reassemble from :c{i} sub-keys.
                    key = g_keys[0]
                    total = g_nbytes[0]
                    meta_entry = batch_meta[idxs[0]]
                    assert meta_entry is not None
                    n_chunks = meta_entry["n_chunks"]
                    sub_keys = [f"{key}:c{i}" for i in range(n_chunks)]
                    final_tensor = torch.empty(tuple(g_shapes[0]), dtype=g_dtypes[0], device=device)
                    for i, sub_key in enumerate(sub_keys):
                        chunk_size = min(buffer_size, total - i * buffer_size)
                        self._batch_get_into_with_retry([sub_key], [staging.ptr], [chunk_size])
                        staging.memcpy_d2d_async(final_tensor.data_ptr() + i * buffer_size, staging.ptr, chunk_size)
                    staging.synchronize()
                    tensors[idxs[0]] = final_tensor
                else:
                    # Normal group: aligned total fits; batch_get_into then unpack.
                    offsets, _ = _aligned_offsets(g_nbytes)
                    sub_ptrs = [staging._ptr + off for off in offsets]
                    self._batch_get_into_with_retry(g_keys, sub_ptrs, g_nbytes)
                    unpacked = staging.unpack(sub_ptrs, g_nbytes, g_dtypes, g_shapes, device)
                    for pos, t in zip(idxs, unpacked, strict=True):
                        tensors[pos] = t

        return tensors, indexes

    def _get_bytes_thread_worker(
        self, batch_keys: list[str], batch_packed_sizes: list[int], indexes: list[int]
    ) -> tuple[list[Any], list[int]]:
        # `batch_packed_sizes` are byte counts. With torch.uint8 (1 byte/element),
        # a 1-D shape of (N,) corresponds to exactly N bytes. We use
        # `allocate_empty_tensors` to get N uint8 views over a single contiguous,
        # register-able region. These are plain byte buffers, not real tensors;
        # consumers apply the actual dtype/shape interpretation when unpacking.
        batch_shapes = [(sz,) for sz in batch_packed_sizes]
        batch_dtypes = [torch.uint8] * len(batch_keys)
        batch_nbytes = get_nbytes(batch_dtypes, batch_shapes)
        batch_buffer_tensors, batch_buffer_ptrs, region_ptrs, region_sizes = allocate_empty_tensors(
            batch_dtypes, batch_shapes
        )

        self._register_all_buffers(region_ptrs, region_sizes)
        try:
            self._batch_get_into_with_retry(batch_keys, batch_buffer_ptrs, batch_nbytes)
        finally:
            self._unregister_all_buffers(region_ptrs)

        return serial_utils.batch_decode_from(batch_buffer_tensors), indexes

    def clear(self, keys: list[str], custom_backend_meta: list[Any] | None = None) -> None:
        """Deletes multiple keys from MooncakeStore.

        Args:
            keys (List[str]): List of keys to remove.
            custom_backend_meta (List[Any], optional): ...
        """
        if self._gdr_staging is not None and custom_backend_meta is not None:
            actual_keys: list[str] = []
            for key, meta in zip(keys, custom_backend_meta, strict=True):
                if isinstance(meta, dict) and "n_chunks" in meta:
                    actual_keys.extend(f"{key}:c{i}" for i in range(meta["n_chunks"]))
                else:
                    actual_keys.append(key)
        else:
            if self._gdr_staging is not None:
                logger.warning(
                    "GDR is enabled but custom_backend_meta is None; chunked sub-keys (if any) will not be removed."
                )
            actual_keys = keys

        ret_codes = self._store.batch_remove(actual_keys, force=True)
        for i, ret in enumerate(ret_codes):
            if not (ret == 0 or ret == -704):
                logger.error(f"remove failed for key `{actual_keys[i]}` with error code: {ret}")

    def close(self):
        """Closes MooncakeStore."""
        if self._gdr_staging is not None:
            self._gdr_staging.close(self._store)
            self._gdr_staging = None
        if self._store:
            self._store.close()
            self._store = None

    def _batch_upsert_with_retry(self, batch_keys: list[str], batch_ptrs: list[int], batch_sizes: list[int]) -> None:
        """Run ``batch_upsert_from`` with per-key retry; raise on permanent failure.

        Caller owns the memory regions (register/unregister and lifetime of the
        backing tensors/buffers).
        """
        results = self._store.batch_upsert_from(batch_keys, batch_ptrs, batch_sizes, config=self.replica_config)
        if len(results) != len(batch_keys):
            raise RuntimeError(f"batch_upsert_from returned {len(results)} results, expected {len(batch_keys)}")

        failed_indices = [j for j, r in enumerate(results) if r != 0]
        if not failed_indices:
            return

        current_failed_keys = [batch_keys[i] for i in failed_indices]
        current_failed_codes = [results[i] for i in failed_indices]
        current_failed_indices = failed_indices

        logger.error(
            f"batch_upsert_from failed for keys {current_failed_keys} with error codes {current_failed_codes}. "
            f"Retrying up to {MAX_RETRIES} times..."
        )

        for attempt in range(1, MAX_RETRIES + 1):
            retry_ptrs = [batch_ptrs[i] for i in current_failed_indices]
            retry_sizes = [batch_sizes[i] for i in current_failed_indices]

            retry_results = self._store.batch_upsert_from(
                current_failed_keys, retry_ptrs, retry_sizes, config=self.replica_config
            )

            next_failed_indices = []
            next_failed_keys = []
            next_failed_codes = []

            for i, ret in enumerate(retry_results):
                if ret != 0:
                    next_failed_indices.append(current_failed_indices[i])
                    next_failed_keys.append(current_failed_keys[i])
                    next_failed_codes.append(ret)

            if not next_failed_indices:
                logger.info("batch_upsert_from succeeded after retransmission.")
                return

            logger.error(
                f"batch_upsert_from retry {attempt}/{MAX_RETRIES} failed for {len(next_failed_keys)} keys "
                f"with error codes {next_failed_codes}."
            )

            current_failed_indices = next_failed_indices
            current_failed_keys = next_failed_keys
            current_failed_codes = next_failed_codes

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"batch_upsert_from failed for keys {current_failed_keys} with error codes "
            f"{current_failed_codes} after retrying {MAX_RETRIES} times."
        )

    def _batch_get_into_with_retry(
        self, batch_keys: list[str], batch_buffer_ptrs: list[int], batch_nbytes: list[int]
    ) -> None:
        """Run ``batch_get_into`` with per-key retry; raise on permanent failure.

        Caller owns the receive buffers (allocate/register/unregister).
        """
        ret_codes = self._store.batch_get_into(batch_keys, batch_buffer_ptrs, batch_nbytes)
        if len(ret_codes) != len(batch_keys):
            raise RuntimeError(f"batch_get_into returned {len(ret_codes)} results, expected {len(batch_keys)}")

        failed_indices = [i for i, ret in enumerate(ret_codes) if ret < 0]
        if not failed_indices:
            return

        current_failed_keys = [batch_keys[i] for i in failed_indices]
        current_failed_codes = [ret_codes[i] for i in failed_indices]
        current_failed_indices = failed_indices

        logger.error(
            f"batch_get_into failed for keys {current_failed_keys} with error codes {current_failed_codes}. "
            f"Retrying up to {MAX_RETRIES} times..."
        )

        for attempt in range(1, MAX_RETRIES + 1):
            # Reuse the originally allocated pointers; no need to allocate/register new buffers.
            retry_ptrs = [batch_buffer_ptrs[i] for i in current_failed_indices]
            retry_nbytes = [batch_nbytes[i] for i in current_failed_indices]

            retry_codes = self._store.batch_get_into(current_failed_keys, retry_ptrs, retry_nbytes)

            next_failed_indices = []
            next_failed_keys = []
            next_failed_codes = []

            for i, ret in enumerate(retry_codes):
                if ret < 0:
                    next_failed_indices.append(current_failed_indices[i])
                    next_failed_keys.append(current_failed_keys[i])
                    next_failed_codes.append(ret)

            if not next_failed_indices:
                logger.info("batch_get_into succeeded after retransmission.")
                return

            logger.error(
                f"batch_get_into retry {attempt}/{MAX_RETRIES} failed for {len(next_failed_keys)} keys "
                f"with error codes {next_failed_codes}."
            )

            current_failed_indices = next_failed_indices
            current_failed_keys = next_failed_keys
            current_failed_codes = next_failed_codes

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"batch_get_into failed for keys {current_failed_keys} with error codes "
            f"{current_failed_codes} after retrying {MAX_RETRIES} times."
        )

    @staticmethod
    def _preprocess_tensors_for_put(values: list[Tensor]) -> tuple[list[int], list[int], list[Tensor]]:
        ptr_list: list[int] = []
        size_list: list[int] = []
        tensor_list: list[Tensor] = []  # hold reference for the contiguous tensor
        for t in values:
            if t.device.type == "cuda":
                t = t.cpu()
            t = t.contiguous()
            tensor_list.append(t)
            ptr_list.append(t.data_ptr())
            size_list.append(t.nbytes)
        return ptr_list, size_list, tensor_list

    def _register_all_buffers(self, ptrs, sizes):
        for ptr, size in zip(ptrs, sizes, strict=True):
            self._store.register_buffer(ptr, size)

    def _unregister_all_buffers(self, ptrs):
        for ptr in ptrs:
            self._store.unregister_buffer(ptr)
