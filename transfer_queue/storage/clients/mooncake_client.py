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

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import StorageClientFactory, StorageKVClient
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.tensor_utils import allocate_empty_tensors, get_nbytes, merge_contiguous_memory

logger = get_logger(__name__)

MOONCAKE_STORE_IMPORTED: bool = True
try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

except ImportError:
    MOONCAKE_STORE_IMPORTED = False

from tensordict import NonTensorData as _NonTensorData

BATCH_SIZE_LIMIT: int = 400
MAX_WORKER_THREADS = 4
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0

# Separator joining an original key to a dict sub-key (e.g. "5@mmi::pixel_values").
_DICT_SUBKEY_SEP: str = "::"

# Sentinel marker key identifying a per-key dict-unpack meta entry.
_TQ_DICT_UNPACK_KEY: str = "__tq_dict_unpack__"

# Reserved sub-key name for the bundled non-tensor blob (a 1D uint8 tensor that
# carries pickle bytes of all non-tensor entries of the original dict).
_TQ_EXTRAS_SUBKEY: str = "__tq_extras__"


def _is_dict_unpack_meta(meta: Any) -> bool:
    """True if value is a dict-unpack meta entry."""
    return isinstance(meta, dict) and meta.get(_TQ_DICT_UNPACK_KEY) is True


def _unwrap_non_tensor(value: Any) -> Any:
    """Unwrap a tensordict.NonTensorData to the underlying Python object."""
    if isinstance(value, _NonTensorData):
        return value.data
    return value


def _dict_has_tensor(value: Any) -> bool:
    """True if value is a non-empty dict containing at least one tensor."""
    value = _unwrap_non_tensor(value)
    return (
        isinstance(value, dict)
        and len(value) > 0
        and any(isinstance(v, torch.Tensor) for v in value.values())
    )


def _expand_dict_slots_fn(
    keys: list[str],
    shapes: list[Any],
    dtypes: list[Any],
    custom_backend_meta: list[Any],
) -> tuple[list[str], list[Any], list[Any], list[tuple]]:
    """Expand dict-unpack slots into a flat list of sub-keys plus instructions
    for rebuilding each original slot.
    """
    flat_keys: list[str] = []
    flat_shapes: list[Any] = []
    flat_dtypes: list[Any] = []
    reconstruct: list[tuple] = []
    for i, key in enumerate(keys):
        meta = custom_backend_meta[i]
        if _is_dict_unpack_meta(meta):
            tensor_sub_idxs: list[int] = []
            for sk, sd, ss in zip(
                meta["tensor_keys"], meta["tensor_dtypes"], meta["tensor_shapes"], strict=True
            ):
                flat_keys.append(f"{key}{_DICT_SUBKEY_SEP}{sk}")
                flat_shapes.append(ss)
                flat_dtypes.append(sd)
                tensor_sub_idxs.append(len(flat_keys) - 1)
            extras_idx = -1
            extras_size = meta.get("extras_size", 0)
            if extras_size > 0:
                flat_keys.append(f"{key}{_DICT_SUBKEY_SEP}{_TQ_EXTRAS_SUBKEY}")
                flat_shapes.append([extras_size])
                flat_dtypes.append(torch.uint8)
                extras_idx = len(flat_keys) - 1
            reconstruct.append(
                (
                    "dict",
                    list(meta["key_order"]),
                    list(meta["tensor_keys"]),
                    tensor_sub_idxs,
                    extras_idx,
                )
            )
        else:
            flat_keys.append(key)
            flat_shapes.append(shapes[i])
            flat_dtypes.append(dtypes[i])
            reconstruct.append(("scalar", len(flat_keys) - 1))
    return flat_keys, flat_shapes, flat_dtypes, reconstruct


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

    def put(self, keys: list[str], values: list[Any]) -> list[Any] | None:
        """Store key-value pairs in MooncakeStore.

        Returns optional per-key backend metadata that ``get`` / ``clear``
        need later; ``None`` when there is nothing to remember.
        """

        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        custom_meta: list[Any] = [None] * len(keys)
        dict_seen: bool = False

        tensor_keys = []
        tensor_values = []
        non_tensor_keys = []
        non_tensor_values = []

        for i, (key, value) in enumerate(zip(keys, values, strict=True)):
            if isinstance(value, torch.Tensor):
                tensor_keys.append(key)
                tensor_values.append(value)
            elif _dict_has_tensor(value):
                # Dict-with-tensor fan-out: avoid the Mooncake bytes pool which
                # silently returns b"" under MB-scale GET pressure (see
                # real_client.cpp:2209 "Failed to allocate buffer"). Each
                # sub-tensor rides the working tensor RDMA path; non-tensor
                # entries are pickled into one uint8 blob and ride the same
                # path as another sub-key. The bytes pool is never touched.
                dict_seen = True
                raw_dict = _unwrap_non_tensor(value)
                key_order = list(raw_dict.keys())
                ts_sub_keys: list[Any] = []
                ts_sub_tensors: list[Tensor] = []
                extras: dict[Any, Any] = {}
                for sk in key_order:
                    v = raw_dict[sk]
                    if isinstance(v, torch.Tensor):
                        ts_sub_keys.append(sk)
                        ts_sub_tensors.append(v)
                    else:
                        extras[sk] = v

                extras_size = 0
                extras_tensor: Tensor | None = None
                if extras:
                    extras_blob = pickle.dumps(extras, protocol=pickle.HIGHEST_PROTOCOL)
                    extras_tensor = torch.frombuffer(bytearray(extras_blob), dtype=torch.uint8)
                    extras_size = extras_tensor.numel()

                custom_meta[i] = {
                    _TQ_DICT_UNPACK_KEY: True,
                    "key_order": key_order,
                    "tensor_keys": ts_sub_keys,
                    "tensor_dtypes": [t.dtype for t in ts_sub_tensors],
                    "tensor_shapes": [list(t.shape) for t in ts_sub_tensors],
                    "extras_size": extras_size,
                }

                for sk, st in zip(ts_sub_keys, ts_sub_tensors, strict=True):
                    tensor_keys.append(f"{key}{_DICT_SUBKEY_SEP}{sk}")
                    tensor_values.append(st)
                if extras_tensor is not None:
                    tensor_keys.append(f"{key}{_DICT_SUBKEY_SEP}{_TQ_EXTRAS_SUBKEY}")
                    tensor_values.append(extras_tensor)
            else:
                non_tensor_keys.append(key)
                non_tensor_values.append(value)

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
            for i in range(0, len(tensor_keys), BATCH_SIZE_LIMIT):
                batch_keys = tensor_keys[i : i + BATCH_SIZE_LIMIT]
                batch_tensors = tensor_values[i : i + BATCH_SIZE_LIMIT]
                futures.append(executor.submit(self._put_tensors_thread_worker, batch_keys, batch_tensors))

            for i in range(0, len(non_tensor_keys), BATCH_SIZE_LIMIT):
                batch_keys = non_tensor_keys[i : i + BATCH_SIZE_LIMIT]
                batch_values = non_tensor_values[i : i + BATCH_SIZE_LIMIT]
                futures.append(executor.submit(self._put_bytes_thread_worker, batch_keys, batch_values))

            for future in as_completed(futures):
                future.result()

        return custom_meta if dict_seen else None

    def _put_tensors_thread_worker(self, batch_keys: list[str], batch_tensors: list[Tensor]) -> None:
        """Worker thread for putting batch of tensors to MooncakeStore."""

        batch_ptrs, batch_sizes, _contiguous_tensors = self._preprocess_tensors_for_put(batch_tensors)
        batch_ptr_reduced, batch_sizes_reduced = merge_contiguous_memory(batch_ptrs, batch_sizes)
        self._register_all_buffers(batch_ptr_reduced, batch_sizes_reduced)

        try:
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
                    break  # All retries in this attempt succeeded.

                logger.error(
                    f"batch_upsert_from retry {attempt}/{MAX_RETRIES} failed for {len(next_failed_keys)} keys "
                    f"with error codes {next_failed_codes}."
                )

                current_failed_indices = next_failed_indices
                current_failed_keys = next_failed_keys
                current_failed_codes = next_failed_codes

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(
                    f"batch_upsert_from failed for keys {current_failed_keys} with error codes "
                    f"{current_failed_codes} after retrying {MAX_RETRIES} times."
                )

        finally:
            self._unregister_all_buffers(batch_ptr_reduced)

    def _put_bytes_thread_worker(self, batch_keys: list[str], batch_values: list[Any]):
        """Worker thread for putting batch of non-tensors to MooncakeStore."""

        serialized_values = [pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL) for v in batch_values]

        # FIXME: When MooncakeStore supports per-key status codes for upsert_batch and get_batch,
        #        switch the bytes write/read paths from whole-batch retry to per-key selective retry,
        #        matching the tensor-path behaviour.
        ret = self._store.upsert_batch(batch_keys, serialized_values, self.replica_config)
        if ret == 0:
            return

        logger.error(
            f"upsert_batch failed for {len(batch_keys)} keys with error code: {ret}. "
            f"Retrying up to {MAX_RETRIES} times..."
        )

        for attempt in range(1, MAX_RETRIES + 1):
            ret = self._store.upsert_batch(batch_keys, serialized_values, self.replica_config)
            if ret == 0:
                logger.info("upsert_batch succeeded after retransmission.")
                return

            logger.error(
                f"upsert_batch retry {attempt}/{MAX_RETRIES} failed for {len(batch_keys)} keys with error code: {ret}."
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"upsert_batch failed for {len(batch_keys)} keys with error code: {ret} after retrying {MAX_RETRIES} times."
        )

    def get(
        self,
        keys: list[str],
        shapes: list[Any] | None = None,
        dtypes: list[Any] | None = None,
        custom_backend_meta: list[Any] | None = None,
    ) -> list[Any]:
        """Fetch values for the given keys from MooncakeStore.

        ``shapes`` and ``dtypes`` describe the expected tensor layout per key
        (use ``None`` for non-tensor slots). ``custom_backend_meta`` carries
        per-key metadata returned by ``put``. Returns values in input order.
        """

        if shapes is None or dtypes is None:
            raise ValueError("MooncakeStoreClient needs shapes and dtypes for zero-copy transfer.")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")
        if custom_backend_meta is not None and len(custom_backend_meta) != len(keys):
            raise ValueError(
                f"Length of custom_backend_meta ({len(custom_backend_meta)}) must match keys ({len(keys)})"
            )

        # Expand dict-of-tensors slots into synthetic sub-keys whose tensors ride
        # the working tensor RDMA path. ``reconstruct`` records how to fold each
        # original slot back from the flat fetch result.
        has_dict = custom_backend_meta is not None and any(
            _is_dict_unpack_meta(m) for m in custom_backend_meta
        )
        if has_dict:
            flat_keys, flat_shapes, flat_dtypes, reconstruct = _expand_dict_slots_fn(
                keys, shapes, dtypes, custom_backend_meta
            )
        else:
            flat_keys, flat_shapes, flat_dtypes = list(keys), list(shapes), list(dtypes)
            reconstruct = None

        flat_results = self._get_flat(flat_keys, flat_shapes, flat_dtypes)

        if reconstruct is None:
            return flat_results

        n_orig = len(keys)
        results: list[Any] = [None] * n_orig
        for orig_i, op in enumerate(reconstruct):
            if op[0] == "scalar":
                results[orig_i] = flat_results[op[1]]
            else:
                _, key_order, tensor_sub_keys, tensor_sub_idxs, extras_idx = op
                tensor_map = {
                    sk: flat_results[j]
                    for sk, j in zip(tensor_sub_keys, tensor_sub_idxs, strict=True)
                }
                if extras_idx >= 0:
                    extras_tensor = flat_results[extras_idx]
                    extras_map = pickle.loads(extras_tensor.numpy().tobytes())
                else:
                    extras_map = {}
                results[orig_i] = {
                    sk: (tensor_map[sk] if sk in tensor_map else extras_map[sk])
                    for sk in key_order
                }
        return results

    def _get_flat(self, keys: list[str], shapes: list[Any], dtypes: list[Any]) -> list[Any]:
        """Fetch a flat list of keys; tensor slots and non-tensor slots are
        dispatched to their respective worker paths.
        """
        tensor_indices = []
        non_tensor_indices = []

        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)

        results = [None] * len(keys)

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
            for i in range(0, len(tensor_indices), BATCH_SIZE_LIMIT):
                batch_indexes = tensor_indices[i : i + BATCH_SIZE_LIMIT]
                batch_keys = [keys[i] for i in batch_indexes]
                batch_shapes = [shapes[i] for i in batch_indexes]
                batch_dtypes = [dtypes[i] for i in batch_indexes]
                futures.append(
                    executor.submit(
                        self._get_tensors_thread_worker, batch_keys, batch_shapes, batch_dtypes, batch_indexes
                    )
                )

            for i in range(0, len(non_tensor_indices), BATCH_SIZE_LIMIT):
                batch_indexes = non_tensor_indices[i : i + BATCH_SIZE_LIMIT]
                batch_keys = [keys[i] for i in batch_indexes]
                futures.append(executor.submit(self._get_bytes_thread_worker, batch_keys, batch_indexes))

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
            ret_codes = self._store.batch_get_into(batch_keys, batch_buffer_ptrs, batch_nbytes)
            if len(ret_codes) != len(batch_keys):
                raise RuntimeError(f"batch_get_into returned {len(ret_codes)} results, expected {len(batch_keys)}")

            failed_indices = [i for i, ret in enumerate(ret_codes) if ret < 0]
            if not failed_indices:
                return batch_buffer_tensors, indexes

            # error handling
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
                    break  # All retries in this attempt succeeded.

                logger.error(
                    f"batch_get_into retry {attempt}/{MAX_RETRIES} failed for {len(next_failed_keys)} keys "
                    f"with error codes {next_failed_codes}."
                )

                # Narrow down to still-failed items for the next retry attempt.
                current_failed_indices = next_failed_indices
                current_failed_keys = next_failed_keys
                current_failed_codes = next_failed_codes

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
            else:
                # All retries exhausted.
                raise RuntimeError(
                    f"batch_get_into failed for keys {current_failed_keys} with error codes "
                    f"{current_failed_codes} after retrying {MAX_RETRIES} times."
                )

        finally:
            self._unregister_all_buffers(region_ptrs)

        return batch_buffer_tensors, indexes

    def _get_bytes_thread_worker(self, batch_keys: list[str], indexes: list[int]) -> tuple[list[Any], list[int]]:
        raw_results = self._store.get_batch(batch_keys)
        if len(raw_results) != len(batch_keys):
            raise RuntimeError(f"get_batch returned {len(raw_results)} items, expected {len(batch_keys)}")

        # FIXME: Use MooncakeStore provided ret codes to detect transmission failures when supported
        # Currently we rely on empty bytes (b'') to detect transmission failures because
        # MooncakeStore does not currently return a separate status code per key.
        failed_indices = [i for i, result in enumerate(raw_results) if result == b""]
        if failed_indices:
            current_failed_keys = [batch_keys[i] for i in failed_indices]
            current_failed_indices = failed_indices

            logger.error(f"get_batch failed for keys {current_failed_keys}. Retrying up to {MAX_RETRIES} times...")

            for attempt in range(1, MAX_RETRIES + 1):
                retry_results = self._store.get_batch(current_failed_keys)

                next_failed_keys = []
                next_failed_indices = []

                for i, result in enumerate(retry_results):
                    original_idx = current_failed_indices[i]
                    if result == b"":
                        next_failed_keys.append(current_failed_keys[i])
                        next_failed_indices.append(original_idx)
                    else:
                        # Write the successfully retried value back to its original slot immediately.
                        raw_results[original_idx] = result

                if not next_failed_indices:
                    logger.info("get_batch succeeded after retransmission.")
                    break  # All retries in this attempt succeeded.

                logger.error(f"get_batch retry {attempt}/{MAX_RETRIES} failed for {len(next_failed_keys)} keys.")

                # Narrow down to still-failed items for the next retry attempt.
                current_failed_keys = next_failed_keys
                current_failed_indices = next_failed_indices

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
            else:
                # All retries exhausted.
                raise RuntimeError(
                    f"get_batch failed for keys {current_failed_keys} after retrying {MAX_RETRIES} times."
                )

        deserialized_results = [pickle.loads(result) if result != b"" else None for result in raw_results]
        return deserialized_results, indexes

    def clear(self, keys: list[str], custom_backend_meta: list[Any] | None = None) -> None:
        """Delete keys from MooncakeStore. If ``custom_backend_meta`` carries
        any dict-unpack entries, the corresponding sub-keys are also removed.
        """
        if custom_backend_meta is not None and len(custom_backend_meta) != len(keys):
            raise ValueError(
                f"Length of custom_backend_meta ({len(custom_backend_meta)}) must match keys ({len(keys)})"
            )

        if custom_backend_meta is None or not any(_is_dict_unpack_meta(m) for m in custom_backend_meta):
            expanded_keys = list(keys)
        else:
            expanded_keys = []
            for key, meta in zip(keys, custom_backend_meta, strict=True):
                if _is_dict_unpack_meta(meta):
                    for sk in meta["tensor_keys"]:
                        expanded_keys.append(f"{key}{_DICT_SUBKEY_SEP}{sk}")
                    if meta.get("extras_size", 0) > 0:
                        expanded_keys.append(f"{key}{_DICT_SUBKEY_SEP}{_TQ_EXTRAS_SUBKEY}")
                else:
                    expanded_keys.append(key)

        ret_codes = self._store.batch_remove(expanded_keys, force=True)
        for i, ret in enumerate(ret_codes):
            if not (ret == 0 or ret == -704):
                logger.error(f"remove failed for key `{expanded_keys[i]}` with error code: {ret}")

    def close(self):
        """Closes MooncakeStore."""
        if self._store:
            self._store.close()
            self._store = None

    @staticmethod
    def _preprocess_tensors_for_put(values: list[Tensor]) -> tuple[list[int], list[int], list[Tensor]]:
        ptr_list: list[int] = []
        size_list: list[int] = []
        tensor_list: list[Tensor] = []  # hold reference for the contiguous tensor
        for t in values:
            # TODO: support gpu direct rdma and use different data paths.
            #       For GPU, it's more reasonable to perform data copy since
            #       The register overhead is much higher than CPU
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
