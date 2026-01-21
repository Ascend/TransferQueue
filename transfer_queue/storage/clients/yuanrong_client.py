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

import logging
import os
import pickle
from typing import Any, Optional

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

NPU_DS_CLIENT_KEYS_LIMIT: int = 9999
CPU_DS_CLIENT_KEYS_LIMIT: int = 1999
YUANRONG_DATASYSTEM_IMPORTED: bool = True
TORCH_NPU_IMPORTED: bool = True
try:
    from yr import datasystem
except ImportError:
    YUANRONG_DATASYSTEM_IMPORTED = False


@StorageClientFactory.register("YuanrongStorageClient")
class YuanrongStorageClient(TransferQueueStorageKVClient):
    """
    Storage client for YuanRong DataSystem.

    Supports storing and fetching both:
    - NPU tensors via DsTensorClient (for high performance).
    - General objects (CPU tensors, str, bool, list, etc.) via KVClient with pickle serialization.
    """

    def __init__(self, config: dict[str, Any]):
        if not YUANRONG_DATASYSTEM_IMPORTED:
            raise ImportError("YuanRong DataSystem not installed.")

        global TORCH_NPU_IMPORTED
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            TORCH_NPU_IMPORTED = False

        self.host = config.get("host")
        self.port = config.get("port")

        self.device_id = None
        self._npu_ds_client = None
        self._cpu_ds_client = None

        if not TORCH_NPU_IMPORTED:
            logger.warning(
                "'torch_npu' import failed. "
                "It results in the inability to quickly put/get tensors on the NPU side, which may affect performance."
            )
        elif not torch.npu.is_available():
            logger.warning(
                "NPU is not available. "
                "It results in the inability to quickly put/get tensors on the NPU side, which may affect performance."
            )
        else:
            self.device_id = torch.npu.current_device()
            self._npu_ds_client = datasystem.DsTensorClient(self.host, self.port, self.device_id)
            self._npu_ds_client.init()

        self._cpu_ds_client = datasystem.KVClient(self.host, self.port)
        self._cpu_ds_client.init()

    def npu_ds_client_is_available(self):
        return self._npu_ds_client is not None

    def cpu_ds_client_is_available(self):
        return self._cpu_ds_client is not None

    def _create_empty_npu_tensorlist(self, shapes, dtypes):
        """
        Create a list of empty NPU tensors with given shapes and dtypes.

        Args:
            shapes (list): List of tensor shapes (e.g., [(3,), (2, 4)])
            dtypes (list): List of torch dtypes (e.g., [torch.float32, torch.int64])
        Returns:
            list: List of uninitialized NPU tensors
        """
        tensors: list[Tensor] = []
        for shape, dtype in zip(shapes, dtypes, strict=False):
            tensor = torch.empty(shape, dtype=dtype, device=f"npu:{self.device_id}")
            tensors.append(tensor)
        return tensors

    def _batch_put(self, keys: list[str], values: list[Any]):
        """Stores a batch of key-value pairs to remote storage, splitting by device type.

        NPU tensors are sent via DsTensorClient (with higher batch limit),
        while all other objects are pickled and sent via KVClient.

        Args:
            keys (List[str]): List of string keys.
            values (List[Any]): Corresponding values (tensors or general objects).
        """
        if self.npu_ds_client_is_available():
            # Classify NPU and CPU data
            npu_keys = []
            npu_values = []

            cpu_keys = []
            cpu_values = []

            for key, value in zip(keys, values, strict=True):
                if isinstance(value, Tensor) and value.device.type == "npu":
                    if not value.is_contiguous():
                        raise ValueError(f"NPU Tensor is not contiguous: {value}")
                    npu_keys.append(key)
                    npu_values.append(value)

                else:
                    cpu_keys.append(key)
                    # TODO: Optimize serialization of tensors
                    # Serializing slice of tensors results in entire tensors being serialized
                    cpu_values.append(pickle.dumps(value.clone() if isinstance(value, Tensor) else value))

            # put NPU data
            for i in range(0, len(npu_keys), NPU_DS_CLIENT_KEYS_LIMIT):
                batch_keys = npu_keys[i : i + NPU_DS_CLIENT_KEYS_LIMIT]
                batch_values = npu_values[i : i + NPU_DS_CLIENT_KEYS_LIMIT]

                # _npu_ds_client.dev_mset doesn't support to overwrite
                try:
                    self._npu_ds_client.dev_delete(batch_keys)
                except Exception as e:
                    logger.warning(f"dev_delete error({e}) before dev_mset")
                self._npu_ds_client.dev_mset(batch_keys, batch_values)

            # put CPU data
            for i in range(0, len(cpu_keys), CPU_DS_CLIENT_KEYS_LIMIT):
                batch_keys = cpu_keys[i : i + CPU_DS_CLIENT_KEYS_LIMIT]
                batch_values = cpu_values[i : i + CPU_DS_CLIENT_KEYS_LIMIT]
                self._cpu_ds_client.mset(batch_keys, batch_values)

        else:
            #  All data goes through CPU path
            pickled_values = [pickle.dumps(v.clone() if isinstance(v, Tensor) else v) for v in values]
            for i in range(0, len(keys), CPU_DS_CLIENT_KEYS_LIMIT):
                batch_keys = keys[i : i + CPU_DS_CLIENT_KEYS_LIMIT]
                batch_vals = pickled_values[i : i + CPU_DS_CLIENT_KEYS_LIMIT]
                self._cpu_ds_client.mset(batch_keys, batch_vals)

    def put(self, keys: list[str], values: list[Any]) -> Optional[list[Any]]:
        """Stores multiple key-value pairs to remote storage.

        Automatically routes NPU tensors to high-performance tensor storage,
        and other objects to general-purpose KV storage.

        Args:
            keys (List[str]): List of unique string identifiers.
            values (List[Any]): List of values to store (tensors, scalars, dicts, etc.).
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")
        self._batch_put(keys, values)
        return None

    def _batch_get(self, keys: list[str], shapes: list, dtypes: list) -> Optional[list[Any]]:
        """Retrieves a batch of values from remote storage using expected metadata.

        NPU tensors are fetched via DsTensorClient using pre-allocated buffers.
        Other objects are fetched via KVClient and unpickled.

        Args:
            keys (List[str]): Keys to fetch.
            shapes (List[List[int]]): Expected shapes for each key (empty list for scalars).
            dtypes (List[Optional[torch.dtype]]): Expected dtypes; None indicates non-tensor data.

        Returns:
            List[Any]: Retrieved values in the same order as input keys.
        """

        if self.npu_ds_client_is_available():
            # classify npu and cpu queries
            npu_indices = []
            npu_keys = []
            npu_shapes = []
            npu_dtypes = []

            cpu_indices = []
            cpu_keys = []

            for idx, (key, shape, dtype) in enumerate(zip(keys, shapes, dtypes, strict=False)):
                if dtype is not None:
                    npu_indices.append(idx)
                    npu_keys.append(key)
                    npu_shapes.append(shape)
                    npu_dtypes.append(dtype)
                else:
                    cpu_indices.append(idx)
                    cpu_keys.append(key)

            results = [None] * len(keys)

            # Fetch NPU tensors
            for i in range(0, len(npu_keys), NPU_DS_CLIENT_KEYS_LIMIT):
                batch_keys = npu_keys[i : i + NPU_DS_CLIENT_KEYS_LIMIT]
                batch_shapes = npu_shapes[i : i + NPU_DS_CLIENT_KEYS_LIMIT]
                batch_dtypes = npu_dtypes[i : i + NPU_DS_CLIENT_KEYS_LIMIT]
                batch_indices = npu_indices[i : i + NPU_DS_CLIENT_KEYS_LIMIT]

                batch_values = self._create_empty_npu_tensorlist(batch_shapes, batch_dtypes)
                failed_subkeys = []
                try:
                    failed_subkeys = self._npu_ds_client.dev_mget(batch_keys, batch_values)
                    # failed_keys = f'{key},{npu_device_id}'
                    failed_subkeys = [f_key.rsplit(",", 1)[0] for f_key in failed_subkeys]
                except Exception:
                    failed_subkeys = batch_keys

                # Fill successfully retrieved tensors
                failed_set = set(failed_subkeys)
                for idx, key, value in zip(batch_indices, batch_keys, batch_values, strict=False):
                    if key not in failed_set:
                        results[idx] = value

                # Add failed keys to CPU fallback queue
                if failed_subkeys:
                    cpu_keys.extend(failed_subkeys)
                    cpu_indices.extend([batch_indices[j] for j, k in enumerate(batch_keys) if k in failed_set])

            # Fetch CPU/general objects (including NPU fallbacks)
            for i in range(0, len(cpu_keys), CPU_DS_CLIENT_KEYS_LIMIT):
                batch_keys = cpu_keys[i : i + CPU_DS_CLIENT_KEYS_LIMIT]
                batch_indices = cpu_indices[i : i + CPU_DS_CLIENT_KEYS_LIMIT]
                raw_values = self._cpu_ds_client.get(batch_keys)
                for idx, raw_val in zip(batch_indices, raw_values, strict=False):
                    results[idx] = pickle.loads(raw_val)

            return results
        else:
            # npu is not available, goes through cpu_ds_client
            results = [None] * len(keys)
            idx = 0
            for i in range(0, len(keys), CPU_DS_CLIENT_KEYS_LIMIT):
                batch_keys = keys[i : i + CPU_DS_CLIENT_KEYS_LIMIT]
                raw_values = self._cpu_ds_client.get(batch_keys)
                for raw_val in raw_values:
                    results[idx] = pickle.loads(raw_val)
                    idx += 1
            return results

    def get(self, keys: list[str], shapes=None, dtypes=None, custom_meta=None) -> list[Any]:
        """Retrieves multiple values from remote storage with expected metadata.

        Requires shape and dtype hints to reconstruct NPU tensors correctly.

        Args:
            keys (List[str]): Keys to fetch.
            shapes (List[List[int]]): Expected tensor shapes (use [] for scalars).
            dtypes (List[Optional[torch.dtype]]): Expected dtypes; use None for non-tensor data.

        Returns:
            List[Any]: Retrieved values in the same order as input keys.
        """
        if shapes is None or dtypes is None:
            raise ValueError("YuanrongStorageClient needs Expected shapes and dtypes")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")
        return self._batch_get(keys, shapes, dtypes)

    def _batch_clear(self, keys: list[str]):
        """Deletes a batch of keys from remote storage.

        Attempts deletion via NPU client first (if available), then falls back to CPU client
        for any keys not handled by NPU.

        Args:
            keys (List[str]): Keys to delete.
        """
        if self.npu_ds_client_is_available():
            # Try to delete all keys via npu client
            for i in range(0, len(keys), NPU_DS_CLIENT_KEYS_LIMIT):
                batch = keys[i : i + NPU_DS_CLIENT_KEYS_LIMIT]
                # Return the keys that failed to delete
                self._npu_ds_client.dev_delete(batch)
                # Delete failed keys via CPU client
            for j in range(0, len(keys), CPU_DS_CLIENT_KEYS_LIMIT):
                sub_batch = keys[j : j + CPU_DS_CLIENT_KEYS_LIMIT]
                self._cpu_ds_client.delete(sub_batch)
        else:
            for i in range(0, len(keys), CPU_DS_CLIENT_KEYS_LIMIT):
                batch = keys[i : i + CPU_DS_CLIENT_KEYS_LIMIT]
                self._cpu_ds_client.delete(batch)

    def clear(self, keys: list[str]):
        """Deletes multiple keys from remote storage.

        Args:
            keys (List[str]): List of keys to remove.
        """
        self._batch_clear(keys)
