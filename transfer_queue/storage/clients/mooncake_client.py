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
from typing import Any, Optional

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory
from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)

MOONCAKE_STORE_IMPORTED: bool = True
try:
    from mooncake.store import MooncakeDistributedStore
except ImportError:
    MOONCAKE_STORE_IMPORTED = False

BATCH_SIZE_LIMIT: int = 500


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

        tensor_keys = []
        tensor_values = []
        non_tensor_keys = []
        non_tensor_values = []

        for key, value in zip(keys, values, strict=True):
            if isinstance(value, torch.Tensor):
                tensor = value.contiguous()
                # TODO: use gpu direct rdma instead
                if tensor.device.type == "cuda":
                    tensor = tensor.cpu()
                tensor_keys.append(key)
                tensor_values.append(tensor)
            else:
                non_tensor_keys.append(key)
                non_tensor_values.append(pickle.dumps(value))

        if tensor_keys:
            self._batch_put_tensors(tensor_keys, tensor_values)

        if non_tensor_keys:
            self._batch_put_bytes(non_tensor_keys, non_tensor_values)

        return None

    def _batch_put_tensors(self, keys: list[str], tensors: list[Tensor]):
        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i : i + BATCH_SIZE_LIMIT]
            batch_tensors = tensors[i : i + BATCH_SIZE_LIMIT]

            results = self._store.batch_put_tensor(batch_keys, batch_tensors)
            if not all(r == 0 for r in results):
                failed_indices = [j for j, r in enumerate(results) if r != 0]
                error_codes = [results[j] for j in failed_indices]
                raise RuntimeError(
                    f"batch_put_tensor failed for indices {failed_indices} with error codes: {error_codes}"
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

        tensor_indices = []
        non_tensor_indices = []

        for i, dtype in enumerate(dtypes):
            if dtype is not None:
                tensor_indices.append(i)
            else:
                non_tensor_indices.append(i)

        results = [None] * len(keys)

        if tensor_indices:
            tensor_keys = [keys[i] for i in tensor_indices]
            tensor_shapes = [shapes[i] for i in tensor_indices]
            tensor_dtypes = [dtypes[i] for i in tensor_indices]
            tensor_results = self._batch_get_tensors(tensor_keys, tensor_shapes, tensor_dtypes)
            # TODO: optimize these for loops
            for idx, tensor in zip(tensor_indices, tensor_results, strict=True):
                results[idx] = tensor

        if non_tensor_indices:
            non_tensor_keys = [keys[i] for i in non_tensor_indices]
            non_tensor_results = self._batch_get_bytes(non_tensor_keys)
            for idx, data in zip(non_tensor_indices, non_tensor_results, strict=True):
                results[idx] = pickle.loads(data)

        return results

    def _batch_get_tensors(self, keys: list[str], shapes: list, dtypes: list) -> list[Tensor]:
        tensors = [None] * len(keys)

        for i in range(0, len(keys), BATCH_SIZE_LIMIT):
            batch_keys = keys[i : i + BATCH_SIZE_LIMIT]
            batch_shapes = shapes[i : i + BATCH_SIZE_LIMIT]
            batch_dtypes = dtypes[i : i + BATCH_SIZE_LIMIT]

            batch_results = self._store.batch_get_tensor(batch_keys)

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
            self._store.close()
            self._store = None
