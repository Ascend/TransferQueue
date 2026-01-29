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

import itertools
from typing import Any, Optional

import ray
import torch

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory


@ray.remote(max_concurrency=8)
class RayObjectRefStorage:
    """Ray object ref storage."""

    def __init__(self):
        self.storage_dict = {}

    def put_obj_ref(self, keys: list[str], obj_refs: list[ray.ObjectRef]):
        """Put object ref to remote storage."""
        self.storage_dict.update(itertools.starmap(lambda k, v: (k, v), zip(keys, obj_refs, strict=True)))

    def get_obj_ref(self, keys: list[str]) -> list[ray.ObjectRef]:
        """Get object ref from remote storage."""
        obj_refs = [self.storage_dict.get(key, None) for key in keys]
        return obj_refs

    def clear_obj_ref(self, keys: list[str]):
        """Clear object ref from remote storage."""
        for key in keys:
            if key in self.storage_dict:
                del self.storage_dict[key]


@StorageClientFactory.register("RayStorageClient")
class RayStorageClient(TransferQueueStorageKVClient):
    """
    Storage client for Ray RDT.
    """

    def __init__(self, config=None):
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Please call ray.init() before creating RayStorageClient.")

        # initialize actor
        try:
            self.storage_actor = ray.get_actor("RayObjectRefStorage")
        except ValueError:
            self.storage_actor = RayObjectRefStorage.options(name="RayObjectRefStorage", get_if_exists=False).remote()

    def put(self, keys: list[str], values: list[Any]) -> Optional[list[Any]]:
        """
        Store tensors to remote storage.
        Args:
            keys (list): List of string keys
            values (list): List of torch.Tensor on GPU(CUDA) or CPU
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError(f"keys and values must be lists, but got {type(keys)} and {type(values)}")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        transports = itertools.repeat("nixl")
        obj_refs = list(
            itertools.starmap(
                lambda v, tx: ray.put(v, _tensor_transport=tx) if isinstance(v, torch.Tensor) else ray.put(v),
                zip(values, transports, strict=False),
            )
        )
        ray.get(self.storage_actor.put_obj_ref.remote(keys, obj_refs))
        return None

    def get(self, keys: list[str], shapes=None, dtypes=None, custom_backend_meta=None) -> list[Any]:
        """
        Retrieve objects from remote storage.
        Args:
            keys (list): List of string keys to fetch.
            shapes (list, optional): Ignored. For compatibility with KVStorageManager.
            dtypes (list, optional): Ignored. For compatibility with KVStorageManager.
            custom_backend_meta (list, optional): Ray object ref for each key
        Returns:
            list: List of retrieved objects
        """

        if not isinstance(keys, list):
            raise ValueError(f"keys must be a list, but got {type(keys)}")

        obj_refs = ray.get(self.storage_actor.get_obj_ref.remote(keys))
        try:
            values = ray.get(obj_refs)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve value for key '{keys}': {e}") from e
        return values

    def clear(self, keys: list[str], custom_meta=None):
        """
        Delete entries from storage by keys.
        Args:
            keys (list): List of keys to delete
            custom_meta (List[Any], optional): ...
        """
        ray.get(self.storage_actor.clear_obj_ref.remote(keys))
