import itertools
from typing import Any

import ray
import torch

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory


@StorageClientFactory.register("RayStorageClient")
class RayStorageClient(TransferQueueStorageKVClient):
    def __init__(self, config=None):
        if not ray.is_initialized():
            raise RuntimeError(
                "Ray is not initialized. Please call ray.init() before creating RayStorageClient."
            )

        # default to 'nixl' transport
        self.tensor_transport = (
            config.get("tensor_transport", "nixl") if config else "nixl"
        )
        # store object refs to avoid early garbage collection
        self.object_ref_storage = {}

    def put(self, keys: list[str], values: list[Any]) -> dict[str, ray.ObjectRef]:
        """
        Store tensors to remote storage.
        Args:
            keys (list): List of string keys
            values (list): List of torch.Tensor on GPU(CUDA) or CPU
        Returns:
            dict: A dictionary mapping keys to their corresponding Ray ObjectRefs
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError(
                f"keys and values must be lists, but got {type(keys)} and {type(values)}"
            )
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        def ray_put_with_transport(v):
            return (
                ray.put(v, _tensor_transport=self.tensor_transport)
                if isinstance(v, torch.Tensor)
                else ray.put(v)
            )

        custom_meta = {k: ray_put_with_transport(v) for k, v in zip(keys, values)}
        self.object_ref_storage.update(custom_meta)
        return custom_meta

    def get(
        self, keys: list[str], shapes=None, dtypes=None, custom_meta=None
    ) -> list[Any]:
        """
        Retrieve objects from remote storage.
        Args:
            keys (list): List of string keys to fetch.
            shapes (list, optional): Ignored. For compatibility with KVStorageManager.
            dtypes (list, optional): Ignored. For compatibility with KVStorageManager.
            custom_meta (dict, optional): A dictionary mapping keys to Ray ObjectRefs.
        Returns:
            list: List of retrieved objects
        """

        if not isinstance(keys, list):
            raise ValueError(f"keys must be a list, but got {type(keys)}")

        if custom_meta is None:
            raise ValueError("custom_meta must be provided")

        obj_refs = custom_meta.values()
        try:
            values = ray.get(obj_refs)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve value for key '{keys}': {e}") from e
        return values

    def clear(self, keys: list[str]):
        """
        Delete entries from storage by keys.
        Args:
            keys (list): List of keys to delete
        """
        self.object_ref_storage = {
            k: v for k, v in self.object_ref_storage.items() if k not in keys
        }
