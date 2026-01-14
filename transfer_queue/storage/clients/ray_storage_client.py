from typing import Any

import ray
import torch

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory


@StorageClientFactory.register("RayStorageClient")
class RayStorageClient(TransferQueueStorageKVClient):
    def __init__(self, config=None):
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Please call ray.init() before creating RayStorageClient.")

        # default to object store transport
        self.tensor_transport = config.get("tensor_transport", "object_store") if config else "object_store"

        # TODO(tianyi): this implementation is temporary. I have to keep a local copy of
        # refs here to avoid early recycle of real objects. But the refs are not
        # guaranteed to be deleted from the same client.
        self.object_storage = {}

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
            raise ValueError(f"keys and values must be lists, but got {type(keys)} and {type(values)}")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        def ray_put_with_transport(v):
            return ray.put(v, _tensor_transport=self.tensor_transport) if isinstance(v, torch.Tensor) else ray.put(v)

        custom_meta = {k: ray_put_with_transport(v) for k, v in zip(keys, values, strict=False)}
        self.object_storage.update(custom_meta)
        return custom_meta

    def get(self, keys: list[str], shapes=None, dtypes=None, custom_meta=None) -> list[Any]:
        """
        Retrieve objects from remote storage.
        Args:
            keys (list): List of string keys to fetch.
            shapes (list, optional): Ignored. For compatibility with KVStorageManager.
            dtypes (list, optional): Ignored. For compatibility with KVStorageManager.
            custom_meta (list, optional): A list of Ray ObjectRefs.
        Returns:
            list: List of retrieved objects
        """

        if not isinstance(keys, list):
            raise ValueError(f"keys must be a list, but got {type(keys)}")

        if custom_meta is None:
            raise ValueError("custom_meta must be provided")

        try:
            print(f"len(custom_meta): {len(custom_meta)}")
            values = ray.get(custom_meta)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve value for key '{keys}': {e}") from e
        return values

    def clear(self, keys: list[str]):
        """
        Delete entries from storage by keys. Notice that the object refs are stored in
        controller DataPartitionStatus.field_custom_metas, so here's nothing to do.
        The object refs will be garbage collected once the controller receives clear meta
        request.
        Args:
            keys (list): List of keys to delete
        """

        for key in keys:
            if key in self.object_storage:
                del self.object_storage[key]
