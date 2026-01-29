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
import struct
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeAlias, Union

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory
from transfer_queue.utils.serial_utils import _decoder, _encoder

bytestr: TypeAlias = bytes | bytearray | memoryview

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

YUANRONG_DATASYSTEM_IMPORTED: bool = True

try:
    from yr import datasystem
except ImportError:
    YUANRONG_DATASYSTEM_IMPORTED = False


class StorageStrategy(ABC):
    @abstractmethod
    @staticmethod
    def init(config: dict) -> Union["StorageStrategy", None]: ...

    @abstractmethod
    def custom_meta(self) -> Any: ...

    @abstractmethod
    def supports_put(self, value: Any) -> bool: ...

    @abstractmethod
    def put(self, keys: list[str], values: list[Any]): ...

    @abstractmethod
    def supports_get(self, custom_meta: Any) -> bool: ...

    @abstractmethod
    def get(self, keys: list[str], **kwargs) -> list[Optional[Any]]: ...

    @abstractmethod
    def supports_clear(self, custom_meta: Any) -> bool: ...

    @abstractmethod
    def clear(self, keys: list[str]): ...


class DsTensorClientAdapter(StorageStrategy):
    KEYS_LIMIT: int = 10_000

    def __init__(self, config: dict):
        host = config.get("host")
        port = config.get("port")

        self.device_id = torch.npu.current_device()
        torch.npu.set_device(self.device_id)

        self._ds_client = datasystem.DsTensorClient(host, port, self.device_id)
        self._ds_client.init()
        logger.info("YuanrongStorageClient: Create DsTensorClient to connect with yuanrong-datasystem backend!")

    @staticmethod
    def init(config: dict) -> Union["StorageStrategy", None]:
        torch_npu_imported: bool = True
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            torch_npu_imported = False
        enable = config.get("enable_yr_npu_optimization", True)
        if not (enable and torch_npu_imported and torch.npu.is_available()):
            return None

        return DsTensorClientAdapter(config)

    def custom_meta(self) -> Any:
        return "DsTensorClient"

    def supports_put(self, value: Any) -> bool:
        if not (isinstance(value, torch.Tensor) and value.device.type == "npu"):
            return False
        # Todo(dpj): perhaps KVClient can process uncontiguous tensor
        if not value.is_contiguous():
            raise ValueError(f"NPU Tensor is not contiguous: {value}")
        return True

    def put(self, keys: list[str], values: list[Any]):
        # _npu_ds_client.dev_mset doesn't support to overwrite
        for i in range(0, len(keys), self.KEYS_LIMIT):
            batch_keys = keys[i : i + self.KEYS_LIMIT]
            batch_values = values[i : i + self.KEYS_LIMIT]
            try:
                self._ds_client.dev_delete(batch_keys)
            except Exception:
                pass
            self._ds_client.dev_mset(batch_keys, batch_values)

    def supports_get(self, custom_meta: str) -> bool:
        return isinstance(custom_meta, str) and custom_meta == self.custom_meta()

    def get(self, keys: list[str], **kwargs) -> list[Optional[Any]]:
        # Fetch NPU tensors
        shapes = kwargs.get("shapes", None)
        dtypes = kwargs.get("dtypes", None)
        if not shapes or not dtypes:
            raise ValueError("YuanrongStorageClient needs Expected shapes and dtypes")
        results = []
        for i in range(0, len(keys), self.KEYS_LIMIT):
            batch_keys = keys[i : i + self.KEYS_LIMIT]
            batch_shapes = shapes[i : i + self.KEYS_LIMIT]
            batch_dtypes = dtypes[i : i + self.KEYS_LIMIT]

            batch_values = self._create_empty_npu_tensorlist(batch_shapes, batch_dtypes)
            self._ds_client.dev_mget(batch_keys, batch_values)
            # Todo(dpj): should we check failed keys?
            # failed_keys = self._ds_client.dev_mget(batch_keys, batch_values)
            # if failed_keys:
            #     logging.warning(f"YuanrongStorageClient: Querying keys using 'DsTensorClient' failed: {failed_keys}")
            results.extend(batch_values)
        return results

    def supports_clear(self, custom_meta: str) -> bool:
        return isinstance(custom_meta, str) and custom_meta == self.custom_meta()

    def clear(self, keys: list[str]):
        for i in range(0, len(keys), self.KEYS_LIMIT):
            batch = keys[i : i + self.KEYS_LIMIT]
            # Todo(dpj): Test call clear when no (key,value) put in ds
            self._ds_client.dev_delete(batch)

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
        for shape, dtype in zip(shapes, dtypes, strict=True):
            tensor = torch.empty(shape, dtype=dtype, device=f"npu:{self.device_id}")
            tensors.append(tensor)
        return tensors


class KVClientAdapter(StorageStrategy):
    PUT_KEYS_LIMIT: int = 2_000
    GET_CLEAR_KEYS_LIMIT: int = 10_000

    # Header: number of entries (uint32, little-endian)
    HEADER_FMT = "<I"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    # Entry: (payload_offset: uint32, payload_size: uint32)
    ENTRY_FMT = "<II"
    ENTRY_SIZE = struct.calcsize(ENTRY_FMT)

    DS_MAX_WORKERS: int = 16

    def __init__(self, config: dict):
        host = config.get("host")
        port = config.get("port")

        self._ds_client = datasystem.KVClient(host, port)
        self._ds_client.init()
        logger.info("YuanrongStorageClient: Create KVClient to connect with yuanrong-datasystem backend!")

    @staticmethod
    def init(config: dict) -> Union["StorageStrategy", None]:
        return KVClientAdapter(config)

    def custom_meta(self) -> Any:
        return "KVClient"

    def supports_put(self, value: Any) -> bool:
        return True

    def put(self, keys: list[str], values: list[Any]):
        for i in range(0, len(keys), self.PUT_KEYS_LIMIT):
            batch_keys = keys[i : i + self.PUT_KEYS_LIMIT]
            batch_vals = values[i : i + self.PUT_KEYS_LIMIT]
            self.mset_zero_copy(batch_keys, batch_vals)

    def supports_get(self, custom_meta: str) -> bool:
        return isinstance(custom_meta, str) and custom_meta == self.custom_meta()

    def get(self, keys: list[str], **kwargs) -> list[Optional[Any]]:
        results = []
        for i in range(0, len(keys), self.GET_CLEAR_KEYS_LIMIT):
            batch_keys = keys[i : i + self.GET_CLEAR_KEYS_LIMIT]
            objects = self.mget_zero_copy(batch_keys)
            results.extend(objects)
        return results

    def supports_clear(self, custom_meta: str) -> bool:
        return isinstance(custom_meta, str) and custom_meta == self.custom_meta()

    def clear(self, keys: list[str]):
        for i in range(0, len(keys), self.GET_CLEAR_KEYS_LIMIT):
            batch = keys[i : i + self.GET_CLEAR_KEYS_LIMIT]
            self._ds_client.delete(batch)

    @staticmethod
    def calc_packed_size(items: list[memoryview]) -> int:
        """
        Calculate the total size (in bytes) required to pack a list of memoryview items
        into the structured binary format used by pack_into.

        Args:
            items: List of memoryview objects to be packed.

        Returns:
            Total buffer size in bytes.
        """
        return (
            KVClientAdapter.HEADER_SIZE + len(items) * KVClientAdapter.ENTRY_SIZE + sum(item.nbytes for item in items)
        )

    @staticmethod
    def pack_into(target: memoryview, items: list[memoryview]):
        """
        Pack multiple contiguous buffers into a single buffer.
            ┌───────────────┐
            │ item_count    │  uint32
            ├───────────────┤
            │ entries       │  N * item entries
            ├───────────────┤
            │ payload blob  │  N * concatenated buffers
            └───────────────┘

        Args:
            target (memoryview): A writable memoryview returned by StateValueBuffer.MutableData().
                It must be large enough to accommodate the total number of bytes of HEADER + ENTRY_TABLE + all items.
                This buffer is usually mapped to shared memory or Zero-Copy memory area.
            items (List[memoryview]): List of read-only memory views (e.g., from serialized objects).
                Each item must support the buffer protocol and be readable as raw bytes.

        """
        struct.pack_into(KVClientAdapter.HEADER_FMT, target, 0, len(items))

        entry_offset = KVClientAdapter.HEADER_SIZE
        payload_offset = KVClientAdapter.HEADER_SIZE + len(items) * KVClientAdapter.ENTRY_SIZE

        target_tensor = torch.frombuffer(target, dtype=torch.uint8)

        for item in items:
            struct.pack_into(KVClientAdapter.ENTRY_FMT, target, entry_offset, payload_offset, item.nbytes)
            src_tensor = torch.frombuffer(item, dtype=torch.uint8)
            target_tensor[payload_offset : payload_offset + item.nbytes].copy_(src_tensor)
            entry_offset += KVClientAdapter.ENTRY_SIZE
            payload_offset += item.nbytes

    @staticmethod
    def unpack_from(source: memoryview) -> list[memoryview]:
        """
        Unpack multiple contiguous buffers from a single packed buffer.
        Args:
            source (memoryview): The packed source buffer.
        Returns:
            list[memoryview]: List of unpacked contiguous buffers.
        """
        mv = memoryview(source)
        item_count = struct.unpack_from(KVClientAdapter.HEADER_FMT, mv, 0)[0]
        offsets = []
        for i in range(item_count):
            offset, length = struct.unpack_from(
                KVClientAdapter.ENTRY_FMT, mv, KVClientAdapter.HEADER_SIZE + i * KVClientAdapter.ENTRY_SIZE
            )
            offsets.append((offset, length))
        return [mv[offset : offset + length] for offset, length in offsets]

    def mset_zero_copy(self, keys: list[str], objs: list[Any]):
        """Store multiple objects in zero-copy mode using parallel serialization and buffer packing.

        Args:
            keys (list[str]): List of string keys under which the objects will be stored.
            objs (list[Any]): List of Python objects to store (e.g., tensors, strings).
        """
        items_list = [[memoryview(b) for b in _encoder.encode(obj)] for obj in objs]
        packed_sizes = [self.calc_packed_size(items) for items in items_list]
        buffers = self._ds_client.mcreate(keys, packed_sizes)
        tasks = [(target.MutableData(), item) for target, item in zip(buffers, items_list, strict=True)]
        with ThreadPoolExecutor(max_workers=self.DS_MAX_WORKERS) as executor:
            list(executor.map(lambda p: self.pack_into(*p), tasks))
        self._ds_client.mset_buffer(buffers)

    def mget_zero_copy(self, keys: list[str]) -> list[Any]:
        """Retrieve multiple objects in zero-copy mode by directly deserializing from shared memory buffers.

        Args:
            keys (list[str]): List of string keys to retrieve from storage.

        Returns:
            list[Any]: List of deserialized objects corresponding to the input keys.
        """
        buffers = self._ds_client.get_buffers(keys)
        return [_decoder.decode(self.unpack_from(buffer)) if buffer is not None else None for buffer in buffers]


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

        self._strategies: list[StorageStrategy] = []
        for strategy_cls in [DsTensorClientAdapter, KVClientAdapter]:
            strategy = strategy_cls.init(config)
            if strategy is not None:
                self._strategies.append(strategy)

        if not self._strategies:
            raise RuntimeError("No storage strategy available for YuanrongStorageClient")

    def put(self, keys: list[str], values: list[Any]) -> Optional[list[Any]]:
        """Stores multiple key-value pairs to remote storage.

        Automatically routes NPU tensors to high-performance tensor storage,
        and other objects to general-purpose KV storage.

        Args:
            keys (List[str]): List of unique string identifiers.
            values (List[Any]): List of values to store (tensors, scalars, dicts, etc.).

        Returns:
            List[Any]: custom metadata of YuanrongStorageCilent in the same order as input keys.
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        routed_indexes = self._route_to_strategies(values, lambda strategy_, item_: strategy_.supports_put(item_))
        custom_metas = [None] * len(keys)
        for strategy, indexes in routed_indexes.items():
            if not indexes:
                continue
            strategy_keys = [keys[i] for i in indexes]
            strategy_values = [values[i] for i in indexes]
            strategy.put(strategy_keys, strategy_values)
            for i in indexes:
                custom_metas[i] = strategy.custom_meta()

        return custom_metas

    def get(self, keys: list[str], shapes=None, dtypes=None, custom_meta=None) -> list[Any]:
        """Retrieves multiple values from remote storage with expected metadata.

        Requires shape and dtype hints to reconstruct NPU tensors correctly.

        Args:
            keys (List[str]): Keys to fetch.
            shapes (List[List[int]]): Expected tensor shapes (use [] for scalars).
            dtypes (List[Optional[torch.dtype]]): Expected dtypes; use None for non-tensor data.
            custom_meta (List[str], optional): Device type (npu/cpu) for each key

        Returns:
            List[Any]: Retrieved values in the same order as input keys.
        """
        if shapes is None or dtypes is None:
            raise ValueError("YuanrongStorageClient needs Expected shapes and dtypes")
        if not (len(keys) == len(shapes) == len(dtypes)):
            raise ValueError("Lengths of keys, shapes, dtypes must match")

        if custom_meta is None:
            raise ValueError("custom_meta is required for YuanrongStorageClient.get()")

        if len(custom_meta) != len(keys):
            raise ValueError("custom_meta length must match keys")

        routed_indexes = self._route_to_strategies(custom_meta, lambda strategy_, item_: strategy_.supports_get(item_))

        # Todo(dpj): Parallel get
        results = [None] * len(keys)
        for strategy, indexes in routed_indexes.items():
            if not indexes:
                continue
            strategy_keys = [keys[i] for i in indexes]
            strategy_shapes = [shapes[i] for i in indexes]
            strategy_dtypes = [dtypes[i] for i in indexes]
            strategy_results = strategy.get(strategy_keys, shapes=strategy_shapes, dtypes=strategy_dtypes)
            for j, i in enumerate(indexes):
                results[i] = strategy_results[j]

        return results

    def clear(self, keys: list[str]):
        """Deletes multiple keys from remote storage.

        Args:
            keys (List[str]): List of keys to remove.
        """
        pass

    def _route_to_strategies(
        self,
        items: list[Any],
        selector: Callable[[StorageStrategy, Any], bool],
    ) -> dict[StorageStrategy, list[int]]:
        """
        Groups item indices by storage strategy.

        Args:
            items: A list of items (e.g., values or custom_meta strings) to be dispatched.
                   The order must correspond to the original keys.
            selector: A function that determines whether a strategy supports an item.
                      Signature: (strategy, item) -> bool

        Returns:
            A dictionary mapping each strategy to a list of indices in `items`.
        """
        routed_indexes: dict[StorageStrategy, list[int]] = {s: [] for s in self._strategies}
        for i, item in enumerate(items):
            for strategy in self._strategies:
                if selector(strategy, item):
                    routed_indexes[strategy].append(i)
                    break
            else:
                raise ValueError(f"No strategy supports item: {item}")
        return routed_indexes
