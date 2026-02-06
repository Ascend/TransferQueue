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
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from transfer_queue.storage.clients.base import TransferQueueStorageKVClient
from transfer_queue.storage.clients.factory import StorageClientFactory
from transfer_queue.utils.serial_utils import _decoder, _encoder

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

YUANRONG_DATASYSTEM_IMPORTED: bool = True

try:
    from yr import datasystem
except ImportError:
    YUANRONG_DATASYSTEM_IMPORTED = False


class StorageStrategy(ABC):
    """Abstract base class for storage strategies."""

    @staticmethod
    @abstractmethod
    def init(config: dict) -> Optional["StorageStrategy"]:
        """Initialize strategy from config; return None if not applicable."""

    @abstractmethod
    def strategy_tag(self) -> Any:
        """Return metadata identifying this strategy (e.g., string name, byte tag)."""

    @abstractmethod
    def supports_put(self, value: Any) -> bool:
        """Check if this strategy can store the given value."""

    @abstractmethod
    def put(self, keys: list[str], values: list[Any]):
        """Store key-value pairs using this strategy."""

    @abstractmethod
    def supports_get(self, strategy_tag: Any) -> bool:
        """Check if this strategy can retrieve data with given tag."""

    @abstractmethod
    def get(self, keys: list[str], **kwargs) -> list[Optional[Any]]:
        """Retrieve values by keys; kwargs may include shapes/dtypes."""

    @abstractmethod
    def supports_clear(self, strategy_tag: Any) -> bool:
        """Check if this strategy owns data identified by metadata."""

    @abstractmethod
    def clear(self, keys: list[str]):
        """Delete keys from storage."""


class NPUTensorKVClientAdapter(StorageStrategy):
    """Adapter for YuanRong's high-performance NPU tensor storage.
    Using yr.datasystem.DsTensorClient to connect datasystem backends.
    """

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
    def init(config: dict) -> Optional["StorageStrategy"]:
        """Initialize only if NPU and torch_npu are available."""
        torch_npu_imported: bool = True
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            torch_npu_imported = False
        enable = config.get("enable_yr_npu_transport", True)
        if not (enable and torch_npu_imported and torch.npu.is_available()):
            return None

        return NPUTensorKVClientAdapter(config)

    def strategy_tag(self) -> str:
        """Strategy tag for NPU tensor storage. Using a single byte is for better performance."""
        return "1"

    def supports_put(self, value: Any) -> bool:
        """Supports contiguous NPU tensors only."""
        if not (isinstance(value, torch.Tensor) and value.device.type == "npu"):
            return False
        # Only contiguous NPU tensors are supported by this adapter.
        return value.is_contiguous()

    def put(self, keys: list[str], values: list[Any]):
        """Store NPU tensors in batches; deletes before overwrite."""
        for i in range(0, len(keys), self.KEYS_LIMIT):
            batch_keys = keys[i : i + self.KEYS_LIMIT]
            batch_values = values[i : i + self.KEYS_LIMIT]
            # _npu_ds_client.dev_mset doesn't support to overwrite
            try:
                self._ds_client.dev_delete(batch_keys)
            except Exception:
                pass
            self._ds_client.dev_mset(batch_keys, batch_values)

    def supports_get(self, strategy_tag: str) -> bool:
        """Matches 'DsTensorClient' Strategy tag."""
        return isinstance(strategy_tag, str) and strategy_tag == self.strategy_tag()

    def get(self, keys: list[str], **kwargs) -> list[Optional[Any]]:
        """Fetch NPU tensors using pre-allocated empty buffers."""
        shapes = kwargs.get("shapes", None)
        dtypes = kwargs.get("dtypes", None)
        if shapes is None or dtypes is None:
            raise ValueError("YuanrongStorageClient needs Expected shapes and dtypes")
        results = []
        for i in range(0, len(keys), self.KEYS_LIMIT):
            batch_keys = keys[i : i + self.KEYS_LIMIT]
            batch_shapes = shapes[i : i + self.KEYS_LIMIT]
            batch_dtypes = dtypes[i : i + self.KEYS_LIMIT]

            batch_values = self._create_empty_npu_tensorlist(batch_shapes, batch_dtypes)
            self._ds_client.dev_mget(batch_keys, batch_values)
            # Todo(dpj): consider checking and logging keys that fail during dev_mget
            results.extend(batch_values)
        return results

    def supports_clear(self, strategy_tag: str) -> bool:
        """Matches 'DsTensorClient' strategy tag."""
        return isinstance(strategy_tag, str) and strategy_tag == self.strategy_tag()

    def clear(self, keys: list[str]):
        """Delete NPU tensor keys in batches."""
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


class GeneralKVClientAdapter(StorageStrategy):
    """Adapter for general-purpose KV storage with serialization.
    Using yr.datasystem.KVClient to connect datasystem backends.
    The serialization method uses '_decoder' and '_encoder' from 'transfer_queue.utils.serial_utils'.
    """

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
    def init(config: dict) -> Optional["StorageStrategy"]:
        """Always enabled for general objects."""
        return GeneralKVClientAdapter(config)

    def strategy_tag(self) -> str:
        """Strategy tag for general KV storage. Using a single byte is for better performance."""
        return "2"

    def supports_put(self, value: Any) -> bool:
        """Accepts any Python object."""
        return True

    def put(self, keys: list[str], values: list[Any]):
        """Store objects via zero-copy serialization in batches."""
        for i in range(0, len(keys), self.PUT_KEYS_LIMIT):
            batch_keys = keys[i : i + self.PUT_KEYS_LIMIT]
            batch_vals = values[i : i + self.PUT_KEYS_LIMIT]
            self.mset_zero_copy(batch_keys, batch_vals)

    def supports_get(self, strategy_tag: str) -> bool:
        """Matches 'KVClient' strategy tag."""
        return isinstance(strategy_tag, str) and strategy_tag == self.strategy_tag()

    def get(self, keys: list[str], **kwargs) -> list[Optional[Any]]:
        """Retrieve and deserialize objects in batches."""
        results = []
        for i in range(0, len(keys), self.GET_CLEAR_KEYS_LIMIT):
            batch_keys = keys[i : i + self.GET_CLEAR_KEYS_LIMIT]
            objects = self.mget_zero_copy(batch_keys)
            results.extend(objects)
        return results

    def supports_clear(self, strategy_tag: str) -> bool:
        """Matches 'KVClient' strategy tag."""
        return isinstance(strategy_tag, str) and strategy_tag == self.strategy_tag()

    def clear(self, keys: list[str]):
        """Delete keys in batches."""
        for i in range(0, len(keys), self.GET_CLEAR_KEYS_LIMIT):
            batch_keys = keys[i : i + self.GET_CLEAR_KEYS_LIMIT]
            self._ds_client.delete(batch_keys)

    @classmethod
    def calc_packed_size(cls, items: list[memoryview]) -> int:
        """
        Calculate the total size (in bytes) required to pack a list of memoryview items
        into the structured binary format used by pack_into.

        Args:
            items: List of memoryview objects to be packed.

        Returns:
            Total buffer size in bytes.
        """
        return cls.HEADER_SIZE + len(items) * cls.ENTRY_SIZE + sum(item.nbytes for item in items)

    @classmethod
    def pack_into(cls, target: memoryview, items: list[memoryview]):
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
        struct.pack_into(cls.HEADER_FMT, target, 0, len(items))

        entry_offset = cls.HEADER_SIZE
        payload_offset = cls.HEADER_SIZE + len(items) * cls.ENTRY_SIZE

        target_tensor = torch.frombuffer(target, dtype=torch.uint8)

        for item in items:
            struct.pack_into(cls.ENTRY_FMT, target, entry_offset, payload_offset, item.nbytes)
            src_tensor = torch.frombuffer(item, dtype=torch.uint8)
            target_tensor[payload_offset : payload_offset + item.nbytes].copy_(src_tensor)
            entry_offset += cls.ENTRY_SIZE
            payload_offset += item.nbytes

    @classmethod
    def unpack_from(cls, source: memoryview) -> list[memoryview]:
        """
        Unpack multiple contiguous buffers from a single packed buffer.
        Args:
            source (memoryview): The packed source buffer.
        Returns:
            list[memoryview]: List of unpacked contiguous buffers.
        """
        mv = memoryview(source)
        item_count = struct.unpack_from(cls.HEADER_FMT, mv, 0)[0]
        offsets = []
        for i in range(item_count):
            offset, length = struct.unpack_from(cls.ENTRY_FMT, mv, cls.HEADER_SIZE + i * cls.ENTRY_SIZE)
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

    Use different storage strategies depending on the data type.
    Supports storing and fetching both:
    - NPU tensors via NPUTensorKVClientAdapter (for high performance).
    - General objects (CPU tensors, str, bool, list, etc.) via GeneralKVClientAdapter with serialization.
    """

    def __init__(self, config: dict[str, Any]):
        if not YUANRONG_DATASYSTEM_IMPORTED:
            raise ImportError("YuanRong DataSystem not installed.")

        super().__init__(config)

        # Storage strategies are prioritized in ascending order of list element index.
        # In other words, the later in the order, the lower the priority.
        storage_strategies_priority = [NPUTensorKVClientAdapter, GeneralKVClientAdapter]
        self._strategies: list[StorageStrategy] = []
        for strategy_cls in storage_strategies_priority:
            strategy = strategy_cls.init(config)
            if strategy is not None:
                self._strategies.append(strategy)

        if not self._strategies:
            raise RuntimeError("No storage strategy available for YuanrongStorageClient")

    def put(self, keys: list[str], values: list[Any]) -> list[str]:
        """Stores multiple key-value pairs to remote storage.

        Automatically routes NPU tensors to high-performance tensor storage,
        and other objects to general-purpose KV storage.

        Args:
            keys (List[str]): List of unique string identifiers.
            values (List[Any]): List of values to store (tensors, scalars, dicts, etc.).

        Returns:
            List[str]: storage strategy tag of YuanrongStorageClient in the same order as input keys.
        """
        if not isinstance(keys, list) or not isinstance(values, list):
            raise ValueError("keys and values must be lists")
        if len(keys) != len(values):
            raise ValueError("Number of keys must match number of values")

        routed_indexes = self._route_to_strategies(values, lambda strategy_, item_: strategy_.supports_put(item_))

        # Define the 'put_task': Slicing the input list and calling the backend strategy.
        # The closure captures local 'keys' and 'values' for zero-overhead parameter passing.
        def put_task(strategy, indexes):
            strategy.put([keys[i] for i in indexes], [values[i] for i in indexes])
            return strategy.strategy_tag(), indexes

        # Dispatch tasks and map strategy_tag back to original positions
        strategy_tags: list[str] = [""] * len(keys)
        for tag, indexes in self._dispatch_tasks(routed_indexes, put_task):
            for original_index in indexes:
                strategy_tags[original_index] = tag
        return strategy_tags

    def get(self, keys: list[str], shapes=None, dtypes=None, custom_backend_meta=None) -> list[Any]:
        """Retrieves multiple values from remote storage with expected metadata.

        Requires shape and dtype hints to reconstruct NPU tensors correctly.

        Args:
            keys (List[str]): Keys to fetch.
            shapes (List[List[int]]): Expected tensor shapes (use [] for scalars).
            dtypes (List[Optional[torch.dtype]]): Expected dtypes; use None for non-tensor data.
            custom_backend_meta (List[str]): StorageStrategy tag for each key

        Returns:
            List[Any]: Retrieved values in the same order as input keys.
        """
        if shapes is None or dtypes is None or custom_backend_meta is None:
            raise ValueError("YuanrongStorageClient.get() needs Expected shapes, dtypes and custom_backend_meta")

        if not (len(keys) == len(shapes) == len(dtypes) == len(custom_backend_meta)):
            raise ValueError("Lengths of keys, shapes, dtypes, custom_backend_meta must match")

        strategy_tags = custom_backend_meta
        routed_indexes = self._route_to_strategies(
            strategy_tags, lambda strategy_, item_: strategy_.supports_get(item_)
        )

        # Define the 'get_task': handles slicing of keys, shapes, and dtypes simultaneously.
        def get_task(strategy, indexes):
            res = strategy.get(
                [keys[i] for i in indexes], shapes=[shapes[i] for i in indexes], dtypes=[dtypes[i] for i in indexes]
            )
            return res, indexes

        # Gather results and restore original order
        results = [None] * len(keys)
        for strategy_res, indexes in self._dispatch_tasks(routed_indexes, get_task):
            for value, original_index in zip(strategy_res, indexes, strict=True):
                results[original_index] = value
        return results

    def clear(self, keys: list[str], custom_backend_meta=None):
        """Deletes multiple keys from remote storage.

        Args:
            keys (List[str]): List of keys to remove.
            custom_backend_meta (List[str]): StorageStrategy tag for each key
        """
        if not isinstance(keys, list) or not isinstance(custom_backend_meta, list):
            raise ValueError("keys and custom_backend_meta must be a list")

        if len(custom_backend_meta) != len(keys):
            raise ValueError("custom_backend_meta length must match keys")

        strategy_tags = custom_backend_meta
        routed_indexes = self._route_to_strategies(
            strategy_tags, lambda strategy_, item_: strategy_.supports_clear(item_)
        )

        def clear_task(strategy, indexes):
            strategy.clear([keys[i] for i in indexes])

        # Execute deletions (no return values needed)
        self._dispatch_tasks(routed_indexes, clear_task)

    def _route_to_strategies(
        self,
        items: list[Any],
        selector: Callable[[StorageStrategy, Any], bool],
    ) -> dict[StorageStrategy, list[int]]:
        """Groups item indices by the first strategy that supports them.

        Used to route data to storage strategies by grouped indexes.

        Args:
            items: A list used to distinguish which storage strategy the data is routed to.
                   e.g., route <keys, values> for put based on types of values,
                   or route <keys, Optional[shapes], Optional[dtypes]> for get/clear based on strategy_tags.
                   The order must correspond to the original keys.
            selector: A function that determines whether a strategy supports an item.
                     Signature: `(strategy: StorageStrategy, item: Any) -> bool`.

        Returns:
            A dictionary mapping each active strategy to a list of indexes in `items`
            that it should handle. Every index appears exactly once.
        """
        routed_indexes: dict[StorageStrategy, list[int]] = {s: [] for s in self._strategies}
        for i, item in enumerate(items):
            for strategy in self._strategies:
                if selector(strategy, item):
                    routed_indexes[strategy].append(i)
                    break
            else:
                raise ValueError(
                    f"No strategy supports item of type {type(item).__name__}: {item}. "
                    f"Available strategies: {[type(s).__name__ for s in self._strategies]}"
                )
        return routed_indexes

    @staticmethod
    def _dispatch_tasks(routed_tasks: dict[StorageStrategy, list[int]], task_function: Callable) -> list[Any]:
        """Executes tasks across one or more storage strategies, optionally in parallel.

        Optimizes for common case: if only one strategy is involved, runs synchronously
        to avoid thread overhead. Otherwise, uses thread pool for concurrency.

        Args:
            routed_tasks: Mapping from strategy to list of indexes it should process.
            task_function: Callable accepting `(strategy, list_of_indexes)` and returning any result.

        Returns:
            List of results from `task_function`, one per active strategy, in arbitrary order.
            Each result typically includes data and the corresponding indices for reassembly.
        """
        active_tasks = [(strategy, indexes) for strategy, indexes in routed_tasks.items() if indexes]

        if not active_tasks:
            return []

        # Fast path: single strategy → avoid threading
        if len(active_tasks) == 1:
            return [task_function(*active_tasks[0])]

        # Parallel path: overlap NPU and CPU operations
        # Cap the number of worker threads to avoid resource exhaustion if many
        # strategies are added in the future.
        max_workers = min(len(active_tasks), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # futures' results are from task_function
            futures = [executor.submit(task_function, strategy, indexes) for strategy, indexes in active_tasks]
            return [f.result() for f in futures]
