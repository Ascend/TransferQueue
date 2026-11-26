# Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The TransferQueue Team
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

import os
from uuid import uuid4

import ray
import torch
import zmq
from msgspec import msgpack
from omegaconf import DictConfig

from transfer_queue.controller import DataPartitionStatus, FieldMeta
from transfer_queue.metadata import BatchMeta
from transfer_queue.metrics import REQUEST_LATENCY_BUCKETS, RustControllerMetrics
from transfer_queue.sampler import BaseSampler
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.serial_utils import decode, parse_torch_dtype
from transfer_queue.utils.zmq_utils import ZMQServerInfo, get_node_ip_address

logger = get_logger(__name__)


def _bitmap_indexes(words: list[int], indexes: list[int]) -> list[int]:
    return [
        index for row, index in enumerate(indexes) if row // 64 < len(words) and words[row // 64] & (1 << (row % 64))
    ]


def _partition_from_snapshot(name: str, snapshot: dict) -> DataPartitionStatus:
    """Materialize a Python ``DataPartitionStatus`` view from a Rust partition snapshot."""
    indexes = snapshot["indexes"]
    partition = DataPartitionStatus(partition_id=name, created_at=snapshot["created_at"])
    if indexes:
        partition.ensure_samples_capacity(max(indexes) + 1)
    partition.ensure_fields_capacity(len(snapshot["fields"]))
    partition.global_indexes = set(_bitmap_indexes(snapshot["active"], indexes))
    partition.pre_allocated_global_indexes = set(snapshot["reserved"])
    partition.keys_mapping = snapshot["keys"]
    partition.revert_keys_mapping = snapshot["index_keys"]
    partition.custom_meta = {index: msgpack.decode(raw, strict=False) for index, raw in snapshot["custom_meta"].items()}
    partition.field_custom_backend_meta = {
        index: {field: msgpack.decode(raw, strict=False) for field, raw in fields.items()}
        for index, fields in snapshot["backend_meta"].items()
    }
    partition.links = snapshot["links"]
    partition.ever_linked = set(snapshot["ever_linked"])
    for index, fields in partition.links.items():
        for source in fields.values():
            partition.linked_by.setdefault(source, set()).add(index)
    for col, (name, field) in enumerate(snapshot["fields"].items()):
        partition.field_name_mapping[name] = col
        partition.production_status[_bitmap_indexes(field["ready"], indexes), col] = 1
        schema = field["schema"]
        if schema is not None:
            dtype = schema["dtype"]
            partition.field_metadata[name] = FieldMeta(
                global_indexes=set(_bitmap_indexes(field["present"], indexes)),
                dtype=parse_torch_dtype(dtype, name) if dtype is not None else None,
                shape=tuple(schema["shape"]) if schema["shape"] is not None else None,
                is_nested=schema["nested"],
                is_non_tensor=schema["non_tensor"],
                per_sample_shapes={
                    index: tuple(shape) if shape is not None else None
                    for index, shape in schema["sample_shapes"].items()
                },
            )
    for task, bitmap in snapshot["consumed"].items():
        consumed = torch.zeros(partition.allocated_samples_num, dtype=torch.int8)
        consumed[_bitmap_indexes(bitmap, indexes)] = 1
        partition.consumption_status[task] = consumed
    return partition


@ray.remote(num_cpus=1)
class RustControllerActor:
    """Expose discovery and management methods for the in-process Rust Controller."""

    def __init__(self, sampler: BaseSampler | type[BaseSampler] | str | None = None) -> None:
        if sampler is not None:
            logger.warning(
                f"Rust Controller does not support custom sampler configuration (got {sampler!r}); "
                "using the built-in default FIFO sampling instead."
            )
        try:
            from transfer_queue._controller_engine import ControllerEngine
        except ImportError as error:
            raise RuntimeError("The Rust Controller requires a build with the _controller_engine extension") from error

        controller_id = f"TQ_CONTROLLER_{uuid4().hex[:8]}"
        ip = get_node_ip_address()
        preallocate = int(os.environ.get("TQ_PRE_ALLOC_SAMPLE_NUM", 1))
        self._engine = ControllerEngine(ip, controller_id, preallocate=preallocate)
        handshake_port, request_port = self._engine.ports()
        self._server_info = ZMQServerInfo(
            role=Role.CONTROLLER,
            id=controller_id,
            ip=ip,
            ports={"handshake_socket": handshake_port, "request_handle_socket": request_port},
        )
        self._config: DictConfig | None = None
        self._metrics: RustControllerMetrics | None = None

    def get_zmq_server_info(self) -> ZMQServerInfo:
        """Return the listening endpoints."""
        return self._server_info

    def store_config(self, conf: DictConfig) -> None:
        """Publish the config for attaching clients."""
        self._config = conf
        if conf.backend.storage_backend == "SimpleStorage":
            self._engine.set_num_storage_units(int(conf.backend.SimpleStorage.num_data_storage_units))

    def get_config(self) -> DictConfig | None:
        """Return the config once the owner has published it."""
        return self._config

    def list_partitions(self) -> list[str]:
        """List all available partition IDs.

        Returns:
            List of partition IDs.
        """
        return self._engine.list_partitions()

    def get_partition_snapshot(self, partition_id: str) -> DataPartitionStatus | None:
        """Get a copy of partition status information.

        Args:
            partition_id: ID of the partition to retrieve.

        Returns:
            DataPartitionStatus object if the partition exists, None otherwise.
        """
        snapshot = self._engine.partition_snapshot(partition_id)
        if snapshot is None:
            return None
        return _partition_from_snapshot(partition_id, decode([snapshot]))

    def get_partition_index_range(self, partition_id: str) -> list[int]:
        """Get all global indexes allocated to a specific partition (including pre-allocated slots).

        Args:
            partition_id: Partition identifier.

        Returns:
            List of global indexes allocated to the partition.
        """
        return self._engine.partition_index_range(partition_id)

    def mark_clearing(self, global_indexes: list[int], partition_ids: list[str]) -> None:
        """Mark samples as pending deletion by zeroing their production status.

        This prevents consumers from reading data that is about to be deleted from
        storage, closing the window between storage clear and controller clear.

        Args:
            global_indexes: Global indexes to mark as pending deletion.
            partition_ids: Corresponding partition IDs for each global index.
        """
        self._engine.mark_clearing(global_indexes, partition_ids)

    def clear_meta(self, global_indexes: list[int], partition_ids: list[str]) -> None:
        """Clear metadata for individual samples while preserving the partition.

        Args:
            global_indexes: Global indexes to clear.
            partition_ids: Corresponding partition IDs for each global index.
        """
        self._engine.clear_meta(global_indexes, partition_ids)

    def clear_partition(self, partition_id: str) -> None:
        """Clear metadata for a specific partition and delete the partition.

        Args:
            partition_id: ID of the partition to clear.
        """
        self._engine.clear_partition(partition_id)

    def mark_unlinked_public_clearing(self, stale_keys: list[str] | None = None) -> tuple[BatchMeta, list[str]]:
        """Mark every ``PUBLIC_PARTITION`` key that keys linked to before but none links to now as clearing.

        Args:
            stale_keys: Public keys the caller gave up waiting on; each one no key links to now
                is marked too, even if none ever linked it. Missing keys are ignored.

        Returns:
            The marked rows' BatchMeta (for storage deletion and ``clear_meta``) and the keys of those rows.
        """
        metadata, keys = self._engine.mark_unlinked_public_clearing(stale_keys)
        return decode([metadata]), keys

    def save_checkpoint(self, path: str) -> None:
        """Serialize controller state directly to a native binary checkpoint file.

        Args:
            path: Absolute path for the output ``.msgpack`` checkpoint file.
        """
        self._engine.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Restore controller state directly from a native binary checkpoint file.

        Args:
            path: Absolute path to a checkpoint file previously written by ``save_checkpoint()``.
        """
        self._engine.load_checkpoint(path)

    def start_metrics(self, port: int = 0) -> str:
        """Initialize and start the Prometheus metrics exporter.

        Safe to call multiple times; subsequent calls return the existing endpoint.

        Args:
            port: HTTP port for the ``/metrics`` endpoint (0 = auto-assign).

        Returns:
            The metrics endpoint address in ``host:port`` format.
        """
        if self._metrics is None:
            self._engine.enable_metrics(REQUEST_LATENCY_BUCKETS)
            self._metrics = RustControllerMetrics(
                lambda: msgpack.decode(self._engine.metrics_snapshot()), zmq.Context.instance()
            )
            self._metrics.start(node_ip=self._server_info.ip, port=port)
        return self._metrics.endpoint

    def register_storage_units_for_metrics(self, storage_unit_infos: dict) -> None:
        """Register storage unit ZMQ endpoints so the metrics collector can query them.

        Args:
            storage_unit_infos: Mapping of storage unit names/IDs to ``ZMQServerInfo``.
        """
        if self._metrics is not None:
            self._metrics.register_storage_units(storage_unit_infos)

    def shutdown(self) -> None:
        """Stop the worker and release its sockets."""
        self._engine.close()
