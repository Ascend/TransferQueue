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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from importlib import resources
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack

from transfer_queue import data_dump
from transfer_queue.client import TransferQueueClient
from transfer_queue.controller import PUBLIC_PARTITION, TransferQueueController
from transfer_queue.kv_lock import _close_lock_managers, _lock_settings, _start_lock_managers
from transfer_queue.metadata import KVBatchMeta
from transfer_queue.sampler import *  # noqa: F401
from transfer_queue.sampler import BaseSampler
from transfer_queue.storage.bootstrap import StorageBootstrapProvider
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.yuanrong_utils import cleanup_yuanrong_resources
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = get_logger(__name__)

_TQ_CLIENT: Any = None
_TQ_STORAGE: Any = None
_TQ_CONTROLLER: Any = None
# True only in the process whose init() created the controller. Other processes
# attach to the same named actor, so their close() must not tear it down.
_TQ_IS_OWNER = False

# Idle workers and proxies observe shutdown within one second; leave time for
# Ray dispatch, in-flight work, and SSD cleanup before forcing termination.
_SIMPLE_STORAGE_SHUTDOWN_TIMEOUT_S = 15


def _maybe_create_tq_client(conf: DictConfig | None = None) -> TransferQueueClient:
    global _TQ_CLIENT
    if _TQ_CLIENT is None:
        if conf is None:
            _init_from_existing()
            assert _TQ_CLIENT is not None, (
                "TransferQueueController has not been initialized yet. Please call init() first."
            )
            return _TQ_CLIENT

        pid = os.getpid()
        _TQ_CLIENT = TransferQueueClient(
            client_id=f"TransferQueueClient_{pid}", controller_info=conf.controller.zmq_info
        )

        backend_name = conf.backend.storage_backend

        _TQ_CLIENT.initialize_storage_manager(manager_type=backend_name, config=conf.backend[backend_name])

    return _TQ_CLIENT


def _maybe_create_tq_storage(conf: DictConfig) -> DictConfig:
    global _TQ_STORAGE

    if _TQ_STORAGE is None:
        _TQ_STORAGE = {}
        backend_name = conf.backend.storage_backend
        provider_fn = StorageBootstrapProvider.get_provider(backend_name)
        if provider_fn is not None:
            backend_resources = provider_fn(conf)
            if backend_resources is not None:
                _TQ_STORAGE[backend_name] = backend_resources
            else:
                logger.error(f"Not found available {backend_name} storage resources, please check the config.")
        else:
            logger.error(
                f"Storage backend {backend_name} not registered. Please add it to the StorageBootstrapProvider."
            )
    return conf


def _init_from_existing() -> bool:
    """Initialize the TransferQueueClient from existing controller.

    Returns:
        True if successfully initialized from existing controller, False otherwise.
    """
    global _TQ_CONTROLLER
    try:
        if _TQ_CONTROLLER is None:
            _TQ_CONTROLLER = ray.get_actor("TransferQueueController", namespace="transfer_queue")

    except ValueError:
        logger.info("Called _init_from_existing() but TransferQueueController has not been initialized yet.")
        return False

    logger.info("Found existing TransferQueueController instance. Connecting...")

    conf = None
    while conf is None:
        conf = ray.get(_TQ_CONTROLLER.get_config.remote())
        if conf is not None:
            _maybe_create_tq_client(conf)

            logger.info("TransferQueueClient initialized.")
            return True

        logger.debug("Waiting for controller to initialize... Retrying in 1s")
        time.sleep(1)

    return False


# ==================== Initialization API ====================
def init(conf: DictConfig | None = None) -> DictConfig | None:
    """Initialize the TransferQueue system.

    This function sets up the TransferQueue controller, distributed storage, and client.
    It should be called once at the beginning of the program before any data operations.

    If a controller already exists, reuse it and only initialize the client;
    the provided `conf` will be ignored in this case. Only the process that
    created the controller tears TransferQueue down in `close()`.

    Args:
        conf: Optional custom config merged with default `config.yaml`.
              Only takes effect on first-time initialization, ignored when attaching
              to an existing controller.
    Returns:
        The merged configuration dictionary.

    Example:
        >>> # In process 0, node A
        >>> import transfer_queue as tq
        >>> tq.init()   # Initialize the TransferQueue
        >>> tq.put(...) # then you can use tq for data operations
        >>>
        >>> # In process 1, node B (with Ray connected to node A)
        >>> import transfer_queue as tq
        >>> tq.init()   # This will only initialize a TransferQueueClient and link with existing TQ
        >>> metadata = tq.get_meta(...)
        >>> data = tq.get_data(metadata)
    """
    if _init_from_existing():
        return conf

    logger.info("No TransferQueueController found. Starting first-time initialization...")

    final_conf = OmegaConf.create({}, flags={"allow_objects": True})
    default_conf = OmegaConf.load(resources.files("transfer_queue") / "config.yaml")
    final_conf = OmegaConf.merge(final_conf, default_conf)
    if conf:
        final_conf = OmegaConf.merge(final_conf, conf)

    # TODO(hz): support load custom sampler class from external module.
    try:
        sampler = final_conf.controller.sampler
        if isinstance(sampler, BaseSampler):
            # user pass a pre-initialized sampler instance
            sampler = sampler
        elif isinstance(sampler, type) and issubclass(sampler, BaseSampler):
            # user pass a sampler class
            sampler = sampler()
        elif isinstance(sampler, str):
            # user pass a sampler name str
            # try to convert as sampler class
            sampler = globals()[final_conf.controller.sampler]
    except KeyError:
        raise ValueError(f"Could not find sampler {final_conf.controller.sampler}") from None
    lock_settings = _lock_settings()

    try:
        global _TQ_CONTROLLER, _TQ_IS_OWNER
        _TQ_CONTROLLER = TransferQueueController.options(  # type: ignore[attr-defined]
            name="TransferQueueController", namespace="transfer_queue"
        ).remote(sampler=sampler, polling_mode=final_conf.controller.polling_mode)
        _TQ_IS_OWNER = True
        logger.info("TransferQueueController has been created.")
        _start_lock_managers(lock_settings)
    except ValueError:
        logger.info("Some other rank has initialized TransferQueueController. Try to connect to existing controller.")
        _init_from_existing()
        return final_conf

    controller_zmq_info = process_zmq_server_info(_TQ_CONTROLLER)
    final_conf.controller.zmq_info = controller_zmq_info

    final_conf = _maybe_create_tq_storage(final_conf)

    ray.get(_TQ_CONTROLLER.store_config.remote(final_conf))
    logger.info(f"TransferQueue config: {final_conf}")

    # start Prometheus metrics exporter if enabled
    metrics_conf = final_conf.get("metrics", {})
    if metrics_conf.get("enabled", False):
        metrics_port = metrics_conf.get("port", 0)
        metrics_endpoint = ray.get(_TQ_CONTROLLER.start_metrics.remote(port=metrics_port))
        final_conf.metrics.enabled = True
        final_conf.metrics.endpoint = metrics_endpoint
        # Update stored config so other processes can discover the endpoint
        ray.get(_TQ_CONTROLLER.store_config.remote(final_conf))
        # Register storage units for metrics collection (SimpleStorage only)
        if final_conf.backend.storage_backend == "SimpleStorage":
            storage_zmq_info = final_conf.backend.SimpleStorage.get("zmq_info")
            if storage_zmq_info:
                ray.get(_TQ_CONTROLLER.register_storage_units_for_metrics.remote(storage_zmq_info))
            # Start metrics exporter on each storage unit
            if _TQ_STORAGE and "SimpleStorage" in _TQ_STORAGE:
                futures = [handle.start_metrics.remote(port=0) for handle in _TQ_STORAGE["SimpleStorage"].values()]
                ray.get(futures)
        logger.info(f"Prometheus metrics endpoint: http://{metrics_endpoint}/metrics")

    _maybe_create_tq_client(final_conf)
    return final_conf


def close():
    """Close the TransferQueue system.

    Every process closes its client and drops its handles. Only the process whose
    `init()` created TransferQueue also cleans up distributed storage and kills the
    controller and lock actors; processes that attached leave them running, release
    their own `kv_global_lock` leases, and may `init()` again.

    Note:
        This function should be called when the TransferQueue system is no longer needed.
    """
    global _TQ_CLIENT
    global _TQ_STORAGE
    global _TQ_CONTROLLER
    global _TQ_IS_OWNER

    try:
        if _TQ_STORAGE:
            for key, value in _TQ_STORAGE.items():
                if key == "SimpleStorage":
                    # Only the initializing process owns and shuts down these storage actors.
                    storage_handles = list(value.values())
                    try:
                        ray.get(
                            [storage.shutdown.remote() for storage in storage_handles],
                            timeout=_SIMPLE_STORAGE_SHUTDOWN_TIMEOUT_S,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to gracefully shut down SimpleStorage units: {e}")
                    finally:
                        for storage in storage_handles:
                            ray.kill(storage)
                elif key == "MooncakeStore":
                    check = subprocess.run(["pgrep", "-f", "mooncake_master"], stdout=subprocess.PIPE, text=True)
                    if check.returncode == 0:
                        pids = check.stdout.strip().replace("\n", ", ")
                        logger.warning(
                            f"TransferQueue will not stop mooncake_master process with PID: {pids}. "
                            f"Consider manually killing the mooncake_master."
                        )

                    # Terminate offload client process if it was started
                    if isinstance(value, dict):
                        offload_proc = value.get("offload_client_process")
                        if offload_proc is not None and offload_proc.poll() is None:
                            offload_proc.terminate()
                            try:
                                offload_proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                offload_proc.kill()
                            logger.info(f"Terminated mooncake_client offload process (PID: {offload_proc.pid}).")

                    if _TQ_CLIENT:
                        try:
                            ret = _TQ_CLIENT.storage_manager.storage_client._store.remove_all()
                            if ret < 0:
                                logger.error("Failed to remove existing keys in mooncake_master.")
                            else:
                                logger.info("Successfully removed all existing keys in mooncake_master.")
                        except Exception:
                            pass
                elif key == "Yuanrong":
                    cleanup_yuanrong_resources(value)
                else:
                    logger.warning(f"close for _TQ_STORAGE with key {key} is not supported for now.")

        _TQ_STORAGE = None
    except Exception:
        pass

    if _TQ_CLIENT:
        _TQ_CLIENT.close()
        _TQ_CLIENT = None

    _close_lock_managers()
    if _TQ_IS_OWNER:
        try:
            ray.kill(_TQ_CONTROLLER)
        except Exception:
            pass
    _TQ_CONTROLLER = None
    _TQ_IS_OWNER = False


# ==================== Metrics API ====================
def get_metrics_endpoint() -> str | None:
    """Return the Prometheus metrics endpoint address (``host:port``), or *None* if metrics are disabled.

    Works from any process — the endpoint is stored in the Controller's config
    so that processes joining via ``_init_from_existing()`` can discover it too.

    Example:
        >>> import transfer_queue as tq
        >>> tq.init({"metrics": {"enabled": True}})
        >>> endpoint = tq.get_metrics_endpoint()
        >>> print(endpoint)   # e.g. "10.0.1.42:38271"
        >>> # Use endpoint to register Prometheus scrape target
    """
    if _TQ_CONTROLLER is None:
        _init_from_existing()
    if _TQ_CONTROLLER is None:
        return None
    conf = ray.get(_TQ_CONTROLLER.get_config.remote())
    if conf is None:
        return None
    return conf.get("metrics", {}).get("endpoint", None)


# ==================== High-Level KV Interface API ====================
def kv_put(
    key: str,
    partition_id: str,
    fields: TensorDict | dict[str, Any] | None = None,
    tag: dict[str, Any] | None = None,
    data_parser: Callable[[Any], Any] | None = None,
) -> KVBatchMeta:
    """Put a single key-value pair to TransferQueue.

    This is a convenience method for putting data using a user-specified key
    instead of BatchMeta. Internally, the key is translated to a BatchMeta
    and the data is stored using the regular put mechanism.

    Args:
        key: User-specified key for the data sample (in row)
        partition_id: Logical partition to store the data in
        fields: Data fields to store. Can be a TensorDict or a dict of tensors.
                Each key in `fields` will be treated as a column for the data sample.
                If dict is provided, tensors will be unsqueezed to add batch dimension.
                If not provided, will only update the newly given tag to the key.
        tag: Optional metadata tag to associate with the key
        data_parser: Optional callable to parse reference data (e.g., URLs) into real
                     content. The input is a slice of the `fields` parameter passed to
                     kv_put / kv_batch_put, in plain dict format (not TensorDict),
                     mapping field_name -> batched values. For a regular tensor column
                     the value is a batched tensor; for nested tensors (jagged or
                     strided) and NonTensorStack columns the values are extracted into
                     a list. It must modify values in-place based on the original keys;
                     do not add or remove keys. The number of elements per column must
                     also remain unchanged. Do not change the inner order of values
                     within each column. Only supported by SimpleStorage.

    Returns:
        KVBatchMeta: Metadata containing the key, tags, partition_id, and fields.
                     The `fields` attribute includes all fields stored for this sample,
                     including any new fields written by this put operation.

    Raises:
        ValueError: If neither fields nor tag is provided
        ValueError: If nested tensors are provided (use kv_batch_put instead)
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`
        RuntimeError: If the controller fails to record the written data, e.g. on a dtype conflict.
            Storage changes are not rolled back; overwrites of ready fields may remain visible.

    Example:
        >>> import transfer_queue as tq
        >>> import torch
        >>> tq.init()
        >>> # Put with both fields and tag
        >>> meta = tq.kv_put(
        ...     key="sample_1",
        ...     partition_id="train",
        ...     fields={"input_ids": torch.tensor([1, 2, 3])},
        ...     tag={"score": 0.95}
        ... )
        >>> print(meta.fields)  # ['input_ids']
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(
        async_kv_put(
            key=key,
            partition_id=partition_id,
            fields=fields,
            tag=tag,
            data_parser=data_parser,
        )
    )


def kv_batch_put(
    keys: list[str],
    partition_id: str,
    fields: TensorDict | None = None,
    tags: list[dict[str, Any]] | None = None,
    data_parser: Callable[[Any], Any] | None = None,
) -> KVBatchMeta:
    """Batch put multiple key-value pairs into the TransferQueue.

    This method stores multiple key-value entries in a single operation,
    which is significantly more efficient than repeated calls to ``kv_put``.

    Args:
        keys: List of user-defined unique keys for the data entries.
        partition_id: Logical partition where the data will be stored.
        fields: TensorDict containing batched data for all keys. Must have ``batch_size == len(keys)``.
            If not provided, only the associated tags will be updated.
        tags: List of metadata dictionaries, one per key.  Length must match the number of keys.
        data_parser: Optional callable to parse raw reference data (e.g., URLs) into real content
            before storage. The input is a plain dict (not TensorDict) mapping field names to
            batched values. The parser  **must modify data in-place** without adding/removing
            keys or changing element counts/order. Only supported by ``SimpleStorage`` backend.

    Returns:
        KVBatchMeta: Metadata object containing stored keys, tags, partition ID,
            and field information. The ``fields`` attribute includes all
            persisted fields for the written samples.

    Raises:
        ValueError: When both ``fields`` and ``tags`` are empty.
        ValueError: When ``fields`` batch size mismatches key count.
        ValueError: When ``tags`` length mismatches key count.
        ValueError: When ``keys`` contains duplicates.
        RuntimeError: When retrieved metadata size mismatches input key count.
        RuntimeError: When the controller fails to record the written data, e.g. on a dtype conflict.
            Storage changes are not rolled back; overwrites of ready fields may remain visible.

    Example:
        >>> import transfer_queue as tq
        >>> from tensordict import TensorDict
        >>> tq.init()
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> tags = [{"score": 0.9}, {"score": 0.85}, {"score": 0.95}]
        >>> meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields, tags=tags)
        >>> print(meta.fields)
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(
        async_kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=fields,
            tags=tags,
            data_parser=data_parser,
        )
    )


def kv_batch_get_by_meta(meta: KVBatchMeta, select_fields: list[str] | str | None = None) -> TensorDict:
    """Get data from TransferQueue using KVBatchMeta.

    This is a convenience method for retrieving data using KVBatchMeta returned
    from a previous put operation. It extracts the keys and partition_id from
    the metadata to fetch the corresponding data.

    Args:
        meta: KVBatchMeta object returned from a previous put operation (e.g., kv_put,
              kv_batch_put). It contains keys, partition_id, and fields information.
        select_fields: Optional field(s) to retrieve, which overrides the fields
                       recorded in the given KVBatchMeta. If None, uses all fields
                       from meta.fields. Can be a single field name (str) or a list
                       of field names.

    Returns:
        TensorDict with the requested data

    Raises:
        ValueError: If the partition or any of the keys is not found
        ValueError: If empty fields exist in any key (sample)
        ValueError: If any field in select_fields doesn't exist in KVBatchMeta.fields

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # First put some data
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> meta = tq.kv_batch_put(keys=keys, partition_id="train", fields=fields)
        >>> # Then retrieve it using the returned metadata
        >>> data = tq.kv_batch_get_by_meta(meta)
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(async_kv_batch_get_by_meta(meta=meta, select_fields=select_fields))


def kv_batch_get(
    keys: list[str] | str,
    partition_id: str,
    select_fields: list[str] | str | None = None,
    data_parser: Callable[[dict[str, list], list | None], dict[str, list]] | None = None,
    parser_args: list | None = None,
) -> TensorDict:
    """Get data from TransferQueue using user-specified keys.

    This is a convenience method for retrieving data using keys instead of indexes.

    Args:
        keys: Single key or list of keys to retrieve
        partition_id: Partition containing the keys
        select_fields: Optional field(s) to retrieve. If None, retrieves all fields
        data_parser: Advanced. Optional ``data_parser(data, args) -> dict`` that each storage
            unit runs on its share of the batch before replying, so only the parsed values cross
            the network (the get-side counterpart of ``kv_batch_put``'s ``data_parser``).
            ``data`` maps each selected field to a list with one stored value per sample on that
            unit, and ``args`` holds the matching ``parser_args`` entries (None without
            ``parser_args``). Each field may be parsed differently, but the returned dict must
            keep the keys and the number and order of values per field. Constraints:

            - Do not modify stored values in place (e.g. ``tensor.add_()``, ``value.append()``):
              ``data`` holds the stored objects, so the change would persist and also reach
              later readers. Build new values instead; replacing entries of ``data`` is safe.
            - Keep it free of side effects. Like ``kv_batch_put`` with a parser, a storage-unit
              timeout is not retried and the call raises; retrying it yourself may run the parser
              again for the same samples.
            - It is sent with cloudpickle, so a module-level function must be importable on the
              storage units. SSD-offloaded values are read in full before parsing.

            Only supported by SimpleStorage; KV backends raise ``NotImplementedError``, and so
            does a selected field linked by ``kv_link``.
        parser_args: Optional per-key arguments for ``data_parser``, aligned with ``keys``.
            Requires ``data_parser``. They are serialized like request data, so tuples
            arrive as lists.

    Returns:
        TensorDict with the requested data

    Raises:
        ValueError: If the partition or any of the keys is not found
        ValueError: If empty fields exist in any key (sample)
        ValueError: If a field in ``select_fields`` is not ready in every found key
        ValueError: If ``parser_args`` is given without ``data_parser`` or does not match the
            number of keys
        RuntimeError: If ``data_parser`` fails or breaks its contract on a storage unit
        NotImplementedError: If ``data_parser`` is given and a selected field is linked by ``kv_link``

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Get single key with all fields
        >>> data = tq.kv_batch_get(keys="sample_1", partition_id="train")
        >>> # Get multiple keys with specific fields
        >>> data = tq.kv_batch_get(
        ...     keys=["sample_1", "sample_2"],
        ...     partition_id="train",
        ...     select_fields="input_ids"
        ... )
        >>> # Trim each key's input_ids and pick some of its images on the storage units
        >>> def pick(data, args):
        ...     data["input_ids"] = [ids[:n] for ids, (n, _) in zip(data["input_ids"], args)]
        ...     data["images"] = [[imgs[i] for i in idx] for imgs, (_, idx) in zip(data["images"], args)]
        ...     return data
        >>> data = tq.kv_batch_get(
        ...     keys=["sample_1", "sample_2"],
        ...     partition_id="train",
        ...     select_fields=["input_ids", "images"],
        ...     data_parser=pick,
        ...     parser_args=[(5, [0, 2]), (3, [1])],
        ... )
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(
        async_kv_batch_get(
            keys=keys,
            partition_id=partition_id,
            select_fields=select_fields,
            data_parser=data_parser,
            parser_args=parser_args,
        )
    )


def kv_list(partition_id: str | None = None) -> dict[str, dict[str, Any]]:
    """List all keys and their metadata in one or all partitions.

    Args:
        partition_id: The specific partition_id to query.
            If None (default), returns keys from all partitions.

    Returns:
        A nested dictionary mapping partition IDs to their keys and metadata.

        Structure:
        {
            "partition_id": {
                "key_name": {
                    "tag1": <value>,
                    ... (other metadata)
                },
                ...,
            },
            ...
        }

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Case 1: Retrieve a specific partition
        >>> partitions = tq.kv_list(partition_id="train")
        >>> print(f"Keys: {list(partitions['train'].keys())}")
        >>> print(f"Tags: {list(partitions['train'].values())}")
        >>> # Case 2: Retrieve all partitions
        >>> all_partitions = tq.kv_list()
        >>> for pid, keys in all_partitions.items():
        >>>     print(f"Partition: {pid}, Key count: {len(keys)}")
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(async_kv_list(partition_id=partition_id))


def kv_describe(keys: list[str] | str, partition_id: str) -> dict[str, dict[str, Any]]:
    """Describe which fields each key holds, including unfinalized ones.

    See ``async_kv_describe``.

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> rows = tq.kv_describe(keys=["sample_1", "sample_2"], partition_id="train")
        >>> hidden = {key: row["unfinalized"] for key, row in rows.items() if row["unfinalized"]}
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(async_kv_describe(keys=keys, partition_id=partition_id))


def kv_clear(keys: list[str] | str, partition_id: str, fields: list[str] | str | None = None) -> None:
    """Clear key-value pairs from TransferQueue.

    This removes the specified keys and their associated data from both
    the controller and storage units.

    Args:
        keys: Single key or list of keys to clear
        partition_id: Partition containing the keys
        fields: If given, clear only these fields and keep the keys. A cleared field
            reads as never written: it is not ready and can be put again. Tags and
            consumption status are kept. Missing keys and fields are ignored.

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Clear single key
        >>> tq.kv_clear(keys="sample_1", partition_id="train")
        >>> # Clear multiple keys
        >>> tq.kv_clear(keys=["sample_1", "sample_2"], partition_id="train")
        >>> # Release one field but keep the keys
        >>> tq.kv_clear(keys=["sample_1", "sample_2"], partition_id="train", fields="pixel_values")
    """

    tq_client = _maybe_create_tq_client()
    tq_client._run_coroutine(async_kv_clear(keys=keys, partition_id=partition_id, fields=fields))


def kv_link(keys: list[str] | str, partition_id: str, public_key: str, fields: list[str] | str) -> None:
    """Let ``keys`` read ``fields`` from ``public_key`` in ``PUBLIC_PARTITION`` without another copy.

    Put a payload shared by many keys, e.g. the video of a GRPO prompt read by its n
    responses, once into ``tq.PUBLIC_PARTITION`` and link the keys to it. Its rows are
    spread evenly over the storage units, apart from the rows of every other partition.
    Missing ``keys`` are created, and linked fields read the public key's current value
    and are ready whenever its fields are. Putting a linked field stores a private value
    in place of the link, and ``kv_update`` refuses it; clearing a linked key or field
    drops only the link. ``public_key`` cannot be cleared while keys link to it.

    Args:
        keys: Single key or list of keys that will read the linked fields.
        partition_id: Partition holding ``keys``; not ``PUBLIC_PARTITION``.
        public_key: Key in ``PUBLIC_PARTITION`` that has produced ``fields``.
        fields: Single field name or list of field names to link.

    Raises:
        RuntimeError: If ``public_key`` does not exist or has not produced all of ``fields``,
            or ``partition_id`` is ``PUBLIC_PARTITION``.
        ValueError: If ``keys`` contains duplicates.
        NotImplementedError: If the storage backend is not SimpleStorage.

    Example:
        >>> tq.kv_put(key="prompt_0", partition_id=tq.PUBLIC_PARTITION, fields={"video": video})
        >>> tq.kv_link([f"prompt_0_{i}" for i in range(n)], "train", "prompt_0", "video")
    """
    tq_client = _maybe_create_tq_client()
    tq_client._run_coroutine(async_kv_link(keys, partition_id, public_key, fields))


def kv_clear_unlinked_public() -> list[str]:
    """Clear every key in ``PUBLIC_PARTITION`` that no key links to, and return those keys.

    For example, after clearing a step's trajectories, this frees the prompt videos no
    trajectory reads any more. A public key that is put but not linked yet is cleared too,
    so call it when no writer is between its ``kv_put`` and ``kv_link``. A ``kv_link``
    that races with it is refused for the keys it clears.

    Example:
        >>> tq.kv_clear(trajectory_keys, partition_id="train")
        >>> freed = tq.kv_clear_unlinked_public()
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(async_kv_clear_unlinked_public())


def kv_update(
    key: str,
    partition_id: str,
    fields: dict[str, Any],
    merge_fn: Callable[[Any, Any], Any],
    finalize: bool = True,
) -> KVBatchMeta:
    """Merge new values into the fields of one key, adding the key or fields if missing.

    A missing key is created and a field the controller does not record for the key
    yet is stored as given, without calling ``merge_fn``. Unlike ``kv_put``, it takes
    no ``tags`` or ``data_parser``.

    Args:
        key: User-specified key. Created if missing.
        partition_id: Partition that holds the key.
        fields: New values keyed by field name.
        merge_fn: ``merge_fn(old, new) -> stored`` for fields the key already holds,
            hidden ones included. SimpleStorage runs it once per field on the unit
            that owns the sample.
        finalize: If False, the updated fields stay hidden from consumers until an update
            with True (the default), so a field built over several updates never shows an
            intermediate value. Hiding a field that is already ready is not guaranteed:
            readers that already hold its metadata can still see the new value.

    Returns:
        Metadata for the updated sample.

    Raises:
        TypeError: If ``fields`` is not a non-empty dict or ``merge_fn`` is not callable.
        RuntimeError: If ``merge_fn`` fails or a stored value would change the field's
            tensor/non-tensor type or dtype; the failing storage unit writes nothing. Also
            if a field is linked by ``kv_link``, before anything is registered or written.
        NotImplementedError: If the storage backend is not SimpleStorage.

    Note:
        A timeout has an unknown outcome because the storage unit may finish the
        merge after the caller stops waiting. Do not blindly retry a merge.
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(
        async_kv_update(
            key=key,
            partition_id=partition_id,
            fields=fields,
            merge_fn=merge_fn,
            finalize=finalize,
        )
    )


def kv_batch_update(
    keys: list[str],
    partition_id: str,
    fields: TensorDict,
    merge_fn: Callable[[Any, Any], Any],
    finalize: bool = True,
) -> KVBatchMeta:
    """Merge batched values into the fields of keys, adding keys or fields if missing.

    See ``kv_update``. ``keys`` must be unique; a repeated key raises ``ValueError``
    before anything is registered or written. A timeout has an unknown outcome because
    a unit may commit after the caller stops waiting. Do not blindly retry a merge.
    """
    tq_client = _maybe_create_tq_client()
    return tq_client._run_coroutine(
        async_kv_batch_update(keys=keys, partition_id=partition_id, fields=fields, merge_fn=merge_fn, finalize=finalize)
    )


# ==================== KV Interface API ====================
async def async_kv_put(
    key: str,
    partition_id: str,
    fields: TensorDict | dict[str, Any] | None = None,
    tag: dict[str, Any] | None = None,
    data_parser: Callable[[Any], Any] | None = None,
) -> KVBatchMeta:
    """Asynchronously put a single key-value pair to TransferQueue.

    This is a convenience method for putting data using a user-specified key
    instead of BatchMeta. Internally, the key is translated to a BatchMeta
    and the data is stored using the regular put mechanism.

    Args:
        key: User-specified key for the data sample (in row)
        partition_id: Logical partition to store the data in
        fields: Data fields to store. Can be a TensorDict or a dict of tensors.
                Each key in `fields` will be treated as a column for the data sample.
                If dict is provided, tensors will be unsqueezed to add batch dimension.
                If not provided, will only update the newly given tag to the key.
        tag: Optional metadata tag to associate with the key
        data_parser: Optional callable to parse reference data (e.g., URLs) into real
                     content. The input is a slice of the `fields` parameter passed to
                     kv_put / kv_batch_put, in plain dict format (not TensorDict),
                     mapping field_name -> batched values. For a regular tensor column
                     the value is a batched tensor; for nested tensors (jagged or
                     strided) and NonTensorStack columns the values are extracted into
                     a list. It must modify values in-place based on the original keys;
                     do not add or remove keys. The number of elements per column must
                     also remain unchanged. Do not change the inner order of values
                     within each column. Only supported by SimpleStorage.

    Returns:
        KVBatchMeta: Metadata containing the key, tags, partition_id, and fields.
                     The `fields` attribute includes all fields stored for this sample,
                     including any new fields written by this put operation.

    Raises:
        ValueError: If neither fields nor tag is provided
        ValueError: If nested tensors are provided (use kv_batch_put instead)
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`
        RuntimeError: If the controller fails to record the written data, e.g. on a dtype conflict.
            Storage changes are not rolled back; overwrites of ready fields may remain visible.

    Example:
        >>> import transfer_queue as tq
        >>> import torch
        >>> tq.init()
        >>> # Put with both fields and tag
        >>> meta = await tq.async_kv_put(
        ...     key="sample_1",
        ...     partition_id="train",
        ...     fields={"input_ids": torch.tensor([1, 2, 3])},
        ...     tag={"score": 0.95}
        ... )
        >>> print(meta.fields)  # ['input_ids']
    """

    if fields is None and tag is None:
        raise ValueError("Please provide at least one parameter of fields or tag.")

    tq_client = _maybe_create_tq_client()

    # 1. translate user-specified key to BatchMeta
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=[key], partition_id=partition_id, create=True)

    if batch_meta.size != 1:
        raise RuntimeError(f"Retrieved BatchMeta size {batch_meta.size} does not match with input `key` size of 1!")

    # 2. register the user-specified tag to BatchMeta
    if tag is not None:
        batch_meta.update_custom_meta([tag])

    # 3. put data
    if fields is not None:
        if isinstance(fields, dict):
            # TODO: consider whether to support this...
            batch = {}
            for field_name, value in fields.items():
                if isinstance(value, torch.Tensor):
                    if value.is_nested:
                        raise ValueError("Please use (async)kv_batch_put for batch operation")
                    batch[field_name] = value.unsqueeze(0)
                else:
                    batch[field_name] = NonTensorStack(value)
            fields = TensorDict(batch, batch_size=[1])
        elif not isinstance(fields, TensorDict):
            raise ValueError("`fields` can only be dict or TensorDict")

        # After put, batch_meta.field_names will include the new fields written by user
        batch_meta = await tq_client.async_put(fields, batch_meta, data_parser=data_parser)
    else:
        # Directly update custom_meta (tag) to controller
        await tq_client.async_set_custom_meta(batch_meta)

    fields_to_return = batch_meta.field_names

    return KVBatchMeta(
        keys=[key],
        tags=batch_meta.custom_meta,
        partition_id=partition_id,
        fields=fields_to_return,
        extra_info=batch_meta.extra_info,
    )


def _check_unique_keys(keys: list[str]) -> None:
    # KVBatchMeta rejects duplicates only after the write has landed, so reject them before any RPC.
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"keys must be unique, got duplicates: {duplicates[:20]}")


async def async_kv_batch_put(
    keys: list[str],
    partition_id: str,
    fields: TensorDict | None = None,
    tags: list[dict[str, Any]] | None = None,
    data_parser: Callable[[Any], Any] | None = None,
) -> KVBatchMeta:
    """Asynchronously put multiple key-value pairs to TransferQueue in batch.

    This method stores multiple key-value pairs in a single operation, which is more
    efficient than calling kv_put multiple times.

    Args:
        keys: List of user-specified keys for the data
        partition_id: Logical partition to store the data in
        fields: TensorDict containing data for all keys. Must have batch_size == len(keys).
                If not provided, will only update the newly given tags to the keys.
        tags: List of metadata tags, one for each key
        data_parser: Optional callable to parse reference data (e.g., URLs) into real
                     content. The input is a slice of the `fields` parameter passed to
                     kv_put / kv_batch_put, in plain dict format (not TensorDict),
                     mapping field_name -> batched values. For a regular tensor column
                     the value is a batched tensor; for nested tensors (jagged or
                     strided) and NonTensorStack columns the values are extracted into
                     a list. It must modify values in-place based on the original keys;
                     do not add or remove keys. The number of elements per column must
                     also remain unchanged. Do not change the inner order of values
                     within each column. Only supported by SimpleStorage.

    Returns:
        KVBatchMeta: Metadata containing the keys, tags, partition_id, and fields.
                     The `fields` attribute includes all fields stored for these samples,
                     including any new fields written by this put operation.

    Raises:
        ValueError: If neither `fields` nor `tags` is provided
        ValueError: If length of `keys` doesn't match length of `tags` or the batch_size of `fields` TensorDict
        ValueError: If `keys` contains duplicates
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`
        RuntimeError: If the controller fails to record the written data, e.g. on a dtype conflict.
            Storage changes are not rolled back; overwrites of ready fields may remain visible.

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> tags = [{"score": 0.9}, {"score": 0.85}, {"score": 0.95}]
        >>> meta = await tq.async_kv_batch_put(keys=keys, partition_id="train", fields=fields, tags=tags)
        >>> print(meta.fields)  # ['input_ids', 'attention_mask']
    """

    if fields is None and tags is None:
        raise ValueError("Please provide at least one parameter of `fields` or `tags`.")

    if fields is not None and fields.batch_size[0] != len(keys):
        raise ValueError(
            f"`keys` with length {len(keys)} does not match the `fields` TensorDict with "
            f"batch_size {fields.batch_size[0]}"
        )
    _check_unique_keys(keys)

    tq_client = _maybe_create_tq_client()

    # 1. translate user-specified key to BatchMeta
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=keys, partition_id=partition_id, create=True)

    if batch_meta.size != len(keys):
        raise RuntimeError(
            f"Retrieved BatchMeta size {batch_meta.size} does not match with input `keys` size {len(keys)}!"
        )

    # 2. register the user-specified tags to BatchMeta
    if tags is not None:
        if len(tags) != len(keys):
            raise ValueError(f"keys with length {len(keys)} does not match length of tags {len(tags)}")
        batch_meta.update_custom_meta(tags)

    # 3. put data
    if fields is not None:
        # After put, batch_meta.field_names will include the new fields written by user
        batch_meta = await tq_client.async_put(fields, batch_meta, data_parser=data_parser)
    else:
        # Directly update custom_meta (tags) to controller
        await tq_client.async_set_custom_meta(batch_meta)

    fields_to_return = batch_meta.field_names

    return KVBatchMeta(
        keys=keys,
        tags=batch_meta.custom_meta,
        partition_id=partition_id,
        fields=fields_to_return,
        extra_info=batch_meta.extra_info,
    )


async def async_kv_batch_get_by_meta(meta: KVBatchMeta, select_fields: list[str] | str | None = None) -> TensorDict:
    """Asynchronously get data from TransferQueue using KVBatchMeta.

    This is a convenience method for retrieving data using KVBatchMeta returned
    from a previous put operation. It extracts the keys and partition_id from
    the metadata to fetch the corresponding data.

    Args:
        meta: KVBatchMeta object returned from a previous put operation (e.g., async_kv_put,
              async_kv_batch_put). It contains keys, partition_id, and fields information.
        select_fields: Optional field(s) to retrieve, which overrides the fields
                       recorded in the given KVBatchMeta. If None, uses all fields
                       from meta.fields. Can be a single field name (str) or a list
                       of field names.

    Returns:
        TensorDict with the requested data

    Raises:
        ValueError: If the partition or any of the keys is not found
        ValueError: If empty fields exist in any key (sample)
        ValueError: If any field in select_fields doesn't exist in KVBatchMeta.fields

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # First put some data
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> meta = await tq.async_kv_batch_put(keys=keys, partition_id="train", fields=fields)
        >>> # Then retrieve it using the returned metadata
        >>> data = await tq.async_kv_batch_get_by_meta(meta)
    """
    if meta.partition_id is None:
        raise ValueError("Must provide partition_id in the input KVBatchMeta.")

    fields_to_fetch: list[str] | None
    if select_fields is not None:
        if isinstance(select_fields, str):
            fields_to_fetch = [select_fields]
        else:
            fields_to_fetch = select_fields

        assert fields_to_fetch is not None
        if meta.fields is None or any(f not in meta.fields for f in fields_to_fetch):
            raise ValueError(
                f"Some fields assigned in select_fields not found in the metadata. "
                f"Assigned: {fields_to_fetch}; Fields in KVBatchMeta: {meta.fields}."
            )
    else:
        fields_to_fetch = meta.fields
    return await async_kv_batch_get(keys=meta.keys, partition_id=meta.partition_id, select_fields=fields_to_fetch)


async def async_kv_batch_get(
    keys: list[str] | str,
    partition_id: str,
    select_fields: list[str] | str | None = None,
    data_parser: Callable[[dict[str, list], list | None], dict[str, list]] | None = None,
    parser_args: list | None = None,
) -> TensorDict:
    """Asynchronously get data from TransferQueue using user-specified keys.

    This is a convenience method for retrieving data using keys instead of indexes.

    Args:
        keys: Single key or list of keys to retrieve
        partition_id: Partition containing the keys
        select_fields: Optional field(s) to retrieve. If None, retrieves all fields
        data_parser: Advanced. Optional parser the storage units run before replying; see
            ``kv_batch_get`` for its contract and constraints.
        parser_args: Optional per-key arguments for ``data_parser``, aligned with ``keys``.

    Returns:
        TensorDict with the requested data

    Raises:
        ValueError: If the partition or any of the keys is not found
        ValueError: If empty fields exist in any key (sample)
        ValueError: If a field in ``select_fields`` is not ready in every found key
        ValueError: If ``parser_args`` is given without ``data_parser`` or does not match the
            number of keys
        RuntimeError: If ``data_parser`` fails or breaks its contract on a storage unit

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Get single key with all fields
        >>> data = await tq.async_kv_batch_get(keys="sample_1", partition_id="train")
        >>> # Get multiple keys with specific fields
        >>> data = await tq.async_kv_batch_get(
        ...     keys=["sample_1", "sample_2"],
        ...     partition_id="train",
        ...     select_fields="input_ids"
        ... )
    """
    tq_client = _maybe_create_tq_client()

    batch_meta = await tq_client.async_kv_retrieve_meta(keys=keys, partition_id=partition_id, create=False)

    if batch_meta.size == 0:
        raise ValueError("keys or partition were not found!")
    # The controller drops missing keys from the meta, so the rows would no longer line up with keys.
    requested = [keys] if isinstance(keys, str) else keys
    if batch_meta.size != len(requested):
        found = set(await tq_client.async_kv_retrieve_keys(batch_meta.global_indexes, partition_id))
        missing = [key for key in requested if key not in found]
        raise ValueError(
            f"{len(missing)} of {len(requested)} keys were not found in partition {partition_id!r}: {missing[:20]}"
        )

    if select_fields is not None:
        if isinstance(select_fields, str):
            fields_to_fetch = [select_fields]
        else:
            fields_to_fetch = select_fields
        # The meta keeps only fields ready in every key; select_fields would drop the rest silently.
        not_ready = [f for f in fields_to_fetch if f not in batch_meta.field_names]
        if not_ready:
            raise ValueError(
                f"Fields {not_ready} are not ready in all the requested keys: missing, or written by "
                "kv_update(finalize=False) and not finalized yet"
            )
        batch_meta = batch_meta.select_fields(fields_to_fetch)

    if not batch_meta.is_ready:
        raise ValueError("Some fields are not ready in all the requested keys!")

    data = await tq_client.async_get_data(batch_meta, data_parser=data_parser, parser_args=parser_args)
    return data


async def async_kv_list(partition_id: str | None = None) -> dict[str, dict[str, Any]]:
    """Asynchronously list all keys and their metadata in one or all partitions.

    Args:
        partition_id: The specific partition_id to query.
            If None (default), returns keys from all partitions.

    Returns:
        A nested dictionary mapping partition IDs to their keys and metadata.

        Structure:
        {
            "partition_id": {
                "key_name": {
                    "tag1": <value>,
                    ... (other metadata)
                },
                ...,
            },
            ...
        }


    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Case 1: Retrieve a specific partition
        >>> partitions = await tq.async_kv_list(partition_id="train")
        >>> print(f"Keys: {list(partitions['train'].keys())}")
        >>> print(f"Tags: {list(partitions['train'].values())}")
        >>> # Case 2: Retrieve all partitions
        >>> all_partitions = await tq.async_kv_list()
        >>> for pid, keys in all_partitions.items():
        >>>     print(f"Partition: {pid}, Key count: {len(keys)}")
    """
    tq_client = _maybe_create_tq_client()

    partition_info = await tq_client.async_kv_list(partition_id)

    return partition_info


async def async_kv_describe(keys: list[str] | str, partition_id: str) -> dict[str, dict[str, Any]]:
    """Asynchronously describe which fields each key holds, including unfinalized ones.

    ``kv_batch_get`` returns only fields that are ready in every key, so this is the way
    to find fields written by ``kv_update(finalize=False)`` and not yet finalized.

    Args:
        keys: Single key or list of keys to describe.
        partition_id: Partition containing the keys.

    Returns:
        ``{key: {"fields", "unfinalized", "links"}}``. ``fields`` lists every stored field of
        the key and ``unfinalized`` the subset that is not ready yet. ``links`` maps each field
        linked by ``kv_link`` to its key in ``PUBLIC_PARTITION``, whose readiness it shares.

    Raises:
        RuntimeError: If the partition or any key does not exist.

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> rows = await tq.async_kv_describe(keys="sample_1", partition_id="train")
        >>> print(rows["sample_1"]["unfinalized"])
    """
    keys = [keys] if isinstance(keys, str) else keys
    tq_client = _maybe_create_tq_client()
    description = await tq_client.async_kv_describe(list(dict.fromkeys(keys)), partition_id)
    return {
        key: {"fields": row["fields"], "unfinalized": row["unfinalized"], "links": row["links"]}
        for key, row in description["rows"].items()
    }


async def async_kv_clear(keys: list[str] | str, partition_id: str, fields: list[str] | str | None = None) -> None:
    """Asynchronously clear key-value pairs from TransferQueue.

    This removes the specified keys and their associated data from both
    the controller and storage units.

    Args:
        keys: Single key or list of keys to clear
        partition_id: Partition containing the keys
        fields: If given, clear only these fields and keep the keys. See ``kv_clear``.

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Clear single key
        >>> await tq.async_kv_clear(keys="sample_1", partition_id="train")
        >>> # Clear multiple keys
        >>> await tq.async_kv_clear(keys=["sample_1", "sample_2"], partition_id="train")
        >>> # Release one field but keep the keys
        >>> await tq.async_kv_clear(keys=["sample_1", "sample_2"], partition_id="train", fields="pixel_values")
    """

    if isinstance(keys, str):
        keys = [keys]

    tq_client = _maybe_create_tq_client()
    if fields is not None:
        await tq_client.async_kv_clear_fields(
            keys=keys, partition_id=partition_id, fields=[fields] if isinstance(fields, str) else fields
        )
        return

    batch_meta = await tq_client.async_kv_retrieve_meta(keys=keys, partition_id=partition_id, create=False)

    if batch_meta.size > 0:
        await tq_client.async_clear_samples(batch_meta)


async def async_kv_link(keys: list[str] | str, partition_id: str, public_key: str, fields: list[str] | str) -> None:
    """Asynchronously let ``keys`` read ``fields`` from ``public_key``. See ``kv_link``."""
    tq_client = _maybe_create_tq_client()
    keys = [keys] if isinstance(keys, str) else keys
    _check_unique_keys(keys)
    await tq_client.async_kv_link(keys, partition_id, public_key, [fields] if isinstance(fields, str) else fields)


async def async_kv_clear_unlinked_public() -> list[str]:
    """Asynchronously clear every ``PUBLIC_PARTITION`` key no key links to. See ``kv_clear_unlinked_public``."""
    tq_client = _maybe_create_tq_client()
    return await tq_client.async_clear_unlinked_public()


def _single_update_batch(fields: dict[str, Any]) -> TensorDict:
    """Wrap one sample's field mapping in a one-row TensorDict."""
    if not isinstance(fields, dict) or not fields:
        raise TypeError("fields must be a non-empty dict")
    batch: dict[str, Any] = {}
    for field_name, value in fields.items():
        if not isinstance(field_name, str):
            raise TypeError("field names must be strings")
        if isinstance(value, torch.Tensor):
            if value.is_nested:
                raise ValueError("Use async_kv_batch_update for nested tensors")
            batch[field_name] = value.unsqueeze(0)
        else:
            batch[field_name] = NonTensorStack(value)
    return TensorDict(batch, batch_size=[1])


async def async_kv_update(
    key: str,
    partition_id: str,
    fields: dict[str, Any],
    merge_fn: Callable[[Any, Any], Any],
    finalize: bool = True,
) -> KVBatchMeta:
    """Asynchronously merge or add fields of one key. See ``kv_update``."""
    return await async_kv_batch_update(
        keys=[key],
        partition_id=partition_id,
        fields=_single_update_batch(fields),
        merge_fn=merge_fn,
        finalize=finalize,
    )


async def async_kv_batch_update(
    keys: list[str],
    partition_id: str,
    fields: TensorDict,
    merge_fn: Callable[[Any, Any], Any],
    finalize: bool = True,
) -> KVBatchMeta:
    """Asynchronously merge or add batched fields of keys. See ``kv_update``."""
    if not isinstance(keys, list) or not keys or not all(isinstance(key, str) for key in keys):
        raise TypeError("keys must be a non-empty list of strings")
    if not isinstance(fields, TensorDict) or not fields.keys():
        raise TypeError("fields must be a non-empty TensorDict")
    if fields.batch_size != torch.Size([len(keys)]):
        raise ValueError(f"fields batch size {fields.batch_size} does not match {len(keys)} keys")
    if not callable(merge_fn):
        raise TypeError("merge_fn must be callable")
    _check_unique_keys(keys)
    tq_client = _maybe_create_tq_client()
    # Checked up front: missing keys are registered before storage is asked to merge.
    if not isinstance(tq_client.storage_manager, AsyncSimpleStorageManager):
        raise NotImplementedError(
            f"kv_update is not supported by {type(tq_client.storage_manager).__name__}; "
            "it requires the SimpleStorage backend"
        )
    batch_meta, merge_plan = await tq_client.async_kv_prepare_update(
        keys=keys, partition_id=partition_id, fields=list(fields.keys())
    )

    if batch_meta.size != len(keys):
        raise RuntimeError(f"Retrieved BatchMeta size {batch_meta.size} does not match {len(keys)} keys")

    batch_meta = await tq_client.async_update(
        metadata=batch_meta, values=fields, merge_fn=merge_fn, merge_plan=merge_plan, finalize=finalize
    )

    return KVBatchMeta(
        keys=keys,
        tags=batch_meta.custom_meta,
        partition_id=partition_id,
        fields=batch_meta.field_names,
        extra_info=batch_meta.extra_info,
    )


# ==================== Low-Level Native API ====================
# For low-level API support, please refer to transfer_queue/client.py for details.
def get_client():
    """Get a TransferQueueClient for using low-level API"""
    assert _TQ_CLIENT is not None, "Please initialize the TransferQueue first by calling `tq.init()`!"
    return _TQ_CLIENT


# ==================== Checkpoint API ====================

_METADATA_FILE = "metadata.json"
_CONTROLLER_FILE = "controller_state.pkl"


def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    include_storage: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a full checkpoint of the TransferQueue system state.

    .. note::
        **Multi-node limitation**: checkpoint_dir must reside on a shared network
        filesystem (e.g. NFS, GPFS, Lustre) accessible from all nodes.
        Single-node deployments have no such requirement.

    Args:
        checkpoint_dir: Directory to save the checkpoint (created if not exists).
        include_storage: Whether to save storage backend state. Set to False to
                         save controller metadata only (e.g. for KV backends where
                         data persists externally and does not need to be re-saved).
                         For SimpleStorage (in-memory), this flag is ignored and
                         storage is always saved — skipping it would cause data loss
                         on restart.
        metadata: User-defined key-value pairs written into metadata.json.
                  Example: {"time_stamp": 1234567.891234, "step": 10}

    Raises:
        RuntimeError: TransferQueue is not initialized.
        OSError: Failed to write checkpoint files.
    """
    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    checkpoint_dir = Path(checkpoint_dir)
    tmp_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}.tmp"
    old_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}.old"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        client = _maybe_create_tq_client()

        controller_path = tmp_dir / _CONTROLLER_FILE
        client.save_controller_checkpoint(str(controller_path))
        logger.info("Controller state saved.")

        if not include_storage and isinstance(client.storage_manager, AsyncSimpleStorageManager):
            logger.warning(
                "include_storage=False has no effect for SimpleStorage: "
                "in-memory data would be lost on restart. Forcing include_storage=True."
            )
            include_storage = True

        if include_storage:
            try:
                client.save_storage_checkpoint(str(tmp_dir))
            except NotImplementedError:
                logger.warning("Storage backend does not support checkpoint; storage data will not be saved.")
                include_storage = False

        meta_content = {
            "storage_saved": include_storage,
            "user_metadata": metadata or {},
        }
        with open(tmp_dir / _METADATA_FILE, "w") as f:
            json.dump(meta_content, f, indent=2)

        # Atomic replacement: move old aside, move new in place, delete old
        if checkpoint_dir.exists():
            if old_dir.exists():
                shutil.rmtree(old_dir)
            checkpoint_dir.rename(old_dir)

        tmp_dir.rename(checkpoint_dir)

        if old_dir.exists():
            shutil.rmtree(old_dir)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        if old_dir.exists() and not checkpoint_dir.exists():
            # Restore old checkpoint if new one failed
            old_dir.rename(checkpoint_dir)
        raise


def load_checkpoint(
    checkpoint_dir: str | Path,
) -> None:
    """Restore TransferQueue system state from a checkpoint.

    .. note::
        **Multi-node limitation**: checkpoint_dir must reside on a shared network
        filesystem (e.g. NFS, GPFS, Lustre) accessible from all nodes.
        Single-node deployments have no such requirement.

    Args:
        checkpoint_dir: Path to the checkpoint directory.

    Raises:
        FileNotFoundError: Checkpoint directory or required files do not exist.
        RuntimeError: TransferQueue is not initialized, or restore fails.
    """
    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    checkpoint_dir = Path(checkpoint_dir)
    old_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}.old"

    if not checkpoint_dir.exists() and old_dir.exists():
        # A prior save_checkpoint crashed between renaming checkpoint_dir aside
        # and installing the new version; roll back to the pre-crash checkpoint.
        logger.warning(f"Found interrupted checkpoint replacement; restoring {old_dir} to {checkpoint_dir}")
        old_dir.rename(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    metadata_path = checkpoint_dir / _METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(f"{_METADATA_FILE} not found in {checkpoint_dir}")

    with open(metadata_path) as f:
        meta = json.load(f)

    client = _maybe_create_tq_client()

    controller_path = checkpoint_dir / _CONTROLLER_FILE
    if not controller_path.exists():
        raise FileNotFoundError(f"{_CONTROLLER_FILE} not found in {checkpoint_dir}")

    if meta.get("storage_saved"):
        client.load_storage_checkpoint(str(checkpoint_dir))

    client.load_controller_checkpoint(str(controller_path))

    logger.info(f"Checkpoint loaded from {checkpoint_dir}")


# ==================== Selective Data Dump API ====================


def dump_data_by_key(dump_dir: str | Path, keys: list[str], partition_id: str) -> dict[str, int]:
    """Dump the rows addressed by ``keys`` into ``dump_dir``.

    Each storage unit pickles the rows it owns in its own process, so the payload never
    passes through the caller. The caller writes only a small row index.

    The dump is staged in a uniquely named sibling directory and renamed into place
    once durable. An existing ``dump_dir`` is refused, so a published dump is never
    replaced: loads need no lock and readers no write access.

    .. note::
        **Multi-node limitation**: dump_dir must reside on a shared network filesystem
        (e.g. NFS, GPFS, Lustre) reachable from every storage unit, because each unit
        writes its own shard from its own node.

    Callers must freeze writers for ``keys`` for the duration: TransferQueue has no
    atomic snapshot, so a concurrent put can land between the row index and the shards.

    Args:
        dump_dir: Directory to write the dump into.
        keys: Keys to dump. Duplicates are dropped, first occurrence wins. Fields linked
            by ``kv_link`` are dumped as links; dump their ``PUBLIC_PARTITION`` keys
            separately and load that dump first.
        partition_id: Partition that owns ``keys``.

    Returns:
        ``{"keys", "rows_with_data", "shards", "bytes"}``.

    Raises:
        FileExistsError: ``dump_dir`` already exists.
        RuntimeError: TransferQueue is not initialized, the partition or a key does not
            exist, or a storage unit holds no data for a row it was asked to dump.
        NotImplementedError: The storage backend does not support selective dumps.
    """
    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    dump_dir = Path(dump_dir).resolve()
    if dump_dir.exists():
        raise FileExistsError(f"{dump_dir} already exists; write each dump to a new directory")
    unique_keys = list(dict.fromkeys(keys))
    client = _maybe_create_tq_client()
    # A unique staging name keeps concurrent dumps to the same path from sharing files.
    tmp_dir = dump_dir.with_name(f"{dump_dir.name}.tmp-{uuid4().hex}")
    tmp_dir.mkdir(parents=True)

    try:
        description = client.kv_describe(unique_keys, partition_id) if unique_keys else {"rows": {}, "field_schema": {}}
        row_index = {"partition_id": partition_id, **description}
        rows = row_index["rows"]
        # A row whose fields are all still unproduced has nothing for a storage unit to
        # dump, but it keeps its key and tag so the restore can recreate the row.
        fields_by_index = {row["global_index"]: row["fields"] for row in rows.values() if row["fields"]}
        shard_records = []
        if fields_by_index:
            dumped = client.dump_rows_by_index(str(tmp_dir / data_dump.SHARD_SUBDIR), fields_by_index)
            shard_records = dumped["shards"]
            row_index["field_schema"] = data_dump.dump_field_schema(row_index["field_schema"], dumped["row_schema"])
        data_dump.publish_dump(tmp_dir, dump_dir, row_index, shard_records)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    total_bytes = sum(path.stat().st_size for path in dump_dir.rglob("*") if path.is_file())
    logger.info(f"Dumped {len(unique_keys)} keys of partition {partition_id} to {dump_dir}")
    return {
        "keys": len(unique_keys),
        "rows_with_data": len(fields_by_index),
        "shards": len(shard_records),
        "bytes": total_bytes,
    }


def load_data_by_key(dump_dir: str | Path) -> dict[str, dict]:
    """Merge selected rows into the running system, preserving existing key indexes.

    SimpleStorage units read their assigned indexed records directly and in parallel.
    New keys receive current indexes; unrelated rows and fields remain untouched.
    Writers and clears for these keys must be paused during restore. As with
    ``kv_batch_put``, a failure may leave new keys registered and existing keys' fields
    partially overwritten; retrying is idempotent.

    Args:
        dump_dir: Directory previously written by ``dump_data_by_key``. It must be
            accessible from every storage unit.

    Returns:
        ``{key: tag}`` for every restored key.

    Raises:
        RuntimeError: TransferQueue is not initialized, a storage unit fails, or a key
            linked to ``PUBLIC_PARTITION`` names a public key that does not exist.
        NotImplementedError: The storage backend is not SimpleStorage.
        FileNotFoundError: The dump is incomplete.
        ValueError: The manifest or a row disagrees with the row index, or a linked public
            field is not ready.
    """
    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    client = _maybe_create_tq_client()
    # Checked up front: the client registers the keys before any unit reads a shard.
    if not isinstance(client.storage_manager, AsyncSimpleStorageManager):
        raise NotImplementedError(f"{type(client.storage_manager).__name__} does not support selective data load")

    dump_dir = Path(dump_dir).resolve()
    row_index, shards = data_dump.read_dump(dump_dir)
    partition_id = row_index["partition_id"]
    rows = row_index["rows"]
    client.validate_dump_schema(partition_id, row_index["field_schema"])
    # Checked before loading anything: relinking needs the public fields, restored by their own load.
    public_keys = sorted({key for row in rows.values() for key in row.get("links", {}).values()})
    if public_keys:
        public_rows = client.kv_describe(public_keys, PUBLIC_PARTITION)["rows"]
        unready = sorted(
            {
                key
                for row in rows.values()
                for name, key in row.get("links", {}).items()
                if name not in public_rows[key]["fields"] or name in public_rows[key]["unfinalized"]
            }
        )
        if unready:
            raise ValueError(f"linked fields of {PUBLIC_PARTITION!r} keys {unready} are not ready; load them first")
    bytes_read = client.load_rows_by_key(partition_id, rows, shards)
    # One kv_link per public key and field set.
    link_groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for key, row in rows.items():
        links = row.get("links", {})
        for public_key in set(links.values()):
            link_groups[public_key, tuple(sorted(name for name, k in links.items() if k == public_key))].append(key)
    for (public_key, names), keys in link_groups.items():
        client.kv_link(keys, partition_id, public_key, list(names))
    logger.info(f"Restored {len(rows)} keys into partition {partition_id} from {dump_dir}, reading {bytes_read} bytes")
    return {key: row["tag"] for key, row in rows.items()}
