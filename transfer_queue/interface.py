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

import importlib.resources as pkg_resources
import logging
import math
import os
import time
from typing import Any, Optional

import ray
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from transfer_queue.client import TransferQueueClient
from transfer_queue.controller import TransferQueueController
from transfer_queue.metadata import BatchMeta
from transfer_queue.sampler import *  # noqa: F401
from transfer_queue.sampler import BaseSampler
from transfer_queue.storage.simple_backend import SimpleStorageUnit
from transfer_queue.utils.common import get_placement_group
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

_TRANSFER_QUEUE_CLIENT: Any = None
_TRANSFER_QUEUE_STORAGE: Any = None


def _maybe_create_transferqueue_client(
    conf: Optional[DictConfig] = None,
) -> TransferQueueClient:
    global _TRANSFER_QUEUE_CLIENT
    if _TRANSFER_QUEUE_CLIENT is None:
        if conf is None:
            raise ValueError("Missing config for initializing TransferQueueClient!")
        pid = os.getpid()
        _TRANSFER_QUEUE_CLIENT = TransferQueueClient(
            client_id=f"TransferQueueClient_{pid}", controller_info=conf.controller.zmq_info
        )

        backend_name = conf.backend.storage_backend

        _TRANSFER_QUEUE_CLIENT.initialize_storage_manager(manager_type=backend_name, config=conf.backend[backend_name])

    return _TRANSFER_QUEUE_CLIENT


def _maybe_create_transferqueue_storage(conf: DictConfig) -> DictConfig:
    global _TRANSFER_QUEUE_STORAGE

    if _TRANSFER_QUEUE_STORAGE is None:
        _TRANSFER_QUEUE_STORAGE = {}
        if conf.backend.storage_backend == "SimpleStorage":
            # initialize SimpleStorageUnit
            num_data_storage_units = conf.backend.SimpleStorage.num_data_storage_units
            total_storage_size = conf.backend.SimpleStorage.total_storage_size
            storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=1)

            for storage_unit_rank in range(num_data_storage_units):
                storage_node = SimpleStorageUnit.options(  # type: ignore[attr-defined]
                    placement_group=storage_placement_group,
                    placement_group_bundle_index=storage_unit_rank,
                    name=f"TransferQueueStorageUnit#{storage_unit_rank}",
                    lifetime="detached",
                ).remote(storage_unit_size=math.ceil(total_storage_size / num_data_storage_units))
                _TRANSFER_QUEUE_STORAGE[f"TransferQueueStorageUnit#{storage_unit_rank}"] = storage_node
                logger.info(f"TransferQueueStorageUnit#{storage_unit_rank} has been created.")

            storage_zmq_info = process_zmq_server_info(_TRANSFER_QUEUE_STORAGE)
            backend_name = conf.backend.storage_backend
            conf.backend[backend_name].zmq_info = storage_zmq_info

    return conf


def _init_from_existing() -> None:
    """Initialize the TransferQueueClient from existing controller."""

    controller = ray.get_actor("TransferQueueController")
    logger.info("Found existing TransferQueueController instance. Connecting...")

    conf = None
    while conf is None:
        remote_conf = ray.get(controller.get_config.remote())
        if remote_conf is not None:
            _maybe_create_transferqueue_client(remote_conf)
            logger.info("TransferQueueClient initialized.")
            return

        logger.debug("Waiting for controller to initialize... Retrying in 1s")
        time.sleep(1)


def init(conf: Optional[DictConfig] = None) -> None:
    """Initialize the TransferQueue system.

    This function sets up the TransferQueue controller, distributed storage, and client.
    It should be called once at the beginning of the program before any data operations.

    If a controller already exists (e.g., initialized by another process), this function
    will retrieve the config from existing controller and initialize the TransferQueueClient.
    In this case, the `conf` parameter will be ignored.

    Args:
        conf: Optional configuration dictionary. If provided, it will be merged with
              the default config from 'config.yaml'. This is only used for first-time
              initializing. When connecting to an existing controller, this parameter
              is ignored.

    Raises:
        ValueError: If config is not valid or required configuration keys are missing.

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
    try:
        _init_from_existing()
    except ValueError:
        logger.info("No TransferQueueController found. Starting first-time initialization...")
    else:
        return

    # First-time initialize TransferQueue

    # create config
    final_conf = OmegaConf.create({}, flags={"allow_objects": True})
    with pkg_resources.path("transfer_queue", "config.yaml") as p:
        default_conf = OmegaConf.load(p)
    final_conf = OmegaConf.merge(final_conf, default_conf)
    if conf:
        final_conf = OmegaConf.merge(final_conf, conf)

    # create controller
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

    try:
        # Ray will make sure actor with same name can only be created once
        controller = TransferQueueController.options(name="TransferQueueController", lifetime="detached").remote(  # type: ignore[attr-defined]
            sampler=sampler, polling_mode=final_conf.controller.polling_mode
        )
        logger.info("TransferQueueController has been created.")
    except ValueError:
        logger.info("Some other rank has initialized TransferQueueController. Try to connect to existing controller.")
        _init_from_existing()
        return

    controller_zmq_info = process_zmq_server_info(controller)
    final_conf.controller.zmq_info = controller_zmq_info

    # create distributed storage backends
    final_conf = _maybe_create_transferqueue_storage(final_conf)

    # store the config into controller
    ray.get(controller.store_config.remote(final_conf))
    logger.info(f"TransferQueue config: {final_conf}")

    # create client
    _maybe_create_transferqueue_client(final_conf)


def get_meta(
    data_fields: list[str],
    batch_size: int,
    partition_id: str,
    mode: str = "fetch",
    task_name: Optional[str] = None,
    sampling_config: Optional[dict[str, Any]] = None,
) -> BatchMeta:
    """Synchronously fetch data metadata from the controller via ZMQ.

    Args:
        data_fields: List of data field names to retrieve metadata for
        batch_size: Number of samples to request in the batch
        partition_id: Current data partition id
        mode: Data fetch mode. Options:
            - 'fetch': Get ready data only
            - 'force_fetch': Get data regardless of readiness (may return unready samples)
            - 'insert': Internal usage - should not be used by users
        task_name: Optional task name associated with the request
        sampling_config: Optional sampling configuration for custom samplers.


    Returns:
        BatchMeta: Metadata object containing data structure, sample information, and readiness status

    Raises:
        RuntimeError: If communication fails or controller returns error response

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>>
        >>> # Example 1: Basic fetch metadata
        >>> batch_meta = tq.get_meta(
        ...     data_fields=["input_ids", "attention_mask"],
        ...     batch_size=4,
        ...     partition_id="train_0",
        ...     mode="fetch",
        ...     task_name="generate_sequences"
        ... )
        >>> print(batch_meta.is_ready)  # True if all samples ready
        >>>
        >>> # Example 2: Fetch with self-defined samplers (using GRPOGroupNSampler as an example)
        >>> batch_meta = tq.get_meta(
        ...     data_fields=["input_ids", "attention_mask"],
        ...     batch_size=8,
        ...     partition_id="train_0",
        ...     mode="fetch",
        ...     task_name="generate_sequences",
        ...     sampling_config={"n_samples_per_prompt": 4}
        ... )
        >>> print(batch_meta.is_ready)  # True if all samples ready
        >>>
        >>> # Example 3: Force fetch metadata (bypass production status check and Sampler,
        >>> # so may include unready and already-consumed samples. No filtering by consumption status is applied.)
        >>> batch_meta = tq.get_meta(
        ...     partition_id="train_0",   # optional
        ...     mode="force_fetch",
        ... )
        >>> print(batch_meta.is_ready)  # May be False if some samples not ready
    """

    tq_client = _maybe_create_transferqueue_client()
    return tq_client.get_meta(data_fields, batch_size, partition_id, mode, task_name, sampling_config)


async def async_get_meta(
    data_fields: list[str],
    batch_size: int,
    partition_id: str,
    mode: str = "fetch",
    task_name: Optional[str] = None,
    sampling_config: Optional[dict[str, Any]] = None,
) -> BatchMeta:
    """Asynchronously fetch data metadata from the controller via ZMQ.

    Args:
        data_fields: List of data field names to retrieve metadata for
        batch_size: Number of samples to request in the batch
        partition_id: Current data partition id
        mode: Data fetch mode. Options:
            - 'fetch': Get ready data only
            - 'force_fetch': Get data regardless of readiness (may return unready samples)
            - 'insert': Internal usage - should not be used by users
        task_name: Optional task name associated with the request
        sampling_config: Optional sampling configuration for custom samplers.
        socket: ZMQ async socket for message transmission (injected by decorator)

    Returns:
        BatchMeta: Metadata object containing data structure, sample information, and readiness status

    Raises:
        RuntimeError: If communication fails or controller returns error response

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>>
        >>> # Example 1: Basic fetch metadata
        >>> batch_meta = asyncio.run(tq.async_get_meta(
        ...     data_fields=["input_ids", "attention_mask"],
        ...     batch_size=4,
        ...     partition_id="train_0",
        ...     mode="fetch",
        ...     task_name="generate_sequences"
        ... ))
        >>> print(batch_meta.is_ready)  # True if all samples ready
        >>>
        >>> # Example 2: Fetch with self-defined samplers (using GRPOGroupNSampler as an example)
        >>> batch_meta = asyncio.run(tq.async_get_meta(
        ...     data_fields=["input_ids", "attention_mask"],
        ...     batch_size=8,
        ...     partition_id="train_0",
        ...     mode="fetch",
        ...     task_name="generate_sequences",
        ... ))
        >>> print(batch_meta.is_ready)  # True if all samples ready
        >>>
        >>> # Example 3: Force fetch metadata (bypass production status check and Sampler,
        >>> # so may include unready and already-consumed samples. No filtering by consumption status is applied.)
        >>> batch_meta = asyncio.run(tq.async_get_meta(
        ...     partition_id="train_0",   # optional
        ...     mode="force_fetch",
        ... ))
        >>> print(batch_meta.is_ready)  # May be False if some samples not ready
    """

    tq_client = _maybe_create_transferqueue_client()
    return await tq_client.async_get_meta(data_fields, batch_size, partition_id, mode, task_name, sampling_config)


def get_data(metadata: BatchMeta) -> TensorDict:
    """Synchronously fetch data from storage units and organize into TensorDict.

    Args:
        metadata: Batch metadata containing data location information and global indexes

    Returns:
        TensorDict containing:
            - Requested data fields (e.g., "prompts", "attention_mask")

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>>
        >>> batch_meta = tq.get_data(
        ...     data_fields=["prompts", "attention_mask"],
        ...     batch_size=4,
        ...     partition_id="train_0",
        ...     mode="fetch",
        ...     task_name="generate_sequences",
        ... )
        >>> batch = tq.get_data(batch_meta)
        >>> print(batch)
        >>> # TensorDict with fields "prompts", "attention_mask", and sample order matching metadata global_indexes
    """
    tq_client = _maybe_create_transferqueue_client()
    return tq_client.get_data(metadata)


async def async_get_data(metadata: BatchMeta) -> TensorDict:
    """Asynchronously fetch data from storage units and organize into TensorDict.

    Args:
        metadata: Batch metadata containing data location information and global indexes

    Returns:
        TensorDict containing:
            - Requested data fields (e.g., "prompts", "attention_mask")

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>>
        >>> batch_meta = asyncio.run(tq.async_get_meta(
        ...     data_fields=["prompts", "attention_mask"],
        ...     batch_size=4,
        ...     partition_id="train_0",
        ...     mode="fetch",
        ...     task_name="generate_sequences",
        ... ))
        >>> batch = asyncio.run(tq.async_get_data(batch_meta))
        >>> print(batch)
        >>> # TensorDict with fields "prompts", "attention_mask", and sample order matching metadata global_indexes
    """
    tq_client = _maybe_create_transferqueue_client()
    return await tq_client.async_get_data(metadata)


def put(data: TensorDict, metadata: Optional[BatchMeta] = None, partition_id: Optional[str] = None) -> BatchMeta:
    """Synchronously write data to storage units based on metadata.

    If metadata is not provided, it will be created automatically using insert mode
    with the provided data fields and partition_id.

    During put, the custom_meta in metadata will update the corresponding custom_meta in
    TransferQueue Controller.

    Note:
        When using multiple workers for distributed execution, there may be data
        ordering inconsistencies between workers during put operations.

    Args:
        data: Data to write as TensorDict
        metadata: Records the metadata of a batch of data samples, containing index and
                  storage unit information. If None, metadata will be auto-generated.
        partition_id: Target data partition id (required if metadata is not provided)

    Returns:
        BatchMeta: The metadata used for the put operation (currently returns the input metadata or auto-retrieved
                   metadata; will be updated in a future version to reflect the post-put state)

    Raises:
        ValueError: If metadata is None or empty, or if partition_id is None when metadata is not provided
        RuntimeError: If storage operation fails

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>>
        >>> batch_size = 4
        >>> seq_len = 16
        >>> current_partition_id = "train_0"
        >>> # Example 1: Normal usage with existing metadata
        >>> batch_meta = tq.get_meta(
        ...     data_fields=["prompts", "attention_mask"],
        ...     batch_size=batch_size,
        ...     partition_id=current_partition_id,
        ...     mode="fetch",
        ...     task_name="generate_sequences",
        ... )
        >>> batch = tq.get_data(batch_meta)
        >>> output = TensorDict({"response": torch.randn(batch_size, seq_len)})
        >>> tq.put(data=output, metadata=batch_meta)
        >>>
        >>> # Example 2: Initial data insertion without pre-existing metadata
        >>> # BE CAREFUL: this usage may overwrite any unconsumed data in the given partition_id!
        >>> # Please make sure the corresponding partition_id is empty before calling the async_put()
        >>> # without metadata.
        >>> # Now we only support put all the data of the corresponding partition id in once. You should repeat with
        >>> # interleave the initial data if n_sample > 1 before calling the async_put().
        >>> original_prompts = torch.randn(batch_size, seq_len)
        >>> n_samples = 4
        >>> prompts_repeated = torch.repeat_interleave(original_prompts, n_samples, dim=0)
        >>> prompts_repeated_batch = TensorDict({"prompts": prompts_repeated})
        >>> # This will create metadata in "insert" mode internally.
        >>> metadata = tq.put(data=prompts_repeated_batch, partition_id=current_partition_id)
    """
    tq_client = _maybe_create_transferqueue_client()
    return tq_client.put(data, metadata, partition_id)


async def async_put(
    data: TensorDict,
    metadata: Optional[BatchMeta] = None,
    partition_id: Optional[str] = None,
) -> BatchMeta:
    """Asynchronously write data to storage units based on metadata.

    If metadata is not provided, it will be created automatically using insert mode
    with the provided data fields and partition_id.

    During put, the custom_meta in metadata will update the corresponding custom_meta in
    TransferQueue Controller.

    Note:
        When using multiple workers for distributed execution, there may be data
        ordering inconsistencies between workers during put operations.

    Args:
        data: Data to write as TensorDict
        metadata: Records the metadata of a batch of data samples, containing index and
                  storage unit information. If None, metadata will be auto-generated.
        partition_id: Target data partition id (required if metadata is not provided)

    Returns:
        BatchMeta: The metadata used for the put operation (currently returns the input metadata or auto-retrieved
                   metadata; will be updated in a future version to reflect the post-put state)

    Raises:
        ValueError: If metadata is None or empty, or if partition_id is None when metadata is not provided
        RuntimeError: If storage operation fails

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>>
        >>> batch_size = 4
        >>> seq_len = 16
        >>> current_partition_id = "train_0"
        >>> # Example 1: Normal usage with existing metadata
        >>> batch_meta = asyncio.run(tq.async_get_meta(
        ...     data_fields=["prompts", "attention_mask"],
        ...     batch_size=batch_size,
        ...     partition_id=current_partition_id,
        ...     mode="fetch",
        ...     task_name="generate_sequences",
        ... ))
        >>> batch = asyncio.run(tq.async_get_data(batch_meta))
        >>> output = TensorDict({"response": torch.randn(batch_size, seq_len)})
        >>> asyncio.run(tq.async_put(data=output, metadata=batch_meta))
        >>>
        >>> # Example 2: Initial data insertion without pre-existing metadata
        >>> # BE CAREFUL: this usage may overwrite any unconsumed data in the given partition_id!
        >>> # Please make sure the corresponding partition_id is empty before calling the async_put()
        >>> # without metadata.
        >>> # Now we only support put all the data of the corresponding partition id in once. You should repeat with
        >>> # interleave the initial data if n_sample > 1 before calling the async_put().
        >>> original_prompts = torch.randn(batch_size, seq_len)
        >>> n_samples = 4
        >>> prompts_repeated = torch.repeat_interleave(original_prompts, n_samples, dim=0)
        >>> prompts_repeated_batch = TensorDict({"prompts": prompts_repeated})
        >>> # This will create metadata in "insert" mode internally.
        >>> metadata = asyncio.run(tq.async_put(data=prompts_repeated_batch, partition_id=current_partition_id))
    """
    tq_client = _maybe_create_transferqueue_client()
    return await tq_client.async_put(data, metadata, partition_id)


def set_custom_meta(metadata: BatchMeta) -> None:
    """Synchronously send custom metadata to the controller.

    This method sends per-sample custom metadata (custom_meta) to the controller.
    The custom_meta is stored in the controller and can be retrieved along with
    the BatchMeta in subsequent get_meta calls.

    Args:
        metadata: BatchMeta containing the samples and their custom metadata to store.
                 The custom_meta should be set using BatchMeta.update_custom_meta() or
                 BatchMeta.set_custom_meta() before calling this method.

    Raises:
        RuntimeError: If communication fails or controller returns error response

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>>
        >>> # Create batch with custom metadata
        >>> batch_meta = tq.get_meta(data_fields=["input_ids"], batch_size=4, ...)
        >>> batch_meta.update_custom_meta({0: {"score": 0.9}, 1: {"score": 0.8}})
        >>> tq.set_custom_meta(batch_meta)
    """
    tq_client = _maybe_create_transferqueue_client()
    return tq_client.set_custom_meta(metadata)


async def async_set_custom_meta(
    metadata: BatchMeta,
) -> None:
    """
    Asynchronously send custom metadata to the controller.

    This method sends per-sample custom metadata (custom_meta) to the controller.
    The custom_meta is stored in the controller and can be retrieved along with
    the BatchMeta in subsequent get_meta calls.

    Args:
        metadata: BatchMeta containing the samples and their custom metadata to store.
                 The custom_meta should be set using BatchMeta.update_custom_meta() or
                 BatchMeta.set_custom_meta() before calling this method.
        socket: ZMQ async socket for message transmission (injected by decorator)

    Raises:
        RuntimeError: If communication fails or controller returns error response

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>>
        >>> # Create batch with custom metadata
        >>> batch_meta = tq.get_meta(data_fields=["input_ids"], batch_size=4, ...)
        >>> batch_meta.update_custom_meta({0: {"score": 0.9}, 1: {"score": 0.8}})
        >>> asyncio.run(tq.async_set_custom_meta(batch_meta))
    """
    tq_client = _maybe_create_transferqueue_client()
    return await tq_client.async_set_custom_meta(metadata)


def clear_samples(metadata: BatchMeta):
    """Synchronously clear specific samples from all storage units and the controller.

    Args:
        metadata: The BatchMeta of the corresponding data to be cleared

    Raises:
        RuntimeError: If clear operation fails
    """
    tq_client = _maybe_create_transferqueue_client()
    return tq_client.clear_samples(metadata)


async def async_clear_samples(metadata: BatchMeta):
    """Asynchronously clear specific samples from all storage units and the controller.

    Args:
        metadata: The BatchMeta of the corresponding data to be cleared

    Raises:
        RuntimeError: If clear operation fails
    """
    tq_client = _maybe_create_transferqueue_client()
    return await tq_client.async_clear_samples(metadata)


def clear_partition(partition_id: str):
    """Synchronously clear the whole partition from all storage units and the controller.

    Args:
        partition_id: The partition id to clear data for

    Raises:
        RuntimeError: If clear operation fails
    """
    tq_client = _maybe_create_transferqueue_client()
    return tq_client.clear_partition(partition_id)


async def async_clear_partition(partition_id: str):
    """Asynchronously clear the whole partition from all storage units and the controller.

    Args:
        partition_id: The partition id to clear data for

    Raises:
        RuntimeError: If clear operation fails
    """
    tq_client = _maybe_create_transferqueue_client()
    return await tq_client.async_clear_partition(partition_id)


def close():
    """Close the TransferQueue system."""
    global _TRANSFER_QUEUE_CLIENT
    global _TRANSFER_QUEUE_STORAGE
    if _TRANSFER_QUEUE_CLIENT:
        _TRANSFER_QUEUE_CLIENT.close()
        _TRANSFER_QUEUE_CLIENT = None

    try:
        if _TRANSFER_QUEUE_STORAGE:
            # only the process that do first-time init can clean the distributed storage
            for storage in _TRANSFER_QUEUE_STORAGE.values():
                ray.kill(storage)
        _TRANSFER_QUEUE_STORAGE = None
    except Exception:
        pass

    try:
        controller = ray.get_actor("TransferQueueController")
        ray.kill(controller)
    except Exception:
        pass
