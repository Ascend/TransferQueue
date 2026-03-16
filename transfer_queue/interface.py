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
import math
import os
import subprocess
import time
from importlib import resources
from typing import Any, Optional
from urllib.parse import urlparse

import ray
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack

from transfer_queue.client import TransferQueueClient
from transfer_queue.controller import TransferQueueController
from transfer_queue.sampler import *  # noqa: F401
from transfer_queue.sampler import BaseSampler
from transfer_queue.storage.simple_backend import SimpleStorageUnit
from transfer_queue.utils.common import get_placement_group
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

_TRANSFER_QUEUE_CLIENT: Any = None
_TRANSFER_QUEUE_STORAGE: Any = None
_TRANSFER_QUEUE_CONTROLLER: Any = None


def _maybe_create_transferqueue_client(
    conf: Optional[DictConfig] = None,
) -> TransferQueueClient:
    global _TRANSFER_QUEUE_CLIENT
    if _TRANSFER_QUEUE_CLIENT is None:
        if conf is None:
            _init_from_existing()
            assert _TRANSFER_QUEUE_CLIENT is not None, (
                "TransferQueueController has not been initialized yet. Please call init() first."
            )
            return _TRANSFER_QUEUE_CLIENT

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
            simple_storage_handles = {}
            num_data_storage_units = conf.backend.SimpleStorage.num_data_storage_units
            total_storage_size = conf.backend.SimpleStorage.total_storage_size
            storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=1)

            for storage_unit_rank in range(num_data_storage_units):
                storage_node = SimpleStorageUnit.options(  # type: ignore[attr-defined]
                    placement_group=storage_placement_group,
                    placement_group_bundle_index=storage_unit_rank,
                    name=f"TransferQueueStorageUnit#{storage_unit_rank}",
                    lifetime="detached",
                ).remote(
                    storage_unit_size=math.ceil(total_storage_size / num_data_storage_units),
                )
                simple_storage_handles[f"TransferQueueStorageUnit#{storage_unit_rank}"] = storage_node
                logger.info(f"TransferQueueStorageUnit#{storage_unit_rank} has been created.")

            storage_zmq_info = process_zmq_server_info(simple_storage_handles)
            backend_name = conf.backend.storage_backend
            conf.backend[backend_name].zmq_info = storage_zmq_info
            _TRANSFER_QUEUE_STORAGE["SimpleStorage"] = simple_storage_handles
        if conf.backend.storage_backend == "MooncakeStore":
            if conf.backend.MooncakeStore.auto_init:
                # Try to kill existing mooncake_master processes before starting a new one to avoid potential conflicts
                check = subprocess.run(["pgrep", "-f", "mooncake_master"], stdout=subprocess.PIPE, text=True)
                if check.returncode == 0:
                    pids = check.stdout.strip().replace("\n", ", ")
                    logging.info(f"Find existing mooncake_master (PID: {pids}), try to kill first...")

                    result = os.system('pkill -f "[m]ooncake_master"')
                    if result == 0:
                        logging.info("Successfully killed existing mooncake_master processes.")
                    else:
                        raise RuntimeError(f"Failed to kill existing mooncake_master processes (exit code: {result}).")

                raw_address = conf.backend.MooncakeStore.metadata_server
                if "://" not in raw_address:
                    raw_address = "//" + raw_address

                parsed = urlparse(raw_address)

                if not parsed.hostname or parsed.port is None:
                    raise ValueError(
                        f"Invalid metadata_server '{conf.backend.MooncakeStore.metadata_server}'. "
                        f"Host and port are required (e.g., host:port)."
                    )

                metadata_server_host = parsed.hostname
                metadata_server_port = str(parsed.port)

                cmd = [
                    "mooncake_master",
                    "-default_kv_lease_ttl=999999",
                    "-default_kv_soft_pin_ttl=999999",
                    "--eviction_high_watermark_ratio=1.0",
                    "--eviction_ratio=0.0",
                    "--enable_http_metadata_server=true",
                    "--allow_evict_soft_pinned_objects=false",
                    f"--http_metadata_server_host={metadata_server_host}",
                    f"--http_metadata_server_port={metadata_server_port}",
                ]

                log_file_path = "/tmp/mooncake_master.log"
                with open(log_file_path, "w") as log_file:
                    process = subprocess.Popen(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        start_new_session=True,
                    )
                    time.sleep(3)

                if process.poll() is None:
                    logger.info(
                        f"mooncake_master started, PID: {process.pid}. Logs are at: {os.path.abspath(log_file_path)}"
                    )
                else:
                    error_msg = ""
                    try:
                        with open(log_file_path) as f:
                            error_msg = f.read()
                    except Exception as e:
                        error_msg = f"Failed to read log file: {e}"

                    raise RuntimeError(
                        f"mooncake_master exited with error. Check {log_file_path} for detailed logs. "
                        f"Output:\n{error_msg}"
                    )
                _TRANSFER_QUEUE_STORAGE["MooncakeStore"] = process
    return conf


def _init_from_existing() -> bool:
    """Initialize the TransferQueueClient from existing controller.

    Returns:
        True if successfully initialized from existing controller, False otherwise.
    """
    global _TRANSFER_QUEUE_CONTROLLER
    try:
        if _TRANSFER_QUEUE_CONTROLLER is None:
            _TRANSFER_QUEUE_CONTROLLER = ray.get_actor("TransferQueueController")

    except ValueError:
        logger.info("Called _init_from_existing() but TransferQueueController has not been initialized yet.")
        return False

    logger.info("Found existing TransferQueueController instance. Connecting...")

    conf = None
    while conf is None:
        conf = ray.get(_TRANSFER_QUEUE_CONTROLLER.get_config.remote())
        if conf is not None:
            _maybe_create_transferqueue_client(conf)
            logger.info("TransferQueueClient initialized.")
            return True

        logger.debug("Waiting for controller to initialize... Retrying in 1s")
        time.sleep(1)

    return False


# ==================== Initialization API ====================
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
    if _init_from_existing():
        return

    # First-time initialize TransferQueue
    logger.info("No TransferQueueController found. Starting first-time initialization...")

    # create config
    final_conf = OmegaConf.create({}, flags={"allow_objects": True})
    default_conf = OmegaConf.load(resources.files("transfer_queue") / "config.yaml")
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
        global _TRANSFER_QUEUE_CONTROLLER
        _TRANSFER_QUEUE_CONTROLLER = TransferQueueController.options(name="TransferQueueController").remote(  # type: ignore[attr-defined]
            sampler=sampler, polling_mode=final_conf.controller.polling_mode
        )
        logger.info("TransferQueueController has been created.")
    except ValueError:
        logger.info("Some other rank has initialized TransferQueueController. Try to connect to existing controller.")
        _init_from_existing()
        return

    controller_zmq_info = process_zmq_server_info(_TRANSFER_QUEUE_CONTROLLER)
    final_conf.controller.zmq_info = controller_zmq_info

    # create distributed storage backends
    final_conf = _maybe_create_transferqueue_storage(final_conf)

    # store the config into controller
    ray.get(_TRANSFER_QUEUE_CONTROLLER.store_config.remote(final_conf))
    logger.info(f"TransferQueue config: {final_conf}")

    # create client
    _maybe_create_transferqueue_client(final_conf)


def close():
    """Close the TransferQueue system.

    This function cleans up the TransferQueue system, including:
    - Closing the client and its associated resources
    - Cleaning up distributed storage (only for the process that initialized it)
    - Killing the controller actor

    Note:
        This function should be called when the TransferQueue system is no longer needed.
    """
    global _TRANSFER_QUEUE_CLIENT
    global _TRANSFER_QUEUE_STORAGE
    global _TRANSFER_QUEUE_CONTROLLER

    try:
        if _TRANSFER_QUEUE_STORAGE:
            for key, value in _TRANSFER_QUEUE_STORAGE.items():
                if key == "SimpleStorage":
                    # only the process that do first-time init can clean the distributed storage
                    for storage in value.values():
                        ray.kill(storage)
                elif key == "MooncakeStore":
                    check = subprocess.run(["pgrep", "-f", "mooncake_master"], stdout=subprocess.PIPE, text=True)
                    if check.returncode == 0:
                        pids = check.stdout.strip().replace("\n", ", ")
                        logger.warning(
                            f"TransferQueue will not stop mooncake_master process with PID: {pids}. "
                            f"Consider manually killing the mooncake_master."
                        )

                    if _TRANSFER_QUEUE_CLIENT:
                        try:
                            ret = _TRANSFER_QUEUE_CLIENT.storage_manager.storage_client._store.remove_all()
                            if ret < 0:
                                logger.error("Failed to remove existing keys in mooncake_master.")
                            else:
                                logger.info("Successfully removed all existing keys in mooncake_master.")
                        except Exception:
                            pass
                else:
                    logger.warning(f"close for _TRANSFER_QUEUE_STORAGE with key {key} is not supported for now.")

        _TRANSFER_QUEUE_STORAGE = None
    except Exception:
        pass

    if _TRANSFER_QUEUE_CLIENT:
        _TRANSFER_QUEUE_CLIENT.close()
        _TRANSFER_QUEUE_CLIENT = None

    if _TRANSFER_QUEUE_CONTROLLER:
        try:
            ray.kill(_TRANSFER_QUEUE_CONTROLLER)
        except Exception:
            pass
        _TRANSFER_QUEUE_CONTROLLER = None

    try:
        controller = ray.get_actor("TransferQueueController")
        ray.kill(controller)
    except Exception:
        pass


# ==================== High-Level KV Interface API ====================
def kv_put(
    key: str,
    partition_id: str,
    fields: Optional[TensorDict | dict[str, Any]] = None,
    tag: Optional[dict[str, Any]] = None,
) -> None:
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
        tag: Optional metadata tag to associate with the key

    Raises:
        ValueError: If neither fields nor tag is provided
        ValueError: If nested tensors are provided (use kv_batch_put instead)
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`

    Example:
        >>> import transfer_queue as tq
        >>> import torch
        >>> tq.init()
        >>> # Put with both fields and tag
        >>> tq.kv_put(
        ...     key="sample_1",
        ...     partition_id="train",
        ...     fields={"input_ids": torch.tensor([1, 2, 3])},
        ...     tag={"score": 0.95}
        ... )
    """
    if fields is None and tag is None:
        raise ValueError("Please provide at least one parameter of `fields` or `tag`.")

    tq_client = _maybe_create_transferqueue_client()

    # 1. translate user-specified key to BatchMeta
    batch_meta = tq_client.kv_retrieve_meta(keys=[key], partition_id=partition_id, create=True)

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
            raise ValueError("field can only be dict or TensorDict")

        # custom_meta (tag) will be put to controller through the internal put process
        tq_client.put(fields, batch_meta)
    else:
        # directly update custom_meta (tag) to controller
        tq_client.set_custom_meta(batch_meta)


def kv_batch_put(
    keys: list[str], partition_id: str, fields: Optional[TensorDict] = None, tags: Optional[list[dict[str, Any]]] = None
) -> None:
    """Put multiple key-value pairs to TransferQueue in batch.

    This method stores multiple key-value pairs in a single operation, which is more
    efficient than calling kv_put multiple times.

    Args:
        keys: List of user-specified keys for the data
        partition_id: Logical partition to store the data in
        fields: TensorDict containing data for all keys. Must have batch_size == len(keys)
        tags: List of metadata tags, one for each key

    Raises:
        ValueError: If neither `fields` nor `tags` is provided
        ValueError: If length of `keys` doesn't match length of `tags` or the batch_size of `fields` TensorDict
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`

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
        >>> tq.kv_batch_put(keys=keys, partition_id="train", fields=fields, tags=tags)
    """

    if fields is None and tags is None:
        raise ValueError("Please provide at least one parameter of fields or tag.")

    if fields is not None and fields.batch_size[0] != len(keys):
        raise ValueError(
            f"`keys` with length {len(keys)} does not match the `fields` TensorDict with "
            f"batch_size {fields.batch_size[0]}"
        )

    tq_client = _maybe_create_transferqueue_client()

    # 1. translate user-specified key to BatchMeta
    batch_meta = tq_client.kv_retrieve_meta(keys=keys, partition_id=partition_id, create=True)

    if batch_meta.size != len(keys):
        raise RuntimeError(
            f"Retrieved BatchMeta size {batch_meta.size} does not match with input `keys` size {len(keys)}!"
        )

    # 2. register the user-specified tags to BatchMeta
    if tags:
        if len(tags) != len(keys):
            raise ValueError(f"keys with length {len(keys)} does not match length of tags {len(tags)}")
        batch_meta.update_custom_meta(tags)

    # 3. put data
    if fields is not None:
        tq_client.put(fields, batch_meta)
    else:
        # directly update custom_meta (tags) to controller
        tq_client.set_custom_meta(batch_meta)


def kv_batch_get(keys: list[str] | str, partition_id: str, fields: Optional[list[str] | str] = None) -> TensorDict:
    """Get data from TransferQueue using user-specified keys.

    This is a convenience method for retrieving data using keys instead of indexes.

    Args:
        keys: Single key or list of keys to retrieve
        partition_id: Partition containing the keys
        fields: Optional field(s) to retrieve. If None, retrieves all fields

    Returns:
        TensorDict with the requested data

    Raises:
        RuntimeError: If keys or partition are not found
        RuntimeError: If empty fields exist in any key (sample)

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Get single key with all fields
        >>> data = tq.kv_batch_get(keys="sample_1", partition_id="train")
        >>> # Get multiple keys with specific fields
        >>> data = tq.kv_batch_get(
        ...     keys=["sample_1", "sample_2"],
        ...     partition_id="train",
        ...     fields="input_ids"
        ... )
    """
    tq_client = _maybe_create_transferqueue_client()

    batch_meta = tq_client.kv_retrieve_meta(keys=keys, partition_id=partition_id, create=False)

    if batch_meta.size == 0:
        raise RuntimeError("keys or partition were not found!")

    if fields is not None:
        if isinstance(fields, str):
            fields = [fields]
        batch_meta = batch_meta.select_fields(fields)

    if not batch_meta.is_ready:
        raise RuntimeError("Some fields are not ready in all the requested keys!")

    data = tq_client.get_data(batch_meta)
    return data


def kv_list(partition_id: Optional[str] = None) -> dict[str, dict[str, Any]]:
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
    tq_client = _maybe_create_transferqueue_client()

    partition_info = tq_client.kv_list(partition_id)

    return partition_info


def kv_clear(keys: list[str] | str, partition_id: str) -> None:
    """Clear key-value pairs from TransferQueue.

    This removes the specified keys and their associated data from both
    the controller and storage units.

    Args:
        keys: Single key or list of keys to clear
        partition_id: Partition containing the keys

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Clear single key
        >>> tq.kv_clear(keys="sample_1", partition_id="train")
        >>> # Clear multiple keys
        >>> tq.kv_clear(keys=["sample_1", "sample_2"], partition_id="train")
    """

    if isinstance(keys, str):
        keys = [keys]

    tq_client = _maybe_create_transferqueue_client()
    batch_meta = tq_client.kv_retrieve_meta(keys=keys, partition_id=partition_id, create=False)

    if batch_meta.size > 0:
        tq_client.clear_samples(batch_meta)


# ==================== KV Interface API ====================
async def async_kv_put(
    key: str,
    partition_id: str,
    fields: Optional[TensorDict | dict[str, Any]] = None,
    tag: Optional[dict[str, Any]] = None,
) -> None:
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
        tag: Optional metadata tag to associate with the key

    Raises:
        ValueError: If neither fields nor tag is provided
        ValueError: If nested tensors are provided (use kv_batch_put instead)
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`

    Example:
        >>> import transfer_queue as tq
        >>> import torch
        >>> tq.init()
        >>> # Put with both fields and tag
        >>> await tq.async_kv_put(
        ...     key="sample_1",
        ...     partition_id="train",
        ...     fields={"input_ids": torch.tensor([1, 2, 3])},
        ...     tag={"score": 0.95}
        ... ))
    """

    if fields is None and tag is None:
        raise ValueError("Please provide at least one parameter of fields or tag.")

    tq_client = _maybe_create_transferqueue_client()

    # 1. translate user-specified key to BatchMeta
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=[key], partition_id=partition_id, create=True)

    if batch_meta.size != 1:
        raise RuntimeError(f"Retrieved BatchMeta size {batch_meta.size} does not match with input `key` size of 1!")

    # 2. register the user-specified tag to BatchMeta
    if tag:
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
            raise ValueError("field can only be dict or TensorDict")

        # custom_meta (tag) will be put to controller through the put process
        await tq_client.async_put(fields, batch_meta)
    else:
        # directly update custom_meta (tag) to controller
        await tq_client.async_set_custom_meta(batch_meta)


async def async_kv_batch_put(
    keys: list[str], partition_id: str, fields: Optional[TensorDict] = None, tags: Optional[list[dict[str, Any]]] = None
) -> None:
    """Asynchronously put multiple key-value pairs to TransferQueue in batch.

    This method stores multiple key-value pairs in a single operation, which is more
    efficient than calling kv_put multiple times.

    Args:
        keys: List of user-specified keys for the data
        partition_id: Logical partition to store the data in
        fields: TensorDict containing data for all keys. Must have batch_size == len(keys)
        tags: List of metadata tags, one for each key

    Raises:
        ValueError: If neither `fields` nor `tags` is provided
        ValueError: If length of `keys` doesn't match length of `tags` or the batch_size of `fields` TensorDict
        RuntimeError: If retrieved BatchMeta size doesn't match length of `keys`

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> keys = ["sample_1", "sample_2", "sample_3"]
        >>> fields = TensorDict({
        ...     "input_ids": torch.randn(3, 10),
        ...     "attention_mask": torch.ones(3, 10),
        ... }, batch_size=3)
        >>> tags = [{"score": 0.9}, {"score": 0.85}, {"score": 0.95}]
        >>> await tq.async_kv_batch_put(keys=keys, partition_id="train", fields=fields, tags=tags)
    """

    if fields is None and tags is None:
        raise ValueError("Please provide at least one parameter of `fields` or `tags`.")

    if fields is not None and fields.batch_size[0] != len(keys):
        raise ValueError(
            f"`keys` with length {len(keys)} does not match the `fields` TensorDict with "
            f"batch_size {fields.batch_size[0]}"
        )

    tq_client = _maybe_create_transferqueue_client()

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
        await tq_client.async_put(fields, batch_meta)
    else:
        # directly update custom_meta (tags) to controller
        await tq_client.async_set_custom_meta(batch_meta)


async def async_kv_batch_get(
    keys: list[str] | str, partition_id: str, fields: Optional[list[str] | str] = None
) -> TensorDict:
    """Asynchronously get data from TransferQueue using user-specified keys.

    This is a convenience method for retrieving data using keys instead of indexes.

    Args:
        keys: Single key or list of keys to retrieve
        partition_id: Partition containing the keys
        fields: Optional field(s) to retrieve. If None, retrieves all fields

    Returns:
        TensorDict with the requested data

    Raises:
        RuntimeError: If keys or partition are not found
        RuntimeError: If empty fields exist in any key (sample)

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Get single key with all fields
        >>> data = await tq.async_kv_batch_get(keys="sample_1", partition_id="train")
        >>> # Get multiple keys with specific fields
        >>> data = await tq.async_kv_batch_get(
        ...     keys=["sample_1", "sample_2"],
        ...     partition_id="train",
        ...     fields="input_ids"
        ... )
    """
    tq_client = _maybe_create_transferqueue_client()

    batch_meta = await tq_client.async_kv_retrieve_meta(keys=keys, partition_id=partition_id, create=False)

    if batch_meta.size == 0:
        raise RuntimeError("keys or partition were not found!")

    if fields is not None:
        if isinstance(fields, str):
            fields = [fields]
        batch_meta = batch_meta.select_fields(fields)

    if not batch_meta.is_ready:
        raise RuntimeError("Some fields are not ready in all the requested keys!")

    data = await tq_client.async_get_data(batch_meta)
    return data


async def async_kv_list(partition_id: Optional[str] = None) -> dict[str, dict[str, Any]]:
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
    tq_client = _maybe_create_transferqueue_client()

    partition_info = await tq_client.async_kv_list(partition_id)

    return partition_info


async def async_kv_clear(keys: list[str] | str, partition_id: str) -> None:
    """Asynchronously clear key-value pairs from TransferQueue.

    This removes the specified keys and their associated data from both
    the controller and storage units.

    Args:
        keys: Single key or list of keys to clear
        partition_id: Partition containing the keys

    Example:
        >>> import transfer_queue as tq
        >>> tq.init()
        >>> # Clear single key
        >>> await tq.async_kv_clear(keys="sample_1", partition_id="train")
        >>> # Clear multiple keys
        >>> await tq.async_kv_clear(keys=["sample_1", "sample_2"], partition_id="train")
    """

    if isinstance(keys, str):
        keys = [keys]

    tq_client = _maybe_create_transferqueue_client()
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=keys, partition_id=partition_id, create=False)

    if batch_meta.size > 0:
        await tq_client.async_clear_samples(batch_meta)


# ==================== Low-Level Native API ====================
# For low-level API support, please refer to transfer_queue/client.py for details.
def get_client():
    """Get a TransferQueueClient for using low-level API"""
    if _TRANSFER_QUEUE_CLIENT is None:
        raise RuntimeError("Please initialize the TransferQueue first by calling `tq.init()`!")
    return _TRANSFER_QUEUE_CLIENT
