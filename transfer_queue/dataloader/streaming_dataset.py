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
import time
import uuid
import warnings
from typing import Callable, Iterator

from omegaconf import DictConfig
from tensordict import TensorDict
from torch.utils.data import IterableDataset

from transfer_queue import TransferQueueClient
from transfer_queue.metadata import BatchMeta
from transfer_queue.utils.zmq_utils import ZMQServerInfo

TQ_STREAMING_DATASET_EMPTY_BATCH_SLEEP_INTERVAL = float(
    os.environ.get("TQ_STREAMING_DATASET_EMPTY_BATCH_SLEEP_INTERVAL", 1)
)  # in seconds

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


class StreamingDataset(IterableDataset):
    """Streaming dataset for distributed training with TransferQueue.

    This dataset is designed to work with RankAwareSampler for distributed training
    scenarios where each rank independently retrieves data through TransferQueue.

    Usage Example:
        >>> dataset = StreamingDataset(
        ...     config=config,
        ...     micro_batch_size=4,
        ...     required_fields=["input_ids", "attention_mask"],
        ...     partition_id="train",
        ...     task_name="update_actor",
        ...     dp_rank=dp_rank,          # Same for all ranks in data replica group
        ... )
        >>> dataloader = StreamingDataLoader(
        ...     dataset,
        ...     num_workers=2,          # num_workers for data retrieval, each has a TQ client for async data retrieval
        ...     prefetch_factor=2,      # number of batches loaded in advance by each worker
        ... )
        >>> for batch, batch_meta in dataloader:
        ...     # batch is a TensorDict with the requested fields
        ...     # batch_meta contains metadata for TransferQueue coordination
        ...     pass
    """

    def __init__(
        self,
        config: DictConfig,
        batch_size: int,
        micro_batch_size: int,
        data_fields: list[str],
        partition_id: str,
        task_name: str,
        dp_rank: int,
        fetch_batch_fn: Callable | None = None,
        process_batch_fn: Callable | None = None,
    ):
        """Initialize the StreamingDataset.

        Args:
            config: Configuration dictionary containing:
                - controller.controller_info: ZMQServerInfo for the TransferQueueController
                - backend.storage_backend: Storage backend type (e.g., "SimpleStorage")
                - Other backend-specific configuration
            batch_size: Batch size for data loading per iter.
            micro_batch_size: Number of samples per micro-batch. This is the batch size
                that will be requested from TransferQueue for each iteration.
            data_fields: List of field names to retrieve from storage. Only these
                fields will be included in the returned batch.
            partition_id: Partition ID for data versioning. Different partitions can
                be used for different data versions or splits (e.g., "train", "val").
            task_name: Unique identifier for the training task. This is used to track
                which samples have been consumed by which task.
            dp_rank: The group ID of the current data group. All
                ranks with the same dp_rank will receive identical samples.
            fetch_batch_fn: Optional custom function to retrieve batch data.
                If None, uses default_fetch_batch_fn function.
            process_batch_fn: Optional custom function to post-process
                and split data into micro-batches. If None, uses chunk_batch_fn.

        Raises:
            ValueError: If input parameters are invalid.
        """

        if micro_batch_size < 1:
            raise ValueError(f"micro_batch_size must be >= 1, got {micro_batch_size}")

        if len(data_fields) < 1:
            raise ValueError(f"data_fields must be a list with at least one field name, got {data_fields}")

        if dp_rank < 0:
            raise ValueError(f"dp_rank {dp_rank} must be greater than or equal to 0")

        self.config = config
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.data_fields = data_fields
        self.partition_id = partition_id
        self.task_name = task_name
        self.dp_rank = dp_rank
        self.fetch_batch_fn = fetch_batch_fn if fetch_batch_fn else default_fetch_batch_fn
        self.process_batch_fn = process_batch_fn if process_batch_fn else chunk_batch_fn

        # Build sampling config for controller
        self.sampling_config = {
            "dp_rank": self.dp_rank,
            "task_name": self.task_name,
        }

        self._tq_client = None
        # Buffer for caching fetched batches (list of tuples (TensorDict, BatchMeta)).
        # Purpose:
        # 1) Cache full training batches retrieved from TransferQueue / storage to
        #    make logging, debugging and replaying batches easier.
        # 2) Support multi-pass training on the same samples in some scenarios —
        #    using `batch_index` to iterate over cached batches multiple times
        #    avoids re-fetching them from remote storage and reduces network/storage
        #    overhead.
        # 3) Work together with `reset()` / `step()` to manage iteration state cleanly
        #    and avoid dropping batches that haven't been consumed yet.
        self.buffer: list[tuple[TensorDict, BatchMeta]] = []
        self.batch_index = 0

        super().__init__()

    def _create_client(self):
        """Create and initialize a TransferQueue client.

        This method initializes the TransferQueueClient with the provided configuration
        and storage backend, and sets up the storage manager for data retrieval.

        Raises:
            ValueError: If controller_info or storage_backend is missing or invalid.
        """
        client_id = uuid.uuid4().hex[:8]

        # TODO: DEPRECATE in future
        controller_config = self.config.get("controller", None)
        if controller_config:
            controller_info = controller_config.get("zmq_info", None)
        else:
            controller_info = self.config.get("controller_info", None)
            if controller_info:
                warnings.warn(
                    "Config entry `controller_info` will be deprecated in 0.1.7, please "
                    "use `controller.zmq_info` instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )

        if not controller_info or not isinstance(controller_info, ZMQServerInfo):
            raise ValueError("Invalid or missing controller.zmq_info in config")

        backend_config = self.config.get("backend", None)
        if not backend_config:
            storage_backend = self.config.get("storage_backend", None)
            backend_config = self.config
            if storage_backend:
                warnings.warn(
                    "Config entry `storage_backend` will be deprecated in 0.1.7, please "
                    "use `backend.storage_backend` instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
        else:
            storage_backend = backend_config.get("storage_backend", None)
            backend_config = self.config.backend[storage_backend]

        if not storage_backend:
            raise ValueError("Missing storage_backend in config")

        self._tq_client = TransferQueueClient(client_id, controller_info)
        self._tq_client.initialize_storage_manager(manager_type=storage_backend, config=backend_config)

    def __iter__(self) -> Iterator[tuple[TensorDict, BatchMeta]]:
        """Iterate over the dataset, yielding batches of data.

        Yields:
            Tuple[TensorDict, BatchMeta]: A tuple containing:
                - TensorDict: Batch of data with the requested fields.
                - BatchMeta: Corresponding metadata to interact with TransferQueue.
        Note:
            This iterator runs indefinitely until the data source is exhausted.
            The caller should handle StopIteration when appropriate (e.g., when
            all data has been consumed and no more data will be produced).
        """
        if self._tq_client is None:
            self._create_client()

        assert self._tq_client is not None, "Failed to create TransferQueue client"

        # Note: For fully streamed production-consumption, please set the environment variable
        # TQ_PRE_ALLOC_SAMPLE_NUM to the required global_batch_size to make sure consumers can accurately
        # determine consumption status even before producers have generated the samples.
        while (
            not self._tq_client.check_consumption_status(self.task_name, self.partition_id)
            or self.batch_index <= len(self.buffer) - 1
        ):
            try:
                if self.batch_index <= len(self.buffer) - 1:
                    current_data = self.buffer[self.batch_index]
                    self.batch_index += 1
                    yield from self.process_batch_fn(*current_data, micro_batch_size=self.micro_batch_size)

                else:
                    batch_data, batch_meta = self.fetch_batch_fn(
                        self._tq_client,
                        self.data_fields,
                        self.batch_size,
                        self.partition_id,
                        self.task_name,
                        self.sampling_config,
                        self.batch_index,
                    )
                    if batch_data is not None:
                        self.buffer.append((batch_data, batch_meta))
                    else:
                        time.sleep(1)

            except Exception as e:
                logger.error(f"[StreamingDataset]: Error in data iteration: {e}")
                raise

    def reset(self):
        """Reset the dataset iterator to the beginning.

        Clears the buffer and resets the batch index for a fresh iteration.
        """
        self.batch_index = 0

    def step(self, partition_id):
        """Switch to a new partition and reset the dataset state.

        This method clears the buffer, resets the batch index, and updates the partition_id
        to fetch data from a different partition (e.g., switching from "train" to "val").

        Args:
            partition_id: The new partition ID to switch to.
        """
        self.buffer = []
        self.batch_index = 0
        self.partition_id = partition_id


def default_fetch_batch_fn(tq_client, data_fields, batch_size, partition_id, task_name, sampling_config, batch_index):
    """Retrieve a batch of data from TransferQueue.

    This function queries the TransferQueue controller for batch metadata and retrieves
    the actual data if available. It handles empty batches gracefully.

    Args:
        tq_client: The TransferQueueClient instance for data retrieval.
        data_fields: List of field names to retrieve from the batch.
        batch_size: The requested batch size.
        partition_id: The partition ID for data versioning.
        task_name: Unique identifier for the training task.
        sampling_config: Configuration dictionary for sampling strategy.
        batch_index: Current batch index for tracking consumption progress.

    Returns:
        tuple: A tuple containing:
            - batch: TensorDict with the retrieved data, or None if batch is empty.
            - batch_meta: BatchMeta object containing batch metadata.
    """
    # Get metadata from controller
    config = {**sampling_config, "batch_index": batch_index, "partition_id": partition_id}
    batch_meta = tq_client.get_meta(
        data_fields=data_fields,
        batch_size=batch_size,
        partition_id=partition_id,
        task_name=task_name,
        sampling_config=config,
    )

    # Check if we got valid data
    if batch_meta.size == 0:
        logger.debug(
            f"[StreamingDataset]: Received empty batch, waiting for more data... "
            f"Required batch_size={batch_size}, data_fields={data_fields},"
            f"partition_id={partition_id}, task_name={task_name}."
        )
        return None, batch_meta
    else:
        batch = tq_client.get_data(batch_meta)
        return batch, batch_meta


def chunk_batch_fn(td, batch_meta, micro_batch_size=1):
    """Split TensorDict into micro-batches along the batch dimension.

    This function chunks a TensorDict into smaller micro-batches with the specified size,
    along with corresponding metadata chunks. Handles cases where batch size is not
    evenly divisible by micro_batch_size.

    Args:
        td: Input TensorDict with non-empty batch_size.
        batch_meta: BatchMeta object to be chunked along with the TensorDict.
        micro_batch_size: Target size for each micro-batch (positive integer, default: 1).

    Returns:
        list: List of tuples (micro_batch_td, micro_batch_meta) where each tuple
              contains a TensorDict chunk and corresponding metadata chunk.

    Raises:
        TypeError: If td is not a TensorDict.
        ValueError: If micro_batch_size is not a positive integer, batch_size is empty,
                   or micro_batch_size exceeds total batch size.
    """
    if not isinstance(td, TensorDict):
        raise TypeError(f"Expected TensorDict, got {type(td).__name__}")

    if not isinstance(micro_batch_size, int) or micro_batch_size <= 0:
        raise ValueError(f"micro_batch_size must be a positive integer, got {micro_batch_size}")

    if len(td.batch_size) == 0:
        raise ValueError("Input TensorDict must have non-empty batch_size")

    total_size = td.batch_size[0]
    if micro_batch_size > total_size:
        raise ValueError(f"micro_batch_size ({micro_batch_size}) exceeds total batch size ({total_size})")

    # Calculate number of splits (handles uneven division)
    num_splits = (total_size + micro_batch_size - 1) // micro_batch_size
    splits = []
    batch_meta_list = batch_meta.chunk(num_splits)

    # Chunk the TensorDict and pair with corresponding metadata chunks
    for i in range(num_splits):
        start = i * micro_batch_size
        end = min(start + micro_batch_size, total_size)
        splits.append((td[start:end], batch_meta_list[i]))

    return splits
