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
from typing import Optional

import torch
from tensordict import TensorDict

from transfer_queue.dataloader.streaming_dataset import StreamingDataset
from transfer_queue.metadata import BatchMeta

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)


def _identity_collate_fn(data: tuple[TensorDict, BatchMeta]) -> tuple[TensorDict, BatchMeta]:
    """Identity collate function for TransferQueue.

    This function acts as a pass-through, preserving the `(TensorDict, BatchMeta)`
    structure yielded by `StreamingDataset`. It prevents PyTorch from attempting
    to stack or modify the already-batched data.
    """
    return data


class StreamingDataLoader(torch.utils.data.DataLoader):
    """StreamingDataLoader interface for TransferQueue.

    This DataLoader wraps StreamingDataset and provides a PyTorch DataLoader
    interface for distributed training with streaming data access.

    Key Features:
    - Compatible with PyTorch training loops (for loop iteration)
    - Works with StreamingDataset for streaming data access
    - Supports distributed training via RankAwareSampler coordination


    Note:
        This DataLoader is typically used with StreamingDataset which manages
        batch size internally. The standard PyTorch DataLoader batch_size
        parameter is set to None because batching is handled by the dataset
        in coordination with TransferQueue's sampling logic.

    Example:
        >>> dataset = StreamingDataset(
        ...     config=config,
        ...     micro_batch_size=4,
        ...     required_fields=["input_ids", "attention_mask"],
        ...     partition_id="train",
        ...     task_name="update_actor",
        ...     data_replica_group=0,
        ...     data_replica_rank=0,
        ...     data_replica_world_size=1,
        ... )
        >>> dataloader = StreamingDataLoader(dataset, num_workers=0)
        >>> for batch, batch_meta in dataloader:
        ...     # batch: TensorDict with requested fields
        ...     # batch_meta: Metadata for TransferQueue coordination
        ...     loss = model(batch)
        ...     loss.backward()
    """

    def __init__(
        self,
        dataset: StreamingDataset,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
        worker_init_fn=None,
        multiprocessing_context=None,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        """Initialize the StreamingDataLoader.

        Args:
            dataset: StreamingDataset instance.
            num_workers: Number of subprocesses for data loading.
            collate_fn: Function to collate samples into batches.
            pin_memory: If True, pin memory for GPU transfer.
            worker_init_fn: Worker initialization function.
            multiprocessing_context: Multiprocessing context.
            prefetch_factor: Number of batches to prefetch per worker.
            persistent_workers: Keep workers alive between epochs.
            pin_memory_device: Device for pin_memory.

        Note:
            This DataLoader is designed to work with StreamingDataset which handles
            batch size internally via the micro_batch_size parameter. The batch_size
            parameter in PyTorch DataLoader is set to None because batching is managed
            by the StreamingDataset in coordination with RankAwareSampler.
        """
        self.dataset: StreamingDataset = dataset

        if collate_fn is None:
            # use identical collate function to directly return the self-defined
            # [TensorDict, BatchMeta] output of StreamingDataset
            final_collate_fn = _identity_collate_fn
        else:
            final_collate_fn = collate_fn

        super().__init__(
            dataset=dataset,
            batch_size=None,  # Batch size is handled by the dataset
            shuffle=None,
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=final_collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            timeout=0,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )

    def reset(self):
        """Reset the dataset iterator to the beginning.

        Clears the buffer and resets the batch index for a fresh iteration.
        """
        self.dataset.reset()

    def step(self, partition_id):
        """Switch to a new partition and reset the dataset state.

        This method clears the buffer, resets the batch index, and updates the partition_id
        to fetch data from a different partition (e.g., switching from "train" to "val").

        Args:
            partition_id: The new partition ID to switch to.
        """
        self.dataset.step(partition_id)

    def get_buffer(self):
        """Get the current buffer from the underlying dataset.

        Returns the batch buffer maintained by StreamingDataset, which stores
        pre-fetched batches for efficient data access.

        Returns:
            list: Buffer containing pre-fetched (TensorDict, BatchMeta) tuples.
        """
        return self.dataset.buffer
