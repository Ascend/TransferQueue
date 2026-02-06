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

from typing import Any

from transfer_queue.sampler import BaseSampler


class RankAwareSampler(BaseSampler):
    """Rank-aware sampler for distributed training with TransferQueue.

    This sampler is designed for distributed data parallel training scenarios
    where each rank retrieves data independently.

    This sampler guarantees that all ranks within the same data replica group receive
    the same sample indices.

    The sampler maintains inner state to coordinate sampling across ranks:

    - First rank in a data replica group to call :meth:`sample` performs actual sampling from
      ``ready_indexes`` and caches the result for other ranks in the same group
    - Subsequent ranks in the same group retrieve the cached indices.
    - If no cached indices are available, sampling is performed again and cached for others.


    Please refer to our roadmap for more details:
    [Roadmap] StreamingDataLoader for task-separated RL post-training
    https://github.com/Ascend/TransferQueue/issues/1
    """

    def __init__(self):
        """Initialize the RankAwareSampler.

        The sampler maintains internal state to coordinate sampling across ranks
        within the same data replica group. This state tracks which samples have been sampled
        and how many times they have been fetched.
        """

        super().__init__()

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        dp_rank: int,
        batch_index: int,
        task_name: str,
        partition_id: str,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Sample indices for the current rank, coordinating with other data replica ranks.

        This method implements coordinated sampling for distributed training.
        The first rank in each data replica group to call this method performs actual sampling
        from ``ready_indexes`` and caches the result. Subsequent ranks in the same
        data replica group receive the cached indices directly.

        Internal state structure (self._states):

        .. code-block:: python

            self._states = {
                "partition_id": {
                    "task_name": {
                        dp_rank: {
                            "batch_index": [sampled_indexes]
                        }
                    }
                }
            }

        State lifecycle:
        1. First rank samples from ``ready_indexes``, caches results for other ranks
        2. Other ranks pop and retrieve the cached indices

        Args:
            ready_indexes: List of global indices for which all required fields of the
                corresponding samples have been produced, and the samples are not labeled
                as consumed in the corresponding task.
            batch_size: Number of samples to select. If larger than available
                ready samples, no samples are returned and both lists are empty.
            dp_rank: Data parallel rank ID that this worker belongs to
                The same Ranks receive the same data samples.
            batch_index: Current batch index for tracking consumption progress.
            task_name: Identifier for the task.
            partition_id: Partition ID for data management.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of two lists:
            - List of sampled global indices. Typically, has length ``batch_size``,
              or empty if samples are insufficient.
            - List of global indices to mark as consumed (excluded from future
              retrieval by other data_replica_groups).

        Raises:
            ValueError: If ``data_replica_rank`` or ``data_replica_world_size`` is invalid.

        """

        if dp_rank < 0:
            raise ValueError(f"dp_rank {dp_rank} must be greater than or equal to 0")

        if partition_id not in self._states:
            self._states[partition_id] = {}

        if task_name not in self._states[partition_id]:
            self._states[partition_id][task_name] = {}

        if dp_rank not in self._states[partition_id][task_name]:
            self._states[partition_id][task_name][dp_rank] = {}

        if batch_index not in self._states[partition_id][task_name][dp_rank]:
            # Select first batch_size indices from ready_indexes
            sampled_indexes = ready_indexes[:batch_size]

            if len(sampled_indexes) < batch_size:
                return [], []

            consumed_indexes = sampled_indexes

            self._states[partition_id][task_name][dp_rank][batch_index] = sampled_indexes
        else:
            # Return the cached indices (identical to what first rank received)
            sampled_indexes = self._states[partition_id][task_name][dp_rank][batch_index]
            consumed_indexes = sampled_indexes

        return sampled_indexes, consumed_indexes
