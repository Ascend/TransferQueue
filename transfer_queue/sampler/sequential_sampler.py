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


class SequentialSampler(BaseSampler):
    """Sequential sampler for basic data consumption patterns.

    This sampler implements sequential sampling without replacement, selecting samples
    from the beginning of the ready_indexes list in order. It's the default sampling
    strategy for TransferQueueController and provides simple, deterministic data consumption
    with minimal overhead.

    The sampler is ideal for standard supervised learning scenarios, data preprocessing
    pipelines, and any use case where ordered, predictable data consumption is preferred.
    It ensures each sample is consumed exactly once, maintaining a clean progression through
    the available data.

    This sampler is typically used as the default sampler in TransferQueueController:

    ```python
    # Default usage (SequentialSampler is the default)
    controller = TransferQueueController.remote()
    # or explicitly:
    controller = TransferQueueController.remote(sampler=SequentialSampler)
    ```
    """

    def __init__(
        self,
    ):
        """Initialize the SequentialSampler.

        SequentialSampler requires no initialization parameters and maintains
        minimal internal state for optimal performance.
        """
        super().__init__()

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        """Select first batch_size elements from ready_indexes.

        Args:
            ready_indexes: Available sample indices.
            batch_size: Number of samples to select. If larger than available ready samples,
                all available samples will be returned.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple of (sampled_indexes, consumed_indexes), where consumed_indexes = sampled_indexes.
        """
        sampled_indexes = ready_indexes[:batch_size]
        consumed_indexes = sampled_indexes

        return sampled_indexes, consumed_indexes
