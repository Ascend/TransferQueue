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
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    action="ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage*",
    category=UserWarning,
    module=r"torch\.nested",
)

warnings.filterwarnings(
    action="ignore",
    message=r"Tip: In future versions of Ray, Ray will no longer override accelerator visible "
    r"devices env var if num_gpus=0 or num_gpus=None.*",
    category=FutureWarning,
    module=r"ray\._private\.worker",
)

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402
from transfer_queue.sampler import BaseSampler  # noqa: E402


class RandomSamplerWithReplacement(BaseSampler):
    """
    Sampler 1: Random Sampling with Replacement

    Samples data randomly with replacement.
    Useful when you want to revisit samples multiple times.
    """

    def __init__(self, seed: int = None):
        super().__init__()
        self.seed = seed
        self._states["rng"] = np.random.RandomState(seed)

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        rng = self._states["rng"]

        if len(ready_indexes) < batch_size:
            raise ValueError("Not enough ready indexes to sample from.")

        # Do sample
        sampled_indexes = rng.choice(ready_indexes, size=batch_size, replace=False).tolist()

        # Label no consumption since samples can be reused
        consumed_indexes = []

        return sampled_indexes, consumed_indexes


class RandomSamplerWithoutReplacement(BaseSampler):
    """
    Sampler 2: Random Sampling without Replacement

    Samples data randomly without replacement.
    Useful for training without data ordering bias.
    """

    def __init__(self, seed: int = None):
        super().__init__()
        self.seed = seed
        self._states["rng"] = np.random.RandomState(seed)

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        rng = self._states["rng"]

        if len(ready_indexes) < batch_size:
            raise ValueError("Not enough ready indexes to sample from.")

        # Do sample
        sampled_indexes = rng.choice(ready_indexes, size=batch_size, replace=False).tolist()

        # Consumed indexes are same as sampled
        consumed_indexes = sampled_indexes.copy()

        return sampled_indexes, consumed_indexes


class PrioritySampler(BaseSampler):
    """
    Sampler 3: Priority Sampling

    Samples based on priority scores (e.g., loss, uncertainty, etc.).
    Priority can be longer than ready_indexes - use partial sampling.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def sample(
        self,
        ready_indexes: list[int],
        batch_size: int,
        priority_scores: np.ndarray = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[int], list[int]]:
        if len(ready_indexes) < batch_size:
            raise ValueError("Not enough ready indexes to sample from.")

        if priority_scores is None:
            priority_scores = np.ones(len(ready_indexes), dtype=float)
        elif len(priority_scores) > len(ready_indexes):
            # Priority longer than ready_indexes - use partial
            priority_scores = priority_scores[ready_indexes]

        # Convert scores to probabilities
        priority_scores = priority_scores / priority_scores.sum()

        # Sample without replacement
        sampled_indexes = np.random.choice(
            ready_indexes, size=min(batch_size, len(ready_indexes)), replace=False, p=priority_scores
        ).tolist()

        consumed_indexes = sampled_indexes.copy()
        return sampled_indexes, consumed_indexes


def setup_transfer_queue_with_sampler(sampler):
    """Setup TransferQueue with custom sampler."""
    if not ray.is_initialized():
        ray.init(namespace="TransferQueueTutorial")

    config = OmegaConf.create(
        {"controller": {"sampler": sampler}, "backend": {"SimpleStorage": {"num_data_storage_units": 2}}},
        flags={"allow_objects": True},
    )

    tq.init(config)


def demonstrate_random_sampler_with_replacement():
    print("\n" + "=" * 80)
    print("Example 1: Use Customized RandomSamplerWithReplacement in TransferQueue")
    print("=" * 80)

    print("\nSetup TransferQueue with RandomSamplerWithReplacement...")

    sampler = RandomSamplerWithReplacement()
    setup_transfer_queue_with_sampler(sampler)

    # Add 5 samples
    print("\n[Step 1] Adding 5 samples...")
    data = TensorDict(
        {
            "input": torch.tensor([[i] for i in range(5)]),
        },
        batch_size=5,
    )
    tq.put(data=data, partition_id="test")
    print("  ✓ 5 samples added")

    # Get batch 1 (should get 2 random samples)
    print("\n[Step 2] Get batch 1 (2 samples)...")
    meta1 = tq.get_meta(data_fields=["input"], batch_size=2, partition_id="test", task_name="demo_task")
    print(f"  ✓ Got samples: {meta1.global_indexes}")

    # Get batch 2 (should get 1 random sample with replacement - may have duplicate with previous batch!)
    print("\n[Step 3] Get batch 2 (1 sample)...")
    meta2 = tq.get_meta(data_fields=["input"], batch_size=1, partition_id="test", task_name="demo_task")
    print(f"  ✓ Got samples: {meta2.global_indexes}")

    # Get batch 3 (should get 2 random samples with replacement - may have duplicate with previous batches!)
    print("\n[Step 4] Get batch 3 (2 samples)...")
    meta3 = tq.get_meta(data_fields=["input"], batch_size=2, partition_id="test", task_name="demo_task")
    print(f"  ✓ Got samples: {meta3.global_indexes}")

    print("\n[Verification]")
    print("  ✓ With replacement: Same sample can appear multiple times")
    print("  ✓ Check: Are there duplicates in the batches?")
    all_sampled = meta1.global_indexes + meta2.global_indexes + meta3.global_indexes
    print(f"  ✓ All sampled: {all_sampled}")

    # Cleanup
    tq.clear_partition(partition_id="test")
    tq.close()
    ray.shutdown()


def demonstrate_random_sampler_without_replacement():
    print("\n" + "=" * 80)
    print("Example 2: Use Customized RandomSamplerWithoutReplacement in TransferQueue")
    print("=" * 80)

    print("\nSetup TransferQueue with RandomSamplerWithoutReplacement...")

    sampler = RandomSamplerWithoutReplacement()
    setup_transfer_queue_with_sampler(sampler)

    # Add 6 samples
    print("\n[Step 1] Adding 6 samples...")
    data = TensorDict(
        {
            "input": torch.tensor([[i] for i in range(6)]),
        },
        batch_size=6,
    )
    tq.put(data=data, partition_id="test")
    print("  ✓ 6 samples added")

    # Get batch 1 (should get 3 random samples without replacement)
    print("\n[Step 2] Get batch 1 (3 samples)...")
    meta1 = tq.get_meta(data_fields=["input"], batch_size=3, partition_id="test", task_name="demo_task")
    print(f"  ✓ Got samples: {meta1.global_indexes}")

    # Get batch 2 (should randomly get 1 sample that are different from previous batch)
    print("\n[Step 3] Get batch 2 (1 samples)...")
    meta2 = tq.get_meta(data_fields=["input"], batch_size=1, partition_id="test", task_name="demo_task")
    print(f"  ✓ Got samples: {meta2.global_indexes}")

    # Get batch 3 (should randomly get 2 samples that are different from previous batch)
    print("\n[Step 4] Get batch 3 (2 samples)...")
    meta3 = tq.get_meta(data_fields=["input"], batch_size=2, partition_id="test", task_name="demo_task")
    print(f"  ✓ Got samples: {meta3.global_indexes}")

    print("\n[Verification]")
    print("  ✓ Without replacement: Each sample consumed only once")
    print(f"  ✓ Batch 1: {meta1.global_indexes}")
    print(f"  ✓ Batch 2: {meta2.global_indexes}")
    print(f"  ✓ Batch 3: {meta3.global_indexes} (none left)")

    # Cleanup
    tq.clear_partition(partition_id="test")
    tq.close()
    ray.shutdown()


def demonstrate_priority_sampler():
    print("\n" + "=" * 80)
    print("Example 3: Use Customized PrioritySampler in TransferQueue")
    print("=" * 80)

    print("\nSetup TransferQueue with PrioritySampler...")

    sampler = PrioritySampler()
    setup_transfer_queue_with_sampler(sampler)

    # Add 8 samples
    print("\n[Step 1] Adding 8 samples...")
    data = TensorDict(
        {
            "input": torch.tensor([[i] for i in range(8)]),
        },
        batch_size=8,
    )
    tq.put(data=data, partition_id="test")
    print("  ✓ 8 samples added")

    time.sleep(1)

    # Priority scores (higher = more important)
    # Index 2, 7, 3 have highest priority
    priority_scores = np.array([0.01, 0.01, 88, 999, 0.01, 0.01, 0.01, 10])

    print("\n[Step 2] Get batch with priority (1 sample)...")
    print(f"Priority scores: {priority_scores}")

    # Get batch using priority sampling
    meta1 = tq.get_meta(
        data_fields=["input"],
        batch_size=1,
        partition_id="test",
        task_name="demo_task",
        sampling_config={"priority_scores": priority_scores},
    )
    print(f"  ✓ Got samples (high priority): {meta1.global_indexes}")
    print(f"  ✓ Priority of sampled: {[priority_scores[i] for i in meta1.global_indexes]}")

    # Get another batch
    print("\n[Step 3] Get another batch (2 samples)...")
    meta2 = tq.get_meta(
        data_fields=["input"],
        batch_size=2,
        partition_id="test",
        task_name="demo_task",
        sampling_config={"priority_scores": priority_scores},
    )
    print(f"  ✓ Got samples: {meta2.global_indexes}")
    print(f"  ✓ Priority of sampled: {[priority_scores[i] for i in meta2.global_indexes]}")

    print("\n[Verification]")
    print("  ✓ Priority sampling: Higher priority samples more likely to be chosen")
    print(f"  ✓ Batch 1 high-priority indices: {[i for i in meta1.global_indexes if priority_scores[i] >= 0.1]}")
    print(f"  ✓ Batch 2 high-priority indices: {[i for i in meta2.global_indexes if priority_scores[i] >= 0.1]}")

    # Cleanup
    tq.clear_partition(partition_id="test")
    tq.close()
    ray.shutdown()


def main():
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 4: Custom Sampler Development

        This script demonstrates how to develop custom samplers for TransferQueue.
        Samplers control HOW data is consumed from the queue.

        Core Interface:
        - BaseSampler.sample(ready_indexes, batch_size, *args, **kwargs)
        - Returns: (sampled_indexes, consumed_indexes)
        - sampled_indexes has length = batch_size; consumed_indexes may be empty or have a different length

        Key Concepts:
        - ready_indexes: Samples ready for consumption (all fields produced & has not been consumed by the task)
        - sampled_indexes: Which samples to return in this batch
        - consumed_indexes: Which samples to mark as consumed (never returned to this task again)
        """
        )
    )
    print("=" * 80)

    try:
        demonstrate_random_sampler_with_replacement()
        demonstrate_random_sampler_without_replacement()
        demonstrate_priority_sampler()

        print("\n" + "=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("Key Takeaways:")
        print("1. Samplers control HOW data is consumed from TransferQueue")
        print("2. Implement BaseSampler.sample(ready_indexes, batch_size, *args, **kwargs)")
        print("3. Return (sampled_indexes, consumed_indexes)")
        print("4. Initialize controller with custom sampler: TransferQueueController.remote(sampler=YourSampler())")
        print("5. Pass parameters via sampling_config in get_meta calls")

    except Exception as e:
        print(f"Error during tutorial: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
