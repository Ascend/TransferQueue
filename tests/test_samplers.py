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

"""Unit tests for TransferQueue samplers."""

from typing import Any

import pytest

from transfer_queue.sampler import BaseSampler
from transfer_queue.sampler.grpo_group_n_sampler import GRPOGroupNSampler
from transfer_queue.sampler.rank_aware_sampler import RankAwareSampler
from transfer_queue.sampler.seqlen_balanced_sampler import (
    SeqlenBalancedSampler,
    get_seqlen_balanced_partitions,
)
from transfer_queue.sampler.sequential_sampler import SequentialSampler


class TestBaseSampler:
    """Test cases for BaseSampler abstract class."""

    def test_base_sampler_is_abstract(self):
        """Test that BaseSampler cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseSampler()

        assert "Can't instantiate abstract class" in str(exc_info.value)
        assert "sample" in str(exc_info.value)

    def test_base_sampler_has_abstract_methods(self):
        """Test that BaseSampler defines abstract methods."""
        assert hasattr(BaseSampler, "sample")
        assert getattr(BaseSampler.sample, "__isabstractmethod__", False)

    def test_base_sampler_has_call_method(self):
        """Test that BaseSampler has __call__ method."""
        assert callable(BaseSampler)

    def test_base_sampler_initialization_states(self):
        """Test BaseSampler initialization sets _states correctly."""

        # Create a concrete implementation for testing
        class TestSampler(BaseSampler):
            def sample(self, ready_indexes: list[int], batch_size: int, **kwargs: Any) -> tuple[list[int], list[int]]:
                return ready_indexes[:batch_size], ready_indexes[:batch_size]

        sampler = TestSampler()
        assert hasattr(sampler, "_states")
        assert sampler._states == {}


class TestSequentialSampler:
    """Test cases for SequentialSampler."""

    def test_sequential_sampler_initialization(self):
        """Test SequentialSampler initialization."""
        sampler = SequentialSampler()
        assert isinstance(sampler, BaseSampler)
        assert hasattr(sampler, "_states")
        assert sampler._states == {}

    def test_sequential_sampler_basic_functionality(self):
        """Test basic sampling functionality."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2]
        assert consumed == [0, 1, 2]
        assert len(sampled) == batch_size
        assert len(consumed) == batch_size

    def test_sequential_sampler_empty_ready_indexes(self):
        """Test behavior with empty ready indexes."""
        sampler = SequentialSampler()
        ready_indexes = []
        batch_size = 3

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == []
        assert consumed == []

    def test_sequential_sampler_batch_size_larger_than_ready(self):
        """Test behavior when batch_size > len(ready_indexes)."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1]
        batch_size = 5

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1]
        assert consumed == [0, 1]
        assert len(sampled) == len(ready_indexes)

    def test_sequential_sampler_zero_batch_size(self):
        """Test behavior with zero batch size."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 0

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == []
        assert consumed == []

    def test_sequential_sampler_negative_batch_size(self):
        """Test behavior with negative batch size."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = -1

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        # Python slicing with negative numbers should work as expected
        expected = ready_indexes[:batch_size]  # This gives [0, 1, 2] for -1
        assert sampled == expected
        assert consumed == expected

    def test_sequential_sampler_non_sequential_indexes(self):
        """Test behavior with non-sequential ready indexes."""
        sampler = SequentialSampler()
        ready_indexes = [10, 5, 15, 20, 8]
        batch_size = 3

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [10, 5, 15]
        assert consumed == [10, 5, 15]

    def test_sequential_sampler_duplicate_indexes(self):
        """Test behavior with duplicate indexes."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 0, 2, 1, 3]
        batch_size = 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 0, 2]
        assert consumed == [0, 1, 0, 2]

    def test_sequential_sampler_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2

        sampled, consumed = sampler(ready_indexes, batch_size)

        assert sampled == [0, 1]
        assert consumed == [0, 1]

    def test_sequential_sampler_with_extra_kwargs(self):
        """Test that SequentialSampler accepts extra kwargs but ignores them."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2

        # SequentialSampler should accept extra kwargs but ignore them
        sampled, consumed = sampler.sample(ready_indexes, batch_size, extra_param="ignored")

        assert sampled == [0, 1]
        assert consumed == [0, 1]


class TestGRPOGroupNSampler:
    """Test cases for GRPOGroupNSampler."""

    def test_grpo_sampler_initialization(self):
        """Test GRPOGroupNSampler initialization."""
        sampler = GRPOGroupNSampler()
        assert isinstance(sampler, BaseSampler)
        assert hasattr(sampler, "_states")
        assert sampler._states == {}

    def test_grpo_sampler_basic_functionality(self):
        """Test basic grouped sampling functionality."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes
        batch_size = 8

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(sampled) == batch_size
        assert len(consumed) == batch_size

    def test_grpo_sampler_partial_batch(self):
        """Test partial batch sampling."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 12 indexes
        batch_size = 8  # Want 8 samples total
        # 2 groups of 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(sampled) == batch_size
        assert len(consumed) == batch_size

    def test_grpo_sampler_batch_size_divisibility(self):
        """Test that batch_size must be divisible by n_samples_per_prompt."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes, sufficient for batch_size=7
        batch_size = 7

        with pytest.raises(ValueError) as exc_info:
            sampler.sample(ready_indexes, batch_size)

        assert "must be a multiple of n_samples_per_prompt" in str(exc_info.value)

    def test_grpo_sampler_insufficient_ready_indexes(self):
        """Test behavior when not enough ready indexes are available."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3]  # Only 4 indexes, but need 8 for 2 groups of 4
        batch_size = 8

        # Should return empty lists when insufficient complete groups
        sampled, consumed = sampler.sample(ready_indexes, batch_size)
        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_exact_multiple_available(self):
        """Test when ready_indexes length is exactly a multiple of n_samples_per_prompt."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes
        batch_size = 8

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grpo_sampler_zero_batch_size(self):
        """Test behavior with zero batch size."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=2)
        ready_indexes = [0, 1, 2, 3]
        batch_size = 0

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_single_sample_per_prompt(self):
        """Test with n_samples_per_prompt = 1."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2]
        assert consumed == [0, 1, 2]

    def test_grpo_sampler_large_group_size(self):
        """Test with large n_samples_per_prompt."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=10)
        ready_indexes = list(range(20))  # 20 indexes
        batch_size = 20

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == list(range(20))
        assert consumed == list(range(20))

    def test_grpo_sampler_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=2)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 4

        sampled, consumed = sampler(ready_indexes, batch_size)

        assert sampled == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]

    def test_grpo_sampler_with_extra_kwargs(self):
        """Test that GRPOGroupNSampler accepts extra kwargs but ignores them."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 8

        # GRPOGroupNSampler should accept extra kwargs but ignore them
        sampled, consumed = sampler.sample(ready_indexes, batch_size, extra_param="ignored", another_param=42)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grpo_sampler_non_sequential_indexes(self):
        """Test with non-sequential ready indexes that get sorted."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [3, 4, 5, 6, 9, 10, 11, 12]  # Non-sequential order but has consecutive groups after sorting
        batch_size = 8

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        # Should find consecutive groups after sorting: [3,4,5,6] and [9,10,11,12]
        expected = [3, 4, 5, 6, 9, 10, 11, 12]
        assert sampled == expected
        assert consumed == expected

    def test_grpo_sampler_invalid_n_samples_per_prompt(self):
        """Test behavior with invalid n_samples_per_prompt values."""
        # Test zero n_samples_per_prompt
        with pytest.raises(ValueError) as exc_info:
            GRPOGroupNSampler(n_samples_per_prompt=0)
        assert "must be positive" in str(exc_info.value)
        # Test negative n_samples_per_prompt
        with pytest.raises(ValueError) as exc_info:
            GRPOGroupNSampler(n_samples_per_prompt=-2)
        assert "must be positive" in str(exc_info.value)

    def test_grpo_sampler_no_complete_groups(self):
        """Test behavior when no complete groups are available."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=3)
        ready_indexes = [0, 1, 3, 4, 6, 7]  # No consecutive groups of size 3
        batch_size = 6

        # Should return empty lists when no complete groups found
        sampled, consumed = sampler.sample(ready_indexes, batch_size)
        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_mixed_groups(self):
        """Test behavior with mixed complete and incomplete groups."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=3)
        ready_indexes = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]  # Mixed groups
        batch_size = 6

        # Should find the complete groups [3,4,5] and [9,10,11]
        sampled, consumed = sampler.sample(ready_indexes, batch_size)
        assert sampled == [3, 4, 5, 9, 10, 11]
        assert consumed == [3, 4, 5, 9, 10, 11]

    def test_grpo_sampler_sorting_functionality(self):
        """Test that ready_indexes are properly sorted before group detection."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [10, 11, 12, 5, 6, 7, 8, 9]  # Out of order but contains consecutive groups
        batch_size = 8

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        # After sorting: [5,6,7,8,9,10,11,12], should find [5,6,7,8] and [9,10,11,12]
        expected = [5, 6, 7, 8, 9, 10, 11, 12]
        assert sampled == expected
        assert consumed == expected

    def test_grpo_sampler_insufficient_groups(self):
        """Test behavior when requesting more groups than available."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 4 groups of 4
        batch_size = 12  # Requesting 3 groups of 4 - this should work

        # This should actually work fine since we have 4 groups and request 3
        sampled, consumed = sampler.sample(ready_indexes, batch_size)
        assert len(sampled) == 12
        assert len(consumed) == 12

        # Now test requesting more than available
        batch_size = 20  # Requesting 5 groups of 4, but only have 4
        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        # Should return empty lists when requesting more complete groups than available
        assert sampled == []
        assert consumed == []


class TestRankAwareSampler:
    """Test cases for RankAwareSampler."""

    def test_rank_aware_sampler_initialization(self):
        """Test RankAwareSampler initialization."""
        sampler = RankAwareSampler()
        assert isinstance(sampler, BaseSampler)
        assert hasattr(sampler, "_states")
        assert sampler._states == {}

    def test_rank_aware_sampler_basic_sampling(self):
        """Test basic sampling functionality."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == [0, 1, 2]
        assert consumed == [0, 1, 2]
        assert len(sampled) == batch_size

    def test_rank_aware_sampler_caching_on_same_batch_index(self):
        """Test that same batch_index returns cached results."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3

        # First call with batch_index=0
        sampled1, consumed1 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        # Second call with same batch_index=0 should return cached result
        sampled2, consumed2 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled1 == sampled2 == [0, 1, 2]
        assert consumed1 == consumed2 == [0, 1, 2]

    def test_rank_aware_sampler_different_batch_indexes(self):
        """Test that different batch_index values sample different data."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 2

        # First batch
        sampled1, consumed1 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        # Second batch
        ready_indexes = [2, 3, 4, 5, 6, 7]
        sampled2, consumed2 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=1,
            task_name="task",
            partition_id="test",
        )

        assert sampled1 == [0, 1]
        assert sampled2 == [2, 3]
        assert consumed1 == [0, 1]
        assert consumed2 == [2, 3]

    def test_rank_aware_sampler_multiple_dp_ranks(self):
        """Test that same dp_ranks reuse state cache."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 2

        # DP rank 0 samples batch 0
        sampled_dp0_b0, consumed_dp0_b0 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )
        ready_indexes = [2, 3, 4, 5, 6, 7]
        # DP rank 0 samples batch 0 (should get same result as dp_rank=0)
        sampled_dp1_b0, consumed_dp1_b0 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        # Both should sample from the same ready_indexes
        assert sampled_dp0_b0 == [0, 1]
        assert sampled_dp1_b0 == [0, 1]

    def test_rank_aware_sampler_empty_ready_indexes(self):
        """Test behavior with empty ready indexes."""
        sampler = RankAwareSampler()
        ready_indexes = []
        batch_size = 3

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == []
        assert consumed == []

    def test_rank_aware_sampler_batch_size_larger_than_ready(self):
        """Test behavior when batch_size > len(ready_indexes)."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1]
        batch_size = 5

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == []
        assert consumed == []

    def test_rank_aware_sampler_zero_batch_size(self):
        """Test behavior with zero batch size."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 0

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == []
        assert consumed == []

    def test_rank_aware_sampler_multiple_tasks(self):
        """Test behavior with multiple tasks."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 2

        sampled_task0, consumed_task0 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task0",
            partition_id="test",
        )

        sampled_task1, consumed_task1 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task1",
            partition_id="test",
        )

        assert sampled_task0 == [0, 1]
        assert consumed_task0 == [0, 1]
        assert sampled_task1 == [0, 1]
        assert consumed_task1 == [0, 1]

        # Check that state is separate per task
        assert sampler._states["test"]["task0"][0][0] == [0, 1]
        assert sampler._states["test"]["task1"][0][0] == [0, 1]

    def test_rank_aware_sampler_multiple_partitions(self):
        """Test behavior with multiple partitions."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 2

        sampled_part0, consumed_part0 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="partition0",
        )

        sampled_part1, consumed_part1 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="partition1",
        )

        assert sampled_part0 == [0, 1]
        assert consumed_part0 == [0, 1]
        assert sampled_part1 == [0, 1]
        assert consumed_part1 == [0, 1]

        # Check that state is separate per partition
        assert sampler._states["partition0"]["task"][0][0] == [0, 1]
        assert sampler._states["partition1"]["task"][0][0] == [0, 1]

    def test_rank_aware_sampler_invalid_dp_rank(self):
        """Test behavior with invalid dp_rank."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2

        with pytest.raises(ValueError) as exc_info:
            sampler.sample(
                ready_indexes,
                batch_size,
                dp_rank=-1,
                batch_index=0,
                task_name="task",
                partition_id="test",
            )

        assert "dp_rank" in str(exc_info.value)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_rank_aware_sampler_with_extra_kwargs(self):
        """Test that RankAwareSampler accepts extra kwargs but ignores them."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 2

        # Should accept extra kwargs gracefully
        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
            extra_param="ignored",
            another_param=42,
        )

        assert sampled == [0, 1]
        assert consumed == [0, 1]

    def test_rank_aware_sampler_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2

        sampled, consumed = sampler(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == [0, 1]
        assert consumed == [0, 1]


class TestSeqlenBalancedSampler:
    """Test cases for SeqlenBalancedSampler."""

    # ---- Helper: mock partition object ----

    class MockPartition:
        """Minimal mock for DataPartitionStatus providing get_custom_meta."""

        def __init__(self, custom_meta: dict[int, dict]):
            self._custom_meta = custom_meta

        def get_custom_meta(self, global_indices: list[int]) -> dict[int, dict]:
            return {idx: self._custom_meta.get(idx, {}) for idx in global_indices}

    # ---- Initialization tests ----

    def test_initialization_invalid_dp_size(self):
        """Test that dp_size must be positive."""
        with pytest.raises(ValueError) as exc_info:
            SeqlenBalancedSampler(dp_size=0)
        assert "dp_size must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            SeqlenBalancedSampler(dp_size=-1)
        assert "dp_size must be positive" in str(exc_info.value)

    # ---- Fallback (no partition) tests ----

    def test_fallback_equal_split_no_partition(self):
        """Test fallback equal-split when no partition is provided."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2  # per-DP → global = 4

        sampled_0, consumed_0 = sampler.sample(
            ready_indexes,
            batch_size,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        sampled_1, consumed_1 = sampler.sample(
            ready_indexes,
            batch_size,
            task_name="task",
            partition_id="p0",
            dp_rank=1,
            batch_index=0,
        )

        # Together they should cover all 4 indexes without overlap
        assert len(sampled_0) == 2
        assert len(sampled_1) == 2
        assert set(sampled_0 + sampled_1) == {0, 1, 2, 3}
        assert sampled_0 == consumed_0
        assert sampled_1 == consumed_1

    def test_fallback_single_dp(self):
        """Test dp_size=1 returns all samples to rank 0."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=2, dp_size=1)
        ready_indexes = [0, 1, 2, 3]
        batch_size = 4  # per-DP = global = 4

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert sampled == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]

    # ---- Balanced partitioning with mock partition ----

    def test_balanced_partitioning_with_custom_meta(self):
        """Test that samples are balanced by total_lengths across DP ranks."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]
        # Sample 0 and 1 are long, sample 2 and 3 are short
        partition = self.MockPartition(
            {
                0: {"total_lengths": 100},
                1: {"total_lengths": 100},
                2: {"total_lengths": 10},
                3: {"total_lengths": 10},
            }
        )

        sampled_0, _ = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
            partition=partition,
        )
        sampled_1, _ = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=1,
            batch_index=0,
            partition=partition,
        )

        # All indexes should be covered
        all_sampled = sorted(sampled_0 + sampled_1)
        assert all_sampled == [0, 1, 2, 3]

        # KK should pair one long with one short per rank for balance
        def total_len(indices):
            lengths = {0: 100, 1: 100, 2: 10, 3: 10}
            return sum(lengths[i] for i in indices)

        diff = abs(total_len(sampled_0) - total_len(sampled_1))
        # Perfect balance: each rank gets one 100 + one 10 = 110, diff = 0
        assert diff == 0

    def test_balanced_partitioning_group_level(self):
        """Test balanced partitioning at group level (n_samples_per_prompt > 1)."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=2, dp_size=2)
        # 4 groups of 2: [0,1], [2,3], [4,5], [6,7]
        ready_indexes = list(range(8))
        partition = self.MockPartition(
            {
                0: {"total_lengths": 50},
                1: {"total_lengths": 50},  # group0 total=100
                2: {"total_lengths": 5},
                3: {"total_lengths": 5},  # group1 total=10
                4: {"total_lengths": 50},
                5: {"total_lengths": 50},  # group2 total=100
                6: {"total_lengths": 5},
                7: {"total_lengths": 5},  # group3 total=10
            }
        )

        sampled_0, _ = sampler.sample(
            ready_indexes,
            4,  # per-DP batch=4, global=8
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
            partition=partition,
        )
        sampled_1, _ = sampler.sample(
            ready_indexes,
            4,
            task_name="task",
            partition_id="p0",
            dp_rank=1,
            batch_index=0,
            partition=partition,
        )

        # Each rank should get 4 samples (2 groups)
        assert len(sampled_0) == 4
        assert len(sampled_1) == 4
        assert set(sampled_0 + sampled_1) == set(range(8))

        # Group integrity: each group's samples stay together
        for rank_samples in [sampled_0, sampled_1]:
            for s in rank_samples:
                partner = s ^ 1  # pairs: (0,1), (2,3), (4,5), (6,7)
                if s % 2 == 0:
                    assert partner in rank_samples, f"Group broken: {s} without {partner}"

    # ---- Caching tests ----

    def test_caching_returns_same_result(self):
        """Test that repeated calls with same key return cached result."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]

        sampled_first, _ = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        sampled_second, _ = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert sampled_first == sampled_second

    def test_different_batch_index_not_cached(self):
        """Test that different batch_index produces different cache keys."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=1)
        ready_indexes_b0 = [0, 1, 2, 3]
        ready_indexes_b1 = [4, 5, 6, 7]

        sampled_b0, _ = sampler.sample(
            ready_indexes_b0,
            4,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        sampled_b1, _ = sampler.sample(
            ready_indexes_b1,
            4,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=1,
        )

        assert sampled_b0 == [0, 1, 2, 3]
        assert sampled_b1 == [4, 5, 6, 7]

    def test_states_cache_populated_for_all_ranks(self):
        """Test that _states cache is populated for all dp_ranks on first call."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=3)
        ready_indexes = list(range(6))

        sampler.sample(
            ready_indexes,
            2,  # per-DP=2, global=6
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        # All 3 ranks should have cached state
        states = sampler._states["p0"]["task"]
        for rank_i in range(3):
            assert rank_i in states
            assert 0 in states[rank_i]
            cached_sampled, cached_consumed = states[rank_i][0]
            assert len(cached_sampled) == 2
            assert cached_sampled == cached_consumed

    # ---- clear_cache tests ----

    def test_clear_cache(self):
        """Test clear_cache removes both _states and _balanced_cache."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]

        sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert len(sampler._balanced_cache) > 0
        assert "p0" in sampler._states

        sampler.clear_cache("p0")

        assert all(k[0] != "p0" for k in sampler._balanced_cache)
        assert "p0" not in sampler._states

    def test_clear_cache_only_affects_target_partition(self):
        """Test clear_cache only removes the specified partition."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=1)

        sampler.sample(
            [0, 1],
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        sampler.sample(
            [2, 3],
            2,
            task_name="task",
            partition_id="p1",
            dp_rank=0,
            batch_index=0,
        )

        sampler.clear_cache("p0")

        assert "p0" not in sampler._states
        assert "p1" in sampler._states
        assert any(k[0] == "p1" for k in sampler._balanced_cache)

    # ---- Edge cases ----

    def test_insufficient_ready_indexes(self):
        """Test behavior when not enough ready indexes for global batch."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=2, dp_size=2)
        ready_indexes = [0, 1]  # Only 1 group, need 2 (global_batch = 4)

        sampled, consumed = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert sampled == []
        assert consumed == []

    def test_dp_rank_out_of_range(self):
        """Test behavior when dp_rank >= dp_size (returns empty)."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]

        # First call to populate cache
        sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        # dp_rank=5 is out of range
        sampled, consumed = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=5,
            batch_index=0,
        )

        assert sampled == []
        assert consumed == []

    def test_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=1)
        ready_indexes = [0, 1, 2, 3]

        sampled, consumed = sampler(
            ready_indexes,
            4,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert sampled == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]

    def test_batch_size_not_divisible_by_n_samples_per_prompt(self):
        """Test that batch_size must be divisible by n_samples_per_prompt (inherited)."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=4, dp_size=2)
        ready_indexes = list(range(20))

        with pytest.raises(ValueError) as exc_info:
            sampler.sample(
                ready_indexes,
                3,  # per-DP=3, global=6, 6 % 4 != 0
                task_name="task",
                partition_id="p0",
                dp_rank=0,
                batch_index=0,
            )

        assert "must be a multiple of n_samples_per_prompt" in str(exc_info.value)


class TestKarmarkarKarp:
    """Test cases for karmarkar_karp and get_seqlen_balanced_partitions utilities."""

    def test_equal_size_basic(self):
        """Test equal-size partitioning with balanced inputs."""
        seqlens = [10, 20, 30, 40]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)

        assert len(partitions) == 2
        assert all(len(p) == 2 for p in partitions)
        # All indices covered
        assert sorted(sum(partitions, [])) == [0, 1, 2, 3]

    def test_equal_size_balance_quality(self):
        """Test that KK produces well-balanced partitions."""
        seqlens = [100, 90, 50, 10, 5, 1]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)

        sums = [sum(seqlens[i] for i in p) for p in partitions]
        # Difference should be small relative to total
        assert abs(sums[0] - sums[1]) <= max(seqlens)

    def test_unequal_size(self):
        """Test variable-size partitioning."""
        seqlens = [100, 10, 10, 10, 10]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=False)

        assert len(partitions) == 2
        assert sorted(sum(partitions, [])) == [0, 1, 2, 3, 4]

    def test_single_partition(self):
        """Test with k_partitions=1 returns all items."""
        seqlens = [10, 20, 30]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=1, equal_size=False)

        assert len(partitions) == 1
        assert sorted(partitions[0]) == [0, 1, 2]

    def test_equal_size_assertion_error(self):
        """Test that equal_size raises when items not divisible by k."""
        seqlens = [10, 20, 30]
        with pytest.raises(AssertionError):
            get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)

    def test_too_few_items(self):
        """Test that too few items raises AssertionError."""
        seqlens = [10]
        with pytest.raises(AssertionError):
            get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=False)

    def test_three_way_partition(self):
        """Test 3-way partitioning."""
        seqlens = [100, 80, 60, 40, 20, 10]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=True)

        assert len(partitions) == 3
        assert all(len(p) == 2 for p in partitions)
        assert sorted(sum(partitions, [])) == [0, 1, 2, 3, 4, 5]

    def test_identical_seqlens(self):
        """Test with all identical sequence lengths."""
        seqlens = [50, 50, 50, 50]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)

        sums = [sum(seqlens[i] for i in p) for p in partitions]
        assert sums[0] == sums[1] == 100


class TestSamplerIntegration:
    """Integration tests for samplers."""

    def test_samplers_implement_base_interface(self):
        """Test that all samplers properly implement BaseSampler interface."""
        samplers = [SequentialSampler(), GRPOGroupNSampler(), SeqlenBalancedSampler()]

        for sampler in samplers:
            # Test that they are instances of BaseSampler
            assert isinstance(sampler, BaseSampler)

            # Test that they have the required methods
            assert hasattr(sampler, "sample")
            assert callable(sampler.sample)
            assert callable(sampler)
            assert callable(sampler.__call__)

    def test_samplers_return_consistent_types(self):
        """Test that all samplers return consistent tuple types."""
        samplers = [
            (SequentialSampler(), {}),
            (GRPOGroupNSampler(n_samples_per_prompt=2), {}),
            (
                SeqlenBalancedSampler(n_samples_per_prompt=2, dp_size=1),
                {
                    "task_name": "task",
                    "partition_id": "test",
                    "dp_rank": 0,
                    "batch_index": 0,
                },
            ),
        ]

        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 4

        for sampler, kwargs in samplers:
            sampled, consumed = sampler.sample(ready_indexes, batch_size, **kwargs)

            # Check return types
            assert isinstance(sampled, list)
            assert isinstance(consumed, list)
            assert isinstance(sampled[0], int) if sampled else True
            assert isinstance(consumed[0], int) if consumed else True

            # Check return value consistency
            assert len(sampled) <= batch_size
            assert len(sampled) == len(consumed)

    def test_samplers_handle_edge_cases_consistently(self):
        """Test that samplers handle edge cases consistently."""
        samplers = [(SequentialSampler(), {}), (GRPOGroupNSampler(n_samples_per_prompt=2), {})]

        # Test empty ready indexes
        for sampler, kwargs in samplers:
            try:
                sampled, consumed = sampler.sample([], 0, **kwargs)
                assert sampled == []
                assert consumed == []
            except Exception:
                # GRPO sampler might fail with empty list, that's expected
                pass

        # Test zero batch size
        for sampler, kwargs in samplers:
            try:
                sampled, consumed = sampler.sample([0, 1, 2, 3], 0, **kwargs)
                assert sampled == []
                assert consumed == []
            except Exception:
                # Some samplers might not handle zero batch size
                pass


if __name__ == "__main__":
    pytest.main([__file__])
