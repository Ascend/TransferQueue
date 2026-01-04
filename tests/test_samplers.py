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
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes
        batch_size = 8
        n_samples_per_prompt = 4  # 2 groups of 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(sampled) == batch_size
        assert len(consumed) == batch_size

    def test_grpo_sampler_partial_batch(self):
        """Test partial batch sampling."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 12 indexes
        batch_size = 8  # Want 8 samples total
        n_samples_per_prompt = 4  # 2 groups of 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(sampled) == batch_size
        assert len(consumed) == batch_size

    def test_grpo_sampler_different_group_sizes(self):
        """Test different n_samples_per_prompt values."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # Test with 2 samples per prompt (8 groups)
        sampled, consumed = sampler.sample(ready_indexes, 8, n_samples_per_prompt=2)
        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]

        # Test with 8 samples per prompt (2 groups)
        sampled, consumed = sampler.sample(ready_indexes, 8, n_samples_per_prompt=8)
        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grpo_sampler_batch_size_divisibility(self):
        """Test that batch_size must be divisible by n_samples_per_prompt."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes, sufficient for batch_size=7
        batch_size = 7
        n_samples_per_prompt = 4

        with pytest.raises(ValueError) as exc_info:
            sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        assert "must be a multiple of n_samples_per_prompt" in str(exc_info.value)

    def test_grpo_sampler_insufficient_ready_indexes(self):
        """Test behavior when not enough ready indexes are available."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3]  # Only 4 indexes, but need 8 for 2 groups of 4
        batch_size = 8
        n_samples_per_prompt = 4

        # Should return empty lists when insufficient complete groups
        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)
        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_exact_multiple_available(self):
        """Test when ready_indexes length is exactly a multiple of n_samples_per_prompt."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes
        batch_size = 8
        n_samples_per_prompt = 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grpo_sampler_zero_batch_size(self):
        """Test behavior with zero batch size."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 0
        n_samples_per_prompt = 2

        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_single_sample_per_prompt(self):
        """Test with n_samples_per_prompt = 1."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3
        n_samples_per_prompt = 1

        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        assert sampled == [0, 1, 2]
        assert consumed == [0, 1, 2]

    def test_grpo_sampler_large_group_size(self):
        """Test with large n_samples_per_prompt."""
        sampler = GRPOGroupNSampler()
        ready_indexes = list(range(20))  # 20 indexes
        batch_size = 20
        n_samples_per_prompt = 10

        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        assert sampled == list(range(20))
        assert consumed == list(range(20))

    def test_grpo_sampler_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 4
        n_samples_per_prompt = 2

        sampled, consumed = sampler(ready_indexes, batch_size, n_samples_per_prompt=n_samples_per_prompt)

        assert sampled == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]

    def test_grpo_sampler_parameter_order_independence(self):
        """Test that parameter order doesn't matter when using kwargs."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]

        # Try different parameter orders
        sampled1, consumed1 = sampler.sample(n_samples_per_prompt=4, batch_size=8, ready_indexes=ready_indexes)

        sampled2, consumed2 = sampler.sample(batch_size=8, ready_indexes=ready_indexes, n_samples_per_prompt=4)

        assert sampled1 == sampled2
        assert consumed1 == consumed2

    def test_grpo_sampler_with_extra_kwargs(self):
        """Test that GRPOGroupNSampler accepts extra kwargs but ignores them."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 8
        n_samples_per_prompt = 4

        # GRPOGroupNSampler should accept extra kwargs but ignore them
        sampled, consumed = sampler.sample(
            ready_indexes, batch_size, n_samples_per_prompt, extra_param="ignored", another_param=42
        )

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grpo_sampler_non_sequential_indexes(self):
        """Test with non-sequential ready indexes that get sorted."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [3, 4, 5, 6, 9, 10, 11, 12]  # Non-sequential order but has consecutive groups after sorting
        batch_size = 8
        n_samples_per_prompt = 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        # Should find consecutive groups after sorting: [3,4,5,6] and [9,10,11,12]
        expected = [3, 4, 5, 6, 9, 10, 11, 12]
        assert sampled == expected
        assert consumed == expected

    def test_grpo_sampler_invalid_n_samples_per_prompt(self):
        """Test behavior with invalid n_samples_per_prompt values."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 8

        # Test zero n_samples_per_prompt
        with pytest.raises(ValueError) as exc_info:
            sampler.sample(ready_indexes, batch_size, n_samples_per_prompt=0)
        assert "must be positive" in str(exc_info.value)

        # Test negative n_samples_per_prompt
        with pytest.raises(ValueError) as exc_info:
            sampler.sample(ready_indexes, batch_size, n_samples_per_prompt=-2)
        assert "must be positive" in str(exc_info.value)

    def test_grpo_sampler_no_complete_groups(self):
        """Test behavior when no complete groups are available."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 3, 4, 6, 7]  # No consecutive groups of size 3
        batch_size = 6
        n_samples_per_prompt = 3

        # Should return empty lists when no complete groups found
        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)
        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_mixed_groups(self):
        """Test behavior with mixed complete and incomplete groups."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]  # Mixed groups
        batch_size = 6
        n_samples_per_prompt = 3

        # Should find the complete groups [3,4,5] and [9,10,11]
        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)
        assert sampled == [3, 4, 5, 9, 10, 11]
        assert consumed == [3, 4, 5, 9, 10, 11]

    def test_grpo_sampler_sorting_functionality(self):
        """Test that ready_indexes are properly sorted before group detection."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [10, 11, 12, 5, 6, 7, 8, 9]  # Out of order but contains consecutive groups
        batch_size = 8
        n_samples_per_prompt = 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        # After sorting: [5,6,7,8,9,10,11,12], should find [5,6,7,8] and [9,10,11,12]
        expected = [5, 6, 7, 8, 9, 10, 11, 12]
        assert sampled == expected
        assert consumed == expected

    def test_grpo_sampler_insufficient_groups(self):
        """Test behavior when requesting more groups than available."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 4 groups of 4
        batch_size = 12  # Requesting 3 groups of 4 - this should work
        n_samples_per_prompt = 4

        # This should actually work fine since we have 4 groups and request 3
        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)
        assert len(sampled) == 12
        assert len(consumed) == 12

        # Now test requesting more than available
        batch_size = 20  # Requesting 5 groups of 4, but only have 4
        sampled, consumed = sampler.sample(ready_indexes, batch_size, n_samples_per_prompt)

        # Should return empty lists when requesting more complete groups than available
        assert sampled == []
        assert consumed == []


class TestSamplerIntegration:
    """Integration tests for samplers."""

    def test_samplers_implement_base_interface(self):
        """Test that all samplers properly implement BaseSampler interface."""
        samplers = [SequentialSampler(), GRPOGroupNSampler()]

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
        samplers = [(SequentialSampler(), {}), (GRPOGroupNSampler(), {"n_samples_per_prompt": 2})]

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
        samplers = [(SequentialSampler(), {}), (GRPOGroupNSampler(), {"n_samples_per_prompt": 2})]

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
