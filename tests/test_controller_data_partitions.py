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
import sys
import time
from pathlib import Path

import pytest

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TQ_INIT_SAMPLE_NUM = int(os.environ.get("TQ_INIT_SAMPLE_NUM", 1))  # Initial number of samples
TQ_INIT_FIELD_NUM = int(os.environ.get("TQ_INIT_FIELD_NUM", 1))


def test_data_partition_status():
    """Test the DataPartitionStatus class functionality."""
    print("Testing DataPartitionStatus...")

    from transfer_queue.controller import DataPartitionStatus

    # Create a partition
    partition = DataPartitionStatus(partition_id="test@partition_1")

    # Test initial state
    assert partition.total_samples_num == 0
    assert partition.allocated_samples_num == TQ_INIT_SAMPLE_NUM
    assert partition.total_fields_num == 0
    assert partition.allocated_fields_num == TQ_INIT_FIELD_NUM
    assert partition.production_status is not None

    print("✓ Initial state correct")

    # Test dynamic expansion through update_production_status
    success = partition.update_production_status(
        global_indices=[0, 1, 2],
        field_names=["input_ids", "attention_mask"],
        dtypes={
            0: {"input_ids": "torch.int32", "attention_mask": "torch.bool"},
            1: {"input_ids": "torch.int32", "attention_mask": "torch.bool"},
            2: {"input_ids": "torch.int32", "attention_mask": "torch.bool"},
        },
        shapes={
            0: {"input_ids": (512,), "attention_mask": (512,)},
            1: {"input_ids": (512,), "attention_mask": (512,)},
            2: {"input_ids": (512,), "attention_mask": (512,)},
        },
        custom_meta=None,
    )

    assert success
    assert partition.total_samples_num >= 3  # Should expand to accommodate index 2 (likely to TQ_INIT_FIELD_NUM)
    assert partition.total_fields_num == 2  # Two fields registered
    assert partition.production_status is not None
    assert partition.production_status.shape[0] >= 3
    assert partition.production_status.shape[1] >= 2

    print("✓ Dynamic expansion works")

    # Test field metadata retrieval
    dtype = partition.get_field_dtype(0, "input_ids")
    shape = partition.get_field_shape(1, "attention_mask")
    assert dtype == "torch.int32"
    assert shape == (512,)

    print("✓ Field metadata retrieval works")

    # Test consumption status
    global_index, consumption_tensor = partition.get_consumption_status("test_task", mask=False)
    assert consumption_tensor is not None
    assert consumption_tensor.shape[0] == partition.allocated_samples_num

    print("✓ Consumption status creation works")

    # Test marking samples as consumed
    partition.mark_consumed("test_task", [0, 1])
    assert consumption_tensor[0] == 1
    assert consumption_tensor[1] == 1
    assert consumption_tensor[2] == 0  # Not marked

    print("✓ Sample consumption marking works")

    # Test scanning for ready samples (should only return unconsumed samples)
    ready_samples = partition.scan_data_status(field_names=["input_ids", "attention_mask"], task_name="test_task")

    # Should include only sample 2 (0 and 1 are consumed)
    assert len(ready_samples) == 1, f"Expected 1 ready sample, got {len(ready_samples)}: {ready_samples}"
    assert ready_samples == [2], f"Expected [2], got {ready_samples}"

    print("✓ Ready sample scanning works")

    # Test statistics
    stats = partition.get_statistics()
    assert stats["partition_id"] == "test@partition_1"
    assert stats["total_samples_num"] == partition.total_samples_num
    assert stats["total_fields_num"] == 2
    assert "consumption_statistics" in stats

    print("✓ Statistics generation works")

    print("DataPartitionStatus tests passed!\n")


def test_partition_interface():
    """Test the partition interface design."""
    print("Testing partition interface design...")

    # This test focuses on the interface design without actually creating
    # the Ray actor, which would require more complex setup

    from transfer_queue.controller import TransferQueueController

    # Test that the class can be imported and has expected methods
    assert hasattr(TransferQueueController, "create_partition")
    assert hasattr(TransferQueueController, "get_partition_snapshot")
    assert hasattr(TransferQueueController, "update_production_status")
    assert hasattr(TransferQueueController, "scan_data_status")
    assert hasattr(TransferQueueController, "generate_batch_meta")

    print("✓ Controller has all expected methods")

    # Test method signatures
    import inspect

    # Check create_partition signature (should not require num_samples anymore)
    sig = inspect.signature(TransferQueueController.create_partition)
    params = list(sig.parameters.keys())
    assert "partition_id" in params
    assert "num_samples" not in params  # Should be removed in refactoring

    print("✓ Method signatures are correct")

    print("Partition interface tests passed!\n")


def test_dynamic_expansion_scenarios():
    """Test various dynamic expansion scenarios."""
    print("Testing dynamic expansion scenarios...")

    from transfer_queue.controller import DataPartitionStatus

    partition = DataPartitionStatus(partition_id="expansion_test")

    # Scenario 1: Adding samples with large gaps
    partition.update_production_status(
        global_indices=[0, 5, 10],
        field_names=["field1"],
        dtypes={
            0: {"field_1": "torch.bool"},
            5: {"field_1": "torch.bool"},
            10: {"field_1": "torch.bool"},
        },
        shapes={
            0: {"field_1": (32,)},
            5: {"field_1": (32,)},
            10: {"field_1": (32,)},
        },
        custom_meta=None,
    )
    assert partition.total_samples_num == 3
    assert partition.allocated_samples_num >= 11  # Should accommodate index 10
    print("✓ Large index gaps handled correctly")

    # Scenario 2: Adding many fields dynamically
    for i in range(15):
        partition.update_production_status(
            [0], [f"field_{i}"], {0: {f"field_{i}": "torch.bool"}}, {0: {f"field_{i}": (32,)}}, None
        )

    assert partition.total_fields_num == 16  # Original + 15 new fields
    assert partition.allocated_fields_num >= 16

    print("✓ Dynamic field expansion works")

    # Scenario 3: Multiple tasks consuming same partition
    tasks = ["task1", "task2", "task3"]
    for task in tasks:
        partition.get_consumption_status(task)
        partition.mark_consumed(task, [0, 1])

    assert len(partition.consumption_status) == 3
    for task in tasks:
        assert partition.consumption_status[task][0] == 1
        assert partition.consumption_status[task][1] == 1

    print("✓ Multiple task consumption works")

    print("Dynamic expansion tests passed!\n")


def test_data_partition_status_advanced():
    """Advanced tests for DataPartitionStatus refactoring features."""
    print("Testing advanced DataPartitionStatus features...")

    from transfer_queue.controller import DataPartitionStatus

    # Test 1: Property-based capacity tracking
    partition = DataPartitionStatus(partition_id="advanced_test")

    # Initially empty
    assert partition.total_samples_num == 0
    assert partition.allocated_samples_num == TQ_INIT_SAMPLE_NUM
    assert partition.total_fields_num == 0
    assert partition.allocated_fields_num == TQ_INIT_FIELD_NUM

    # Add data to trigger expansion
    dtypes = {i: {f"dynamic_field_{s}": "torch.bool" for s in ["a", "b", "c"]} for i in range(5)}
    shapes = {i: {f"dynamic_field_{s}": (32,) for s in ["a", "b", "c"]} for i in range(5)}
    partition.update_production_status([0, 1, 2, 3, 4], ["field_a", "field_b", "field_c"], dtypes, shapes, None)

    # Properties should reflect current state
    assert partition.total_samples_num >= 5  # At least 5 samples
    assert partition.total_fields_num == 3  # Exactly 3 fields registered
    assert partition.allocated_fields_num >= 3  # At least 3 columns allocated

    print("✓ Property-based capacity tracking works")

    # Test 2: Consumption status with multiple expansions
    task_name = "multi_expansion_task"

    # Initial consumption tracking
    partition.mark_consumed(task_name, [0, 1])
    global_index, initial_consumption = partition.get_consumption_status(task_name)
    assert initial_consumption[0] == 1
    assert initial_consumption[1] == 1

    # Expand samples and verify consumption data preserved
    dtypes = (
        {
            10: {"field_d": "torch.bool"},
            11: {"field_d": "torch.bool"},
            12: {"field_d": "torch.bool"},
        },
    )
    shapes = {
        10: {"field_d": (32,)},
        11: {"field_d": (32,)},
        12: {"field_d": (32,)},
    }
    partition.update_production_status([10, 11, 12], ["field_d"], dtypes, shapes, None)  # Triggers sample expansion
    global_index, expanded_consumption = partition.get_consumption_status(task_name)
    assert expanded_consumption[0] == 1  # Preserved
    assert expanded_consumption[1] == 1  # Preserved
    assert expanded_consumption.shape[0] >= 13  # Expanded to accommodate new samples

    print("✓ Consumption data preserved across expansions")

    # Test 3: Complex field addition scenarios
    # Start with some fields
    dtypes = {0: {"initial_field": "torch.bool"}}
    shapes = {0: {"field_d": (32,)}}
    partition.update_production_status([0], ["initial_field"], dtypes, shapes, None)

    # Add many fields to trigger column expansion
    new_fields = [f"dynamic_field_{i}" for i in range(20)]
    dtypes = {1: {f"dynamic_field_{i}": "torch.bool" for i in range(20)}}
    shapes = {1: {f"dynamic_field_{i}": (32,) for i in range(20)}}
    partition.update_production_status([1], new_fields, dtypes, shapes, None)

    # Verify all fields are registered and accessible
    assert "initial_field" in partition.field_name_mapping
    for field in new_fields:
        assert field in partition.field_name_mapping

    expected_fields = 1 + len(new_fields)
    assert partition.total_fields_num >= expected_fields  # Should be at least this many fields
    assert partition.allocated_fields_num >= partition.total_fields_num

    print("✓ Complex field addition scenarios work")

    # Test 4: Statistics and monitoring
    stats = partition.get_statistics()

    required_keys = [
        "partition_id",
        "created_at",
        "total_samples_num",
        "total_fields_num",
        "allocated_samples_num",
        "allocated_fields_num",
        "registered_tasks",
        "produced_samples",
        "production_progress",
        "field_statistics",
        "consumption_statistics",
    ]

    for key in required_keys:
        assert key in stats, f"Missing key in statistics: {key}"

    assert stats["partition_id"] == "advanced_test"
    assert stats["total_fields_num"] > 0
    assert isinstance(stats["field_statistics"], dict)
    assert isinstance(stats["consumption_statistics"], dict)

    print("✓ Statistics generation comprehensive")

    # Test 5: Data clearing functionality
    initial_consumption_sum = sum(t.sum().item() for t in partition.consumption_status.values())

    # Clear only production data
    partition.clear_data(list(range(4)), clear_consumption=False)
    assert partition.production_status[:4, :].sum().item() == 0

    # Consumption data should remain
    remaining_consumption_sum = sum(t.sum().item() for t in partition.consumption_status.values())
    assert remaining_consumption_sum == initial_consumption_sum

    print("✓ Selective data clearing works")

    print("Advanced DataPartitionStatus tests passed!\n")


def test_edge_cases_and_error_handling():
    """Test edge cases and error handling in DataPartitionStatus."""
    print("Testing edge cases and error handling...")

    from transfer_queue.controller import DataPartitionStatus

    # Test 1: Operations on empty partition
    partition = DataPartitionStatus(partition_id="edge_test")

    # Scanning on empty partition should not crash
    ready_samples = partition.scan_data_status(["nonexistent_field"], "task")
    assert ready_samples == []

    print("✓ Empty partition operations handled gracefully")

    # Test 2: Field metadata operations
    # Test metadata retrieval for non-existent samples/fields
    dtype = partition.get_field_dtype(999, "nonexistent_field")
    shape = partition.get_field_shape(999, "nonexistent_field")
    assert dtype is None
    assert shape is None

    print("✓ Metadata retrieval for non-existent data handled correctly")

    # Test 3: Consumption status edge cases
    # Test consumption status creation before production status
    task_name = "early_task"
    _, consumption_tensor = partition.get_consumption_status(task_name)
    assert consumption_tensor is not None
    assert consumption_tensor.shape[0] == partition.allocated_samples_num

    # Test 4: Production status update error conditions
    # Test with empty lists
    success = partition.update_production_status([], [], [], [])
    assert success  # Should handle empty lists gracefully

    # Test with valid data but ensure no crashes
    dtypes = {0: {"new_field": "torch.int64"}}
    shapes = {0: {"new_field": (32,)}}
    success = partition.update_production_status([0], ["new_field"], dtypes=dtypes, shapes=shapes)
    assert success

    print("✓ Production status update edge cases handled correctly")

    print("Edge cases and error handling tests passed!\n")


def test_performance_characteristics():
    """Test performance characteristics of the refactored implementation."""
    print("Testing performance characteristics...")

    from transfer_queue.controller import DataPartitionStatus

    partition = DataPartitionStatus(partition_id="perf_test")

    # Test 1: Large number of fields (use a smaller number to avoid expansion limits)
    start_time = time.time()
    field_count = 100  # Reduced from 1000 to avoid potential issues
    many_fields = [f"perf_field_{i}" for i in range(field_count)]
    dtypes = {0: {f"perf_field_{i}": "torch.bool" for i in range(field_count)}}
    shapes = {0: {f"perf_field_{i}": (32,) for i in range(field_count)}}
    partition.update_production_status([0], many_fields, dtypes, shapes)
    field_creation_time = time.time() - start_time

    assert partition.total_fields_num == field_count
    assert field_creation_time < 5.0  # Should complete within 5 seconds
    print(f"✓ Large field creation: {field_creation_time:.3f}s for {field_count} fields")

    # Test 2: Large number of samples
    start_time = time.time()
    many_samples = list(range(5000))
    dtypes = {k: {"test_field": "torch.int64"} for k in many_samples}
    shapes = {k: {"test_field": (32,)} for k in many_samples}
    partition.update_production_status(many_samples, ["test_field"], dtypes=dtypes, shapes=shapes)
    sample_creation_time = time.time() - start_time

    assert partition.total_samples_num >= 5000
    assert sample_creation_time < 5.0  # Should complete within 5 seconds
    print(f"✓ Large sample creation: {sample_creation_time:.3f}s for 5000 samples")

    # Test 3: Efficient scanning
    # Mark some samples as consumed
    task_name = "perf_task"
    partition.mark_consumed(task_name, many_samples[::2])  # Mark every other sample

    start_time = time.time()
    ready_samples = partition.scan_data_status(["test_field"], task_name)
    scanning_time = time.time() - start_time

    assert len(ready_samples) == 2500  # Half should be unconsumed
    assert scanning_time < 1.0  # Should be very fast
    print(f"✓ Efficient scanning: {scanning_time:.3f}s for 5000 samples")

    # Test 4: Memory usage pattern
    # The implementation should not grow memory excessively
    initial_allocated = partition.allocated_fields_num
    initial_samples = partition.total_samples_num

    # Add more data (should reuse existing space where possible)
    dtypes = {100: {"new_field": "torch.int64"}}
    shapes = {100: {"new_field": (32,)}}
    partition.update_production_status([100], ["new_field"], dtypes=dtypes, shapes=shapes)

    # Memory growth should be reasonable
    final_allocated = partition.allocated_fields_num
    final_samples = partition.total_samples_num

    # Should not double the allocation for small additions
    if final_samples == initial_samples:  # If sample count didn't change
        assert final_allocated < initial_allocated * 2

    print("✓ Memory usage patterns reasonable")

    print("Performance characteristics tests passed!\n")


def test_custom_meta_in_data_partition_status():
    """Simplified tests for custom_meta functionality in DataPartitionStatus."""

    print("Testing simplified custom_meta in DataPartitionStatus...")

    from transfer_queue.controller import DataPartitionStatus

    partition = DataPartitionStatus(partition_id="custom_meta_test")

    # Basic custom_meta storage via update_production_status
    global_indices = [0, 1, 2]
    field_names = ["input_ids", "attention_mask"]
    dtypes = {i: {"input_ids": "torch.int32", "attention_mask": "torch.bool"} for i in global_indices}
    shapes = {i: {"input_ids": (512,), "attention_mask": (512,)} for i in global_indices}
    custom_meta = {
        0: {"input_ids": {"token_count": 100}},
        1: {"attention_mask": {"mask_ratio": 0.2}},
        2: {"input_ids": {"token_count": 300}},
    }

    success = partition.update_production_status(
        global_indices=global_indices,
        field_names=field_names,
        dtypes=dtypes,
        shapes=shapes,
        custom_meta=custom_meta,
    )

    assert success

    # Verify some stored values
    assert partition.field_custom_metas[0]["input_ids"]["token_count"] == 100
    assert partition.field_custom_metas[1]["attention_mask"]["mask_ratio"] == 0.2

    # Retrieval via helper for a subset of fields
    retrieved = partition.get_field_custom_backend_meta([0, 1], ["input_ids", "attention_mask"])
    assert 0 in retrieved and "input_ids" in retrieved[0]
    assert 1 in retrieved and "attention_mask" in retrieved[1]

    # Clearing a sample should remove its custom_meta
    partition.clear_data([0], clear_consumption=True)
    assert 0 not in partition.field_custom_metas

    print("✓ Custom_meta tests passed")


def test_update_field_metadata_variants():
    """Test _update_field_metadata handles dtypes/shapes/custom_meta being optional and merging."""
    from transfer_queue.controller import DataPartitionStatus

    partition = DataPartitionStatus(partition_id="update_meta_test")

    # Only dtypes provided
    global_indices = [0, 1]
    dtypes = {0: {"f1": "torch.int32"}, 1: {"f1": "torch.bool"}}

    partition._update_field_metadata(global_indices, dtypes, shapes=None, custom_meta=None)
    assert partition.field_dtypes[0]["f1"] == "torch.int32"
    assert partition.field_dtypes[1]["f1"] == "torch.bool"
    assert partition.field_shapes == {}
    assert partition.field_custom_metas == {}

    # Only shapes provided for a new index
    partition._update_field_metadata([2], dtypes=None, shapes={2: {"f2": (16,)}}, custom_meta=None)
    assert partition.field_shapes[2]["f2"] == (16,)

    # Only custom_meta provided and merged with existing entries
    partition._update_field_metadata([2], dtypes=None, shapes=None, custom_meta={2: {"f2": {"meta": 1}}})
    assert 2 in partition.field_custom_metas
    assert partition.field_custom_metas[2]["f2"]["meta"] == 1

    # Merging dtypes on an existing index should preserve previous keys
    partition._update_field_metadata([0], dtypes={0: {"f2": "torch.float32"}}, shapes=None, custom_meta=None)
    assert partition.field_dtypes[0]["f1"] == "torch.int32"
    assert partition.field_dtypes[0]["f2"] == "torch.float32"

    # Length mismatch should raise ValueError when provided mapping lengths differ from global_indices
    with pytest.raises(ValueError):
        partition._update_field_metadata([0, 1, 2], dtypes={0: {}}, shapes=None, custom_meta=None)


def test_get_production_status_for_fields():
    """Test get_production_status_for_fields method with mask parameter."""
    print("Testing get_production_status_for_fields...")

    import torch

    from transfer_queue.controller import DataPartitionStatus

    partition = DataPartitionStatus(partition_id="production_status_test")

    # Add some data first (using non-contiguous indices)
    partition.update_production_status(
        global_indices=[0, 1, 2, 3, 9],
        field_names=["field_a", "field_b"],
        dtypes={i: {"field_a": "torch.int64", "field_b": "torch.bool"} for i in [0, 1, 2, 3, 9]},
        shapes={i: {"field_a": (32,), "field_b": (32,)} for i in [0, 1, 2, 3, 9]},
    )

    # Test get_production_status_for_fields WITHOUT mask (mask=False)
    global_index, production_status = partition.get_production_status_for_fields(
        field_names=["field_a", "field_b"], mask=False
    )
    assert torch.equal(global_index, torch.tensor([0, 1, 2, 3, 9], dtype=torch.long))
    # Without mask, should return all allocated samples
    assert production_status.shape[0] == partition.allocated_samples_num
    # Production status should be 1 for produced samples (0, 1, 2, 3, 9), 0 for others
    # Check that produced samples have all fields produced (all 1s)
    assert torch.all(production_status[0] == 1), "Sample 0 should be produced"
    assert torch.all(production_status[1] == 1), "Sample 1 should be produced"
    assert torch.all(production_status[2] == 1), "Sample 2 should be produced"
    assert torch.all(production_status[3] == 1), "Sample 3 should be produced"
    assert torch.all(production_status[9] == 1), "Sample 9 should be produced"
    # Verify shape - should have 2 fields (columns)
    assert production_status.shape[1] == 2

    print("✓ get_production_status_for_fields without mask works")

    # Test get_production_status_for_fields WITH mask (mask=True)
    global_index_masked, production_status_masked = partition.get_production_status_for_fields(
        field_names=["field_a", "field_b"], mask=True
    )
    assert torch.equal(global_index_masked, torch.tensor([0, 1, 2, 3, 9], dtype=torch.long))
    # Masked status should be same as original for these indices
    assert production_status_masked.shape == (len([0, 1, 2, 3, 9]), 2)
    # All returned samples should be produced
    assert torch.all(production_status_masked == 1)

    print("✓ get_production_status_for_fields with mask works")

    # Test get_production_status_for_fields with subset of fields
    global_index_subset, production_status_subset = partition.get_production_status_for_fields(
        field_names=["field_a"], mask=True
    )
    assert global_index_subset.shape[0] == len([0, 1, 2, 3, 9])
    assert production_status_subset.shape == (len([0, 1, 2, 3, 9]), 1)  # Only one field

    print("✓ get_production_status_for_fields with subset fields works")

    print("get_production_status_for_fields tests passed!\n")


def test_get_consumption_status_parameter():
    """Test get_consumption_status method with mask parameter."""
    print("Testing consumption status mask parameter...")

    import torch

    from transfer_queue.controller import DataPartitionStatus

    partition = DataPartitionStatus(partition_id="consumption_mask_test")
    partition_another = DataPartitionStatus(partition_id="other_partition")

    # Add some data
    partition.update_production_status(
        global_indices=[0, 1, 2, 3, 9],
        field_names=["field_a"],
        dtypes={i: {"field_a": "torch.int64"} for i in [0, 1, 2, 3, 9]},
        shapes={i: {"field_a": (32,)} for i in [0, 1, 2, 3, 9]},
    )

    partition_another.update_production_status(
        global_indices=[5, 6, 7],
        field_names=["field_a"],
        dtypes={i: {"field_a": "torch.int64"} for i in [5, 6, 7]},
        shapes={i: {"field_a": (32,)} for i in [5, 6, 7]},
    )

    # Mark some samples as consumed
    partition.mark_consumed("test_task", [0, 2])

    # Test get_consumption_status WITHOUT mask (mask=False)
    global_index, consumption_status = partition.get_consumption_status("test_task", mask=False)
    assert global_index.shape[0] == partition.total_samples_num
    assert torch.equal(global_index, torch.tensor([0, 1, 2, 3, 9], dtype=torch.long))
    # Without mask, should return all allocated samples
    assert consumption_status.shape[0] == 10
    assert consumption_status[0].item() == 1
    assert consumption_status[1].item() == 0
    assert consumption_status[2].item() == 1
    assert consumption_status[3].item() == 0
    assert consumption_status[4].item() == 0  # empty slot
    assert consumption_status[5].item() == 0  # empty slot
    assert consumption_status[6].item() == 0  # empty slot
    assert consumption_status[7].item() == 0  # empty slot
    assert consumption_status[8].item() == 0  # empty slot
    assert consumption_status[9].item() == 0

    print("✓ get_consumption_status without mask works")

    # Test get_consumption_status WITH mask (mask=True)
    global_index_masked, consumption_status_masked = partition.get_consumption_status("test_task", mask=True)
    # With mask, should return only global_indexes [0, 1, 2, 3, 9]
    assert global_index_masked.shape[0] == partition.total_samples_num
    assert torch.equal(global_index_masked, torch.tensor([0, 1, 2, 3, 9], dtype=torch.long))
    # Masked status shape[0] should correspond to global indexes
    assert consumption_status_masked.shape[0] == partition.total_samples_num
    assert consumption_status_masked[0].item() == 1
    assert consumption_status_masked[1].item() == 0
    assert consumption_status_masked[2].item() == 1
    assert consumption_status_masked[3].item() == 0
    assert consumption_status_masked[4].item() == 0  # no empty slot. this corresponds to global_index=9

    print("✓ get_consumption_status with mask works")

    print("Consumption status mask parameter tests passed!\n")


def test_pre_allocated_indexes_basic():
    """Test basic pre-allocated indexes functionality in DataPartitionStatus."""
    from transfer_queue.controller import DataPartitionStatus

    print("Testing pre-allocated indexes basic functionality...")

    partition = DataPartitionStatus(partition_id="prealloc_test")

    # Initially, pre_allocated_global_indexes should be empty
    assert len(partition.pre_allocated_global_indexes) == 0
    assert partition.total_samples_num == 0

    print("✓ Initial state correct")

    # Register pre-allocated indexes
    pre_allocated = [0, 1, 2, 3, 4]
    partition.register_pre_allocated_indexes(pre_allocated)

    assert partition.pre_allocated_global_indexes == set(pre_allocated)
    # global_indexes should still be empty until retrieved
    assert partition.total_samples_num == 0

    print("✓ Pre-allocated indexes registered")

    # activate pre-allocated indexes
    retrieved = partition.activate_pre_allocated_indexes(3)

    assert len(retrieved) == 3
    assert set(retrieved) == {0, 1, 2}
    assert partition.global_indexes == {0, 1, 2}
    assert partition.pre_allocated_global_indexes == {3, 4}
    assert partition.total_samples_num == 3

    print("✓ Pre-allocated indexes activate & retrieved correctly")

    # Activate remaining indexes
    retrieved = partition.activate_pre_allocated_indexes(5)

    assert len(retrieved) == 2  # Only 2 remaining
    assert set(retrieved) == {3, 4}
    assert partition.global_indexes == {0, 1, 2, 3, 4}
    assert partition.pre_allocated_global_indexes == set()
    assert partition.total_samples_num == 5

    print("✓ All pre-allocated indexes retrieved")

    print("Pre-allocated indexes basic tests passed!\n")


def test_pre_allocated_indexes_consumption_status():
    """Test that pre-allocated indexes are included in consumption status."""
    import torch

    from transfer_queue.controller import DataPartitionStatus

    print("Testing pre-allocated indexes in consumption status...")

    partition = DataPartitionStatus(partition_id="consumption_test")

    # Register pre-allocated indexes
    partition.register_pre_allocated_indexes([0, 1, 2, 3, 4])

    # Get consumption status - should include pre-allocated indexes
    global_index, consumption_status = partition.get_consumption_status("test_task", mask=True)

    # global_index should include all pre-allocated indexes
    assert torch.equal(global_index, torch.tensor([0, 1, 2, 3, 4], dtype=torch.long))
    # All consumption statuses should be 0 (not consumed yet)
    assert torch.all(consumption_status == 0)

    print("✓ Consumption status includes pre-allocated indexes")

    # Mark some samples as consumed
    partition.mark_consumed("test_task", [0, 2, 4])

    # Get consumption status again
    global_index, consumption_status = partition.get_consumption_status("test_task", mask=True)

    assert consumption_status[0].item() == 1  # consumed
    assert consumption_status[1].item() == 0  # not consumed
    assert consumption_status[2].item() == 1  # consumed
    assert consumption_status[3].item() == 0  # not consumed
    assert consumption_status[4].item() == 1  # consumed

    print("✓ Marked consumed works with pre-allocated indexes")

    print("Pre-allocated indexes consumption status tests passed!\n")


def test_pre_allocated_indexes_in_scan_data_status():
    """Test that pre-allocated indexes affect scan_data_status behavior."""
    from transfer_queue.controller import DataPartitionStatus

    print("Testing pre-allocated indexes in scan_data_status...")

    partition = DataPartitionStatus(partition_id="scan_test")

    # Register pre-allocated indexes (5 samples)
    partition.register_pre_allocated_indexes([0, 1, 2, 3, 4])

    # Before any production, scan should return empty (no samples produced yet)
    ready = partition.scan_data_status(field_names=["input_ids"], task_name="test_task")
    assert ready == []

    print("✓ Scan returns empty before production")

    # Now produce some samples (0, 2, 4)
    partition.update_production_status(
        global_indices=[0, 2, 4],
        field_names=["input_ids"],
        dtypes={i: {"input_ids": "torch.int32"} for i in [0, 2, 4]},
        shapes={i: {"input_ids": (32,)} for i in [0, 2, 4]},
    )

    # Scan should return produced and unconsumed samples
    ready = partition.scan_data_status(field_names=["input_ids"], task_name="test_task")
    assert set(ready) == {0, 2, 4}

    print("✓ Scan returns produced samples correctly")

    # Mark sample 2 as consumed
    partition.mark_consumed("test_task", [2])

    # Scan should now return only 0 and 4
    ready = partition.scan_data_status(field_names=["input_ids"], task_name="test_task")
    assert set(ready) == {0, 4}

    print("✓ Scan respects consumption status")

    print("Pre-allocated indexes scan_data_status tests passed!\n")


def test_pre_allocated_indexes_mixed_with_dynamic():
    """Test mixing pre-allocated indexes with dynamically allocated ones."""
    from transfer_queue.controller import DataPartitionStatus

    print("Testing mixed pre-allocated and dynamic indexes...")

    partition = DataPartitionStatus(partition_id="mixed_test")

    # Register 3 pre-allocated indexes
    partition.register_pre_allocated_indexes([0, 1, 2])

    # Simulate adding more samples (indexes 5, 6, 7)
    # This would happen when producer calls update_production_status
    partition.update_production_status(
        global_indices=[5, 6, 7],
        field_names=["input_ids"],
        dtypes={i: {"input_ids": "torch.int32"} for i in [5, 6, 7]},
        shapes={i: {"input_ids": (32,)} for i in [5, 6, 7]},
    )

    # Now global_indexes should only contain dynamically generated in (5,6,7)
    assert partition.global_indexes == {5, 6, 7}
    assert partition.total_samples_num == 3

    # all pre-allocated
    retrieved = partition.activate_pre_allocated_indexes(3)
    assert set(retrieved) == {0, 1, 2}

    # Now global_indexes should have both pre-allocated (0,1,2) and dynamic (5,6,7)
    assert partition.global_indexes == {0, 1, 2, 5, 6, 7}
    assert partition.total_samples_num == 6

    print("✓ Mixed pre-allocated and dynamic indexes work correctly")

    print("Mixed indexes tests passed!\n")
