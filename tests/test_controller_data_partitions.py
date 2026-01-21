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
    consumption_tensor = partition.get_consumption_status("test_task")
    assert consumption_tensor is not None
    assert consumption_tensor.shape[0] == partition.total_samples_num

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
    initial_consumption = partition.get_consumption_status(task_name)
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
    expanded_consumption = partition.get_consumption_status(task_name)
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
    consumption_tensor = partition.get_consumption_status(task_name)
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
    """Test custom_meta functionality in DataPartitionStatus."""
    print("Testing custom_meta in DataPartitionStatus...")

    from transfer_queue.controller import DataPartitionStatus

    partition = DataPartitionStatus(partition_id="custom_meta_test")

    # Test 1: Basic custom_meta storage via update_production_status
    global_indices = [0, 1, 2]
    field_names = ["input_ids", "attention_mask"]
    dtypes = {
        0: {"input_ids": "torch.int32", "attention_mask": "torch.bool"},
        1: {"input_ids": "torch.int32", "attention_mask": "torch.bool"},
        2: {"input_ids": "torch.int32", "attention_mask": "torch.bool"},
    }
    shapes = {
        0: {"input_ids": (512,), "attention_mask": (512,)},
        1: {"input_ids": (512,), "attention_mask": (512,)},
        2: {"input_ids": (512,), "attention_mask": (512,)},
    }
    custom_meta = {
        0: {"input_ids": {"token_count": 100}, "attention_mask": {"mask_ratio": 0.1}},
        1: {"input_ids": {"token_count": 200}, "attention_mask": {"mask_ratio": 0.2}},
        2: {"input_ids": {"token_count": 300}, "attention_mask": {"mask_ratio": 0.3}},
    }

    success = partition.update_production_status(
        global_indices=global_indices,
        field_names=field_names,
        dtypes=dtypes,
        shapes=shapes,
        custom_meta=custom_meta,
    )

    assert success
    assert len(partition.field_custom_metas) == 3

    # Verify custom_meta is stored correctly
    assert partition.field_custom_metas[0]["input_ids"]["token_count"] == 100
    assert partition.field_custom_metas[1]["attention_mask"]["mask_ratio"] == 0.2
    assert partition.field_custom_metas[2]["input_ids"]["token_count"] == 300

    print("✓ Basic custom_meta storage works")

    # Test 2: get_field_custom_meta retrieval
    retrieved_meta = partition.get_field_custom_meta([0, 1, 2], ["input_ids", "attention_mask"])

    assert 0 in retrieved_meta
    assert 1 in retrieved_meta
    assert 2 in retrieved_meta
    assert retrieved_meta[0]["input_ids"]["token_count"] == 100
    assert retrieved_meta[1]["attention_mask"]["mask_ratio"] == 0.2

    print("✓ get_field_custom_meta retrieval works")

    # Test 3: get_field_custom_meta with partial field filter
    partial_meta = partition.get_field_custom_meta([0, 1], ["input_ids"])

    assert 0 in partial_meta
    assert 1 in partial_meta
    assert "input_ids" in partial_meta[0]
    assert "attention_mask" not in partial_meta[0]  # Should not include non-requested fields

    print("✓ get_field_custom_meta with partial fields works")

    # Test 4: get_field_custom_meta with non-existent global_index
    empty_meta = partition.get_field_custom_meta([999], ["input_ids"])
    assert 999 not in empty_meta  # Should not include non-existent indices

    print("✓ get_field_custom_meta handles non-existent indices correctly")

    # Test 5: custom_meta update (merge) on same global_index
    additional_custom_meta = {
        0: {"new_field": {"new_key": "new_value"}},
    }
    success = partition.update_production_status(
        global_indices=[0],
        field_names=["new_field"],
        dtypes={0: {"new_field": "torch.float32"}},
        shapes={0: {"new_field": (64,)}},
        custom_meta=additional_custom_meta,
    )

    assert success
    # Original custom_meta should be preserved
    assert partition.field_custom_metas[0]["input_ids"]["token_count"] == 100
    # New custom_meta should be merged
    assert partition.field_custom_metas[0]["new_field"]["new_key"] == "new_value"

    print("✓ Custom_meta merge on update works")

    # Test 6: custom_meta cleared on clear_data
    partition.clear_data([0], clear_consumption=True)

    assert 0 not in partition.field_custom_metas
    assert 1 in partition.field_custom_metas  # Other samples should remain
    assert 2 in partition.field_custom_metas

    print("✓ Custom_meta cleared on clear_data works")

    # Test 7: custom_meta None does not create entries
    partition2 = DataPartitionStatus(partition_id="custom_meta_test_2")
    success = partition2.update_production_status(
        global_indices=[0, 1],
        field_names=["field1"],
        dtypes={0: {"field1": "torch.int32"}, 1: {"field1": "torch.int32"}},
        shapes={0: {"field1": (32,)}, 1: {"field1": (32,)}},
        custom_meta=None,
    )

    assert success
    assert len(partition2.field_custom_metas) == 0

    print("✓ Custom_meta None handling works")

    # Test 8: custom_meta length mismatch raises ValueError
    partition3 = DataPartitionStatus(partition_id="custom_meta_test_3")
    mismatched_custom_meta = {
        0: {"field1": {"key": "value"}},
        # Missing entries for 1 and 2
    }
    success = partition3.update_production_status(
        global_indices=[0, 1, 2],
        field_names=["field1"],
        dtypes={0: {"field1": "torch.int32"}, 1: {"field1": "torch.int32"}, 2: {"field1": "torch.int32"}},
        shapes={0: {"field1": (32,)}, 1: {"field1": (32,)}, 2: {"field1": (32,)}},
        custom_meta=mismatched_custom_meta,
    )

    # Should return False due to length mismatch (caught by exception handler)
    assert success is False

    print("✓ Custom_meta length mismatch error handling works")

    # Test 9: Complex nested custom_meta
    partition4 = DataPartitionStatus(partition_id="custom_meta_test_4")
    complex_custom_meta = {
        0: {
            "field1": {
                "nested": {"level1": {"level2": {"value": 42}}},
                "list_data": [1, 2, 3],
                "mixed": {"str": "test", "int": 100, "float": 3.14, "bool": True},
            }
        },
    }
    success = partition4.update_production_status(
        global_indices=[0],
        field_names=["field1"],
        dtypes={0: {"field1": "torch.int32"}},
        shapes={0: {"field1": (32,)}},
        custom_meta=complex_custom_meta,
    )

    assert success
    stored_meta = partition4.field_custom_metas[0]["field1"]
    assert stored_meta["nested"]["level1"]["level2"]["value"] == 42
    assert stored_meta["list_data"] == [1, 2, 3]
    assert stored_meta["mixed"]["str"] == "test"
    assert stored_meta["mixed"]["bool"] is True

    print("✓ Complex nested custom_meta storage works")

    # Test 10: custom_meta preserved in snapshot
    snapshot = partition4.to_snapshot()
    assert 0 in snapshot.field_custom_metas
    assert snapshot.field_custom_metas[0]["field1"]["nested"]["level1"]["level2"]["value"] == 42

    print("✓ Custom_meta preserved in snapshot")

    print("Custom_meta in DataPartitionStatus tests passed!\n")
