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
from pathlib import Path

import pytest
import ray
import torch

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transfer_queue import TransferQueueController  # noqa: E402
from transfer_queue.utils.enum_utils import ProductionStatus  # noqa: E402


@pytest.fixture(scope="function")
def ray_setup():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"RAY_DEBUG": "1", "RAY_DEDUP_LOGS": "0"}},
        log_to_driver=True,
    )
    yield
    if ray.is_initialized():
        ray.shutdown()
        logger.info("Ray has been shut down completely after test")


class TestTransferQueueController:
    def test_controller_with_single_partition(self, ray_setup):
        gbs = 8
        num_n_samples = 4

        tq_controller = TransferQueueController.remote()

        # Test get metadata in insert mode
        partition_id = "train_0"
        data_fields = ["prompt_ids", "attention_mask"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="insert",
            )
        )

        assert metadata.global_indexes == list(range(gbs * num_n_samples))
        assert metadata.samples[0].partition_id == "train_0"
        assert sum([int(sample.fields.get("prompt_ids").production_status) for sample in metadata.samples]) == int(
            ProductionStatus.NOT_PRODUCED
        )
        assert sum([int(sample.fields.get("attention_mask").production_status) for sample in metadata.samples]) == int(
            ProductionStatus.NOT_PRODUCED
        )
        partition_index_range = ray.get(tq_controller.get_partition_index_range.remote(partition_id))
        assert partition_index_range == list(range(gbs * num_n_samples))

        print("✓ Initial get metadata correct")

        # Test update production status
        dtypes = {k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,)} for k in metadata.global_indexes}
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_backend_meta=None,
            )
        )
        assert success
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition.production_status is not None
        assert partition.production_status.size(0) == gbs * num_n_samples

        # Test for get production status
        global_index, production_status = ray.get(
            tq_controller.get_production_status.remote(
                partition_id=partition_id,
                data_fields=data_fields,
            )
        )
        # Verify global_index contains all expected indexes
        assert torch.equal(global_index, torch.tensor(range(gbs * num_n_samples), dtype=torch.long))
        # Verify all samples are produced for all fields (status should be 1)
        expected_production_status = torch.ones(gbs * num_n_samples, len(metadata.field_names), dtype=torch.int8)
        assert torch.equal(production_status, expected_production_status)
        print("✓ Get production status returns correct global_index and production_status")

        # Total fields should match the number of fields we added
        assert partition.total_fields_num == len(data_fields)

        # Allocated fields should be at least the number of actual fields
        assert partition.allocated_fields_num >= partition.total_fields_num

        # Check production status for the fields we added
        assert torch.equal(
            sum(partition.production_status[:, : len(data_fields)]),
            torch.Tensor([gbs * num_n_samples, gbs * num_n_samples]),
        )

        # Any additional allocated fields should be zero (unused)
        if partition.allocated_fields_num > len(data_fields):
            assert torch.equal(
                sum(partition.production_status[:, len(data_fields) :]),
                torch.zeros(1 * (partition.allocated_fields_num - len(data_fields))),
            )

        print(f"✓ Updated production status for partition {partition_id}")

        # Test for get consumption status BEFORE consumption
        global_index, consumption_status = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        # Verify global_index
        assert torch.equal(global_index, torch.tensor(range(gbs * num_n_samples), dtype=torch.long))
        # Verify all samples are NOT consumed yet (status should be 0)
        expected_consumption_status_before = torch.zeros(gbs * num_n_samples, dtype=torch.int8)
        assert torch.equal(consumption_status, expected_consumption_status_before)
        print("✓ Get consumption status returns correct global_index and status (before consumption)")

        # Test get metadate in fetch mode
        gen_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=["prompt_ids"],
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="fetch",
                task_name="generate_sequences",
            )
        )

        assert gen_meta.global_indexes == list(range(gbs * num_n_samples))
        assert gen_meta.samples[0].partition_id == "train_0"
        assert gen_meta.field_names == ["prompt_ids"]
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert torch.equal(partition.consumption_status["generate_sequences"], torch.ones(gbs * num_n_samples))
        print("✓ Get metadata in fetch mode correct")

        # Test for get consumption status AFTER consumption
        global_index, consumption_status = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        # Verify global_index
        assert torch.equal(global_index, torch.tensor(range(gbs * num_n_samples), dtype=torch.long))
        # Verify all samples are consumed (status should be 1)
        expected_consumption_status_after = torch.ones(gbs * num_n_samples, dtype=torch.int8)
        assert torch.equal(consumption_status, expected_consumption_status_after)
        print("✓ Get consumption status returns correct global_index and status (after consumption)")

        # Test get clear meta
        clear_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=[],
                partition_id=partition_id,
                mode="insert",
            )
        )
        assert clear_meta.global_indexes == list(range(gbs * num_n_samples))
        assert [sample.fields for sample in clear_meta.samples] == [{}] * (gbs * num_n_samples)
        print("✓ Clear metadata correct")

        # Test clear_partition
        ray.get(tq_controller.clear_partition.remote(partition_id))
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        partition_index_range = ray.get(tq_controller.get_partition_index_range.remote(partition_id))
        assert partition_index_range == []
        assert partition is None
        print("✓ Clear partition correct")

    def test_controller_reset_consumption(self, ray_setup):
        """Test reset_consumption functionality - allows data to be re-consumed"""
        gbs = 4
        num_n_samples = 2
        partition_id = "test_reset_consumption"

        tq_controller = TransferQueueController.remote()

        # Step 1: Create metadata in insert mode
        data_fields = ["prompt_ids", "attention_mask"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="insert",
            )
        )
        assert metadata.global_indexes == list(range(gbs * num_n_samples))

        # Step 2: Update production status
        dtypes = {k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,)} for k in metadata.global_indexes}
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
            )
        )
        assert success

        # Step 3: Verify consumption status BEFORE consumption (should be all zeros)
        global_index, consumption_status = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        expected_consumption_before = torch.zeros(gbs * num_n_samples, dtype=torch.int8)
        assert torch.equal(consumption_status, expected_consumption_before)
        print("✓ Consumption status before fetch is all zeros")

        # Step 4: Fetch data (mark as consumed)
        gen_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=["prompt_ids"],
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="fetch",
                task_name="generate_sequences",
            )
        )
        assert gen_meta.global_indexes == list(range(gbs * num_n_samples))

        # Step 5: Verify consumption status AFTER consumption (should be all ones)
        global_index, consumption_status = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        expected_consumption_after = torch.ones(gbs * num_n_samples, dtype=torch.int8)
        assert torch.equal(consumption_status, expected_consumption_after)
        print("✓ Consumption status after fetch is all ones")

        # Step 6: Reset consumption for specific task
        ray.get(
            tq_controller.reset_consumption.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )

        # Step 7: Verify consumption status is reset (should be all zeros again)
        global_index, consumption_status = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        expected_consumption_reset = torch.zeros(gbs * num_n_samples, dtype=torch.int8)
        assert torch.equal(consumption_status, expected_consumption_reset)
        print("✓ Consumption status after reset is all zeros")

        # Step 8: Consume again and test reset all tasks
        gen_meta_2 = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=["prompt_ids"],
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="fetch",
                task_name="generate_sequences",
            )
        )
        assert gen_meta_2.global_indexes == list(range(gbs * num_n_samples))

        # Also consume with another task
        gen_meta_3 = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=["attention_mask"],
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="fetch",
                task_name="another_task",
            )
        )
        assert gen_meta_3.global_indexes == list(range(gbs * num_n_samples))

        # Verify both tasks have consumed
        _, consumption_status_task1 = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        _, consumption_status_task2 = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="another_task",
            )
        )
        assert torch.equal(consumption_status_task1, torch.ones(gbs * num_n_samples, dtype=torch.int8))
        assert torch.equal(consumption_status_task2, torch.ones(gbs * num_n_samples, dtype=torch.int8))
        print("✓ Both tasks consumed successfully")

        # Step 9: Reset all tasks (task_name=None)
        ray.get(
            tq_controller.reset_consumption.remote(
                partition_id=partition_id,
                task_name=None,  # Reset all tasks
            )
        )

        # Step 10: Verify all tasks are reset
        _, consumption_status_task1_reset = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        _, consumption_status_task2_reset = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="another_task",
            )
        )
        assert torch.equal(consumption_status_task1_reset, torch.zeros(gbs * num_n_samples, dtype=torch.int8))
        assert torch.equal(consumption_status_task2_reset, torch.zeros(gbs * num_n_samples, dtype=torch.int8))
        print("✓ Reset all tasks successful - both tasks have zero consumption status")

        # Clean up
        ray.get(tq_controller.clear_partition.remote(partition_id))
        print("✓ Reset consumption test completed successfully")

    def test_controller_with_multi_partitions(self, ray_setup):
        gbs_1 = 8
        num_n_samples_1 = 4
        partition_id_1 = "train_0"

        gbs_2 = 16
        num_n_samples_2 = 1
        partition_id_2 = "val_0"

        gbs_3 = 32
        num_n_samples_3 = 2
        partition_id_3 = "train_1"

        tq_controller = TransferQueueController.remote()

        # Test get metadata in insert mode
        data_fields = ["prompt_ids", "attention_mask"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs_1 * num_n_samples_1,
                partition_id=partition_id_1,
                mode="insert",
            )
        )

        # Test update production status
        dtypes = {k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,)} for k in metadata.global_indexes}
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id_1,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
            )
        )
        assert success

        # Verify get production status returns correct data
        global_index_1, production_status_1 = ray.get(
            tq_controller.get_production_status.remote(
                partition_id=partition_id_1,
                data_fields=data_fields,
            )
        )
        expected_global_index_1 = torch.tensor(range(gbs_1 * num_n_samples_1), dtype=torch.long)
        assert torch.equal(global_index_1, expected_global_index_1)
        expected_production_status_1 = torch.ones(gbs_1 * num_n_samples_1, len(data_fields), dtype=torch.int8)
        assert torch.equal(production_status_1, expected_production_status_1)
        print("✓ Get production status for partition_1 returns correct global_index and status")

        # Test get metadate in fetch mode
        gen_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=["prompt_ids"],
                batch_size=gbs_1 * num_n_samples_1,
                partition_id=partition_id_1,
                mode="fetch",
                task_name="generate_sequences",
            )
        )
        assert gen_meta

        # Verify get consumption status after fetch (samples should be consumed)
        global_index_1_consumed, consumption_status_1 = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id_1,
                task_name="generate_sequences",
            )
        )
        assert torch.equal(global_index_1_consumed, expected_global_index_1)
        expected_consumption_status_1 = torch.ones(gbs_1 * num_n_samples_1, dtype=torch.int8)
        assert torch.equal(consumption_status_1, expected_consumption_status_1)
        print("✓ Get consumption status for partition_1 returns correct global_index and status (after fetch)")

        # Test get clear meta
        clear_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=[],
                partition_id=partition_id_1,
                mode="insert",
            )
        )
        assert clear_meta

        # =========================partition 2=============================#
        data_fields = ["prompt_ids", "attention_mask"]
        val_metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs_2 * num_n_samples_2,
                partition_id=partition_id_2,
                mode="insert",
            )
        )

        part1_index_range = gbs_1 * num_n_samples_1
        part2_index_range = gbs_2 * num_n_samples_2
        assert val_metadata.global_indexes == list(range(part1_index_range, part2_index_range + part1_index_range))
        assert val_metadata.samples[0].partition_id == "val_0"
        assert sum([int(sample.fields.get("prompt_ids").production_status) for sample in val_metadata.samples]) == int(
            ProductionStatus.NOT_PRODUCED
        )
        assert sum(
            [int(sample.fields.get("attention_mask").production_status) for sample in val_metadata.samples]
        ) == int(ProductionStatus.NOT_PRODUCED)
        partition_index_range = ray.get(tq_controller.get_partition_index_range.remote(partition_id_2))
        assert partition_index_range == list(range(part1_index_range, part2_index_range + part1_index_range))

        # Update production status
        dtypes = {k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool"} for k in val_metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,)} for k in val_metadata.global_indexes}
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id_2,
                global_indexes=val_metadata.global_indexes,
                field_names=val_metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
            )
        )
        assert success

        # Verify get production status for partition_2
        global_index_2, production_status_2 = ray.get(
            tq_controller.get_production_status.remote(
                partition_id=partition_id_2,
                data_fields=data_fields,
            )
        )
        expected_global_index_2 = torch.tensor(
            range(part1_index_range, part2_index_range + part1_index_range), dtype=torch.long
        )
        assert torch.equal(global_index_2, expected_global_index_2)
        expected_production_status_2 = torch.ones(part2_index_range, len(data_fields), dtype=torch.int8)
        assert torch.equal(production_status_2, expected_production_status_2)
        print("✓ Get production status for partition_2 returns correct global_index and status")

        # Verify get consumption status for partition_2 (before consumption - should be all zeros)
        global_index_2_consumed, consumption_status_2 = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id_2,
                task_name="generate_sequences",
            )
        )
        assert torch.equal(global_index_2_consumed, expected_global_index_2)
        expected_consumption_status_2 = torch.zeros(part2_index_range, dtype=torch.int8)
        assert torch.equal(consumption_status_2, expected_consumption_status_2)
        print("✓ Get consumption status for partition_2 returns correct global_index and status (before consumption)")

        # Clear partition 1
        partition_index_range_1 = ray.get(tq_controller.get_partition_index_range.remote(partition_id_1))
        assert partition_index_range_1
        ray.get(tq_controller.clear_partition.remote(partition_id_1))
        partition_1_after_clear = ray.get(tq_controller.get_partition_snapshot.remote(partition_id_1))
        partition_index_range_1_after_clear = ray.get(tq_controller.get_partition_index_range.remote(partition_id_1))

        assert not partition_index_range_1_after_clear
        assert partition_1_after_clear is None
        assert partition_index_range_1_after_clear == []

        partition_2 = ray.get(tq_controller.get_partition_snapshot.remote(partition_id_2))
        partition_index_range_2 = ray.get(tq_controller.get_partition_index_range.remote(partition_id_2))
        assert partition_index_range_2 == [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
        assert torch.all(
            partition_2.production_status[list(partition_index_range_2), : len(val_metadata.field_names)] == 1
        )
        print("✓ Only clear partition 1 correct")

        # =========================partition 3=============================#
        metadata_2 = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs_3 * num_n_samples_3,
                partition_id=partition_id_3,
                mode="insert",
            )
        )
        assert metadata_2.global_indexes == list(range(32)) + list(range(48, 80))
        assert metadata_2.samples[0].partition_id == "train_1"
        assert sum([int(sample.fields.get("prompt_ids").production_status) for sample in metadata_2.samples]) == int(
            ProductionStatus.NOT_PRODUCED
        )
        assert sum(
            [int(sample.fields.get("attention_mask").production_status) for sample in metadata_2.samples]
        ) == int(ProductionStatus.NOT_PRODUCED)
        partition_index_range = ray.get(tq_controller.get_partition_index_range.remote(partition_id_3))
        assert partition_index_range == list(range(32)) + list(range(48, 80))
        print("✓ Correctly assign partition_3")

    def test_controller_clear_meta(self, ray_setup):
        """Test clear_meta functionality for individual samples"""
        gbs = 4
        num_n_samples = 2
        partition_id = "test_clear_meta"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode
        data_fields = ["prompt_ids", "attention_mask"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs * num_n_samples,
                partition_id=partition_id,
                mode="insert",
            )
        )

        assert metadata.global_indexes == list(range(gbs * num_n_samples))

        # Update production status
        dtypes = {k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,)} for k in metadata.global_indexes}
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
            )
        )
        assert success

        # Get partition snapshot before clear
        partition_before = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition_before is not None
        assert len(partition_before.global_indexes) == gbs * num_n_samples
        assert set(partition_before.global_indexes) == set(range(gbs * num_n_samples))

        # Test clear_meta - clear first 4 samples (indexes 0-3)
        global_indexes_to_clear = [0, 1, 2, 3, 6]
        partition_ids_to_clear = [partition_id] * len(global_indexes_to_clear)

        ray.get(
            tq_controller.clear_meta.remote(
                global_indexes=global_indexes_to_clear,
                partition_ids=partition_ids_to_clear,
            )
        )

        # Check that only the cleared samples are affected
        partition_after = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition_after is not None

        # Verify production status is cleared for the specified indexes
        assert set(partition_after.global_indexes) == set([4, 5, 7])

        print("✓ Clear meta correct")


class TestTransferQueueControllerCustomMeta:
    """Integration tests for TransferQueueController custom_meta and custom_backend_meta methods.

    Note: In this codebase:
    - custom_meta: per-sample metadata (simple key-value pairs per sample)
    - custom_backend_meta: per-sample per-field metadata (stored via update_production_status)
    """

    def test_controller_with_custom_meta(self, ray_setup):
        """Test TransferQueueController with custom_backend_meta and custom_meta functionality"""

        batch_size = 3
        partition_id = "custom_meta_test"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode
        data_fields = ["prompt_ids", "attention_mask"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size,
                partition_id=partition_id,
                mode="insert",
            )
        )

        assert metadata.global_indexes == list(range(batch_size))

        # Build custom_backend_meta (per-sample per-field metadata)
        custom_backend_meta = {
            0: {"prompt_ids": {"token_count": 100}, "attention_mask": {"mask_ratio": 0.1}},
            1: {"prompt_ids": {"token_count": 120}, "attention_mask": {"mask_ratio": 0.15}},
            2: {"prompt_ids": {"token_count": 90}, "attention_mask": {"mask_ratio": 0.12}},
        }

        # Update production status with custom_backend_meta
        dtypes = {k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,)} for k in metadata.global_indexes}
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_backend_meta=custom_backend_meta,
            )
        )
        assert success

        # Get partition snapshot and verify custom_backend_meta is stored
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition is not None

        # Verify custom_backend_meta via get_field_custom_backend_meta
        result = partition.get_field_custom_backend_meta(list(range(batch_size)), ["prompt_ids", "attention_mask"])
        assert len(result) == batch_size
        assert result[0]["prompt_ids"]["token_count"] == 100
        assert result[2]["attention_mask"]["mask_ratio"] == 0.12

        print("✓ Controller set custom_backend_meta via update_production_status correct")

        # Now set custom_meta (per-sample metadata)
        # Format: {partition_id: {global_index: custom_meta_dict}}
        custom_meta = {
            partition_id: {
                0: {"sample_score": 0.9, "quality": "high"},
                1: {"sample_score": 0.8, "quality": "medium"},
                # You can set partial samples with custom_meta.
            }
        }

        # Verify set_custom_meta method exists and can be called
        ray.get(tq_controller.set_custom_meta.remote(partition_custom_meta=custom_meta))

        # Verify via partition snapshot
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        result = partition.get_custom_meta([0, 1])
        assert 0 in result
        assert result[0]["sample_score"] == 0.9
        assert result[0]["quality"] == "high"
        assert 1 in result
        assert result[1]["sample_score"] == 0.8
        assert 2 not in result

        # Init another partition
        new_partition_id = "custom_meta_test2"
        # Create metadata in insert mode
        data_fields = ["prompt_ids", "attention_mask"]
        new_metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=batch_size,
                partition_id=new_partition_id,
                mode="insert",
            )
        )

        # Update production status
        dtypes = {k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool"} for k in new_metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,)} for k in new_metadata.global_indexes}
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=new_partition_id,
                global_indexes=new_metadata.global_indexes,
                field_names=new_metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_backend_meta=None,
            )
        )
        assert success

        # Provide complicated case: update custom_meta with mixed partitions, and update previous custom_meta
        new_custom_meta = {
            new_partition_id: {
                3: {"sample_score": 1, "quality": "high"},
                4: {"sample_score": 0, "quality": "low"},
            },
            partition_id: {
                2: {"sample_score": 0.7, "quality": "high"},
                0: {"sample_score": 0.001, "quality": "low"},
            },
        }

        # update with new_custom_meta
        ray.get(tq_controller.set_custom_meta.remote(partition_custom_meta=new_custom_meta))

        # Verify via partition snapshot
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        result = partition.get_custom_meta([0, 1, 2])
        assert 0 in result
        assert result[0]["sample_score"] == 0.001  # updated!
        assert result[0]["quality"] == "low"  # updated!
        assert 1 in result  # unchanged
        assert result[1]["sample_score"] == 0.8  # unchanged
        assert 2 in result  # unchanged
        assert result[2]["sample_score"] == 0.7  # new

        new_partition = ray.get(tq_controller.get_partition_snapshot.remote(new_partition_id))
        result = new_partition.get_custom_meta([3, 4, 5])
        assert 3 in result
        assert result[3]["sample_score"] == 1
        assert result[3]["quality"] == "high"
        assert 4 in result
        assert result[4]["sample_score"] == 0
        assert 5 not in result  # 5 has no custom_meta, it will not return even we retrieve for 5

        # Clean up
        ray.get(tq_controller.clear_partition.remote(partition_id))
