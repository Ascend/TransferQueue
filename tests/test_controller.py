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
from transfer_queue.utils.utils import ProductionStatus  # noqa: E402


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
        assert partition_index_range == set(range(gbs * num_n_samples))

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
            )
        )
        assert success
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition.production_status is not None
        assert partition.production_status.size(0) == gbs * num_n_samples

        # Test for get production status
        production_status = ray.get(
            tq_controller.get_production_status.remote(
                partition_id=partition_id,
                data_fields=data_fields,
            )
        )
        assert production_status

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

        # Test for get consumption status
        consumption_status = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        assert torch.equal(consumption_status, torch.zeros(gbs * num_n_samples))

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

        # Test for get consumption status
        consumption_status = ray.get(
            tq_controller.get_consumption_status.remote(
                partition_id=partition_id,
                task_name="generate_sequences",
            )
        )
        assert torch.equal(consumption_status, torch.ones(gbs * num_n_samples))

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
        assert partition_index_range == set()
        assert partition is None
        print("✓ Clear partition correct")

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
        assert partition_index_range == set(range(part1_index_range, part2_index_range + part1_index_range))

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

        # Clear partition 1
        partition_index_range_1 = ray.get(tq_controller.get_partition_index_range.remote(partition_id_1))
        assert partition_index_range_1
        ray.get(tq_controller.clear_partition.remote(partition_id_1))
        partition_1_after_clear = ray.get(tq_controller.get_partition_snapshot.remote(partition_id_1))
        partition_index_range_1_after_clear = ray.get(tq_controller.get_partition_index_range.remote(partition_id_1))

        assert not partition_index_range_1_after_clear
        assert partition_1_after_clear is None
        assert partition_index_range_1_after_clear == set()

        partition_2 = ray.get(tq_controller.get_partition_snapshot.remote(partition_id_2))
        partition_index_range_2 = ray.get(tq_controller.get_partition_index_range.remote(partition_id_2))
        assert partition_index_range_2 == set([32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47])
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
        assert partition_index_range == set(list(range(32)) + list(range(48, 80)))
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


class TestCustomMeta:
    """Test suite for custom_meta functionality in TransferQueueController"""

    def test_custom_meta_basic_storage_and_retrieval(self, ray_setup):
        """Test basic custom_meta storage via update_production_status and retrieval via get_metadata"""
        gbs = 4
        partition_id = "test_custom_meta_basic"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode
        data_fields = ["prompt_ids", "attention_mask"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="insert",
            )
        )

        assert metadata.global_indexes == list(range(gbs))

        # Update production status with custom_meta
        dtypes = {k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,)} for k in metadata.global_indexes}
        custom_meta = {
            k: {"prompt_ids": {"token_count": 100 + k}, "attention_mask": {"mask_ratio": 0.1 * k}}
            for k in metadata.global_indexes
        }

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_meta=custom_meta,
            )
        )
        assert success

        # Verify custom_meta is stored in partition
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition is not None
        assert len(partition.field_custom_metas) == gbs

        for idx in metadata.global_indexes:
            assert idx in partition.field_custom_metas
            assert "prompt_ids" in partition.field_custom_metas[idx]
            assert "attention_mask" in partition.field_custom_metas[idx]
            assert partition.field_custom_metas[idx]["prompt_ids"]["token_count"] == 100 + idx
            assert partition.field_custom_metas[idx]["attention_mask"]["mask_ratio"] == 0.1 * idx

        print("✓ Basic custom_meta storage correct")

        # Retrieve via get_metadata in fetch mode and verify custom_meta is in batch_meta
        fetch_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="fetch",
                task_name="test_task",
            )
        )

        assert fetch_meta is not None
        custom_meta_retrieved = fetch_meta.get_all_custom_meta()
        assert custom_meta_retrieved is not None

        for idx in metadata.global_indexes:
            assert idx in custom_meta_retrieved
            assert "prompt_ids" in custom_meta_retrieved[idx]
            assert "attention_mask" in custom_meta_retrieved[idx]

        print("✓ Basic custom_meta retrieval via get_metadata correct")

    def test_custom_meta_with_partial_fields(self, ray_setup):
        """Test custom_meta retrieval when only requesting subset of fields"""
        gbs = 4
        partition_id = "test_custom_meta_partial"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode with multiple fields
        data_fields = ["prompt_ids", "attention_mask", "labels"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="insert",
            )
        )

        # Update production status with custom_meta for all fields
        dtypes = {
            k: {"prompt_ids": "torch.int64", "attention_mask": "torch.bool", "labels": "torch.int64"}
            for k in metadata.global_indexes
        }
        shapes = {k: {"prompt_ids": (32,), "attention_mask": (32,), "labels": (32,)} for k in metadata.global_indexes}
        custom_meta = {
            k: {
                "prompt_ids": {"meta_prompt": f"prompt_{k}"},
                "attention_mask": {"meta_mask": f"mask_{k}"},
                "labels": {"meta_label": f"label_{k}"},
            }
            for k in metadata.global_indexes
        }

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_meta=custom_meta,
            )
        )
        assert success

        # Fetch with only a subset of fields
        subset_fields = ["prompt_ids", "labels"]
        fetch_meta = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=subset_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="fetch",
                task_name="test_task",
            )
        )

        assert fetch_meta is not None
        custom_meta_retrieved = fetch_meta.get_all_custom_meta()
        assert custom_meta_retrieved is not None

        # Verify only requested fields are in custom_meta
        for idx in metadata.global_indexes:
            assert idx in custom_meta_retrieved
            assert "prompt_ids" in custom_meta_retrieved[idx]
            assert "labels" in custom_meta_retrieved[idx]
            # attention_mask should not be in the custom_meta since it wasn't requested
            assert "attention_mask" not in custom_meta_retrieved[idx]

        print("✓ Custom_meta with partial fields correct")

    def test_custom_meta_length_mismatch_returns_false(self, ray_setup):
        """Test that custom_meta length mismatch with global_indices returns False"""
        gbs = 4
        partition_id = "test_custom_meta_mismatch"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode
        data_fields = ["prompt_ids"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="insert",
            )
        )

        # Prepare mismatched custom_meta (fewer entries than global_indexes)
        dtypes = {k: {"prompt_ids": "torch.int64"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,)} for k in metadata.global_indexes}
        # Only provide custom_meta for half the samples
        custom_meta = {k: {"prompt_ids": {"meta": k}} for k in metadata.global_indexes[:2]}

        # The method should return False when there's a length mismatch
        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_meta=custom_meta,
            )
        )
        assert success is False, "Expected update_production_status to return False for length mismatch"

        print("✓ Custom_meta length mismatch error handling correct")

    def test_custom_meta_none_does_not_store(self, ray_setup):
        """Test that passing None for custom_meta doesn't create custom_meta entries"""
        gbs = 4
        partition_id = "test_custom_meta_none"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode
        data_fields = ["prompt_ids"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="insert",
            )
        )

        # Update production status without custom_meta (None)
        dtypes = {k: {"prompt_ids": "torch.int64"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,)} for k in metadata.global_indexes}

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_meta=None,
            )
        )
        assert success

        # Verify no custom_meta is stored
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition is not None
        assert len(partition.field_custom_metas) == 0

        print("✓ Custom_meta None handling correct")

    def test_custom_meta_preserved_after_partial_clear(self, ray_setup):
        """Test that custom_meta for non-cleared samples is preserved after clear_meta"""
        gbs = 4
        partition_id = "test_custom_meta_clear"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode
        data_fields = ["prompt_ids"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="insert",
            )
        )

        # Update production status with custom_meta
        dtypes = {k: {"prompt_ids": "torch.int64"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,)} for k in metadata.global_indexes}
        custom_meta = {k: {"prompt_ids": {"sample_id": k * 10}} for k in metadata.global_indexes}

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_meta=custom_meta,
            )
        )
        assert success

        # Clear only first 2 samples
        global_indexes_to_clear = [0, 1]
        partition_ids_to_clear = [partition_id] * len(global_indexes_to_clear)

        ray.get(
            tq_controller.clear_meta.remote(
                global_indexes=global_indexes_to_clear,
                partition_ids=partition_ids_to_clear,
            )
        )

        # Verify custom_meta is cleared for cleared samples and preserved for others
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition is not None

        # Cleared samples should not have custom_meta
        assert 0 not in partition.field_custom_metas
        assert 1 not in partition.field_custom_metas

        # Non-cleared samples should still have custom_meta
        assert 2 in partition.field_custom_metas
        assert 3 in partition.field_custom_metas
        assert partition.field_custom_metas[2]["prompt_ids"]["sample_id"] == 20
        assert partition.field_custom_metas[3]["prompt_ids"]["sample_id"] == 30

        print("✓ Custom_meta preserved after partial clear correct")

    def test_custom_meta_update_merges_values(self, ray_setup):
        """Test that updating custom_meta for the same sample merges values"""
        gbs = 2
        partition_id = "test_custom_meta_merge"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode with first field
        data_fields_1 = ["prompt_ids"]
        metadata_1 = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields_1,
                batch_size=gbs,
                partition_id=partition_id,
                mode="insert",
            )
        )

        # First update with custom_meta for prompt_ids
        dtypes_1 = {k: {"prompt_ids": "torch.int64"} for k in metadata_1.global_indexes}
        shapes_1 = {k: {"prompt_ids": (32,)} for k in metadata_1.global_indexes}
        custom_meta_1 = {k: {"prompt_ids": {"first_update": True}} for k in metadata_1.global_indexes}

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata_1.global_indexes,
                field_names=metadata_1.field_names,
                dtypes=dtypes_1,
                shapes=shapes_1,
                custom_meta=custom_meta_1,
            )
        )
        assert success

        # Second update with new field and its custom_meta
        data_fields_2 = ["attention_mask"]
        dtypes_2 = {k: {"attention_mask": "torch.bool"} for k in metadata_1.global_indexes}
        shapes_2 = {k: {"attention_mask": (32,)} for k in metadata_1.global_indexes}
        custom_meta_2 = {k: {"attention_mask": {"second_update": True}} for k in metadata_1.global_indexes}

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata_1.global_indexes,
                field_names=data_fields_2,
                dtypes=dtypes_2,
                shapes=shapes_2,
                custom_meta=custom_meta_2,
            )
        )
        assert success

        # Verify both custom_meta entries are present (merged)
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition is not None

        for idx in metadata_1.global_indexes:
            assert idx in partition.field_custom_metas
            assert "prompt_ids" in partition.field_custom_metas[idx]
            assert "attention_mask" in partition.field_custom_metas[idx]
            assert partition.field_custom_metas[idx]["prompt_ids"]["first_update"] is True
            assert partition.field_custom_metas[idx]["attention_mask"]["second_update"] is True

        print("✓ Custom_meta merge on update correct")

    def test_custom_meta_with_complex_nested_data(self, ray_setup):
        """Test custom_meta with complex nested data structures"""
        gbs = 2
        partition_id = "test_custom_meta_complex"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode
        data_fields = ["prompt_ids"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="insert",
            )
        )

        # Create complex nested custom_meta
        dtypes = {k: {"prompt_ids": "torch.int64"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,)} for k in metadata.global_indexes}
        custom_meta = {
            k: {
                "prompt_ids": {
                    "nested_dict": {"level1": {"level2": {"value": k}}},
                    "list_data": [1, 2, 3, k],
                    "mixed_types": {"string": "test", "number": 42, "float": 3.14, "bool": True},
                }
            }
            for k in metadata.global_indexes
        }

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_meta=custom_meta,
            )
        )
        assert success

        # Verify complex nested data is preserved
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition is not None

        for idx in metadata.global_indexes:
            stored_meta = partition.field_custom_metas[idx]["prompt_ids"]
            assert stored_meta["nested_dict"]["level1"]["level2"]["value"] == idx
            assert stored_meta["list_data"] == [1, 2, 3, idx]
            assert stored_meta["mixed_types"]["string"] == "test"
            assert stored_meta["mixed_types"]["number"] == 42
            assert stored_meta["mixed_types"]["float"] == 3.14
            assert stored_meta["mixed_types"]["bool"] is True

        print("✓ Complex nested custom_meta correct")

    def test_custom_meta_cleared_on_partition_clear(self, ray_setup):
        """Test that custom_meta is fully cleared when partition is cleared"""
        gbs = 4
        partition_id = "test_custom_meta_partition_clear"

        tq_controller = TransferQueueController.remote()

        # Create metadata in insert mode
        data_fields = ["prompt_ids"]
        metadata = ray.get(
            tq_controller.get_metadata.remote(
                data_fields=data_fields,
                batch_size=gbs,
                partition_id=partition_id,
                mode="insert",
            )
        )

        # Update production status with custom_meta
        dtypes = {k: {"prompt_ids": "torch.int64"} for k in metadata.global_indexes}
        shapes = {k: {"prompt_ids": (32,)} for k in metadata.global_indexes}
        custom_meta = {k: {"prompt_ids": {"data": k}} for k in metadata.global_indexes}

        success = ray.get(
            tq_controller.update_production_status.remote(
                partition_id=partition_id,
                global_indexes=metadata.global_indexes,
                field_names=metadata.field_names,
                dtypes=dtypes,
                shapes=shapes,
                custom_meta=custom_meta,
            )
        )
        assert success

        # Clear the entire partition
        ray.get(tq_controller.clear_partition.remote(partition_id))

        # Verify partition is gone
        partition = ray.get(tq_controller.get_partition_snapshot.remote(partition_id))
        assert partition is None

        print("✓ Custom_meta cleared on partition clear correct")
