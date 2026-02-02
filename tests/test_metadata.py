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

"""Unit tests for TransferQueue metadata module - Learning Examples."""

import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta  # noqa: E402
from transfer_queue.utils.enum_utils import ProductionStatus  # noqa: E402


class TestFieldMeta:
    """FieldMeta learning examples."""

    def test_field_meta_is_ready(self):
        """Test the is_ready property based on production status."""
        field_ready = FieldMeta(
            name="test_field", dtype=torch.float32, shape=(2, 3), production_status=ProductionStatus.READY_FOR_CONSUME
        )
        assert field_ready.is_ready is True

        field_not_ready = FieldMeta(
            name="test_field", dtype=torch.float32, shape=(2, 3), production_status=ProductionStatus.NOT_PRODUCED
        )
        assert field_not_ready.is_ready is False


class TestSampleMeta:
    """SampleMeta learning examples."""

    def test_sample_meta_union(self):
        """Example: Union fields from two samples with matching global indexes."""
        # Create first sample
        fields1 = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
        }
        sample1 = SampleMeta(partition_id="partition_0", global_index=0, fields=fields1)

        # Create second sample with additional fields
        fields2 = {
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
            "field3": FieldMeta(name="field3", dtype=torch.bool, shape=(4,)),
        }
        sample2 = SampleMeta(partition_id="partition_0", global_index=0, fields=fields2)

        # Union samples
        result = sample1.union(sample2)

        # Result contains all fields from both samples
        assert "field1" in result.fields
        assert "field2" in result.fields  # From sample2
        assert "field3" in result.fields

    def test_sample_meta_union_validation_error(self):
        """Example: Union validation catches mismatched global indexes."""
        sample1 = SampleMeta(
            partition_id="partition_0",
            global_index=0,
            fields={"field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,))},
        )

        sample2 = SampleMeta(
            partition_id="partition_0",
            global_index=1,  # Different global index
            fields={"field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,))},
        )

        with pytest.raises(ValueError) as exc_info:
            sample1.union(sample2, validate=True)
        assert "Global indexes" in str(exc_info.value)

    def test_sample_meta_add_fields(self):
        """Example: Add new fields to a sample."""
        initial_fields = {
            "field1": FieldMeta(
                name="field1", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        sample = SampleMeta(partition_id="partition_0", global_index=0, fields=initial_fields)

        new_fields = {
            "field2": FieldMeta(
                name="field2", dtype=torch.int64, shape=(3,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        sample.add_fields(new_fields)

        assert "field1" in sample.fields
        assert "field2" in sample.fields
        assert sample.is_ready is True

    def test_sample_meta_select_fields(self):
        """Example: Select specific fields from a sample."""
        fields = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
            "field3": FieldMeta(name="field3", dtype=torch.bool, shape=(4,)),
        }
        sample = SampleMeta(partition_id="partition_0", global_index=0, fields=fields)

        # Select only field1 and field3
        selected_sample = sample.select_fields(["field1", "field3"])

        assert "field1" in selected_sample.fields
        assert "field3" in selected_sample.fields
        assert "field2" not in selected_sample.fields
        # Original sample is unchanged
        assert len(sample.fields) == 3
        # Selected sample has correct metadata
        assert selected_sample.fields["field1"].dtype == torch.float32
        assert selected_sample.fields["field1"].shape == (2,)
        assert selected_sample.global_index == 0
        assert selected_sample.partition_id == "partition_0"

    def test_sample_meta_select_fields_with_nonexistent_fields(self):
        """Example: Select fields ignores non-existent field names."""
        fields = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
        }
        sample = SampleMeta(partition_id="partition_0", global_index=0, fields=fields)

        # Try to select a field that doesn't exist
        selected_sample = sample.select_fields(["field1", "nonexistent_field"])

        # Only existing field is selected
        assert "field1" in selected_sample.fields
        assert "nonexistent_field" not in selected_sample.fields
        assert "field2" not in selected_sample.fields

    def test_sample_meta_select_fields_empty_list(self):
        """Example: Select with empty field list returns sample with no fields."""
        fields = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
        }
        sample = SampleMeta(partition_id="partition_0", global_index=0, fields=fields)

        # Select with empty list
        selected_sample = sample.select_fields([])

        assert len(selected_sample.fields) == 0
        assert selected_sample.global_index == 0
        assert selected_sample.partition_id == "partition_0"


class TestBatchMeta:
    """BatchMeta learning examples - Core Operations."""

    def test_batch_meta_chunk(self):
        """Example: Split a batch into multiple chunks."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [SampleMeta(partition_id="partition_0", global_index=i, fields=fields) for i in range(10)]
        batch = BatchMeta(
            samples=samples,
            custom_meta={i: {"uid": i} for i in range(10)},
            _custom_backend_meta={i: {"test_field": {"dtype": torch.float32}} for i in range(10)},
        )

        # Chunk into 3 parts
        chunks = batch.chunk(3)

        assert len(chunks) == 3
        assert len(chunks[0]) == 4  # First chunk gets extra element
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3

        # validate custom_meta is chunked
        assert 0 in chunks[0].custom_meta
        assert 1 in chunks[0].custom_meta
        assert 2 in chunks[0].custom_meta
        assert 3 in chunks[0].custom_meta
        assert 4 not in chunks[0].custom_meta
        assert 4 in chunks[1].custom_meta

        # validate _custom_backend_meta is chunked
        assert 0 in chunks[0]._custom_backend_meta
        assert 1 in chunks[0]._custom_backend_meta
        assert 2 in chunks[0]._custom_backend_meta
        assert 3 in chunks[0]._custom_backend_meta
        assert 4 not in chunks[0]._custom_backend_meta
        assert 4 in chunks[1]._custom_backend_meta

    def test_batch_meta_chunk_by_partition(self):
        """Example: Split a batch into multiple chunks."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [SampleMeta(partition_id=f"partition_{i % 4}", global_index=i, fields=fields) for i in range(10)]
        batch = BatchMeta(
            samples=samples,
            custom_meta={i: {"uid": i} for i in range(10)},
            _custom_backend_meta={i: {"test_field": {"dtype": torch.float32}} for i in range(10)},
        )

        # Chunk according to partition_id
        chunks = batch.chunk_by_partition()

        assert len(chunks) == 4
        assert len(chunks[0]) == 3
        assert chunks[0].partition_ids == ["partition_0", "partition_0", "partition_0"]
        assert chunks[0].global_indexes == [0, 4, 8]
        assert len(chunks[1]) == 3
        assert chunks[1].partition_ids == ["partition_1", "partition_1", "partition_1"]
        assert chunks[1].global_indexes == [1, 5, 9]
        assert len(chunks[2]) == 2
        assert chunks[2].partition_ids == ["partition_2", "partition_2"]
        assert chunks[2].global_indexes == [2, 6]
        assert len(chunks[3]) == 2
        assert chunks[3].partition_ids == ["partition_3", "partition_3"]
        assert chunks[3].global_indexes == [3, 7]

        # validate _custom_backend_meta is chunked
        assert 0 in chunks[0].custom_meta
        assert 4 in chunks[0].custom_meta
        assert 8 in chunks[0].custom_meta
        assert 1 not in chunks[0].custom_meta
        assert 1 in chunks[1].custom_meta

        # validate _custom_backend_meta is chunked
        assert 0 in chunks[0]._custom_backend_meta
        assert 4 in chunks[0]._custom_backend_meta
        assert 8 in chunks[0]._custom_backend_meta
        assert 1 not in chunks[0]._custom_backend_meta
        assert 1 in chunks[1]._custom_backend_meta

    def test_batch_meta_init_validation_error_different_field_names(self):
        """Example: Init validation catches samples with different field names."""
        # Create first sample with field1
        fields1 = {"field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,))}
        sample1 = SampleMeta(partition_id="partition_0", global_index=0, fields=fields1)

        # Create second sample with field2
        fields2 = {"field2": FieldMeta(name="field2", dtype=torch.float32, shape=(2,))}
        sample2 = SampleMeta(partition_id="partition_0", global_index=1, fields=fields2)

        # Attempt to create BatchMeta with samples having different field names
        with pytest.raises(ValueError) as exc_info:
            BatchMeta(samples=[sample1, sample2])
        assert "All samples in BatchMeta must have the same field_names." in str(exc_info.value)

    def test_batch_meta_concat(self):
        """Example: Concatenate multiple batches."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }

        # Create two batches
        batch1 = BatchMeta(
            samples=[
                SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
                SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
            ],
            custom_meta={i: {"uid": i} for i in [0, 1]},
            _custom_backend_meta={i: {"test_field": {"dtype": torch.float32}} for i in [0, 1]},
        )

        batch2 = BatchMeta(
            samples=[
                SampleMeta(partition_id="partition_0", global_index=2, fields=fields),
                SampleMeta(partition_id="partition_0", global_index=3, fields=fields),
            ],
            custom_meta={i: {"uid": i} for i in [2, 3]},
            _custom_backend_meta={i: {"test_field": {"dtype": torch.float32}} for i in [2, 3]},
        )

        # Concatenate batches
        result = BatchMeta.concat([batch1, batch2])

        assert len(result) == 4
        assert result.global_indexes == [0, 1, 2, 3]
        assert result.custom_meta == {i: {"uid": i} for i in [0, 1, 2, 3]}
        assert result._custom_backend_meta == {i: {"test_field": {"dtype": torch.float32}} for i in [0, 1, 2, 3]}

    def test_batch_meta_concat_with_tensor_extra_info(self):
        """Example: Concat handles tensor extra_info by concatenating along dim=0."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }

        batch1 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])
        batch1.extra_info["tensor"] = torch.randn(3, 4)
        batch1.extra_info["scalar"] = torch.tensor(1.0)

        batch2 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=1, fields=fields)])
        batch2.extra_info["tensor"] = torch.randn(3, 4)
        batch2.extra_info["scalar"] = torch.tensor(2.0)

        result = BatchMeta.concat([batch1, batch2])

        # Tensors are concatenated along dim=0
        assert result.extra_info["tensor"].shape == (6, 4)
        # Scalars are stacked
        assert result.extra_info["scalar"].shape == (2,)

    def test_batch_meta_concat_with_non_tensor_stack(self):
        """Example: Concat handles NonTensorStack extra_info."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }

        batch1 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])
        batch1.extra_info["non_tensor"] = NonTensorStack(1, 2, 3)

        batch2 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=1, fields=fields)])
        batch2.extra_info["non_tensor"] = NonTensorStack(4, 5, 6)

        result = BatchMeta.concat([batch1, batch2])

        # NonTensorStack is stacked
        assert isinstance(result.extra_info["non_tensor"], NonTensorStack)
        assert result.extra_info["non_tensor"].batch_size == torch.Size([2, 3])

    def test_batch_meta_concat_with_list_extra_info(self):
        """Example: Concat handles list extra_info by flattening."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }

        batch1 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])
        batch1.extra_info["list"] = [1, 2, 3]

        batch2 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=1, fields=fields)])
        batch2.extra_info["list"] = [4, 5, 6]

        result = BatchMeta.concat([batch1, batch2])

        # Lists are flattened
        assert result.extra_info["list"] == [1, 2, 3, 4, 5, 6]

    def test_batch_meta_concat_with_mixed_types(self):
        """Example: Concat handles mixed extra_info types correctly."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }

        batch1 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])
        batch1.extra_info["tensor"] = torch.randn(3, 4)
        batch1.extra_info["list"] = [1, 2, 3]
        batch1.extra_info["string"] = "hello"
        batch1.extra_info["non_tensor"] = NonTensorStack(1, 2, 3)

        batch2 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=1, fields=fields)])
        batch2.extra_info["tensor"] = torch.randn(3, 4)
        batch2.extra_info["list"] = [4, 5]
        batch2.extra_info["string"] = "world"
        batch2.extra_info["non_tensor"] = NonTensorStack(4, 5, 6)

        result = BatchMeta.concat([batch1, batch2])

        # Each type is handled appropriately
        assert result.extra_info["tensor"].shape == (6, 4)  # Concatenated
        assert result.extra_info["list"] == [1, 2, 3, 4, 5]  # Flattened
        assert result.extra_info["string"] == "world"  # Last value wins
        assert isinstance(result.extra_info["non_tensor"], NonTensorStack)  # Stacked

    def test_batch_meta_union(self):
        """Example: Union two batches with matching global indexes."""
        fields1 = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
        }
        fields2 = {
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
            "field3": FieldMeta(name="field3", dtype=torch.bool, shape=(4,)),
        }

        batch1 = BatchMeta(
            samples=[
                SampleMeta(partition_id="partition_0", global_index=0, fields=fields1),
                SampleMeta(partition_id="partition_0", global_index=1, fields=fields1),
            ],
            _custom_backend_meta={
                i: {"field1": {"dtype": torch.float32}, "field2": {"dtype": torch.int64}} for i in [0, 1]
            },
        )
        batch1.extra_info["info1"] = "value1"

        batch2 = BatchMeta(
            samples=[
                SampleMeta(partition_id="partition_0", global_index=0, fields=fields2),
                SampleMeta(partition_id="partition_0", global_index=1, fields=fields2),
            ],
            _custom_backend_meta={
                i: {"field2": {"dtype": torch.int64}, "field3": {"dtype": torch.bool}} for i in [0, 1]
            },
        )
        batch2.extra_info["info2"] = "value2"

        result = batch1.union(batch2)

        assert len(result) == 2
        # All fields are present
        for sample in result.samples:
            assert "field1" in sample.fields
            assert "field2" in sample.fields
            assert "field3" in sample.fields
        # Extra info is merged
        assert result.extra_info["info1"] == "value1"
        assert result.extra_info["info2"] == "value2"

        # _custom_backend_meta is merged
        assert result._custom_backend_meta == {
            i: {"field1": {"dtype": torch.float32}, "field2": {"dtype": torch.int64}, "field3": {"dtype": torch.bool}}
            for i in [0, 1]
        }

    def test_batch_meta_union_validation(self):
        """Example: Union validation catches mismatched conditions."""
        fields = {"test_field": FieldMeta(name="test_field", dtype=torch.float32, shape=(2,))}

        batch1 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])

        batch2 = BatchMeta(
            samples=[
                SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
                SampleMeta(partition_id="partition_0", global_index=1, fields=fields),  # Different size
            ]
        )

        with pytest.raises(ValueError) as exc_info:
            batch1.union(batch2, validate=True)
        assert "Batch sizes do not match" in str(exc_info.value)

    def test_batch_meta_reorder(self):
        """Example: Reorder samples in a batch."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=4, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=5, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=6, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Reorder to [2, 0, 1]
        batch.reorder([2, 0, 1])

        assert batch.global_indexes == [6, 4, 5]
        # Batch indexes are updated
        assert batch.samples[0].batch_index == 0
        assert batch.samples[1].batch_index == 1
        assert batch.samples[2].batch_index == 2

    def test_batch_meta_add_fields(self):
        """Example: Add fields from TensorDict to all samples."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Create TensorDict with new fields
        tensor_dict = TensorDict({"new_field1": torch.randn(2, 3), "new_field2": torch.randn(2, 5)}, batch_size=[2])

        batch.add_fields(tensor_dict)

        # Fields are added to all samples
        for sample in batch.samples:
            assert "new_field1" in sample.fields
            assert "new_field2" in sample.fields
            assert sample.is_ready is True

    def test_batch_meta_select_fields(self):
        """Example: Select specific fields from all samples in a batch."""
        fields = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
            "field3": FieldMeta(name="field3", dtype=torch.bool, shape=(4,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(
            samples=samples,
            extra_info={"test_key": "test_value"},
            _custom_backend_meta={
                i: {
                    "field1": {"dtype": torch.float32},
                    "field2": {"dtype": torch.int64},
                    "field3": {"dtype": torch.bool},
                }
                for i in [0, 1]
            },
        )

        # Select only field1 and field3
        selected_batch = batch.select_fields(["field1", "field3"])

        # Check all samples have correct fields
        assert len(selected_batch) == 2
        for sample in selected_batch.samples:
            assert "field1" in sample.fields
            assert "field3" in sample.fields
            assert "field2" not in sample.fields
        # Original batch is unchanged
        assert len(batch.samples[0].fields) == 3
        # Extra info is preserved
        assert selected_batch.extra_info["test_key"] == "test_value"
        # Global indexes are preserved
        assert selected_batch.global_indexes == [0, 1]

        # _custom_backend_meta is selected
        assert "field1" in selected_batch._custom_backend_meta[0]
        assert "field2" not in selected_batch._custom_backend_meta[0]
        assert "field3" in selected_batch._custom_backend_meta[0]
        assert "field1" in selected_batch._custom_backend_meta[1]
        assert "field2" not in selected_batch._custom_backend_meta[1]
        assert "field3" in selected_batch._custom_backend_meta[1]

    def test_batch_meta_select_fields_with_nonexistent_fields(self):
        """Example: Select fields ignores non-existent field names in batch."""
        fields = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Try to select fields including non-existent ones
        selected_batch = batch.select_fields(["field1", "nonexistent_field"])

        # Only existing fields are selected
        for sample in selected_batch.samples:
            assert "field1" in sample.fields
            assert "nonexistent_field" not in sample.fields
            assert "field2" not in sample.fields

    def test_batch_meta_select_fields_empty_list(self):
        """Example: Select with empty field list returns batch with no fields."""
        fields = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Select with empty list
        selected_batch = batch.select_fields([])

        assert len(selected_batch) == 2
        for sample in selected_batch.samples:
            assert len(sample.fields) == 0
        # Global indexes are preserved
        assert selected_batch.global_indexes == [0, 1]

    def test_batch_meta_select_fields_single_sample(self):
        """Example: Select fields works correctly for batch with single sample."""
        fields = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
        }
        sample = SampleMeta(partition_id="partition_0", global_index=0, fields=fields)
        batch = BatchMeta(samples=[sample])

        # Select only field2
        selected_batch = batch.select_fields(["field2"])

        assert len(selected_batch) == 1
        assert "field2" in selected_batch.samples[0].fields
        assert "field1" not in selected_batch.samples[0].fields

    def test_batch_meta_select_fields_preserves_field_metadata(self):
        """Example: Selected fields preserve their original metadata."""
        fields = {
            "field1": FieldMeta(
                name="field1", dtype=torch.float32, shape=(2, 3), production_status=ProductionStatus.READY_FOR_CONSUME
            ),
            "field2": FieldMeta(
                name="field2", dtype=torch.int64, shape=(5,), production_status=ProductionStatus.NOT_PRODUCED
            ),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Select field1
        selected_batch = batch.select_fields(["field1"])
        selected_field = selected_batch.samples[0].fields["field1"]

        assert selected_field.dtype == torch.float32
        assert selected_field.shape == (2, 3)
        assert selected_field.production_status == ProductionStatus.READY_FOR_CONSUME
        assert selected_field.name == "field1"

    def test_batch_meta_select_samples(self):
        """Example: Select specific samples from a batch."""
        fields = {
            "field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,)),
            "field2": FieldMeta(name="field2", dtype=torch.int64, shape=(3,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=4, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=5, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=6, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=7, fields=fields),
        ]
        batch = BatchMeta(samples=samples, extra_info={"test_key": "test_value"})

        # Select samples at indices [0, 2]
        selected_batch = batch.select_samples([0, 2])  # This will select the first two samples with global_index=4/5

        # Check number of samples
        assert len(selected_batch) == 2
        # Check global indexes
        assert selected_batch.global_indexes == [4, 6]
        # Check fields are preserved
        for sample in selected_batch.samples:
            assert "field1" in sample.fields
            assert "field2" in sample.fields
        # Original batch is unchanged
        assert len(batch) == 4
        # Extra info is preserved
        assert selected_batch.extra_info["test_key"] == "test_value"

    def test_batch_meta_select_samples_all_indices(self):
        """Example: Select all samples using complete index list."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=4, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=5, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=6, fields=fields),
        ]
        batch = BatchMeta(samples=samples, extra_info={"test_key": "test_value"})

        # Select all samples
        selected_batch = batch.select_samples([0, 1, 2])

        # All samples are selected
        assert len(selected_batch) == 3
        assert selected_batch.global_indexes == [4, 5, 6]
        # Extra info is preserved
        assert selected_batch.extra_info["test_key"] == "test_value"

    def test_batch_meta_select_samples_single_sample(self):
        """Example: Select a single sample from batch."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=2, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Select only the middle sample
        selected_batch = batch.select_samples([1])

        assert len(selected_batch) == 1
        assert selected_batch.global_indexes == [1]
        assert selected_batch.samples[0].batch_index == 0  # New batch index

    def test_batch_meta_select_samples_empty_list(self):
        """Example: Select with empty list returns empty batch."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples, extra_info={"test_key": "test_value"})

        # Select with empty list
        selected_batch = batch.select_samples([])

        assert len(selected_batch) == 0
        assert selected_batch.global_indexes == []
        # Extra info is still preserved
        assert selected_batch.extra_info["test_key"] == "test_value"

    def test_batch_meta_select_samples_reverse_order(self):
        """Example: Select samples in reverse order."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=2, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Select samples in reverse order
        selected_batch = batch.select_samples([2, 1, 0])

        assert len(selected_batch) == 3
        assert selected_batch.global_indexes == [2, 1, 0]
        # Batch indexes are re-assigned
        assert selected_batch.samples[0].global_index == 2
        assert selected_batch.samples[1].global_index == 1
        assert selected_batch.samples[2].global_index == 0

    def test_batch_meta_select_samples_with_extra_info(self):
        """Example: Select samples preserves all extra info types."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Add various extra info types
        batch.extra_info["tensor"] = torch.randn(3, 4)
        batch.extra_info["string"] = "test_string"
        batch.extra_info["number"] = 42
        batch.extra_info["list"] = [1, 2, 3]

        # Select one sample
        selected_batch = batch.select_samples([0])

        # All extra info is preserved
        assert "tensor" in selected_batch.extra_info
        assert selected_batch.extra_info["string"] == "test_string"
        assert selected_batch.extra_info["number"] == 42
        assert selected_batch.extra_info["list"] == [1, 2, 3]

    # =====================================================
    # Custom Meta Tests
    # =====================================================

    def test_batch_meta_set_custom_meta_basic(self):
        """Test set_custom_meta sets metadata for a sample by global_index."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
            "field_b": FieldMeta(name="field_b", dtype=torch.int64, shape=(3,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Set custom_meta for sample 0
        batch.set_custom_meta(0, {"sample_score": 0.9, "quality": "high"})

        result = batch.get_all_custom_meta()
        assert 0 in result
        assert result[0]["sample_score"] == 0.9
        assert result[0]["quality"] == "high"
        # Sample 1 should not have custom_meta
        assert 1 not in result

    def test_batch_meta_set_custom_meta_overwrites(self):
        """Test set_custom_meta overwrites existing metadata."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Set initial custom_meta
        batch.set_custom_meta(0, {"sample_score": 0.9, "quality": "high"})

        # Overwrite with new custom_meta
        batch.set_custom_meta(0, {"sample_score": 0.1, "quality": "low"})

        result = batch.get_all_custom_meta()
        assert result[0]["sample_score"] == 0.1
        assert result[0]["quality"] == "low"

    def test_batch_meta_set_custom_meta_invalid_global_index(self):
        """Test set_custom_meta raises error for invalid global_index."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Try to set with non-existent global index
        with pytest.raises(ValueError) as exc_info:
            batch.set_custom_meta(999, {"sample_score": 0.9})
        assert "not found in global_indexes" in str(exc_info.value)

    def test_batch_meta_update_custom_meta(self):
        """Test update_custom_meta adds metadata for different global indices."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
            "field_b": FieldMeta(name="field_b", dtype=torch.int64, shape=(3,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Initial custom_meta for sample 0
        batch.update_custom_meta({0: {"sample_score": 0.9}})

        # Update with metadata for sample 1
        batch.update_custom_meta({1: {"sample_score": 0.1}})

        result = batch.get_all_custom_meta()
        assert result[0]["sample_score"] == 0.9
        assert result[1]["sample_score"] == 0.1

    def test_batch_meta_update_custom_meta_overwrites(self):
        """Test update_custom_meta overwrites existing metadata at same key."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Initial custom_meta
        batch.update_custom_meta({0: {"sample_score": 0.9, "quality": "high"}})

        # Update with new value for same field - dict.update replaces
        batch.update_custom_meta({0: {"sample_score": 0.1, "quality": "low"}})

        result = batch.get_all_custom_meta()
        assert result[0]["sample_score"] == 0.1
        assert result[0]["quality"] == "low"

    def test_batch_meta_update_custom_meta_with_none(self):
        """Test update_custom_meta with None does nothing."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Set initial value
        batch.update_custom_meta({0: {"sample_score": 0.9}})

        # Update with None should not change anything
        batch.update_custom_meta(None)

        result = batch.get_all_custom_meta()
        assert result[0]["sample_score"] == 0.9

    def test_batch_meta_update_custom_meta_with_empty_dict(self):
        """Test update_custom_meta with empty dict does nothing."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Set initial value
        batch.update_custom_meta({0: {"sample_score": 0.9}})

        # Update with empty dict should not change anything
        batch.update_custom_meta({})

        result = batch.get_all_custom_meta()
        assert result[0]["sample_score"] == 0.9

    def test_batch_meta_update_custom_meta_invalid_global_index(self):
        """Test update_custom_meta raises error for invalid global_index."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Try to update with non-existent global index
        with pytest.raises(ValueError) as exc_info:
            batch.update_custom_meta({999: {"sample_score": 0.9}})
        assert "non-exist global_indexes" in str(exc_info.value)

    def test_batch_meta_clear_custom_meta(self):
        """Test clear_custom_meta removes all custom metadata."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # Set custom_meta
        batch.set_custom_meta(0, {"sample_score": 0.9})
        batch.set_custom_meta(1, {"sample_score": 0.1})

        # Clear all
        batch.clear_custom_meta()

        result = batch.get_all_custom_meta()
        assert result == {}

    def test_batch_meta_get_all_custom_meta_returns_deep_copy(self):
        """Test get_all_custom_meta returns a deep copy."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        custom_meta = {0: {"sample_score": 0.9, "nested": {"value": 1}}}
        batch.update_custom_meta(custom_meta)

        # Get all custom_meta
        result = batch.get_all_custom_meta()

        # Verify it's a deep copy - modifying result should not affect original
        result[0]["sample_score"] = 0.1
        result[0]["nested"]["value"] = 999

        original = batch.get_all_custom_meta()
        assert original[0]["sample_score"] == 0.9
        assert original[0]["nested"]["value"] == 1

    def test_batch_meta_get_all_custom_meta_empty(self):
        """Test get_all_custom_meta with no custom_meta returns empty dict."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        result = batch.get_all_custom_meta()
        assert result == {}

    def test_batch_meta_custom_meta_with_nested_data(self):
        """Test custom_meta supports nested dictionary data."""
        fields = {
            "field_a": FieldMeta(name="field_a", dtype=torch.float32, shape=(2,)),
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        nested_meta = {
            "model_info": {"name": "llama", "version": "7b", "config": {"hidden_size": 4096, "num_layers": 32}},
            "tags": ["training", "inference"],
        }
        batch.set_custom_meta(0, nested_meta)

        result = batch.get_all_custom_meta()
        assert result[0]["model_info"]["name"] == "llama"
        assert result[0]["model_info"]["version"] == "7b"
        assert result[0]["model_info"]["config"]["hidden_size"] == 4096
        assert result[0]["tags"] == ["training", "inference"]

    # =====================================================
    # Extra Info Methods Tests
    # =====================================================

    def test_batch_meta_update_extra_info(self):
        """Test update_extra_info adds multiple values."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        batch = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])

        # Update with multiple values
        batch.update_extra_info({"key1": "value1", "key2": "value2", "key3": "value3"})

        # Verify all exist
        assert "key1" in batch.extra_info
        assert "key2" in batch.extra_info
        assert "key3" in batch.extra_info
        assert batch.extra_info["key1"] == "value1"
        assert batch.extra_info["key2"] == "value2"

    def test_batch_meta_extra_info_preserved_in_operations(self):
        """Test extra_info is preserved in batch operations."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        batch1 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])
        batch1.extra_info["test_key1"] = "test_value"

        batch2 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=1, fields=fields)])
        batch2.extra_info["test_key2"] = "test_value_2"

        result = BatchMeta.concat([batch1, batch2])

        # Extra info is preserved
        assert "test_key1" in result.extra_info

    def test_batch_meta_extra_info_with_concat(self):
        """Test extra_info handling in concat with mixed types."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }

        batch1 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])
        batch1.extra_info["string"] = "hello"
        batch1.extra_info["number"] = 42

        batch2 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=1, fields=fields)])
        batch2.extra_info["string"] = "world"
        batch2.extra_info["number"] = 100

        result = BatchMeta.concat([batch1, batch2])

        # String: last value wins
        assert result.extra_info["string"] == "world"


class TestEdgeCases:
    """Edge cases and important boundaries."""

    def test_batch_meta_chunk_with_more_chunks_than_samples(self):
        """Example: Chunking when chunks > samples produces empty chunks."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }
        samples = [
            SampleMeta(partition_id="partition_0", global_index=0, fields=fields),
            SampleMeta(partition_id="partition_0", global_index=1, fields=fields),
        ]
        batch = BatchMeta(samples=samples)

        # 5 chunks for 2 samples
        chunks = batch.chunk(5)

        assert len(chunks) == 5
        # First 2 chunks have samples
        assert len(chunks[0]) == 1
        assert len(chunks[1]) == 1
        # Last 3 chunks are empty
        assert len(chunks[2]) == 0
        assert len(chunks[3]) == 0
        assert len(chunks[4]) == 0

    def test_batch_meta_concat_with_empty_batches(self):
        """Example: Concat handles empty batches gracefully."""
        fields = {
            "test_field": FieldMeta(
                name="test_field", dtype=torch.float32, shape=(2,), production_status=ProductionStatus.READY_FOR_CONSUME
            )
        }

        batch1 = BatchMeta(samples=[])
        batch2 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields)])
        batch3 = BatchMeta(samples=[])

        # Empty batches are filtered out
        result = BatchMeta.concat([batch1, batch2, batch3])
        assert len(result) == 1
        assert result.global_indexes == [0]

    def test_batch_meta_concat_validation_error(self):
        """Example: Concat validation catches field name mismatches."""
        fields1 = {"field1": FieldMeta(name="field1", dtype=torch.float32, shape=(2,))}
        fields2 = {"field2": FieldMeta(name="field2", dtype=torch.float32, shape=(2,))}

        batch1 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=0, fields=fields1)])

        batch2 = BatchMeta(samples=[SampleMeta(partition_id="partition_0", global_index=1, fields=fields2)])

        with pytest.raises(ValueError) as exc_info:
            BatchMeta.concat([batch1, batch2], validate=True)
        assert "Field names do not match" in str(exc_info.value)
