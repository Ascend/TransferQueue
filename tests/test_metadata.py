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

"""Unit tests for TransferQueue metadata module - Columnar BatchMeta + KVBatchMeta."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.metadata import BatchMeta, KVBatchMeta  # noqa: E402

# ==============================================================================
# Columnar BatchMeta Tests
# ==============================================================================


class TestBatchMetaColumnar:
    """Columnar BatchMeta using field_schema + production_status (numpy array)."""

    def _make_batch(self, batch_size=3, field_names=None):
        """Helper: create a simple columnar BatchMeta."""
        if field_names is None:
            field_names = ["field_a", "field_b"]
        field_schema = {
            field_name: {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}
            for field_name in field_names
        }
        production_status = np.ones(batch_size, dtype=np.int8)
        return BatchMeta(
            global_indexes=list(range(batch_size)),
            partition_ids=["partition_0"] * batch_size,
            field_schema=field_schema,
            production_status=production_status,
        )

    def test_basic_init(self):
        """Test basic columnar BatchMeta initialization."""
        batch = self._make_batch()
        assert len(batch) == 3
        assert batch.global_indexes == [0, 1, 2]
        assert batch.partition_ids == ["partition_0", "partition_0", "partition_0"]
        assert "field_a" in batch.field_schema
        assert "field_b" in batch.field_schema
        assert batch.field_names == ["field_a", "field_b"]

    def test_production_status_vector(self):
        """Test that production_status is accessible per sample."""
        batch = self._make_batch()
        assert batch.production_status is not None
        assert len(batch.production_status) == 3
        assert all(batch.production_status == 1)

    def test_chunk(self):
        """Test splitting a batch into chunks."""
        batch = BatchMeta(
            global_indexes=list(range(10)),
            partition_ids=["partition_0"] * 10,
            field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(10, dtype=np.int8),
            custom_meta=[{"uid": i} for i in range(10)],
            _custom_backend_meta=[{"f": {"key": i}} for i in range(10)],
        )
        chunks = batch.chunk(3)
        assert len(chunks) == 3
        # First chunk gets extra element (ceil division)
        assert len(chunks[0]) == 4
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3
        # custom_meta is chunked correctly (positional)
        assert chunks[0].custom_meta[0] == {"uid": 0}
        assert chunks[0].custom_meta[3] == {"uid": 3}
        assert len(chunks[0].custom_meta) == 4
        assert chunks[1].custom_meta[0] == {"uid": 4}

    def test_chunk_by_partition(self):
        """Test splitting by partition_id."""
        batch = BatchMeta(
            global_indexes=[10, 11, 12, 13],
            partition_ids=["part_A", "part_B", "part_A", "part_B"],
            field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        )
        chunks = batch.chunk_by_partition()
        assert len(chunks) == 2
        part_ids = [c.partition_ids[0] for c in chunks]
        assert "part_A" in part_ids
        assert "part_B" in part_ids

    def test_concat(self):
        """Test concatenating two batches."""
        batch1 = self._make_batch(batch_size=2)
        batch2 = BatchMeta(
            global_indexes=[2, 3],
            partition_ids=["partition_0", "partition_0"],
            field_schema=batch1.field_schema,
            production_status=np.ones(2, dtype=np.int8),
        )
        result = BatchMeta.concat([batch1, batch2])
        assert len(result) == 4
        assert result.global_indexes == [0, 1, 2, 3]

    def test_custom_meta_update(self):
        """Test update_custom_meta method."""
        batch = self._make_batch(batch_size=2)
        batch.update_custom_meta([{"tag": "alpha"}, {"tag": "beta"}])
        assert batch.custom_meta[0]["tag"] == "alpha"
        assert batch.custom_meta[1]["tag"] == "beta"

    def test_custom_backend_meta(self):
        """Test _custom_backend_meta attribute."""
        batch = self._make_batch(batch_size=2)
        batch._custom_backend_meta[0]["field_a"] = {"storage_key": "abc"}
        assert batch._custom_backend_meta[0]["field_a"]["storage_key"] == "abc"

    def test_size_property(self):
        """Test size == len property."""
        batch = self._make_batch(batch_size=5)
        assert batch.size == 5
        assert len(batch) == 5

    def test_pickle_roundtrip_preserves_batchmeta(self):
        """BatchMeta must survive pickle round-trip with all fields intact."""
        import pickle

        batch = BatchMeta(
            global_indexes=[0, 1],
            partition_ids=["p0", "p0"],
            field_schema={
                "tensor_field": {
                    "dtype": torch.float32,
                    "shape": (4,),
                    "is_nested": False,
                    "is_non_tensor": False,
                },
                "scalar_field": {
                    "dtype": torch.float32,
                    "shape": (),
                    "is_nested": False,
                    "is_non_tensor": False,
                },
            },
            production_status=np.ones(2, dtype=np.int8),
            extra_info={"step": 42},
            custom_meta=[{"score": 0.9}, {"score": 0.8}],
        )

        data = pickle.dumps(batch)
        restored = pickle.loads(data)

        assert restored.global_indexes == batch.global_indexes
        assert restored.partition_ids == batch.partition_ids
        assert restored.field_schema["tensor_field"]["dtype"] == torch.float32
        assert restored.field_schema["scalar_field"]["shape"] == ()
        assert list(restored.production_status) == list(batch.production_status)
        assert restored.extra_info == {"step": 42}
        assert restored.custom_meta == [{"score": 0.9}, {"score": 0.8}]

    def test_concat_extra_info_scalar_conflict_raises_value_error(self):
        """concat raises ValueError when scalar extra_info values conflict."""
        batch1 = BatchMeta(
            global_indexes=[0],
            partition_ids=["p0"],
            field_schema={"f": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(1, dtype=np.int8),
            extra_info={"step": 1},
        )
        batch2 = BatchMeta(
            global_indexes=[1],
            partition_ids=["p0"],
            field_schema={"f": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(1, dtype=np.int8),
            extra_info={"step": 2},
        )
        with pytest.raises(ValueError, match="conflicting values"):
            BatchMeta.concat([batch1, batch2])

    def test_concat_extra_info_key_union_with_warning(self):
        """concat unions extra_info keys when sets differ, with a warning."""
        batch1 = BatchMeta(
            global_indexes=[0],
            partition_ids=["p0"],
            field_schema={"f": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(1, dtype=np.int8),
            extra_info={"common": "ok", "only_a": 1},
        )
        batch2 = BatchMeta(
            global_indexes=[1],
            partition_ids=["p0"],
            field_schema={"f": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(1, dtype=np.int8),
            extra_info={"common": "ok", "only_b": 2},
        )
        result = BatchMeta.concat([batch1, batch2])
        assert result.extra_info["common"] == "ok"
        assert result.extra_info["only_a"] == 1
        assert result.extra_info["only_b"] == 2

    def test_concat_extra_info_tensor_equal_preserved(self):
        """concat preserves identical Tensor extra_info values."""
        t = torch.tensor([1.0, 2.0, 3.0])
        batch1 = BatchMeta(
            global_indexes=[0],
            partition_ids=["p0"],
            field_schema={"f": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(1, dtype=np.int8),
            extra_info={"embedding": t.clone()},
        )
        batch2 = BatchMeta(
            global_indexes=[1],
            partition_ids=["p0"],
            field_schema={"f": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(1, dtype=np.int8),
            extra_info={"embedding": t.clone()},
        )
        result = BatchMeta.concat([batch1, batch2])
        assert torch.equal(result.extra_info["embedding"], t)

    def test_setstate_readonly_production_status(self):
        """__setstate__ must make read-only production_status writable.

        When Ray deserializes a BatchMeta via Arrow zero-copy, numpy arrays
        become read-only. Since pickle skips __init__/__post_init__, the
        .copy() guard is bypassed. __setstate__ must fix this.
        """
        batch = self._make_batch()
        # Simulate pickle round-trip with Arrow zero-copy (read-only array)

        state = batch.__getstate__()
        # Convert tuple to list for modification
        state = list(state)
        slot_idx = list(BatchMeta.__slots__).index("production_status")
        state[slot_idx] = state[slot_idx].copy()
        state[slot_idx].flags.writeable = False
        state = tuple(state)

        restored = BatchMeta.__new__(BatchMeta)
        restored.__setstate__(state)

        # production_status must be writable after __setstate__
        assert restored.production_status.flags.writeable
        # Verify add_fields works without ValueError
        from tensordict import TensorDict

        td = TensorDict({"new_field": torch.randn(3, 4)}, batch_size=3)
        restored.add_fields(td)  # Should not raise
        assert restored.is_ready

    def test_shallow_copy_isolation_global_indexes(self):
        """Modifying the original global_indexes list does not affect BatchMeta."""
        original_indexes = [0, 1, 2]
        batch = BatchMeta(
            global_indexes=original_indexes,
            partition_ids=["p"] * 3,
        )
        original_indexes.append(99)
        assert batch.global_indexes == [0, 1, 2]
        assert len(batch) == 3

    def test_shallow_copy_isolation_extra_info(self):
        """Modifying the original extra_info dict does not affect BatchMeta."""
        original_info = {"key": "value"}
        batch = BatchMeta(
            global_indexes=[0],
            partition_ids=["p"],
            extra_info=original_info,
        )
        original_info["key"] = "corrupted"
        original_info["new_key"] = "new"
        assert batch.extra_info == {"key": "value"}

    def test_shallow_copy_isolation_field_schema(self):
        """Modifying the original field_schema dict does not affect BatchMeta."""
        original_schema = {"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}}
        batch = BatchMeta(
            global_indexes=[0],
            partition_ids=["p"],
            field_schema=original_schema,
        )
        original_schema["f"]["dtype"] = torch.int64
        assert batch.field_schema["f"]["dtype"] == torch.float32

    def test_select_fields_isolation_extra_info(self):
        """select_fields result has isolated extra_info from the original."""
        batch = self._make_batch()
        batch.set_extra_info("key", "original")
        selected = batch.select_fields(["field_a"])
        selected.set_extra_info("key", "modified")
        selected.set_extra_info("new_key", "new")
        assert batch.extra_info["key"] == "original"
        assert "new_key" not in batch.extra_info

    def test_select_fields_isolation_custom_meta(self):
        """select_fields result has isolated custom_meta from the original."""
        batch = self._make_batch()
        batch.update_custom_meta([{"score": 0.9}, {"score": 0.8}, {"score": 0.7}])
        selected = batch.select_fields(["field_a"])
        selected.update_custom_meta([{"score": 0.0}, {"score": 0.0}, {"score": 0.0}])
        assert batch.custom_meta[0]["score"] == 0.9

    def test_concat_no_double_copy_regression(self):
        """concat still works correctly after removing double-copy in __post_init__."""
        batch1 = self._make_batch(batch_size=2)
        batch2 = BatchMeta(
            global_indexes=[2, 3],
            partition_ids=["partition_0", "partition_0"],
            field_schema=batch1.field_schema,
            production_status=np.ones(2, dtype=np.int8),
            custom_meta=[{"id": 2}, {"id": 3}],
        )
        result = BatchMeta.concat([batch1, batch2])
        assert len(result) == 4
        assert result.global_indexes == [0, 1, 2, 3]
        assert result.custom_meta[2] == {"id": 2}
        assert result.custom_meta[3] == {"id": 3}

    def test_concat_extra_info_identical_scalars_preserved(self):
        """concat preserves identical scalar extra_info (int, str, dict)."""
        common_info = {"step": 42, "mode": "train", "config": {"lr": 0.01}}
        batch1 = BatchMeta(
            global_indexes=[0],
            partition_ids=["p0"],
            field_schema={"f": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(1, dtype=np.int8),
            extra_info=dict(common_info),
        )
        batch2 = BatchMeta(
            global_indexes=[1],
            partition_ids=["p0"],
            field_schema={"f": {"dtype": torch.float32, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(1, dtype=np.int8),
            extra_info=dict(common_info),
        )
        result = BatchMeta.concat([batch1, batch2])
        assert result.extra_info == common_info
        assert len(result) == 2

    def test_chunk_concat_roundtrip_preserves_extra_info(self):
        """chunk followed by concat preserves extra_info without errors."""
        batch = BatchMeta(
            global_indexes=list(range(6)),
            partition_ids=["p0"] * 6,
            field_schema={"f": {"dtype": torch.float32, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
            production_status=np.ones(6, dtype=np.int8),
            extra_info={"metrics": {"loss": 0.5}, "step": 100, "tags": ["train"]},
        )
        chunks = batch.chunk(3)
        restored = BatchMeta.concat(chunks)
        assert restored.extra_info == {"metrics": {"loss": 0.5}, "step": 100, "tags": ["train"]}
        assert len(restored) == 6
        assert restored.global_indexes == list(range(6))


# ==============================================================================
# KVBatchMeta Tests (all migrated from main with no modification)
# ==============================================================================


class TestKVBatchMeta:
    """KVBatchMeta Tests"""

    def test_kv_batch_meta_basic_init(self):
        """Example: Basic KVBatchMeta initialization."""
        kv_meta = KVBatchMeta(
            keys=["key1", "key2", "key3"],
            tags=[{"sample_id": 0}, {"sample_id": 1}, {"sample_id": 2}],
            partition_id="partition_0",
            fields=["field1", "field2"],
        )

        assert kv_meta.size == 3
        assert len(kv_meta) == 3
        assert kv_meta.keys == ["key1", "key2", "key3"]
        assert kv_meta.partition_id == "partition_0"
        assert kv_meta.fields == ["field1", "field2"]

    def test_kv_batch_meta_empty_init(self):
        """Example: Empty KVBatchMeta initialization."""
        kv_meta = KVBatchMeta()

        assert kv_meta.size == 0
        assert len(kv_meta) == 0
        assert kv_meta.keys == []
        assert kv_meta.tags == []
        assert kv_meta.partition_id is None
        assert kv_meta.fields is None

    def test_kv_batch_meta_init_validation_keys_tags_mismatch(self):
        """Example: Init validation catches keys and tags length mismatch."""
        with pytest.raises(ValueError) as exc_info:
            KVBatchMeta(
                keys=["key1", "key2"],
                tags=[{"sample_id": 0}],  # Only one tag
            )
        assert "keys and tags must have same length" in str(exc_info.value)

    def test_kv_batch_meta_init_validation_duplicate_keys(self):
        """Example: Init validation catches duplicate keys."""
        with pytest.raises(ValueError) as exc_info:
            KVBatchMeta(
                keys=["key1", "key1"],
                tags=[{"sample_id": 0}, {"sample_id": 1}],
                partition_id="partition_0",
            )
        assert "Got duplicated keys" in str(exc_info.value)

    def test_kv_batch_meta_init_validation_duplicate_fields(self):
        """Example: Init validation catches duplicate fields."""
        with pytest.raises(ValueError) as exc_info:
            KVBatchMeta(
                keys=["key1"],
                tags=[{"sample_id": 0}],
                partition_id="partition_0",
                fields=["field1", "field1"],
            )
        assert "Got duplicated fields" in str(exc_info.value)

    def test_kv_batch_meta_select_keys(self):
        """Example: Select specific keys from KVBatchMeta."""
        kv_meta = KVBatchMeta(
            keys=["key1", "key2", "key3"],
            tags=[{"idx": 0}, {"idx": 1}, {"idx": 2}],
            partition_id="partition_0",
            fields=["field1", "field2"],
            extra_info={"test": "value"},
        )

        selected = kv_meta.select_keys(["key1", "key3"])

        assert selected.keys == ["key1", "key3"]
        assert selected.tags == [{"idx": 0}, {"idx": 2}]
        assert selected.partition_id == "partition_0"
        assert selected.fields == ["field1", "field2"]
        assert selected.extra_info == {"test": "value"}

    def test_kv_batch_meta_select_keys_validation_duplicate(self):
        """Example: Select keys validation catches duplicate keys in input."""
        kv_meta = KVBatchMeta(
            keys=["key1", "key2", "key3"],
            tags=[{}, {}, {}],
        )

        with pytest.raises(ValueError) as exc_info:
            kv_meta.select_keys(["key1", "key1"])
        assert "Contain duplicate keys" in str(exc_info.value)

    def test_kv_batch_meta_select_keys_validation_nonexistent(self):
        """Example: Select keys validation catches non-existent keys."""
        kv_meta = KVBatchMeta(
            keys=["key1", "key2", "key3"],
            tags=[{}, {}, {}],
        )

        with pytest.raises(RuntimeError) as exc_info:
            kv_meta.select_keys(["key1", "nonexistent"])
        assert "not found in current batch" in str(exc_info.value)

    def test_kv_batch_meta_reorder(self):
        """Example: Reorder samples in KVBatchMeta."""
        kv_meta = KVBatchMeta(
            keys=["key1", "key2", "key3"],
            tags=[{"idx": 0}, {"idx": 1}, {"idx": 2}],
        )

        kv_meta.reorder([2, 0, 1])

        assert kv_meta.keys == ["key3", "key1", "key2"]
        assert kv_meta.tags == [{"idx": 2}, {"idx": 0}, {"idx": 1}]

    def test_kv_batch_meta_reorder_validation_size_mismatch(self):
        """Example: Reorder validation catches size mismatch."""
        kv_meta = KVBatchMeta(
            keys=["key1", "key2", "key3"],
            tags=[{}, {}, {}],
        )

        with pytest.raises(ValueError) as exc_info:
            kv_meta.reorder([0, 1])  # Only 2 indexes for 3 samples
        assert "does not match" in str(exc_info.value)

    def test_kv_batch_meta_reorder_validation_duplicate_indexes(self):
        """Example: Reorder validation catches duplicate indexes."""
        kv_meta = KVBatchMeta(
            keys=["key1", "key2", "key3"],
            tags=[{}, {}, {}],
        )

        with pytest.raises(ValueError) as exc_info:
            kv_meta.reorder([0, 0, 1])  # Duplicate index 0
        assert "Contain duplicate indexes" in str(exc_info.value)

    def test_kv_batch_meta_chunk(self):
        """Example: Split KVBatchMeta into multiple chunks."""
        kv_meta = KVBatchMeta(
            keys=[f"key{i}" for i in range(10)],
            tags=[{"idx": i} for i in range(10)],
            partition_id="partition_0",
            fields=["field1"],
            extra_info={"test": "value"},
        )

        chunks = kv_meta.chunk(3)

        assert len(chunks) == 3
        assert len(chunks[0]) == 4  # First chunk gets extra element
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3

        # Verify partition_id and fields are preserved
        assert chunks[0].partition_id == "partition_0"
        assert chunks[0].fields == ["field1"]
        assert chunks[0].extra_info == {"test": "value"}

        # Verify keys and tags are correctly chunked
        assert chunks[0].keys == ["key0", "key1", "key2", "key3"]
        assert chunks[0].tags == [{"idx": 0}, {"idx": 1}, {"idx": 2}, {"idx": 3}]
        assert chunks[1].keys == ["key4", "key5", "key6"]
        assert chunks[1].tags == [{"idx": 4}, {"idx": 5}, {"idx": 6}]

    def test_kv_batch_meta_chunk_with_more_chunks_than_samples(self):
        """Example: Chunking when chunks > samples produces empty chunks."""
        kv_meta = KVBatchMeta(
            keys=["key1", "key2"],
            tags=[{"idx": 0}, {"idx": 1}],
        )

        chunks = kv_meta.chunk(5)

        assert len(chunks) == 5
        assert len(chunks[0]) == 1
        assert len(chunks[1]) == 1
        assert len(chunks[2]) == 0
        assert len(chunks[3]) == 0
        assert len(chunks[4]) == 0

    def test_kv_batch_meta_concat(self):
        """Example: Concatenate multiple KVBatchMeta chunks."""
        kv_meta1 = KVBatchMeta(
            keys=["key0", "key1"],
            tags=[{"idx": 0}, {"idx": 1}],
            partition_id="partition_0",
            fields=["field1"],
            extra_info={"test": "value1"},
        )

        kv_meta2 = KVBatchMeta(
            keys=["key2", "key3"],
            tags=[{"idx": 2}, {"idx": 3}],
            partition_id="partition_0",
            fields=["field1"],
            extra_info={"test": "value1"},
        )

        result = KVBatchMeta.concat([kv_meta1, kv_meta2])

        assert result.size == 4
        assert result.keys == ["key0", "key1", "key2", "key3"]
        assert result.tags == [{"idx": 0}, {"idx": 1}, {"idx": 2}, {"idx": 3}]
        assert result.partition_id == "partition_0"
        assert result.fields == ["field1"]

    def test_kv_batch_meta_concat_with_empty_chunks(self):
        """Example: Concat handles empty KVBatchMeta chunks gracefully."""
        kv_meta1 = KVBatchMeta()
        kv_meta2 = KVBatchMeta(keys=["key0"], tags=[{"idx": 0}])
        kv_meta3 = KVBatchMeta()

        result = KVBatchMeta.concat([kv_meta1, kv_meta2, kv_meta3])

        assert result.size == 1
        assert result.keys == ["key0"]
        assert result.tags == [{"idx": 0}]

    def test_kv_batch_meta_concat_validation_field_mismatch(self):
        """Example: Concat validation catches field name mismatches."""
        kv_meta1 = KVBatchMeta(
            keys=["key0"],
            tags=[{}],
            fields=["field1"],
        )
        kv_meta2 = KVBatchMeta(
            keys=["key1"],
            tags=[{}],
            fields=["field2"],  # Different field
        )

        with pytest.raises(ValueError) as exc_info:
            KVBatchMeta.concat([kv_meta1, kv_meta2])
        assert "Field names do not match" in str(exc_info.value)

    def test_kv_batch_meta_concat_validation_partition_mismatch(self):
        """Example: Concat validation catches partition_id mismatches."""
        kv_meta1 = KVBatchMeta(
            keys=["key0"],
            tags=[{}],
            partition_id="partition_0",
        )
        kv_meta2 = KVBatchMeta(
            keys=["key1"],
            tags=[{}],
            partition_id="partition_1",  # Different partition
        )

        with pytest.raises(ValueError) as exc_info:
            KVBatchMeta.concat([kv_meta1, kv_meta2])
        assert "Partition do not match" in str(exc_info.value)

    def test_kv_batch_meta_concat_empty_list(self):
        """Example: Concat with empty list returns empty KVBatchMeta."""
        result = KVBatchMeta.concat([])

        assert result.size == 0
        assert result.keys == []
        assert result.tags == []

    def test_kv_batch_meta_deepcopy_tags(self):
        """Example: Tags are deep copied to prevent mutation."""
        original_tags = [{"data": [1, 2, 3]}]
        kv_meta = KVBatchMeta(
            keys=["key1"],
            tags=original_tags,
        )

        # Modify the tag in the KVBatchMeta
        kv_meta.tags[0]["data"].append(4)

        # Original should not be modified
        assert original_tags[0]["data"] == [1, 2, 3]

    def test_kv_batch_meta_deepcopy_extra_info(self):
        """Example: Extra info is deep copied to prevent mutation."""
        original_extra = {"nested": {"value": 1}}
        kv_meta = KVBatchMeta(
            keys=["key1"],
            tags=[{}],
            extra_info=original_extra,
        )

        # Modify extra_info
        kv_meta.extra_info["nested"]["value"] = 999

        # Original should not be modified
        assert original_extra["nested"]["value"] == 1

    def test_kv_batch_meta_concat_extra_info_conflict_raises(self):
        """KVBatchMeta.concat raises ValueError on conflicting extra_info values."""
        kv1 = KVBatchMeta(
            keys=["k0"],
            tags=[{}],
            extra_info={"step": 1},
        )
        kv2 = KVBatchMeta(
            keys=["k1"],
            tags=[{}],
            extra_info={"step": 2},
        )
        with pytest.raises(ValueError, match="conflicting"):
            KVBatchMeta.concat([kv1, kv2])


# ==============================================================================
# StorageUnitData Tests
# ==============================================================================


class TestStorageUnitDataStrict:
    """Tests for StorageUnitData length validation."""

    def test_put_data_length_mismatch_raises(self):
        """put_data must raise when global_indexes and field values have different lengths."""
        from transfer_queue.storage.simple_backend import StorageUnitData

        sud = StorageUnitData(storage_size=10)
        # 3 indexes but only 2 values — must raise, not silently drop
        with pytest.raises(ValueError, match="length mismatch"):
            sud.put_data({"field_a": [1, 2]}, global_indexes=[0, 1, 2])
