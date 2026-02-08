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

"""End-to-end tests for KV interface in transfer_queue.interface.

This test module validates the KV interface functionality by:
1. Using external interfaces (kv_put, kv_batch_put, kv_batch_get, kv_list, kv_clear) for read/write
2. Verifying correctness by calling TransferQueueController's internal methods directly
"""

import os
import sys
from pathlib import Path

import pytest
import ray
import torch
from tensordict import TensorDict

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402

# Configure Ray for tests
os.environ["RAY_DEDUP_LOGS"] = "0"


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for the test module."""
    if not ray.is_initialized():
        ray.init(namespace="TestKVInterfaceE2E")
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def tq_system(ray_init):
    """Initialize TransferQueue system for the test module."""
    tq.init()
    yield
    tq.close()


@pytest.fixture
def controller(tq_system):
    """Get the TransferQueueController actor for direct verification."""
    controller = ray.get_actor("TransferQueueController")
    yield controller


@pytest.fixture(autouse=True)
def cleanup_partition(controller):
    """Cleanup partition after each test."""
    yield
    try:
        ray.get(controller.clear_partition.remote("test_partition"))
    except Exception:
        pass


def get_controller_partition(controller, partition_id: str):
    """Get partition snapshot from controller for verification."""
    return ray.get(controller.get_partition_snapshot.remote(partition_id))


def assert_tensor_equal(tensor_a, tensor_b, msg=""):
    """Assert two tensors are equal."""
    assert torch.equal(tensor_a, tensor_b), f"{msg} Tensors are not equal: {tensor_a} vs {tensor_b}"


def assert_tensor_close(tensor_a, tensor_b, rtol=1e-5, atol=1e-8, msg=""):
    """Assert two tensors are close."""
    assert torch.allclose(tensor_a, tensor_b, rtol=rtol, atol=atol), f"{msg} Tensors are not close"


class TestKVPutE2E:
    """End-to-end tests for kv_put functionality."""

    def test_kv_put_with_dict_fields(self, controller):
        """Test kv_put with dict fields (auto-converted to TensorDict)."""
        partition_id = "test_partition"
        key = "sample_0"

        # Put with dict fields - will be auto-unsqueezed
        tq.kv_put(
            key=key, partition_id=partition_id, fields={"data": torch.tensor([1, 2, 3, 4])}, tag={"type": "dict_test"}
        )

        # Verify - retrieved data will have batch dimension
        retrieved = tq.kv_batch_get(keys=key, partition_id=partition_id)
        expected = torch.tensor([[1, 2, 3, 4]])  # unsqueezed
        assert_tensor_equal(retrieved["data"], expected)

    def test_kv_put_with_tensordict_fields(self, controller):
        """Test kv_put with tensordict fields."""
        partition_id = "test_partition"
        key = "sample_1"

        tensordict_data = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
            },
            batch_size=1,
        )
        # Put with dict fields - will be auto-unsqueezed
        tq.kv_put(key=key, partition_id=partition_id, fields=tensordict_data, tag={"type": "tensordict_test"})

        # Verify - retrieved data will have batch dimension
        retrieved = tq.kv_batch_get(keys=key, partition_id=partition_id)
        expected = torch.tensor([[1, 2, 3, 4]])  # unsqueezed
        assert_tensor_equal(retrieved["input_ids"], expected)

    def test_kv_put_single_sample_with_fields_and_tag(self, controller):
        """Test putting a single sample with fields and tag."""
        partition_id = "test_partition"
        key = "sample_2"
        # Use 1D tensors - kv_put with dict will auto-unsqueeze to add batch dimension
        input_ids = torch.tensor([1, 2, 3])
        attention_mask = torch.ones(3)
        tag = {"global_steps": 0, "status": "running"}

        # Put data using interface
        tq.kv_put(
            key=key,
            partition_id=partition_id,
            fields={"input_ids": input_ids, "attention_mask": attention_mask},
            tag=tag,
        )

        # Verify via controller internal state
        partition = get_controller_partition(controller, partition_id)
        assert partition is not None, "Partition should exist"

        # Check key->global_index mapping
        assert key in partition.keys_mapping, f"Key {key} should be in keys_mapping"
        global_idx = partition.keys_mapping[key]
        assert global_idx in partition.global_indexes, f"Global index {global_idx} should be in global_indexes"

        # Check custom_meta (tag)
        assert global_idx in partition.custom_meta, f"Custom meta should exist for global index {global_idx}"
        assert partition.custom_meta[global_idx]["global_steps"] == 0
        assert partition.custom_meta[global_idx]["status"] == "running"

        # Check production status - fields should be marked as produced
        assert "input_ids" in partition.field_name_mapping, "input_ids field should be registered"
        assert "attention_mask" in partition.field_name_mapping, "attention_mask field should be registered"
        input_ids_col_idx = partition.field_name_mapping["input_ids"]
        assert partition.production_status[global_idx, input_ids_col_idx] == 1, "input_ids should be marked as produced"

        # Retrieve and verify data via kv_batch_get - tensors will have batch dimension
        retrieved = tq.kv_batch_get(keys=key, partition_id=partition_id)
        assert "input_ids" in retrieved.keys()
        assert "attention_mask" in retrieved.keys()
        # After unsqueeze, tensors become 2D [batch_size=1, original_size]
        expected_input_ids = input_ids.unsqueeze(0)
        expected_attention_mask = attention_mask.unsqueeze(0)
        assert_tensor_equal(retrieved["input_ids"], expected_input_ids)
        assert_tensor_equal(retrieved["attention_mask"], expected_attention_mask)

    def test_kv_put_update_tag_only(self, controller):
        """Test updating only tag without providing fields."""
        partition_id = "test_partition"
        key = "sample_3"

        # First put with fields - use TensorDict as another example
        single_data = TensorDict({"value": torch.tensor([[10]])}, batch_size=1)
        tq.kv_put(key=key, partition_id=partition_id, fields=single_data, tag={"version": 1})

        # Update only tag
        new_tag = {"version": 2, "status": "updated"}
        tq.kv_put(key=key, partition_id=partition_id, fields=None, tag=new_tag)

        # Verify via controller
        partition = get_controller_partition(controller, partition_id)
        global_idx = partition.keys_mapping[key]
        assert partition.custom_meta[global_idx]["version"] == 2
        assert partition.custom_meta[global_idx]["status"] == "updated"

        # Data should still be accessible
        retrieved = tq.kv_batch_get(keys=key, partition_id=partition_id)
        assert_tensor_equal(retrieved["value"], torch.tensor([[10]]))

    def test_kv_put_partial_update(self, controller):
        """Test adding new fields to existing sample."""
        partition_id = "test_partition"
        key = "sample_4"

        # First put initial data
        initial_data = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
            },
            batch_size=1,
        )
        tq.kv_put(key=key, partition_id=partition_id, fields=initial_data, tag={"v": 1})

        # Add new fields to subset of keys
        new_fields = TensorDict(
            {
                "response": torch.tensor([[5, 6]]),
            },
            batch_size=1,
        )
        tq.kv_put(key=key, partition_id=partition_id, fields=new_fields, tag={"v": 2})

        # Verify via controller - only keys[1] should have response field
        partition = get_controller_partition(controller, partition_id)
        global_idx = partition.keys_mapping[key]

        # Check that fields were added
        assert "response" in partition.field_name_mapping
        response_col_idx = partition.field_name_mapping["response"]

        # key should have response marked as produced
        assert partition.production_status[global_idx, response_col_idx] == 1, "Key should have response"


class TestKVBatchPutE2E:
    """End-to-end tests for kv_batch_put functionality."""

    def test_kv_batch_put_multiple_samples(self, controller):
        """Test batch putting multiple samples."""
        partition_id = "test_partition"
        keys = ["batch_0", "batch_1", "batch_2", "batch_3"]
        batch_input_ids = torch.tensor(
            [
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
            ]
        )
        batch_attention_mask = torch.ones_like(batch_input_ids)

        fields = TensorDict(
            {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
            },
            batch_size=4,
        )

        tags = [{"idx": i, "batch": True} for i in range(4)]

        # Batch put using interface
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)

        # Verify via controller
        partition = get_controller_partition(controller, partition_id)
        assert partition is not None

        # All keys should be registered
        for key in keys:
            assert key in partition.keys_mapping, f"Key {key} should be in keys_mapping"

        # Verify tags
        for i, key in enumerate(keys):
            global_idx = partition.keys_mapping[key]
            assert partition.custom_meta[global_idx]["idx"] == i
            assert partition.custom_meta[global_idx]["batch"] is True

        # Verify all data via kv_batch_get
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id)
        assert_tensor_equal(retrieved["input_ids"], batch_input_ids)
        assert_tensor_equal(retrieved["attention_mask"], batch_attention_mask)

    def test_kv_batch_put_partial_update(self, controller):
        """Test adding new fields to existing samples."""
        partition_id = "test_partition"
        keys = ["partial_0", "partial_1"]

        # First put initial data
        initial_data = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
            },
            batch_size=2,
        )
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=initial_data, tags=[{"v": 1}, {"v": 1}])

        # Add new fields to subset of keys
        new_fields = TensorDict(
            {
                "response": torch.tensor([[5, 6]]),  # Only for 1 sample
            },
            batch_size=1,
        )
        tq.kv_batch_put(keys=[keys[1]], partition_id=partition_id, fields=new_fields, tags=[{"v": 2}])

        # Verify via controller - only keys[1] should have response field
        partition = get_controller_partition(controller, partition_id)
        global_idx_1 = partition.keys_mapping[keys[1]]

        # Check that fields were added
        assert "response" in partition.field_name_mapping
        response_col_idx = partition.field_name_mapping["response"]

        # keys[0] should NOT have response marked as produced
        global_idx_0 = partition.keys_mapping[keys[0]]
        assert partition.production_status[global_idx_0, response_col_idx] == 0, "Keys[0] should not have response"

        # keys[1] should have response marked as produced
        assert partition.production_status[global_idx_1, response_col_idx] == 1, "Keys[1] should have response"


class TestKVGetE2E:
    """End-to-end tests for kv_batch_get functionality."""

    def test_kv_batch_get_single_key(self, controller):
        """Test getting data for a single key."""
        partition_id = "test_partition"
        key = "get_single"
        # Use TensorDict to avoid auto-unsqueeze issue with dict input
        expected_data = torch.tensor([[100, 200, 300]])
        fields = TensorDict({"data": expected_data}, batch_size=1)

        tq.kv_put(key=key, partition_id=partition_id, fields=fields, tag=None)

        retrieved = tq.kv_batch_get(keys=key, partition_id=partition_id)
        assert_tensor_equal(retrieved["data"], expected_data)

    def test_kv_batch_get_multiple_keys(self, controller):
        """Test getting data for multiple keys."""
        partition_id = "test_partition"
        keys = ["get_multi_0", "get_multi_1", "get_multi_2"]
        expected_data = torch.tensor([[1, 2], [3, 4], [5, 6]])

        fields = TensorDict({"data": expected_data}, batch_size=3)
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=[{}, {}, {}])

        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id)
        assert_tensor_equal(retrieved["data"], expected_data)

    def test_kv_batch_get_partial_keys(self, controller):
        """Test getting data for partial keys."""
        partition_id = "test_partition"
        keys = ["get_multi_3", "get_multi_4", "get_multi_5"]
        partial_keys = ["get_multi_3", "get_multi_5"]
        input_data = torch.tensor([[1, 2], [3, 4], [5, 6]])
        expected_data = torch.tensor([[1, 2], [5, 6]])

        fields = TensorDict({"data": input_data}, batch_size=3)
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=[{}, {}, {}])

        retrieved = tq.kv_batch_get(keys=partial_keys, partition_id=partition_id)
        assert_tensor_equal(retrieved["data"], expected_data)

    def test_kv_batch_get_partial_fields(self, controller):
        """Test getting only partial fields."""
        partition_id = "test_partition"
        key = "get_fields"
        # Use TensorDict to avoid auto-unsqueeze issue
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)
        response = torch.tensor([[10, 20]])

        fields = TensorDict(
            {"input_ids": input_ids, "attention_mask": attention_mask, "response": response}, batch_size=1
        )

        # Put all fields
        tq.kv_put(key=key, partition_id=partition_id, fields=fields, tag=None)

        # Get only input_ids
        retrieved = tq.kv_batch_get(keys=key, partition_id=partition_id, fields="input_ids")
        assert "input_ids" in retrieved.keys()
        assert "attention_mask" not in retrieved.keys()
        assert "response" not in retrieved.keys()
        assert_tensor_equal(retrieved["input_ids"], input_ids)

        # Get multiple specific fields
        retrieved = tq.kv_batch_get(keys=key, partition_id=partition_id, fields=["input_ids", "response"])
        assert "input_ids" in retrieved.keys()
        assert "response" in retrieved.keys()
        assert "attention_mask" not in retrieved.keys()
        assert_tensor_equal(retrieved["input_ids"], input_ids)
        assert_tensor_equal(retrieved["response"], response)

    def test_kv_batch_get_nonexistent_key(self, controller):
        """Test that getting data for non-existent key returns empty result."""
        partition_id = "test_partition"

        # Try to get data for a key that doesn't exist - should return empty or raise error
        try:
            retrieved = tq.kv_batch_get(keys="nonexistent_key", partition_id=partition_id)
            # If it returns, it should be empty
            assert retrieved.batch_size[0] == 0
        except RuntimeError as e:
            # Or it might raise an error about keys not found
            assert "not found" in str(e).lower() or "empty" in str(e).lower()


class TestKVListE2E:
    """End-to-end tests for kv_list functionality."""

    def test_kv_list_all_keys(self, controller):
        """Test listing all keys in a partition."""
        partition_id = "test_partition"
        keys = ["list_0", "list_1", "list_2"]

        for i, key in enumerate(keys):
            tq.kv_put(key=key, partition_id=partition_id, fields={"data": torch.tensor([[i]])}, tag={"id": i})

        # List all keys
        listed_keys, tags = tq.kv_list(partition_id=partition_id)

        assert len(listed_keys) == 3
        for key in keys:
            assert key in listed_keys

        # Verify tags match
        for i, (key, tag) in enumerate(zip(listed_keys, tags, strict=False)):
            assert tag["id"] == i

    def test_kv_list_empty_partition(self):
        """Test listing empty partition."""
        partition_id = "test_partition_empty"

        keys, tags = tq.kv_list(partition_id=partition_id)

        assert len(keys) == 0
        assert len(tags) == 0


class TestKVClearE2E:
    """End-to-end tests for kv_clear functionality."""

    def test_kv_clear_single_key(self, controller):
        """Test clearing a single key."""
        partition_id = "test_partition"
        key = "clear_single"
        other_key = "clear_other"

        tq.kv_put(key=key, partition_id=partition_id, fields={"data": torch.tensor([[1]])}, tag={"id": "single"})
        tq.kv_put(key=other_key, partition_id=partition_id, fields={"data": torch.tensor([[2]])}, tag={"id": "other"})

        # Clear single key
        tq.kv_clear(keys=key, partition_id=partition_id)

        # Verify via kv_list
        listed_keys, _ = tq.kv_list(partition_id=partition_id)
        assert key not in listed_keys
        assert other_key in listed_keys

        # Verify via controller - key should be removed
        partition = get_controller_partition(controller, partition_id)
        assert key not in partition.keys_mapping
        assert other_key in partition.keys_mapping

    def test_kv_clear_multiple_keys(self, controller):
        """Test clearing multiple keys."""
        partition_id = "test_partition"
        keys = ["clear_multi_0", "clear_multi_1", "clear_multi_2", "clear_multi_3"]

        for i, key in enumerate(keys):
            tq.kv_put(key=key, partition_id=partition_id, fields={"data": torch.tensor([[i]])}, tag=None)

        # Clear first 2 keys
        tq.kv_clear(keys=keys[:2], partition_id=partition_id)

        # Verify
        listed_keys, _ = tq.kv_list(partition_id=partition_id)
        assert len(listed_keys) == 2
        assert keys[0] not in listed_keys
        assert keys[1] not in listed_keys
        assert keys[2] in listed_keys
        assert keys[3] in listed_keys


class TestKVE2ECornerCases:
    """End-to-end tests for corner cases."""

    def test_field_expansion_across_samples(self, controller):
        """Test that new fields can be added across samples."""
        partition_id = "test_partition"
        keys = ["expand_0", "expand_1"]

        # Put initial fields
        tq.kv_put(key=keys[0], partition_id=partition_id, fields={"field_a": torch.tensor([[1]])}, tag=None)

        # Add new field to first key
        tq.kv_put(key=keys[0], partition_id=partition_id, fields={"field_b": torch.tensor([[2]])}, tag=None)

        # Add different field to second key
        tq.kv_put(
            key=keys[1],
            partition_id=partition_id,
            fields={"field_a": torch.tensor([[3]]), "field_c": torch.tensor([[4]])},
            tag=None,
        )

        # Verify field expansion in controller
        partition = get_controller_partition(controller, partition_id)

        # All fields should be registered, but only samples with the actual fields are labeled as READY_FOR_CONSUME
        assert "field_a" in partition.field_name_mapping
        assert "field_b" in partition.field_name_mapping
        assert "field_c" in partition.field_name_mapping

        # We can only fetch "field_a" because not all requested keys has other fields
        data = tq.kv_batch_get(keys=keys, partition_id=partition_id)
        assert "field_a" in data
        assert "field_b" not in data
        assert "field_c" not in data


def run_tests():
    """Run all e2e tests manually for debugging."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
