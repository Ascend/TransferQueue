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

import asyncio
import os
import sys
from pathlib import Path

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402


class TQAPIWrapper:
    """Wrapper that routes kv_* calls to sync or async interface based on use_async flag."""

    def __init__(self, use_async: bool):
        self.use_async = use_async

    def __getattr__(self, name):
        if name.startswith("kv_"):
            if self.use_async:
                async_func = getattr(tq, f"async_{name}")
                return lambda *args, **kwargs: asyncio.run(async_func(*args, **kwargs))
            else:
                return getattr(tq, name)
        # For non-kv_ attributes (init, close), pass through directly
        return getattr(tq, name)


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def tq_api(request):
    """Returns a unified TQ API handle that routes to sync or async interface.

    When use_async=False (sync mode), calls tq.kv_* directly.
    When use_async=True (async mode), calls tq.async_kv_* via asyncio.run().
    """
    return TQAPIWrapper(use_async=request.param)


# Configure Ray for tests
os.environ["RAY_DEDUP_LOGS"] = "0"

# Backend configurations for E2E tests
# Adjust values for GitHub CI environment (smaller memory footprint)
BACKEND_CONFIGS = {
    "SimpleStorage": {
        "controller": {
            "polling_mode": True,
        },
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "total_storage_size": 200,
                "num_data_storage_units": 2,
            },
        },
    },
    "MooncakeStore": {
        "controller": {
            "polling_mode": True,
        },
        "backend": {
            "storage_backend": "MooncakeStore",
            "MooncakeStore": {
                # Reduced memory sizes for CI/testing environment
                "global_segment_size": 134217728,  # 128MB
                "local_buffer_size": 134217728,  # 128MB
                "metadata_server": os.environ.get("TQ_MOONCAKE_METADATA_SERVER", "localhost:50050"),
                "master_server_address": os.environ.get("TQ_MOONCAKE_MASTER_SERVER", "localhost:50051"),
                "protocol": "tcp",
                "device_name": "",
            },
        },
    },
}


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for the test module."""
    if not ray.is_initialized():
        ray.init(namespace="TestKVInterfaceE2E")
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def backend_name():
    """Get the backend name from environment variable.

    Environment variables:
        TQ_TEST_BACKEND: Backend name (SimpleStorage or MooncakeStore)

    To run tests for a specific backend:
        TQ_TEST_BACKEND=SimpleStorage pytest tests/e2e/test_kv_interface_e2e.py
        TQ_TEST_BACKEND=MooncakeStore pytest tests/e2e/test_kv_interface_e2e.py
    """
    return os.environ.get("TQ_TEST_BACKEND", "SimpleStorage")


@pytest.fixture(scope="module")
def tq_system(ray_init, backend_name):
    """Initialize TransferQueue system for the test module.

    Args:
        ray_init: Ray cluster fixture
        backend_name: Backend name from TQ_TEST_BACKEND env var
    """
    if backend_name not in BACKEND_CONFIGS:
        raise ValueError(f"Unknown backend: {backend_name}. Available: {list(BACKEND_CONFIGS.keys())}")

    config = BACKEND_CONFIGS[backend_name]
    tq.init(OmegaConf.create(config))
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
        partition_ids = ray.get(controller.list_partitions.remote())
        for partition_id in partition_ids:
            ray.get(controller.clear_partition.remote(partition_id))
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


def assert_nested_tensor_equal(nested_a, nested_b, msg=""):
    """Assert two nested (jagged) tensors are equal component-wise."""
    components_a = list(nested_a)
    components_b = list(nested_b)
    assert len(components_a) == len(components_b), f"{msg} Length mismatch: {len(components_a)} vs {len(components_b)}"
    for i, (a, b) in enumerate(zip(components_a, components_b, strict=True)):
        assert torch.equal(a, b), f"{msg} Component {i} not equal: {a} vs {b}"


class TestKVPutE2E:
    """End-to-end tests for kv_put functionality."""

    def test_kv_put_with_dict_fields(self, controller, tq_api):
        """Test kv_put with dict fields (auto-converted to TensorDict)."""
        partition_id = "test_partition"
        key = "sample_0"

        # Put with dict fields - will be auto-unsqueezed
        tq_api.kv_put(
            key=key, partition_id=partition_id, fields={"data": torch.tensor([1, 2, 3, 4])}, tag={"type": "dict_test"}
        )

        # Verify - retrieved data will have batch dimension
        retrieved = tq_api.kv_batch_get(keys=key, partition_id=partition_id)
        expected = torch.tensor([[1, 2, 3, 4]])  # unsqueezed
        assert_tensor_equal(retrieved["data"], expected)

        # delete the key (MooncakeStore does not support updating existing key, so we need to clear it before next test)
        tq_api.kv_clear(keys=key, partition_id=partition_id)

    def test_kv_put_with_tensordict_fields(self, controller, tq_api):
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
        tq_api.kv_put(key=key, partition_id=partition_id, fields=tensordict_data, tag={"type": "tensordict_test"})

        # Verify - retrieved data will have batch dimension
        retrieved = tq_api.kv_batch_get(keys=key, partition_id=partition_id)
        expected = torch.tensor([[1, 2, 3, 4]])  # unsqueezed
        assert_tensor_equal(retrieved["input_ids"], expected)

        tq_api.kv_clear(keys=key, partition_id=partition_id)

    def test_kv_put_single_sample_with_fields_and_tag(self, controller, tq_api):
        """Test putting a single sample with fields and tag."""
        partition_id = "test_partition"
        key = "sample_2"
        # Use 1D tensors - kv_put with dict will auto-unsqueeze to add batch dimension
        input_ids = torch.tensor([1, 2, 3])
        attention_mask = torch.ones(3)
        tag = {"global_steps": 0, "status": "running"}

        # Put data using interface
        tq_api.kv_put(
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
        retrieved = tq_api.kv_batch_get(keys=key, partition_id=partition_id)
        assert "input_ids" in retrieved.keys()
        assert "attention_mask" in retrieved.keys()
        # After unsqueeze, tensors become 2D [batch_size=1, original_size]
        expected_input_ids = input_ids.unsqueeze(0)
        expected_attention_mask = attention_mask.unsqueeze(0)
        assert_tensor_equal(retrieved["input_ids"], expected_input_ids)
        assert_tensor_equal(retrieved["attention_mask"], expected_attention_mask)

        tq_api.kv_clear(keys=key, partition_id=partition_id)

    def test_kv_put_update_tag_only(self, controller, tq_api):
        """Test updating only tag without providing fields."""
        partition_id = "test_partition"
        key = "sample_3"

        # First put with fields - use TensorDict as another example
        single_data = TensorDict({"value": torch.tensor([[10]])}, batch_size=1)
        tq_api.kv_put(key=key, partition_id=partition_id, fields=single_data, tag={"version": 1})

        # Update only tag
        new_tag = {"version": 2, "status": "updated"}
        tq_api.kv_put(key=key, partition_id=partition_id, fields=None, tag=new_tag)

        # Verify via controller
        partition = get_controller_partition(controller, partition_id)
        global_idx = partition.keys_mapping[key]
        assert partition.custom_meta[global_idx]["version"] == 2
        assert partition.custom_meta[global_idx]["status"] == "updated"

        # Data should still be accessible
        retrieved = tq_api.kv_batch_get(keys=key, partition_id=partition_id)
        assert_tensor_equal(retrieved["value"], torch.tensor([[10]]))

        tq_api.kv_clear(keys=key, partition_id=partition_id)

    def test_kv_put_partial_update(self, controller, tq_api):
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
        tq_api.kv_put(key=key, partition_id=partition_id, fields=initial_data, tag={"v": 1})

        # Add new fields to subset of keys
        new_fields = TensorDict(
            {
                "response": torch.tensor([[5, 6]]),
            },
            batch_size=1,
        )
        tq_api.kv_put(key=key, partition_id=partition_id, fields=new_fields, tag={"v": 2})

        # Verify via controller - only keys[1] should have response field
        partition = get_controller_partition(controller, partition_id)
        global_idx = partition.keys_mapping[key]

        # Check that fields were added
        assert "response" in partition.field_name_mapping
        response_col_idx = partition.field_name_mapping["response"]

        # key should have response marked as produced
        assert partition.production_status[global_idx, response_col_idx] == 1, "Key should have response"

        tq_api.kv_clear(keys=key, partition_id=partition_id)

    def test_kv_put_returns_cumulative_fields(self, controller, tq_api):
        """Test that kv_put returns KVBatchMeta with cumulative fields (previous + new)."""
        partition_id = "test_partition"
        key = "sample_cumulative"

        # First put: only input_ids
        first_data = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
            },
            batch_size=1,
        )
        first_meta = tq_api.kv_put(key=key, partition_id=partition_id, fields=first_data, tag={"step": 1})

        # Verify first meta contains only input_ids
        assert first_meta.fields is not None
        assert "input_ids" in first_meta.fields
        assert len(first_meta.fields) == 1

        # Second put: add attention_mask
        second_data = TensorDict(
            {
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
            batch_size=1,
        )
        second_meta = tq_api.kv_put(key=key, partition_id=partition_id, fields=second_data, tag={"step": 2})

        # Verify second meta contains BOTH previous (input_ids) and new (attention_mask) fields
        assert second_meta.fields is not None
        assert "input_ids" in second_meta.fields, "Previous field 'input_ids' should be in returned fields"
        assert "attention_mask" in second_meta.fields, "New field 'attention_mask' should be in returned fields"
        assert len(second_meta.fields) == 2, f"Expected 2 fields, got {second_meta.fields}"

        # Third put: add response field
        third_data = TensorDict(
            {
                "response": torch.tensor([[10, 20]]),
            },
            batch_size=1,
        )
        third_meta = tq_api.kv_put(key=key, partition_id=partition_id, fields=third_data, tag={"step": 3})

        # Verify third meta contains ALL three fields
        assert third_meta.fields is not None
        assert "input_ids" in third_meta.fields, "Previous field 'input_ids' should still be present"
        assert "attention_mask" in third_meta.fields, "Previous field 'attention_mask' should still be present"
        assert "response" in third_meta.fields, "New field 'response' should be present"
        assert len(third_meta.fields) == 3, f"Expected 3 fields, got {third_meta.fields}"

        tq_api.kv_clear(keys=key, partition_id=partition_id)


class TestKVBatchPutE2E:
    """End-to-end tests for kv_batch_put functionality."""

    def test_kv_batch_put_multiple_samples(self, controller, tq_api):
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
        tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)

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
        retrieved = tq_api.kv_batch_get(keys=keys, partition_id=partition_id)
        assert_tensor_equal(retrieved["input_ids"], batch_input_ids)
        assert_tensor_equal(retrieved["attention_mask"], batch_attention_mask)

        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_batch_put_partial_update(self, controller, tq_api):
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
        tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=initial_data, tags=[{"v": 1}, {"v": 1}])

        # Add new fields to subset of keys
        new_fields = TensorDict(
            {
                "response": torch.tensor([[5, 6]]),  # Only for 1 sample
            },
            batch_size=1,
        )
        tq_api.kv_batch_put(keys=[keys[1]], partition_id=partition_id, fields=new_fields, tags=[{"v": 2}])

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

        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_batch_put_returns_cumulative_fields(self, controller, tq_api):
        """Test that kv_batch_put returns KVBatchMeta with cumulative fields (previous + new)."""
        partition_id = "test_partition"
        keys = ["batch_cumulative_0", "batch_cumulative_1"]

        # First batch put: only input_ids
        first_data = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            },
            batch_size=2,
        )
        first_meta = tq_api.kv_batch_put(
            keys=keys, partition_id=partition_id, fields=first_data, tags=[{"step": 1}, {"step": 1}]
        )

        # Verify first meta contains only input_ids
        assert first_meta.fields is not None
        assert "input_ids" in first_meta.fields
        assert len(first_meta.fields) == 1

        # Second batch put: add attention_mask for both keys
        second_data = TensorDict(
            {
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            },
            batch_size=2,
        )
        second_meta = tq_api.kv_batch_put(
            keys=keys, partition_id=partition_id, fields=second_data, tags=[{"step": 2}, {"step": 2}]
        )

        # Verify second meta contains BOTH previous (input_ids) and new (attention_mask) fields
        assert second_meta.fields is not None
        assert "input_ids" in second_meta.fields, "Previous field 'input_ids' should be in returned fields"
        assert "attention_mask" in second_meta.fields, "New field 'attention_mask' should be in returned fields"
        assert len(second_meta.fields) == 2, f"Expected 2 fields, got {second_meta.fields}"

        tq_api.kv_clear(keys=keys, partition_id=partition_id)


class TestKVGetE2E:
    """End-to-end tests for kv_batch_get functionality."""

    def test_kv_batch_get_nested_tensor(self, controller, tq_api):
        # test put a regular tensor with batch size 1 and get it back as a nested tensor
        partition_id = "test_partition"
        keys = []
        data_list = []

        for i in range(1, 4):
            key = f"nested_tensor_{i}"
            keys.append(key)
            data = torch.randn(size=(i,))
            data_list.append(data)
            fields = TensorDict({"data": data.unsqueeze(0)}, batch_size=1)
            tq_api.kv_put(key=key, partition_id=partition_id, fields=fields, tag=None)

        retrieved = tq_api.kv_batch_get(keys=keys, partition_id=partition_id)

        assert_nested_tensor_equal(retrieved["data"], torch.nested.as_nested_tensor(data_list, layout=torch.jagged))
        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_batch_get_single_key(self, controller, tq_api):
        """Test getting data for a single key."""
        partition_id = "test_partition"
        key = "get_single"
        # Use TensorDict to avoid auto-unsqueeze issue with dict input
        expected_data = torch.tensor([[100, 200, 300]])
        fields = TensorDict({"data": expected_data}, batch_size=1)

        tq_api.kv_put(key=key, partition_id=partition_id, fields=fields, tag=None)

        retrieved = tq_api.kv_batch_get(keys=key, partition_id=partition_id)
        assert_tensor_equal(retrieved["data"], expected_data)

        tq_api.kv_clear(keys=key, partition_id=partition_id)

    def test_kv_batch_get_multiple_keys(self, controller, tq_api):
        """Test getting data for multiple keys."""
        partition_id = "test_partition"
        keys = ["get_multi_0", "get_multi_1", "get_multi_2"]
        expected_data = torch.tensor([[1, 2], [3, 4], [5, 6]])

        fields = TensorDict({"data": expected_data}, batch_size=3)
        tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=[{}, {}, {}])

        retrieved = tq_api.kv_batch_get(keys=keys, partition_id=partition_id)
        assert_tensor_equal(retrieved["data"], expected_data)

        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_batch_get_partial_keys(self, controller, tq_api):
        """Test getting data for partial keys."""
        partition_id = "test_partition"
        keys = ["get_multi_3", "get_multi_4", "get_multi_5"]
        partial_keys = ["get_multi_3", "get_multi_5"]
        input_data = torch.tensor([[1, 2], [3, 4], [5, 6]])
        nested_data = torch.nested.nested_tensor([[10, 11, 12], [20], [30, 31]])
        three_d_nested_data = torch.nested.nested_tensor(
            [[[10, 11], [12, 13]], [[20, 21], [22, 23]], [[30, 31], [32, 33]]]
        )
        expected_three_d_nested_data = [torch.tensor([[10, 11], [12, 13]]), torch.tensor([[30, 31], [32, 33]])]
        expected_data = torch.tensor([[1, 2], [5, 6]])
        expected_nested_data = [torch.tensor([10, 11, 12]), torch.tensor([30, 31])]

        fields = TensorDict(
            {"data": input_data, "nested_data": nested_data, "three_d_nested_data": three_d_nested_data}, batch_size=3
        )
        tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=[{}, {}, {}])

        retrieved = tq_api.kv_batch_get(keys=partial_keys, partition_id=partition_id)
        assert_tensor_equal(retrieved["data"], expected_data)

        assert_nested_tensor_equal(retrieved["nested_data"], expected_nested_data)
        assert_nested_tensor_equal(retrieved["three_d_nested_data"], expected_three_d_nested_data)

        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_batch_get_partial_fields(self, controller, tq_api):
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
        tq_api.kv_put(key=key, partition_id=partition_id, fields=fields, tag=None)

        # Get only input_ids
        retrieved = tq_api.kv_batch_get(keys=key, partition_id=partition_id, select_fields="input_ids")
        assert "input_ids" in retrieved.keys()
        assert "attention_mask" not in retrieved.keys()
        assert "response" not in retrieved.keys()
        assert_tensor_equal(retrieved["input_ids"], input_ids)

        # Get multiple specific fields
        retrieved = tq_api.kv_batch_get(keys=key, partition_id=partition_id, select_fields=["input_ids", "response"])
        assert "input_ids" in retrieved.keys()
        assert "response" in retrieved.keys()
        assert "attention_mask" not in retrieved.keys()
        assert_tensor_equal(retrieved["input_ids"], input_ids)
        assert_tensor_equal(retrieved["response"], response)

        tq_api.kv_clear(keys=key, partition_id=partition_id)

    def test_kv_batch_get_nonexistent_key(self, controller, tq_api):
        """Test that getting data for non-existent key returns empty result."""
        partition_id = "test_partition"

        # Try to get data for a key that doesn't exist - should return empty or raise error
        try:
            retrieved = tq_api.kv_batch_get(keys="nonexistent_key", partition_id=partition_id)
            # If it returns, it should be empty
            assert retrieved.batch_size[0] == 0
        except ValueError as e:
            # Or it might raise an error about keys not found
            assert "not found" in str(e).lower() or "empty" in str(e).lower()


class TestKVBatchGetByMetaE2E:
    """End-to-end tests for kv_batch_get_by_meta functionality."""

    def test_kv_batch_get_by_meta_select_fields_override(self, controller, tq_api):
        """Test kv_batch_get_by_meta with select_fields to override meta.fields."""
        partition_id = "test_partition"
        keys = ["meta_override_0", "meta_override_1", "meta_override_2"]
        expected_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_attention_mask = torch.ones_like(expected_input_ids)
        expected_response = torch.tensor([[10, 20], [30, 40], [50, 60]])

        fields = TensorDict(
            {
                "input_ids": expected_input_ids,
                "attention_mask": expected_attention_mask,
                "response": expected_response,
            },
            batch_size=3,
        )
        tags = [{"idx": i} for i in range(3)]

        # Batch put all fields
        meta = tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)

        # Verify meta.fields contains all fields
        assert "input_ids" in meta.fields
        assert "attention_mask" in meta.fields
        assert "response" in meta.fields
        assert len(meta.fields) == 3

        # Retrieve using kv_batch_get_by_meta with select_fields override - only input_ids
        retrieved = tq_api.kv_batch_get_by_meta(meta, select_fields="input_ids")
        assert "input_ids" in retrieved.keys()
        assert "attention_mask" not in retrieved.keys()
        assert "response" not in retrieved.keys()
        assert_tensor_equal(retrieved["input_ids"], expected_input_ids)

        # Retrieve using kv_batch_get_by_meta with select_fields override - subset of fields
        retrieved = tq_api.kv_batch_get_by_meta(meta, select_fields=["attention_mask", "response"])
        assert "input_ids" not in retrieved.keys()
        assert "attention_mask" in retrieved.keys()
        assert "response" in retrieved.keys()
        assert_tensor_equal(retrieved["attention_mask"], expected_attention_mask)
        assert_tensor_equal(retrieved["response"], expected_response)

        # Retrieve without select_fields - should get all fields from meta
        retrieved = tq_api.kv_batch_get_by_meta(meta)
        assert "input_ids" in retrieved.keys()
        assert "attention_mask" in retrieved.keys()
        assert "response" in retrieved.keys()
        assert_tensor_equal(retrieved["input_ids"], expected_input_ids)
        assert_tensor_equal(retrieved["attention_mask"], expected_attention_mask)
        assert_tensor_equal(retrieved["response"], expected_response)

        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_batch_get_by_meta_select_fields_invalid(self, controller, tq_api):
        """Test kv_batch_get_by_meta raises error when select_fields contains invalid field."""
        partition_id = "test_partition"
        keys = ["meta_invalid_0", "meta_invalid_1"]
        fields = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            },
            batch_size=2,
        )

        meta = tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=[{}, {}])

        # Try to retrieve with a field that doesn't exist in meta
        with pytest.raises(ValueError, match=r"select_fields.*not found"):
            tq_api.kv_batch_get_by_meta(meta, select_fields="nonexistent_field")

        # Try to retrieve with mix of valid and invalid fields
        with pytest.raises(ValueError, match=r"select_fields.*not found"):
            tq_api.kv_batch_get_by_meta(meta, select_fields=["input_ids", "invalid_field"])

        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_batch_get_by_meta_from_kv_batch_put(self, controller, tq_api):
        """Test kv_batch_get_by_meta using KVBatchMeta returned from kv_batch_put."""
        partition_id = "test_partition"
        keys = ["meta_batch_0", "meta_batch_1", "meta_batch_2"]
        expected_input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_attention_mask = torch.ones_like(expected_input_ids)

        fields = TensorDict(
            {
                "input_ids": expected_input_ids,
                "attention_mask": expected_attention_mask,
            },
            batch_size=3,
        )
        tags = [{"idx": i} for i in range(3)]

        # Batch put and get KVBatchMeta
        meta = tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)

        # Retrieve using kv_batch_get_by_meta
        retrieved = tq_api.kv_batch_get_by_meta(meta)
        assert_tensor_equal(retrieved["input_ids"], expected_input_ids)
        assert_tensor_equal(retrieved["attention_mask"], expected_attention_mask)

        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_batch_get_by_meta_multiple_puts(self, controller, tq_api):
        """Test kv_batch_get_by_meta with data from multiple sequential puts."""
        partition_id = "test_partition"
        keys = ["meta_multi_0", "meta_multi_1"]

        # First put
        first_data = TensorDict({"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}, batch_size=2)
        tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=first_data, tags=[{}, {}])

        # Second put adds more fields
        second_data = TensorDict({"attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])}, batch_size=2)
        second_meta = tq_api.kv_batch_put(keys=keys, partition_id=partition_id, fields=second_data, tags=[{}, {}])

        # Use second meta (contains both fields)
        retrieved = tq_api.kv_batch_get_by_meta(second_meta)
        assert "input_ids" in retrieved.keys()
        assert "attention_mask" in retrieved.keys()
        assert_tensor_equal(retrieved["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]]))
        assert_tensor_equal(retrieved["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 1]]))

        tq_api.kv_clear(keys=keys, partition_id=partition_id)


class TestKVListE2E:
    """End-to-end tests for kv_list functionality."""

    def test_kv_list_single_partition(self, controller, tq_api):
        """Test listing all keys and tags in single partition."""
        partition_id = "test_partition"
        keys = ["list_0", "list_1", "list_2"]

        for i, key in enumerate(keys):
            tq_api.kv_put(key=key, partition_id=partition_id, fields={"data": torch.tensor([[i]])}, tag={"id": i})

        # List all keys
        partition_info = tq_api.kv_list(partition_id=partition_id)

        assert len(partition_info.keys()) == 1
        assert "test_partition" in partition_info.keys()
        assert len(partition_info["test_partition"]) == 3
        for key in keys:
            assert key in partition_info["test_partition"]

        # Verify tags match
        for i, (key, tag) in enumerate(partition_info["test_partition"].items()):
            assert tag["id"] == i

        tq_api.kv_clear(keys=keys, partition_id=partition_id)

    def test_kv_list_all_partitions(self, controller, tq_api):
        """Test listing keys and tags in all partitions."""
        partition_id = ["test_partition0", "test_partition1", "test_partition2"]

        keys_partition0 = ["list_0", "list_1", "list_2"]
        keys_partition1 = ["list_0", "list_1", "list_2"]  # deliberately set same keys
        keys_partition2 = ["list_3", "list_4", "list_5", "list_6"]

        fields_partition0 = TensorDict({"data": torch.tensor([[0], [1], [2]])}, batch_size=3)
        fields_partition1 = TensorDict({"data": torch.tensor([[3], [4], [5]])}, batch_size=3)
        fields_partition2 = TensorDict({"data": torch.tensor([[6], [7], [8], [9]])}, batch_size=4)

        tags_partition0 = [{"id": i} for i in range(3)]
        tags_partition1 = [{"id": i + 3} for i in range(3)]
        tags_partition2 = [{"id": i + 6} for i in range(4)]

        # Put to TQ
        tq_api.kv_batch_put(
            keys=keys_partition0, partition_id=partition_id[0], fields=fields_partition0, tags=tags_partition0
        )
        tq_api.kv_batch_put(
            keys=keys_partition1, partition_id=partition_id[1], fields=fields_partition1, tags=tags_partition1
        )
        tq_api.kv_batch_put(
            keys=keys_partition2, partition_id=partition_id[2], fields=fields_partition2, tags=tags_partition2
        )

        # List all keys
        partition_info = tq_api.kv_list()

        # Verify all partitions are exist
        assert len(partition_info.keys()) == 3
        assert "test_partition0" in partition_info.keys()
        assert "test_partition1" in partition_info.keys()
        assert "test_partition2" in partition_info.keys()

        assert len(partition_info["test_partition0"]) == 3
        for key in keys_partition0:
            assert key in partition_info["test_partition0"]

        assert len(partition_info["test_partition1"]) == 3
        for key in keys_partition1:
            assert key in partition_info["test_partition1"]

        assert len(partition_info["test_partition2"]) == 4
        for key in keys_partition2:
            assert key in partition_info["test_partition2"]

        # Verify tags match
        for i, (key, tag) in enumerate(partition_info["test_partition0"].items()):
            assert tag["id"] == i
        for i, (key, tag) in enumerate(partition_info["test_partition1"].items()):
            assert tag["id"] == i + 3
        for i, (key, tag) in enumerate(partition_info["test_partition2"].items()):
            assert tag["id"] == i + 6

        tq_api.kv_clear(keys=keys_partition0, partition_id=partition_id[0])
        tq_api.kv_clear(keys=keys_partition1, partition_id=partition_id[1])
        tq_api.kv_clear(keys=keys_partition2, partition_id=partition_id[2])

    def test_kv_list_empty_partition(self, tq_api):
        """Test listing empty partition."""
        partition_id = "test_partition_empty"

        partition_info = tq_api.kv_list(partition_id=partition_id)

        assert len(partition_info) == 0


class TestKVClearE2E:
    """End-to-end tests for kv_clear functionality."""

    def test_kv_clear_single_key(self, controller, tq_api):
        """Test clearing a single key."""
        partition_id = "test_partition"
        key = "clear_single"
        other_key = "clear_other"

        tq_api.kv_put(key=key, partition_id=partition_id, fields={"data": torch.tensor([[1]])}, tag={"id": "single"})
        tq_api.kv_put(
            key=other_key, partition_id=partition_id, fields={"data": torch.tensor([[2]])}, tag={"id": "other"}
        )

        # Clear single key
        tq_api.kv_clear(keys=key, partition_id=partition_id)

        # Verify via kv_list
        partition_info = tq_api.kv_list(partition_id=partition_id)
        assert key not in partition_info[partition_id]
        assert other_key in partition_info[partition_id]

        # Verify via controller - key should be removed
        partition = get_controller_partition(controller, partition_id)
        assert key not in partition.keys_mapping
        assert other_key in partition.keys_mapping

        tq_api.kv_clear(keys=other_key, partition_id=partition_id)

    def test_kv_clear_multiple_keys(self, controller, tq_api):
        """Test clearing multiple keys."""
        partition_id = "test_partition"
        keys = ["clear_multi_0", "clear_multi_1", "clear_multi_2", "clear_multi_3"]

        for i, key in enumerate(keys):
            tq_api.kv_put(key=key, partition_id=partition_id, fields={"data": torch.tensor([[i]])}, tag=None)

        # Clear first 2 keys
        tq_api.kv_clear(keys=keys[:2], partition_id=partition_id)

        # Verify
        partition_info = tq_api.kv_list(partition_id=partition_id)
        assert len(partition_info[partition_id]) == 2
        assert keys[0] not in partition_info[partition_id]
        assert keys[1] not in partition_info[partition_id]
        assert keys[2] in partition_info[partition_id]
        assert keys[3] in partition_info[partition_id]

        tq_api.kv_clear(keys=keys[2:], partition_id=partition_id)


class TestKVE2ECornerCases:
    """End-to-end tests for corner cases."""

    def test_field_expansion_across_samples(self, controller, tq_api):
        """Test that new fields can be added across samples."""
        partition_id = "test_partition"
        keys = ["expand_0", "expand_1"]

        # Put initial fields
        tq_api.kv_put(key=keys[0], partition_id=partition_id, fields={"field_a": torch.tensor([[1]])}, tag=None)

        # Add new field to first key
        tq_api.kv_put(key=keys[0], partition_id=partition_id, fields={"field_b": torch.tensor([[2]])}, tag=None)

        # Add different field to second key
        tq_api.kv_put(
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
        data = tq_api.kv_batch_get(keys=keys, partition_id=partition_id)
        assert "field_a" in data
        assert "field_b" not in data
        assert "field_c" not in data

        tq_api.kv_clear(keys=keys, partition_id=partition_id)


def run_tests():
    """Run all e2e tests manually for debugging."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    run_tests()
