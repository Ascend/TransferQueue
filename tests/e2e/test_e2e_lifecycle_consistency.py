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

"""E2E lifecycle consistency tests for TransferQueue."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

# Setup paths (transfer_queue is not pip-installed)
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Module-level default fields to avoid repeated generation
DEFAULT_FIELDS = [
    "tensor_f32",
    "tensor_i64",
    "tensor_bf16",
    "nested_jagged",
    "nested_strided",
    "list_int",
    "list_str",
    "list_obj",
    "np_array",
    "np_obj",
    "special_val",
    "non_tensor_stack",
]


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def e2e_client(ray_cluster):
    """Create a client using transfer_queue.init() for lifecycle testing."""
    from omegaconf import OmegaConf

    import transfer_queue

    config = {
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
    }
    transfer_queue.init(OmegaConf.create(config))
    client = transfer_queue.get_client()
    yield client
    transfer_queue.close()


def generate_complex_data(indices: list[int]) -> TensorDict:
    """Generate complex TensorDict with all supported field types."""
    n = len(indices)

    # Standard Tensor (Float32)
    tensor_f32 = torch.stack([torch.arange(i, i + 5, dtype=torch.float32) for i in indices])

    # Standard Tensor (Int64)
    tensor_i64 = torch.stack([torch.arange(i, i + 5, dtype=torch.int64) for i in indices])

    # Nested Tensor (Jagged)
    nested_list = []
    for i in indices:
        length = 2 + (i % 4)  # Variable length: 2-5
        nested_list.append(torch.arange(i, i + length, dtype=torch.float32))
    # Inject special values into jagged tensor components
    nested_list[0][0] = float("inf")
    if len(nested_list) > 1:
        nested_list[1][0] = float("nan")
    nested_jagged = torch.nested.as_nested_tensor(nested_list, layout=torch.jagged)

    # Nested Tensor (Strided)
    strided_tensors = [torch.full((3, 4), float(i)) for i in indices]
    nested_strided = torch.nested.nested_tensor(strided_tensors, layout=torch.strided)

    # Python Lists
    list_int = [i * 10 for i in indices]
    list_str = [f"sample_{i}" for i in indices]

    # NumPy Arrays
    np_array = np.array([np.arange(i, i + 3) for i in indices], dtype=np.float64)
    np_obj = np.array([f"obj_{i}" for i in indices], dtype=object)

    # Special Values (NaN and Inf)
    special_val = torch.zeros(n, 3)
    special_val[:, 0] = float("inf")
    special_val[:, 1] = float("nan")
    special_val[:, 2] = torch.tensor(indices, dtype=torch.float32)

    # NonTensorData
    non_tensor_data = [{"idx": i, "text": f"non_tensor_{i}"} for i in indices]
    non_tensor_stack = NonTensorData(data=non_tensor_data, batch_size=(n,), device=None)

    # BFloat16 Tensor
    tensor_bf16 = torch.stack([torch.arange(i, i + 5, dtype=torch.bfloat16) for i in indices])

    # List of objects (dicts)
    list_obj = [{"key": f"value_{i}", "num": i} for i in indices]

    field_values = {
        "tensor_f32": tensor_f32,
        "tensor_i64": tensor_i64,
        "tensor_bf16": tensor_bf16,
        "nested_jagged": nested_jagged,
        "nested_strided": nested_strided,
        "list_int": list_int,
        "list_str": list_str,
        "list_obj": list_obj,
        "np_array": np_array,
        "np_obj": np_obj,
        "special_val": special_val,
        "non_tensor_stack": non_tensor_stack,
    }

    # Validate: field_values must exactly match DEFAULT_FIELDS
    assert set(field_values.keys()) == set(DEFAULT_FIELDS), (
        f"generate_complex_data fields mismatch with DEFAULT_FIELDS: "
        f"extra={set(field_values.keys()) - set(DEFAULT_FIELDS)}, "
        f"missing={set(DEFAULT_FIELDS) - set(field_values.keys())}"
    )

    return TensorDict(
        {field: field_values[field] for field in DEFAULT_FIELDS},
        batch_size=n,
    )


def poll_for_meta(client, partition_id, data_fields, batch_size, task_name, mode="fetch", max_retries=10):
    """Poll until metadata is ready or max retries reached."""
    for _ in range(max_retries):
        meta = client.get_meta(
            partition_id=partition_id,
            data_fields=data_fields,
            batch_size=batch_size,
            mode=mode,
            task_name=task_name,
        )
        if meta is not None and meta.size > 0:
            return meta
        time.sleep(0.3)
    return None


# Helper Functions for Data Verification
def verify_special_values(retrieved: torch.Tensor, expected: torch.Tensor) -> bool:
    """Verify special values (NaN, Inf) are preserved."""
    # Check Inf column
    if not torch.all(torch.isinf(retrieved[:, 0]) & (retrieved[:, 0] > 0)):
        return False
    # Check NaN column
    if not torch.all(torch.isnan(retrieved[:, 1])):
        return False
    # Check regular values column
    if not torch.allclose(retrieved[:, 2], expected[:, 2]):
        return False
    return True


def verify_nested_tensor_equal(retrieved, expected) -> bool:
    """Verify nested tensors element by element, handling NaN/Inf."""
    r_list = retrieved.unbind()
    e_list = expected.unbind()
    if len(r_list) != len(e_list):
        return False
    for r, e in zip(r_list, e_list, strict=True):
        # Handle NaN: positions must match
        r_nan = torch.isnan(r)
        e_nan = torch.isnan(e)
        if not torch.equal(r_nan, e_nan):
            return False
        # Compare non-NaN values (allclose handles Inf correctly)
        mask = ~r_nan
        if mask.any() and not torch.allclose(r[mask], e[mask]):
            return False
    return True


def verify_non_tensor_data(retrieved, expected) -> bool:
    """Verify NonTensorData content element by element."""
    if hasattr(retrieved, "tolist"):
        retrieved = retrieved.tolist()
    elif hasattr(retrieved, "data"):
        retrieved = retrieved.data
    if hasattr(expected, "tolist"):
        expected = expected.tolist()
    elif hasattr(expected, "data"):
        expected = expected.data
    if isinstance(retrieved, list) and isinstance(expected, list):
        if len(retrieved) != len(expected):
            return False
        return all(r == e for r, e in zip(retrieved, expected, strict=True))
    return retrieved == expected


def verify_list_equal(retrieved, expected) -> bool:
    """Verify list content.

    Note: TensorDict may materialize Python lists as Tensors or NonTensorStack during
    storage/retrieval, so we normalize both sides to native Python types before comparison.
    """
    from tensordict.tensorclass import NonTensorStack  # local import to avoid circular deps

    if isinstance(retrieved, NonTensorStack):
        retrieved = retrieved.tolist()
    elif isinstance(retrieved, torch.Tensor):
        retrieved = retrieved.tolist()
    if isinstance(expected, NonTensorStack):
        expected = expected.tolist()
    elif isinstance(expected, torch.Tensor):
        expected = expected.tolist()
    return retrieved == expected


def _reorder_tensordict(td: TensorDict, order: list[int]) -> TensorDict:
    """Reorder a TensorDict by the given index order.

    Handles regular tensors, nested/jagged tensors, NonTensorStack, lists, and other
    indexable types.
    """
    from tensordict.tensorclass import NonTensorStack  # local import to avoid circular deps

    reordered = {}
    for key in td.keys():
        field = td[key]
        if isinstance(field, NonTensorStack):
            # NonTensorStack: reorder by converting to list and re-wrapping
            items = field.tolist()
            reordered_items = [items[i] for i in order]
            reordered[key] = NonTensorStack(*reordered_items, batch_size=[len(order)])
        elif hasattr(field, "unbind"):
            items = field.unbind(0)
            reordered_items = [items[i] for i in order]
            try:
                reordered[key] = torch.stack(reordered_items)
            except (RuntimeError, TypeError):
                # RuntimeError: shape mismatch (jagged); TypeError: non-Tensor items
                reordered[key] = torch.nested.as_nested_tensor(reordered_items, layout=field.layout)
        elif isinstance(field, list):
            reordered[key] = [field[i] for i in order]
        else:
            reordered[key] = field[torch.tensor(order)]
    return TensorDict(reordered, batch_size=td.batch_size)


# Scenario One: Core Read/Write Consistency
def test_core_consistency(e2e_client):
    """Put full complex data then get — verify all field types are correctly round-tripped."""
    client = e2e_client
    partition_id = "test_core_consistency"
    batch_size = 20
    task_name = "core_consistency_task"

    # 1. Put full complex data
    indices = list(range(batch_size))
    original_data = generate_complex_data(indices)
    fields = DEFAULT_FIELDS

    meta = client.put(data=original_data, partition_id=partition_id)
    assert meta.size == batch_size, f"Expected batch_size {batch_size}, got {meta.size}"
    try:
        # 2. Get data
        retrieved_meta = poll_for_meta(client, partition_id, fields, batch_size, task_name, mode="fetch")
        assert retrieved_meta is not None and retrieved_meta.size == batch_size, "Failed to retrieve metadata"
        retrieved_data = client.get_data(retrieved_meta)

        # 3. Verify Standard Tensors
        assert torch.allclose(retrieved_data["tensor_f32"], original_data["tensor_f32"]), "tensor_f32 mismatch"
        assert torch.equal(retrieved_data["tensor_i64"], original_data["tensor_i64"]), "tensor_i64 mismatch"
        assert torch.equal(retrieved_data["tensor_bf16"], original_data["tensor_bf16"]), "tensor_bf16 mismatch"

        # 4. Verify Nested Tensors (Jagged)
        assert verify_nested_tensor_equal(retrieved_data["nested_jagged"], original_data["nested_jagged"]), (
            "Jagged nested tensor mismatch"
        )

        # 5. Verify Nested Tensors (Strided)
        assert verify_nested_tensor_equal(retrieved_data["nested_strided"], original_data["nested_strided"]), (
            "Strided nested tensor mismatch"
        )

        # 6. Verify Python Lists
        assert verify_list_equal(retrieved_data["list_int"], original_data["list_int"]), "list_int mismatch"
        assert verify_list_equal(retrieved_data["list_str"], original_data["list_str"]), "list_str mismatch"
        assert verify_list_equal(retrieved_data["list_obj"], original_data["list_obj"]), "list_obj mismatch"

        # 7. Verify NumPy Arrays
        assert np.allclose(retrieved_data["np_array"], original_data["np_array"]), "np_array mismatch"
        # np_obj may be returned as NonTensorStack; normalize to list before comparing
        retrieved_np_obj = retrieved_data["np_obj"]
        if hasattr(retrieved_np_obj, "tolist"):
            retrieved_np_obj = retrieved_np_obj.tolist()
        expected_np_obj = original_data["np_obj"]
        if hasattr(expected_np_obj, "tolist") and not isinstance(expected_np_obj, np.ndarray):
            expected_np_obj = expected_np_obj.tolist()
        assert list(retrieved_np_obj) == list(expected_np_obj), "np_obj mismatch"

        # 8. Verify Special Values (NaN and Inf)
        assert verify_special_values(retrieved_data["special_val"], original_data["special_val"]), (
            "Special values (NaN/Inf) not preserved"
        )

        # 9. Verify NonTensorData
        assert verify_non_tensor_data(retrieved_data["non_tensor_stack"], original_data["non_tensor_stack"]), (
            "NonTensorData content mismatch"
        )
    finally:
        client.clear_partition(partition_id)


# Scenario Two: Cross-Shard Update
def test_cross_shard_complex_update(e2e_client):
    """Cross-shard update: put A + put B, update overlapping region, verify all regions."""
    client = e2e_client
    partition_id = "test_cross_shard_update"
    task_name = "cross_shard_task"

    # Define index ranges
    idx_a = list(range(0, 20))  # Put A
    idx_b = list(range(20, 40))  # Put B
    idx_update = list(range(10, 30))  # Update (cross-shard)
    base_fields = DEFAULT_FIELDS

    # 1. Allocate full partition
    alloc_meta = client.get_meta(
        partition_id=partition_id,
        data_fields=base_fields,
        batch_size=40,
        mode="insert",
        task_name="allocator",
    )
    assert len(alloc_meta.global_indexes) == 40, "Failed to allocate 40 samples"

    try:
        # 2. Put A: indices 0-19
        data_a = generate_complex_data(idx_a)
        meta_a = alloc_meta.select_samples(list(range(0, 20)))
        client.put(data=data_a, metadata=meta_a)

        # 3. Put B: indices 20-39
        data_b = generate_complex_data(idx_b)
        meta_b = alloc_meta.select_samples(list(range(20, 40)))
        client.put(data=data_b, metadata=meta_b)

        # 4. Update indices 10-29 with modified values and new fields
        modified_indices = [i + 1000 for i in idx_update]  # Offset to make values distinguishable
        data_update = generate_complex_data(modified_indices)

        # Add new fields
        new_extra_tensor = torch.stack([torch.ones(3) * i for i in idx_update])  # Shape: (20, 3)
        new_extra_non_tensor = NonTensorData(
            data=[{"new_field": f"new_{i}"} for i in idx_update],
            batch_size=(len(idx_update),),
            device=None,
        )
        data_update["new_extra_tensor"] = new_extra_tensor
        data_update["new_extra_non_tensor"] = new_extra_non_tensor

        # Put update data
        meta_update = alloc_meta.select_samples(list(range(10, 30)))
        client.put(data=data_update, metadata=meta_update)

        # 5. Get Full: indices 0-39, only base fields first
        full_meta = poll_for_meta(client, partition_id, base_fields, 40, task_name, mode="force_fetch")
        assert full_meta is not None and full_meta.size == 40, "Failed to retrieve full metadata"
        full_data = client.get_data(full_meta)

        # Reorder by global_indexes for deterministic positional assertions
        sorted_order = sorted(range(full_meta.size), key=lambda i: full_meta.global_indexes[i])
        if sorted_order != list(range(full_meta.size)):
            full_data = _reorder_tensordict(full_data, sorted_order)

        # 6. Verify region 0-9: original Put A values
        original_data_0_9 = generate_complex_data(list(range(0, 10)))
        assert torch.allclose(full_data["tensor_f32"][:10], original_data_0_9["tensor_f32"]), (
            "Region 0-9 tensor_f32 should match original Put A"
        )

        # 7. Verify region 10-29: updated values (using offset indices 1010-1029)
        updated_expected = generate_complex_data([i + 1000 for i in range(10, 30)])
        assert torch.allclose(full_data["tensor_f32"][10:30], updated_expected["tensor_f32"]), (
            "Region 10-29 tensor_f32 should match updated values"
        )

        # 8. Verify region 30-39: original Put B values
        original_data_30_39 = generate_complex_data(list(range(30, 40)))
        assert torch.allclose(full_data["tensor_f32"][30:40], original_data_30_39["tensor_f32"]), (
            "Region 30-39 tensor_f32 should match original Put B"
        )

        # 9. Verify new fields exist in update region (indices 10-29 only have new fields).
        # Build extended_meta from full_meta (which has valid _custom_backend_meta/_su_id)
        # by selecting the subset of samples whose global_indexes match meta_update.
        # Using meta_update directly would fail because it was derived from alloc_meta
        # before put(), so its _custom_backend_meta lacks _su_id.
        update_gis = set(meta_update.global_indexes)
        update_positions_in_full = [i for i, gi in enumerate(full_meta.global_indexes) if gi in update_gis]
        update_meta_with_backend = full_meta.select_samples(update_positions_in_full)
        extended_meta = update_meta_with_backend.with_data_fields(
            base_fields + ["new_extra_tensor", "new_extra_non_tensor"]
        )
        update_region_data = client.get_data(extended_meta)
        assert "new_extra_tensor" in update_region_data.keys(), "new_extra_tensor should exist"
        assert "new_extra_non_tensor" in update_region_data.keys(), "new_extra_non_tensor should exist"
    finally:
        client.clear_partition(partition_id)


# Scenario Three: Production Status Lifecycle
def test_production_status_lifecycle(e2e_client):
    """Multi-round partial put: verify production & consumption status transitions."""
    client = e2e_client
    partition_id = "test_production_lifecycle"
    batch_size = 10
    task_name = "production_lifecycle_task"

    # Define field sets
    set_a_fields = ["tensor_f32", "tensor_i64", "list_int", "list_str"]
    set_b_fields = ["nested_jagged", "np_array", "special_val"]
    all_fields = set_a_fields + set_b_fields

    # 1. Allocate partition with all fields
    alloc_meta = client.get_meta(
        partition_id=partition_id,
        data_fields=all_fields,
        batch_size=batch_size,
        mode="insert",
        task_name="allocator",
    )
    indices = alloc_meta.global_indexes
    assert len(indices) == batch_size, f"Expected {batch_size} indices, got {len(indices)}"

    try:
        # 2. Round 1: Put only Set_A fields
        full_data = generate_complex_data(indices)
        set_a_data = full_data.select(*set_a_fields)
        client.put(data=set_a_data, metadata=alloc_meta)

        # 3. Check Production Status - Set_A should be ready, Set_B should not
        set_a_ready = client.check_production_status(data_fields=set_a_fields, partition_id=partition_id)
        assert set_a_ready, "Set_A fields should be ready after Round 1"

        set_b_ready = client.check_production_status(data_fields=set_b_fields, partition_id=partition_id)
        assert not set_b_ready, "Set_B fields should NOT be ready after Round 1"

        all_ready_before = client.check_production_status(data_fields=all_fields, partition_id=partition_id)
        assert not all_ready_before, "All fields should NOT be ready before Round 2"

        # 4. Round 2: Put Set_B fields
        set_b_data = full_data.select(*set_b_fields)
        client.put(data=set_b_data, metadata=alloc_meta)

        # 5. Check Production Status - All should be ready
        all_ready_after = client.check_production_status(data_fields=all_fields, partition_id=partition_id)
        assert all_ready_after, "All fields should be ready after Round 2"

        # 6. Consumption Status Check - should be False initially
        is_consumed = client.check_consumption_status(task_name=task_name, partition_id=partition_id)
        assert not is_consumed, "Data should not be consumed initially"

        # 7. Consume Data (consumption is marked during get_meta(fetch))
        meta = poll_for_meta(client, partition_id, all_fields, batch_size, task_name, mode="fetch")
        assert meta is not None, "Failed to poll metadata"

        # Consumption is marked during get_meta(fetch), verify before get_data
        is_consumed_mid = client.check_consumption_status(task_name=task_name, partition_id=partition_id)
        assert is_consumed_mid, "Data should already be consumed after get_meta in fetch mode"

        client.get_data(meta)

        # 8. Post-Consumption Check - should be True
        is_consumed_after = client.check_consumption_status(task_name=task_name, partition_id=partition_id)
        assert is_consumed_after, "Data should be consumed after get_meta in fetch mode"
    finally:
        client.clear_partition(partition_id)


# Scenario Four: Custom Metadata Persistence
def test_custom_metadata_persistence(e2e_client):
    """Set per-sample custom metadata, retrieve it, and verify persistence."""
    client = e2e_client
    partition_id = "test_custom_meta"
    batch_size = 8
    task_name = "custom_meta_task"
    fields = DEFAULT_FIELDS

    # 1. Allocate and Put Data
    meta = client.put(
        data=generate_complex_data(list(range(batch_size))),
        partition_id=partition_id,
    )
    assert meta.size == batch_size, f"Expected batch_size {batch_size}, got {meta.size}"

    try:
        # 2. Create Custom Metadata for each sample
        custom_metadata_list = [
            {
                "score": float(i) / 10.0,
                "label": f"label_{i}",
                "tags": [f"tag_{i}_a", f"tag_{i}_b"],
            }
            for i in range(batch_size)
        ]
        meta.update_custom_meta(custom_metadata_list)

        # 3. Upload Custom Metadata
        client.set_custom_meta(meta)

        # 4. Retrieve Metadata and Verify Custom Meta
        retrieved_meta = poll_for_meta(client, partition_id, fields, batch_size, task_name, mode="force_fetch")
        assert retrieved_meta is not None, "Failed to retrieve metadata"

        # Verify custom metadata content
        retrieved_custom = retrieved_meta.get_all_custom_meta()
        assert len(retrieved_custom) == batch_size, (
            f"Expected {batch_size} custom_meta entries, got {len(retrieved_custom)}"
        )
        for i, (actual, expected) in enumerate(zip(retrieved_custom, custom_metadata_list, strict=True)):
            assert actual["score"] == expected["score"], f"Score mismatch at sample {i}"
            assert actual["label"] == expected["label"], f"Label mismatch at sample {i}"
            assert actual["tags"] == expected["tags"], f"Tags mismatch at sample {i}"
    finally:
        client.clear_partition(partition_id)


# Scenario Five: Reset & Clear
def test_reset_consumption(e2e_client):
    """Consume data, reset consumption status, verify re-consumability."""
    client = e2e_client
    partition_id = "test_reset_consumption"
    batch_size = 10
    task_name = "reset_test_task"
    fields = DEFAULT_FIELDS

    # 1. Put Data
    data = generate_complex_data(list(range(batch_size)))
    client.put(data=data, partition_id=partition_id)

    try:
        # 2. Initial Consumption Status Check - should be False (not consumed)
        is_consumed_initial = client.check_consumption_status(task_name=task_name, partition_id=partition_id)
        assert not is_consumed_initial, "Data should not be consumed initially"

        # 3. Consume Data (consumption is marked during get_meta(fetch))
        meta = poll_for_meta(client, partition_id, fields, batch_size, task_name, mode="fetch")
        assert meta is not None and meta.size == batch_size, "Failed to poll metadata"

        # Consumption is marked during get_meta(fetch), verify before get_data
        is_consumed_mid = client.check_consumption_status(task_name=task_name, partition_id=partition_id)
        assert is_consumed_mid, "Data should already be consumed after get_meta in fetch mode"

        retrieved_data = client.get_data(meta)
        assert retrieved_data.batch_size[0] == batch_size, "Retrieved data batch_size mismatch"

        # 4. Post-Consumption Status Check - should be True
        is_consumed_after = client.check_consumption_status(task_name=task_name, partition_id=partition_id)
        assert is_consumed_after, "Data should be consumed after get_meta in fetch mode"

        # 5. Reset Consumption
        success = client.reset_consumption(partition_id=partition_id, task_name=task_name)
        assert success, "reset_consumption should return True"

        # 6. Post-Reset Status Check - should be False again
        is_consumed_reset = client.check_consumption_status(task_name=task_name, partition_id=partition_id)
        assert not is_consumed_reset, "Consumption status should be False after reset"

        # 7. Verify data can be re-consumed
        meta_again = poll_for_meta(client, partition_id, fields, batch_size, task_name, mode="fetch")
        assert meta_again is not None and meta_again.size == batch_size, "Should be able to fetch metadata again"
    finally:
        client.clear_partition(partition_id)


def test_clear_partition(e2e_client):
    """Clear partition: verify data removal and production status reset."""
    client = e2e_client
    partition_id = "test_clear_partition"
    batch_size = 15
    task_name = "clear_test_task"
    fields = DEFAULT_FIELDS

    # 1. Put Data
    data = generate_complex_data(list(range(batch_size)))
    client.put(data=data, partition_id=partition_id)

    try:
        # 2. Verify Data Exists - production status should be True
        is_ready = client.check_production_status(data_fields=fields, partition_id=partition_id)
        assert is_ready, "Data should be ready after put"

        # 3. Get Data to confirm accessibility
        meta = poll_for_meta(client, partition_id, fields, batch_size, task_name, mode="force_fetch")
        assert meta is not None and meta.size == batch_size, "Failed to poll metadata"

        # 4. Verify partition exists before clear
        partition_list_before = client.get_partition_list()
        assert partition_id in partition_list_before, "Partition should exist before clear"

        # 5. Clear Partition
        client.clear_partition(partition_id)

        # 6. Verify partition is removed from list
        partition_list_after = client.get_partition_list()
        assert partition_id not in partition_list_after, "Partition should be removed after clear"

        # 7. Verify Production Status returns False for cleared partition
        is_ready_after_clear = client.check_production_status(data_fields=fields, partition_id=partition_id)
        assert not is_ready_after_clear, "Production status should be False after clear"
    finally:
        # Ensure cleanup even if assertions fail
        try:
            client.clear_partition(partition_id)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
