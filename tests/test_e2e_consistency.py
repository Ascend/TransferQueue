import sys
import time
from pathlib import Path

import numpy as np
import pytest
import ray
import torch
from tensordict import NonTensorStack, TensorDict

# Setup paths
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (  # noqa: E402
    SimpleStorageUnit,
    TransferQueueClient,
    TransferQueueController,
)


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def tq_setup(ray_cluster):
    # 1. Start Controller
    controller_actor = TransferQueueController.options(
        name="test_controller",
        get_if_exists=True,
    ).remote()
    controller_info = ray.get(controller_actor.get_zmq_server_info.remote())

    # 2. Start Storage Unit
    storage_actor = SimpleStorageUnit.options(
        name="test_storage",
        get_if_exists=True,
    ).remote(storage_unit_size=10000)
    storage_info = ray.get(storage_actor.get_zmq_server_info.remote())

    # 3. Setup Client

    client_id = "test_e2e_client"
    client = TransferQueueClient(
        client_id=client_id,
        controller_info=controller_info,
    )

    # Initialize Storage Manager (AsyncSimpleStorageManager)
    # We need to manually configure it to know about our specific storage unit
    config = {
        "controller_info": controller_info,
        "storage_unit_infos": {storage_info.id: storage_info},
        "storage_backend_config": {"storage_unit_size": 10000},
    }

    client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

    yield client, controller_actor, storage_actor


def assert_data_equal(original, retrieved, msg=""):
    """Recursively check data equality for various types."""
    if isinstance(original, torch.Tensor):
        assert isinstance(retrieved, torch.Tensor), f"{msg} Type mismatch: expected Tensor, got {type(retrieved)}"
        # Check for nested tensor
        if original.is_nested:
            assert retrieved.is_nested, f"{msg} Expected nested tensor"
            assert len(original) == len(retrieved), f"{msg} Nested tensor length mismatch"
            for i in range(len(original)):
                # Recurse for nested elements
                assert_data_equal(original[i], retrieved[i], msg=f"{msg} Nested index {i} mismatch")
        else:
            # Handle potential NaN/Inf
            # equal_nan=True is generally safe for equality checks in tests
            torch.testing.assert_close(original, retrieved, msg=f"{msg} Tensor mismatch", equal_nan=True)

    elif isinstance(original, list | tuple):
        # If it's a list, retrieved might be a NonTensorStack or list
        if isinstance(retrieved, NonTensorStack | list | tuple):
            assert len(original) == len(retrieved), f"{msg} Length mismatch"
            for i, (o, r) in enumerate(zip(original, retrieved, strict=False)):
                assert_data_equal(o, r, msg=f"{msg} List index {i}")
        else:
            pytest.fail(f"{msg} Type mismatch: expected List/Tuple, got {type(retrieved)}")

    elif isinstance(original, np.ndarray):
        np.testing.assert_array_equal(original, retrieved, err_msg=f"{msg} Numpy array mismatch")

    elif isinstance(original, TensorDict | dict):
        assert isinstance(retrieved, TensorDict | dict), f"{msg} Type mismatch: expected Dict, got {type(retrieved)}"
        for k in original.keys():
            assert k in retrieved, f"{msg} Missing key {k}"
            assert_data_equal(original[k], retrieved[k], msg=f"{msg} Key {k}")

    else:
        # Primitive types
        assert original == retrieved, f"{msg} Value mismatch: {original} != {retrieved}"


@pytest.mark.timeout(60)
def test_consistency_core_types(tq_setup):
    """
    Test Case 1: Core Data Types Coverage
    - Tensor, NestedTensor, Non-Tensor (stackable/non-stackable)
    """
    client, _, _ = tq_setup
    partition_id = "test_core_types"

    # Define test data
    batch_size = 5

    # 1. Normal Tensor
    tensor_data = torch.randn(batch_size, 10)

    # 2. Nested Tensor (Ragged)
    nested_list = [torch.randn(i + 2) for i in range(batch_size)]
    nested_tensor = torch.nested.as_nested_tensor(nested_list, layout=torch.jagged)

    # 3. Stackable Non-Tensor (List of ints)
    list_int = [i * 10 for i in range(batch_size)]

    # 4. Non-Stackable / Scalar-like mixed (Strings)
    list_str = [f"sample_{i}" for i in range(batch_size)]

    # 5. List of numpy arrays
    list_numpy = [np.array([i, i + 1]) for i in range(batch_size)]

    # 6. Numpy Object (Strings/Mixed)
    # TransferQueue should handle this as NonTensor or specific serialization
    np_obj = np.array([f"obj_{i}" for i in range(batch_size)], dtype=object)

    # 7. Special Values (Inf/NaN) & Bool
    special_tensor = torch.zeros(batch_size, 3)
    special_tensor[:, 0] = float("inf")
    special_tensor[:, 1] = float("nan")
    bool_tensor = torch.rand(batch_size, 5) > 0.5

    # 8. Non-contiguous Tensor
    large_t = torch.randn(batch_size, 20)
    non_contiguous = large_t[:, ::2]  # Stride 2

    data = TensorDict(
        {
            "tensor_field": tensor_data,
            "nested_field": nested_tensor,
            "list_int_field": list_int,
            "list_str_field": list_str,
            "list_numpy_field": list_numpy,
            "np_object_field": np_obj,
            "special_field": special_tensor,
            "bool_field": bool_tensor,
            "non_orig_field": non_contiguous,
        },
        batch_size=batch_size,
    )

    # Put Data
    client.put(partition_id=partition_id, data=data)

    # Get Data

    # Poll for metadata until ready
    max_retries = 10
    retrieved_data = None

    fields = [
        "tensor_field",
        "nested_field",
        "list_int_field",
        "list_str_field",
        "np_object_field",
        "special_field",
        "bool_field",
        "non_orig_field",
    ]

    meta = None
    for _ in range(max_retries):
        try:
            meta = client.get_meta(
                partition_id=partition_id,
                data_fields=fields,
                batch_size=batch_size,
                mode="fetch",
                task_name="test_worker",
            )
            break
        except Exception:
            time.sleep(0.5)

    assert meta is not None, "Failed to retrieve metadata"

    retrieved_data = client.get_data(meta)

    # Verification
    assert_data_equal(data["tensor_field"], retrieved_data["tensor_field"], "Tensor Field")
    assert_data_equal(data["nested_field"], retrieved_data["nested_field"], "Nested Field")

    # For Non-Tensor, TransferQueue might return them as NonTensorStack or list
    assert_data_equal(data["list_int_field"], retrieved_data["list_int_field"], "List Int Field")
    assert_data_equal(data["list_str_field"], retrieved_data["list_str_field"], "List Str Field")

    # Verify complex types
    assert_data_equal(data["np_object_field"], retrieved_data["np_object_field"], "Numpy Object Field")

    # Special Floats - NaN needs special check in assert checking/allclose
    assert_data_equal(data["special_field"], retrieved_data["special_field"], "Special Float Field")

    assert_data_equal(data["bool_field"], retrieved_data["bool_field"], "Bool Field")
    assert_data_equal(data["non_orig_field"], retrieved_data["non_orig_field"], "Non-contiguous Field")


@pytest.mark.timeout(120)
def test_consistency_multi_round_put_get(tq_setup):
    """
    Test Case 2: Multi-round Put & Field Merge
    Simulate fragmented writing and field stitching.
    """
    client, _, _ = tq_setup
    partition_id = "test_multi_round"

    # Define Indices
    idx_round1 = list(range(0, 20))
    idx_round2 = list(range(20, 41))  # 21 items

    # ... (gen_data functions same) ...
    # Data Generators with Descriptive Names
    # Group 1: Standard & Scalar Types (Tensor, List[str], List[int], Special, Bool)
    def gen_group_standard(indices):
        n = len(indices)
        # 1. Normal Tensor
        tensor_data = torch.randn(n, 5) + indices[0]
        # 2. List of Strings
        list_str = [f"str_{i}" for i in indices]
        # 3. List of Ints
        list_int = [i * 10 for i in indices]
        # 4. Special Floats
        special_tensor = torch.zeros(n, 3)
        special_tensor[:, 0] = float("inf")
        special_tensor[:, 1] = float("nan")
        # 5. Bool
        bool_tensor = torch.rand(n, 5) > 0.5

        return TensorDict(
            {
                "f_tensor": tensor_data,
                "f_list_str": list_str,
                "f_list_int": list_int,
                "f_special": special_tensor,
                "f_bool": bool_tensor,
            },
            batch_size=n,
        )

    # Group 2: Complex & Nested Types (NestedTensor, List[numpy], NumpyObj, Non-Contig)
    def gen_group_complex(indices):
        n = len(indices)
        # 6. Nested Tensor
        nested_list = [torch.full((i % 5 + 1,), float(i)) for i in indices]
        nested_tensor = torch.nested.as_nested_tensor(nested_list, layout=torch.jagged)
        # 7. List of Numpy Arrays
        list_numpy = [np.array([i, i * 2]) for i in indices]
        # 8. Numpy Object
        np_obj = np.array([f"obj_{i}" for i in indices], dtype=object)
        # 9. Non-contiguous Tensor
        large_t = torch.randn(n, 20)
        non_contiguous = large_t[:, ::2]

        return TensorDict(
            {"f_nested": nested_tensor, "f_list_numpy": list_numpy, "f_np_obj": np_obj, "f_non_contig": non_contiguous},
            batch_size=n,
        )

    # Helper to support updates on specific indices
    def update_samples(global_indices, data):
        import asyncio

        from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta

        samples = []
        field_names = list(data.keys())
        for i, idx in enumerate(global_indices):
            fields_dict = {name: FieldMeta(name=name, dtype=None, shape=None) for name in field_names}
            samples.append(SampleMeta(partition_id=partition_id, global_index=idx, fields=fields_dict))

        meta = BatchMeta(samples=samples)

        future = asyncio.run_coroutine_threadsafe(client.storage_manager.put_data(data, meta), client._loop)
        try:
            future.result(timeout=10)
        except Exception:
            raise

    # Step 1: Pre-allocate Indices in TWO separate batches to test cross-batch retrieval
    # Batch 1 (0-19) | Batch 2 (20-40)
    all_fields = [
        "f_tensor",
        "f_list_str",
        "f_list_int",
        "f_special",
        "f_bool",
        "f_nested",
        "f_list_numpy",
        "f_np_obj",
        "f_non_contig",
    ]

    # Allocation 1: Indices 0-19
    meta_alloc_1 = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=20, mode="insert", task_name="allocator_1"
    )
    idx_round1 = meta_alloc_1.global_indexes
    assert len(idx_round1) == 20

    # Allocation 2: Indices 20-40
    meta_alloc_2 = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=21, mode="insert", task_name="allocator_2"
    )
    idx_round2 = meta_alloc_2.global_indexes
    assert len(idx_round2) == 21

    # Full list for reference
    all_indices = idx_round1 + idx_round2

    # --- Write Operations with Cross-Batch Logic ---

    # Op 1: Put Standard Group for Batch 1 (0-19)
    data_1_std = gen_group_standard(idx_round1)
    update_samples(idx_round1, data_1_std)

    # Op 2: Put Standard Group for Batch 2 (20-40)
    data_2_std = gen_group_standard(idx_round2)
    update_samples(idx_round2, data_2_std)

    # Op 3: Cross-batch Put for Complex Group (indices 15-25)
    idx_cross = all_indices[15:26]  # 15 to 25 inclusive
    data_cross_complex = gen_group_complex(idx_cross)
    update_samples(idx_cross, data_cross_complex)

    # Op 4: Fill remaining Complex Group (0-14)
    idx_remaining_1 = all_indices[0:15]
    data_rem_1_complex = gen_group_complex(idx_remaining_1)
    update_samples(idx_remaining_1, data_rem_1_complex)

    # Op 5: Fill remaining Complex Group (26-40)
    idx_remaining_2 = all_indices[26:]
    data_rem_2_complex = gen_group_complex(idx_remaining_2)
    update_samples(idx_remaining_2, data_rem_2_complex)

    # Verifications

    # 1. Get 0-10
    meta_1 = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=10, mode="fetch", task_name="verifier_1"
    )
    res_1 = client.get_data(meta_1)
    assert len(res_1["f_tensor"]) == 10

    # Verify presence of all fields
    for field in all_fields:
        assert field in res_1, f"Missing field {field}"

    # Verify NestedTensor property
    assert res_1["f_nested"].is_nested, "f_nested should be a NestedTensor"

    # 2. Get 10-30
    meta_2 = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=20, mode="fetch", task_name="verifier_1"
    )
    res_2 = client.get_data(meta_2)
    assert len(res_2["f_tensor"]) == 20

    # 3. Get 30-40
    meta_3 = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=11, mode="fetch", task_name="verifier_1"
    )
    res_3 = client.get_data(meta_3)
    assert len(res_3["f_tensor"]) == 11


@pytest.mark.timeout(60)
def test_consistency_slicing_and_subset(tq_setup):
    """
    Test Case 3: Slicing and Field Subsetting
    """
    client, _, _ = tq_setup
    partition_id = "test_slicing"

    # Pre-allocate Partition

    all_fields = [
        "F_tensor",
        "F_nested",
        "F_list_int",
        "F_list_str",
        "F_list_numpy",
        "F_np_obj",
        "F_special",
        "F_bool",
        "F_non_contig",
    ]
    meta_alloc = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=20, mode="insert", task_name="allocator"
    )
    # USE THE ALLOCATED INDICES
    indices = meta_alloc.global_indexes
    assert len(indices) == 20

    # Create Data using these indices
    n = len(indices)

    # 1. Tensor
    tensor_data = torch.randn(n, 5) + indices[0]
    # 2. Nested
    nested_list = [torch.tensor([i, i * 2, i * 3]) if i % 2 == 0 else torch.tensor([i]) for i in indices]
    nested_tensor = torch.nested.as_nested_tensor(nested_list, layout=torch.jagged)
    # 3. List int
    list_int = [i * 10 for i in indices]
    # 4. List str
    list_str = [f"slice_{i}" for i in indices]
    # 5. List numpy
    list_numpy = [np.array([i]) for i in indices]
    # 6. Numpy Object
    np_obj = np.array([f"obj_{i}" for i in indices], dtype=object)
    # 7. Special
    special_tensor = torch.zeros(n, 3)
    special_tensor[:, 0] = float("inf")
    # 8. Bool
    bool_tensor = torch.rand(n, 1) > 0.5
    # 9. Non-contig
    large_t = torch.randn(n, 10)
    non_contig = large_t[:, ::2]

    data = TensorDict(
        {
            "F_tensor": tensor_data,
            "F_nested": nested_tensor,
            "F_list_int": list_int,
            "F_list_str": list_str,
            "F_list_numpy": list_numpy,
            "F_np_obj": np_obj,
            "F_special": special_tensor,
            "F_bool": bool_tensor,
            "F_non_contig": non_contig,
        },
        batch_size=20,
    )

    # Helper for manual put
    def update_samples(global_indices, data):
        import asyncio

        from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta

        samples = []
        field_names = list(data.keys())
        for i, idx in enumerate(global_indices):
            # Populate fields just in case
            fields_dict = {name: FieldMeta(name=name, dtype=None, shape=None) for name in field_names}
            samples.append(SampleMeta(partition_id=partition_id, global_index=idx, fields=fields_dict))

        meta = BatchMeta(samples=samples)
        future = asyncio.run_coroutine_threadsafe(client.storage_manager.put_data(data, meta), client._loop)
        future.result()

    update_samples(indices, data)

    # 1. Field Subset: Get only F_nested (Ragged)
    meta_subset = client.get_meta(
        partition_id=partition_id,
        data_fields=["F_nested"],
        batch_size=20,  # Ignored in force_fetch
        mode="force_fetch",
        task_name="inspector",
    )
    res_subset = client.get_data(meta_subset)

    assert "F_nested" in res_subset
    assert "F_tensor" not in res_subset
    assert len(res_subset["F_nested"]) == 20

    fetched_indices = meta_subset.global_indexes
    fetched_values = res_subset["F_nested"]

    # Verify F_nested is NestedTensor
    assert fetched_values.is_nested

    for idx, val in zip(fetched_indices, fetched_values, strict=False):
        if idx % 2 == 0:
            expected = torch.tensor([idx, idx * 2, idx * 3])
        else:
            expected = torch.tensor([idx])

        torch.testing.assert_close(val, expected.to(val.dtype))

    # 2. Get F_np_obj (Numpy Objects/Mixed) & F_special (Inf/Nan)
    meta_mixed = client.get_meta(
        partition_id=partition_id,
        data_fields=["F_np_obj", "F_special"],
        batch_size=20,
        mode="force_fetch",
        task_name="inspector_2",
    )
    res_mixed = client.get_data(meta_mixed)
    assert "F_np_obj" in res_mixed
    assert "F_special" in res_mixed

    # Verify F_np_obj
    fetched_indices_mixed = meta_mixed.global_indexes
    fetched_obj = res_mixed["F_np_obj"]

    assert len(fetched_obj) == 20
    for idx, val in zip(fetched_indices_mixed, fetched_obj, strict=False):
        expected_val = f"obj_{idx}"
        assert val == expected_val, f"Mismatch for index {idx}: {val} != {expected_val}"

    # Verify F_special (checking logic roughly, we know it has Inf)
    fetched_special = res_mixed["F_special"]
    assert torch.isinf(fetched_special).any(), "Expected Inf values in special tensor"


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
