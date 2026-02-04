# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import collections.abc
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ray
import torch
from tensordict import TensorDict

# Setup paths
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (  # noqa: E402
    SimpleStorageUnit,
    TransferQueueClient,
    TransferQueueController,
)
from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta  # noqa: E402


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def e2e_client(ray_cluster):
    controller_actor = TransferQueueController.options(
        name="test_controller",
        get_if_exists=True,
    ).remote()
    controller_info = ray.get(controller_actor.get_zmq_server_info.remote())

    # Start two storage units to test sharding/scatter
    storage_actor_1 = SimpleStorageUnit.options(
        name="test_storage_1",
        get_if_exists=True,
    ).remote(storage_unit_size=10000)
    storage_info_1 = ray.get(storage_actor_1.get_zmq_server_info.remote())

    storage_actor_2 = SimpleStorageUnit.options(
        name="test_storage_2",
        get_if_exists=True,
    ).remote(storage_unit_size=10000)
    storage_info_2 = ray.get(storage_actor_2.get_zmq_server_info.remote())

    client_id = "test_e2e_client"
    client = TransferQueueClient(
        client_id=client_id,
        controller_info=controller_info,
    )

    # Initialize Storage Manager (AsyncSimpleStorageManager) and configure it
    config = {
        "controller_info": controller_info,
        "storage_unit_infos": {
            storage_info_1.id: storage_info_1,
            storage_info_2.id: storage_info_2,
        },
        "storage_backend_config": {"storage_unit_size": 10000},
    }

    client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

    yield client


def generate_consistency_data(indices: list[int]) -> TensorDict:
    """
    Generates a TensorDict with deterministic data based on the provided indices.
    Includes all field types used in consistency tests.
    """
    n = len(indices)

    # Standard Tensor: shape (n, 5), values based on index
    tensor_data = torch.stack([torch.arange(i, i + 5, dtype=torch.float32) for i in indices])

    # Nested Tensor (Jagged):
    nested_list = []
    for i in indices:
        if i % 2 == 0:
            nested_list.append(torch.tensor([i, i * 2, i * 3], dtype=torch.float32))
        else:
            nested_list.append(torch.tensor([i], dtype=torch.float32))
    nested_tensor = torch.nested.as_nested_tensor(nested_list, layout=torch.jagged)

    # Strided Nested Tensor (or fallback)
    try:
        tensors = [torch.full((3, 4), float(i)) for i in indices]
        strided_nested = torch.nested.nested_tensor(tensors, layout=torch.strided)
    except Exception:
        # Fallback for environments without strided nested support
        tensors = [torch.full((3, 4), float(i)) for i in indices]
        strided_nested = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

    # List of Ints
    int_values = [i * 10 for i in indices]

    # List of Strings
    str_values = [f"str_{i}" for i in indices]

    # List of Numpy Arrays
    numpy_list = [np.array([i, i + 1], dtype=np.int64) for i in indices]

    # Numpy Object Array (Strings)
    numpy_objects = np.array([f"obj_{i}" for i in indices], dtype=object)

    # Special Values (Inf/NaN)
    special_tensor = torch.zeros(n, 3)
    special_tensor[:, 0] = float("inf")
    special_tensor[:, 1] = float("nan")
    special_tensor[:, 2] = torch.tensor(indices, dtype=torch.float32)

    # Bool Tensor
    bool_tensor = torch.tensor([[i % 2 == 0] for i in indices], dtype=torch.bool)

    # Non-contiguous Tensor
    # orig shape (n, 20), slice ::2 -> (n, 10).
    large_tensor = torch.stack([torch.full((20,), float(i)) for i in indices])
    non_contiguous_tensor = large_tensor[:, ::2]

    return TensorDict(
        {
            "tensor_field": tensor_data,
            "nested_field": nested_tensor,
            "strided_nested_field": strided_nested,
            "list_int_field": int_values,
            "list_str_field": str_values,
            "list_numpy_field": numpy_list,
            "np_object_field": numpy_objects,
            "special_field": special_tensor,
            "bool_field": bool_tensor,
            "non_orig_field": non_contiguous_tensor,
        },
        batch_size=n,
    )


def compute_data_hash(data: Any, algorithm="sha256") -> str:
    """
    Computes a structure-agnostic hash of the data.
    - Flattens nested structures (lists, tuples, tensors).
    - Ignores container types.
    - Normalizes Tensors (detach, cpu).
    - Consistent for NestedTensor vs List[Tensor].
    - Consistent for List[int] vs Tensor(int).
    """
    hasher = hashlib.new(algorithm)
    _update_hash(hasher, data)
    return hasher.hexdigest()


def _update_hash(hasher, data):
    if isinstance(data, dict | TensorDict) or (hasattr(data, "keys") and hasattr(data, "__getitem__")):
        # Check if it behaves like a mapping
        try:
            keys = sorted(list(data.keys()))
            for k in keys:
                _hash_scalar(hasher, str(k))
                _update_hash(hasher, data[k])
            return
        except TypeError:
            pass

    if isinstance(data, torch.Tensor):
        if data.is_nested:
            for t in data.unbind():
                _update_hash(hasher, t)
            return

        data = data.detach().cpu()
        if data.ndim == 0:
            _hash_scalar(hasher, data.item())
        return

    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            _hash_scalar(hasher, data.item())
        elif data.dtype == object:
            for item in data:
                _update_hash(hasher, item)
        else:
            for item in data:
                _update_hash(hasher, item)
        return

    if isinstance(data, str | bytes):
        _hash_scalar(hasher, data)
        return

    # Handles list, tuple, and any custom Sequence
    if isinstance(data, collections.abc.Sequence):
        for item in data:
            _update_hash(hasher, item)
        return

    _hash_scalar(hasher, data)


def _hash_scalar(hasher, data):
    s = str(data)
    if s == "nan":
        s = "NAN"
    hasher.update(s.encode("utf-8"))
    hasher.update(b"|")


def put_data_with_indices(client, partition_id, indices, data):
    """
    Helper to put data with specific global indices.
    """
    samples = []
    field_names = list(data.keys())

    # Pre-compute field metas
    for idx in indices:
        fields_dict = {name: FieldMeta(name=name, dtype=None, shape=None) for name in field_names}
        samples.append(SampleMeta(partition_id=partition_id, global_index=idx, fields=fields_dict))

    meta = BatchMeta(samples=samples)

    future = asyncio.run_coroutine_threadsafe(client.storage_manager.put_data(data, meta), client._loop)
    # Wait for result
    future.result(timeout=30)


def get_fields_subset(indices, field_names):
    """Helper to slice TensorDict by fields."""
    full_data = generate_consistency_data(indices)
    # Select only specific fields
    # Note: TensorDict.select returns a new TensorDict with shared storage
    return full_data.select(*field_names)


def verify_data_consistency(
    client, partition_id: str, task_name: str, data_fields: list[str], batch_size: int, mode: str = "fetch"
) -> TensorDict:
    """Helper to retrieve data and verify it matches deterministic generation logic."""
    # Poll for metadata until ready
    max_retries = 10
    meta = None
    for _ in range(max_retries):
        try:
            meta = client.get_meta(
                partition_id=partition_id,
                data_fields=data_fields,
                batch_size=batch_size,
                mode=mode,
                task_name=task_name,
            )
            break
        except Exception:
            time.sleep(0.5)

    assert meta is not None, f"Failed to retrieve metadata for {task_name}"

    retrieved_data = client.get_data(meta)

    # Generate partial expected data based on retrieved indices
    full_expected = generate_consistency_data(meta.global_indexes)
    expected_data = full_expected.select(*data_fields)

    # Verify Hash
    retrieved_hash = compute_data_hash(retrieved_data)
    expected_hash = compute_data_hash(expected_data)
    assert retrieved_hash == expected_hash, f"Hash mismatch for {task_name}"

    return retrieved_data


@pytest.mark.timeout(60)
def test_consistency_core_types(e2e_client):
    """
    Test Case 1: Core Data Types Coverage
    - Tensor, NestedTensor, Non-Tensor (stackable/non-stackable)
    """
    client = e2e_client

    # Use distinct partition to avoid conflict if tests run shared (though scope is module)
    partition_id = "test_core_types"

    batch_size = 5

    # Get fields list from dummy data
    dummy_data = generate_consistency_data([0])
    fields = list(dummy_data.keys())

    # 1. Allocate Partition & Indices
    allocation_meta = client.get_meta(
        partition_id=partition_id, data_fields=fields, batch_size=batch_size, mode="insert", task_name="allocator"
    )
    indices = allocation_meta.global_indexes
    assert len(indices) == batch_size

    # 2. Generate Data using allocated indices
    data = generate_consistency_data(indices)

    put_data_with_indices(client, partition_id, indices, data)

    # 3. Get Data and Verify
    verify_data_consistency(client, partition_id, "test_worker", list(data.keys()), batch_size)


@pytest.mark.timeout(120)
def test_consistency_multi_round_put_get(e2e_client):
    """
    Test Case 2: Multi-round Put & Field Merge
    Simulate fragmented writing and field stitching.
    """
    client = e2e_client
    partition_id = "test_multi_round"

    # Define Indices
    idx_round1 = list(range(0, 20))
    idx_round2 = list(range(20, 41))  # 21 items

    # Step 1: Allocations
    # Define Field Groups
    # Group 1: Standard
    group_std_fields = ["tensor_field", "list_str_field", "list_int_field", "special_field", "bool_field"]
    # Group 2: Complex
    group_complex_fields = [
        "nested_field",
        "strided_nested_field",
        "list_numpy_field",
        "np_object_field",
        "non_orig_field",
    ]

    all_fields = group_std_fields + group_complex_fields

    # Allocation 1: Indices 0-19
    allocation_meta_1 = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=20, mode="insert", task_name="allocator_1"
    )
    assert len(allocation_meta_1.global_indexes) == 20

    # Allocation 2: Indices 20-40
    allocation_meta_2 = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=21, mode="insert", task_name="allocator_2"
    )
    assert len(allocation_meta_2.global_indexes) == 21

    # Full list for reference
    all_indices = idx_round1 + idx_round2

    # --- Write Operations with Cross-Batch Logic ---

    # Op 1: Put Standard Group for Batch 1 (0-19)
    data_1_std = get_fields_subset(idx_round1, group_std_fields)
    put_data_with_indices(client, partition_id, idx_round1, data_1_std)

    # Op 2: Put Standard Group for Batch 2 (20-40)
    data_2_std = get_fields_subset(idx_round2, group_std_fields)
    put_data_with_indices(client, partition_id, idx_round2, data_2_std)

    # Op 3: Cross-batch Put for Complex Group (indices 15-25)
    idx_cross = all_indices[15:26]  # 15 to 25 inclusive
    data_cross_complex = get_fields_subset(idx_cross, group_complex_fields)
    put_data_with_indices(client, partition_id, idx_cross, data_cross_complex)

    # Op 4: Fill remaining Complex Group (0-14)
    idx_remaining_1 = all_indices[0:15]
    data_rem_1_complex = get_fields_subset(idx_remaining_1, group_complex_fields)
    put_data_with_indices(client, partition_id, idx_remaining_1, data_rem_1_complex)

    # Op 5: Fill remaining Complex Group (26-40)
    idx_remaining_2 = all_indices[26:]
    data_rem_2_complex = get_fields_subset(idx_remaining_2, group_complex_fields)
    put_data_with_indices(client, partition_id, idx_remaining_2, data_rem_2_complex)

    # Verifications

    # 1. Get 0-10
    verify_data_consistency(client, partition_id, "verifier_1", all_fields, 10)

    # 2. Get 10-30 (Cross boundary)
    verify_data_consistency(client, partition_id, "verifier_2", all_fields, 20)

    # 3. Get 30-40
    verify_data_consistency(client, partition_id, "verifier_3", all_fields, 11)


@pytest.mark.timeout(60)
def test_consistency_slicing_and_subset(e2e_client):
    """
    Test Case 3: Slicing and Field Subsetting
    """
    client = e2e_client
    partition_id = "test_slicing"

    # Pre-allocate Partition
    all_fields = [
        "tensor_field",
        "nested_field",
        "strided_nested_field",
        "list_int_field",
        "list_str_field",
        "list_numpy_field",
        "np_object_field",
        "special_field",
        "bool_field",
        "non_orig_field",
    ]
    allocation_meta = client.get_meta(
        partition_id=partition_id, data_fields=all_fields, batch_size=20, mode="insert", task_name="allocator"
    )
    indices = allocation_meta.global_indexes
    assert len(indices) == 20

    # Create Data using these indices
    data = generate_consistency_data(indices)

    # Put Data
    put_data_with_indices(client, partition_id, indices, data)

    # 1. Field Subset: Get only nested_field (Ragged)
    result_subset = verify_data_consistency(client, partition_id, "inspector", ["nested_field"], 20, mode="force_fetch")

    assert "nested_field" in result_subset
    assert "tensor_field" not in result_subset
    assert len(result_subset["nested_field"]) == 20

    # 2. Get np_object_field & special_field
    result_mixed = verify_data_consistency(
        client, partition_id, "inspector_2", ["np_object_field", "special_field"], 20, mode="force_fetch"
    )

    assert "np_object_field" in result_mixed
    assert "special_field" in result_mixed

    # 3. Get strided_nested_field explicitly
    verify_data_consistency(client, partition_id, "inspector_3", ["strided_nested_field"], 20, mode="force_fetch")


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
