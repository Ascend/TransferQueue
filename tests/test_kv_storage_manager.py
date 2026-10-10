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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers.base import KVStorageManager


def test_close_is_quiet_when_config_is_rejected_before_super_init():
    """A config rejected before super().__init__() leaves nothing for close() to release.

    Every KV manager validates its config first, so none of the base attributes close()
    reaches for exist yet. __del__ calls close() regardless, and an AttributeError there
    buries the real ValueError under a spurious teardown error.
    """
    built = []

    class Probe(KVStorageManager):
        def __init__(self, *args, **kwargs):
            built.append(self)
            super().__init__(*args, **kwargs)

    with pytest.raises(ValueError, match="Missing client_name"):
        Probe(None, {})

    manager = built[0]
    assert not hasattr(manager, "notify_pool"), "the test no longer exercises the pre-super path"
    manager.close()


def get_meta(data, global_indexes=None):
    if not global_indexes:
        global_indexes = list(range(data.batch_size[0]))

    # Build columnar field_schema from the data
    field_schema = {}
    for field_name in data.keys():
        field_data = data[field_name]
        if isinstance(field_data, torch.Tensor) and field_data.is_nested:
            per_sample_shapes = [t.shape for t in field_data.unbind()]
            field_schema[field_name] = {
                "dtype": field_data.dtype,
                "shape": per_sample_shapes[0],
                "per_sample_shapes": per_sample_shapes,
                "is_nested": True,
                "is_non_tensor": False,
            }
        else:
            tensor = field_data[0]
            field_schema[field_name] = {
                "dtype": tensor.dtype if isinstance(tensor, torch.Tensor) else type(tensor),
                "shape": tensor.shape if isinstance(tensor, torch.Tensor) else None,
                "is_nested": False,
                "is_non_tensor": not isinstance(tensor, torch.Tensor),
            }

    import numpy as np

    production_status = np.ones(len(global_indexes), dtype=np.int8)

    metadata = BatchMeta(
        global_indexes=list(global_indexes),
        partition_ids=["0"] * len(global_indexes),
        field_schema=field_schema,
        production_status=production_status,
    )
    return metadata


@pytest.fixture
def test_data():
    """Fixture providing test configuration, data, and metadata."""
    cfg = {
        "controller_info": MagicMock(),
        "client_name": "YuanrongStorageClient",
        "worker_port": 31501,
        "device_id": 0,
    }
    global_indexes = [8, 9, 10]

    data = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(
                [
                    torch.tensor([1, 2, 3, 4, 5]),
                    torch.tensor([6, 7, 8, 9]),
                    torch.tensor([10, 11]),
                ],
                layout=torch.jagged,
            ),
            "prompt_ids": torch.nested.as_nested_tensor(
                [
                    torch.tensor([1, 2]),
                    torch.tensor([6, 7, 8]),
                    torch.tensor([10]),
                ],
                layout=torch.jagged,
            ),
            "response_ids": torch.nested.as_nested_tensor(
                [
                    torch.tensor([3, 4, 5]),
                    torch.tensor([9]),
                    torch.tensor([11]),
                ],
                layout=torch.jagged,
            ),
            "response_mask": torch.nested.as_nested_tensor(
                [
                    torch.tensor([0, 0, 1, 1, 1]),
                    torch.tensor([0, 0, 0, 1]),
                    torch.tensor([0, 1]),
                ],
                layout=torch.jagged,
            ),
        },
        batch_size=3,
    )
    metadata = get_meta(data, global_indexes)

    return {
        "cfg": cfg,
        "field_names": data.keys(),
        "global_indexes": global_indexes,
        "data": data,
        "metadata": metadata,
    }


def test_generate_keys(test_data):
    """Test whether _generate_keys can generate the correct key list."""
    keys = KVStorageManager._generate_keys(test_data["data"].keys(), test_data["metadata"].global_indexes)
    expected = [
        "8@input_ids",
        "9@input_ids",
        "10@input_ids",
        "8@prompt_ids",
        "9@prompt_ids",
        "10@prompt_ids",
        "8@response_ids",
        "9@response_ids",
        "10@response_ids",
        "8@response_mask",
        "9@response_mask",
        "10@response_mask",
    ]
    assert keys == expected
    assert len(keys) == 12  # 4 fields * 3 indexes


def test_generate_values(test_data):
    """
    Test whether _generate_values can flatten the TensorDict into an ordered list of tensors,
    using field_name as the primary key and global_index as the secondary key.
    """
    values = KVStorageManager._generate_values(test_data["data"])
    expected_length = len(test_data["field_names"]) * len(test_data["global_indexes"])  # 12
    expected_values = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9],
        [10, 11],  # input_ids
        [1, 2],
        [6, 7, 8],
        [10],  # prompt_ids
        [3, 4, 5],
        [9],
        [11],  # response_ids
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 1],  # response_mask
    ]

    expected_values = [torch.tensor(value) for value in expected_values]

    assert len(values) == expected_length

    for i in range(len(values)):
        assert torch.equal(values[i], expected_values[i])


@patch("transfer_queue.storage.managers.base.StorageClientFactory.create")
@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
def test_merge_tensors_to_tensordict(mock_create, test_data):
    """Test whether _merge_kv_to_tensordict can correctly reconstruct the TensorDict."""
    mock_client = MagicMock()
    mock_create.return_value = mock_client

    manager = KVStorageManager(controller_info=MagicMock(), config=test_data["cfg"])
    assert manager.storage_client is mock_client
    assert manager._multi_threads_executor is None

    # Generate values
    values = manager._generate_values(test_data["data"])

    # Reconstruct TensorDict
    reconstructed = manager._merge_tensors_to_tensordict(test_data["metadata"], values)

    # Check presence of keys
    assert "input_ids" in reconstructed
    assert "prompt_ids" in reconstructed
    assert "response_ids" in reconstructed
    assert "response_mask" in reconstructed

    # Check tensor equality (nested tensor vs nested tensor)
    for key in ["input_ids", "prompt_ids", "response_ids", "response_mask"]:
        unbound_a = reconstructed[key].unbind(0)
        unbound_b = test_data["data"][key].unbind(0)
        assert len(unbound_a) == len(unbound_b), f"Length mismatch for {key}: {len(unbound_a)} vs {len(unbound_b)}"
        for t1, t2 in zip(unbound_a, unbound_b, strict=True):
            assert torch.equal(t1, t2)

    # Check batch size
    assert reconstructed.batch_size == torch.Size([3])

    # verify nested tensors and non tensors
    complex_data = TensorDict(
        {
            "input_ids": torch.nested.nested_tensor([[1, 2], [3], [4]]),
            "prompt": ["5", "6", "7"],
            "extra": [torch.Tensor([8]), "9", torch.Tensor([10])],
        },
        batch_size=[3],
    )

    complex_meta = get_meta(complex_data)
    complex_values = manager._generate_values(complex_data)
    complex_tensordict = manager._merge_tensors_to_tensordict(complex_meta, complex_values)
    assert "input_ids" in complex_tensordict
    assert "prompt" in complex_tensordict
    for key in complex_tensordict.keys():
        if isinstance(complex_tensordict[key], torch.Tensor):
            unbound_a = complex_tensordict[key].unbind(0)
            unbound_b = complex_data[key].unbind(0)
            assert len(unbound_a) == len(unbound_b), f"Length mismatch for {key}: {len(unbound_a)} vs {len(unbound_b)}"
            for t1, t2 in zip(unbound_a, unbound_b, strict=True):
                assert torch.equal(t1, t2)
        else:
            assert complex_tensordict[key] == complex_data[key]


def test_get_shape_type_custom_backend_meta_list_without_custom_backend_meta(test_data):
    """Test _get_shape_type_custom_backend_meta_list returns correct shapes and dtypes without custom_backend_meta."""
    shapes, dtypes, custom_backend_meta_list = KVStorageManager._get_shape_type_custom_backend_meta_list(
        test_data["metadata"]
    )

    # Expected order: sorted by field name (label, mask, text), then by global_index order
    # 3 fields * 3 samples = 9 entries
    # Check shapes - order is input_ids, prompt_ids, response_ids, response_mask (sorted alphabetically)
    # input_ids shapes: [5, 4, 2], prompt_ids shapes: [2, 3, 1],
    # response_ids shapes: [3, 1, 1], response_mask shapes: [5, 4, 2]
    expected_shapes = [
        torch.Size([5]),  # input_ids[0]
        torch.Size([4]),  # input_ids[1]
        torch.Size([2]),  # input_ids[2]
        torch.Size([2]),  # prompt_ids[0]
        torch.Size([3]),  # prompt_ids[1]
        torch.Size([1]),  # prompt_ids[2]
        torch.Size([3]),  # response_ids[0]
        torch.Size([1]),  # response_ids[1]
        torch.Size([1]),  # response_ids[2]
        torch.Size([5]),  # response_mask[0]
        torch.Size([4]),  # response_mask[1]
        torch.Size([2]),  # response_mask[2]
    ]
    expected_dtypes = [torch.int64] * (len(test_data["field_names"]) * len(test_data["global_indexes"]))
    # No custom_backend_meta provided, so all should be None
    expected_custom_backend_meta = [None] * (len(test_data["field_names"]) * len(test_data["global_indexes"]))

    assert shapes == expected_shapes
    assert dtypes == expected_dtypes
    assert custom_backend_meta_list == expected_custom_backend_meta


def test_get_shape_type_custom_backend_meta_list_with_custom_backend_meta(test_data):
    """Test _get_shape_type_custom_backend_meta_list returns correct custom_backend_meta when provided."""
    # Add custom_backend_meta to metadata (columnar: list aligned with global_indexes [8, 9, 10])
    metadata = test_data["metadata"]
    metadata._custom_backend_meta = [
        {
            "input_ids": {"key1": "value1"},
            "prompt_ids": {"key2": "value2"},
            "response_ids": {"key3": "value3"},
            "response_mask": {"key4": "value4"},
        },  # global_index=8
        {
            "input_ids": {"key5": "value5"},
            "prompt_ids": {"key6": "value6"},
            "response_ids": {"key7": "value7"},
            "response_mask": {"key8": "value8"},
        },  # global_index=9
        {
            "input_ids": {"key9": "value9"},
            "prompt_ids": {"key10": "value10"},
            "response_ids": {"key11": "value11"},
            "response_mask": {"key12": "value12"},
        },  # global_index=10
    ]

    shapes, dtypes, custom_backend_meta_list = KVStorageManager._get_shape_type_custom_backend_meta_list(metadata)

    # Check custom_backend_meta - order is input_ids, prompt_ids, response_ids,
    # response_mask (sorted alphabetically) by global_index
    expected_custom_backend_meta = [
        {"key1": "value1"},  # input_ids, global_index=8
        {"key5": "value5"},  # input_ids, global_index=9
        {"key9": "value9"},  # input_ids, global_index=10
        {"key2": "value2"},  # prompt_ids, global_index=8
        {"key6": "value6"},  # prompt_ids, global_index=9
        {"key10": "value10"},  # prompt_ids, global_index=10
        {"key3": "value3"},  # response_ids, global_index=8
        {"key7": "value7"},  # response_ids, global_index=9
        {"key11": "value11"},  # response_ids, global_index=10
        {"key4": "value4"},  # response_mask, global_index=8
        {"key8": "value8"},  # response_mask, global_index=9
        {"key12": "value12"},  # response_mask, global_index=10
    ]
    assert custom_backend_meta_list == expected_custom_backend_meta


def test_get_shape_type_custom_backend_meta_list_with_partial_custom_backend_meta(test_data):
    """Test _get_shape_type_custom_backend_meta_list handles partial custom_backend_meta correctly."""
    # Add custom_backend_meta only for some fields (columnar: list aligned with global_indexes [8, 9, 10])
    metadata = test_data["metadata"]
    metadata._custom_backend_meta = [
        {"input_ids": {"key1": "value1"}},  # global_index=8: only input_ids field
        {},  # global_index=9: no custom_backend_meta
        {
            "prompt_ids": {"key2": "value2"},
            "response_ids": {"key3": "value3"},
        },  # global_index=10: prompt_ids and response_ids only
    ]

    shapes, dtypes, custom_backend_meta_list = KVStorageManager._get_shape_type_custom_backend_meta_list(metadata)

    # Check custom_backend_meta - order is input_ids, prompt_ids, response_ids,
    # response_mask (sorted alphabetically) by global_index
    expected_custom_backend_meta = [
        {"key1": "value1"},  # input_ids, global_index=8
        None,  # input_ids, global_index=9 (not in custom_backend_meta)
        None,  # input_ids, global_index=10 (not in custom_backend_meta)
        None,  # prompt_ids, global_index=8 (not in custom_backend_meta)
        None,  # prompt_ids, global_index=9 (not in custom_backend_meta)
        {"key2": "value2"},  # prompt_ids, global_index=10
        None,  # response_ids, global_index=8 (not in custom_backend_meta)
        None,  # response_ids, global_index=9 (not in custom_backend_meta)
        {"key3": "value3"},  # response_ids, global_index=10
        None,  # response_mask, global_index=8 (not in custom_backend_meta)
        None,  # response_mask, global_index=9 (not in custom_backend_meta)
        None,  # response_mask, global_index=10 (not in custom_backend_meta)
    ]
    assert custom_backend_meta_list == expected_custom_backend_meta


@pytest.fixture
def test_data_for_put_data():
    """Provide test fixtures for put_data tests."""
    field_names = ["text", "label"]
    global_indexes = [0, 1, 2]

    # Create test data
    data = TensorDict(
        {
            "text": torch.tensor([[1, 2], [3, 4], [5, 6]]),
            "label": torch.tensor([0, 1, 2]),
        },
        batch_size=3,
    )

    metadata = get_meta(data, global_indexes)

    return {
        "field_names": field_names,
        "global_indexes": global_indexes,
        "data": data,
        "metadata": metadata,
    }


STORAGE_CLIENT_FACTORY_PATH = "transfer_queue.storage.managers.base.StorageClientFactory"


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
@patch.object(KVStorageManager, "notify_data_update", new_callable=AsyncMock)
def test_put_data_with_custom_backend_meta_from_storage_client(mock_notify, test_data_for_put_data):
    """Test that put_data correctly processes custom_backend_meta returned by storage client."""
    # Create a mock storage client
    mock_storage_client = MagicMock()
    # Simulate storage client returning custom_backend_meta (one per key)
    # Keys order: label[0,1,2], text[0,1,2] (sorted by field name)
    mock_custom_backend_meta = [
        {"storage_key": "0@label"},
        {"storage_key": "1@label"},
        {"storage_key": "2@label"},
        {"storage_key": "0@text"},
        {"storage_key": "1@text"},
        {"storage_key": "2@text"},
    ]
    mock_storage_client.put.return_value = mock_custom_backend_meta

    # Create manager with mocked dependencies
    config = {"client_name": "MockClient"}
    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=mock_storage_client):
        manager = KVStorageManager(controller_info=MagicMock(), config=config)

    # Run put_data
    asyncio.run(manager.put_data(test_data_for_put_data["data"], test_data_for_put_data["metadata"]))

    # Verify storage client was called with correct keys and values
    mock_storage_client.put.assert_called_once()
    call_args = mock_storage_client.put.call_args
    keys = call_args[0][0]
    values = call_args[0][1]

    # Verify keys are correct
    expected_keys = ["0@label", "1@label", "2@label", "0@text", "1@text", "2@text"]
    assert keys == expected_keys
    assert len(values) == 6

    # Verify notify_data_update was called with correct custom_backend_meta structure
    mock_notify.assert_called_once()
    notify_call_args = mock_notify.call_args
    per_field_custom_backend_meta = notify_call_args[0][3]  # 4th positional argument (custom_backend_meta)

    # Verify custom_backend_meta is structured correctly: {global_index: {field: meta}}
    assert 0 in per_field_custom_backend_meta
    assert 1 in per_field_custom_backend_meta
    assert 2 in per_field_custom_backend_meta

    assert per_field_custom_backend_meta[0]["label"] == {"storage_key": "0@label"}
    assert per_field_custom_backend_meta[0]["text"] == {"storage_key": "0@text"}
    assert per_field_custom_backend_meta[1]["label"] == {"storage_key": "1@label"}
    assert per_field_custom_backend_meta[1]["text"] == {"storage_key": "1@text"}
    assert per_field_custom_backend_meta[2]["label"] == {"storage_key": "2@label"}
    assert per_field_custom_backend_meta[2]["text"] == {"storage_key": "2@text"}

    # Verify metadata was updated with custom_backend_meta
    all_custom_backend_meta = test_data_for_put_data["metadata"]._custom_backend_meta
    assert len(all_custom_backend_meta) == 3
    assert all_custom_backend_meta[0]["label"] == {"storage_key": "0@label"}
    assert all_custom_backend_meta[2]["text"] == {"storage_key": "2@text"}


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
@patch.object(KVStorageManager, "notify_data_update", new_callable=AsyncMock)
def test_put_data_without_custom_backend_meta(mock_notify, test_data_for_put_data):
    """Test that put_data works correctly when storage client returns no custom_backend_meta."""
    # Create a mock storage client that returns None for custom_backend_meta
    mock_storage_client = MagicMock()
    mock_storage_client.put.return_value = None

    # Create manager with mocked dependencies
    config = {"controller_info": MagicMock(), "client_name": "MockClient"}
    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=mock_storage_client):
        manager = KVStorageManager(controller_info=MagicMock(), config=config)

    # Run put_data
    asyncio.run(manager.put_data(test_data_for_put_data["data"], test_data_for_put_data["metadata"]))

    # Verify notify_data_update was called with empty dict for custom_backend_meta
    mock_notify.assert_called_once()
    notify_call_args = mock_notify.call_args
    per_field_custom_backend_meta = notify_call_args[0][3]  # 4th positional argument (custom_backend_meta)
    assert per_field_custom_backend_meta == {}


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
def test_put_data_custom_backend_meta_length_mismatch_raises_error(test_data_for_put_data):
    """Test that put_data raises ValueError when custom_backend_meta length doesn't match keys."""
    # Create a mock storage client that returns mismatched custom_backend_meta length
    mock_storage_client = MagicMock()
    # Return only 3 custom_backend_meta entries when 6 are expected
    mock_storage_client.put.return_value = [{"key": "1"}, {"key": "2"}, {"key": "3"}]

    # Create manager with mocked dependencies
    config = {"controller_info": MagicMock(), "client_name": "MockClient"}
    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=mock_storage_client):
        manager = KVStorageManager(controller_info=MagicMock(), config=config)

    # Run put_data and expect ValueError
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(manager.put_data(test_data_for_put_data["data"], test_data_for_put_data["metadata"]))

    assert "does not match" in str(exc_info.value)
