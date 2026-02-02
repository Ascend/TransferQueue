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
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta  # noqa: E402
from transfer_queue.storage.managers.base import KVStorageManager  # noqa: E402


def get_meta(data, global_indexes=None):
    if not global_indexes:
        global_indexes = range(data.batch_size[0])
    samples = []
    for sample_id in range(data.batch_size[0]):
        fields_dict = {}
        for field_name in data.keys():
            tensor = data[field_name][sample_id]
            field_meta = FieldMeta(
                name=field_name,
                dtype=tensor.dtype if isinstance(tensor, torch.Tensor) else None,
                shape=tensor.shape if isinstance(tensor, torch.Tensor) else None,
                production_status=1,
            )
            fields_dict[field_name] = field_meta
        sample = SampleMeta(
            partition_id=0,
            global_index=global_indexes[sample_id],
            fields=fields_dict,
        )
        samples.append(sample)
    metadata = BatchMeta(samples=samples)
    return metadata


@pytest.fixture
def test_data():
    """Fixture providing test configuration, data, and metadata."""
    cfg = {
        "controller_info": MagicMock(),
        "client_name": "YuanrongStorageClient",
        "host": "127.0.0.1",
        "port": 31501,
        "device_id": 0,
    }
    global_indexes = [8, 9, 10]

    data = TensorDict(
        {
            "text": torch.tensor([[1, 2], [3, 4], [5, 6]]),  # shape: [3, 2]
            "label": torch.tensor([0, 1, 2]),  # shape: [3]
            "mask": torch.tensor([[1], [1], [0]]),  # shape: [3, 1]
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
    expected = ["8@label", "9@label", "10@label", "8@mask", "9@mask", "10@mask", "8@text", "9@text", "10@text"]
    assert keys == expected
    assert len(keys) == 9  # 3 fields * 3 indexes


def test_generate_values(test_data):
    """
    Test whether _generate_values can flatten the TensorDict into an ordered list of tensors,
    using field_name as the primary key and global_index as the secondary key.
    """
    values = KVStorageManager._generate_values(test_data["data"])
    expected_length = len(test_data["field_names"]) * len(test_data["global_indexes"])  # 9
    expected_values = [0, 1, 2, [1], [1], [0], [1, 2], [3, 4], [5, 6]]

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

    manager = KVStorageManager(test_data["cfg"])
    assert manager.storage_client is mock_client
    assert manager._multi_threads_executor is None

    # Generate values
    values = manager._generate_values(test_data["data"])

    # Reconstruct TensorDict
    reconstructed = manager._merge_tensors_to_tensordict(test_data["metadata"], values)

    # Check presence of keys
    assert "text" in reconstructed
    assert "label" in reconstructed
    assert "mask" in reconstructed

    # Check tensor equality
    assert torch.equal(reconstructed["text"], test_data["data"]["text"])
    assert torch.equal(reconstructed["label"], test_data["data"]["label"])
    assert torch.equal(reconstructed["mask"], test_data["data"]["mask"])

    # Check batch size
    assert reconstructed.batch_size == torch.Size([3])

    # verify nested tensors and non tensors
    complex_data = TensorDict(
        {
            "text": torch.nested.nested_tensor([[1, 2], [3], [4]]),
            "prompt": ["5", "6", "7"],
            "extra": [torch.Tensor([8]), "9", torch.Tensor([10])],
        },
        batch_size=[3],
    )

    complex_meta = get_meta(complex_data)
    complex_values = manager._generate_values(complex_data)
    complex_tensordict = manager._merge_tensors_to_tensordict(complex_meta, complex_values)
    assert "text" in complex_tensordict
    assert "prompt" in complex_tensordict
    for key in complex_tensordict.keys():
        if isinstance(complex_tensordict[key], torch.Tensor):
            for t1, t2 in zip(complex_tensordict[key], complex_data[key], strict=False):
                assert torch.equal(t1, t2)
        else:
            assert complex_tensordict[key] == complex_data[key]


def test_get_shape_type_custom_backend_meta_list_without_custom_meta(test_data):
    """Test _get_shape_type_custom_backend_meta_list returns correct shapes and dtypes without custom_meta."""
    shapes, dtypes, custom_backend_meta_list = KVStorageManager._get_shape_type_custom_backend_meta_list(
        test_data["metadata"]
    )

    # Expected order: sorted by field name (label, mask, text), then by global_index order
    # 3 fields * 3 samples = 9 entries
    # Check shapes - order is label, mask, text (sorted alphabetically)
    # label shapes: [()]*3, mask shapes: [(1,)]*3, text shapes: [(2,)]*3
    expected_shapes = [
        torch.Size([]),  # label[0]
        torch.Size([]),  # label[1]
        torch.Size([]),  # label[2]
        torch.Size([1]),  # mask[0]
        torch.Size([1]),  # mask[1]
        torch.Size([1]),  # mask[2]
        torch.Size([2]),  # text[0]
        torch.Size([2]),  # text[1]
        torch.Size([2]),  # text[2]
    ]
    expected_dtypes = [torch.int64] * (len(test_data["field_names"]) * len(test_data["global_indexes"]))
    # No custom_meta provided, so all should be None
    expected_custom_backend_meta = [None] * (len(test_data["field_names"]) * len(test_data["global_indexes"]))

    assert shapes == expected_shapes
    assert dtypes == expected_dtypes
    assert custom_backend_meta_list == expected_custom_backend_meta


def test_get_shape_type_custom_backend_meta_list_with_custom_meta(test_data):
    """Test _get_shape_type_custom_meta_list returns correct custom_meta when provided."""
    # Add custom_meta to metadata
    custom_backend_meta = {
        8: {"text": {"key1": "value1"}, "label": {"key2": "value2"}, "mask": {"key3": "value3"}},
        9: {"text": {"key4": "value4"}, "label": {"key5": "value5"}, "mask": {"key6": "value6"}},
        10: {"text": {"key7": "value7"}, "label": {"key8": "value8"}, "mask": {"key9": "value9"}},
    }
    metadata = test_data["metadata"]
    metadata._custom_backend_meta.update(custom_backend_meta)

    shapes, dtypes, custom_backend_meta_list = KVStorageManager._get_shape_type_custom_backend_meta_list(metadata)

    # Check custom_meta - order is label, mask, text (sorted alphabetically) by global_index
    expected_custom_meta = [
        {"key2": "value2"},  # label, global_index=8
        {"key5": "value5"},  # label, global_index=9
        {"key8": "value8"},  # label, global_index=10
        {"key3": "value3"},  # mask, global_index=8
        {"key6": "value6"},  # mask, global_index=9
        {"key9": "value9"},  # mask, global_index=10
        {"key1": "value1"},  # text, global_index=8
        {"key4": "value4"},  # text, global_index=9
        {"key7": "value7"},  # text, global_index=10
    ]
    assert custom_backend_meta_list == expected_custom_meta


def test_get_shape_type_custom_backend_meta_list_with_partial_custom_meta(test_data):
    """Test _get_shape_type_custom_backend_meta_list handles partial custom_meta correctly."""
    # Add custom_meta only for some global_indexes and fields
    custom_backend_meta = {
        8: {"text": {"key1": "value1"}},  # Only text field
        # global_index 9 has no custom_meta
        10: {"label": {"key2": "value2"}, "mask": {"key3": "value3"}},  # label and mask only
    }
    metadata = test_data["metadata"]
    metadata._custom_backend_meta.update(custom_backend_meta)

    shapes, dtypes, custom_backend_meta_list = KVStorageManager._get_shape_type_custom_backend_meta_list(metadata)

    # Check custom_meta - order is label, mask, text (sorted alphabetically) by global_index
    expected_custom_backend_meta = [
        None,  # label, global_index=8 (not in custom_meta)
        None,  # label, global_index=9 (not in custom_meta)
        {"key2": "value2"},  # label, global_index=10
        None,  # mask, global_index=8 (not in custom_meta)
        None,  # mask, global_index=9 (not in custom_meta)
        {"key3": "value3"},  # mask, global_index=10
        {"key1": "value1"},  # text, global_index=8
        None,  # text, global_index=9 (not in custom_meta)
        None,  # text, global_index=10 (not in custom_meta for text)
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
def test_put_data_with_custom_meta_from_storage_client(mock_notify, test_data_for_put_data):
    """Test that put_data correctly processes custom_meta returned by storage client."""
    # Create a mock storage client
    mock_storage_client = MagicMock()
    # Simulate storage client returning custom_meta (one per key)
    # Keys order: label[0,1,2], text[0,1,2] (sorted by field name)
    mock_custom_meta = [
        {"storage_key": "0@label"},
        {"storage_key": "1@label"},
        {"storage_key": "2@label"},
        {"storage_key": "0@text"},
        {"storage_key": "1@text"},
        {"storage_key": "2@text"},
    ]
    mock_storage_client.put.return_value = mock_custom_meta

    # Create manager with mocked dependencies
    config = {"controller_info": MagicMock(), "client_name": "MockClient"}
    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=mock_storage_client):
        manager = KVStorageManager(config)

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

    # Verify notify_data_update was called with correct custom_meta structure
    mock_notify.assert_called_once()
    notify_call_args = mock_notify.call_args
    per_field_custom_meta = notify_call_args[0][5]  # 6th positional argument

    # Verify custom_meta is structured correctly: {global_index: {field: meta}}
    assert 0 in per_field_custom_meta
    assert 1 in per_field_custom_meta
    assert 2 in per_field_custom_meta

    assert per_field_custom_meta[0]["label"] == {"storage_key": "0@label"}
    assert per_field_custom_meta[0]["text"] == {"storage_key": "0@text"}
    assert per_field_custom_meta[1]["label"] == {"storage_key": "1@label"}
    assert per_field_custom_meta[1]["text"] == {"storage_key": "1@text"}
    assert per_field_custom_meta[2]["label"] == {"storage_key": "2@label"}
    assert per_field_custom_meta[2]["text"] == {"storage_key": "2@text"}

    # Verify metadata was updated with custom_meta
    all_custom_meta = test_data_for_put_data["metadata"].get_all_custom_meta()
    assert all_custom_meta[0]["label"] == {"storage_key": "0@label"}
    assert all_custom_meta[2]["text"] == {"storage_key": "2@text"}


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
@patch.object(KVStorageManager, "notify_data_update", new_callable=AsyncMock)
def test_put_data_without_custom_meta(mock_notify, test_data_for_put_data):
    """Test that put_data works correctly when storage client returns no custom_meta."""
    # Create a mock storage client that returns None for custom_meta
    mock_storage_client = MagicMock()
    mock_storage_client.put.return_value = None

    # Create manager with mocked dependencies
    config = {"controller_info": MagicMock(), "client_name": "MockClient"}
    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=mock_storage_client):
        manager = KVStorageManager(config)

    # Run put_data
    asyncio.run(manager.put_data(test_data_for_put_data["data"], test_data_for_put_data["metadata"]))

    # Verify notify_data_update was called with empty dict for custom_meta
    mock_notify.assert_called_once()
    notify_call_args = mock_notify.call_args
    per_field_custom_meta = notify_call_args[0][5]  # 6th positional argument
    assert per_field_custom_meta == {}


@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
def test_put_data_custom_meta_length_mismatch_raises_error(test_data_for_put_data):
    """Test that put_data raises ValueError when custom_meta length doesn't match keys."""
    # Create a mock storage client that returns mismatched custom_meta length
    mock_storage_client = MagicMock()
    # Return only 3 custom_meta entries when 6 are expected
    mock_storage_client.put.return_value = [{"key": "1"}, {"key": "2"}, {"key": "3"}]

    # Create manager with mocked dependencies
    config = {"controller_info": MagicMock(), "client_name": "MockClient"}
    with patch(f"{STORAGE_CLIENT_FACTORY_PATH}.create", return_value=mock_storage_client):
        manager = KVStorageManager(config)

    # Run put_data and expect ValueError
    with pytest.raises(ValueError) as exc_info:
        asyncio.run(manager.put_data(test_data_for_put_data["data"], test_data_for_put_data["metadata"]))

    assert "does not match" in str(exc_info.value)
