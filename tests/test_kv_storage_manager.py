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
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from tensordict import TensorDict

# Setup path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta  # noqa: E402
from transfer_queue.storage.managers.base import KVStorageManager  # noqa: E402


class Test(unittest.TestCase):
    def setUp(self):
        self.cfg = {"client_name": "Yuanrong", "host": "127.0.0.1", "port": 31501, "device_id": 0}
        # metadata
        self.field_names = ["text", "label", "mask"]
        self.global_indexes = [8, 9, 10]

        # data: TensorDict
        self.data = TensorDict(
            {
                "text": torch.tensor([[1, 2], [3, 4], [5, 6]]),  # shape: [3, 2]
                "label": torch.tensor([0, 1, 2]),  # shape: [3]
                "mask": torch.tensor([[1], [1], [0]]),  # shape: [3, 1]
            },
            batch_size=3,
        )
        samples = []

        for sample_id in range(self.data.batch_size[0]):
            fields_dict = {}
            for field_name in self.data.keys():
                tensor = self.data[field_name][sample_id]
                field_meta = FieldMeta(name=field_name, dtype=tensor.dtype, shape=tensor.shape, production_status=1)
                fields_dict[field_name] = field_meta
            sample = SampleMeta(
                partition_id=0,
                global_index=self.global_indexes[sample_id],
                fields=fields_dict,
            )
            samples.append(sample)
        self.metadata = BatchMeta(samples=samples)

    # def test_create(self):
    #     self.sm = YuanrongStorageManager(self.cfg)

    def test_generate_keys(self):
        """Test whether _generate_keys can generate the correct key list."""
        keys = KVStorageManager._generate_keys(self.data.keys(), self.metadata.global_indexes)
        expected = ["8@label", "9@label", "10@label", "8@mask", "9@mask", "10@mask", "8@text", "9@text", "10@text"]
        self.assertEqual(keys, expected)
        self.assertEqual(len(keys), 9)  # 3 fields * 3 indexes

    def test_generate_values(self):
        """
        Test whether _generate_values can flatten the TensorDict into an ordered list of tensors,
        using field_name as the primary key and global_index as the secondary key.
        """
        values = KVStorageManager._generate_values(self.data)
        expected_length = len(self.field_names) * len(self.global_indexes)  # 9
        self.assertEqual(len(values), expected_length)

    def test_merge_kv_to_tensordict(self):
        """Test whether _merge_kv_to_tensordict can correctly reconstruct the TensorDict."""
        # generate values firstly
        values = KVStorageManager._generate_values(self.data)

        # merge values to TensorDict
        reconstructed = KVStorageManager._merge_tensors_to_tensordict(self.metadata, values)

        self.assertIn("text", reconstructed)
        self.assertIn("label", reconstructed)
        self.assertIn("mask", reconstructed)

        self.assertTrue(torch.equal(reconstructed["text"], self.data["text"]))
        self.assertTrue(torch.equal(reconstructed["label"], self.data["label"]))
        self.assertTrue(torch.equal(reconstructed["mask"], self.data["mask"]))

        self.assertEqual(reconstructed.batch_size, torch.Size([3]))

    def test_get_shape_type_custom_meta_list_without_custom_meta(self):
        """Test _get_shape_type_custom_meta_list returns correct shapes and dtypes without custom_meta."""
        shapes, dtypes, custom_meta_list = KVStorageManager._get_shape_type_custom_meta_list(self.metadata)

        # Expected order: sorted by field name (label, mask, text), then by global_index order
        # 3 fields * 3 samples = 9 entries
        self.assertEqual(len(shapes), 9)
        self.assertEqual(len(dtypes), 9)
        self.assertEqual(len(custom_meta_list), 9)

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
        self.assertEqual(shapes, expected_shapes)

        # All dtypes should be torch.int64
        for dtype in dtypes:
            self.assertEqual(dtype, torch.int64)

        # No custom_meta provided, so all should be None
        for meta in custom_meta_list:
            self.assertIsNone(meta)

    def test_get_shape_type_custom_meta_list_with_custom_meta(self):
        """Test _get_shape_type_custom_meta_list returns correct custom_meta when provided."""
        # Add custom_meta to metadata
        custom_meta = {
            8: {"text": {"key1": "value1"}, "label": {"key2": "value2"}, "mask": {"key3": "value3"}},
            9: {"text": {"key4": "value4"}, "label": {"key5": "value5"}, "mask": {"key6": "value6"}},
            10: {"text": {"key7": "value7"}, "label": {"key8": "value8"}, "mask": {"key9": "value9"}},
        }
        self.metadata.update_custom_meta(custom_meta)

        shapes, dtypes, custom_meta_list = KVStorageManager._get_shape_type_custom_meta_list(self.metadata)

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
        self.assertEqual(custom_meta_list, expected_custom_meta)

    def test_get_shape_type_custom_meta_list_with_partial_custom_meta(self):
        """Test _get_shape_type_custom_meta_list handles partial custom_meta correctly."""
        # Add custom_meta only for some global_indexes and fields
        custom_meta = {
            8: {"text": {"key1": "value1"}},  # Only text field
            # global_index 9 has no custom_meta
            10: {"label": {"key2": "value2"}, "mask": {"key3": "value3"}},  # label and mask only
        }
        self.metadata.update_custom_meta(custom_meta)

        shapes, dtypes, custom_meta_list = KVStorageManager._get_shape_type_custom_meta_list(self.metadata)

        # Check custom_meta - order is label, mask, text (sorted alphabetically) by global_index
        expected_custom_meta = [
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
        self.assertEqual(custom_meta_list, expected_custom_meta)


class TestPutDataWithCustomMeta(unittest.TestCase):
    """Test put_data with custom_meta functionality."""

    def setUp(self):
        """Set up test fixtures for put_data tests."""
        self.field_names = ["text", "label"]
        self.global_indexes = [0, 1, 2]

        # Create test data
        self.data = TensorDict(
            {
                "text": torch.tensor([[1, 2], [3, 4], [5, 6]]),
                "label": torch.tensor([0, 1, 2]),
            },
            batch_size=3,
        )

        # Create metadata without production status set (for insert mode)
        samples = []
        for sample_id in range(self.data.batch_size[0]):
            fields_dict = {}
            for field_name in self.data.keys():
                tensor = self.data[field_name][sample_id]
                field_meta = FieldMeta(name=field_name, dtype=tensor.dtype, shape=tensor.shape, production_status=0)
                fields_dict[field_name] = field_meta
            sample = SampleMeta(
                partition_id="test_partition",
                global_index=self.global_indexes[sample_id],
                fields=fields_dict,
            )
            samples.append(sample)
        self.metadata = BatchMeta(samples=samples)

    @patch.object(KVStorageManager, "_connect_to_controller")
    @patch.object(KVStorageManager, "notify_data_update", new_callable=AsyncMock)
    def test_put_data_with_custom_meta_from_storage_client(self, mock_notify, mock_connect):
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
        config = {"client_name": "MockClient"}
        with patch(
            "transfer_queue.storage.managers.base.StorageClientFactory.create", return_value=mock_storage_client
        ):
            manager = KVStorageManager(config)

        # Run put_data
        asyncio.run(manager.put_data(self.data, self.metadata))

        # Verify storage client was called with correct keys and values
        mock_storage_client.put.assert_called_once()
        call_args = mock_storage_client.put.call_args
        keys = call_args[0][0]
        values = call_args[0][1]

        # Verify keys are correct
        expected_keys = ["0@label", "1@label", "2@label", "0@text", "1@text", "2@text"]
        self.assertEqual(keys, expected_keys)
        self.assertEqual(len(values), 6)

        # Verify notify_data_update was called with correct custom_meta structure
        mock_notify.assert_called_once()
        notify_call_args = mock_notify.call_args
        per_field_custom_meta = notify_call_args[0][5]  # 6th positional argument

        # Verify custom_meta is structured correctly: {global_index: {field: meta}}
        self.assertIn(0, per_field_custom_meta)
        self.assertIn(1, per_field_custom_meta)
        self.assertIn(2, per_field_custom_meta)

        self.assertEqual(per_field_custom_meta[0]["label"], {"storage_key": "0@label"})
        self.assertEqual(per_field_custom_meta[0]["text"], {"storage_key": "0@text"})
        self.assertEqual(per_field_custom_meta[1]["label"], {"storage_key": "1@label"})
        self.assertEqual(per_field_custom_meta[1]["text"], {"storage_key": "1@text"})
        self.assertEqual(per_field_custom_meta[2]["label"], {"storage_key": "2@label"})
        self.assertEqual(per_field_custom_meta[2]["text"], {"storage_key": "2@text"})

        # Verify metadata was updated with custom_meta
        all_custom_meta = self.metadata.get_all_custom_meta()
        self.assertEqual(all_custom_meta[0]["label"], {"storage_key": "0@label"})
        self.assertEqual(all_custom_meta[2]["text"], {"storage_key": "2@text"})

    @patch.object(KVStorageManager, "_connect_to_controller")
    @patch.object(KVStorageManager, "notify_data_update", new_callable=AsyncMock)
    def test_put_data_without_custom_meta(self, mock_notify, mock_connect):
        """Test that put_data works correctly when storage client returns no custom_meta."""
        # Create a mock storage client that returns None for custom_meta
        mock_storage_client = MagicMock()
        mock_storage_client.put.return_value = None

        # Create manager with mocked dependencies
        config = {"client_name": "MockClient"}
        with patch(
            "transfer_queue.storage.managers.base.StorageClientFactory.create", return_value=mock_storage_client
        ):
            manager = KVStorageManager(config)

        # Run put_data
        asyncio.run(manager.put_data(self.data, self.metadata))

        # Verify notify_data_update was called with empty dict for custom_meta
        mock_notify.assert_called_once()
        notify_call_args = mock_notify.call_args
        per_field_custom_meta = notify_call_args[0][5]  # 6th positional argument
        self.assertEqual(per_field_custom_meta, {})

    @patch.object(KVStorageManager, "_connect_to_controller")
    @patch.object(KVStorageManager, "notify_data_update", new_callable=AsyncMock)
    def test_put_data_custom_meta_length_mismatch_raises_error(self, mock_notify, mock_connect):
        """Test that put_data raises ValueError when custom_meta length doesn't match keys."""
        # Create a mock storage client that returns mismatched custom_meta length
        mock_storage_client = MagicMock()
        # Return only 3 custom_meta entries when 6 are expected
        mock_storage_client.put.return_value = [{"key": "1"}, {"key": "2"}, {"key": "3"}]

        # Create manager with mocked dependencies
        config = {"client_name": "MockClient"}
        with patch(
            "transfer_queue.storage.managers.base.StorageClientFactory.create", return_value=mock_storage_client
        ):
            manager = KVStorageManager(config)

        # Run put_data and expect ValueError
        with self.assertRaises(ValueError) as context:
            asyncio.run(manager.put_data(self.data, self.metadata))

        self.assertIn("does not match", str(context.exception))


if __name__ == "__main__":
    unittest.main()
