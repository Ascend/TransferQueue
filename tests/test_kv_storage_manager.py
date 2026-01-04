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

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
