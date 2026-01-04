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

import unittest
from importlib.util import find_spec

import pytest
import torch

from transfer_queue.storage.clients.factory import StorageClientFactory
from transfer_queue.storage.clients.yuanrong_client import YuanrongStorageClient


class Test(unittest.TestCase):
    def setUp(self):
        self.cfg = {"host": "127.0.0.1", "port": 31501, "device_id": 0}

    @pytest.mark.skipif(find_spec("datasystem") is None, reason="datasystem is not available")
    def test_create_client(self):
        self.assertIn("YuanrongStorageClient", StorageClientFactory._registry)
        self.assertIs(StorageClientFactory._registry["YuanrongStorageClient"], YuanrongStorageClient)
        StorageClientFactory.create("YuanrongStorageClient", self.cfg)

        with self.assertRaises(ValueError) as cm:
            StorageClientFactory.create("abc", self.cfg)
        self.assertIn("Unknown StorageClient", str(cm.exception))

    @pytest.mark.skipif(
        find_spec("torch_npu") is None or find_spec("datasystem") is None, reason="torch_npu is not available"
    )
    def test_client_create_empty_tensorlist(self):
        tensors = [torch.Tensor([2, 1]), torch.Tensor([1, 5]), torch.Tensor([0]), torch.Tensor([-1.5])]
        shapes = []
        dtypes = []
        for t in tensors:
            shapes.append(t.shape)
            dtypes.append(t.dtype)
        client = StorageClientFactory.create("YuanrongStorageClient", self.cfg)

        empty_tensors = client._create_empty_npu_tensorlist(shapes, dtypes)
        self.assertEqual(len(tensors), len(empty_tensors))
        for t, et in zip(tensors, empty_tensors, strict=False):
            self.assertEqual(t.shape, et.shape)
            self.assertEqual(t.dtype, et.dtype)


if __name__ == "__main__":
    unittest.main()
