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
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue.storage.clients.yuanrong_client import (  # noqa: E402
    KVClientAdapter,
)


class MockBuffer:
    def __init__(self, size):
        self.data = bytearray(size)

    def MutableData(self):
        return self.data


class TestYuanrongKVClientZCopy:
    @pytest.fixture
    def mock_kv_client(self, mocker):
        mock_client = MagicMock()
        mock_client.init.return_value = None

        mocker.patch("yr.datasystem.KVClient", return_value=mock_client)
        mocker.patch("yr.datasystem.DsTensorClient")
        mocker.patch("transfer_queue.storage.clients.yuanrong_client.TORCH_NPU_IMPORTED", False)

        return mock_client

    @pytest.fixture
    def storage_client(self, mock_kv_client):
        return KVClientAdapter({"host": "127.0.0.1", "port": 31501})

    def test_mset_mget_p2p(self, storage_client, mocker):
        # Mock serialization/deserialization
        def mock_serialization(obj):
            if isinstance(obj, torch.Tensor):
                return [obj.numpy().tobytes()]
            return [str(obj).encode("utf-8")]

        def mock_deserialization(items):
            data = items[0]
            if len(data) == 12:
                return torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
            try:
                return data.tobytes().decode("utf-8")
            except UnicodeDecodeError:
                return data

        mocker.patch("transfer_queue.storage.clients.yuanrong_client._encoder.encode", side_effect=mock_serialization)
        mocker.patch("transfer_queue.storage.clients.yuanrong_client._decoder.decode", side_effect=mock_deserialization)

        stored_raw_buffers = []

        def side_effect_mcreate(keys, sizes):
            buffers = [MockBuffer(size) for size in sizes]
            for b in buffers:
                stored_raw_buffers.append(b.MutableData())
            return buffers

        storage_client._ds_client.mcreate.side_effect = side_effect_mcreate
        storage_client._ds_client.get_buffers.return_value = stored_raw_buffers

        storage_client.mset_zero_copy(
            ["tensor_key", "string_key"], [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32), "hello yuanrong"]
        )
        results = storage_client.mget_zero_copy(["tensor_key", "string_key"])

        assert torch.allclose(results[0], torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        assert results[1] == "hello yuanrong"
