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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from transfer_queue.storage.clients.yuanrong_client import GeneralKVClientAdapter

pytest.importorskip("yr")


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
        mocker.patch("transfer_queue.storage.clients.yuanrong_client.find_reachable_host", return_value="127.0.0.1")

        return mock_client

    @pytest.fixture
    def storage_client(self, mock_kv_client):
        return GeneralKVClientAdapter({"worker_port": 31501})

    @pytest.mark.parametrize("ttl", [0, 3600])
    def test_clear_honours_data_ttl(self, mock_kv_client, ttl):
        """With a TTL configured, clear expires keys instead of deleting them.

        Both halves have to hold together: the TTL must also reach mcreate, because a
        cleared index is reused and only an explicit TTL re-arms the deadline on the
        rewritten key.
        """
        client = GeneralKVClientAdapter({"worker_port": 31501, "data_ttl_second": ttl})
        assert client._ttl_second == ttl

        keys = ["k0", "k1"]
        client.clear(keys)
        if ttl:
            mock_kv_client.expire.assert_called_once_with(keys, client.CLEAR_EXPIRE_SECOND)
            mock_kv_client.delete.assert_not_called()
        else:
            mock_kv_client.delete.assert_called_once_with(keys)
            mock_kv_client.expire.assert_not_called()

        mock_kv_client.mcreate.side_effect = lambda ks, sizes, ttl_second=0: [MockBuffer(s) for s in sizes]
        client.mset_zero_copy(keys, [b"a", b"b"])
        assert mock_kv_client.mcreate.call_args.kwargs["ttl_second"] == ttl

    def test_clear_batches_beyond_the_key_limit(self, mock_kv_client):
        """datasystem rejects more than GET_CLEAR_KEYS_LIMIT keys in one call."""
        client = GeneralKVClientAdapter({"worker_port": 31501, "data_ttl_second": 60})
        n = client.GET_CLEAR_KEYS_LIMIT + 5
        client.clear([f"k{i}" for i in range(n)])
        assert mock_kv_client.expire.call_count == 2
        assert len(mock_kv_client.expire.call_args_list[0].args[0]) == client.GET_CLEAR_KEYS_LIMIT
        assert len(mock_kv_client.expire.call_args_list[1].args[0]) == 5

    def test_mset_mget_p2p(self, storage_client, mocker):
        # Mock serialization/deserialization
        def mock_encode(obj):
            if isinstance(obj, torch.Tensor):
                return [obj.numpy().tobytes()]
            return [str(obj).encode("utf-8")]

        def mock_decode(frames):
            data = frames[0]
            if len(data) == 12:
                return torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
            try:
                return data.tobytes().decode("utf-8")
            except UnicodeDecodeError:
                return data

        mocker.patch("transfer_queue.utils.serial_utils.encode", side_effect=mock_encode)
        mocker.patch("transfer_queue.utils.serial_utils.decode", side_effect=mock_decode)

        stored_raw_buffers = []

        def side_effect_mcreate(keys, sizes, ttl_second=0):
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
