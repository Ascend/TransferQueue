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
from unittest import mock

import pytest
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


# --- Mock Backend Implementation ---


class MockDsTensorClient:
    def __init__(self, host, port, device_id):
        self.storage = {}

    def init(self):
        pass

    def dev_mset(self, keys, values):
        for k, v in zip(keys, values, strict=True):
            assert v.device.type == "npu"
            self.storage[k] = v.clone()

    def dev_mget(self, keys, out_tensors):
        for i, k in enumerate(keys):
            if k in self.storage:
                out_tensors[i].copy_(self.storage[k])
            # Note: If key is missing, tensor remains unchanged (mock limitation)

    def dev_delete(self, keys):
        for k in keys:
            self.storage.pop(k, None)


class MockKVClient:
    def __init__(self, host, port):
        self.storage = {}

    def init(self):
        pass

    def mcreate(self, keys, sizes):
        class MockBuffer:
            def __init__(self, size):
                self._data = bytearray(size)

            def MutableData(self):
                return memoryview(self._data)

        self._current_keys = keys
        return [MockBuffer(s) for s in sizes]

    def mset_buffer(self, buffers):
        for key, buf in zip(self._current_keys, buffers, strict=True):
            self.storage[key] = bytes(buf.MutableData())

    def get_buffers(self, keys):
        return [memoryview(self.storage[k]) if k in self.storage else None for k in keys]

    def delete(self, keys):
        for k in keys:
            self.storage.pop(k, None)


# --- Fixtures ---


@pytest.fixture
def mock_yr_datasystem():
    """Wipe real 'yr' modules and inject mocks."""

    # 1. Clean up sys.modules to force a fresh import under mock conditions
    # This ensures top-level code in yuanrong_client.py is re-executed
    to_delete = [k for k in sys.modules if k.startswith("yr")]
    for mod in to_delete:
        del sys.modules[mod]

    # 2. Setup Mock Objects
    ds_mock = mock.MagicMock()
    ds_mock.DsTensorClient = MockDsTensorClient
    ds_mock.KVClient = MockKVClient

    yr_mock = mock.MagicMock(datasystem=ds_mock)

    # 3. Apply patches
    # - sys.modules: Redirects 'import yr' to our mocks
    # - YUANRONG_DATASYSTEM_IMPORTED: Forces the existence check to True so initialize the client successfully
    # - datasystem: Direct attribute patch for the module
    with (
        mock.patch.dict("sys.modules", {"yr": yr_mock, "yr.datasystem": ds_mock}),
        mock.patch("transfer_queue.storage.clients.yuanrong_client.YUANRONG_DATASYSTEM_IMPORTED", True, create=True),
        mock.patch("transfer_queue.storage.clients.yuanrong_client.datasystem", ds_mock),
    ):
        yield


@pytest.fixture
def config():
    return {"host": "127.0.0.1", "port": 12345, "enable_yr_npu_optimization": True}


def assert_tensors_equal(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape and a.dtype == b.dtype
    # Move to CPU for cross-device comparison
    assert torch.equal(a.cpu(), b.cpu())


# --- Test Suite ---


class TestYuanrongStorageE2E:
    @pytest.fixture(autouse=True)
    def setup_client(self, mock_yr_datasystem, config):
        # Lazy import to ensure mocks are active
        from transfer_queue.storage.clients.yuanrong_client import YuanrongStorageClient

        self.client_cls = YuanrongStorageClient
        self.config = config

    def _create_data(self, mode="cpu"):
        if mode == "cpu":
            keys = ["t", "s", "i"]
            vals = [torch.randn(2), "hi", 1]
        elif mode == "npu":
            if not (hasattr(torch, "npu") and torch.npu.is_available()):
                pytest.skip("NPU required")
            keys = ["n1", "n2"]
            vals = [torch.randn(2).npu(), torch.tensor([1]).npu()]
        else:  # mixed
            if not (hasattr(torch, "npu") and torch.npu.is_available()):
                pytest.skip("NPU required")
            keys = ["n1", "c1"]
            vals = [torch.randn(2).npu(), "cpu"]

        shapes = [list(v.shape) if isinstance(v, torch.Tensor) else [] for v in vals]
        dtypes = [v.dtype if isinstance(v, torch.Tensor) else None for v in vals]
        return keys, vals, shapes, dtypes

    def test_cpu_only_flow(self, config):
        client = self.client_cls(config)
        keys, vals, shp, dt = self._create_data("cpu")

        # Put & Verify Meta
        meta = client.put(keys, vals)
        assert all(m == "KVClient" for m in meta)

        # Get & Verify Values
        ret = client.get(keys, shp, dt, meta)
        for o, r in zip(vals, ret, strict=True):
            if isinstance(o, torch.Tensor):
                assert_tensors_equal(o, r)
            else:
                assert o == r

        # Clear & Verify
        client.clear(keys, meta)
        assert all(v is None for v in client.get(keys, shp, dt, meta))

    def test_npu_only_flow(self, config):
        keys, vals, shp, dt = self._create_data("npu")
        client = self.client_cls(config)

        meta = client.put(keys, vals)
        assert all(m == "DsTensorClient" for m in meta)

        ret = client.get(keys, shp, dt, meta)
        for o, r in zip(vals, ret, strict=True):
            assert_tensors_equal(o, r)

        client.clear(keys, meta)

    def test_mixed_flow(self, config):
        keys, vals, shp, dt = self._create_data("mixed")
        client = self.client_cls(config)

        meta = client.put(keys, vals)
        assert set(meta) == {"DsTensorClient", "KVClient"}

        ret = client.get(keys, shp, dt, meta)
        for o, r in zip(vals, ret, strict=True):
            if isinstance(o, torch.Tensor):
                assert_tensors_equal(o, r)
            else:
                assert o == r
