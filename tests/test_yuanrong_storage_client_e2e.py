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

import pytest
import torch
from unittest import mock
from typing import Any, Optional

from transfer_queue.storage.clients.factory import StorageClientFactory


# --- Mock Backend Clients ---
class MockDsTensorClient:
    def __init__(self, host, port, device_id):
        self.host = host
        self.port = port
        self.device_id = device_id
        self.storage = {}  # key -> tensor

    def init(self):
        pass

    def dev_mset(self, keys, values):
        for k, v in zip(keys, values):
            assert v.device.type == "npu"
            self.storage[k] = v.clone()  # simulate store

    def dev_mget(self, keys, out_tensors):
        for i, k in enumerate(keys):
            if k in self.storage:
                out_tensors[i].copy_(self.storage[k])

    def dev_delete(self, keys):
        for k in keys:
            self.storage.pop(k, None)


class MockKVClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.storage = {}  # key -> bytes

    def init(self):
        pass

    def mcreate(self, keys, sizes):
        # We don't actually need to return real buffers if we intercept packing
        # But to keep interface, return dummy buffers
        class MockBuffer:
            def __init__(self, size):
                self._data = bytearray(size)
                self.size = size
            def MutableData(self):
                return memoryview(self._data)
        self._current_keys = keys  # HACK: remember keys for mset_buffer
        return [MockBuffer(s) for s in sizes]

    def mset_buffer(self, buffers):
        from transfer_queue.utils.serial_utils import _decoder, _encoder
        # Reconstruct objects from buffers (simulate what pack_into did)
        # But we can't easily unpack without knowing original obj...
        # Instead, during test, we can assume that whatever was packed is recoverable.
        # But for mock, let's just store the raw bytes of the first item in buffer
        for key, buf in zip(self._current_keys, buffers):
            # Extract the full content of the buffer
            raw_bytes = bytes(buf.MutableData())
            self.storage[key] = raw_bytes
        del self._current_keys

    def get_buffers(self, keys):
        results = []
        for k in keys:
            raw = self.storage.get(k)
            if raw is None:
                results.append(None)
            else:
                results.append(memoryview(raw))
        return results

    def delete(self, keys):
        for k in keys:
            self.storage.pop(k, None)


# --- Patch utilities ---
def make_mock_datasystem():
    """Returns a mock module that replaces yr.datasystem"""
    mock_ds = mock.MagicMock()
    mock_ds.DsTensorClient = MockDsTensorClient
    mock_ds.KVClient = MockKVClient
    return mock_ds


# --- Test Fixtures ---
@pytest.fixture
def mock_yr_datasystem():
    with mock.patch.dict("sys.modules", {"yr": mock.MagicMock(), "yr.datasystem": make_mock_datasystem()}):
        import yr  # noqa
        yield


@pytest.fixture
def config():
    return {
        "host": "127.0.0.1",
        "port": 12345,
        "enable_yr_npu_optimization": True,
    }


# --- Helper: Check tensor equality with device awareness ---
def assert_tensors_equal(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    if a.device.type == "npu" or b.device.type == "npu":
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        assert torch.equal(a_cpu, b_cpu)
    else:
        assert torch.equal(a, b)


# --- Main Test Class ---
class TestYuanrongStorageE2E:

    @pytest.fixture(autouse=True)
    def setup_client(self, mock_yr_datasystem, config):
        # Re-register YuanrongStorageClient in case it was loaded before
        from transfer_queue.storage.clients.yuanrong import YuanrongStorageClient  # adjust import path as needed
        self.client_cls = YuanrongStorageClient
        self.config = config

    def _create_test_data_cpu(self):
        """Pure CPU data: tensors + primitives"""
        keys = ["cpu_tensor", "string", "int_val", "bool_val", "list_val"]
        values = [
            torch.randn(3, 4),          # CPU tensor
            "hello world",
            42,
            True,
            [1, 2, {"a": 3}],
        ]
        shapes = [list(v.shape) if isinstance(v, torch.Tensor) else [] for v in values]
        dtypes = [v.dtype if isinstance(v, torch.Tensor) else None for v in values]
        return keys, values, shapes, dtypes

    def _create_test_data_npu(self):
        """Pure NPU tensors (only if NPU available)"""
        if not hasattr(torch, 'npu') or not torch.npu.is_available():
            pytest.skip("NPU not available")
        keys = ["npu_tensor1", "npu_tensor2"]
        values = [
            torch.randn(2, 3).npu(),
            torch.tensor([1, 2, 3], dtype=torch.int64).npu(),
        ]
        shapes = [list(v.shape) for v in values]
        dtypes = [v.dtype for v in values]
        return keys, values, shapes, dtypes

    def _create_test_data_mixed(self):
        """Mixed NPU + CPU"""
        if not hasattr(torch, 'npu') or not torch.npu.is_available():
            pytest.skip("NPU not available")
        keys = ["npu_t", "cpu_t", "str_val"]
        values = [
            torch.randn(1, 2).npu(),
            torch.tensor([5.0]),  # CPU
            "mixed",
        ]
        shapes = [list(v.shape) if isinstance(v, torch.Tensor) else [] for v in values]
        dtypes = [v.dtype if isinstance(v, torch.Tensor) else None for v in values]
        return keys, values, shapes, dtypes

    def test_put_get_clear_cpu_only(self, config):
        client = self.client_cls(config)
        keys, values, shapes, dtypes = self._create_test_data_cpu()

        # Put
        custom_meta = client.put(keys, values)
        assert len(custom_meta) == len(keys)
        assert all(cm == "KVClient" for cm in custom_meta)

        # Get
        retrieved = client.get(keys, shapes=shapes, dtypes=dtypes, custom_meta=custom_meta)
        assert len(retrieved) == len(values)
        for orig, ret in zip(values, retrieved):
            if isinstance(orig, torch.Tensor):
                assert_tensors_equal(orig, ret)
            else:
                assert orig == ret

        # Clear
        client.clear(keys, custom_meta=custom_meta)

        # Verify cleared (optional: try get again → should be None or error)
        # Since our mock returns zeros for missing NPU keys but None for KV,
        # and KV mock returns None for missing keys:
        after_clear = client.get(keys, shapes=shapes, dtypes=dtypes, custom_meta=custom_meta)
        assert all(v is None for v in after_clear)
        # Note: For simplicity, we don't deeply verify deletion; focus on no crash.

    def test_put_get_clear_npu_only(self, config):
        if not (hasattr(torch, 'npu') and torch.npu.is_available()):
            pytest.skip("NPU not available")

        # Force enable NPU path
        config["enable_yr_npu_optimization"] = True
        client = self.client_cls(config)

        keys, values, shapes, dtypes = self._create_test_data_npu()

        # Put
        custom_meta = client.put(keys, values)
        assert all(cm == "DsTensorClient" for cm in custom_meta)

        # Get
        retrieved = client.get(keys, shapes=shapes, dtypes=dtypes, custom_meta=custom_meta)
        for orig, ret in zip(values, retrieved):
            assert_tensors_equal(orig, ret)

        # Clear
        client.clear(keys, custom_meta=custom_meta)

    def test_put_get_clear_mixed(self, config):
        if not (hasattr(torch, 'npu') and torch.npu.is_available()):
            pytest.skip("NPU not available")

        config["enable_yr_npu_optimization"] = True
        client = self.client_cls(config)

        keys, values, shapes, dtypes = self._create_test_data_mixed()

        # Put
        custom_meta = client.put(keys, values)
        # Should have both strategies
        assert "DsTensorClient" in custom_meta
        assert "KVClient" in custom_meta

        # Get
        retrieved = client.get(keys, shapes=shapes, dtypes=dtypes, custom_meta=custom_meta)
        for orig, ret in zip(values, retrieved):
            if isinstance(orig, torch.Tensor):
                assert_tensors_equal(orig, ret)
            else:
                assert orig == ret

        # Clear
        client.clear(keys, custom_meta=custom_meta)

    def test_without_npu_fallback_to_kv(self, config):
        """Ensure NPU tensor is serialized via KVClient when NPU optimization is disabled."""
        if not (hasattr(torch, 'npu') and torch.npu.is_available()):
            pytest.skip("NPU not available")

        config = config.copy()
        config["enable_yr_npu_optimization"] = False
        client = self.client_cls(config)

        keys = ["fallback_tensor"]
        values = [torch.randn(2).npu()]  # NPU tensor

        # Should go to KVClient
        custom_meta = client.put(keys, values)
        assert custom_meta == ["KVClient"]

        # Prepare metadata for get
        shapes = [list(values[0].shape)]  # e.g., [2]
        dtypes = [values[0].dtype]        # e.g., torch.float32

        retrieved = client.get(keys, shapes=shapes, dtypes=dtypes, custom_meta=custom_meta)
        # Should be deserialized as CPU tensor
        assert retrieved[0].device.type == "cpu"
        assert torch.equal(values[0].cpu(), retrieved[0])

        # Clear and verify
        client.clear(keys, custom_meta=custom_meta)
        after_clear = client.get(keys, shapes=shapes, dtypes=dtypes, custom_meta=custom_meta)
        assert after_clear[0] is None