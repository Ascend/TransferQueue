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

"""Tests for the per-thread NPU device binding used by MooncakeStoreClient.

Mooncake's Ascend transport is bound to a thread-local ACL context, but
register/unregister run on per-call ``ThreadPoolExecutor`` workers. These tests
stay hardware-free by faking the accelerator.
"""

from transfer_queue.storage.clients import mooncake_client as mc


class _FakeNpu:
    def __init__(self, current=5, initialized=True):
        self._current = current
        self._initialized = initialized
        self.set_devices = []
        self.current_device_calls = 0

    def is_initialized(self):
        return self._initialized

    def current_device(self):
        self.current_device_calls += 1
        return self._current

    def set_device(self, device):
        self.set_devices.append(device)


def _new_client(npu_device):
    client = object.__new__(mc.MooncakeStoreClient)
    client._npu_device = npu_device
    return client


def test_resolve_npu_device_captures_owning_thread_device(monkeypatch):
    fake_npu = _FakeNpu(current=3)
    monkeypatch.setattr(mc, "NPU_IMPORTED", True)
    monkeypatch.setattr(mc.torch, "npu", fake_npu, raising=False)

    assert mc._resolve_npu_device() == 3
    assert fake_npu.current_device_calls == 1


def test_resolve_npu_device_none_without_npu(monkeypatch):
    monkeypatch.setattr(mc, "NPU_IMPORTED", False)

    assert mc._resolve_npu_device() is None


def test_resolve_npu_device_none_when_not_initialized(monkeypatch):
    fake_npu = _FakeNpu(initialized=False)
    monkeypatch.setattr(mc, "NPU_IMPORTED", True)
    monkeypatch.setattr(mc.torch, "npu", fake_npu, raising=False)

    assert mc._resolve_npu_device() is None
    assert fake_npu.current_device_calls == 0


def test_worker_binds_device_captured_at_construction(monkeypatch):
    # A context-less worker reports device 0; the client captured 3 in __init__.
    fake_npu = _FakeNpu(current=0)
    monkeypatch.setattr(mc.torch, "npu", fake_npu, raising=False)

    _new_client(npu_device=3)._bind_npu_device()

    assert fake_npu.set_devices == [3]
    assert fake_npu.current_device_calls == 0


def test_worker_bind_noop_without_npu_device(monkeypatch):
    fake_npu = _FakeNpu()
    monkeypatch.setattr(mc.torch, "npu", fake_npu, raising=False)

    _new_client(npu_device=None)._bind_npu_device()

    assert fake_npu.set_devices == []


def test_worker_bind_logs_and_swallows_failure(monkeypatch):
    class _FailingNpu(_FakeNpu):
        def set_device(self, device):
            raise RuntimeError("set_device failed")

    fake_npu = _FailingNpu()
    monkeypatch.setattr(mc.torch, "npu", fake_npu, raising=False)
    warnings = []
    monkeypatch.setattr(mc.logger, "warning", lambda *a, **k: warnings.append((a, k)))

    _new_client(npu_device=3)._bind_npu_device()  # must not raise

    assert warnings and warnings[0][1].get("exc_info") is True
