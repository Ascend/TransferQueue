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

"""Tests for the per-thread accelerator context used by MooncakeStoreClient.

Mooncake's Ascend transport is bound to a thread-local ACL context, but
register/unregister run on per-call ``ThreadPoolExecutor`` workers. These tests
stay hardware-free by faking the accelerator and the executor.
"""

import sys

from transfer_queue.storage.clients import mooncake_client as mc


class _FakeStore:
    def __init__(self):
        self.registered = []
        self.unregistered = []

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))

    def unregister_buffer(self, ptr):
        self.unregistered.append(ptr)


class _FakeNpu:
    def __init__(self, available=True, current=5):
        self._available = available
        self._current = current
        self.set_devices = []

    def is_available(self):
        return self._available

    def current_device(self):
        return self._current

    def set_device(self, device):
        self.set_devices.append(device)


def _new_client():
    client = object.__new__(mc.MooncakeStoreClient)
    client._store = _FakeStore()
    client.use_gdr = False
    client._gdr_staging = None
    return client


def test_ensure_accelerator_context_sets_device_and_warms_once(monkeypatch):
    fake_npu = _FakeNpu(current=5)
    monkeypatch.setattr(mc.torch, "npu", fake_npu, raising=False)
    warmups = []
    monkeypatch.setattr(mc.torch, "zeros", lambda *a, **k: warmups.append(k))

    mc._ensure_accelerator_context._initialized = False
    mc._ensure_accelerator_context()
    mc._ensure_accelerator_context()

    assert fake_npu.set_devices == [5, 5]
    assert len(warmups) == 1  # context warmup runs once per process


def test_ensure_accelerator_context_noop_without_npu(monkeypatch):
    monkeypatch.setattr(mc.torch, "npu", None, raising=False)
    # Force the optional torch_npu import to fail so torch.npu stays absent.
    monkeypatch.setitem(sys.modules, "torch_npu", None)

    mc._ensure_accelerator_context()  # must not raise or touch a device


def _patch_executor(monkeypatch):
    captured = []
    real_executor = mc.ThreadPoolExecutor

    class _RecordingExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            captured.append(kwargs.get("initializer"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mc, "ThreadPoolExecutor", _RecordingExecutor)
    return captured


def test_put_pool_initializer_binds_thread_context(monkeypatch):
    calls = []
    monkeypatch.setattr(mc, "_ensure_accelerator_context", lambda: calls.append(1))
    captured = _patch_executor(monkeypatch)

    assert _new_client().put([], []) == []
    assert captured and all(fn is not None for fn in captured)

    captured[-1]()
    assert calls == [1]


def test_get_pool_initializer_binds_thread_context(monkeypatch):
    calls = []
    monkeypatch.setattr(mc, "_ensure_accelerator_context", lambda: calls.append(1))
    captured = _patch_executor(monkeypatch)

    assert _new_client().get([], [], [], []) == []
    assert captured and all(fn is not None for fn in captured)

    captured[-1]()
    assert calls == [1]
