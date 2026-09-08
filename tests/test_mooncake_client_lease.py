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

"""Receive-buffer leasing on the MooncakeStore tensor read path.

The store and the lease pool are faked, so these tests need neither mooncake nor RDMA.
"""

import ctypes
import sys

import pytest
import torch

from transfer_queue.storage.clients import mooncake_client as mcc

DTYPES = [torch.float32, torch.int64, torch.float32, torch.int16]
SHAPES = [(4, 3), (5,), (), (2, 8)]
KEYS = ["k0", "k1", "k2", "k3"]


def expected_tensors():
    """Deterministic payloads, one per key in KEYS."""
    out = []
    for seed, (dtype, shape) in enumerate(zip(DTYPES, SHAPES, strict=True)):
        numel = torch.empty(shape).numel()
        values = torch.arange(seed * 100, seed * 100 + numel)
        out.append(values.to(dtype).reshape(shape))
    return out


class FakeStore:
    """Writes the stored payload into whatever pointer batch_get_into is given."""

    def __init__(self):
        self.objects = {
            key: bytes(t.contiguous().numpy().tobytes()) for key, t in zip(KEYS, expected_tensors(), strict=True)
        }
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []

    def setup(self, *args):
        return 0

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 0

    def unregister_buffer(self, ptr):
        self.unregistered.append(ptr)
        return 0

    def batch_get_into(self, keys, ptrs, sizes):
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            ctypes.memmove(ptr, self.objects[key], size)
        return list(sizes)


class FakeLease:
    def __init__(self, pool, nbytes):
        self._pool = pool
        self._memory = torch.empty(nbytes, dtype=torch.uint8)
        self.ptr = self._memory.data_ptr()
        # A numpy array (not its .data memoryview): torch.frombuffer keeps a reference
        # to it, so a staged view left alive at release() shows up as an extra refcount.
        self.buffer = self._memory.numpy()
        self._baseline_refs = sys.getrefcount(self.buffer)

    def release(self):
        # Mirror mooncake: the pool refuses to return a lease while a view of its
        # buffer is still alive (a live torch.frombuffer tensor holds a reference).
        if sys.getrefcount(self.buffer) > self._baseline_refs:
            raise RuntimeError("cannot release buffer while exported views exist")
        self._pool.released += 1


class FakePool:
    """Serves leases up to ``capacity`` bytes; larger requests cannot be served."""

    def __init__(self, capacity=1 << 30):
        self.capacity = capacity
        self.acquired: list[int] = []
        self.released = 0

    def acquire(self, nbytes, block=True):
        if nbytes > self.capacity:
            return None
        self.acquired.append(nbytes)
        return FakeLease(self, nbytes)


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(mcc, "MOONCAKE_STORE_IMPORTED", True)
    # raising=False: these symbols are absent unless mooncake is installed.
    monkeypatch.setattr(mcc, "MooncakeDistributedStore", lambda: fake, raising=False)
    monkeypatch.setattr(mcc, "ReplicateConfig", type("ReplicateConfig", (), {}), raising=False)
    return fake


def make_client(local_buffer_size=1 << 30):
    return mcc.MooncakeStoreClient(
        {
            "local_hostname": "127.0.0.1",
            "metadata_server": "127.0.0.1:8080",
            "master_server_address": "127.0.0.1:8081",
            "local_buffer_size": local_buffer_size,
        }
    )


def read_all(client):
    tensors, indexes = client._get_tensors_thread_worker(KEYS, SHAPES, DTYPES, list(range(len(KEYS))))
    assert indexes == list(range(len(KEYS)))
    return tensors


def assert_payloads(tensors):
    for got, want in zip(tensors, expected_tensors(), strict=True):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert torch.equal(got, want)


def install_pool(monkeypatch, capacity=1 << 30):
    """Make the client believe mooncake provides a lease pool, and hand it a fake one."""
    fake = FakePool(capacity)
    monkeypatch.setattr(mcc, "MOONCAKE_BUFFER_POOL_IMPORTED", True)
    monkeypatch.setattr(mcc, "BufferPool", lambda _store, max_bytes: fake, raising=False)
    return fake


def test_reads_land_in_leased_buffer(store, monkeypatch):
    pool = install_pool(monkeypatch)

    tensors = read_all(make_client())

    assert_payloads(tensors)
    # The whole point: no registration on the data path, and no lease left behind.
    assert store.registered == []
    assert len(pool.acquired) == 1 and pool.released == 1


def test_batch_larger_than_lease_share_is_read_in_rounds(store, monkeypatch):
    pool = install_pool(monkeypatch)

    # Small local buffer: each thread's share holds only part of the batch.
    client = make_client(local_buffer_size=256 * mcc.MAX_BATCH_WORKER_THREADS)
    tensors = read_all(client)

    assert_payloads(tensors)
    assert len(pool.acquired) > 1
    assert all(nbytes <= client._lease_bytes for nbytes in pool.acquired)
    assert pool.released == len(pool.acquired)
    assert store.registered == []


def test_falls_back_to_own_buffers_when_pool_cannot_serve(store, monkeypatch):
    pool = install_pool(monkeypatch, capacity=0)

    tensors = read_all(make_client())

    assert_payloads(tensors)
    assert pool.acquired == []
    assert store.registered and len(store.unregistered) == len(store.registered)


def test_registers_receive_regions_without_lease_support(store, monkeypatch):
    monkeypatch.setattr(mcc, "MOONCAKE_BUFFER_POOL_IMPORTED", False)

    tensors = read_all(make_client())

    assert_payloads(tensors)
    assert store.registered and len(store.unregistered) == len(store.registered)


def test_uniform_group_copied_in_one_strided_pass(store, monkeypatch):
    # Many identical small tensors (the fragmented-read case) take the single strided
    # copy-out path instead of a per-tensor loop; the payloads must still round-trip.
    pool = install_pool(monkeypatch)
    n = 8
    dtypes = [torch.float32] * n
    shapes = [(16,)] * n
    keys = [f"u{i}" for i in range(n)]
    payloads = [torch.arange(i, i + 16, dtype=torch.float32) for i in range(n)]
    store.objects = {k: bytes(t.numpy().tobytes()) for k, t in zip(keys, payloads, strict=True)}

    tensors, indexes = make_client()._get_tensors_thread_worker(keys, shapes, dtypes, list(range(n)))

    assert indexes == list(range(n))
    for got, want in zip(tensors, payloads, strict=True):
        assert torch.equal(got, want)
    assert pool.acquired and pool.released == len(pool.acquired)
    assert store.registered == []
