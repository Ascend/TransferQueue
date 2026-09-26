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

"""Pooled receive buffers on the MooncakeStore tensor read path.

The store and the buffer pool are faked, so these tests need neither mooncake nor RDMA.
The fake pool keeps the upstream properties this path depends on: buffers are carved out
of one fixed arena and reused, exhaustion raises, a buffer cannot be returned while an
exported view of it is alive, and the pool cannot close while a buffer is out on loan.
"""

import ctypes
import logging
import threading
import weakref

import pytest
import torch

from transfer_queue.storage.clients import mooncake_client as mcc

LOGGER_NAME = "transfer_queue.storage.clients.mooncake_client"
ALIGN = 256

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
        self.closed = False
        self.transfer_error: Exception | None = None
        self.on_transfer = None

    def setup(self, *args):
        return 0

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 0

    def unregister_buffer(self, ptr):
        self.unregistered.append(ptr)
        return 0

    def batch_get_into(self, keys, ptrs, sizes):
        if self.transfer_error is not None:
            raise self.transfer_error
        if self.on_transfer is not None:
            self.on_transfer()
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            ctypes.memmove(ptr, self.objects[key], size)
        return list(sizes)

    def close(self):
        self.closed = True


class FakeBuffer:
    """One region carved out of the pool's arena."""

    def __init__(self, pool, offset, nbytes):
        self._pool = pool
        self._offset = offset
        self.nbytes = nbytes
        self.ptr = pool.arena.data_ptr() + offset
        self._views: list[weakref.ref] = []
        self.returned = False

    @property
    def buffer(self):
        # Mooncake hands out a fresh view per access and counts it as an export.
        view = self._pool.arena_bytes[self._offset : self._offset + max(self.nbytes, 1)]
        self._views.append(weakref.ref(view))
        return view

    def has_live_views(self):
        return any(ref() is not None for ref in self._views)

    def release(self):
        if self.has_live_views():
            raise RuntimeError("cannot release buffer while exported views exist")
        self._pool._give_back(self._offset)
        self.returned = True


class FakePool:
    """Sub-allocates one arena, the way mooncake carves up its registered local buffer."""

    def __init__(self, capacity, max_regions=None):
        self.capacity = capacity
        self.max_regions = max_regions
        self.arena = torch.empty(max(capacity, 1), dtype=torch.uint8)
        self.arena_bytes = self.arena.numpy()
        self.acquired: list[int] = []
        self.offsets: list[int] = []
        self.returned = 0
        self.active: dict[int, int] = {}
        self.peak_bytes = 0
        self.peak_regions = 0
        self.closed = False
        self._lock = threading.Lock()

    def acquire(self, nbytes, block=True):
        with self._lock:
            if self.closed:
                raise RuntimeError("buffer pool is closed")
            size = max(nbytes, 1)
            if size > self.capacity:
                raise RuntimeError("requested buffer size exceeds pool capacity")
            if self.max_regions is not None and len(self.active) >= self.max_regions:
                raise RuntimeError("buffer pool is exhausted")
            offset = self._first_fit(size)
            if offset is None:
                raise RuntimeError("buffer pool is exhausted")
            self.active[offset] = size
            self.acquired.append(nbytes)
            self.offsets.append(offset)
            self.peak_bytes = max(self.peak_bytes, sum(self.active.values()))
            self.peak_regions = max(self.peak_regions, len(self.active))
            return FakeBuffer(self, offset, nbytes)

    def _first_fit(self, size):
        cursor = 0
        for offset in sorted(self.active):
            if offset - cursor >= size:
                return cursor
            cursor = -(-(offset + self.active[offset]) // ALIGN) * ALIGN
        return cursor if self.capacity - cursor >= size else None

    def _give_back(self, offset):
        with self._lock:
            del self.active[offset]
            self.returned += 1

    def close(self):
        with self._lock:
            if self.active:
                raise RuntimeError("cannot close buffer pool with active leases")
            self.closed = True


class PoolFactory:
    """Stands in for mooncake.store.BufferPool, recording how the client configured it."""

    def __init__(self, capacity=None):
        self._capacity = capacity
        self.kwargs: dict | None = None
        self.pool: FakePool | None = None

    def __call__(self, store, max_bytes=0, max_regions=None):
        self.kwargs = {"max_bytes": max_bytes, "max_regions": max_regions}
        self.pool = FakePool(max_bytes if self._capacity is None else self._capacity, max_regions)
        return self.pool


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    monkeypatch.setattr(mcc, "MOONCAKE_STORE_IMPORTED", True)
    # raising=False: these symbols are absent unless mooncake is installed.
    monkeypatch.setattr(mcc, "MooncakeDistributedStore", lambda: fake, raising=False)
    monkeypatch.setattr(mcc, "ReplicateConfig", type("ReplicateConfig", (), {}), raising=False)
    return fake


def make_client(local_buffer_size=1 << 30, **extra):
    config = {
        "local_hostname": "127.0.0.1",
        "metadata_server": "127.0.0.1:8080",
        "master_server_address": "127.0.0.1:8081",
        "local_buffer_size": local_buffer_size,
    }
    config.update(extra)
    return mcc.MooncakeStoreClient(config)


def install_pool(monkeypatch, capacity=None):
    """Make the client believe mooncake provides a buffer pool, and hand it a fake one."""
    factory = PoolFactory(capacity)
    monkeypatch.setattr(mcc, "MOONCAKE_BUFFER_POOL_IMPORTED", True)
    monkeypatch.setattr(mcc, "BufferPool", factory, raising=False)
    return factory


def make_client_and_pool(monkeypatch, local_buffer_size=1 << 30, capacity=None, **extra):
    factory = install_pool(monkeypatch, capacity)
    client = make_client(local_buffer_size, **extra)
    return client, factory.pool


def read(client, keys, shapes, dtypes):
    tensors, indexes = client._get_tensors_thread_worker(keys, shapes, dtypes, list(range(len(keys))))
    assert indexes == list(range(len(keys)))
    return tensors


def read_all(client):
    return read(client, KEYS, SHAPES, DTYPES)


def assert_payloads(tensors):
    for got, want in zip(tensors, expected_tensors(), strict=True):
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        assert torch.equal(got, want)


def test_reads_land_in_pooled_buffer(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch)

    tensors = read_all(client)

    # Mixed dtypes and a scalar all round-trip through the staged copy-out.
    assert_payloads(tensors)
    # The whole point: no registration on the data path, and no buffer left on loan.
    assert store.registered == []
    assert len(pool.acquired) == 1 and pool.returned == 1 and pool.active == {}


def test_pool_budget_is_capped_by_the_registered_local_buffer(store, monkeypatch):
    factory = install_pool(monkeypatch)
    client = make_client(local_buffer_size=1 << 20)

    # max_bytes=0 would let mooncake register fresh memory per lease once the local
    # buffer is full, which is exactly the cost this path removes.
    assert factory.kwargs == {"max_bytes": 1 << 20, "max_regions": mcc.MAX_BATCH_WORKER_THREADS}
    # Half of an equal per-thread share, leaving the store room for its own staging.
    assert client._lease_bytes == (1 << 20) // (2 * mcc.MAX_BATCH_WORKER_THREADS)


def test_batch_larger_than_share_is_read_in_rounds(store, monkeypatch):
    # Small local buffer: each thread's share holds only part of the batch.
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)

    tensors = read_all(client)

    assert_payloads(tensors)
    assert len(pool.acquired) > 1
    assert all(nbytes <= client._lease_bytes for nbytes in pool.acquired)
    assert pool.returned == len(pool.acquired)
    assert store.registered == []


def test_tensor_larger_than_share_registers_its_own_buffer(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)
    keys, shapes, dtypes = ["big", "small"], [(256,), (4,)], [torch.float32, torch.float32]
    payloads = [torch.arange(256, dtype=torch.float32), torch.arange(4, dtype=torch.float32)]
    store.objects = {k: bytes(t.numpy().tobytes()) for k, t in zip(keys, payloads, strict=True)}

    tensors = read(client, keys, shapes, dtypes)

    for got, want in zip(tensors, payloads, strict=True):
        assert torch.equal(got, want)
    # The oversized tensor must not lease the headroom the other readers need; the
    # small one still takes the pooled path.
    assert pool.acquired and all(nbytes <= client._lease_bytes for nbytes in pool.acquired)
    assert store.registered and len(store.unregistered) == len(store.registered)
    assert pool.active == {}


def test_falls_back_to_own_buffers_when_pool_is_exhausted(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)
    # Mooncake's own staging holds the arena, so acquire() raises instead of returning None.
    held = pool.acquire(pool.capacity)

    tensors = read_all(client)

    assert_payloads(tensors)
    assert pool.acquired == [pool.capacity]
    assert store.registered and len(store.unregistered) == len(store.registered)
    held.release()


def test_exhaustion_warns_once(store, monkeypatch, caplog):
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)
    held = pool.acquire(pool.capacity)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        read_all(client)
        read_all(client)

    # Every group of every read falls back here; the hot path must not flood the log.
    fallbacks = [r for r in caplog.records if "falling back to registering" in r.message]
    assert len(fallbacks) == 1
    held.release()


def test_registers_receive_regions_without_pool_support(store, monkeypatch):
    monkeypatch.setattr(mcc, "MOONCAKE_BUFFER_POOL_IMPORTED", False)

    tensors = read_all(make_client())

    assert_payloads(tensors)
    assert store.registered and len(store.unregistered) == len(store.registered)


def test_uniform_group_copied_in_one_strided_pass(store, monkeypatch):
    # Many identical small tensors (the fragmented-read case) take the single strided
    # copy-out path instead of a per-tensor loop; the payloads must still round-trip.
    client, pool = make_client_and_pool(monkeypatch)
    n = 8
    keys = [f"u{i}" for i in range(n)]
    payloads = [torch.arange(i, i + 16, dtype=torch.float32) for i in range(n)]
    store.objects = {k: bytes(t.numpy().tobytes()) for k, t in zip(keys, payloads, strict=True)}

    tensors = read(client, keys, [(16,)] * n, [torch.float32] * n)

    for got, want in zip(tensors, payloads, strict=True):
        assert torch.equal(got, want)
    assert pool.acquired and pool.returned == len(pool.acquired)
    assert store.registered == []


def test_returned_tensors_survive_buffer_reuse(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)

    first = read_all(client)
    rounds = len(pool.acquired)
    second = read_all(client)

    # The second read gets the same arena offsets back, so the first read's tensors
    # prove the copy-out really handed ownership to the caller.
    assert pool.offsets[rounds:] == pool.offsets[:rounds]
    assert pool.returned == len(pool.acquired)
    assert_payloads(first)
    assert_payloads(second)


def test_failed_transfer_returns_the_buffer_and_keeps_the_cause(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch)
    store.transfer_error = RuntimeError("transport is down")

    with pytest.raises(RuntimeError, match="transport is down"):
        read_all(client)

    assert pool.active == {} and pool.returned == len(pool.acquired)


def test_failed_copy_out_returns_the_buffer_and_keeps_the_cause(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch)

    def boom(self, source, *args, **kwargs):
        # Drop the staged view the caller passed in: a real copy_ failure comes from C++
        # and leaves no Python frame holding it, so only the client's own reference counts.
        del source
        raise RuntimeError("copy-out failed")

    monkeypatch.setattr(torch.Tensor, "copy_", boom)

    # A staged view left alive by the failing copy would make release() raise and
    # replace the real cause, so the view has to be dropped on the error path too.
    with pytest.raises(RuntimeError, match="copy-out failed"):
        read_all(client)

    assert pool.active == {} and pool.returned == len(pool.acquired)


def test_concurrent_readers_stay_within_the_local_buffer(store, monkeypatch):
    local_buffer_size = 8192
    readers = 8
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=local_buffer_size)
    barrier = threading.Barrier(readers)
    # Hold every reader inside its transfer, so all of them need memory at the same time.
    store.on_transfer = lambda: barrier.wait(timeout=30)
    results: list[list[torch.Tensor]] = [None] * readers  # type: ignore[list-item]
    failures: list[Exception] = []

    def reader(slot):
        try:
            results[slot] = read_all(client)
        except Exception as e:  # reported after the join below
            failures.append(e)

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(readers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not failures
    for tensors in results:
        assert_payloads(tensors)
    # Readers past the region cap fall back instead of eating into the store's headroom.
    assert pool.peak_regions <= mcc.MAX_BATCH_WORKER_THREADS
    assert pool.peak_bytes <= local_buffer_size // 2
    assert pool.active == {}


@pytest.mark.parametrize(
    ("config", "warns"),
    [
        ({"protocol": "rdma"}, True),
        ({"protocol": "tcp"}, False),
        ({"protocol": "rdma", "use_gdr": True}, False),
    ],
)
def test_missing_pool_support_warns_only_for_cpu_rdma_reads(store, monkeypatch, caplog, config, warns):
    monkeypatch.setattr(mcc, "MOONCAKE_BUFFER_POOL_IMPORTED", False)
    # GDR staging defers cudaMalloc, so it can stand in without a CUDA device here.
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        make_client(**config)

    assert any("BufferPool is unavailable" in r.message for r in caplog.records) is warns


def test_close_returns_the_pool_before_the_store(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch)

    read_all(client)
    client.close()

    assert pool.closed and store.closed and client._buffer_pool is None


def test_close_keeps_the_store_open_while_a_buffer_is_out(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch)
    held = pool.acquire(1024)

    # The store owns the memory the buffer points into, so it must outlive the pool.
    with pytest.raises(RuntimeError, match="while receive buffers are leased"):
        client.close()

    assert not store.closed and client._buffer_pool is pool
    held.release()
    client.close()
    assert pool.closed and store.closed
