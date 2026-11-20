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

"""Pooled receive and send buffers on the MooncakeStore tensor read and put paths.

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
        self.put_calls: list[tuple[list[str], list[int], list[int]]] = []

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

    def batch_upsert_from(self, keys, ptrs, sizes, config):
        self.put_calls.append((list(keys), list(ptrs), list(sizes)))
        if self.transfer_error is not None:
            raise self.transfer_error
        if self.on_transfer is not None:
            self.on_transfer()
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            self.objects[key] = ctypes.string_at(ptr, size)
        return [0] * len(keys)

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
        # The pool only runs on RDMA; default to it so the pooled-path tests exercise it.
        "protocol": "rdma",
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
    assert client._pool_share_bytes == (1 << 20) // (2 * mcc.MAX_BATCH_WORKER_THREADS)


def test_pool_is_disabled_on_tcp(store, monkeypatch):
    # TCP has no registration cost to save, so the pool would only add the copy-out.
    install_pool(monkeypatch)
    client = make_client(protocol="tcp")

    assert client._buffer_pool is None
    tensors = read_all(client)

    assert_payloads(tensors)
    assert store.registered and len(store.unregistered) == len(store.registered)


def test_batch_larger_than_share_is_read_in_rounds(store, monkeypatch):
    # Small local buffer: each thread's share holds only part of the batch.
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)

    tensors = read_all(client)

    assert_payloads(tensors)
    assert len(pool.acquired) > 1
    assert all(nbytes <= client._pool_share_bytes for nbytes in pool.acquired)
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
    assert pool.acquired and all(nbytes <= client._pool_share_bytes for nbytes in pool.acquired)
    assert store.registered and len(store.unregistered) == len(store.registered)
    assert pool.active == {}


def test_fallback_registers_everything_in_one_pass(store, monkeypatch):
    # Pool exhausted up front: every group falls back, but a single same-dtype batch must
    # register one merged region, not one per group (the pre-pool cost, not N times it).
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)
    held = pool.acquire(pool.capacity)

    n = 8
    keys = [f"f{i}" for i in range(n)]
    payloads = [torch.arange(i, i + 32, dtype=torch.float32) for i in range(n)]
    store.objects = {k: bytes(t.numpy().tobytes()) for k, t in zip(keys, payloads, strict=True)}

    tensors = read(client, keys, [(32,)] * n, [torch.float32] * n)

    for got, want in zip(tensors, payloads, strict=True):
        assert torch.equal(got, want)
    assert len(store.registered) == 1 and len(store.unregistered) == 1
    held.release()


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
    fallbacks = [r for r in caplog.records if "Falling back to registering" in r.message]
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


@pytest.mark.parametrize("operation", ["get", "put"])
def test_failed_transfer_returns_the_buffer_and_keeps_the_cause(store, monkeypatch, operation):
    client, pool = make_client_and_pool(monkeypatch)
    store.transfer_error = RuntimeError("transport is down")

    with pytest.raises(RuntimeError, match="transport is down") as exc:
        if operation == "get":
            read_all(client)
        else:
            client._put_tensors_thread_worker(KEYS, expected_tensors())

    assert exc.value is store.transfer_error
    assert pool.active == {} and pool.returned == len(pool.acquired) == 1


@pytest.mark.parametrize("operation", ["get", "put"])
def test_failed_copy_returns_the_buffer_and_keeps_the_cause(store, monkeypatch, operation):
    client, pool = make_client_and_pool(monkeypatch)
    error = RuntimeError("copy failed")

    def boom(self, source, *args, **kwargs):
        # A real copy_ failure leaves no Python mock frame holding either tensor.
        # Drop both references so the mock does not keep pooled memory alive.
        del self, source
        raise error

    monkeypatch.setattr(torch.Tensor, "copy_", boom)

    with pytest.raises(RuntimeError, match="copy failed") as exc:
        if operation == "get":
            read_all(client)
        else:
            client._put_tensors_thread_worker(KEYS, expected_tensors())

    assert exc.value is error
    assert pool.active == {} and pool.returned == len(pool.acquired) == 1


@pytest.mark.parametrize("mixed", [False, True], ids=["get_only", "mixed"])
def test_concurrent_transfers_stay_within_the_local_buffer(store, monkeypatch, mixed):
    local_buffer_size = 8192
    workers = 8
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=local_buffer_size)
    barrier = threading.Barrier(workers)
    # Hold every worker inside its transfer, so all of them need memory at the same time.
    store.on_transfer = lambda: barrier.wait(timeout=30)
    failures: list[Exception] = []

    def transfer(slot):
        try:
            if mixed and slot % 2:
                client._put_tensors_thread_worker([f"put{slot}/{key}" for key in KEYS], expected_tensors())
            else:
                assert_payloads(read_all(client))
        except Exception as e:  # reported after the join below
            failures.append(e)

    threads = [threading.Thread(target=transfer, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=35)

    assert not any(t.is_alive() for t in threads)
    assert not failures
    if mixed:
        for slot in range(1, workers, 2):
            for key, tensor in zip(KEYS, expected_tensors(), strict=True):
                assert store.objects[f"put{slot}/{key}"] == tensor.numpy().tobytes()
    # Workers past the region cap fall back instead of eating into the store's headroom.
    assert pool.peak_regions <= mcc.MAX_BATCH_WORKER_THREADS
    assert pool.peak_bytes <= local_buffer_size // 2
    assert pool.active == {} and pool.returned == len(pool.acquired)


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

    # The store owns the memory the buffer points into, so a refused pool close must not
    # tear it down underneath an active lease.
    with pytest.raises(RuntimeError, match="cannot close buffer pool with active leases"):
        client.close()

    assert not store.closed and client._buffer_pool is pool
    held.release()
    client.close()
    assert pool.closed and store.closed


def test_empty_tensor_reads_like_the_register_path(store, monkeypatch):
    # mooncake rejects a 0-byte put, so this is unreachable in practice; the guard only keeps
    # the pooled path from raising where the register path returns, should it ever be hit.
    keys, shapes, dtypes = ["empty", "normal"], [(0,), (4,)], [torch.float32, torch.float32]
    payloads = [torch.empty(0), torch.arange(4, dtype=torch.float32)]
    store.objects = {k: bytes(t.numpy().tobytes()) for k, t in zip(keys, payloads, strict=True)}

    monkeypatch.setattr(mcc, "MOONCAKE_BUFFER_POOL_IMPORTED", False)
    registered = read(make_client(), keys, shapes, dtypes)
    client, pool = make_client_and_pool(monkeypatch)
    pooled = read(client, keys, shapes, dtypes)

    for got, want in zip(pooled, registered, strict=True):
        assert got.shape == want.shape and torch.equal(got, want)
    assert pool.returned == len(pool.acquired)


def test_public_put_get_and_overwrite_mixed_values(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch)
    keys = ["matrix", "metadata", "scalar", "tokens"]
    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not matrix.is_contiguous()
    values = [matrix, {"step": 1, "labels": ["a", "b"]}, torch.tensor(17), torch.arange(5, dtype=torch.int16)]
    shapes = [tuple(matrix.shape), None, (), (5,)]
    dtypes = [torch.float32, None, torch.int64, torch.int16]

    meta = client.put(keys, values)

    assert meta[0] is None and meta[2:] == [None, None]
    assert meta[1] == {"packed_size": len(store.objects["metadata"])}
    assert store.registered == [] and store.unregistered == []
    first = client.get(keys, shapes, dtypes, meta)
    for index in [0, 2, 3]:
        assert first[index].dtype == values[index].dtype
        assert first[index].shape == values[index].shape
        assert torch.equal(first[index], values[index])
    assert first[1] == values[1]

    updated = [matrix + 10, {"step": 2, "labels": []}, torch.tensor(-3), values[3] + 4]
    store.registered.clear()
    store.unregistered.clear()
    meta = client.put(keys, updated)

    assert store.registered == [] and store.unregistered == []
    for _, ptrs, sizes in store.put_calls:
        for ptr, size in zip(ptrs, sizes, strict=True):
            assert pool.arena.data_ptr() <= ptr < ptr + size <= pool.arena.data_ptr() + pool.capacity
    second = client.get(keys, shapes, dtypes, meta)
    for index in [0, 2, 3]:
        assert torch.equal(second[index], updated[index])
        # Reusing pooled memory must not overwrite tensors returned by an earlier read.
        assert torch.equal(first[index], values[index])
    assert second[1] == updated[1]
    assert pool.active == {} and pool.returned == len(pool.acquired)


@pytest.mark.parametrize("fallback", ["missing_pool", "tcp", "exhausted"])
def test_put_falls_back_to_registered_buffers(store, monkeypatch, fallback):
    held = None
    if fallback == "missing_pool":
        monkeypatch.setattr(mcc, "MOONCAKE_BUFFER_POOL_IMPORTED", False)
        client = make_client()
        pool = None
    else:
        client, pool = make_client_and_pool(monkeypatch, protocol="tcp" if fallback == "tcp" else "rdma")
        if fallback == "exhausted":
            held = pool.acquire(pool.capacity)
    tensor = torch.arange(12, dtype=torch.int16).reshape(3, 4).t()
    assert not tensor.is_contiguous()
    values = [tensor, {"fallback": fallback}]

    try:
        meta = client.put(["tensor", "object"], values)

        assert store.registered and len(store.unregistered) == len(store.registered)
        assert store.objects["tensor"] == values[0].numpy().tobytes()
        assert meta == [None, {"packed_size": len(store.objects["object"])}]
        result = client.get(["tensor", "object"], [(4, 3), None], [torch.int16, None], meta)
        assert torch.equal(result[0], tensor) and result[1] == values[1]
        if pool is not None:
            assert pool.acquired == [pool.capacity]
    finally:
        if held is not None:
            held.release()


def test_put_splits_groups_and_registers_only_oversized_values(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)
    keys = ["big", "small0", "small1", "small2", "object"]
    values = [
        torch.arange(256, dtype=torch.float32),
        torch.arange(4),
        torch.arange(5),
        torch.arange(6),
        {"text": "x" * 1024},
    ]

    meta = client.put(keys, values)

    assert len(pool.acquired) == 2
    assert all(nbytes <= client._pool_share_bytes for nbytes in pool.acquired)
    assert store.registered and len(store.registered) == len(store.unregistered)
    for key, value in zip(keys[:4], values[:4], strict=True):
        assert store.objects[key] == value.numpy().tobytes()
    assert meta[4] == {"packed_size": len(store.objects["object"])}
    assert pool.active == {} and pool.returned == len(pool.acquired)


def test_put_repeated_key_preserves_input_order(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch, local_buffer_size=4096)
    values = [torch.arange(256, dtype=torch.float32), torch.arange(4, dtype=torch.float32)]

    assert client.put(["repeated", "repeated"], values) == [None, None]

    assert store.objects["repeated"] == values[-1].numpy().tobytes()
    assert pool.active == {}


def test_put_keeps_the_same_lease_during_partial_retry(store, monkeypatch):
    client, pool = make_client_and_pool(monkeypatch)
    upsert = store.batch_upsert_from

    def retry_once(keys, ptrs, sizes, config):
        assert len(pool.active) == 1 and pool.returned == 0
        results = upsert(keys, ptrs, sizes, config)
        return [-1, 0] if len(store.put_calls) == 1 else results

    monkeypatch.setattr(store, "batch_upsert_from", retry_once)
    tensors = [torch.arange(4), torch.arange(5)]

    client._put_tensors_thread_worker(["retry", "done"], tensors)

    initial, retry = store.put_calls
    assert retry == (["retry"], [initial[1][0]], [initial[2][0]])
    assert store.objects["retry"] == tensors[0].numpy().tobytes()
    assert pool.active == {} and pool.returned == len(pool.acquired) == 1
