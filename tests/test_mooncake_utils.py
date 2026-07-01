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

"""Unit tests for transfer_queue.utils.mooncake_utils and MooncakeStoreClient.clear()."""

import logging
import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from transfer_queue.utils.mooncake_utils import (
    GdrStaging,
    _aligned_offsets,
    chunk_subkeys,
    split_by_bytes,
)

_DEFAULT_ALIGN = 256


def _aligned(n: int) -> int:
    return (n + _DEFAULT_ALIGN - 1) // _DEFAULT_ALIGN * _DEFAULT_ALIGN


# ===========================================================================
# _aligned_offsets
# ===========================================================================


class TestAlignedOffsets:
    def test_empty(self):
        offsets, total = _aligned_offsets([])
        assert offsets == []
        assert total == 0

    def test_single_already_aligned(self):
        offsets, total = _aligned_offsets([256])
        assert offsets == [0]
        assert total == 256

    def test_single_unaligned(self):
        offsets, total = _aligned_offsets([100])
        assert offsets == [0]
        assert total == 256  # ceil(100/256)*256

    def test_multiple_unaligned(self):
        # 100 → pad to 256, 200 → pad to 256, 300 → pad to 512
        offsets, total = _aligned_offsets([100, 200, 300])
        assert offsets == [0, 256, 512]
        assert total == 512 + 512  # last slot: 300 → 512

    def test_exact_multiples(self):
        offsets, total = _aligned_offsets([256, 512, 256])
        assert offsets == [0, 256, 768]
        assert total == 1024


# ===========================================================================
# chunk_subkeys
# ===========================================================================


class TestChunkSubkeys:
    def test_fits_exactly(self):
        assert chunk_subkeys("k", 1024, 1024) == ["k"]

    def test_fits_below(self):
        assert chunk_subkeys("k", 100, 1024) == ["k"]

    def test_oversized_two_chunks(self):
        result = chunk_subkeys("k", 1025, 1024)
        assert result == ["k:c0", "k:c1"]

    def test_oversized_exact_multiple(self):
        result = chunk_subkeys("k", 2048, 1024)
        assert result == ["k:c0", "k:c1"]

    def test_oversized_three_chunks(self):
        result = chunk_subkeys("k", 2049, 1024)
        assert result == ["k:c0", "k:c1", "k:c2"]

    def test_key_format_preserved(self):
        result = chunk_subkeys("field@0", 3000, 1024)
        assert all(s.startswith("field@0:c") for s in result)
        assert [s.split(":c")[1] for s in result] == ["0", "1", "2"]

    def test_zero_bytes_fits(self):
        assert chunk_subkeys("k", 0, 1024) == ["k"]


# ===========================================================================
# split_by_bytes
# ===========================================================================


class TestSplitByBytes:
    def test_empty(self):
        assert split_by_bytes([], 1024) == []

    def test_single_fits(self):
        groups = split_by_bytes([100], 1024)
        assert groups == [[0]]

    def test_all_fit_one_group(self):
        # 100+100+100 aligned = 256*3 = 768 <= 1024
        groups = split_by_bytes([100, 100, 100], 1024)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_splits_into_two_groups(self):
        # 500 aligned=512, 500 aligned=512; 512+512=1024 fits; third pushes to new group
        groups = split_by_bytes([500, 500, 500], 1024)
        assert len(groups) == 2
        total_indices = sorted(idx for g in groups for idx in g)
        assert total_indices == [0, 1, 2]

    def test_oversized_singleton(self):
        # 2000 > 1024 → own group
        groups = split_by_bytes([2000], 1024)
        assert groups == [[0]]

    def test_oversized_in_mixed_batch(self):
        # [100, 2000, 100]: the 2000-byte tensor must be its own singleton group
        groups = split_by_bytes([100, 2000, 100], 1024)
        singleton_groups = [g for g in groups if len(g) == 1]
        multi_groups = [g for g in groups if len(g) > 1]
        oversized_idx = next(g[0] for g in singleton_groups if g[0] == 1)
        assert oversized_idx == 1
        assert sorted(multi_groups[0]) == [0, 2]

    def test_multiple_oversized_each_gets_singleton(self):
        groups = split_by_bytes([2000, 3000, 2000], 1024)
        assert len(groups) == 3
        assert all(len(g) == 1 for g in groups)

    def test_ascending_sort_prevents_fragmentation(self):
        # Without ascending sort, processing [200, 900, 200, 200] in order would produce 3 groups:
        #   group0=[0], group1=[1(900)], group2=[2,3]
        # With ascending sort the three 200-byte tensors pack together first, 900-byte goes last:
        #   group0=[0,2,3], group1=[1]  → only 2 groups
        #
        # buffer_size=1024; aligned sizes: 200→256, 900→1024
        # 256*3=768 ≤ 1024 (three smalls fit); adding 900's 1024 would overflow → separate group
        groups = split_by_bytes([200, 900, 200, 200], 1024)
        assert len(groups) == 2
        # The three small-tensor indices must share a group
        small_indices = {0, 2, 3}
        assert any(small_indices == set(g) for g in groups)
        # The large tensor must be alone
        assert any(g == [1] for g in groups)

    def test_alignment_boundary(self):
        # Two tensors each aligned to exactly buffer_size/2 should fit in one group
        # 512 bytes each → aligned=512; 512+512=1024 == buffer_size
        groups = split_by_bytes([512, 512], 1024)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1]

    def test_all_indices_covered(self):
        nbytes = [100, 200, 900, 50, 1200, 300]
        groups = split_by_bytes(nbytes, 1024)
        covered = sorted(idx for g in groups for idx in g)
        assert covered == list(range(len(nbytes)))


# ===========================================================================
# GdrStaging – lock only (no CUDA required)
# ===========================================================================


class TestGdrStagingLock:
    def test_acquire_blocks_concurrent_thread(self):
        staging = GdrStaging(1024 * 1024)
        entered = threading.Event()
        released = threading.Event()
        results: list[str] = []

        def holder():
            with staging.acquire():
                entered.set()
                released.wait(timeout=2.0)
                results.append("holder_done")

        def waiter():
            entered.wait(timeout=2.0)
            with staging.acquire():
                results.append("waiter_in")

        t1 = threading.Thread(target=holder)
        t2 = threading.Thread(target=waiter)
        t1.start()
        t2.start()

        entered.wait(timeout=1.0)
        time.sleep(0.05)
        assert "waiter_in" not in results  # still blocked

        released.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        assert results == ["holder_done", "waiter_in"]


# ===========================================================================
# GdrStaging – CUDA-dependent tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGdrStagingCuda:
    def _mock_store(self):
        store = MagicMock()
        store.register_buffer.return_value = None
        store.unregister_buffer.return_value = None
        return store

    def test_lazy_init_idempotent(self):
        store = self._mock_store()
        staging = GdrStaging(1024 * 1024)
        staging.lazy_init(store)
        staging.lazy_init(store)
        assert store.register_buffer.call_count == 1
        assert staging._initialized
        staging.close(store)

    def test_close_resets_state(self):
        store = self._mock_store()
        staging = GdrStaging(1024 * 1024)
        staging.lazy_init(store)
        staging.close(store)
        assert not staging._initialized
        assert store.unregister_buffer.call_count == 1

    def test_pack_unpack_roundtrip_single(self):
        store = self._mock_store()
        staging = GdrStaging(4 * 1024 * 1024)
        staging.lazy_init(store)
        try:
            original = torch.arange(1024, dtype=torch.float32, device="cuda")
            with staging.acquire():
                sub_ptrs, sizes = staging.pack([original])
                result = staging.unpack(sub_ptrs, sizes, [original.dtype], [tuple(original.shape)], original.device)
            assert torch.equal(result[0], original)
        finally:
            staging.close(store)

    def test_pack_unpack_roundtrip_multiple(self):
        store = self._mock_store()
        staging = GdrStaging(4 * 1024 * 1024)
        staging.lazy_init(store)
        try:
            tensors = [
                torch.randn(128, dtype=torch.float32, device="cuda"),
                torch.randint(0, 100, (64,), dtype=torch.int64, device="cuda"),
                torch.ones(256, dtype=torch.float16, device="cuda"),
            ]
            dtypes = [t.dtype for t in tensors]
            shapes = [tuple(t.shape) for t in tensors]
            with staging.acquire():
                sub_ptrs, sizes = staging.pack(tensors)
                results = staging.unpack(sub_ptrs, sizes, dtypes, shapes, torch.device("cuda"))
            for orig, got in zip(tensors, results, strict=True):
                assert torch.equal(orig, got)
        finally:
            staging.close(store)

    def test_pack_contiguous_required(self):
        # Non-contiguous tensor is contiguous()-ed by the caller before pack; staging
        # itself only receives contiguous tensors. Verify pack/unpack still works.
        store = self._mock_store()
        staging = GdrStaging(4 * 1024 * 1024)
        staging.lazy_init(store)
        try:
            base = torch.arange(256, dtype=torch.float32, device="cuda").reshape(16, 16)
            t = base[:, :8].contiguous()  # caller makes contiguous
            with staging.acquire():
                sub_ptrs, sizes = staging.pack([t])
                result = staging.unpack(sub_ptrs, sizes, [t.dtype], [tuple(t.shape)], t.device)
            assert torch.equal(result[0], t)
        finally:
            staging.close(store)


# ===========================================================================
# MooncakeStoreClient.clear() – sub-key expansion and cleanup
# ===========================================================================


def _make_clear_client(use_gdr: bool = True):
    """Construct a minimal MooncakeStoreClient-like object for clear() testing.

    Bypasses __init__ to avoid needing a real Mooncake store connection.
    """
    from transfer_queue.storage.clients.mooncake_client import MooncakeStoreClient

    client = object.__new__(MooncakeStoreClient)
    client._store = MagicMock()
    # Default: all batch_remove calls succeed
    client._store.batch_remove.side_effect = lambda keys, force: [0] * len(keys)
    client._gdr_staging = MagicMock() if use_gdr else None
    return client


class TestClear:
    def test_no_gdr_keys_pass_through(self):
        client = _make_clear_client(use_gdr=False)
        keys = ["a", "b", "c"]
        client.clear(keys)
        client._store.batch_remove.assert_called_once_with(keys, force=True)

    def test_gdr_all_normal_meta_no_expansion(self):
        client = _make_clear_client(use_gdr=True)
        keys = ["a", "b"]
        client.clear(keys, custom_backend_meta=[None, None])
        client._store.batch_remove.assert_called_once_with(["a", "b"], force=True)

    def test_gdr_single_chunked_key_expands(self):
        client = _make_clear_client(use_gdr=True)
        client.clear(["big"], custom_backend_meta=[{"n_chunks": 3}])
        client._store.batch_remove.assert_called_once_with(["big:c0", "big:c1", "big:c2"], force=True)

    def test_gdr_mixed_chunked_and_normal(self):
        client = _make_clear_client(use_gdr=True)
        client.clear(["normal", "big"], custom_backend_meta=[None, {"n_chunks": 2}])
        client._store.batch_remove.assert_called_once_with(["normal", "big:c0", "big:c1"], force=True)

    def test_gdr_multiple_chunked_keys_all_expanded(self):
        client = _make_clear_client(use_gdr=True)
        keys = ["a", "b", "c"]
        meta = [{"n_chunks": 2}, {"n_chunks": 3}, None]
        client.clear(keys, custom_backend_meta=meta)
        client._store.batch_remove.assert_called_once_with(["a:c0", "a:c1", "b:c0", "b:c1", "b:c2", "c"], force=True)

    def test_gdr_no_subkeys_leaked_after_clear(self):
        # All keys (original + sub-keys) must appear exactly once in batch_remove;
        # none should be silently dropped.
        client = _make_clear_client(use_gdr=True)
        keys = ["x", "y"]
        meta = [{"n_chunks": 4}, {"n_chunks": 2}]
        client.clear(keys, custom_backend_meta=meta)
        call_args = client._store.batch_remove.call_args
        removed = call_args[0][0]
        assert removed == ["x:c0", "x:c1", "x:c2", "x:c3", "y:c0", "y:c1"]
        # No original keys leaked through
        assert "x" not in removed
        assert "y" not in removed

    def test_gdr_meta_none_warns_and_uses_original_keys(self, caplog):
        client = _make_clear_client(use_gdr=True)
        keys = ["k0", "k1"]
        with caplog.at_level(logging.WARNING, logger="transfer_queue.storage.clients.mooncake_client"):
            client.clear(keys, custom_backend_meta=None)
        assert "custom_backend_meta is None" in caplog.text
        client._store.batch_remove.assert_called_once_with(keys, force=True)

    def test_no_gdr_meta_none_no_warning(self, caplog):
        client = _make_clear_client(use_gdr=False)
        with caplog.at_level(logging.WARNING):
            client.clear(["k0"], custom_backend_meta=None)
        assert "custom_backend_meta" not in caplog.text

    def test_error_code_triggers_log(self, caplog):
        client = _make_clear_client(use_gdr=False)
        client._store.batch_remove.side_effect = lambda keys, force: [-1] * len(keys)
        with caplog.at_level(logging.ERROR, logger="transfer_queue.storage.clients.mooncake_client"):
            client.clear(["k0"])
        assert "remove failed" in caplog.text

    def test_already_removed_code_704_is_silent(self, caplog):
        client = _make_clear_client(use_gdr=False)
        client._store.batch_remove.side_effect = lambda keys, force: [-704] * len(keys)
        with caplog.at_level(logging.ERROR):
            client.clear(["k0"])
        assert "remove failed" not in caplog.text

    def test_success_code_zero_is_silent(self, caplog):
        client = _make_clear_client(use_gdr=False)
        with caplog.at_level(logging.ERROR):
            client.clear(["k0"])
        assert "remove failed" not in caplog.text
