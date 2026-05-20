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

"""Tests for the dict-unpack / restore feature in MooncakeStoreClient.

A dict value with at least one tensor is split: each sub-tensor rides the
tensor RDMA path under a synthetic sub-key; non-tensor entries are pickled
into a single uint8 blob that rides the same RDMA path as another sub-key.
The bytes pool (which silently returns ``b""`` under MB-scale GET pressure
because the internal allocator fails) is never touched for such dicts.
"""

import ctypes
import pickle
from typing import Any
from unittest import mock

import pytest
import torch

from transfer_queue.storage.clients import mooncake_client as mc
from transfer_queue.storage.clients.mooncake_client import (
    _DICT_SUBKEY_SEP,
    _TQ_DICT_UNPACK_KEY,
    _TQ_EXTRAS_SUBKEY,
    _dict_has_tensor,
    _expand_dict_slots_fn,
    _is_dict_unpack_meta,
    _unwrap_non_tensor,
)


def _make_dict_unpack_meta(
    tensor_keys,
    tensor_dtypes,
    tensor_shapes,
    extras_size: int = 0,
    key_order=None,
):
    """Helper: build a dict-unpack meta entry with the sentinel key set.

    If ``key_order`` is not provided, defaults to ``tensor_keys`` (i.e. no
    non-tensor entries in the original dict).
    """
    return {
        _TQ_DICT_UNPACK_KEY: True,
        "key_order": list(key_order) if key_order is not None else list(tensor_keys),
        "tensor_keys": list(tensor_keys),
        "tensor_dtypes": list(tensor_dtypes),
        "tensor_shapes": list(tensor_shapes),
        "extras_size": extras_size,
    }


# ---------------------------------------------------------------------------
# Pure helper tests — no storage needed
# ---------------------------------------------------------------------------
class TestDictHasTensor:
    def test_dict_of_tensors_is_true(self):
        v = {"a": torch.zeros(3), "b": torch.ones(2, 4)}
        assert _dict_has_tensor(v)

    def test_empty_dict_is_false(self):
        assert not _dict_has_tensor({})

    def test_non_dict_is_false(self):
        assert not _dict_has_tensor(torch.zeros(3))
        assert not _dict_has_tensor([1, 2, 3])
        assert not _dict_has_tensor("multi_modal_inputs")

    def test_mixed_dict_is_true(self):
        """At least one tensor present → unpack triggers; non-tensors get
        bundled into the extras blob."""
        v = {"a": torch.zeros(3), "b": "not a tensor"}
        assert _dict_has_tensor(v)

    def test_dict_with_no_tensors_is_false(self):
        """Falls back to the bytes path."""
        v = {"name": "alice", "age": 7}
        assert not _dict_has_tensor(v)

    def test_nested_dict_only_is_false(self):
        """A dict whose values are sub-dicts (not tensors) doesn't trigger."""
        v = {"a": {"inner": torch.zeros(3)}}
        assert not _dict_has_tensor(v)

    def test_non_tensor_data_wrapped_dict_is_true(self):
        """The KV storage manager hands the client NonTensorData-wrapped dicts;
        the dict-unpack path must unwrap them before classification."""
        try:
            from tensordict import NonTensorData
        except ImportError:
            pytest.skip("tensordict not installed in this env")
        v = NonTensorData({"a": torch.zeros(3), "b": torch.ones(2, 4)})
        assert _dict_has_tensor(v)
        unwrapped = _unwrap_non_tensor(v)
        assert isinstance(unwrapped, dict)
        assert unwrapped["a"].shape == (3,)
        assert unwrapped["b"].shape == (2, 4)

    def test_non_tensor_data_wrapped_non_dict_is_false(self):
        try:
            from tensordict import NonTensorData
        except ImportError:
            pytest.skip("tensordict not installed in this env")
        assert not _dict_has_tensor(NonTensorData("scalar-string"))
        assert not _dict_has_tensor(NonTensorData({}))  # empty dict


class TestExpandDictSlots:
    """Stand-alone test of the static helper used by ``get``."""

    def test_no_dict_slots_passes_through_unchanged(self):
        keys = ["0@input_ids", "1@input_ids", "0@reward"]
        shapes = [(10,), (10,), ()]
        dtypes = [torch.int64, torch.int64, torch.float32]
        cbm = [None, None, None]
        flat_keys, flat_shapes, flat_dtypes, recon = _expand_dict_slots_fn(keys, shapes, dtypes, cbm)
        assert flat_keys == keys
        assert flat_shapes == shapes
        assert flat_dtypes == dtypes
        assert recon == [("scalar", 0), ("scalar", 1), ("scalar", 2)]

    def test_pure_dict_slot_expands(self):
        meta = _make_dict_unpack_meta(
            tensor_keys=["pixel_values", "image_grid_thw"],
            tensor_dtypes=[torch.float32, torch.int64],
            tensor_shapes=[(1176, 3), (1, 3)],
        )
        keys = ["0@multi_modal_inputs"]
        shapes = [None]
        dtypes = [None]
        cbm = [meta]
        flat_keys, flat_shapes, flat_dtypes, recon = _expand_dict_slots_fn(keys, shapes, dtypes, cbm)
        assert flat_keys == [
            f"0@multi_modal_inputs{_DICT_SUBKEY_SEP}pixel_values",
            f"0@multi_modal_inputs{_DICT_SUBKEY_SEP}image_grid_thw",
        ]
        assert flat_shapes == [(1176, 3), (1, 3)]
        assert flat_dtypes == [torch.float32, torch.int64]
        assert recon == [
            (
                "dict",
                ["pixel_values", "image_grid_thw"],
                ["pixel_values", "image_grid_thw"],
                [0, 1],
                -1,
            )
        ]

    def test_dict_with_extras_appends_blob_subkey(self):
        """A mixed dict expands to tensor sub-keys + one extras sub-key (uint8)."""
        meta = _make_dict_unpack_meta(
            tensor_keys=["pixel_values"],
            tensor_dtypes=[torch.float32],
            tensor_shapes=[(2, 3)],
            extras_size=42,
            key_order=["pixel_values", "caption", "tag"],
        )
        keys = ["0@mmi"]
        cbm = [meta]
        flat_keys, flat_shapes, flat_dtypes, recon = _expand_dict_slots_fn(
            keys, [None], [None], cbm
        )
        assert flat_keys == [
            f"0@mmi{_DICT_SUBKEY_SEP}pixel_values",
            f"0@mmi{_DICT_SUBKEY_SEP}{_TQ_EXTRAS_SUBKEY}",
        ]
        assert flat_shapes == [(2, 3), [42]]
        assert flat_dtypes == [torch.float32, torch.uint8]
        assert recon == [
            (
                "dict",
                ["pixel_values", "caption", "tag"],
                ["pixel_values"],
                [0],
                1,
            )
        ]

    def test_mixed_slots_interleave(self):
        meta = _make_dict_unpack_meta(
            tensor_keys=["pixel_values"],
            tensor_dtypes=[torch.float32],
            tensor_shapes=[(4, 4)],
        )
        keys = ["0@input_ids", "0@multi_modal_inputs", "0@reward"]
        shapes = [(10,), None, ()]
        dtypes = [torch.int64, None, torch.float32]
        cbm = [None, meta, None]
        flat_keys, flat_shapes, flat_dtypes, recon = _expand_dict_slots_fn(keys, shapes, dtypes, cbm)
        assert flat_keys == [
            "0@input_ids",
            f"0@multi_modal_inputs{_DICT_SUBKEY_SEP}pixel_values",
            "0@reward",
        ]
        assert flat_dtypes == [torch.int64, torch.float32, torch.float32]
        assert recon == [
            ("scalar", 0),
            ("dict", ["pixel_values"], ["pixel_values"], [1], -1),
            ("scalar", 2),
        ]

    def test_sparse_columns_image_only_then_video_only(self):
        """Two samples with disjoint sub-key sets (image-only / video-only)."""
        meta_img = _make_dict_unpack_meta(
            tensor_keys=["pixel_values", "image_grid_thw"],
            tensor_dtypes=[torch.float32, torch.int64],
            tensor_shapes=[(8, 4), (1, 3)],
        )
        meta_vid = _make_dict_unpack_meta(
            tensor_keys=["pixel_values_videos", "video_grid_thw"],
            tensor_dtypes=[torch.float32, torch.int64],
            tensor_shapes=[(16, 4), (1, 3)],
        )
        keys = ["0@mmi", "1@mmi"]
        shapes = [None, None]
        dtypes = [None, None]
        cbm = [meta_img, meta_vid]
        flat_keys, flat_shapes, flat_dtypes, recon = _expand_dict_slots_fn(keys, shapes, dtypes, cbm)
        assert flat_keys == [
            f"0@mmi{_DICT_SUBKEY_SEP}pixel_values",
            f"0@mmi{_DICT_SUBKEY_SEP}image_grid_thw",
            f"1@mmi{_DICT_SUBKEY_SEP}pixel_values_videos",
            f"1@mmi{_DICT_SUBKEY_SEP}video_grid_thw",
        ]
        assert recon[0] == (
            "dict",
            ["pixel_values", "image_grid_thw"],
            ["pixel_values", "image_grid_thw"],
            [0, 1],
            -1,
        )
        assert recon[1] == (
            "dict",
            ["pixel_values_videos", "video_grid_thw"],
            ["pixel_values_videos", "video_grid_thw"],
            [2, 3],
            -1,
        )

    def test_mismatched_lengths_raise_in_helper(self):
        with pytest.raises((ValueError, IndexError)):
            _expand_dict_slots_fn(["a", "b"], [(1,)], [torch.int64], [None, None])


# ---------------------------------------------------------------------------
# In-memory mock MooncakeDistributedStore
# ---------------------------------------------------------------------------
class _FakeMooncakeStore:
    """A minimal in-memory stand-in for MooncakeDistributedStore.

    Implements just enough of the API surface for ``MooncakeStoreClient.put /
    get / clear`` in non-GDR mode to round-trip. Tensor put/get rides raw
    memory (ctypes.memmove); bytes path stores the pickled payload as-is.
    """

    def __init__(self):
        self._data: dict[str, bytes] = {}

    def setup(self, *args, **kwargs) -> int:
        return 0

    def close(self) -> None:
        self._data.clear()

    def register_buffer(self, ptr: int, size: int) -> int:
        return 0

    def unregister_buffer(self, ptr: int) -> int:
        return 0

    def batch_upsert_from(self, keys, ptrs, sizes, config=None):
        for k, p, s in zip(keys, ptrs, sizes, strict=True):
            if s == 0:
                self._data[k] = b""
            else:
                self._data[k] = ctypes.string_at(p, s)
        return [0] * len(keys)

    def upsert_batch(self, keys, values, config=None):
        for k, v in zip(keys, values, strict=True):
            self._data[k] = bytes(v)
        return 0

    def batch_get_into(self, keys, ptrs, sizes):
        ret = []
        for k, p, s in zip(keys, ptrs, sizes, strict=True):
            stored = self._data.get(k)
            if stored is None:
                ret.append(-1)
                continue
            if len(stored) != s:
                ret.append(-2)
                continue
            if s > 0:
                ctypes.memmove(p, stored, s)
            ret.append(0)
        return ret

    def get_batch(self, keys):
        return [self._data.get(k, b"") for k in keys]

    def batch_remove(self, keys, force=False):
        ret = []
        for k in keys:
            if self._data.pop(k, None) is None:
                ret.append(-704)
            else:
                ret.append(0)
        return ret


class _FakeReplicateConfig:
    """Stand-in for mooncake.store.ReplicateConfig."""

    with_hard_pin: bool = False


@pytest.fixture
def client():
    """Construct a MooncakeStoreClient backed by the in-memory fake store."""
    with (
        mock.patch.object(mc, "MOONCAKE_STORE_IMPORTED", True),
        mock.patch.object(mc, "MooncakeDistributedStore", _FakeMooncakeStore),
        mock.patch.object(mc, "ReplicateConfig", _FakeReplicateConfig),
    ):
        config: dict[str, Any] = {
            "local_hostname": "127.0.0.1",
            "metadata_server": "127.0.0.1:8080",
            "master_server_address": "127.0.0.1:8081",
        }
        c = mc.MooncakeStoreClient(config)
        try:
            yield c
        finally:
            c.close()


# ---------------------------------------------------------------------------
# End-to-end round-trip via the fake store
# ---------------------------------------------------------------------------
def _assert_tensors_equal(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert torch.equal(a, b)


def _assert_tensor_dicts_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert set(a.keys()) == set(b.keys())
    for k in a:
        _assert_tensors_equal(a[k], b[k])


class TestRoundTrip:
    def test_pure_dict_value_round_trip(self, client):
        v = {
            "pixel_values": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        }
        keys = ["0@multi_modal_inputs"]
        meta = client.put(keys, [v])
        assert isinstance(meta, list) and len(meta) == 1
        assert _is_dict_unpack_meta(meta[0])
        assert meta[0]["tensor_keys"] == ["pixel_values", "image_grid_thw"]
        assert meta[0]["key_order"] == ["pixel_values", "image_grid_thw"]
        assert meta[0]["extras_size"] == 0

        out = client.get(
            keys=keys,
            shapes=[None],
            dtypes=[None],
            custom_backend_meta=meta,
        )
        assert len(out) == 1
        _assert_tensor_dicts_equal(out[0], v)

    def test_mixed_dict_value_round_trip(self, client):
        """Dict with both tensors and non-tensors: tensors ride RDMA as
        sub-keys, non-tensors get pickled into a uint8 blob and ride RDMA
        as one more sub-key. The bytes pool is never touched."""
        v = {
            "pixel_values": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "caption": "a cat in a box",
            "image_grid_thw": torch.tensor([[1, 4, 3]], dtype=torch.int64),
            "tags": ["cat", "box"],
            "n_patches": 12,
        }
        keys = ["0@multi_modal_inputs"]
        meta = client.put(keys, [v])
        assert isinstance(meta, list) and len(meta) == 1
        m = meta[0]
        assert _is_dict_unpack_meta(m)
        assert m["tensor_keys"] == ["pixel_values", "image_grid_thw"]
        assert m["key_order"] == ["pixel_values", "caption", "image_grid_thw", "tags", "n_patches"]
        assert m["extras_size"] > 0

        # The extras sub-key must be present in the fake store under the
        # reserved name and carry uint8 payload.
        extras_key = f"0@multi_modal_inputs{_DICT_SUBKEY_SEP}{_TQ_EXTRAS_SUBKEY}"
        assert extras_key in client._store._data
        assert len(client._store._data[extras_key]) == m["extras_size"]
        # Confirm the bytes-pool path was NOT used for this mixed dict — only
        # the per-tensor and extras sub-keys should be in the store.
        expected_keys = {
            f"0@multi_modal_inputs{_DICT_SUBKEY_SEP}pixel_values",
            f"0@multi_modal_inputs{_DICT_SUBKEY_SEP}image_grid_thw",
            extras_key,
        }
        assert set(client._store._data.keys()) == expected_keys

        out = client.get(
            keys=keys,
            shapes=[None],
            dtypes=[None],
            custom_backend_meta=meta,
        )
        assert len(out) == 1
        restored = out[0]
        assert list(restored.keys()) == list(v.keys())  # insertion order preserved
        _assert_tensors_equal(restored["pixel_values"], v["pixel_values"])
        _assert_tensors_equal(restored["image_grid_thw"], v["image_grid_thw"])
        assert restored["caption"] == v["caption"]
        assert restored["tags"] == v["tags"]
        assert restored["n_patches"] == v["n_patches"]

    def test_pure_tensor_value_still_returns_none(self, client):
        t = torch.arange(8, dtype=torch.int64)
        meta = client.put(["0@input_ids"], [t])
        assert meta is None
        out = client.get(["0@input_ids"], shapes=[(8,)], dtypes=[torch.int64], custom_backend_meta=None)
        _assert_tensors_equal(out[0], t)

    def test_dict_with_no_tensors_falls_back_to_bytes_path(self, client):
        """A dict with zero tensors does NOT trigger unpack — it goes through
        the bytes path (small payloads only; large all-non-tensor dicts could
        still hit the upstream bug, but this is intentionally out of scope)."""
        meta = client.put(["0@reward_extra"], [{"misc": "not-a-tensor", "score": 0.5}])
        assert meta is None

    def test_mixed_batch_unpacks_only_dict_slots(self, client):
        mmi = {
            "pixel_values": torch.full((2, 3), 7.0, dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 4, 9]], dtype=torch.int64),
        }
        ids = torch.arange(6, dtype=torch.int64).reshape(1, 6)
        scalar = 0.42  # bytes path
        keys = ["0@multi_modal_inputs", "0@input_ids", "0@reward"]
        values = [mmi, ids, scalar]
        meta = client.put(keys, values)

        assert isinstance(meta, list) and len(meta) == 3
        assert _is_dict_unpack_meta(meta[0])
        assert meta[1] is None
        assert meta[2] is None

        out = client.get(
            keys=keys,
            shapes=[None, (1, 6), None],
            dtypes=[None, torch.int64, None],
            custom_backend_meta=meta,
        )
        _assert_tensor_dicts_equal(out[0], mmi)
        _assert_tensors_equal(out[1], ids)
        assert out[2] == scalar

    def test_sparse_columns_image_only_and_video_only(self, client):
        img = {
            "pixel_values": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "image_grid_thw": torch.tensor([[1, 2, 4]], dtype=torch.int64),
        }
        vid = {
            "pixel_values_videos": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "video_grid_thw": torch.tensor([[2, 2, 4]], dtype=torch.int64),
        }
        keys = ["0@mmi", "1@mmi"]
        meta = client.put(keys, [img, vid])
        assert isinstance(meta, list) and len(meta) == 2
        assert _is_dict_unpack_meta(meta[0])
        assert _is_dict_unpack_meta(meta[1])
        assert set(meta[0]["tensor_keys"]) == {"pixel_values", "image_grid_thw"}
        assert set(meta[1]["tensor_keys"]) == {"pixel_values_videos", "video_grid_thw"}

        out = client.get(
            keys=keys,
            shapes=[None, None],
            dtypes=[None, None],
            custom_backend_meta=meta,
        )
        _assert_tensor_dicts_equal(out[0], img)
        _assert_tensor_dicts_equal(out[1], vid)

    def test_clear_removes_dict_sub_keys(self, client):
        v = {
            "pixel_values": torch.zeros(2, 3, dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        }
        keys = ["0@mmi"]
        meta = client.put(keys, [v])
        assert f"0@mmi{_DICT_SUBKEY_SEP}pixel_values" in client._store._data
        assert f"0@mmi{_DICT_SUBKEY_SEP}image_grid_thw" in client._store._data

        client.clear(keys=keys, custom_backend_meta=meta)
        assert f"0@mmi{_DICT_SUBKEY_SEP}pixel_values" not in client._store._data
        assert f"0@mmi{_DICT_SUBKEY_SEP}image_grid_thw" not in client._store._data

    def test_clear_removes_extras_subkey(self, client):
        """A mixed dict's clear must remove both tensor sub-keys and the
        extras blob sub-key."""
        v = {
            "pixel_values": torch.zeros(2, 3, dtype=torch.float32),
            "caption": "hello",
        }
        keys = ["0@mmi"]
        meta = client.put(keys, [v])
        extras_key = f"0@mmi{_DICT_SUBKEY_SEP}{_TQ_EXTRAS_SUBKEY}"
        assert f"0@mmi{_DICT_SUBKEY_SEP}pixel_values" in client._store._data
        assert extras_key in client._store._data

        client.clear(keys=keys, custom_backend_meta=meta)
        assert f"0@mmi{_DICT_SUBKEY_SEP}pixel_values" not in client._store._data
        assert extras_key not in client._store._data

    def test_get_rejects_mismatched_custom_backend_meta_length(self, client):
        with pytest.raises(ValueError, match="custom_backend_meta"):
            client.get(
                keys=["a", "b"],
                shapes=[(1,), (1,)],
                dtypes=[torch.int64, torch.int64],
                custom_backend_meta=[None],
            )

    def test_clear_rejects_mismatched_custom_backend_meta_length(self, client):
        with pytest.raises(ValueError, match="custom_backend_meta"):
            client.clear(keys=["a", "b"], custom_backend_meta=[None])

    def test_non_tensor_data_wrapped_dict_round_trip(self, client):
        """End-to-end: the KV manager's _generate_values yields NonTensorData-
        wrapped dicts; client.put must still unpack and client.get must rebuild
        as a plain dict (mirroring the verl onethinker workload)."""
        try:
            from tensordict import NonTensorData
        except ImportError:
            pytest.skip("tensordict not installed in this env")

        raw = {
            "pixel_values": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "image_grid_thw": torch.tensor([[1, 4, 3]], dtype=torch.int64),
        }
        wrapped = NonTensorData(raw)
        keys = ["0@multi_modal_inputs"]
        meta = client.put(keys, [wrapped])
        assert isinstance(meta, list) and len(meta) == 1
        assert _is_dict_unpack_meta(meta[0]), (
            f"PUT should have unpacked the NonTensorData-wrapped dict, "
            f"but custom_backend_meta[0]={type(meta[0]).__name__}"
        )
        assert meta[0]["tensor_keys"] == ["pixel_values", "image_grid_thw"]

        out = client.get(
            keys=keys,
            shapes=[None],
            dtypes=[None],
            custom_backend_meta=meta,
        )
        assert len(out) == 1
        _assert_tensor_dicts_equal(out[0], raw)

    def test_meta_is_pickleable(self):
        """Dict-unpack meta must survive pickle round-trip with all fields intact."""
        m = _make_dict_unpack_meta(
            tensor_keys=["pixel_values", "image_grid_thw"],
            tensor_dtypes=[torch.float32, torch.int64],
            tensor_shapes=[(2, 3), (1, 3)],
            extras_size=17,
            key_order=["pixel_values", "caption", "image_grid_thw"],
        )
        restored = pickle.loads(pickle.dumps(m))
        assert _is_dict_unpack_meta(restored)
        assert restored["tensor_keys"] == m["tensor_keys"]
        assert restored["tensor_dtypes"] == m["tensor_dtypes"]
        assert restored["tensor_shapes"] == m["tensor_shapes"]
        assert restored["key_order"] == m["key_order"]
        assert restored["extras_size"] == m["extras_size"]

    def test_meta_survives_tq_msgpack_pipeline(self):
        """REGRESSION: an earlier implementation made the dict-unpack meta a
        ``@dataclass``, which msgspec auto-flattened into a typeless dict on
        the controller round-trip; ``isinstance`` checks then failed at GET
        and the bytes-pool fallback re-triggered the original bug. Using a
        plain ``dict`` with a sentinel key sidesteps the issue — dicts are a
        native msgpack map type, so the structure (including the
        ``_TQ_DICT_UNPACK_KEY`` marker and all fields) round-trips
        losslessly.
        """
        try:
            from transfer_queue.utils.serial_utils import decode, encode
        except ImportError:
            pytest.skip("transfer_queue serial_utils unavailable")
        m = _make_dict_unpack_meta(
            tensor_keys=["pixel_values", "image_grid_thw"],
            tensor_dtypes=[torch.float32, torch.int64],
            # NOTE: shapes are list-of-list (not tuple) by design — msgpack has
            # only one ordered-sequence type (array), so tuples on encode come
            # back as lists on decode. Production ``put`` writes lists too, so
            # before- and after-ZMQ representations match exactly.
            tensor_shapes=[[2, 3], [1, 3]],
            extras_size=23,
            key_order=["pixel_values", "caption", "image_grid_thw"],
        )
        # Mimic the controller round-trip: ZMQ body is a dict containing
        # custom_backend_meta as a per-global-idx mapping.
        payload = {"custom_backend_meta": {0: {"multi_modal_inputs": m}}}
        encoded = encode(payload)
        decoded = decode(encoded)
        restored = decoded["custom_backend_meta"][0]["multi_modal_inputs"]
        assert _is_dict_unpack_meta(restored), (
            f"dict-unpack meta sentinel lost after msgspec round-trip; got "
            f"type={type(restored).__name__} value={restored!r}"
        )
        assert restored["tensor_keys"] == m["tensor_keys"]
        assert restored["tensor_dtypes"] == m["tensor_dtypes"]
        assert restored["tensor_shapes"] == m["tensor_shapes"]
        assert restored["key_order"] == m["key_order"]
        assert restored["extras_size"] == m["extras_size"]
