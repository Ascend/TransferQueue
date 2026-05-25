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

"""Unit tests for the packed-buffer batch serialization helpers in
``transfer_queue.utils.serial_utils``:

* ``calc_packed_size``
* ``pack_into`` / ``unpack_from``
* ``batch_encode_into``
* ``batch_decode_from``
"""

import numpy as np
import pytest
import torch

from transfer_queue.utils import serial_utils

# ============================================================================
# low-level: calc_packed_size + pack_into + unpack_from (raw bytes layer)
# ============================================================================


def test_calc_packed_size_then_pack_unpack_roundtrip():
    items = [b"hello", b"world!", b"x"]
    size = serial_utils.calc_packed_size(items)
    buf = bytearray(size)
    serial_utils.pack_into(buf, items)
    recovered = serial_utils.unpack_from(buf)
    assert [bytes(mv) for mv in recovered] == items


def test_pack_into_writes_only_within_its_slice():
    items = [b"alpha", b"beta", b"gamma"]
    sz = serial_utils.calc_packed_size(items)
    pad_before, pad_after = 17, 23
    big = bytearray(pad_before + sz + pad_after)
    serial_utils.pack_into(memoryview(big)[pad_before : pad_before + sz], items)

    assert all(b == 0 for b in big[:pad_before])
    assert all(b == 0 for b in big[pad_before + sz :])

    recovered = serial_utils.unpack_from(memoryview(big)[pad_before : pad_before + sz])
    assert [bytes(mv) for mv in recovered] == items


def test_unpack_from_zero_item_buffer():
    items: list[bytes] = []
    sz = serial_utils.calc_packed_size(items)
    buf = bytearray(sz)
    serial_utils.pack_into(buf, items)
    assert serial_utils.unpack_from(buf) == []


# ============================================================================
# batch_encode_into + batch_decode_from (high-level batch layer)
# ============================================================================


def _mooncake_alloc(sizes: list[int]) -> list[torch.Tensor]:
    """Single big torch.uint8 tensor sliced into N views (mooncake-style)."""
    big = torch.empty(sum(sizes), dtype=torch.uint8)
    buffers: list[torch.Tensor] = []
    offset = 0
    for s in sizes:
        buffers.append(big[offset : offset + s])
        offset += s
    return buffers


def _yuanrong_alloc(sizes: list[int]) -> list[bytearray]:
    """N independent bytearrays (yuanrong-style per-key buffer)."""
    return [bytearray(s) for s in sizes]


def _decode_from_returned(buffers, alloc_kind):
    if alloc_kind == "mooncake":
        return serial_utils.batch_decode_from(buffers)
    return serial_utils.batch_decode_from([bytes(b) for b in buffers])


def _roundtrip(values, alloc, alloc_kind, *, num_workers: int = 1):
    buffers, sizes = serial_utils.batch_encode_into(values, alloc, num_workers=num_workers)
    decoded = _decode_from_returned(buffers, alloc_kind)
    return decoded, buffers, sizes


# ---- structural: return shapes / alloc contract ----


def test_batch_encode_into_return_shapes():
    values = [{"x": 1}, "a string", torch.arange(8, dtype=torch.float32)]
    buffers, sizes = serial_utils.batch_encode_into(values, _mooncake_alloc)

    assert len(buffers) == len(values)
    assert len(sizes) == len(values)
    for b, s in zip(buffers, sizes, strict=True):
        assert b.nbytes == s


def test_batch_encode_into_allows_padded_buffers():
    """Alloc may return buffers larger than requested sizes; batch_sizes still
    reports the actual packed length, and the data round-trips correctly."""
    pad = 32

    def padded_alloc(sizes):
        return [bytearray(s + pad) for s in sizes]

    values = [b"alpha", {"k": "v"}, torch.arange(4, dtype=torch.float32)]
    buffers, sizes = serial_utils.batch_encode_into(values, padded_alloc)

    for b, s in zip(buffers, sizes, strict=True):
        assert len(b) == s + pad

    # decoding uses only the first `s` bytes; the pad tail is harmless
    decoded = serial_utils.batch_decode_from([bytes(b[:s]) for b, s in zip(buffers, sizes, strict=True)])
    _assert_equal_payloads(decoded, values)


# ---- semantic: encode → decode roundtrip preserves values ----


_ROUNDTRIP_PARAMS = [
    pytest.param([42, 3.14, "hello", b"bytes"], id="primitives"),
    pytest.param([{"a": 1, "b": [1, 2, 3]}, {"nested": {"k": "v"}}], id="nested-dicts"),
    pytest.param([torch.arange(10, dtype=torch.float32)], id="single-tensor"),
    pytest.param(
        [
            torch.arange(100, dtype=torch.float32),
            torch.randn(4, 4, dtype=torch.bfloat16),
            torch.zeros(3, 5, dtype=torch.int64),
        ],
        id="mixed-tensors",
    ),
    pytest.param(
        [np.arange(50, dtype=np.float64), np.ones((3, 3), dtype=np.int32)],
        id="numpy-arrays",
    ),
    pytest.param(
        [{"meta": "v1", "arr": torch.arange(5, dtype=torch.float32)}, [1, 2, "three"]],
        id="heterogeneous",
    ),
    pytest.param(
        [
            torch.randn(2, 3, 4, 5, dtype=torch.float32),
            torch.randn(2, 3, 4, 5, 6, dtype=torch.bfloat16),
        ],
        id="high-rank-tensors",
    ),
    pytest.param(
        [
            torch.nested.nested_tensor(
                [torch.arange(3, dtype=torch.float32), torch.arange(5, dtype=torch.float32)],
                layout=torch.strided,
            ),
            torch.nested.nested_tensor(
                [torch.randn(3, dtype=torch.bfloat16), torch.randn(5, dtype=torch.bfloat16)],
                layout=torch.strided,
            ),
            torch.nested.nested_tensor(
                [torch.arange(4, dtype=torch.float32), torch.arange(7, dtype=torch.float32)],
                layout=torch.jagged,
            ),
            torch.nested.nested_tensor(
                [torch.randn(4, dtype=torch.bfloat16), torch.randn(7, dtype=torch.bfloat16)],
                layout=torch.jagged,
            ),
        ],
        id="nested-tensors",
    ),
    pytest.param(
        [{"only": "one", "tensor": torch.arange(3, dtype=torch.float32)}],
        id="single-value",
    ),
]


@pytest.mark.parametrize("values", _ROUNDTRIP_PARAMS)
def test_batch_encode_decode_roundtrip_mooncake(values):
    decoded, *_ = _roundtrip(values, _mooncake_alloc, "mooncake")
    _assert_equal_payloads(decoded, values)


@pytest.mark.parametrize("values", _ROUNDTRIP_PARAMS)
def test_batch_encode_decode_roundtrip_yuanrong(values):
    decoded, *_ = _roundtrip(values, _yuanrong_alloc, "yuanrong")
    _assert_equal_payloads(decoded, values)


def test_batch_encode_decode_empty_list():
    calls = []

    def alloc(sizes):
        calls.append(list(sizes))
        return []

    buffers, sizes = serial_utils.batch_encode_into([], alloc)
    assert buffers == [] and sizes == []
    assert calls == [[]]
    assert serial_utils.batch_decode_from([]) == []


# ---- num_workers: parallel pack must produce identical bytes vs serial ----


@pytest.mark.parametrize("values", _ROUNDTRIP_PARAMS)
def test_batch_encode_into_parallel_matches_serial(values):
    serial_buffers, serial_sizes = serial_utils.batch_encode_into(values, _yuanrong_alloc, num_workers=1)
    par_buffers, par_sizes = serial_utils.batch_encode_into(values, _yuanrong_alloc, num_workers=4)

    assert serial_sizes == par_sizes
    assert [bytes(b) for b in serial_buffers] == [bytes(b) for b in par_buffers]


def test_batch_encode_into_parallel_roundtrip_many_objects():
    rng = np.random.default_rng(42)
    values = []
    for _ in range(64):
        n = int(rng.integers(1, 257))
        values.append(torch.from_numpy(rng.random(n).astype(np.float32)))

    decoded, *_ = _roundtrip(values, _yuanrong_alloc, "yuanrong", num_workers=8)
    _assert_equal_payloads(decoded, values)


# ============================================================================
# helpers
# ============================================================================


def _assert_equal_payloads(decoded, original):
    assert len(decoded) == len(original)
    for got, want in zip(decoded, original, strict=True):
        if isinstance(want, torch.Tensor):
            assert isinstance(got, torch.Tensor)
            assert got.dtype == want.dtype
            if want.is_nested:
                assert got.is_nested
                assert got.layout == want.layout
                got_subs = got.unbind()
                want_subs = want.unbind()
                assert len(got_subs) == len(want_subs)
                for g, w in zip(got_subs, want_subs, strict=True):
                    assert g.shape == w.shape
                    assert torch.equal(g, w)
            else:
                assert got.shape == want.shape
                assert torch.equal(got, want)
        elif isinstance(want, np.ndarray):
            assert isinstance(got, np.ndarray)
            assert got.dtype == want.dtype
            assert got.shape == want.shape
            assert np.array_equal(got, want)
        elif isinstance(want, dict):
            assert isinstance(got, dict)
            assert got.keys() == want.keys()
            for k in want:
                _assert_equal_payloads([got[k]], [want[k]])
        elif isinstance(want, list):
            assert isinstance(got, list)
            _assert_equal_payloads(got, want)
        else:
            assert got == want
