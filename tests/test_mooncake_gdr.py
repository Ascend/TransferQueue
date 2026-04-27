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

"""Tests for GDR (GPU Direct RDMA) support in MooncakeStoreClient.

Unit tests for TensorMetadata pack/unpack and dtype mapping can run anywhere.
Integration tests that need a live MooncakeStore + CUDA are marked accordingly.
"""

import ctypes
import struct

import pytest
import torch

from transfer_queue.storage.clients.mooncake_client import (
    TENSOR_METADATA_SIZE,
    _MC_ENUM_TO_TORCH_DTYPE,
    _TENSOR_METADATA_FMT,
    _TORCH_DTYPE_TO_MC_ENUM,
    _pack_tensor_metadata,
    _unpack_tensor_metadata,
)


# ---------------------------------------------------------------------------
# TensorMetadata binary layout tests
# ---------------------------------------------------------------------------
class TestTensorMetadataLayout:
    """Verify that Python packing matches the C++ TensorMetadata struct."""

    def test_struct_size_is_40_bytes(self):
        assert struct.calcsize(_TENSOR_METADATA_FMT) == 40
        assert TENSOR_METADATA_SIZE == 40

    def test_field_offsets(self):
        """dtype at 0, ndim at 4, shape[0] at 8."""
        packed = _pack_tensor_metadata(torch.float16, (10, 20))
        # dtype = 11 (FLOAT16), ndim = 2
        dtype_val = struct.unpack_from("<i", packed, 0)[0]
        ndim_val = struct.unpack_from("<i", packed, 4)[0]
        shape0 = struct.unpack_from("<q", packed, 8)[0]
        shape1 = struct.unpack_from("<q", packed, 16)[0]
        shape2 = struct.unpack_from("<q", packed, 24)[0]
        shape3 = struct.unpack_from("<q", packed, 32)[0]
        assert dtype_val == 11
        assert ndim_val == 2
        assert shape0 == 10
        assert shape1 == 20
        assert shape2 == -1
        assert shape3 == -1


# ---------------------------------------------------------------------------
# Dtype enum mapping tests
# ---------------------------------------------------------------------------
class TestDtypeEnumMapping:
    """Ensure torch ↔ Mooncake enum mapping is complete and consistent."""

    @pytest.mark.parametrize(
        "dtype,expected_enum",
        [
            (torch.float32, 0),
            (torch.float64, 1),
            (torch.int8, 2),
            (torch.uint8, 3),
            (torch.int16, 4),
            (torch.uint16, 5),
            (torch.int32, 6),
            (torch.uint32, 7),
            (torch.int64, 8),
            (torch.uint64, 9),
            (torch.bool, 10),
            (torch.float16, 11),
            (torch.bfloat16, 12),
        ],
    )
    def test_dtype_to_enum(self, dtype, expected_enum):
        assert _TORCH_DTYPE_TO_MC_ENUM[dtype] == expected_enum

    def test_roundtrip_all_dtypes(self):
        for dtype, enum_val in _TORCH_DTYPE_TO_MC_ENUM.items():
            assert _MC_ENUM_TO_TORCH_DTYPE[enum_val] is dtype

    @pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 not available")
    def test_float8_e4m3fn_mapping(self):
        assert _TORCH_DTYPE_TO_MC_ENUM[torch.float8_e4m3fn] == 13

    @pytest.mark.skipif(not hasattr(torch, "float8_e5m2"), reason="float8 not available")
    def test_float8_e5m2_mapping(self):
        assert _TORCH_DTYPE_TO_MC_ENUM[torch.float8_e5m2] == 14


# ---------------------------------------------------------------------------
# pack / unpack round-trip tests
# ---------------------------------------------------------------------------
class TestPackUnpack:

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.bool],
    )
    @pytest.mark.parametrize(
        "shape",
        [(), (1024,), (32, 128), (2, 3, 4), (2, 3, 4, 5)],
    )
    def test_roundtrip(self, dtype, shape):
        packed = _pack_tensor_metadata(dtype, shape)
        assert len(packed) == TENSOR_METADATA_SIZE
        out_dtype, out_shape = _unpack_tensor_metadata(packed)
        assert out_dtype is dtype
        assert out_shape == shape

    def test_scalar_tensor_metadata(self):
        """ndim=0 → all shape slots are -1."""
        packed = _pack_tensor_metadata(torch.float32, ())
        _, ndim, *shape_raw = struct.unpack(_TENSOR_METADATA_FMT, packed)
        assert ndim == 0
        assert shape_raw == [-1, -1, -1, -1]

    def test_rejects_5d_tensor(self):
        with pytest.raises(ValueError, match="at most 4 dimensions"):
            _pack_tensor_metadata(torch.float32, (1, 2, 3, 4, 5))

    def test_rejects_unsupported_dtype(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _pack_tensor_metadata(torch.complex64, (10,))

    def test_unpack_rejects_unknown_enum(self):
        bad_data = struct.pack(_TENSOR_METADATA_FMT, 999, 1, 10, -1, -1, -1)
        with pytest.raises(ValueError, match="Unknown dtype enum"):
            _unpack_tensor_metadata(bad_data)


# ---------------------------------------------------------------------------
# Batch metadata buffer tests (ctypes layout used by GDR path)
# ---------------------------------------------------------------------------
class TestBatchMetadataBuffer:
    """Verify that packing multiple metadata entries into a contiguous
    bytearray produces correct per-slot pointers."""

    def test_contiguous_buffer_packing(self):
        dtypes = [torch.float32, torch.bfloat16, torch.int64]
        shapes = [(100,), (32, 64), (2, 3, 4)]
        n = len(dtypes)

        meta_buf = bytearray(TENSOR_METADATA_SIZE * n)
        meta_ctypes = (ctypes.c_char * len(meta_buf)).from_buffer(meta_buf)
        meta_ptr = ctypes.addressof(meta_ctypes)

        for j, (dtype, shape) in enumerate(zip(dtypes, shapes)):
            offset = j * TENSOR_METADATA_SIZE
            meta_bytes = _pack_tensor_metadata(dtype, shape)
            meta_buf[offset : offset + TENSOR_METADATA_SIZE] = meta_bytes

        # Read back each slot and verify
        for j, (dtype, shape) in enumerate(zip(dtypes, shapes)):
            offset = j * TENSOR_METADATA_SIZE
            slot_bytes = bytes(meta_buf[offset : offset + TENSOR_METADATA_SIZE])
            out_dtype, out_shape = _unpack_tensor_metadata(slot_bytes)
            assert out_dtype is dtype
            assert out_shape == shape

        # Verify pointer arithmetic is consistent
        assert meta_ptr > 0
        for j in range(n):
            slot_ptr = meta_ptr + j * TENSOR_METADATA_SIZE
            assert slot_ptr == meta_ptr + j * 40


# ---------------------------------------------------------------------------
# GPU tensor handling (requires CUDA)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGDRTensorHandling:

    def test_gpu_tensor_data_ptr_nonzero(self):
        t = torch.randn(1024, device="cuda:0")
        assert t.data_ptr() != 0

    def test_contiguous_gpu_tensor(self):
        t = torch.randn(10, 10, device="cuda:0").t()
        assert not t.is_contiguous()
        t_c = t.contiguous()
        assert t_c.is_contiguous()
        assert t_c.data_ptr() != 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_metadata_for_gpu_tensor(self, dtype):
        t = torch.zeros(32, 64, dtype=dtype, device="cuda:0")
        packed = _pack_tensor_metadata(t.dtype, tuple(t.shape))
        out_dtype, out_shape = _unpack_tensor_metadata(packed)
        assert out_dtype is dtype
        assert out_shape == (32, 64)

    def test_pre_allocate_gpu_tensor_for_get(self):
        """Simulate the GET path: pre-allocate, then verify data_ptr is usable."""
        shape = [16, 128]
        dtype = torch.bfloat16
        t = torch.empty(shape, dtype=dtype, device="cuda:0")
        assert t.data_ptr() != 0
        assert t.numel() * t.element_size() == 16 * 128 * 2
