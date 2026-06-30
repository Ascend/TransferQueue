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

"""Utilities for Mooncake GDR transfers: persistent staging buffer and sub-key helpers."""

import contextlib
import threading
from math import ceil
from typing import Any

import torch

_DEFAULT_ALIGN = 256


def _aligned_offsets(sizes: list[int], align: int = _DEFAULT_ALIGN) -> tuple[list[int], int]:
    """Lay sizes out back-to-back with ``align``-byte alignment; return (offsets, total)."""
    offsets: list[int] = []
    off = 0
    for sz in sizes:
        offsets.append(off)
        off += (sz + align - 1) // align * align
    return offsets, off


def chunk_subkeys(key: str, nbytes: int, buffer_size: int) -> list[str]:
    """Return the list of storage keys for a tensor of ``nbytes`` under ``buffer_size``.

    - nbytes <= buffer_size: returns [key]  (no chunking)
    - nbytes >  buffer_size: returns ["{key}:c0", ..., "{key}:c{n-1}"]
    """
    if nbytes <= buffer_size:
        return [key]
    n = ceil(nbytes / buffer_size)
    return [f"{key}:c{i}" for i in range(n)]


def split_by_bytes(nbytes: list[int], buffer_size: int) -> list[list[int]]:
    """Partition tensor indices into groups that fit within the staging buffer.

    Args:
        nbytes:      Per-tensor byte counts (same order as the tensor list being transferred).
        buffer_size: Capacity of the staging buffer in bytes.

    Returns:
        List of groups, each group is a list of indices into ``nbytes``.
        Every group's 256-byte-aligned cumulative size fits within ``buffer_size``.
        A tensor whose nbytes > buffer_size gets its own singleton group;
        the caller handles it via the chunked ``:c{i}`` sub-key path.
        Indices are processed in ascending size order so that large tensors do not
        fragment the packing of small tensors.

    Call this before acquiring the staging-buffer lock; it does only integer arithmetic.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    current_total = 0

    for i in sorted(range(len(nbytes)), key=lambda i: nbytes[i]):
        nb = nbytes[i]
        aligned = (nb + _DEFAULT_ALIGN - 1) // _DEFAULT_ALIGN * _DEFAULT_ALIGN
        if nb > buffer_size:
            if current:
                groups.append(current)
                current, current_total = [], 0
            groups.append([i])
        elif current and current_total + aligned > buffer_size:
            groups.append(current)
            current, current_total = [i], aligned
        else:
            current.append(i)
            current_total += aligned

    if current:
        groups.append(current)
    return groups


class GdrStaging:
    """Process-level persistent CUDA staging buffer for GDR transfers.

    One cudaMalloc buffer, registered once for the process lifetime.
    All callers (PUT and GET) serialize through a single lock.
    """

    def __init__(self, buffer_size_bytes: int) -> None:
        self._size = buffer_size_bytes
        self._ptr: int = 0
        self._lock = threading.Lock()
        self._rt: Any = None
        self._stream: torch.cuda.Stream | None = None
        self._initialized = False

    def lazy_init(self, store) -> None:
        """Import cuda-python, cudaMalloc, register_buffer; idempotent."""
        if self._initialized:
            return
        try:
            from cuda import cuda as cuda_driver
            from cuda import cudart
        except ImportError as exc:
            raise ImportError(
                "cuda-python is required for GDR transfers; install with: pip install 'TransferQueue[mooncake]'"
            ) from exc
        self._rt = cudart
        err, device_ordinal = cudart.cudaGetDevice()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGetDevice() failed: {err.name}")
        _, supported = cuda_driver.cuDeviceGetAttribute(
            cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
            device_ordinal,
        )
        if not supported:
            raise RuntimeError(
                f"GPUDirect RDMA is not supported on device {device_ordinal}. "
                "Please ensure the device supports GDR, or set use_gdr=False."
            )
        err, ptr = cudart.cudaMalloc(self._size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc({self._size}) failed: {err.name}")
        self._ptr = ptr
        store.register_buffer(self._ptr, self._size)
        self._stream = torch.cuda.Stream()
        self._initialized = True

    def close(self, store) -> None:
        """store.unregister_buffer + cudaFree. Called by MooncakeStoreClient.close()."""
        if self._initialized:
            store.unregister_buffer(self._ptr)
            (err,) = self._rt.cudaFree(self._ptr)
            if err != self._rt.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaFree(0x{self._ptr:x}) failed: {err.name}")
            self._initialized = False

    @contextlib.contextmanager
    def acquire(self):
        """Context manager that holds the internal mutex for the duration of one transfer."""
        with self._lock:
            yield

    def memcpy_d2d_async(self, dst: int, src: int, nbytes: int) -> None:
        """Enqueue a D2D async copy on the internal stream; call synchronize() when done."""
        assert self._stream is not None
        rt = self._rt
        (err,) = rt.cudaMemcpyAsync(
            dst, src, nbytes, rt.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self._stream.cuda_stream
        )
        if err != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpyAsync D2D failed: {err.name}")

    def memcpy_h2d_async(self, dst: int, src: int, nbytes: int) -> None:
        """Enqueue a H2D async copy on the internal stream; call synchronize() when done."""
        assert self._stream is not None
        rt = self._rt
        (err,) = rt.cudaMemcpyAsync(
            dst, src, nbytes, rt.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream.cuda_stream
        )
        if err != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpyAsync H2D failed: {err.name}")

    def synchronize(self) -> None:
        """Synchronize the internal CUDA stream."""
        assert self._stream is not None
        self._stream.synchronize()

    def pack(self, tensors: list[torch.Tensor]) -> tuple[list[int], list[int]]:
        """Memcpy each tensor into the staging buffer at 256-byte aligned offsets.

        Supports both CPU (H2D) and CUDA (D2D) tensors transparently.
        Caller must hold the lock (call inside acquire()).
        Total packed size must fit in buffer_size (caller must ensure this).
        Returns (sub_ptrs, sizes).
        """
        sizes = [t.nbytes for t in tensors]
        offsets, _ = _aligned_offsets(sizes)
        for t, off in zip(tensors, offsets, strict=True):
            if t.is_cuda:
                self.memcpy_d2d_async(self._ptr + off, t.data_ptr(), t.nbytes)
            else:
                self.memcpy_h2d_async(self._ptr + off, t.data_ptr(), t.nbytes)
        self.synchronize()
        sub_ptrs = [self._ptr + off for off in offsets]
        return sub_ptrs, sizes

    def unpack(
        self,
        sub_ptrs: list[int],
        sizes: list[int],
        dtypes: list[torch.dtype],
        shapes: list[tuple],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """D2D memcpyAsync from each sub_ptr in staging into fresh tensors on device.

        Caller must hold the lock (call inside acquire()).
        """
        out: list[torch.Tensor] = []
        for sub_ptr, sz, dt, shp in zip(sub_ptrs, sizes, dtypes, shapes, strict=True):
            t = torch.empty(tuple(shp), dtype=dt, device=device)
            self.memcpy_d2d_async(t.data_ptr(), sub_ptr, sz)
            out.append(t)
        self.synchronize()
        return out

    @property
    def ptr(self) -> int:
        """Raw CUDA device pointer to the start of the staging buffer."""
        return self._ptr

    @property
    def size(self) -> int:
        """Capacity of the staging buffer in bytes."""
        return self._size
