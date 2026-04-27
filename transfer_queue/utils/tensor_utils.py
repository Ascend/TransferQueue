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

import logging
import operator
import os
from functools import reduce

import torch
from torch import Tensor

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


def allocate_empty_tensors(
    dtypes: list[torch.dtype], shapes: list[tuple]
) -> tuple[list[Tensor], list[int], list[int], list[int]]:
    """Allocate empty tensors, grouping same dtypes into shared memory blocks.

    Instead of allocating each tensor separately, this function groups tensors
    by their dtype and allocates one large contiguous memory block per dtype.
    Each tensor is then created as a view into this shared memory.

    Args:
        dtypes: List of torch dtypes for each tensor.
        shapes: List of shapes (tuples) for each tensor.

    Returns:
        A tuple containing:
            - List of tensors sharing memory within their dtype groups.
            - List of memory pointers (data_ptr) for each tensor.
            - List of base pointers for each allocated memory region (one per dtype).
            - List of total bytes for each allocated memory region (one per dtype).

    Example:
        >>> dtypes = [torch.float32, torch.float32, torch.int32, torch.float32]
        >>> shapes = [(10,), (20,), (5,), (15,)]
        >>> tensors, ptrs, region_ptrs, region_sizes = allocate_empty_tensors(dtypes, shapes)
        >>> # tensors[0], [1], [3] share the same dtype and memory block
    """
    assert len(dtypes) == len(shapes), "dtypes and shapes must have the same length"

    if len(dtypes) == 0:
        return [], [], [], []

    # Group indices by dtype
    dtype_groups: dict[torch.dtype, list[int]] = {}
    for i, dtype in enumerate(dtypes):
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append(i)

    tensor_list = [torch.empty(()) for _ in range(len(dtypes))]
    ptr_list = [0] * len(dtypes)
    region_ptrs: list[int] = []
    region_sizes: list[int] = []

    # For each dtype group, allocate one big tensor and create views
    for dtype, indices in dtype_groups.items():
        # Calculate total number of elements needed for this dtype
        total_elements = 0
        shape_info = []  # Store (index, shape, num_elements, offset)

        for idx in indices:
            shape = tuple(shapes[idx])
            num_elements = reduce(operator.mul, shape, 1)
            shape_info.append((idx, shape, num_elements, total_elements))
            total_elements += num_elements

        # Allocate one big contiguous memory block for this dtype
        big_tensor = torch.empty(total_elements, dtype=dtype)
        region_ptrs.append(big_tensor.data_ptr())
        region_sizes.append(big_tensor.nbytes)

        # Create views into the big tensor for each small tensor
        for idx, shape, num_elements, offset in shape_info:
            # Use as_strided to create a view with the correct shape
            small_tensor = big_tensor.as_strided(size=shape, stride=compute_stride(shape), storage_offset=offset)
            tensor_list[idx] = small_tensor
            ptr_list[idx] = small_tensor.data_ptr()

    return tensor_list, ptr_list, region_ptrs, region_sizes


def compute_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Compute stride for a contiguous row-major (C-style) tensor.

    Args:
        shape: The shape of the tensor.

    Returns:
        Stride tuple for contiguous storage.

    Example:
        >>> compute_stride((2, 3, 4))
        (12, 4, 1)
    """
    stride = []
    cumulative = 1
    # Iterate from last dimension to first
    for dim in reversed(shape):
        stride.append(cumulative)
        cumulative *= dim
    return tuple(reversed(stride))


def get_nbytes(dtypes, shapes) -> list[int]:
    """Calculate number of bytes according to tensor dtypes and shapes."""
    assert len(dtypes) == len(shapes)
    nbytes = []
    for i in range(len(dtypes)):
        elem_size = torch.tensor([], dtype=dtypes[i]).element_size()
        shape = tuple(shapes[i])
        numel = reduce(operator.mul, shape, 1)
        nbytes.append(elem_size * numel)

    return nbytes


def merge_contiguous_memory(ptrs: list[int], sizes: list[int]) -> tuple[list[int], list[int]]:
    """Merge contiguous memory regions to reduce register_buffer overhead

    Args:
        ptrs: List of memory pointers (starting addresses).
        sizes: List of memory region sizes corresponding to each pointer.

    Returns:
        A tuple of (merged_ptrs, merged_sizes) where contiguous regions
        have been merged into single regions.

    Example:
        >>> merge_contiguous_memory([0, 10, 30], [10, 20, 10])
        ([0, 30], [30, 10])

        >>> merge_contiguous_memory([0, 5, 20], [5, 5, 10])
        ([0, 20], [10, 10])
    """
    if len(ptrs) != len(sizes):
        raise ValueError("ptrs and sizes must have the same length")

    if not ptrs:
        return [], []

    # Create list of (ptr, size) pairs and sort by pointer address
    regions = sorted(zip(ptrs, sizes, strict=False), key=lambda x: x[0])

    merged_ptrs = []
    merged_sizes = []

    # Initialize with the first region
    current_ptr, current_size = regions[0]

    for ptr, size in regions[1:]:
        # Check if current region is contiguous with the next one
        # A region is contiguous if: ptr == current_ptr + current_size
        if ptr == current_ptr + current_size:
            # Merge: extend the current region
            current_size += size
        else:
            # Not contiguous: save the current region and start a new one
            merged_ptrs.append(current_ptr)
            merged_sizes.append(current_size)
            current_ptr, current_size = ptr, size

    # Add the last region
    merged_ptrs.append(current_ptr)
    merged_sizes.append(current_size)

    return merged_ptrs, merged_sizes
