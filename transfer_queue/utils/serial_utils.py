# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
# Copyright 2025 The vLLM project
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

# This implementation is inspired by https://github.com/vllm-project/vllm/blob/main/vllm/v1/serial_utils.py

import itertools
import logging
import os
import pickle
from collections.abc import Sequence
from inspect import isclass
from types import FunctionType
from typing import Any, Optional, TypeAlias

import cloudpickle
import numpy as np
import torch
import zmq
from msgspec import msgpack

from transfer_queue.utils.utils import get_env_bool

try:
    from torch.distributed.rpc.internal import _internal_rpc_pickler

    HAS_RPC_PICKLER = True
except ImportError:
    HAS_RPC_PICKLER = False

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3

TQ_ZERO_COPY_SERIALIZATION = get_env_bool("TQ_ZERO_COPY_SERIALIZATION", default=False) and HAS_RPC_PICKLER

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame
tensorenc = tuple[str, tuple[int, ...], int | memoryview]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.WARNING))


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.
    """

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: Optional[list[bytestr]] = None

    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.aux_buffers = None

    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_tensor(self, obj: torch.Tensor) -> tuple[str, list[tensorenc]] | tensorenc:
        assert self.aux_buffers is not None
        assert obj.device.type == "cpu", f"MsgpackEncoder only supports CPU tensors, got {obj.device}"
        assert not obj.is_sparse, "Sparse tensors are not supported yet for MsgpackEncoder."
        # view the tensor as a contiguous 1D array of bytes
        arr = obj.flatten().contiguous().view(torch.uint8).numpy()
        data = len(self.aux_buffers)
        self.aux_buffers.append(arr.data)
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self, t: Optional[Any] = None):
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(*args, ext_hook=self.ext_hook, dec_hook=self.dec_hook)
        self.aux_buffers: Sequence[bytestr] = ()

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])  # type: ignore[index]
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
        return obj

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        # Copy from inline representation, to decouple the memory storage
        # of the message from the original buffer. And also make Torch
        # not complain about a readonly memoryview.
        buffer = self.aux_buffers[data] if isinstance(data, int) else bytearray(data)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array and convert read-only buffer into writable bytearray
        arr = torch.frombuffer(bytearray(buffer), dtype=torch.uint8)
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)

        raise NotImplementedError(f"Extension type code {code} is not supported")


_encoder = MsgpackEncoder()
_decoder = MsgpackDecoder(torch.Tensor)


# Process tensors and collect nested tensor info efficiently
def _process_tensor(tensor: torch.Tensor) -> Any:
    if tensor.is_nested and tensor.layout == torch.strided:
        tensor_list = tensor.unbind()
        tensor_count = len(tensor_list)
        serialized_tensors = [_encoder.encode(inner_tensor) for inner_tensor in tensor_list]
        return tensor_count, serialized_tensors  # tensor_count may equal to 1 for single nested tensor
    else:
        return -1, [_encoder.encode(tensor)]  # use -1 to indicate regular single tensor


def serialization(obj: Any) -> list[bytestr]:
    """
    Serializes any object.

    Returns:
        list[bytestr]: If TQ_ZERO_COPY_SERIALIZATION is enabled, returns a list where the first element
        is the pickled bytes of the message, followed by the flattened serialized tensor parts as
        [pickled_bytes, <bytes>, |<bytes>, <memoryview>, |<bytes>, <memoryview>|...].
        From the third element, two elements is a group that will be used to restore a tensor.

        If TQ_ZERO_COPY_SERIALIZATION is disabled, returns a single-element list containing only the pickled bytes
        through pickle.
    """

    logger.debug(f"Serializing an obj with TQ_ZERO_COPY_SERIALIZATION={TQ_ZERO_COPY_SERIALIZATION}")

    if TQ_ZERO_COPY_SERIALIZATION:
        pickled_bytes, tensors = _internal_rpc_pickler.serialize(obj)

        # Use map to process all tensors in parallel-like fashion
        nested_tensor_info_and_serialized_tensors = list(map(_process_tensor, tensors))

        # Extract nested_tensor_info and flatten serialized tensors using itertools
        nested_tensor_info = np.array([info for info, _ in nested_tensor_info_and_serialized_tensors])
        double_layer_serialized_tensors: list[list[bytestr]] = list(
            itertools.chain.from_iterable(serialized for _, serialized in nested_tensor_info_and_serialized_tensors)
        )
        serialized_tensors: list[bytestr] = list(itertools.chain.from_iterable(double_layer_serialized_tensors))
        return [pickled_bytes, pickle.dumps(nested_tensor_info), *serialized_tensors]
    else:
        return [pickle.dumps(obj)]


def deserialization(data: list[bytestr] | bytestr) -> Any:
    """Deserialize any object from serialized data."""

    logger.debug(f"Deserializing an obj with TQ_ZERO_COPY_SERIALIZATION={TQ_ZERO_COPY_SERIALIZATION}")

    if TQ_ZERO_COPY_SERIALIZATION:
        if isinstance(data, list):
            # contain tensors
            pickled_bytes = data[0]
            nested_tensor_info = pickle.loads(data[1])
            serialized_tensors = data[2:]
            if len(serialized_tensors) % 2 != 0:
                # Note: data is a list of [pickled_bytes, <bytes>, |<bytes>, <memoryview>,
                # |<bytes>, <memoryview>|...].
                # From the third element, two elements is a group that will be used to restore a tensor.

                raise ValueError(
                    f"When TQ_ZERO_COPY_SERIALIZATION is enabled, input data should "
                    f"be a list containing an even number of elements, but got {len(data)}."
                )
            # deserializing each single tensor
            single_tensors: list[torch.Tensor] = [
                _decoder.decode(pair) for pair in zip(serialized_tensors[::2], serialized_tensors[1::2], strict=False)
            ]
        else:
            raise ValueError(
                f"When TQ_ZERO_COPY_SERIALIZATION is enabled, input data should be a list, but got {type(data)}."
            )

        tensor_nums = np.abs(nested_tensor_info).sum()
        if tensor_nums != len(single_tensors):
            raise ValueError(f"Expecting {tensor_nums} tensors, but got {len(single_tensors)}.")

        tensors = [None] * len(nested_tensor_info)
        current_idx = 0
        for i, tensor_num in enumerate(nested_tensor_info):
            if tensor_num == -1:
                tensors[i] = single_tensors[current_idx]
                current_idx += 1
            else:
                tensors[i] = torch.nested.as_nested_tensor(single_tensors[current_idx : current_idx + tensor_num])
                current_idx += tensor_num

        return _internal_rpc_pickler.deserialize(pickled_bytes, tensors)
    else:
        if isinstance(data, bytestr):
            return pickle.loads(data)
        elif isinstance(data, list):
            if len(data) > 1:
                raise ValueError(
                    f"When TQ_ZERO_COPY_SERIALIZATION is disabled, must have only 1 element in"
                    f" list for deserialization, but got {len(data)}."
                )
            return pickle.loads(data[0])
        else:
            raise ValueError(
                f"When TQ_ZERO_COPY_SERIALIZATION is disabled, input data should be a list of bytestr,"
                f" but got {type(data)}."
            )


def zero_copy_serialization_enabled() -> bool:
    """Check if zero-copy serialization is enabled."""
    return TQ_ZERO_COPY_SERIALIZATION
