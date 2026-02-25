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


import pickle
import warnings
from collections.abc import Sequence
from contextvars import ContextVar
from types import FunctionType
from typing import Any, TypeAlias

import cloudpickle
import numpy as np
import torch
import zmq
from msgspec import msgpack
from tensordict import TensorDictBase

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_TENSOR = 3  # For tensor with buffer reference
CUSTOM_TYPE_NESTED_TENSOR = 4  # For nested tensor (strided or jagged)
CUSTOM_TYPE_BATCHMETA = 5  # For BatchMeta serialization

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame

# Ignore warnings about non-writable buffers from torch.frombuffer. Upper codes will ensure
# the tensors are writable to users.
warnings.filterwarnings(action="ignore", message=r"The given buffer is not writable*", category=UserWarning)

# ContextVar for thread/coroutine-safe buffer storage during serialization/deserialization
# This enables the global _encoder/_decoder instances to be safely used across threads
_encoder_aux_buffers: ContextVar[list[bytestr] | None] = ContextVar("encoder_aux_buffers", default=None)
_decoder_aux_buffers: ContextVar[Sequence[bytestr] | None] = ContextVar("decoder_aux_buffers", default=None)


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    This implementation uses ContextVar for thread-safe buffer storage,
    allowing the global encoder instance to be safely used across multiple
    threads and async coroutines.

    """

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)

    @property
    def aux_buffers(self) -> list[bytestr]:
        """Get the current context's aux_buffers."""
        buffers = _encoder_aux_buffers.get()
        assert buffers is not None, "aux_buffers accessed outside of encode() context"
        return buffers

    def encode(self, obj: Any) -> Sequence[bytestr]:
        """Encode a given object to a byte array."""
        # Pre-process to convert BatchMeta to Ext; msgspec auto-serializes dataclasses and won't call enc_hook for them.
        obj = self._preprocess_for_batchmeta(obj)

        bufs: list[bytestr] = [b""]
        token = _encoder_aux_buffers.set(bufs)
        try:
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            _encoder_aux_buffers.reset(token)

    def _preprocess_for_batchmeta(self, obj: Any) -> Any:
        """Recursively preprocess object to convert BatchMeta to Ext.

        This is necessary because msgspec auto-serializes dataclasses and
        won't call enc_hook for them.
        """
        from transfer_queue.metadata import BatchMeta

        if isinstance(obj, BatchMeta):
            return self._encode_batchmeta(obj)
        elif isinstance(obj, dict):
            return {k: self._preprocess_for_batchmeta(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._preprocess_for_batchmeta(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._preprocess_for_batchmeta(item) for item in obj)
        return obj

    def enc_hook(self, obj: Any) -> Any:
        """Custom encoding hook for types msgspec doesn't natively support.

        For zero-copy tensor serialization, we need to handle:
        - torch.Tensor: Extract buffer, store metadata
        - TensorDict: Convert to dict structure for recursive processing
        - numpy.ndarray: Convert to tensor for unified handling

        Note: BatchMeta is handled by _preprocess_for_batchmeta() before encode() is called,
        so it will never reach this hook.
        """
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # Handle TensorDict explicitly for recursive zero-copy
        if isinstance(obj, TensorDictBase):
            return self._encode_tensordict(obj)

        # Handle numpy arrays by converting to tensor
        # Only numeric dtypes are supported by torch.from_numpy:
        # f=float, i=signed int, u=unsigned int, b=bool, c=complex
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in ("f", "i", "u", "b", "c"):
                try:
                    return self._encode_tensor(torch.from_numpy(obj))
                except (TypeError, RuntimeError):
                    # Fallback to pickle for unsupported dtypes (e.g., float16 on some platforms)
                    pass
            # For object arrays, strings, or other unsupported types, use pickle
            return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        if isinstance(obj, FunctionType):
            # cloudpickle for functions/methods
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        # Fallback to pickle for unknown types
        return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_batchmeta(self, obj: Any) -> msgpack.Ext:
        """Encode BatchMeta for serialization.

        BatchMeta is small, so we serialize it via pickle (which handles torch.dtype natively).
        """
        meta_dict = obj.to_dict()
        return msgpack.Ext(CUSTOM_TYPE_BATCHMETA, pickle.dumps(meta_dict, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_tensordict(self, obj: Any) -> dict:
        """Convert TensorDict to a dict structure for recursive msgpack processing.

        This allows msgpack to recursively call enc_hook for each tensor inside,
        enabling zero-copy serialization of nested tensors.
        """
        # Convert to dict, preserving structure
        # TensorDict.to_dict() returns nested dicts with tensors as leaves
        data_dict = dict(obj.items())

        # Return a marked dict that decoder will recognize
        return {
            "__tq_tensordict__": True,
            "batch_size": list(obj.batch_size),  # torch.Size -> list for msgpack
            "data": data_dict,
        }

    def _encode_tensor(self, obj: torch.Tensor) -> msgpack.Ext:
        """Encode tensor with zero-copy buffer extraction.

        Features:
        - Auto GPU->CPU conversion
        - Auto contiguous conversion
        - Direct memoryview extraction via uint8 view (for BFloat16 support)
        - Nested tensors: unbind and serialize each sub-tensor with zero-copy

        Returns Ext type so decoding goes through ext_hook (which has buffer access).
        """
        assert len(self.aux_buffers) > 0

        # Handle nested tensors (strided or jagged) via unbind
        if obj.is_nested:
            return self._encode_nested_tensor(obj)

        return self._encode_regular_tensor(obj)

    def _encode_nested_tensor(self, obj: torch.Tensor) -> msgpack.Ext:
        """Encode nested tensor by unbinding into sub-tensors for zero-copy."""
        # Unbind nested tensor into list of regular tensors
        sub_tensors = obj.unbind()

        # Encode each sub-tensor with zero-copy
        encoded_sub_tensors = []
        for t in sub_tensors:
            # Get tensor metadata (dtype, shape, buffer_idx)
            meta = self._encode_regular_tensor_meta(t)
            encoded_sub_tensors.append(meta)

        # Pack: layout type + list of tensor metas
        layout = "jagged" if obj.layout == torch.jagged else "strided"
        nested_meta = {
            "layout": layout,
            "tensors": encoded_sub_tensors,
        }
        return msgpack.Ext(CUSTOM_TYPE_NESTED_TENSOR, pickle.dumps(nested_meta, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_regular_tensor_meta(self, obj: torch.Tensor) -> tuple:
        """Encode a regular tensor and return its metadata tuple."""
        # Handle non-contiguous tensors

        if not obj.is_contiguous():
            obj = obj.contiguous()

        # Handle GPU tensors
        if obj.device.type != "cpu":
            obj = obj.cpu()

        # Zero-copy buffer extraction via uint8 view
        arr = obj.flatten().view(torch.uint8).numpy()
        buf = memoryview(arr)
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)

        dtype = str(obj.dtype).removeprefix("torch.")
        return (dtype, tuple(obj.shape), idx)

    def _encode_regular_tensor(self, obj: torch.Tensor) -> msgpack.Ext:
        """Encode a regular (non-nested) tensor with zero-copy."""
        # Handle non-contiguous tensors

        if not obj.is_contiguous():
            obj = obj.contiguous()

        # Handle GPU tensors
        if obj.device.type != "cpu":
            obj = obj.cpu()

        if obj.is_sparse:
            # Sparse tensors fallback to pickle
            return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

        # Note: view(uint8) is a byte-level view, NOT a value conversion.
        arr = obj.flatten().view(torch.uint8).numpy()
        buf = memoryview(arr)
        idx = len(self.aux_buffers)
        self.aux_buffers.append(buf)

        # Pack tensor metadata as Ext type
        dtype = str(obj.dtype).removeprefix("torch.")
        meta = (dtype, tuple(obj.shape), idx)
        return msgpack.Ext(CUSTOM_TYPE_TENSOR, pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    This implementation uses ContextVar for thread-safe buffer storage,
    allowing the global decoder instance to be safely used across multiple
    threads and async coroutines.
    """

    def __init__(self):
        self.decoder = msgpack.Decoder(ext_hook=self.ext_hook)

    @property
    def aux_buffers(self) -> Sequence[bytestr]:
        """Get the current context's aux_buffers."""
        buffers = _decoder_aux_buffers.get()
        assert buffers is not None, "aux_buffers accessed outside of decode() context"
        return buffers

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        """Decode a list of bytes."""
        if isinstance(bufs, bytestr):
            result = self.decoder.decode(bufs)
        else:
            token = _decoder_aux_buffers.set(bufs)
            try:
                result = self.decoder.decode(bufs[0])  # type: ignore[index]
            finally:
                _decoder_aux_buffers.reset(token)

        # Post-process to reconstruct TensorDict from marked dicts
        return self._reconstruct_special_types(result)

    def _reconstruct_special_types(self, obj: Any) -> Any:
        """Recursively reconstruct special types (TensorDict) from their dict representation."""
        if isinstance(obj, dict):
            # Check if this is a TensorDict marker
            if obj.get("__tq_tensordict__"):
                return self._reconstruct_tensordict(obj)
            # Recursively process dict values
            return {k: self._reconstruct_special_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._reconstruct_special_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._reconstruct_special_types(item) for item in obj)
        return obj

    def _reconstruct_tensordict(self, obj: dict) -> Any:
        """Reconstruct TensorDict from marked dict structure."""
        try:
            from tensordict import TensorDict

            batch_size = obj["batch_size"]
            data = obj["data"]
            # Recursively process nested data
            processed_data = self._reconstruct_special_types(data)
            return TensorDict(processed_data, batch_size=batch_size)
        except ImportError:
            # If tensordict not available, return as dict
            return obj

    def _decode_tensor(self, meta: tuple) -> torch.Tensor:
        """Decode tensor from (dtype, shape, buffer_idx) tuple."""
        dtype, shape, idx = meta
        buffer = self.aux_buffers[idx]
        torch_dtype = getattr(torch, dtype)

        if not buffer:  # Handle empty tensors
            return torch.empty(shape, dtype=torch_dtype)

        # Create uint8 tensor from buffer, then view as original dtype and reshape
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    def _decode_nested_tensor(self, nested_meta: dict) -> torch.Tensor:
        """Decode nested tensor from serialized sub-tensors."""
        layout = nested_meta["layout"]
        tensor_metas = nested_meta["tensors"]

        # Decode each sub-tensor
        sub_tensors = [self._decode_tensor(meta) for meta in tensor_metas]

        # Reconstruct nested tensor with appropriate layout
        if layout == "jagged":
            return torch.nested.as_nested_tensor(sub_tensors, layout=torch.jagged)
        else:  # strided
            return torch.nested.as_nested_tensor(sub_tensors, layout=torch.strided)

    def ext_hook(self, code: int, data: memoryview) -> Any:
        """Custom decoding hook for types msgspec doesn't natively support.

        For zero-copy tensor serialization, we need to handle:
        - torch.Tensor: Extract buffer, store metadata
        - TensorDict: Convert to dict structure for recursive processing
        - numpy.ndarray: Convert to tensor for unified handling
        - BatchMeta: Reconstruct from pickle
        """
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)
        if code == CUSTOM_TYPE_TENSOR:
            meta = pickle.loads(data)
            return self._decode_tensor(meta)
        if code == CUSTOM_TYPE_NESTED_TENSOR:
            nested_meta = pickle.loads(data)
            return self._decode_nested_tensor(nested_meta)
        if code == CUSTOM_TYPE_BATCHMETA:
            from transfer_queue.metadata import BatchMeta

            meta_dict = pickle.loads(data)
            return BatchMeta.from_dict(meta_dict)

        raise NotImplementedError(f"Extension type code {code} is not supported")


_encoder = MsgpackEncoder()
_decoder = MsgpackDecoder()
