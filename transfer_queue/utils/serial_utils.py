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
from collections.abc import Sequence
from types import FunctionType
from typing import Any, Optional, TypeAlias

import cloudpickle
import torch
import zmq
from msgspec import msgpack


CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_TENSOR = 3  # For tensor with buffer reference
CUSTOM_TYPE_NESTED_TENSOR = 4  # For nested tensor (strided or jagged)

bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


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
        """Custom encoding hook for types msgspec doesn't natively support.
        
        For zero-copy tensor serialization, we need to handle:
        - torch.Tensor: Extract buffer, store metadata
        - TensorDict: Convert to dict structure for recursive processing
        - numpy.ndarray: Convert to tensor for unified handling
        """
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)
        
        # Handle TensorDict explicitly for recursive zero-copy
        # Import here to avoid circular dependency and make it optional
        try:
            from tensordict import TensorDictBase
            if isinstance(obj, TensorDictBase):
                return self._encode_tensordict(obj)
        except ImportError:
            pass
        
        # Handle numpy arrays by converting to tensor
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return self._encode_tensor(torch.from_numpy(obj))
        except ImportError:
            pass

        if isinstance(obj, FunctionType):
            # cloudpickle for functions/methods
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        # Fallback to pickle for unknown types
        return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    
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
        assert self.aux_buffers is not None
        
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
        if obj.device.type != 'cpu':
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
        if obj.device.type != 'cpu':
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

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self):
        self.decoder = msgpack.Decoder(ext_hook=self.ext_hook)
        self.aux_buffers: Sequence[bytestr] = ()

    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):
            result = self.decoder.decode(bufs)
        else:
            self.aux_buffers = bufs
            try:
                result = self.decoder.decode(bufs[0])  # type: ignore[index]
            finally:
                self.aux_buffers = ()
        
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
        arr_tensor = torch.frombuffer(bytearray(buffer), dtype=torch.uint8)
        return arr_tensor.view(torch_dtype).reshape(shape)
    
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

        raise NotImplementedError(f"Extension type code {code} is not supported")


_encoder = MsgpackEncoder()
_decoder = MsgpackDecoder()

