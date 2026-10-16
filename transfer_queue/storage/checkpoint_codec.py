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

"""Opt-in lossless integer encoding at the SimpleStorage checkpoint boundary."""

import pickle

import torch


def _restore_int16_from_uint8_v1(value):
    # A named reconstruction function makes older readers reject this encoding
    # before installing checkpoint data, rather than exposing packed uint8 values.
    if type(value) is not torch.Tensor or value.dtype != torch.uint8:
        raise ValueError("Invalid compact int16 checkpoint tensor")
    return value.to(torch.int16)


class _CheckpointPickler(pickle.Pickler):
    def __init__(self, stream, state, fields):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self._selected = {
            id(value)
            for field in fields
            for value in state["field_data"].get(field, {}).values()
            if type(value) is torch.Tensor
        }

    def reducer_override(self, value):
        if (
            id(value) in self._selected
            and type(value) is torch.Tensor
            and value.dtype == torch.int16
            and value.device.type == "cpu"
            and value.layout == torch.strided
            and value.is_contiguous()
            and value.numel()
            and value.min().item() >= 0
            and value.max().item() <= 255
        ):
            return _restore_int16_from_uint8_v1, (value.to(torch.uint8),)
        return NotImplemented


def dump_checkpoint(state, stream, uint8_fields=()):
    """Pack eligible int16 tensors from explicit fields; retain all other values."""
    if uint8_fields:
        _CheckpointPickler(stream, state, uint8_fields).dump(state)
    else:
        pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
