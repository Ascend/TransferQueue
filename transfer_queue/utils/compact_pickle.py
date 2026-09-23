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

"""Pickle selected values without retaining the storage of unselected tensor rows."""

import pickle

import torch


class Pickler(pickle.Pickler):
    def reducer_override(self, value):
        # Tensor views can retain an entire batch, including unselected keys. This
        # also handles tensors nested in picklable payload objects or metadata.
        if isinstance(value, torch.Tensor):
            return value.clone().__reduce_ex__(pickle.HIGHEST_PROTOCOL)
        return NotImplemented


def dump(value, file):
    Pickler(file, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
