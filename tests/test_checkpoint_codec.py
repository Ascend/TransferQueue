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

import io
import pickle

import pytest
import torch

from transfer_queue.storage.checkpoint_codec import dump_checkpoint


def test_compact_checkpoint_restores_dtype_values_and_other_fields():
    routes = torch.arange(32768, dtype=torch.int16).remainder(256).reshape(128, 16, 16)
    pending = [{"role": "user", "content": "Keep the complete pending prompt."}]
    state = {
        "storage_unit_id": "storage",
        "storage_unit_size": 10,
        "active_keys": {2, 9},
        "field_data": {
            "routed_experts": {2: routes},
            "raw_prompt": {9: pending},
            "metrics": {2: {"duration": 1.25}},
            "input_ids": {2: torch.arange(128)},
            "response_weight_versions": {2: torch.full((128,), 2)},
        },
    }
    stream = io.BytesIO()
    dump_checkpoint(state, stream, ["routed_experts"])
    assert stream.tell() < len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)) - routes.numel() // 2
    restored = pickle.loads(stream.getvalue())
    assert restored["active_keys"] == {2, 9}
    assert restored["field_data"]["raw_prompt"][9] == pending
    assert restored["field_data"]["metrics"] == state["field_data"]["metrics"]
    for field in ("routed_experts", "input_ids", "response_weight_versions"):
        expected = state["field_data"][field][2]
        actual = restored["field_data"][field][2]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    assert state["field_data"]["routed_experts"][2] is routes
    assert routes.dtype == torch.int16


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([-1, 0, 15], dtype=torch.int16),
        torch.tensor([0, 255, 256], dtype=torch.int16),
        torch.tensor([0.0, 1.5]),
        torch.tensor([0, 15], dtype=torch.int64),
        torch.empty(0, dtype=torch.int16),
        torch.arange(16, dtype=torch.int16).reshape(4, 4).t(),
    ],
)
def test_unsupported_tensor_keeps_legacy_representation(value):
    state = {"field_data": {"routed_experts": {0: value}}}
    stream = io.BytesIO()
    dump_checkpoint(state, stream, ["routed_experts"])
    assert stream.getvalue() == pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(stream.getvalue())["field_data"]["routed_experts"][0]
    assert restored.dtype == value.dtype
    assert restored.stride() == value.stride()
    assert torch.equal(restored, value)


def test_default_and_unselected_fields_keep_legacy_representation():
    routes = torch.arange(16, dtype=torch.int16)
    state = {"field_data": {"routed_experts": {0: routes}}}
    for fields in ((), ["another_field"]):
        stream = io.BytesIO()
        dump_checkpoint(state, stream, fields)
        assert stream.getvalue() == pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
