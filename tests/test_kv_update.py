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

import errno
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from transfer_queue.controller import DataPartitionStatus
from transfer_queue.interface import _normalize_kv_update_args
from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.mooncake_manager import MooncakeStorageManager
from transfer_queue.storage.managers.ray_storage_manager import RayStorageManager
from transfer_queue.storage.managers.simple_storage_manager import _build_update_field_schema
from transfer_queue.storage.managers.yuanrong_manager import YuanrongStorageManager
from transfer_queue.storage.simple_storage import HybridStorageUnitData, StorageUnitData


def _concat(old, new):
    if old is None:
        return new
    return torch.cat([old, new])


def test_normalize_empty_rejects_values_and_parser():
    with pytest.raises(ValueError, match="must not specify values"):
        _normalize_kv_update_args("tokens", torch.tensor([1]), None, empty=True)
    with pytest.raises(ValueError, match="must not specify parser"):
        _normalize_kv_update_args("tokens", None, _concat, empty=True)
    names, batch, parser, use_empty = _normalize_kv_update_args("tokens", None, None, empty=True)
    assert names == ["tokens"]
    assert batch is None
    assert parser is None
    assert use_empty is True


def test_normalize_custom_parser_requires_values():
    with pytest.raises(ValueError, match="requires values"):
        _normalize_kv_update_args("tokens", None, _concat)


def test_normalize_rejects_non_callable_parser():
    with pytest.raises(TypeError, match="parser must be callable unless empty=True"):
        _normalize_kv_update_args("tokens", torch.tensor([1]), None)


def test_normalize_single_field_wraps_a_batch():
    names, batch, parser, use_empty = _normalize_kv_update_args("tokens", torch.tensor([4, 5]), _concat)
    assert names == ["tokens"]
    assert use_empty is False
    assert parser is _concat
    assert batch is not None
    assert batch.batch_size == torch.Size([1])
    assert torch.equal(batch["tokens"][0], torch.tensor([4, 5]))


def test_normalize_multi_field_requires_matching_dict():
    with pytest.raises(TypeError, match="must be a dict"):
        _normalize_kv_update_args(["a", "b"], torch.tensor([1]), _concat)
    with pytest.raises(ValueError, match="same columns"):
        _normalize_kv_update_args(["a", "b"], {"a": 1}, _concat)


def test_apply_update_concat_prompt_and_response_keeps_field_name():
    """Stored prompt_ids plus new response_ids become the sequence; the field name is unchanged."""
    prompt_ids = torch.tensor([10, 11, 12])
    response_ids = torch.tensor([20, 21])
    data = StorageUnitData()
    data.put_data({"sequence_ids": [prompt_ids.clone()]}, [0])

    described = data.apply_update([0], ["sequence_ids"], {"sequence_ids": [response_ids]}, _concat, False)

    assert list(described) == ["sequence_ids"]
    assert torch.equal(data.field_data["sequence_ids"][0], torch.tensor([10, 11, 12, 20, 21]))
    assert "prompt_ids" not in data.field_data
    assert "response_ids" not in data.field_data


def test_apply_update_concatenates_and_is_atomic_on_parser_error():
    data = StorageUnitData()
    data.put_data({"tokens": [torch.tensor([1, 2, 3])]}, [7])

    described = data.apply_update([7], ["tokens"], {"tokens": [torch.tensor([4, 5])]}, _concat, False)
    assert described["tokens"] == {"dtype": torch.int64, "shapes": [(5,)]}
    assert torch.equal(data.field_data["tokens"][7], torch.tensor([1, 2, 3, 4, 5]))

    def boom(old, new):
        raise RuntimeError("parser failed")

    with pytest.raises(RuntimeError, match="parser failed"):
        data.apply_update([7], ["tokens"], {"tokens": [torch.tensor([9])]}, boom, False)
    assert torch.equal(data.field_data["tokens"][7], torch.tensor([1, 2, 3, 4, 5]))


def test_apply_update_empty_stores_none():
    data = StorageUnitData()
    data.put_data({"tokens": [torch.tensor([1])], "keep": [torch.tensor([2])]}, [3])
    described = data.apply_update([3], ["tokens"], None, None, True)
    assert described["tokens"]["shapes"] is None
    assert data.field_data["tokens"][3] is None
    assert torch.equal(data.field_data["keep"][3], torch.tensor([2]))


def test_apply_update_missing_field_passes_none_as_old():
    data = StorageUnitData()
    seen = []

    def record(old, new):
        seen.append(old)
        return new

    data.apply_update([1], ["fresh"], {"fresh": [torch.tensor([8])]}, record, False)
    assert seen == [None]
    assert torch.equal(data.field_data["fresh"][1], torch.tensor([8]))


def test_apply_update_decodes_ssd_offloaded_old_value(tmp_path):
    """With SSD offload on, the parser must see the stored tensor, not its file reference."""
    data = HybridStorageUnitData(
        storage_size=4, threshold_bytes=64, ssd_path=str(tmp_path), run_id="run", unit_id="unit"
    )
    prompt = torch.arange(32)
    data.put_data({"tokens": [prompt]}, [0])
    assert data.ssd_active_values == 1, "precondition: the prompt must live on SSD"

    data.apply_update([0], ["tokens"], {"tokens": [torch.tensor([99])]}, _concat, False)

    assert torch.equal(data.get_data(["tokens"], [0])["tokens"][0], torch.cat([prompt, torch.tensor([99])]))
    assert data.ssd_active_values == 1, "the replaced SSD file must be released, not leaked"


def test_apply_update_is_atomic_when_a_later_field_fails_to_reach_ssd(tmp_path, monkeypatch):
    """Concat is not idempotent, so a partly applied update would double-apply on retry."""
    data = HybridStorageUnitData(
        storage_size=4, threshold_bytes=64, ssd_path=str(tmp_path), run_id="run", unit_id="unit"
    )
    try:
        prompt, mask = torch.arange(32), torch.ones(32, dtype=torch.int64)
        data.put_data({"tokens": [prompt], "mask": [mask]}, [0])
        old_files = set(tmp_path.rglob("*.bin"))

        write_values = data._ssd_store.write_values
        calls = []

        def fail_second_field(encoded_values):
            calls.append(encoded_values)
            if len(calls) == 2:
                raise OSError(errno.ENOSPC, "No space left on device")
            return write_values(encoded_values)

        monkeypatch.setattr(data._ssd_store, "write_values", fail_second_field)
        new_data = {"tokens": [torch.tensor([99])], "mask": [torch.tensor([1])]}
        with pytest.raises(OSError, match="No space left"):
            data.apply_update([0], ["tokens", "mask"], new_data, _concat, False)

        stored = data.get_data(["tokens", "mask"], [0])
        assert torch.equal(stored["tokens"][0], prompt)
        assert torch.equal(stored["mask"][0], mask)
        assert set(tmp_path.rglob("*.bin")) == old_files
        assert (data.ssd_active_values, data.ssd_active_bytes) == (2, prompt.nbytes + mask.nbytes)
    finally:
        data.close()


def test_build_update_field_schema_orders_shapes_across_units():
    """Units describe only their own rows; the batch schema must follow metadata order."""
    described = _build_update_field_schema(
        [0, 1, 2, 3],
        [
            ([0, 2], {"tokens": {"dtype": torch.int64, "shapes": [(4,), (4,)]}}),
            ([1, 3], {"tokens": {"dtype": torch.int64, "shapes": [(9,), (9,)]}}),
        ],
    )

    assert described["tokens"]["is_nested"] is True
    assert described["tokens"]["shape"] is None
    assert described["tokens"]["per_sample_shapes"] == [(4,), (9,), (4,), (9,)]


def test_build_update_field_schema_keeps_uniform_column_flat():
    described = _build_update_field_schema(
        [0, 1],
        [
            ([0], {"tokens": {"dtype": torch.int64, "shapes": [(4,)]}}),
            ([1], {"tokens": {"dtype": torch.int64, "shapes": [(4,)]}}),
        ],
    )

    assert described["tokens"] == {
        "dtype": torch.int64,
        "shape": (4,),
        "is_nested": False,
        "is_non_tensor": False,
    }


def test_build_update_field_schema_marks_column_non_tensor_if_any_unit_is():
    described = _build_update_field_schema(
        [0, 1],
        [
            ([0], {"tokens": {"dtype": torch.int64, "shapes": [(4,)]}}),
            ([1], {"tokens": {"dtype": None, "shapes": None}}),
        ],
    )

    assert described["tokens"]["is_non_tensor"] is True
    assert described["tokens"]["shape"] is None


def test_empty_marks_the_controller_field_non_tensor():
    """tq.kv_empty stores None, so the controller must stop describing the column as a tensor."""
    partition = DataPartitionStatus(partition_id="p")
    partition._update_field_metadata(
        [0], {"tokens": {"dtype": torch.int64, "shape": (5,), "is_nested": False, "is_non_tensor": False}}
    )

    partition._update_field_metadata(
        [0], {"tokens": {"dtype": None, "shape": None, "is_nested": False, "is_non_tensor": True}}
    )

    tokens = partition.field_metadata["tokens"]
    assert tokens.is_non_tensor is True
    assert tokens.shape is None
    assert tokens.to_batch_schema([0])["is_non_tensor"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_cls", [MooncakeStorageManager, YuanrongStorageManager, RayStorageManager, KVStorageManager]
)
@patch("transfer_queue.storage.managers.base.StorageClientFactory.create")
@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
async def test_kv_backends_reject_update(mock_create, manager_cls):
    """Every KV backend must refuse kv_update by name; only SimpleStorage implements it."""
    mock_create.return_value = MagicMock()
    # Each manager validates its own config before reaching update_data.
    config = {
        KVStorageManager: {"client_name": "YuanrongStorageClient"},
        YuanrongStorageManager: {"worker_port": 31501},
    }.get(manager_cls, {})
    manager = manager_cls(controller_info=MagicMock(), config=config)
    meta = BatchMeta(
        global_indexes=[0],
        partition_ids=["p"],
        field_schema={"x": {"dtype": torch.int64, "shape": (1,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(1, dtype=np.int8),
    )
    with pytest.raises(NotImplementedError, match=f"not supported by {manager_cls.__name__}"):
        await manager.update_data(meta, ["x"], empty=True)
