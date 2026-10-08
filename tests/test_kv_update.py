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
from tensordict import TensorDict

from transfer_queue.interface import _single_update_batch
from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers.base import KVStorageManager
from transfer_queue.storage.managers.mooncake_manager import MooncakeStorageManager
from transfer_queue.storage.managers.ray_storage_manager import RayStorageManager
from transfer_queue.storage.managers.simple_storage_manager import _build_update_field_schema
from transfer_queue.storage.managers.yuanrong_manager import YuanrongStorageManager
from transfer_queue.storage.simple_storage import HybridStorageUnitData, StorageUnitData


def _concat(old, new):
    return torch.cat([old, new])


def _tensor_schema(*fields, dtype=torch.int64):
    return {field: {"dtype": dtype, "shape": None, "is_nested": True, "is_non_tensor": False} for field in fields}


def test_single_update_batch_wraps_tensor_and_python_values():
    batch = _single_update_batch({"tokens": torch.tensor([4, 5]), "meta": {"step": 1}})

    assert batch.batch_size == torch.Size([1])
    assert torch.equal(batch["tokens"][0], torch.tensor([4, 5]))
    assert batch["meta"][0] == {"step": 1}


@pytest.mark.parametrize("fields", [{}, [], {"tokens": torch.tensor([1]), 2: "bad"}])
def test_single_update_batch_rejects_invalid_fields(fields):
    with pytest.raises(TypeError):
        _single_update_batch(fields)


def test_apply_update_concatenates_and_keeps_field_name():
    data = StorageUnitData()
    data.put_data({"sequence_ids": [torch.tensor([10, 11, 12])]}, [0])

    described = data.apply_update(
        [0],
        {"sequence_ids": [torch.tensor([20, 21])]},
        _concat,
        _tensor_schema("sequence_ids"),
    )

    assert described["sequence_ids"] == {"dtype": torch.int64, "shapes": [(5,)]}
    assert torch.equal(data.field_data["sequence_ids"][0], torch.tensor([10, 11, 12, 20, 21]))
    assert set(data.field_data) == {"sequence_ids"}


def test_apply_update_is_atomic_on_merge_error():
    data = StorageUnitData()
    original = torch.tensor([1, 2, 3])
    data.put_data({"tokens": [original.clone()]}, [7])

    def fail(_old, _new):
        raise RuntimeError("merge failed")

    with pytest.raises(RuntimeError, match="merge failed"):
        data.apply_update([7], {"tokens": [torch.tensor([9])]}, fail, _tensor_schema("tokens"))
    assert torch.equal(data.field_data["tokens"][7], original)


def test_apply_update_rejects_unproduced_field():
    data = StorageUnitData()
    data.put_data({"tokens": [torch.tensor([1])]}, [3])

    with pytest.raises(ValueError, match="unproduced field"):
        data.apply_update([3], {"fresh": [torch.tensor([2])]}, lambda _old, new: new, _tensor_schema("fresh"))
    assert "fresh" not in data.field_data


@pytest.mark.parametrize(
    "merge_fn, error",
    [
        (lambda old, _new: old.to(torch.float32), "keep dtype"),
        (lambda _old, _new: {"not": "a tensor"}, "keep dtype"),
    ],
)
def test_apply_update_rejects_incompatible_tensor_results_before_writing(merge_fn, error):
    data = StorageUnitData()
    original = torch.tensor([1, 2])
    data.put_data({"tokens": [original.clone()]}, [0])

    with pytest.raises(TypeError, match=error):
        data.apply_update([0], {"tokens": [torch.tensor([3])]}, merge_fn, _tensor_schema("tokens"))
    assert torch.equal(data.field_data["tokens"][0], original)


def test_apply_update_rejects_tensor_result_for_non_tensor_field():
    data = StorageUnitData()
    data.put_data({"meta": [{"step": 1}]}, [0])
    schema = {"meta": {"dtype": None, "shape": None, "is_nested": False, "is_non_tensor": True}}

    with pytest.raises(TypeError, match="must remain non-tensor"):
        data.apply_update([0], {"meta": [{"step": 2}]}, lambda _old, _new: torch.tensor([2]), schema)
    assert data.field_data["meta"][0] == {"step": 1}


def test_apply_update_decodes_ssd_offloaded_old_value(tmp_path):
    data = HybridStorageUnitData(
        storage_size=4, threshold_bytes=64, ssd_path=str(tmp_path), run_id="run", unit_id="unit"
    )
    prompt = torch.arange(32)
    data.put_data({"tokens": [prompt]}, [0])
    assert data.ssd_active_values == 1

    data.apply_update([0], {"tokens": [torch.tensor([99])]}, _concat, _tensor_schema("tokens"))

    assert torch.equal(data.get_data(["tokens"], [0])["tokens"][0], torch.cat([prompt, torch.tensor([99])]))
    assert data.ssd_active_values == 1


def test_apply_update_is_atomic_when_a_later_field_fails_to_reach_ssd(tmp_path, monkeypatch):
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
            data.apply_update([0], new_data, _concat, _tensor_schema("tokens", "mask"))

        stored = data.get_data(["tokens", "mask"], [0])
        assert torch.equal(stored["tokens"][0], prompt)
        assert torch.equal(stored["mask"][0], mask)
        assert set(tmp_path.rglob("*.bin")) == old_files
        assert (data.ssd_active_values, data.ssd_active_bytes) == (2, prompt.nbytes + mask.nbytes)
    finally:
        data.close()


def test_build_update_field_schema_orders_shapes_across_units():
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "manager_cls", [MooncakeStorageManager, YuanrongStorageManager, RayStorageManager, KVStorageManager]
)
@patch("transfer_queue.storage.managers.base.StorageClientFactory.create")
@patch.object(KVStorageManager, "_connect_to_controller", lambda self: None)
async def test_kv_backends_reject_update(mock_create, manager_cls):
    mock_create.return_value = MagicMock()
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
    values = TensorDict({"x": torch.ones(1, 1, dtype=torch.int64)}, batch_size=1)

    with pytest.raises(NotImplementedError, match=f"not supported by {manager_cls.__name__}"):
        await manager.update_data(meta, values, lambda _old, new: new)
