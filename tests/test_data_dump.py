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

"""Selective dump integrity and publication tests."""

import pickle
from types import SimpleNamespace

import pytest
import torch

from transfer_queue import data_dump, interface
from transfer_queue.storage.simple_storage import SimpleStorageUnit, StorageUnitData
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


@pytest.fixture
def unit():
    cls = SimpleStorageUnit.__ray_metadata__.modified_class
    unit = cls.__new__(cls)
    unit.storage_unit_id = "test_unit"
    unit.storage_data = StorageUnitData()
    return unit


@pytest.mark.parametrize("row_count", [1, 32])
def test_dump_excludes_unselected_tensor_storage(unit, tmp_path, row_count):
    batch = torch.arange(64 * 4096, dtype=torch.float32).reshape(64, 4096)
    unit.storage_data.put_data({"x": batch}, list(range(64)))
    path = tmp_path / "shard.pkl"
    indexes = list(range(row_count))
    reply = unit._handle_dump_rows(
        ZMQMessage.create(
            request_type=ZMQRequestType.DUMP_ROWS,
            sender_id="test",
            body={"path": str(path), "global_indexes": indexes},
        )
    )
    assert reply.body["success"]
    with path.open("rb") as f:
        shard = pickle.load(f)
    assert shard["global_indexes"] == indexes
    for index, value in shard["field_data"]["x"].items():
        torch.testing.assert_close(value, batch[index])
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    assert path.stat().st_size < row_count * batch[0].numel() * batch.element_size() * 2
    assert unit.storage_data.field_data["x"][0].untyped_storage().nbytes() == batch.numel() * batch.element_size()


def test_row_index_compacts_tensors_inside_tags(monkeypatch, tmp_path):
    batch = torch.arange(64 * 4096).reshape(64, 4096)
    tag = {"nested": [SimpleNamespace(value=batch[0])]}
    client = SimpleNamespace(
        describe_rows_by_key=lambda *_: {
            "key": {"global_index": 0, "fields": [], "tag": tag},
        }
    )
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", object())
    monkeypatch.setattr(interface, "_maybe_create_tq_client", lambda: client)
    data_dump.dump_data_by_key(tmp_path / "dump", ["key"], "p")
    restored = data_dump.read_row_index(tmp_path / "dump")["rows"]["key"]["tag"]["nested"][0].value
    torch.testing.assert_close(restored, batch[0])
    assert restored.untyped_storage().nbytes() == restored.numel() * restored.element_size()


@pytest.fixture
def empty_dump_client(monkeypatch):
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", object())
    monkeypatch.setattr(interface, "_maybe_create_tq_client", lambda: object())


def test_failed_publication_preserves_previous_dump(empty_dump_client, monkeypatch, tmp_path):
    dump = tmp_path / "dump"
    data_dump.dump_data_by_key(dump, [], "old")
    rename = type(dump).rename

    def fail_publish(path, target):
        if path == tmp_path / "dump.tmp":
            raise OSError("publication failed")
        return rename(path, target)

    monkeypatch.setattr(type(dump), "rename", fail_publish)
    with pytest.raises(OSError, match="publication failed"):
        data_dump.dump_data_by_key(dump, [], "new")
    assert data_dump.read_row_index(dump)["partition_id"] == "old"
    assert not (tmp_path / "dump.tmp").exists()


@pytest.mark.parametrize("next_operation", ["read", "load", "dump"])
def test_interrupted_publication_recovers_on_next_access(empty_dump_client, monkeypatch, tmp_path, next_operation):
    dump = tmp_path / "dump"
    data_dump.dump_data_by_key(dump, [], "old")
    rename = type(dump).rename

    def interrupt_publish(path, target):
        if path == tmp_path / "dump.tmp":
            raise KeyboardInterrupt
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(type(dump), "rename", interrupt_publish)
        with pytest.raises(KeyboardInterrupt):
            data_dump.dump_data_by_key(dump, [], "new")
    assert not dump.exists()
    assert (tmp_path / "dump.old").exists()
    if next_operation == "read":
        assert data_dump.read_row_index(dump)["partition_id"] == "old"
    elif next_operation == "load":
        assert data_dump.load_data_by_key(dump)["keys"] == 0
        assert data_dump.read_row_index(dump)["partition_id"] == "old"
    else:
        data_dump.dump_data_by_key(dump, [], "replacement")
        assert data_dump.read_row_index(dump)["partition_id"] == "replacement"


def test_backup_cleanup_failure_does_not_fail_published_dump(empty_dump_client, monkeypatch, tmp_path):
    dump = tmp_path / "dump"
    data_dump.dump_data_by_key(dump, [], "old")
    rmtree = data_dump.shutil.rmtree

    def fail_cleanup(path):
        if path == tmp_path / "dump.old":
            raise OSError("cleanup failed")
        return rmtree(path)

    monkeypatch.setattr(data_dump.shutil, "rmtree", fail_cleanup)
    data_dump.dump_data_by_key(dump, [], "new")
    assert data_dump.read_row_index(dump)["partition_id"] == "new"
