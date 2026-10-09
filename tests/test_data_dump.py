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

import builtins
import io
import pickle
from types import SimpleNamespace

import pytest
import torch

from transfer_queue import data_dump, interface
from transfer_queue.storage.dump_io import validate_dump_values
from transfer_queue.storage.simple_storage import SimpleStorageUnit, StorageUnitData
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


@pytest.fixture
def unit():
    unit = SimpleStorageUnit.__ray_metadata__.modified_class({})
    yield unit
    unit.shutdown()


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
            body={"path": str(path), "fields_by_index": {index: ["x"] for index in indexes}},
        )
    )
    assert reply.body["success"]
    assert set(reply.body["row_offsets"]) == set(indexes)
    with path.open("rb") as f:
        for index, (offset, length) in reply.body["row_offsets"].items():
            f.seek(offset)
            row = pickle.loads(f.read(length))
            assert row["global_index"] == index
            value = row["fields"]["x"]
            torch.testing.assert_close(value, batch[index])
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    assert path.stat().st_size < row_count * batch[0].numel() * batch.element_size() * 2
    assert unit.storage_data.field_data["x"][0].untyped_storage().nbytes() == batch.numel() * batch.element_size()


def test_row_index_compacts_tensors_inside_tags(monkeypatch, tmp_path):
    batch = torch.arange(64 * 4096).reshape(64, 4096)
    tag = {"nested": [SimpleNamespace(value=batch[0])]}
    client = SimpleNamespace(
        describe_data_dump=lambda *_: {
            "partition_id": "p",
            "rows": {"key": {"global_index": 0, "fields": [], "tag": tag}},
            "field_schema": {},
        }
    )
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", object())
    monkeypatch.setattr(interface, "_maybe_create_tq_client", lambda: client)
    interface.dump_data_by_key(tmp_path / "dump", ["key"], "p")
    restored = data_dump._read_row_index(tmp_path / "dump")["rows"]["key"]["tag"]["nested"][0].value
    torch.testing.assert_close(restored, batch[0])
    assert restored.untyped_storage().nbytes() == restored.numel() * restored.element_size()


@pytest.fixture
def empty_dump_client(monkeypatch):
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", object())
    monkeypatch.setattr(
        interface, "_maybe_create_tq_client", lambda: SimpleNamespace(validate_dump_schema=lambda *_: None)
    )


def test_failed_dump_leaves_no_staging_directory(empty_dump_client, monkeypatch, tmp_path):
    def fail(path):
        raise OSError("disk full")

    monkeypatch.setattr(data_dump, "_fsync_directory", fail)
    with pytest.raises(OSError, match="disk full"):
        interface.dump_data_by_key(tmp_path / "dump", [], "p")
    assert not list(tmp_path.iterdir())


def test_racing_dumps_to_one_path_publish_only_the_first(empty_dump_client, monkeypatch, tmp_path):
    dump = tmp_path / "dump"
    fsync = data_dump._fsync_directory
    raced = []

    def publish_another_dump_first(path):
        if path.name.startswith("dump.tmp-") and not raced:
            raced.append(path)
            interface.dump_data_by_key(dump, [], "first")
        fsync(path)

    monkeypatch.setattr(data_dump, "_fsync_directory", publish_another_dump_first)
    with pytest.raises(OSError):
        interface.dump_data_by_key(dump, [], "second")
    assert data_dump._read_row_index(dump)["partition_id"] == "first"
    assert [path.name for path in tmp_path.iterdir()] == ["dump"]


def test_unit_reads_only_assigned_ranges_and_merges(unit, tmp_path, monkeypatch):
    batch = torch.arange(8 * 4096, dtype=torch.float32).reshape(8, 4096)
    unit.storage_data.put_data({"x": batch, "stale": batch}, list(range(8)))
    path = tmp_path / "shard.pkl"
    reply = unit._handle_dump_rows(
        ZMQMessage.create(
            request_type=ZMQRequestType.DUMP_ROWS,
            sender_id="test",
            body={"path": str(path), "fields_by_index": {index: ["x"] for index in range(8)}},
        )
    )
    assert reply.body["success"]
    content = path.read_bytes()
    reads = []

    class TrackedFile(io.BytesIO):
        name = str(path)

        def read(self, size=-1):
            reads.append((self.tell(), size))
            return super().read(size)

    open_file = builtins.open
    monkeypatch.setattr(
        builtins,
        "open",
        lambda name, *a, **kw: TrackedFile(content) if str(name) == str(path) else open_file(name, *a, **kw),
    )
    unit.storage_data = StorageUnitData()
    unit.storage_data.put_data({"keep": ["value"]}, [101])
    records = []
    for source, target in [(1, 101), (6, 106)]:
        offset, length = reply.body["row_offsets"][source]
        records.append(
            {"source_index": source, "target_index": target, "fields": ["x"], "offset": offset, "length": length}
        )
    loaded = unit._handle_load_rows(
        ZMQMessage.create(
            request_type=ZMQRequestType.LOAD_ROWS,
            sender_id="test",
            body={
                "shards": [
                    {
                        "path": str(path),
                        "records": records,
                        "field_schema": {
                            "x": {"dtype": torch.float32, "shape": (4096,), "is_nested": False, "is_non_tensor": False}
                        },
                    }
                ]
            },
        )
    )
    assert loaded.body["success"], loaded.body
    assert reads == [(row["offset"], row["length"]) for row in records]
    assert loaded.body["bytes_read"] == sum(row["length"] for row in records)
    assert set(unit.storage_data.field_data) == {"x", "keep"}
    assert unit.storage_data.field_data["keep"][101] == "value"
    for row in records:
        torch.testing.assert_close(unit.storage_data.field_data["x"][row["target_index"]], batch[row["source_index"]])
    assert sorted(index for update in loaded.body["updates"] for index in update["global_indexes"]) == [101, 106]


@pytest.mark.parametrize("problem", ["wrong_index", "truncated", "missing_field"])
def test_unit_rejects_invalid_records(unit, tmp_path, problem):
    path = tmp_path / "row.pkl"
    path.write_bytes(pickle.dumps({"global_index": 1, "fields": {"x": "value"}}))
    record = {"source_index": 1, "target_index": 2, "offset": 0, "length": path.stat().st_size, "fields": ["x"]}
    if problem == "wrong_index":
        record["source_index"] = 3
    elif problem == "truncated":
        record["length"] += 1
    else:
        record["fields"] = ["missing"]
    reply = unit._handle_load_rows(
        ZMQMessage.create(
            request_type=ZMQRequestType.LOAD_ROWS,
            sender_id="test",
            body={"shards": [{"path": str(path), "records": [record], "field_schema": {"x": {"is_non_tensor": True}}}]},
        )
    )
    assert not reply.body["success"]
    assert not unit.storage_data._active_keys


def test_load_refuses_other_backends_before_registering_keys(monkeypatch, tmp_path):
    client = SimpleNamespace(
        storage_manager=object(),
        kv_retrieve_meta=lambda *_, **__: pytest.fail("registered keys on an unsupported backend"),
    )
    monkeypatch.setattr(interface, "_TQ_CONTROLLER", object())
    monkeypatch.setattr(interface, "_maybe_create_tq_client", lambda: client)
    with pytest.raises(NotImplementedError, match="does not support selective data load"):
        interface.load_data_by_key(tmp_path)


def test_dump_reports_every_rows_stored_types(unit, tmp_path):
    unit.storage_data.put_data({"x": [torch.arange(2), torch.arange(3)], "y": [None, {"key": "value"}]}, [9, 10])
    response = unit._handle_dump_rows(
        ZMQMessage.create(
            request_type=ZMQRequestType.DUMP_ROWS,
            sender_id="test",
            body={"path": str(tmp_path / "shard.pkl"), "fields_by_index": {9: ["x", "y"], 10: ["x", "y"]}},
        )
    )
    assert response.body["success"]
    assert response.body["row_schema"] == {
        9: {"x": (torch.int64, (2,)), "y": None},
        10: {"x": (torch.int64, (3,)), "y": None},
    }


_DENSE = {"is_nested": False, "is_non_tensor": False}
_NON_TENSOR = {"dtype": None, "shape": None, "is_nested": False, "is_non_tensor": True}


@pytest.mark.parametrize(
    ("declared", "rows", "expected"),
    [
        # A dense field whose later row a put wrapped as a string.
        (_DENSE, [(torch.float32, (2,)), None], _NON_TENSOR),
        # ... or as a tensor of another dtype.
        (_DENSE, [(torch.float32, (2,)), (torch.int64, (2,))], _NON_TENSOR),
        # ... or as a tensor of another shape.
        (
            _DENSE,
            [(torch.float32, (2,)), (torch.float32, (3,))],
            {
                "dtype": torch.float32,
                "shape": None,
                "is_nested": True,
                "is_non_tensor": False,
                "per_sample_shapes": {0: (2,), 1: (3,)},
            },
        ),
        (_DENSE, [(torch.int64, ()), (torch.int64, ())], {**_DENSE, "dtype": torch.int64, "shape": (1,)}),
        (
            {"is_nested": True, "is_non_tensor": False},
            [(torch.int64, (2,)), (torch.int64, (2,))],
            {
                "dtype": torch.int64,
                "shape": None,
                "is_nested": True,
                "is_non_tensor": False,
                "per_sample_shapes": {0: (2,), 1: (2,)},
            },
        ),
        ({"is_nested": False, "is_non_tensor": True}, [(torch.int64, (2,)), (torch.int64, (2,))], _NON_TENSOR),
    ],
)
def test_dump_schema_follows_the_stored_rows(declared, rows, expected):
    row_schema = {index: {"x": meta} for index, meta in enumerate(rows)}
    assert data_dump.dump_field_schema({"x": declared}, row_schema) == {"x": expected}


def test_saved_missing_tensor_shape_is_reported_as_invalid_dump():
    schema = {
        "x": {
            "dtype": torch.int64,
            "shape": None,
            "is_nested": True,
            "is_non_tensor": False,
            "per_sample_shapes": {1: None},
        }
    }
    with pytest.raises(ValueError, match="has no saved shape at row 1"):
        validate_dump_values({"x": torch.arange(2)}, schema, 1)
