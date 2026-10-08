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

"""End-to-end tests for dump_data_by_key / load_data_by_key.

Each storage unit writes its own shard from the node it runs on, so
``TQ_DUMP_TEST_ROOT`` must point at a filesystem shared by the whole cluster.
Single-node runs default to pytest-managed temporary storage.

Run with:
    pytest tests/e2e/test_data_dump_e2e.py -v
"""

import builtins
import json
import os
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

import transfer_queue as tq

os.environ["RAY_DEDUP_LOGS"] = "0"

_NUM_STORAGE_UNITS = 4


def _tq_config(num_storage_units: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": 400,
                    "num_data_storage_units": num_storage_units,
                },
            },
        }
    )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_init():
    if not ray.is_initialized():
        ray.init(namespace="TestDataDumpE2E")
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def tq_system(ray_init):
    tq.init(_tq_config(_NUM_STORAGE_UNITS))
    yield
    tq.close()


@pytest.fixture
def controller(tq_system):
    return ray.get_actor("TransferQueueController", namespace="transfer_queue")


@pytest.fixture(autouse=True)
def cleanup_partitions(controller):
    yield
    try:
        for pid in ray.get(controller.list_partitions.remote()):
            ray.get(controller.clear_partition.remote(pid))
    except Exception:
        pass


@pytest.fixture
def dump_dir(dump_test_root, request):
    case = dump_test_root / request.node.name.replace("/", "_")
    yield case / "dump"
    shutil.rmtree(case, ignore_errors=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row_input_ids(row: int) -> torch.Tensor:
    """Deterministic per-row payload so a restored row can be traced to its key."""
    return torch.tensor([row * 10, row * 10 + 1, row * 10 + 2])


def _put_rows(partition_id: str, keys: list[str]) -> None:
    tq.kv_batch_put(
        keys=keys,
        partition_id=partition_id,
        fields=TensorDict(
            {
                "input_ids": torch.stack([_row_input_ids(row) for row in range(len(keys))]),
                "attention_mask": torch.ones(len(keys), 3),
            },
            batch_size=len(keys),
        ),
        tags=[{"idx": row} for row in range(len(keys))],
    )


def _assert_rows_equal(actual: torch.Tensor, expected_rows: list[torch.Tensor]) -> None:
    """Compare a retrieved field row by row.

    TransferQueue returns a batched field as a nested tensor, which ``torch.equal``
    cannot consume directly.
    """
    actual_rows = list(actual.unbind()) if actual.is_nested else list(actual)
    assert len(actual_rows) == len(expected_rows)
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        assert torch.equal(actual_row, expected_row)


def _keys_mapping(controller, partition_id: str) -> dict[str, int]:
    snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
    return dict(snapshot.keys_mapping)


# ---------------------------------------------------------------------------
# dump / load roundtrip
# ---------------------------------------------------------------------------


class TestDumpLoadRoundtrip:
    def test_only_selected_keys_are_restored(self, tq_system, dump_dir, controller):
        # Define test data
        partition_id = "d_basic"
        keys = [f"k{i}" for i in range(6)]
        selected = ["k1", "k4"]
        _put_rows(partition_id, keys)

        # Dump
        report = tq.dump_data_by_key(dump_dir, selected, partition_id)
        assert report == {"keys": 2, "rows_with_data": 2, "shards": report["shards"], "bytes": report["bytes"]}

        # Wipe, then restore
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_data_by_key(dump_dir)

        # Check restored state
        assert sorted(_keys_mapping(controller, partition_id)) == selected
        retrieved = tq.kv_batch_get(keys=selected, partition_id=partition_id, select_fields=["input_ids"])
        _assert_rows_equal(retrieved["input_ids"], [_row_input_ids(keys.index(key)) for key in selected])

    def test_tags_survive_the_roundtrip(self, tq_system, dump_dir, controller):
        # Define test data
        partition_id = "d_tags"
        keys = [f"t{i}" for i in range(5)]
        selected = ["t0", "t3"]
        _put_rows(partition_id, keys)

        # Dump + wipe + load
        tq.dump_data_by_key(dump_dir, selected, partition_id)
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_data_by_key(dump_dir)

        # Check restored state
        snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
        for key in selected:
            assert snapshot.custom_meta[snapshot.keys_mapping[key]]["idx"] == keys.index(key)

    def test_jagged_and_non_tensor_fields_survive(self, tq_system, dump_dir, controller):
        """The shard stores per-row values; packing them back must reproduce the container."""
        # Define test data: variable-length rows plus a string field
        partition_id = "d_jagged"
        keys = ["j0", "j1", "j2"]
        for row, key in enumerate(keys):
            tq.kv_put(
                key=key,
                partition_id=partition_id,
                fields=TensorDict(
                    {
                        "seq": torch.arange(row + 1, dtype=torch.float).unsqueeze(0),
                        "text": NonTensorStack(f"row-{row}"),
                    },
                    batch_size=1,
                ),
                tag=None,
            )

        # Dump + wipe + load
        tq.dump_data_by_key(dump_dir, keys, partition_id)
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_data_by_key(dump_dir)

        # Check restored state
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["seq", "text"])
        for row, component in enumerate(retrieved["seq"].unbind()):
            assert torch.equal(component, torch.arange(row + 1, dtype=torch.float))
        assert list(retrieved["text"]) == [f"row-{row}" for row in range(len(keys))]

    def test_heterogeneous_field_sets_are_grouped(self, tq_system, dump_dir, controller):
        """A selective dump routinely mixes rows that finished different fields."""
        # Define test data: h1 has an extra field the others lack
        partition_id = "d_hetero"
        _put_rows(partition_id, ["h0", "h1", "h2"])
        tq.kv_put(
            key="h1",
            partition_id=partition_id,
            fields=TensorDict({"routed_experts": torch.tensor([[7, 8]])}, batch_size=1),
            tag=None,
        )

        # Dump + wipe + load
        tq.dump_data_by_key(dump_dir, ["h0", "h1", "h2"], partition_id)
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_data_by_key(dump_dir)

        # Check restored state: the extra field came back only on its own row
        rows = tq.read_row_index(dump_dir)["rows"]
        assert "routed_experts" in rows["h1"]["fields"]
        assert "routed_experts" not in rows["h0"]["fields"]
        retrieved = tq.kv_batch_get(keys=["h1"], partition_id=partition_id, select_fields=["routed_experts"])
        _assert_rows_equal(retrieved["routed_experts"], [torch.tensor([7, 8])])

    def test_other_partitions_are_untouched(self, tq_system, dump_dir, controller):
        """Restoring merges by key, unlike a checkpoint load which replaces everything."""
        # Define test data
        _put_rows("d_target", ["a0", "a1"])
        _put_rows("d_bystander", ["b0", "b1"])

        # Dump one partition, then restore it without wiping the other
        tq.dump_data_by_key(dump_dir, ["a0"], "d_target")
        ray.get(controller.clear_partition.remote("d_target"))
        tq.load_data_by_key(dump_dir)

        # Check state: the bystander partition and its rows survived
        assert sorted(ray.get(controller.list_partitions.remote())) == ["d_bystander", "d_target"]
        retrieved = tq.kv_batch_get(keys=["b0", "b1"], partition_id="d_bystander", select_fields=["input_ids"])
        _assert_rows_equal(retrieved["input_ids"], [_row_input_ids(0), _row_input_ids(1)])

    def test_row_without_produced_fields_keeps_its_key(self, tq_system, dump_dir, controller):
        # Define test data: retrieve_meta with create=True registers a key with no fields
        partition_id = "d_keyonly"
        _put_rows(partition_id, ["p0"])
        client = tq.get_client()
        client.kv_retrieve_meta(keys=["empty0"], partition_id=partition_id, create=True)

        # Dump both rows
        report = tq.dump_data_by_key(dump_dir, ["p0", "empty0"], partition_id)
        assert report["keys"] == 2
        assert report["rows_with_data"] == 1

        # Wipe + load
        ray.get(controller.clear_partition.remote(partition_id))
        tq.load_data_by_key(dump_dir)

        # Check restored state: the field-less row still exists
        assert sorted(_keys_mapping(controller, partition_id)) == ["empty0", "p0"]

    def test_duplicate_keys_are_deduplicated(self, tq_system, dump_dir):
        # Define test data
        partition_id = "d_dupes"
        _put_rows(partition_id, ["d0", "d1", "d2"])

        # Dump with a repeated key
        report = tq.dump_data_by_key(dump_dir, ["d1", "d1", "d2"], partition_id)

        # Check report and dump info
        assert report["keys"] == 2
        with open(dump_dir / "dump_info.json", encoding="utf-8") as f:
            assert json.load(f)["num_keys"] == 2

    def test_empty_key_set_writes_a_readable_dump(self, tq_system, dump_dir, controller):
        # Dump nothing
        report = tq.dump_data_by_key(dump_dir, [], "d_empty")

        # Check saved state: still a complete, loadable dump
        assert report["keys"] == 0
        assert report["rows_with_data"] == 0
        assert (dump_dir / "dump_info.json").exists()

        # Check that loading it is a no-op rather than an error
        assert tq.load_data_by_key(dump_dir)["keys"] == 0

    def test_live_partition_survives_a_dump(self, tq_system, dump_dir, controller):
        # Define test data
        partition_id = "d_nonmutating"
        keys = [f"l{i}" for i in range(4)]
        _put_rows(partition_id, keys)
        before = _keys_mapping(controller, partition_id)

        # Dump a subset
        tq.dump_data_by_key(dump_dir, ["l1"], partition_id)

        # Check live state: untouched rows keep their indexes and payloads
        assert _keys_mapping(controller, partition_id) == before
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["input_ids"])
        _assert_rows_equal(retrieved["input_ids"], [_row_input_ids(row) for row in range(len(keys))])

    def test_dump_refuses_an_existing_directory(self, tq_system, dump_dir):
        partition_id = "d_existing"
        _put_rows(partition_id, ["s0"])
        tq.dump_data_by_key(dump_dir, ["s0"], partition_id)
        with pytest.raises(FileExistsError):
            tq.dump_data_by_key(dump_dir, ["s0"], partition_id)
        assert sorted(path.name for path in dump_dir.parent.iterdir()) == [dump_dir.name]
        assert tq.read_row_index(dump_dir)["partition_id"] == partition_id

    def test_published_dump_loads_from_a_read_only_directory(self, tq_system, dump_dir):
        partition_id = "d_read_only"
        _put_rows(partition_id, ["s0"])
        tq.dump_data_by_key(dump_dir, ["s0"], partition_id)
        tq.kv_clear(["s0"], partition_id)
        dump_dir.parent.chmod(0o555)
        try:
            assert sorted(tq.read_row_index(dump_dir)["rows"]) == ["s0"]
            tq.load_data_by_key(dump_dir)
            assert sorted(path.name for path in dump_dir.parent.iterdir()) == [dump_dir.name]
        finally:
            dump_dir.parent.chmod(0o755)
        _assert_rows_equal(tq.kv_batch_get(["s0"], partition_id, ["input_ids"])["input_ids"], [_row_input_ids(0)])


# ---------------------------------------------------------------------------
# row index
# ---------------------------------------------------------------------------


class TestRowIndex:
    def test_row_index_describes_keys_without_reading_payload(self, tq_system, dump_dir):
        # Define test data
        partition_id = "d_index"
        keys = ["i0", "i1"]
        _put_rows(partition_id, keys)

        # Dump
        tq.dump_data_by_key(dump_dir, keys, partition_id)

        # Check the index
        row_index = tq.read_row_index(dump_dir)
        assert row_index["partition_id"] == partition_id
        assert sorted(row_index["rows"]) == keys
        for row, key in enumerate(keys):
            assert row_index["rows"][key]["fields"] == ["attention_mask", "input_ids"]
            assert row_index["rows"][key]["tag"] == {"idx": row}

    def test_read_row_index_rejects_a_missing_dump(self, tq_system, dump_dir):
        with pytest.raises(FileNotFoundError, match="row_index.pt"):
            tq.read_row_index(dump_dir)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


class TestDumpErrors:
    def test_unknown_key_raises_and_leaves_no_directory(self, tq_system, dump_dir):
        # Define test data
        partition_id = "d_err_key"
        _put_rows(partition_id, ["e0"])

        # Dump a key that was never put
        with pytest.raises(RuntimeError, match="keys not found"):
            tq.dump_data_by_key(dump_dir, ["e0", "nope"], partition_id)

        # Check saved state: no partial directory left behind
        assert not list(dump_dir.parent.iterdir())

    def test_unknown_partition_raises(self, tq_system, dump_dir):
        _put_rows("d_err_part", ["e0"])
        with pytest.raises(RuntimeError, match="does not exist"):
            tq.dump_data_by_key(dump_dir, ["e0"], "d_never_created")

    def test_load_rejects_a_dump_without_info(self, tq_system, dump_dir):
        dump_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="dump_info.json"):
            tq.load_data_by_key(dump_dir)

    def test_load_rejects_a_missing_shard(self, tq_system, dump_dir):
        # Define test data + dump
        partition_id = "d_err_shard"
        _put_rows(partition_id, ["m0", "m1"])
        tq.dump_data_by_key(dump_dir, ["m0", "m1"], partition_id)

        # Tamper: delete one shard file
        shard = next((dump_dir / "shards").glob("shard_*.pkl"))
        shard.unlink()

        with pytest.raises(FileNotFoundError, match="Missing dump shard"):
            tq.load_data_by_key(dump_dir)

    def test_load_rejects_an_unknown_format_version(self, tq_system, dump_dir):
        # Define test data + dump
        partition_id = "d_err_version"
        _put_rows(partition_id, ["v0"])
        tq.dump_data_by_key(dump_dir, ["v0"], partition_id)

        # Tamper: bump the format version beyond what this build reads
        info_path = dump_dir / "dump_info.json"
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        info["format_version"] = tq.data_dump.DUMP_FORMAT_VERSION + 1
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f)

        with pytest.raises(ValueError, match="Unsupported dump format version"):
            tq.load_data_by_key(dump_dir)


def test_direct_load_bypasses_caller_payload_io(tq_system, dump_dir, controller, monkeypatch):
    partition = "direct_load"
    keys = [f"key-{i}" for i in range(16)]
    _put_rows(partition, keys)
    tq.dump_data_by_key(dump_dir, keys, partition)
    before = _keys_mapping(controller, partition)
    tq.kv_put(keys[0], partition, fields=TensorDict({"extra": torch.tensor([[42]])}, batch_size=1), tag={"keep": True})
    _put_rows(partition, ["bystander"])
    client = tq.get_client()
    manager = client.storage_manager
    original_load = manager._load_selected_rows
    responses = []

    async def load(*args, **kwargs):
        response = await original_load(*args, **kwargs)
        responses.append((kwargs["target_storage_unit"], response))
        return response

    real_open = builtins.open

    def no_payload_open(path, *args, **kwargs):
        if isinstance(path, str | Path) and Path(path).name.startswith("shard_") and str(path).endswith(".pkl"):
            raise AssertionError("Caller opened a payload shard")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_payload_open)
    monkeypatch.setattr(manager, "_load_selected_rows", load)
    monkeypatch.setattr(manager, "put_data", AsyncMock(side_effect=AssertionError("Caller sent payload through put")))
    tq.load_data_by_key(dump_dir)
    assert len({unit for unit, _ in responses}) == _NUM_STORAGE_UNITS
    shard_bytes = sum(path.stat().st_size for path in (dump_dir / "shards").glob("shard_*.pkl"))
    assert sum(response["bytes_read"] for _, response in responses) == shard_bytes
    after = _keys_mapping(controller, partition)
    assert all(after[key] == before[key] for key in keys)
    assert "bystander" in after
    actual = tq.kv_batch_get(keys, partition, select_fields=["input_ids"])
    _assert_rows_equal(actual["input_ids"], [_row_input_ids(i) for i in range(16)])
    extra = tq.kv_batch_get([keys[0]], partition, select_fields=["extra"])
    _assert_rows_equal(extra["extra"], [torch.tensor([42])])
    snapshot = ray.get(controller.get_partition_snapshot.remote(partition))
    assert snapshot.custom_meta[after[keys[0]]]["keep"]
    assert snapshot.custom_meta[after[keys[0]]]["idx"] == 0


@pytest.mark.parametrize("row_count", [3, 127, 128, 129, 130])
@pytest.mark.parametrize("last_kind", ["tensor", "none", "object"])
def test_chunks_with_a_wrapped_last_row_roundtrip(tq_system, dump_dir, row_count, last_kind, monkeypatch):
    partition = "wrapped_last_row"
    keys = [f"k{i}" for i in range(row_count)]
    tensors = [torch.arange(i % 3 + 1, dtype=torch.int64) for i in range(row_count)]
    last = tensors[-1] if last_kind == "tensor" else None if last_kind == "none" else {"pixels": tensors[-1]}
    expected = [*tensors[:-1], last]
    chunk = dump_dir.parent / "legacy-chunk.pt"
    chunk.parent.mkdir(parents=True)
    first = TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor(tensors[:-1], layout=torch.jagged),
            "multi_modal_inputs#images": torch.nested.as_nested_tensor(tensors[:-1], layout=torch.jagged),
            "wrapped": NonTensorStack(*tensors[:-1]),
        },
        batch_size=row_count - 1,
    )
    tail = TensorDict(
        {
            name: NonTensorStack(value)
            for name, value in {
                "input_ids": last,
                "multi_modal_inputs#images": last,
                "wrapped": tensors[-1],
            }.items()
        },
        batch_size=1,
    )
    # Legacy wrappers reload each saved TensorDict without normalizing its container type.
    for chunk_keys, fields in [(keys[:-1], first), (keys[-1:], tail)]:
        torch.save({"keys": chunk_keys, "fields": fields}, chunk)
        saved = torch.load(chunk, weights_only=False)
        tq.kv_batch_put(saved["keys"], partition, saved["fields"], tags=[{"key": key} for key in chunk_keys])

    open_file = builtins.open

    def no_shard_read(path, *args, **kwargs):
        if isinstance(path, str | Path) and Path(path).name.startswith("shard_") and str(path).endswith(".pkl"):
            pytest.fail("The dump schema merge read a payload shard in the caller")
        return open_file(path, *args, **kwargs)

    for target in [dump_dir, dump_dir.parent / "second-dump"]:
        with monkeypatch.context() as patcher:
            patcher.setattr(builtins, "open", no_shard_read)
            tq.dump_data_by_key(target, keys, partition)
        index = tq.read_row_index(target)
        for name in ["input_ids", "multi_modal_inputs#images"]:
            schema = index["field_schema"][name]
            assert schema["is_non_tensor"] == (last_kind != "tensor")
            if last_kind == "tensor":
                assert schema["is_nested"]
                for key, value in zip(keys, expected, strict=True):
                    assert tuple(schema["per_sample_shapes"][index["rows"][key]["global_index"]]) == tuple(value.shape)
        assert index["field_schema"]["wrapped"]["is_non_tensor"]
        # The controller still declares a tensor field, which the dump does not change.
        if last_kind != "tensor" and target == dump_dir:
            with pytest.raises(RuntimeError, match="tensor/non-tensor type mismatch"):
                tq.load_data_by_key(target)
        tq.kv_clear(keys, partition)
        tq.load_data_by_key(target)
        for start in range(0, row_count, 128):
            end = min(start + 128, row_count)
            restored = tq.kv_batch_get(
                keys[start:end], partition, ["input_ids", "multi_modal_inputs#images", "wrapped"]
            )
            for name in ["input_ids", "multi_modal_inputs#images"]:
                for value, reference in zip(restored[name], expected[start:end], strict=True):
                    torch.testing.assert_close(value, reference)
            for value, reference in zip(restored["wrapped"], tensors[start:end], strict=True):
                torch.testing.assert_close(value, reference)
        tags = tq.kv_list(partition)[partition]
        assert tags == {key: {"key": key} for key in keys}


def test_corrupt_shard_keeps_new_rows_unproduced(tq_system, dump_dir, controller):
    partition = "corrupt"
    _put_rows(partition, ["key"])
    tq.dump_data_by_key(dump_dir, ["key"], partition)
    path = next((dump_dir / "shards").glob("shard_*.pkl"))
    path.write_bytes(b"!" * path.stat().st_size)
    ray.get(controller.clear_partition.remote(partition))
    with pytest.raises(RuntimeError, match="failed to load rows"):
        tq.load_data_by_key(dump_dir)
    snapshot = ray.get(controller.get_partition_snapshot.remote(partition))
    assert "key" in snapshot.keys_mapping
    assert not snapshot.field_metadata


def test_incompatible_schema_rejected_before_writes(tq_system, dump_dir, controller):
    tq.kv_batch_put(["k"], "schema", TensorDict({"x": torch.tensor([[3]], dtype=torch.int64)}, batch_size=1))
    tq.dump_data_by_key(dump_dir, ["k"], "schema")
    tq.get_client().clear_partition("schema")
    tq.kv_batch_put(["k"], "schema", TensorDict({"x": torch.tensor([[1.5]])}, batch_size=1))
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        tq.load_data_by_key(dump_dir)
    _assert_rows_equal(tq.kv_batch_get(["k"], "schema", ["x"])["x"], [torch.tensor([1.5])])


@pytest.mark.parametrize("last", ["text", torch.tensor([3.0, 4.0, 5.0])], ids=["string", "ragged"])
def test_dense_field_with_a_wrapped_later_row_roundtrips(tq_system, dump_dir, last):
    partition = "dense_then_wrapped"
    first = torch.tensor([[1.0, 2.0]])
    tq.kv_batch_put(["first"], partition, TensorDict({"x": first}, batch_size=1))
    tq.kv_batch_put(["last"], partition, TensorDict({"x": NonTensorStack(last)}, batch_size=1))
    tq.dump_data_by_key(dump_dir, ["first", "last"], partition)
    tq.kv_clear(["first", "last"], partition)
    tq.load_data_by_key(dump_dir)
    restored = list(tq.kv_batch_get(["first", "last"], partition, ["x"])["x"])
    torch.testing.assert_close(restored[0], first[0])
    if isinstance(last, str):
        assert restored[1] == last
    else:
        torch.testing.assert_close(restored[1], last)


def test_failed_load_publishes_no_metadata_and_retry_is_idempotent(tq_system, dump_dir, controller, monkeypatch):
    _put_rows("retry", ["key"])
    tq.dump_data_by_key(dump_dir, ["key"], "retry")
    client = tq.get_client()
    client.clear_partition("retry")
    manager = client.storage_manager
    original_load = manager._load_selected_rows

    async def lose_reply(*args, **kwargs):
        await original_load(*args, **kwargs)
        raise RuntimeError("lost reply")

    with monkeypatch.context() as patcher:
        patcher.setattr(manager, "_load_selected_rows", lose_reply)
        with pytest.raises(RuntimeError, match="lost reply"):
            tq.load_data_by_key(dump_dir)
    snapshot = ray.get(controller.get_partition_snapshot.remote("retry"))
    index = snapshot.keys_mapping["key"]
    assert not snapshot.field_metadata

    tq.load_data_by_key(dump_dir)
    snapshot = ray.get(controller.get_partition_snapshot.remote("retry"))
    assert snapshot.keys_mapping["key"] == index
    _assert_rows_equal(tq.kv_batch_get(["key"], "retry", ["input_ids"])["input_ids"], [_row_input_ids(0)])
    assert tq.kv_list("retry")["retry"] == {"key": {"idx": 0}}


def test_dump_and_load_do_not_use_the_put_get_timeout_pool(tq_system, dump_dir, monkeypatch):
    _put_rows("own_pool", ["key"])
    manager = tq.get_client().storage_manager
    # Any lease from the ordinary put/get pool now fails, so only the dump pool can serve these.
    monkeypatch.setattr(manager, "storage_rpc_pool", None)
    tq.dump_data_by_key(dump_dir, ["key"], "own_pool")
    tq.load_data_by_key(dump_dir)
    monkeypatch.undo()
    _assert_rows_equal(tq.kv_batch_get(["key"], "own_pool", ["input_ids"])["input_ids"], [_row_input_ids(0)])
