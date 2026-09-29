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
A node-local path such as pytest's ``tmp_path`` fails on a multi-node cluster
with ``FileNotFoundError``.

Run with:
    pytest tests/e2e/test_data_dump_e2e.py -v
"""

import json
import os
import shutil
import uuid
from pathlib import Path

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

import transfer_queue as tq

os.environ["RAY_DEDUP_LOGS"] = "0"

_NUM_STORAGE_UNITS = 4
_DEFAULT_DUMP_ROOT = "/apdcephfs_hldy/share_303541817/tq_dump_tests"


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


@pytest.fixture(scope="module")
def shared_root():
    root = Path(os.environ.get("TQ_DUMP_TEST_ROOT", _DEFAULT_DUMP_ROOT)) / uuid.uuid4().hex
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def dump_dir(shared_root, request):
    case = shared_root / request.node.name.replace("/", "_")
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

    def test_dump_replaces_preexisting_directory(self, tq_system, dump_dir):
        # Define test data
        partition_id = "d_replace"
        _put_rows(partition_id, ["s0", "s1"])
        dump_dir.mkdir(parents=True)
        (dump_dir / "stale.pkl").write_bytes(b"stale")

        # Dump
        tq.dump_data_by_key(dump_dir, ["s0"], partition_id)

        # Check saved state
        assert not (dump_dir / "stale.pkl").exists()
        assert (dump_dir / "dump_info.json").exists()


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
        assert not dump_dir.exists()
        assert not dump_dir.with_name(dump_dir.name + ".tmp").exists()

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
