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

"""A dump taken with N storage units must restore into a system with M storage units.

This is the property that separates a data dump from a checkpoint. ``load_checkpoint``
sends each storage unit file back to the unit at the same position, so it requires the
same unit count; ``load_data_by_key`` writes rows back by key and lets TransferQueue
route them for the current topology.

Each test restarts TransferQueue with a different unit count, so this lives apart from
``test_data_dump_e2e.py``, whose fixtures hold one system for the whole module.

Run with:
    pytest tests/e2e/test_data_dump_cross_topology_e2e.py -v
"""

import os

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

import transfer_queue as tq

os.environ["RAY_DEDUP_LOGS"] = "0"


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


@pytest.fixture(scope="module")
def ray_init():
    if not ray.is_initialized():
        ray.init(namespace="TestDataDumpCrossTopology")
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def dump_dir(dump_test_root, request):
    return dump_test_root / request.node.name / "dump"


def _row_input_ids(row: int) -> torch.Tensor:
    return torch.tensor([row * 10, row * 10 + 1, row * 10 + 2])


def _put_rows(partition_id: str, keys: list[str]) -> None:
    tq.kv_batch_put(
        keys=keys,
        partition_id=partition_id,
        fields=TensorDict(
            {"input_ids": torch.stack([_row_input_ids(row) for row in range(len(keys))])},
            batch_size=len(keys),
        ),
        tags=[{"idx": row} for row in range(len(keys))],
    )


def _assert_rows_equal(actual: torch.Tensor, expected_rows: list[torch.Tensor]) -> None:
    actual_rows = list(actual.unbind()) if actual.is_nested else list(actual)
    assert len(actual_rows) == len(expected_rows)
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        assert torch.equal(actual_row, expected_row)


# The controller and each storage unit reserve one CPU, and CI runners have four,
# so more than three units never get scheduled there.
@pytest.mark.parametrize(
    ("dump_units", "load_units"),
    [(3, 2), (2, 3), (1, 3)],
)
def test_dump_restores_across_storage_unit_counts(ray_init, dump_dir, dump_units, load_units):
    # Define test data
    partition_id = "cross"
    keys = [f"c{i}" for i in range(8)]

    # Dump with one topology
    tq.init(_tq_config(dump_units))
    try:
        _put_rows(partition_id, keys)
        report = tq.dump_data_by_key(dump_dir, keys, partition_id)
        assert report["rows_with_data"] == len(keys)
        assert report["shards"] <= dump_units
    finally:
        tq.close()

    # Restore into a different topology
    tq.init(_tq_config(load_units))
    try:
        _put_rows("bystander", ["unrelated"])
        _put_rows(partition_id, [keys[3]])
        old_index = tq.get_client().kv_retrieve_meta([keys[3]], partition_id).global_indexes[0]
        tq.load_data_by_key(dump_dir)
        assert tq.get_client().kv_retrieve_meta([keys[3]], partition_id).global_indexes == [old_index]
        assert tq.kv_batch_get(["unrelated"], "bystander", ["input_ids"]).batch_size[0] == 1

        # Check restored state: every row readable, payload and tag intact
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["input_ids"])
        _assert_rows_equal(retrieved["input_ids"], [_row_input_ids(row) for row in range(len(keys))])

        controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
        snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
        for row, key in enumerate(keys):
            assert snapshot.custom_meta[snapshot.keys_mapping[key]]["idx"] == row
    finally:
        tq.close()


def test_preserves_nontensor_schema_across_topology(ray_init, dump_dir):
    # Spans two 128-row load batches; test_data_dump_e2e.py covers each batch boundary on one system.
    row_count = 130
    keys = [f"k{i}" for i in range(row_count)]
    values = [torch.tensor([i], dtype=torch.int64) for i in range(row_count - 1)] + [torch.tensor([1.5])]
    values[-2] = None
    tq.init(_tq_config(1))
    try:
        tq.kv_batch_put(keys, "mixed", TensorDict({"x": NonTensorStack(*values)}, batch_size=row_count))
        tq.dump_data_by_key(dump_dir, keys, "mixed")
    finally:
        tq.close()
    tq.init(_tq_config(2))
    try:
        tq.load_data_by_key(dump_dir)
        schema = tq.get_client().kv_retrieve_meta(keys, "mixed").field_schema["x"]
        assert schema["is_non_tensor"]
        assert schema["dtype"] is None
        assert not schema["is_nested"]
        for key, expected in zip(keys, values, strict=True):
            value = tq.kv_batch_get([key], "mixed", ["x"])["x"][0]
            if expected is None:
                assert value is None
            else:
                torch.testing.assert_close(value, expected)
    finally:
        tq.close()
