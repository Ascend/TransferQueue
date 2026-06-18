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

"""End-to-end tests for save_checkpoint and load_checkpoint.

Run with:
    pytest tests/e2e/test_checkpoint_e2e.py -v
"""

import json
import os

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

import transfer_queue as tq

os.environ["RAY_DEDUP_LOGS"] = "0"

_TQ_CONFIG = OmegaConf.create(
    {
        "controller": {"polling_mode": True},
        "backend": {
            "storage_backend": "SimpleStorage",
            "SimpleStorage": {
                "total_storage_size": 200,
                "num_data_storage_units": 2,
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
        ray.init(namespace="TestCheckpointE2E")
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="module")
def tq_system(ray_init):
    tq.init(_TQ_CONFIG)
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
def checkpoint_dir(tmp_path):
    return tmp_path / "checkpoint"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _assert_tensor_equal(a, b, msg=""):
    if (isinstance(a, torch.Tensor) and a.is_nested) or (isinstance(b, torch.Tensor) and b.is_nested):
        for t1, t2 in zip(list(a), list(b), strict=True):
            assert torch.equal(t1, t2), f"{msg} mismatch"
    else:
        assert torch.equal(a, b), f"{msg} mismatch"


# ---------------------------------------------------------------------------
# basic save / load roundtrip
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    """Standard data → save → verify files → wipe → load → verify data."""

    def test_tensor_fields(self, tq_system, checkpoint_dir, controller):
        # Define test data
        keys = ["k0", "k1"]
        partition_id = "p_tensor"
        input_ids = torch.tensor([[1, 2], [3, 4]])
        attention_mask = torch.ones(2, 2)

        # Put
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=TensorDict({"input_ids": input_ids, "attention_mask": attention_mask}, batch_size=len(keys)),
            tags=[{} for _ in keys],
        )

        # Save
        tq.save_checkpoint(checkpoint_dir)

        # Check saved state: expected files exist
        assert (checkpoint_dir / "metadata.json").exists()
        assert (checkpoint_dir / "controller_state.pkl").exists()
        su_dir = checkpoint_dir / "simple_storage"
        assert su_dir.exists()
        assert (su_dir / "storage_unit_info.json").exists()
        with open(checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["storage_saved"] is True

        # Wipe controller state so load has real work to do
        ray.get(controller.clear_partition.remote(partition_id))
        assert ray.get(controller.list_partitions.remote()) == []

        # Load
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state: partition and data restored
        assert partition_id in ray.get(controller.list_partitions.remote())
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id)
        _assert_tensor_equal(retrieved["input_ids"], input_ids)
        _assert_tensor_equal(retrieved["attention_mask"], attention_mask)

    def test_controller_metadata(self, tq_system, checkpoint_dir, controller):
        # Define test data
        keys = ["a0", "a1", "a2"]
        partition_id = "p_meta"
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tags = [{"idx": i} for i in range(3)]

        # Put
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=TensorDict({"input_ids": input_ids, "attention_mask": torch.ones(3, 3)}, batch_size=len(keys)),
            tags=tags,
        )

        # Save
        tq.save_checkpoint(checkpoint_dir)

        # Wipe
        ray.get(controller.clear_partition.remote(partition_id))

        # Load
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state: key mapping and tags intact
        snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
        for i, key in enumerate(keys):
            assert key in snapshot.keys_mapping
            gidx = snapshot.keys_mapping[key]
            assert snapshot.custom_meta[gidx]["idx"] == i

    def test_multiple_partitions(self, tq_system, checkpoint_dir, controller):
        # Define test data
        partitions_data = {f"part_{i}": (torch.full((2, 4), i, dtype=torch.long), torch.ones(2, 4)) for i in range(3)}

        # Put
        for pid, (iids, mask) in partitions_data.items():
            tq.kv_batch_put(
                keys=[f"{pid}_k0", f"{pid}_k1"],
                partition_id=pid,
                fields=TensorDict({"input_ids": iids, "attention_mask": mask}, batch_size=2),
                tags=[{}, {}],
            )

        # Save
        tq.save_checkpoint(checkpoint_dir)

        # Wipe
        for pid in partitions_data:
            ray.get(controller.clear_partition.remote(pid))

        # Load
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state
        for pid, (iids, _) in partitions_data.items():
            retrieved = tq.kv_batch_get(keys=[f"{pid}_k0", f"{pid}_k1"], partition_id=pid, select_fields=["input_ids"])
            _assert_tensor_equal(retrieved["input_ids"], iids)

    def test_user_metadata_preserved(self, tq_system, checkpoint_dir):
        # Define test data
        keys = ["m0"]

        # Put
        tq.kv_batch_put(
            keys=keys,
            partition_id="p_usermeta",
            fields=TensorDict(
                {"input_ids": torch.tensor([[10, 20]]), "attention_mask": torch.ones(1, 2)}, batch_size=1
            ),
            tags=[{}],
        )

        # Save with user metadata
        tq.save_checkpoint(checkpoint_dir, metadata={"iteration": 42, "loss": 0.5})

        # Check saved state: user metadata written correctly
        with open(checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["user_metadata"]["iteration"] == 42
        assert meta["user_metadata"]["loss"] == pytest.approx(0.5)

    def test_non_tensor_fields(self, tq_system, checkpoint_dir, controller):
        # Define test data
        keys = ["t0", "t1"]
        partition_id = "p_str"
        input_ids = torch.tensor([[1, 2], [3, 4]])
        fields = TensorDict(
            {"input_ids": input_ids, "text": NonTensorStack("hello", "world")},
            batch_size=2,
        )

        # Put
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=[{}, {}])

        # Save
        tq.save_checkpoint(checkpoint_dir)

        # Wipe
        ray.get(controller.clear_partition.remote(partition_id))

        # Load
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["input_ids"])
        _assert_tensor_equal(retrieved["input_ids"], input_ids)

    def test_nested_tensor_fields(self, tq_system, checkpoint_dir, controller):
        # Define test data
        keys = ["j0", "j1", "j2"]
        partition_id = "p_jagged"

        # Put (variable-length sequences)
        for i, key in enumerate(keys):
            tq.kv_put(
                key=key,
                partition_id=partition_id,
                fields=TensorDict({"seq": torch.arange(i + 1, dtype=torch.float).unsqueeze(0)}, batch_size=1),
                tag=None,
            )

        # Save
        tq.save_checkpoint(checkpoint_dir)

        # Wipe
        ray.get(controller.clear_partition.remote(partition_id))

        # Load
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["seq"])
        for i, component in enumerate(retrieved["seq"].unbind()):
            _assert_tensor_equal(component, torch.arange(i + 1, dtype=torch.float))


# ---------------------------------------------------------------------------
# include_storage=False (SimpleStorage override)
# ---------------------------------------------------------------------------


class TestIncludeStorageFalse:
    """For SimpleStorage, include_storage=False is silently forced to True."""

    def test_storage_saved_is_true(self, tq_system, checkpoint_dir):
        # Define test data + Put
        tq.kv_batch_put(
            keys=["n0"],
            partition_id="p_nometa",
            fields=TensorDict({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}, batch_size=1),
            tags=[{}],
        )

        # Save with include_storage=False
        tq.save_checkpoint(checkpoint_dir, include_storage=False)

        # Check saved state: storage_saved must be True and directory must exist
        with open(checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["storage_saved"] is True
        assert (checkpoint_dir / "simple_storage").exists()

    def test_both_restored_after_load(self, tq_system, checkpoint_dir, controller):
        # Define test data
        keys = ["n0", "n1"]
        partition_id = "p_nometa2"
        input_ids = torch.tensor([[5, 6], [7, 8]])

        # Put
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=TensorDict({"input_ids": input_ids, "attention_mask": torch.ones(2, 2)}, batch_size=len(keys)),
            tags=[{} for _ in keys],
        )

        # Save
        tq.save_checkpoint(checkpoint_dir, include_storage=False)

        # Wipe
        ray.get(controller.clear_partition.remote(partition_id))

        # Load
        tq.load_checkpoint(checkpoint_dir)

        # Check loaded state: both controller and storage restored
        assert partition_id in ray.get(controller.list_partitions.remote())
        snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
        for key in keys:
            assert key in snapshot.keys_mapping
        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id)
        _assert_tensor_equal(retrieved["input_ids"], input_ids)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


class TestCheckpointErrors:
    def test_save_raises_if_not_initialized(self, tmp_path):
        import transfer_queue.interface as iface

        original = iface._TQ_CONTROLLER
        try:
            iface._TQ_CONTROLLER = None
            with pytest.raises(RuntimeError, match="not initialized"):
                tq.save_checkpoint(tmp_path / "ck")
        finally:
            iface._TQ_CONTROLLER = original

    def test_load_raises_if_not_initialized(self, tmp_path):
        import transfer_queue.interface as iface

        original = iface._TQ_CONTROLLER
        try:
            iface._TQ_CONTROLLER = None
            with pytest.raises(RuntimeError, match="not initialized"):
                tq.load_checkpoint(tmp_path / "ck")
        finally:
            iface._TQ_CONTROLLER = original

    def test_load_raises_if_dir_missing(self, tq_system, tmp_path):
        with pytest.raises(FileNotFoundError):
            tq.load_checkpoint(tmp_path / "nonexistent")

    def test_load_raises_if_metadata_missing(self, tq_system, tmp_path):
        ck = tmp_path / "ck"
        ck.mkdir()
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            tq.load_checkpoint(ck)

    def test_load_raises_on_storage_unit_count_mismatch(self, tq_system, tmp_path, checkpoint_dir):
        # Define test data + Put + Save
        tq.kv_batch_put(
            keys=["e0"],
            partition_id="p_err",
            fields=TensorDict({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}, batch_size=1),
            tags=[{}],
        )
        tq.save_checkpoint(checkpoint_dir)

        # Tamper: add a fake extra SU entry so count differs
        su_info_path = checkpoint_dir / "simple_storage" / "storage_unit_info.json"
        with open(su_info_path) as f:
            su_info = json.load(f)
        su_info.append({"position": 99, "storage_unit_id": "fake"})
        with open(su_info_path, "w") as f:
            json.dump(su_info, f)

        with pytest.raises(ValueError, match="count mismatch"):
            tq.load_checkpoint(checkpoint_dir)

    def test_no_partial_state_on_failed_save(self, tq_system, tmp_path):
        # Define test data + Put
        tq.kv_batch_put(
            keys=["f0"],
            partition_id="p_fail",
            fields=TensorDict({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}, batch_size=1),
            tags=[{}],
        )

        ck = tmp_path / "ck"

        import unittest.mock as mock

        with mock.patch(
            "transfer_queue.client.TransferQueueClient.save_storage_checkpoint",
            side_effect=RuntimeError("simulated dump failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated dump failure"):
                tq.save_checkpoint(ck)

        # Check saved state: no partial directory left
        assert not ck.exists()
        assert not (tmp_path / "ck.tmp").exists()
