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
from tensordict import TensorDict

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


def _put_batch(keys, partition_id, input_ids, attention_mask, tags=None):
    fields = TensorDict(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        batch_size=len(keys),
    )
    if tags is None:
        tags = [{} for _ in keys]
    tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)


def _get_batch(keys, partition_id):
    return tq.kv_batch_get(keys=keys, partition_id=partition_id)


def assert_tensor_equal(tensor_a, tensor_b, msg=""):
    """Assert two tensors are equal, handling nested vs dense comparisons."""
    if (isinstance(tensor_a, torch.Tensor) and tensor_a.is_nested) or (
        isinstance(tensor_b, torch.Tensor) and tensor_b.is_nested
    ):
        seq_a = list(tensor_a)
        seq_b = list(tensor_b)
        assert len(seq_a) == len(seq_b), f"{msg} Length mismatch: {len(seq_a)} vs {len(seq_b)}"
        for t1, t2 in zip(seq_a, seq_b, strict=True):
            assert torch.equal(t1, t2), f"{msg} Tensors are not equal: {tensor_a} vs {tensor_b}"
    else:
        assert torch.equal(tensor_a, tensor_b), f"{msg} Tensors are not equal: {tensor_a} vs {tensor_b}"


# ---------------------------------------------------------------------------
# basic save / load roundtrip
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_save_creates_expected_files(self, tq_system, checkpoint_dir):
        keys = ["k0", "k1"]
        partition_id = "p0"
        _put_batch(keys, partition_id, torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2))

        tq.save_checkpoint(checkpoint_dir)

        assert (checkpoint_dir / "metadata.json").exists()
        assert (checkpoint_dir / "controller_state.pkl").exists()

        with open(checkpoint_dir / "metadata.json") as f:
            info = json.load(f)

        assert info["storage_saved"] is True
        su_dir = checkpoint_dir / "storage_units"
        assert su_dir.exists()
        assert (su_dir / "su_info.json").exists()

    def test_metadata_json_content(self, tq_system, checkpoint_dir):
        keys = ["m0"]
        _put_batch(keys, "p_meta", torch.tensor([[10, 20]]), torch.ones(1, 2))

        tq.save_checkpoint(checkpoint_dir, metadata={"iteration": 42, "loss": 0.5})

        with open(checkpoint_dir / "metadata.json") as f:
            meta = json.load(f)

        assert meta["user_metadata"]["iteration"] == 42
        assert meta["user_metadata"]["loss"] == pytest.approx(0.5)
        assert "storage_saved" in meta

    def test_load_restores_controller_partitions(self, tq_system, checkpoint_dir, controller):
        keys = ["a0", "a1", "a2"]
        partition_id = "p_ctrl"
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        tags = [{"idx": i} for i in range(3)]
        _put_batch(keys, partition_id, input_ids, torch.ones(3, 3), tags)

        tq.save_checkpoint(checkpoint_dir)

        # wipe controller state
        ray.get(controller.clear_partition.remote(partition_id))
        assert ray.get(controller.list_partitions.remote()) == []

        tq.load_checkpoint(checkpoint_dir)

        # partition must be back
        partitions = ray.get(controller.list_partitions.remote())
        assert partition_id in partitions

        # key-to-global-index mapping must be intact
        snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
        for key in keys:
            assert key in snapshot.keys_mapping

        # tags must be intact
        for i, key in enumerate(keys):
            gidx = snapshot.keys_mapping[key]
            assert snapshot.custom_meta[gidx]["idx"] == i

    def test_load_restores_storage_data(self, tq_system, checkpoint_dir, controller):
        keys = ["s0", "s1"]
        partition_id = "p_storage"
        input_ids = torch.tensor([[10, 20], [30, 40]])
        attention_mask = torch.ones(2, 2)
        _put_batch(keys, partition_id, input_ids, attention_mask)

        tq.save_checkpoint(checkpoint_dir)

        # clear both controller and storage state so load has to restore from scratch
        ray.get(controller.clear_partition.remote(partition_id))

        tq.load_checkpoint(checkpoint_dir)

        retrieved = _get_batch(keys, partition_id)
        assert_tensor_equal(retrieved["input_ids"], input_ids)
        assert_tensor_equal(retrieved["attention_mask"], attention_mask)

    def test_load_restores_multiple_partitions(self, tq_system, checkpoint_dir, controller):
        for i in range(3):
            _put_batch(
                [f"p{i}_k0", f"p{i}_k1"],
                f"part_{i}",
                torch.full((2, 4), i, dtype=torch.long),
                torch.ones(2, 4),
            )

        tq.save_checkpoint(checkpoint_dir)

        for i in range(3):
            ray.get(controller.clear_partition.remote(f"part_{i}"))

        tq.load_checkpoint(checkpoint_dir)

        for i in range(3):
            retrieved = tq.kv_batch_get(
                keys=[f"p{i}_k0", f"p{i}_k1"],
                partition_id=f"part_{i}",
                select_fields=["input_ids"],
            )
            assert_tensor_equal(retrieved["input_ids"], torch.full((2, 4), i, dtype=torch.long))


# ---------------------------------------------------------------------------
# include_storage=False
# ---------------------------------------------------------------------------


class TestCheckpointMetadataOnly:
    def test_save_include_storage_false_simplestorage_override(self, tq_system, checkpoint_dir):
        """For SimpleStorage, include_storage=False is overridden to True because in-memory
        data would be lost on restart. storage_saved must be True and storage_units must exist."""
        _put_batch(["n0"], "p_nometa", torch.tensor([[1, 2]]), torch.ones(1, 2))

        tq.save_checkpoint(checkpoint_dir, include_storage=False)

        with open(checkpoint_dir / "metadata.json") as f:
            info = json.load(f)

        assert info["storage_saved"] is True
        assert (checkpoint_dir / "storage_units").exists()

    def test_load_after_include_storage_false_restores_both(self, tq_system, checkpoint_dir, controller):
        """With SimpleStorage, include_storage=False is forced True, so both controller and
        storage are saved and restored."""
        keys = ["n0", "n1"]
        partition_id = "p_nometa2"
        input_ids = torch.tensor([[5, 6], [7, 8]])
        _put_batch(keys, partition_id, input_ids, torch.ones(2, 2))

        tq.save_checkpoint(checkpoint_dir, include_storage=False)

        ray.get(controller.clear_partition.remote(partition_id))

        tq.load_checkpoint(checkpoint_dir)

        partitions = ray.get(controller.list_partitions.remote())
        assert partition_id in partitions

        snapshot = ray.get(controller.get_partition_snapshot.remote(partition_id))
        for key in keys:
            assert key in snapshot.keys_mapping

        retrieved = _get_batch(keys, partition_id)
        assert_tensor_equal(retrieved["input_ids"], input_ids)


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


class TestCheckpointErrors:
    def test_save_raises_if_not_initialized(self, tmp_path):
        # call save_checkpoint before tq.init() in a fresh module state
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
        _put_batch(["e0"], "p_err", torch.tensor([[1, 2]]), torch.ones(1, 2))
        tq.save_checkpoint(checkpoint_dir)

        # tamper: add a fake extra SU entry in su_info.json so count differs
        su_info_path = checkpoint_dir / "storage_units" / "su_info.json"
        with open(su_info_path) as f:
            su_info = json.load(f)
        su_info.append({"position": 99, "storage_unit_id": "fake"})
        with open(su_info_path, "w") as f:
            json.dump(su_info, f)

        with pytest.raises(ValueError, match="count mismatch"):
            tq.load_checkpoint(checkpoint_dir)

    def test_no_partial_state_on_failed_save(self, tq_system, tmp_path):
        """A failed save must not leave a partial directory."""
        _put_batch(["f0"], "p_fail", torch.tensor([[1, 2]]), torch.ones(1, 2))

        ck = tmp_path / "ck"

        import unittest.mock as mock

        with mock.patch(
            "transfer_queue.client.TransferQueueClient.save_storage_checkpoint",
            side_effect=RuntimeError("simulated dump failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated dump failure"):
                tq.save_checkpoint(ck)

        assert not ck.exists(), "Partial checkpoint directory should have been cleaned up"
        assert not (tmp_path / "ck.tmp").exists(), "Temp directory should have been cleaned up"


# ---------------------------------------------------------------------------
# data variety
# ---------------------------------------------------------------------------


class TestCheckpointDataVariety:
    def test_non_tensor_fields_roundtrip(self, tq_system, checkpoint_dir, controller):
        """String fields should survive save/load."""
        from tensordict import NonTensorStack

        keys = ["t0", "t1"]
        partition_id = "p_str"
        fields = TensorDict(
            {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "text": NonTensorStack("hello", "world"),
            },
            batch_size=2,
        )
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=[{}, {}])

        tq.save_checkpoint(checkpoint_dir)

        ray.get(controller.clear_partition.remote(partition_id))

        tq.load_checkpoint(checkpoint_dir)

        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["input_ids"])
        assert_tensor_equal(retrieved["input_ids"], torch.tensor([[1, 2], [3, 4]]))

    def test_nested_tensor_fields_roundtrip(self, tq_system, checkpoint_dir, controller):
        """Variable-length (jagged) tensor fields should survive save/load."""
        keys = ["j0", "j1", "j2"]
        partition_id = "p_jagged"
        for i, key in enumerate(keys):
            seq = torch.arange(i + 1, dtype=torch.float).unsqueeze(0)
            tq.kv_put(
                key=key,
                partition_id=partition_id,
                fields=TensorDict({"seq": seq}, batch_size=1),
                tag=None,
            )

        tq.save_checkpoint(checkpoint_dir)

        ray.get(controller.clear_partition.remote(partition_id))

        tq.load_checkpoint(checkpoint_dir)

        retrieved = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=["seq"])
        for i, component in enumerate(retrieved["seq"].unbind()):
            assert_tensor_equal(component, torch.arange(i + 1, dtype=torch.float))
