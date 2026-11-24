# Copyright 2026 The TransferQueue Team
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

import logging
from concurrent.futures import ThreadPoolExecutor
from shutil import copytree
from threading import Barrier
from time import monotonic, sleep
from urllib.request import ProxyHandler, build_opener
from uuid import uuid4

import pytest
import ray
import torch
from msgspec import msgpack
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq
from transfer_queue.utils.checkpoint_utils import validate_checkpoint


@pytest.fixture
def ray_cluster(monkeypatch):
    monkeypatch.setenv("TQ_PRE_ALLOC_SAMPLE_NUM", "1")
    monkeypatch.setenv("TQ_METRICS_COLLECT_INTERVAL", "1")
    started_here = not ray.is_initialized()
    if started_here:
        ray.init(address="local", num_cpus=2, include_dashboard=False)
    yield
    if started_here:
        ray.shutdown()


def test_rust_controller_init_bridge_fifo_and_metrics(monkeypatch, ray_cluster, caplog):
    namespace = f"tq-rust-controller-{uuid4().hex}"
    monkeypatch.setenv("TQ_RAY_NAMESPACE", namespace)
    monkeypatch.delenv("TQ_ENABLE_KV_GLOBAL_LOCK", raising=False)
    try:
        # TQ_CONTROLLER_BACKEND takes precedence over an explicit `controller.backend` in `conf`,
        # and custom sampler configuration logs a warning and falls back to built-in FIFO sampling.
        monkeypatch.setenv("TQ_CONTROLLER_BACKEND", "rust")
        with caplog.at_level(logging.WARNING, logger="transfer_queue.interface"):
            final_conf = tq.init(
                OmegaConf.create(
                    {
                        "controller": {"backend": "python", "sampler": tq.GRPOGroupNSampler},
                        "metrics": {"enabled": True},
                        "backend": {"SimpleStorage": {"num_data_storage_units": 1, "total_storage_size": 128}},
                    },
                    flags={"allow_objects": True},
                )
            )
        assert final_conf.controller.backend == "rust"
        assert any("does not support custom sampler" in record.message for record in caplog.records)

        actor = ray.get_actor("TransferQueueController", namespace=namespace)
        assert ray.get(actor.get_config.remote()).controller.backend == "rust"

        client = tq.get_client()
        assert client.get_production_status(["sample"], "absent") == (None, None)
        assert client.get_consumption_status("trainer", "absent") == (None, None)

        allocated = client.get_meta(["sample"], 64, "train", mode="insert")
        allocated.update_custom_meta([{"seq": i} for i in range(64)])
        client.put(TensorDict({"sample": torch.arange(64).unsqueeze(1)}, batch_size=[64]), allocated)
        indexes, ready = client.get_production_status(["sample"], "train")
        assert indexes.dtype == torch.long and ready.dtype == torch.int8
        assert indexes.tolist() == allocated.global_indexes and ready.all()

        start = Barrier(16, timeout=10)
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [
                pool.submit(lambda: (start.wait(), client.get_meta(["sample"], 4, "train", task_name="trainer"))[1])
                for _ in range(16)
            ]
            batches = [future.result(timeout=15) for future in futures]
        assert sorted(idx for batch in batches for idx in batch.global_indexes) == allocated.global_indexes

        _, consumed = client.get_consumption_status("trainer", "train")
        assert consumed.dtype == torch.int8 and consumed.all()
        assert sorted(ray.get(actor.get_partition_index_range.remote("train"))) == allocated.global_indexes
        snapshot = ray.get(actor.get_partition_snapshot.remote("train"))
        assert snapshot.consumption_status["trainer"].all()
        assert snapshot.custom_meta[allocated.global_indexes[0]] == {"seq": 0}

        # Verify built-in insertion-order FIFO across recycled low GlobalIndex slots:
        # clear the first 4 slots, insert 4 brand-new samples (which reuse those low GlobalIndexes),
        # and verify a fresh consumer task receives the older 60 samples BEFORE the 4 newcomers.
        early_slice = allocated.select_samples([0, 1, 2, 3])
        client.clear_samples(early_slice)
        newcomer_meta = client.get_meta(["sample"], 4, "train", mode="insert")
        assert newcomer_meta.global_indexes == early_slice.global_indexes
        client.put(TensorDict({"sample": torch.arange(100, 104).unsqueeze(1)}, batch_size=[4]), newcomer_meta)

        older_fetched = client.get_meta(["sample"], 60, "train", task_name="fifo_reader")
        assert older_fetched.global_indexes == allocated.global_indexes[4:]
        newer_fetched = client.get_meta(["sample"], 4, "train", task_name="fifo_reader")
        assert newer_fetched.global_indexes == newcomer_meta.global_indexes

        _assert_controller_metrics(tq.get_metrics_endpoint(), partition_id="train", samples=64.0, get_meta_total=20.0)

        tq.close()
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(actor.get_config.remote(), timeout=10)
    finally:
        tq.close()


def test_rust_checkpoint_validation_and_cross_backend_rejection(monkeypatch, ray_cluster, tmp_path):
    monkeypatch.delenv("TQ_CONTROLLER_BACKEND", raising=False)
    partition = "checkpoint"
    python_checkpoint = tmp_path / "python"
    rust_checkpoint = tmp_path / "rust"
    missing_field_checkpoint = tmp_path / "missing_field"

    for backend in ("python", "rust"):
        namespace = f"tq-checkpoint-{uuid4().hex}"
        monkeypatch.setenv("TQ_RAY_NAMESPACE", namespace)
        tq.init(
            OmegaConf.create(
                {
                    "controller": {"backend": backend},
                    "backend": {"SimpleStorage": {"num_data_storage_units": 1, "total_storage_size": 8}},
                }
            )
        )
        try:
            if backend == "python":
                tq.kv_put("local", partition, {"value": torch.tensor([11])}, tag={"name": "python"})
                tq.save_checkpoint(python_checkpoint)
                continue

            actor = ray.get_actor("TransferQueueController", namespace=namespace)
            tq.kv_put("local", partition, {"value": torch.tensor([1])}, tag={"name": "local"})
            tq.kv_put("shared", tq.PUBLIC_PARTITION, {"value": torch.tensor([2])})
            tq.kv_link("linked", partition, "shared", "value")
            created_at = ray.get(actor.get_partition_snapshot.remote(partition)).created_at
            tq.save_checkpoint(rust_checkpoint)

            assert validate_checkpoint(rust_checkpoint, "rust", storage_units=1) is True
            assert validate_checkpoint(rust_checkpoint, "rust", storage_units=2) is False
            assert validate_checkpoint(rust_checkpoint, "python") is False
            assert validate_checkpoint(python_checkpoint, "rust") is False

            # Strip a nested PartitionState field (`active`) to verify preflight uses Rust's typed decoder.
            copytree(rust_checkpoint, missing_field_checkpoint)
            ctrl_path = missing_field_checkpoint / "controller_state.msgpack"
            raw = ctrl_path.read_bytes()
            decoded = msgpack.decode(raw[8:], strict=False)
            del decoded["partitions"][partition]["active"]
            ctrl_path.write_bytes(raw[:8] + msgpack.encode(decoded))
            assert validate_checkpoint(missing_field_checkpoint, "rust") is False

            tq.kv_put("local", partition, {"value": torch.tensor([99])}, tag={"step": 99})
            with pytest.raises(ValueError, match="incompatible with the rust backend"):
                tq.load_checkpoint(python_checkpoint)
            with pytest.raises(RuntimeError, match="Invalid controller checkpoint"):
                tq.load_checkpoint(missing_field_checkpoint)
            assert tq.kv_batch_get(["local"], partition)["value"][0].tolist() == [99]
            assert tq.kv_list(partition)[partition]["local"] == {"name": "local", "step": 99}

            tq.load_checkpoint(rust_checkpoint)
            assert ray.get(actor.get_partition_snapshot.remote(partition)).created_at == created_at
            assert tq.kv_batch_get(["local"], partition)["value"][0].tolist() == [1]
            assert tq.kv_list(partition)[partition]["local"] == {"name": "local"}
            assert tq.kv_describe("linked", partition)["linked"]["links"] == {"value": "shared"}
            tq.kv_clear("linked", partition)
            marked_meta, marked_keys = ray.get(actor.mark_unlinked_public_clearing.remote())
            assert marked_keys == ["shared"] and marked_meta.size == 1
        finally:
            tq.close()


def _assert_controller_metrics(
    endpoint: str | None, *, partition_id: str, samples: float, get_meta_total: float
) -> None:
    assert endpoint is not None
    opener = build_opener(ProxyHandler({}))
    deadline = monotonic() + 12
    while True:
        with opener.open(f"http://{endpoint}/metrics", timeout=3) as response:
            metrics = response.read().decode()
        if f'tq_partition_samples_total{{partition_id="{partition_id}"}} {samples}' in metrics:
            assert f'tq_controller_request_total{{op_type="GET_META"}} {get_meta_total}' in metrics
            return
        assert monotonic() < deadline, metrics
        sleep(0.2)
