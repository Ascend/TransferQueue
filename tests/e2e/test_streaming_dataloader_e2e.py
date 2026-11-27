# Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""StreamingDataLoader integration with real metadata sampling and storage."""

import os
from itertools import islice

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq
from transfer_queue.dataloader.streaming_dataloader import StreamingDataLoader
from transfer_queue.dataloader.streaming_dataset import StreamingDataset, default_fetch_batch_fn


@pytest.fixture(scope="module")
def tq_system():
    """Run the rank-aware loader against two SimpleStorage units."""
    if os.environ.get("TQ_CONTROLLER_BACKEND") == "rust":
        pytest.skip("RankAwareSampler requires the Python controller")
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    config = OmegaConf.create(
        {
            "controller": {"sampler": tq.RankAwareSampler},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {"num_data_storage_units": 2, "total_storage_size": 64},
            },
        },
        flags={"allow_objects": True},
    )
    try:
        yield tq.init(config)
    finally:
        tq.close()
        if owns_ray:
            ray.shutdown()


def test_rank_replay_and_loader_reset_step_preserve_microbatches(tq_system):
    """Same-DP replicas and replayed microbatches must keep the same row identity."""
    client = tq.get_client()
    rows = [torch.arange(i + 1, dtype=torch.int64) + i * 10 for i in range(4)]
    client.put(
        TensorDict({"tokens": torch.nested.as_nested_tensor(rows, layout=torch.jagged)}, [4]), partition_id="train_0"
    )
    fetches = []

    def fetch(**kwargs):
        data, meta = default_fetch_batch_fn(**kwargs)
        fetches.append(meta.global_indexes)
        return data, meta

    def split(data, meta, micro_batch_size):
        for start in range(0, meta.size, micro_batch_size):
            yield (
                {"tokens": list(data["tokens"].unbind())[start : start + micro_batch_size]},
                meta[start : start + micro_batch_size],
            )

    dataset = StreamingDataset(
        config=tq_system,
        batch_size=4,
        micro_batch_size=2,
        data_fields=["tokens"],
        partition_id="train_0",
        task_name="actor",
        dp_rank=0,
        fetch_batch_fn=fetch,
        process_batch_fn=split,
    )
    loader = StreamingDataLoader(dataset)

    try:
        batches = list(islice(loader, 2))
        indexes = [index for _, meta in batches for index in meta.global_indexes]
        replica_client = tq.TransferQueueClient("replica", tq_system.controller.zmq_info)
        try:
            replica = replica_client.get_meta(
                ["tokens"],
                4,
                "train_0",
                task_name="actor",
                sampling_config={"dp_rank": 0, "batch_index": 0, "task_name": "actor", "partition_id": "train_0"},
            )
        finally:
            replica_client.close()
        assert replica.global_indexes == indexes
        assert len(fetches) == 1
        loader.reset()
        replay = list(islice(loader, 2))
        assert [meta.global_indexes for _, meta in replay] == [meta.global_indexes for _, meta in batches]
        assert len(fetches) == 1
        for actual, expected in zip([row for data, _ in replay for row in data["tokens"]], rows, strict=True):
            torch.testing.assert_close(actual, expected)

        next_rows = [row + 100 for row in rows]
        client.put(
            TensorDict({"tokens": torch.nested.as_nested_tensor(next_rows, layout=torch.jagged)}, [4]),
            partition_id="train_1",
        )
        loader.step("train_1")
        assert loader.get_buffer() == []
        next_batches = list(islice(loader, 2))
        assert len(fetches) == 2
        assert all(pid == "train_1" for _, meta in next_batches for pid in meta.partition_ids)
        received_rows = [row for data, _ in next_batches for row in data["tokens"]]
        for actual, expected in zip(received_rows, next_rows, strict=True):
            torch.testing.assert_close(actual, expected)
    finally:
        if dataset._tq_client is not None:
            dataset._tq_client.close()
