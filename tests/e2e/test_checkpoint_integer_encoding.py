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

import copy
import tempfile

import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

import transfer_queue as tq


@pytest.mark.parametrize("compact_fields", [[], ["routed_experts"]])
def test_native_checkpoint_preserves_finished_and_pending_rows(tmp_path, compact_fields):
    ray.init(
        address="local",
        num_cpus=3,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        _memory=1024**3,
        _temp_dir=tempfile.mkdtemp(prefix="tq-codec-"),
    )
    try:
        tq.init(
            OmegaConf.create(
                {
                    "controller": {"polling_mode": True},
                    "backend": {
                        "storage_backend": "SimpleStorage",
                        "SimpleStorage": {
                            "total_storage_size": 20,
                            "num_data_storage_units": 1,
                            "checkpoint_uint8_fields": compact_fields,
                        },
                    },
                }
            )
        )
        keys = ["finished", "pending"]
        prompt = [{"role": "user", "content": "The complete pending prompt and tool state."}]
        routes = torch.arange(16384, dtype=torch.int16).remainder(16).reshape(2, 128, 16, 4)
        fields = TensorDict(
            {
                "routed_experts": routes,
                "input_ids": torch.arange(256).reshape(2, 128),
                "loss_mask": torch.ones(2, 128, dtype=torch.bool),
                "rollout_log_probs": torch.zeros(2, 128),
                "rm_scores": torch.tensor([[1.0], [0.0]]),
                "response_weight_versions": torch.full((2, 128), 2),
                "raw_prompt": NonTensorStack(copy.deepcopy(prompt), copy.deepcopy(prompt)),
                "tools_kwargs": NonTensorStack({"task": {"id": "a"}}, {"task": {"id": "b"}}),
            },
            batch_size=2,
        )
        tq.kv_batch_put(
            keys=keys,
            partition_id="train",
            fields=fields,
            tags=[
                {"is_prompt": True, "status": "finished", "global_steps": 2},
                {"is_prompt": True, "status": "running", "global_steps": 2},
            ],
        )
        tags_before = copy.deepcopy(tq.kv_list("train"))
        before = tq.kv_batch_get(keys=keys, partition_id="train")
        checkpoint = tmp_path / "checkpoint"
        tq.save_checkpoint(checkpoint)
        controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
        ray.get(controller.clear_partition.remote("train"))
        tq.load_checkpoint(checkpoint)
        assert tq.kv_list("train") == tags_before
        after = tq.kv_batch_get(keys=keys, partition_id="train")
        for field in fields.keys():
            if field in ("raw_prompt", "tools_kwargs"):
                assert after[field].tolist() == before[field].tolist()
            else:
                assert after[field].dtype == before[field].dtype
                if after[field].is_nested:
                    assert torch.equal(after[field].offsets(), before[field].offsets())
                    for actual, expected in zip(after[field].unbind(), before[field].unbind(), strict=True):
                        assert actual.shape == expected.shape
                        assert torch.equal(actual, expected)
                else:
                    assert torch.equal(after[field], before[field])
    finally:
        tq.close()
        ray.shutdown()
