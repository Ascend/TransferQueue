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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in multi-node E2E coverage for node-local SimpleStorage SSD offload."""

import os
from pathlib import Path

import pytest
import ray
import torch
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict

import transfer_queue as tq

_SSD_ROOT = os.environ.get("TQ_SSD_E2E_ROOT")


@ray.remote
def _local_ssd_file_count(root: str) -> int:
    """Count SSD sample files on the current Ray node."""
    return len(list(Path(root).rglob("*.bin")))


def _counts_by_node(root: str) -> list[int]:
    active_nodes = [node for node in ray.nodes() if node["Alive"]]
    return ray.get(
        [
            _local_ssd_file_count.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node["NodeID"],
                    soft=False,
                )
            ).remote(root)
            for node in active_nodes
        ]
    )


@pytest.mark.skipif(
    _SSD_ROOT is None,
    reason="set TQ_SSD_E2E_ROOT to a node-local SSD path",
)
def test_multinode_ssd_offload_uses_and_cleans_each_node():
    """Exercise SSD-backed public KV operations across two Ray nodes."""
    ray.init(ignore_reinit_error=True)
    alive_nodes = [node for node in ray.nodes() if node["Alive"]]
    assert len(alive_nodes) >= 2

    config = OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": 20,
                    "num_data_storage_units": 2,
                    "ssd_offload": {
                        "enabled": True,
                        "path": _SSD_ROOT,
                    },
                },
            },
        }
    )
    keys = [f"sample-{index}" for index in range(4)]
    values = torch.arange(4 * 262144, dtype=torch.float32).view(4, 262144)
    partition_id = "ssd-multinode"

    try:
        tq.init(config)
        tq.kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=TensorDict({"value": values}, batch_size=4),
        )
        restored = tq.kv_batch_get(keys=keys, partition_id=partition_id)["value"]
        for actual, expected in zip(restored.unbind(), values.unbind(), strict=True):
            torch.testing.assert_close(actual, expected)

        counts = _counts_by_node(_SSD_ROOT)
        assert sum(counts) == len(keys)
        assert sum(count > 0 for count in counts) >= 2

        tq.kv_clear(keys=keys, partition_id=partition_id)
        assert sum(_counts_by_node(_SSD_ROOT)) == 0
    finally:
        tq.close()

    assert sum(_counts_by_node(_SSD_ROOT)) == 0
