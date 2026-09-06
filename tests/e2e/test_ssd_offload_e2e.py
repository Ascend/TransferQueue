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

"""End-to-end coverage for SimpleStorage local SSD offload."""

import pytest
import ray
import torch
from omegaconf import OmegaConf

import transfer_queue as tq


@pytest.fixture(scope="module")
def ssd_root(tmp_path_factory):
    """Return an isolated SSD root shared by this E2E module."""
    return tmp_path_factory.mktemp("tq-ssd-offload")


@pytest.fixture(scope="module")
def tq_system(ssd_root):
    """Initialize a public TransferQueue system with SSD offload enabled."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

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
                        "path": str(ssd_root),
                    },
                },
            },
        }
    )
    tq.init(config)
    yield
    tq.close()
    assert not list(ssd_root.rglob("*.bin"))
    assert not any(path.is_dir() for path in ssd_root.iterdir())
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def controller(tq_system):
    """Return the controller used to clean test partitions."""
    return ray.get_actor("TransferQueueController", namespace="transfer_queue")


@pytest.fixture(autouse=True)
def cleanup_partitions(controller):
    """Remove every partition created by an E2E test."""
    yield
    for partition_id in ray.get(controller.list_partitions.remote()):
        ray.get(controller.clear_partition.remote(partition_id))


def test_public_api_routes_per_sample_and_migrates_on_overwrite(tq_system, ssd_root):
    """Public KV operations preserve values while samples migrate between tiers."""
    partition_id = "ssd-routing"
    small = torch.arange(16, dtype=torch.float32)
    large = torch.arange(262144, dtype=torch.float32)

    tq.kv_put(key="small", partition_id=partition_id, fields={"value": small})
    tq.kv_put(key="large", partition_id=partition_id, fields={"value": large})

    torch.testing.assert_close(
        tq.kv_batch_get(keys=["small"], partition_id=partition_id)["value"][0],
        small,
    )
    torch.testing.assert_close(
        tq.kv_batch_get(keys=["large"], partition_id=partition_id)["value"][0],
        large,
    )
    assert len(list(ssd_root.rglob("*.bin"))) == 1

    tq.kv_put(key="small", partition_id=partition_id, fields={"value": large})
    assert len(list(ssd_root.rglob("*.bin"))) == 2

    tq.kv_put(key="large", partition_id=partition_id, fields={"value": small})
    assert len(list(ssd_root.rglob("*.bin"))) == 1
    torch.testing.assert_close(
        tq.kv_batch_get(keys=["large"], partition_id=partition_id)["value"][0],
        small,
    )

    tq.kv_clear(keys=["small"], partition_id=partition_id)
    assert not list(ssd_root.rglob("*.bin"))


def test_checkpoint_round_trip_recreates_ssd_data(tq_system, ssd_root, tmp_path):
    """Checkpoint restore recreates logical data without depending on old SSD files."""
    partition_id = "ssd-checkpoint"
    key = "large"
    value = torch.arange(262144, dtype=torch.float32)
    checkpoint_dir = tmp_path / "checkpoint"

    tq.kv_put(key=key, partition_id=partition_id, fields={"value": value})
    assert list(ssd_root.rglob("*.bin"))
    tq.save_checkpoint(checkpoint_dir)

    tq.kv_clear(keys=[key], partition_id=partition_id)
    assert not list(ssd_root.rglob("*.bin"))
    tq.load_checkpoint(checkpoint_dir)

    restored = tq.kv_batch_get(keys=[key], partition_id=partition_id)["value"][0]
    torch.testing.assert_close(restored, value)
    assert list(ssd_root.rglob("*.bin"))
