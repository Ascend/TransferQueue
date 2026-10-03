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

"""A Ray worker that attaches to TransferQueue and closes it must not tear it down for the driver."""

import time

import pytest
import ray
import torch
from omegaconf import OmegaConf

import transfer_queue as tq

TIMEOUT_S = 60
CONF = {
    "controller": {"polling_mode": True},
    "backend": {"storage_backend": "SimpleStorage", "SimpleStorage": {"total_storage_size": 100}},
}


@ray.remote
def attach_use_and_close() -> int:
    tq.init()
    tq.kv_put("from_worker", "close_e2e", fields={"v": torch.tensor([7])})
    value = int(tq.kv_batch_get("from_driver", "close_e2e")["v"][0])
    tq.close()
    return value


@pytest.fixture
def ray_session():
    if not ray.is_initialized():
        ray.init(namespace="TestCloseOwnershipE2E")
    yield
    ray.shutdown()


def test_only_the_creating_process_tears_down(ray_session):
    tq.init(OmegaConf.create(CONF))
    tq.kv_put("from_driver", "close_e2e", fields={"v": torch.tensor([3])})

    assert ray.get(attach_use_and_close.remote(), timeout=TIMEOUT_S) == 3

    controller = ray.get_actor("TransferQueueController", namespace="transfer_queue")
    ray.get(controller.list_partitions.remote(), timeout=TIMEOUT_S)
    assert int(tq.kv_batch_get("from_worker", "close_e2e")["v"][0]) == 7

    tq.close()
    deadline = time.monotonic() + TIMEOUT_S
    while True:
        try:
            ray.get_actor("TransferQueueController", namespace="transfer_queue")
        except ValueError:
            break
        assert time.monotonic() < deadline, "controller still alive after the owner's close()"
        time.sleep(0.1)
