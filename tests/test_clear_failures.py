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

"""All-shard cleanup controls with live payloads and real controller ownership."""

import asyncio
import pickle
from uuid import uuid4

import pytest
import ray
import torch
from tensordict import TensorDict

from transfer_queue.client import TransferQueueClient
from transfer_queue.controller import TransferQueueController
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


@ray.remote(num_cpus=1)
class RejectingClearUnit(SimpleStorageUnit.__ray_metadata__.modified_class):
    reject_clear = False

    def set_clear_failure(self, enabled):
        self.reject_clear = enabled

    def _handle_clear(self, request):
        if self.reject_clear:
            return ZMQMessage.create(
                request_type=ZMQRequestType.CLEAR_DATA_ERROR,
                sender_id=self.storage_unit_id,
                body={"message": "intentional live-shard clear failure"},
            )
        return super()._handle_clear(request)


@pytest.fixture(scope="module")
def ray_services():
    assert not ray.is_initialized(), "requires a test-owned local Ray instance"
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=256 * 1024 * 1024,
        namespace="clear-" + uuid4().hex,
    )
    yield
    ray.shutdown()


@pytest.fixture
def queue(ray_services):
    controller = TransferQueueController.remote(polling_mode=True)
    units = [RejectingClearUnit.remote(config={"num_data_storage_units": 2}) for _ in range(2)]
    infos = ray.get([controller.get_zmq_server_info.remote(), *[unit.get_zmq_server_info.remote() for unit in units]])
    client = TransferQueueClient(client_id="clear-" + uuid4().hex, controller_info=infos[0])
    client.initialize_storage_manager("SimpleStorage", {"zmq_info": {info.id: info for info in infos[1:]}})
    try:
        yield client, controller, units, infos[1:]
    finally:
        client.close()
        try:
            ray.get([unit.shutdown.remote() for unit in units], timeout=10)
        finally:
            for unit in units:
                ray.kill(unit, no_restart=True)
            ray.kill(controller, no_restart=True)


def _snapshot(client, path):
    client.save_controller_checkpoint(str(path))
    return pickle.loads(path.read_bytes())


def _put(client):
    return client.put(TensorDict({"value": torch.tensor([[11], [22]])}, [2]), partition_id="partial")


@pytest.mark.parametrize("operation", ["samples", "partition"])
@pytest.mark.parametrize("failed_position", [0, 1])
def test_failed_live_shard_clear_keeps_ownership_and_settles_survivor(queue, tmp_path, operation, failed_position):
    client, _, units, infos = queue
    meta = _put(client)
    assert len(meta.global_indexes) == 2
    ray.get(units[failed_position].set_clear_failure.remote(True))
    if operation == "samples":
        clear = lambda: client.clear_samples(meta)
    else:
        clear = lambda: client.clear_partition("partial")
    with pytest.raises(RuntimeError, match="intentional live-shard clear failure"):
        clear()

    state = _snapshot(client, tmp_path / "failed.pkl")
    owned = set(meta.global_indexes)
    assert owned.issubset(state["index_manager"]["allocated_indexes"])
    assert not owned.intersection(state["index_manager"]["reusable_indexes"])
    assert state["partitions"]["partial"].global_indexes == owned
    survivor = units[1 - failed_position]
    metrics = ray.get(survivor._handle_get_metrics.remote()).body
    assert metrics["active_keys"] == 0
    failed_id = infos[failed_position].id
    group = client.storage_manager._group_by_hash(meta.global_indexes)[failed_id]
    _, values = asyncio.run(
        client.storage_manager._get_from_single_storage_unit(
            group.global_indexes, ["value"], target_storage_unit=failed_id
        )
    )
    assert [v.item() for v in values["value"]] == [11 if failed_position == 0 else 22]

    # Retry is idempotent for the already-cleared shard and releases ownership only on success.
    ray.get(units[failed_position].set_clear_failure.remote(False))
    clear()
    state = _snapshot(client, tmp_path / "success.pkl")
    assert not owned.intersection(state["index_manager"]["allocated_indexes"])
    assert owned.issubset(state["index_manager"]["reusable_indexes"])
    assert all(ray.get(unit._handle_get_metrics.remote()).body["active_keys"] == 0 for unit in units)


@pytest.mark.parametrize("operation", ["samples", "partition"])
def test_successful_all_shard_clear_releases_ownership(queue, tmp_path, operation):
    client, _, units, _ = queue
    meta = _put(client)
    assert [sample.item() for sample in client.get_data(meta)["value"].unbind()] == [11, 22]
    if operation == "samples":
        client.clear_samples(meta)
    else:
        client.clear_partition("partial")
    state = _snapshot(client, tmp_path / "cleared.pkl")
    assert not set(meta.global_indexes).intersection(state["index_manager"]["allocated_indexes"])
    assert set(meta.global_indexes).issubset(state["index_manager"]["reusable_indexes"])
    assert all(ray.get(unit._handle_get_metrics.remote()).body["active_keys"] == 0 for unit in units)
