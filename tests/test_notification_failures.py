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

"""Public producer controls with real controller and storage services."""

import asyncio
import os
import time
from uuid import uuid4

import psutil
import pytest
import ray
import torch
from tensordict import TensorDict

from transfer_queue.client import TransferQueueClient
from transfer_queue.controller import TransferQueueController
from transfer_queue.storage.managers import base
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.zmq_utils import ZMQRequestType


@ray.remote(num_cpus=1)
class NotificationController(TransferQueueController.__ray_metadata__.modified_class):
    response_mode = "normal"

    def set_response_mode(self, mode):
        self.response_mode = mode

    def process_id(self):
        return os.getpid()

    def _handle_notify_data_update_request(self, request):
        if self.response_mode == "normal":
            return super()._handle_notify_data_update_request(request)
        response = self._make_response(request, ZMQRequestType.NOTIFY_DATA_UPDATE_ACK, {"success": True})
        if self.response_mode == "wrong_sender":
            response.sender_id = "unexpected_controller"
        elif self.response_mode == "missing_success":
            response.body = {}
        elif self.response_mode == "error":
            response.request_type = ZMQRequestType.REQUEST_ERROR
            response.body = {"message": "notification rejected by control"}
        elif self.response_mode == "unexpected_type":
            response.request_type = ZMQRequestType.GET_META_RESPONSE
        return response


@pytest.fixture(scope="module")
def ray_services():
    assert not ray.is_initialized(), "requires a test-owned local Ray instance"
    ray.init(
        address="local",
        num_cpus=4,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=256 * 1024 * 1024,
        namespace="notify-" + uuid4().hex,
    )
    yield
    ray.shutdown()


@pytest.fixture
def queue(ray_services, monkeypatch):
    monkeypatch.setattr(base, "TQ_DATA_UPDATE_RESPONSE_TIMEOUT", 2)
    controller = NotificationController.remote(polling_mode=True)
    storage = SimpleStorageUnit.remote(config={"num_data_storage_units": 1})
    info, storage_info = ray.get([controller.get_zmq_server_info.remote(), storage.get_zmq_server_info.remote()])
    client = TransferQueueClient(client_id="producer-" + uuid4().hex, controller_info=info)
    client.initialize_storage_manager("SimpleStorage", {"zmq_info": {storage_info.id: storage_info}})
    try:
        yield client, controller
    finally:
        client.close()
        try:
            ray.get(storage.shutdown.remote(), timeout=10)
        finally:
            ray.kill(storage, no_restart=True)
            ray.kill(controller, no_restart=True)


def _data():
    return TensorDict({"value": torch.tensor([[22]])}, batch_size=1)


def test_successful_ack_completes_public_put(queue):
    client, controller = queue
    meta = client.put(_data(), partition_id="valid")
    assert [sample.item() for sample in client.get_data(meta)["value"].unbind()] == [22]
    partition = ray.get(controller.get_partition_snapshot.remote("valid"))
    assert partition.global_indexes == set(meta.global_indexes)
    assert partition.production_status[meta.global_indexes, : partition.total_fields_num].eq(1).all()


def test_real_negative_ack_reaches_public_put(queue):
    client, controller = queue
    meta = client.put(_data(), partition_id="removed")
    client.clear_partition("removed")
    # The retained handle exercises the real controller's negative ACK without reallocating.
    with pytest.raises(RuntimeError, match="rejected data status update"):
        client.put(_data(), metadata=meta)
    assert "removed" not in ray.get(controller.list_partitions.remote())
    asyncio.run(client.storage_manager.clear_data(meta))


@pytest.mark.parametrize(
    "mode, error",
    [
        ("wrong_sender", "Unexpected data status update sender"),
        ("missing_success", "rejected data status update"),
        ("error", "notification rejected by control"),
        ("unexpected_type", "Controller data status update failed"),
    ],
)
def test_invalid_service_response_reaches_public_put(queue, mode, error):
    client, controller = queue
    ray.get(controller.set_response_mode.remote(mode))
    with pytest.raises(RuntimeError, match=error):
        client.put(_data(), partition_id="invalid")
    partition = ray.get(controller.get_partition_snapshot.remote("invalid"))
    assert partition.production_status.eq(0).all()
    # A failed response must not poison the pooled socket used by the next valid put.
    ray.get(controller.set_response_mode.remote("normal"))
    meta = client.put(_data(), partition_id="next")
    assert [sample.item() for sample in client.get_data(meta)["value"].unbind()] == [22]


def test_missing_controller_ack_fails_public_put_before_deadline(queue):
    client, controller = queue
    meta = client.put(_data(), partition_id="timeout")
    process = psutil.Process(ray.get(controller.process_id.remote()))
    ray.kill(controller, no_restart=True)
    # Ray marks the actor dead before its process necessarily stops serving ZMQ.
    process.wait(timeout=10)
    assert not process.is_running()
    print({"controller_pid": process.pid, "process_terminated": True})
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(controller.list_partitions.remote(), timeout=1)

    async def put():
        await asyncio.wait_for(client.async_put(_data(), metadata=meta), timeout=8)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="no ACK from controller"):
        asyncio.run(put())
    assert time.monotonic() - started < 8
    asyncio.run(client.storage_manager.clear_data(meta))


def test_missing_controller_configuration_fails_public_put(queue):
    client, _ = queue
    meta = client.put(_data(), partition_id="missing")
    info = client.storage_manager.controller_info
    client.storage_manager.controller_info = None
    try:
        with pytest.raises(RuntimeError, match="No controller connected"):
            client.put(_data(), metadata=meta)
    finally:
        client.storage_manager.controller_info = info
