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

"""Restore reservations prevent late writes from corrupting reused indexes."""

from threading import RLock
from unittest.mock import AsyncMock

import pytest
import torch
import zmq

from transfer_queue.client import AsyncTransferQueueClient
from transfer_queue.controller import PartitionIndexManager, TransferQueueController
from transfer_queue.sampler import SequentialSampler
from transfer_queue.storage.simple_storage import SimpleStorageUnit, StorageUnitData
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType


@pytest.fixture
def controller():
    cls = TransferQueueController.__ray_metadata__.modified_class
    controller = cls.__new__(cls)
    controller.controller_id = "test_controller"
    controller.partitions = {}
    controller.index_manager = PartitionIndexManager()
    controller.sampler = SequentialSampler()
    controller._restore_lock = RLock()
    controller._restores = {}
    controller._cancelled_restores = set()
    controller._clearing_indexes = set()
    return controller


def begin(controller, restore_id="r"):
    return controller.begin_restore(restore_id, "/dump", "p", {"k": {"fields": ["x"], "tag": {}}}, ["u"], {})


@pytest.mark.asyncio
@pytest.mark.parametrize(("active_ids", "saved_ids"), [([], []), (["r"], []), ([], ["r"])])
async def test_recovery_checks_backend_before_cancelling(active_ids, saved_ids):
    client = AsyncTransferQueueClient.__new__(AsyncTransferQueueClient)
    client.storage_manager = object()
    client._restore_rpc = AsyncMock(return_value={"restore_ids": active_ids})

    if active_ids or saved_ids:
        with pytest.raises(NotImplementedError, match="does not support selective load recovery"):
            await client.async_recover_data_load("/dump", saved_ids)
    else:
        await client.async_recover_data_load("/dump", saved_ids)

    client._restore_rpc.assert_awaited_once_with(ZMQRequestType.LIST_RESTORES, {"dump_dir": "/dump"})


def test_cancel_before_claim_rejects_late_load_and_late_begin(controller):
    metadata = begin(controller)
    assert controller.finish_restore("r", commit=False)["finished"]
    controller.clear_partition("p")
    current = controller.kv_retrieve_meta(["other"], "other", create=True)
    assert current.global_indexes == metadata.global_indexes
    with pytest.raises(RuntimeError, match="no longer active"):
        controller.restore_unit("r", "u", "claim")
    with pytest.raises(RuntimeError, match="already cancelled"):
        begin(controller)
    controller.finish_restore("not-arrived", commit=False)
    with pytest.raises(RuntimeError, match="already cancelled"):
        begin(controller, "not-arrived")


def test_claimed_load_keeps_indexes_until_terminal_report(controller):
    metadata = begin(controller)
    controller.restore_unit("r", "u", "claim")
    assert not controller.finish_restore("r", commit=False)["finished"]
    for action in (
        lambda: controller.mark_clearing(metadata.global_indexes, ["p"]),
        lambda: controller.clear_meta(metadata.global_indexes, ["p"]),
        lambda: controller.clear_partition("p"),
        lambda: controller.kv_retrieve_meta(["k"], "p", create=True),
        lambda: begin(controller, "overlap"),
    ):
        with pytest.raises(RuntimeError, match="unresolved"):
            action()
    assert controller.list_restores("/dump") == ["r"]
    assert controller.kv_retrieve_meta(["other"], "other", create=True).global_indexes != metadata.global_indexes
    controller.restore_unit("r", "u", "complete", {"success": False})
    assert controller.finish_restore("r", commit=False)["finished"]
    controller.clear_partition("p")


def test_success_publishes_schema_and_tags_only_at_finish(controller):
    metadata = begin(controller)
    index = metadata.global_indexes[0]
    controller.restore_unit("r", "u", "claim")
    schema = {"x": {"dtype": torch.int64, "shape": (1,), "is_nested": False, "is_non_tensor": False}}
    controller.restore_unit(
        "r",
        "u",
        "complete",
        {
            "success": True,
            "updates": [
                {"global_indexes": [index], "field_schema": schema},
            ],
        },
    )
    assert not controller.partitions["p"].field_metadata
    assert controller.finish_restore("r", commit=True)["finished"]
    assert controller.partitions["p"].field_metadata["x"].global_indexes == {index}
    assert not controller.list_restores("/dump")


def test_restore_cannot_start_in_the_middle_of_clear(controller):
    metadata = controller.kv_retrieve_meta(["k"], "p", create=True)
    controller.mark_clearing(metadata.global_indexes, ["p"])
    with pytest.raises(RuntimeError, match="unfinished clear"):
        begin(controller)
    controller.clear_partition("p")
    begin(controller)


def test_unit_rejects_cancelled_permission_before_reading(controller):
    begin(controller)
    controller.finish_restore("r", commit=False)
    cls = SimpleStorageUnit.__ray_metadata__.modified_class
    unit = cls.__new__(cls)
    unit.storage_unit_id = "u"
    unit.storage_data = StorageUnitData()
    unit._restore_results = {}
    unit._restore_controller_request = lambda context, action, result=None: controller.restore_unit(
        "r", "u", action, result
    )
    unit._load_rows = lambda *_: pytest.fail("Cancelled request read payload")
    response = unit._handle_load_rows(
        ZMQMessage.create(
            request_type=ZMQRequestType.LOAD_ROWS,
            sender_id="test",
            body={"restore": {"restore_id": "r"}, "shards": []},
        )
    )
    assert not response.body["success"]


def test_lost_completion_ack_is_recoverable_without_replaying_payload(controller):
    begin(controller)
    cls = SimpleStorageUnit.__ray_metadata__.modified_class
    unit = cls.__new__(cls)
    unit.storage_unit_id = "u"
    unit._restore_results = {}

    def lose_completion(context, action, result=None):
        if action == "claim":
            controller.restore_unit("r", "u", action, result)
        else:
            raise zmq.error.Again()

    unit._restore_controller_request = lose_completion
    unit._load_rows = lambda *_: ZMQMessage.create(
        request_type=ZMQRequestType.LOAD_ROWS_RESPONSE,
        sender_id="u",
        body={"success": True, "updates": [], "bytes_read": 0},
    )
    unit._handle_load_rows(
        ZMQMessage.create(
            request_type=ZMQRequestType.LOAD_ROWS, sender_id="test", body={"restore": {"restore_id": "r"}, "shards": []}
        )
    )
    assert not controller.finish_restore("r", commit=False)["finished"]
    unit._restore_controller_request = lambda context, action, result=None: controller.restore_unit(
        "r", "u", action, result
    )
    response = unit._handle_report_restore(
        ZMQMessage.create(request_type=ZMQRequestType.REPORT_RESTORE, sender_id="test", body={"restore_id": "r"})
    )
    assert response.body["success"]
    assert controller.finish_restore("r", commit=False)["finished"]


def test_lost_claim_ack_can_be_settled_without_writing(controller):
    begin(controller)
    cls = SimpleStorageUnit.__ray_metadata__.modified_class
    unit = cls.__new__(cls)
    unit.storage_unit_id = "u"
    unit._restore_results = {}

    def lose_ack(context, action, result=None):
        controller.restore_unit("r", "u", action, result)
        raise zmq.error.Again()

    unit._restore_controller_request = lose_ack
    unit._load_rows = lambda *_: pytest.fail("Unacknowledged claim wrote data")
    response = unit._handle_load_rows(
        ZMQMessage.create(
            request_type=ZMQRequestType.LOAD_ROWS, sender_id="test", body={"restore": {"restore_id": "r"}, "shards": []}
        )
    )
    assert not response.body["success"]
    assert not controller.finish_restore("r", commit=False)["finished"]
    unit._restore_controller_request = lambda context, action, result=None: controller.restore_unit(
        "r", "u", action, result
    )
    assert unit._handle_report_restore(
        ZMQMessage.create(request_type=ZMQRequestType.REPORT_RESTORE, sender_id="test", body={"restore_id": "r"})
    ).body["success"]
    assert controller.finish_restore("r", commit=False)["finished"]
