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

import asyncio
from threading import RLock
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
import zmq

from transfer_queue.client import AsyncTransferQueueClient
from transfer_queue.controller import PartitionIndexManager, TransferQueueController
from transfer_queue.sampler import SequentialSampler
from transfer_queue.storage.dump_io import RestorePendingError
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
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
    controller._restore_outcomes = {}
    controller._clearing_indexes = set()
    return controller


def begin(controller, restore_id="r"):
    return controller.begin_restore(restore_id, "/dump", "p", {"k": {"fields": ["x"], "tag": {}}}, ["u"], {})


@pytest.mark.asyncio
async def test_report_failures_preserve_other_unit_reports(caplog):
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_unit_infos = dict.fromkeys(["rejected", "slow", "timeout"])
    manager.close = lambda: None
    failed = asyncio.Event()
    completed = []

    async def report(restore, target_storage_unit):
        if target_storage_unit == "rejected":
            failed.set()
            raise RuntimeError("Invalid restore transition pending -> complete")
        await failed.wait()
        if target_storage_unit == "timeout":
            raise zmq.error.Again()
        await asyncio.sleep(0)
        completed.append(target_storage_unit)

    manager._report_restore_unit = report
    errors = await manager.report_restore({"restore_id": "r"})
    assert completed == ["slow"]
    assert set(errors) == {"rejected", "timeout"}
    assert "pending -> complete" in errors["rejected"]
    assert "Again" in errors["timeout"]
    assert "Restore r: unit rejected" in caplog.text
    assert "pending -> complete" in caplog.text


@pytest.mark.asyncio
async def test_report_preserves_controller_rejection_message():
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_manager_id = "test"
    manager.close = lambda: None
    response = ZMQMessage.create(
        request_type=ZMQRequestType.REPORT_RESTORE_RESPONSE,
        sender_id="u",
        body={"success": False, "message": "Invalid restore transition pending -> complete"},
    )
    socket = SimpleNamespace(send_multipart=AsyncMock(), recv_multipart=AsyncMock(return_value=response.serialize()))
    with pytest.raises(RuntimeError, match="Restore r: storage unit u report failed: Invalid restore transition"):
        await manager._report_restore_unit.__wrapped__(manager, {"restore_id": "r"}, "u", socket=socket)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [None, False, True, "timeout"])
async def test_recovery_surfaces_report_errors_only_while_outcome_is_unknown(outcome):
    client = AsyncTransferQueueClient.__new__(AsyncTransferQueueClient)
    errors = {"u": "RuntimeError: completion rejected"}
    client.storage_manager = SimpleNamespace(report_restore=AsyncMock(return_value=errors))
    client._restore_context = lambda restore_id: {"restore_id": restore_id}
    reply = {"finished": outcome is not None, "committed": outcome}
    client._restore_rpc = AsyncMock(
        side_effect=[
            {"restore_ids": ["r"]},
            zmq.error.Again() if outcome == "timeout" else reply,
        ]
    )
    if outcome is None or outcome == "timeout":
        with pytest.raises(RestorePendingError, match="completion rejected") as error:
            await client.async_recover_data_load("/dump")
        assert error.value.report_errors == errors
    else:
        assert await client.async_recover_data_load("/dump") is outcome
    assert client._restore_rpc.await_args.args == (ZMQRequestType.FINISH_RESTORE, {"restore_id": "r", "commit": True})


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
    with pytest.raises(RuntimeError, match="already committed or cancelled"):
        begin(controller)
    controller.finish_restore("not-arrived", commit=False)
    with pytest.raises(RuntimeError, match="already committed or cancelled"):
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


def test_restore_rejects_schema_before_allocating_indexes(controller):
    metadata = controller.kv_retrieve_meta(["existing"], "p", create=True)
    partition = controller.partitions["p"]
    schema = {"x": {"dtype": torch.int64, "shape": (1,), "is_non_tensor": False, "is_nested": False}}
    assert partition.update_production_status(metadata.global_indexes, [], schema)
    conflict = {"x": {**schema["x"], "dtype": torch.float32}}
    with pytest.raises(ValueError, match="dtype mismatch"):
        controller.begin_restore("r", "/dump", "p", {"new": {"fields": ["x"], "tag": {}}}, ["u"], conflict)
    assert set(partition.keys_mapping) == {"existing"}
    assert not controller.list_restores("/dump")


def test_restore_validates_all_updates_before_publishing_readiness(controller):
    metadata = controller.kv_retrieve_meta(["existing"], "p", create=True)
    partition = controller.partitions["p"]
    schema = {"x": {"dtype": torch.int64, "shape": (1,), "is_non_tensor": False, "is_nested": False}}
    assert partition.update_production_status(metadata.global_indexes, [], schema)
    restored = begin(controller)
    controller.restore_unit("r", "u", "claim")
    controller.restore_unit(
        "r",
        "u",
        "complete",
        {
            "success": True,
            "updates": [
                {"global_indexes": restored.global_indexes, "field_schema": {"y": schema["x"]}},
                {
                    "global_indexes": restored.global_indexes,
                    "field_schema": {"x": {**schema["x"], "dtype": torch.float32}},
                },
            ],
        },
    )
    with pytest.raises(ValueError, match="dtype mismatch"):
        controller.finish_restore("r", commit=True)
    assert "y" not in partition.field_metadata
    assert not partition.production_status[restored.global_indexes].any()
    assert controller.finish_restore("r", commit=False)["finished"]


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


def test_failed_unit_cancels_pending_units_but_waits_for_running_units(controller):
    rows = {f"k{i}": {"fields": ["x"], "tag": {}} for i in range(3)}
    controller.begin_restore("r", "/dump", "p", rows, ["u0", "u1", "u2"], {})
    controller.restore_unit("r", "u0", "claim")
    controller.restore_unit("r", "u1", "claim")
    controller.restore_unit("r", "u0", "complete", {"success": False})
    assert not controller.finish_restore("r", commit=True)["finished"]
    with pytest.raises(RuntimeError, match="cannot start"):
        controller.restore_unit("r", "u2", "claim")
    with pytest.raises(RuntimeError, match="unresolved"):
        controller.clear_partition("p")
    controller.restore_unit("r", "u1", "complete", {"success": True, "updates": []})
    assert controller.finish_restore("r", commit=True) == {"finished": True, "committed": False}
    assert not controller.partitions["p"].field_metadata


def test_unknown_restore_stays_pending_until_explicit_cancellation(controller):
    assert not controller.finish_restore("unknown", commit=True)["finished"]
    assert controller.finish_restore("unknown", commit=False) == {"finished": True, "committed": False}
    with pytest.raises(RuntimeError, match="already committed or cancelled"):
        begin(controller, "unknown")


@pytest.mark.parametrize("units", [["only"], ["u0", "u1"], ["u2", "u0", "u1"]])
@pytest.mark.parametrize("has_payload", [True, False])
def test_restore_reserves_only_current_storage_owners(controller, units, has_payload):
    controller.kv_retrieve_meta(["other"], "unrelated", create=True)
    existing = controller.kv_retrieve_meta(["keep", "gap", "empty", "last"], "p", create=True)
    controller.clear_meta([existing.global_indexes[1]], ["p"])
    rows = {
        key: {"fields": ["x"] if has_payload and key != "empty" else [], "tag": {}}
        for key in ["last", "new", "empty", "keep"]
    }
    metadata = controller.begin_restore("r", "/dump", "p", rows, units, {})
    indexes = [index for key, index in zip(rows, metadata.global_indexes, strict=True) if rows[key]["fields"]]
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_unit_infos = dict.fromkeys(units)
    manager.close = lambda: None
    routed = manager._group_by_hash(indexes)
    assert set(controller._restores["r"]["units"]) == set(routed)
    assert metadata.global_indexes[0] == existing.global_indexes[-1]
    assert metadata.global_indexes[-1] == existing.global_indexes[0]
    for unit, group in routed.items():
        assert [indexes[pos] for pos in group.batch_positions] == group.global_indexes
        controller.restore_unit("r", unit, "claim")
        controller.restore_unit("r", unit, "complete", {"success": True, "updates": []})
    assert controller.finish_restore("r", commit=True) == {"finished": True, "committed": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["running_timeout", "lost_load_reply", "lost_commit_reply", "lost_complete"])
async def test_pending_load_can_commit_without_replaying_payload(controller, interruption):
    client = AsyncTransferQueueClient.__new__(AsyncTransferQueueClient)
    client._restore_context = lambda restore_id: {"restore_id": restore_id}
    completed = {}
    fail_commit_reply = interruption == "lost_commit_reply"

    async def rpc(action, body):
        nonlocal fail_commit_reply
        if action == ZMQRequestType.BEGIN_RESTORE:
            return {"metadata": controller.begin_restore(**body)}
        if action == ZMQRequestType.LIST_RESTORES:
            return {"restore_ids": controller.list_restores(body["dump_dir"])}
        result = controller.finish_restore(**body)
        if body["commit"] and result["finished"] and fail_commit_reply:
            fail_commit_reply = False
            raise zmq.error.Again()
        return result

    async def load(shards, context):
        controller.restore_unit("r", "u", "claim")
        index = controller._restores["r"]["metadata"].global_indexes[0]
        schema = {"x": {"dtype": torch.int64, "shape": (1,), "is_nested": False, "is_non_tensor": False}}
        completed.update(success=True, updates=[{"global_indexes": [index], "field_schema": schema}])
        if interruption in ("lost_load_reply", "lost_commit_reply"):
            controller.restore_unit("r", "u", "complete", completed)
        if interruption in ("running_timeout", "lost_load_reply"):
            raise zmq.error.Again()

    async def report(context):
        controller.restore_unit("r", "u", "complete", completed)

    client._restore_rpc = rpc
    client.storage_manager = SimpleNamespace(
        storage_unit_infos={"u": None}, load_rows_by_index=AsyncMock(side_effect=load), report_restore=report
    )
    rows = {"k": {"fields": ["x"], "tag": {"saved": True}}}
    with pytest.raises(RestorePendingError):
        await client.async_load_rows_by_key("p", rows, [], "/dump", "r")
    if "r" in controller._restores:
        assert not controller._restores["r"]["aborting"]
        with pytest.raises(RuntimeError, match="unresolved"):
            controller.clear_partition("p")
    assert await client.async_recover_data_load("/dump", ["r"]) is True
    assert await client.async_recover_data_load("/dump", ["r"]) is True
    client.storage_manager.load_rows_by_index.assert_awaited_once()
    partition = controller.partitions["p"]
    index = partition.keys_mapping["k"]
    assert partition.production_status[index, partition.field_name_mapping["x"]] == 1
    assert partition.custom_meta[index] == {"saved": True}
    # Duplicate finalization must not publish metadata again after the key is cleared.
    controller.clear_partition("p")
    assert controller.finish_restore("r", commit=True) == {"finished": True, "committed": True}
    assert "p" not in controller.partitions
