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

"""Storage-unit retry and timeout-diagnosis tests."""

import asyncio
import errno
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
import zmq
from tensordict import TensorDict

from transfer_queue.metadata import BatchMeta
from transfer_queue.storage.managers import simple_storage_manager as ssm
from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager, StorageUnitTimeout
from transfer_queue.storage.payload_transfer.base import (
    Operation,
    ReplyHeader,
    StorageRejected,
    TransferError,
    TransferTimeout,
)
from transfer_queue.storage.simple_storage import StorageKeyNotFoundError
from transfer_queue.utils import common
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo


def _manager(with_unit: bool = True) -> AsyncSimpleStorageManager:
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_manager_id = "TQ_STORAGE_test"
    manager.storage_unit_infos = {}
    manager.config = {}
    manager._payload_connections = {}
    manager._payload_connecting = {}
    manager._payload_closed = False
    if with_unit:
        manager.storage_unit_infos["unit_a"] = ZMQServerInfo(
            role=Role.STORAGE,
            id="unit_a",
            ip="10.0.0.7",
            ports={"put_get_socket": 5555},
        )
    manager.close = lambda: None
    return manager


def _single_sample_batch() -> tuple[TensorDict, BatchMeta]:
    metadata = BatchMeta(
        global_indexes=[0],
        partition_ids=["0"],
        field_schema={"input_ids": {"dtype": torch.int64, "shape": (2,), "is_nested": False, "is_non_tensor": False}},
        production_status=np.ones(1, dtype=np.int8),
    )
    return TensorDict({"input_ids": torch.zeros(1, 2, dtype=torch.int64)}, batch_size=1), metadata


class _SocketLease:
    def __init__(self, socket):
        self.socket = socket

    async def __aenter__(self):
        return self.socket

    async def __aexit__(self, *_args):
        return None


@pytest.mark.asyncio
async def test_payload_concurrent_requests_share_one_handshake():
    manager = _manager()
    peer = {"backend": "test", "token": "opaque"}
    manager.config = {"payload_peers": {"unit_a": peer}}
    started, release = asyncio.Event(), asyncio.Event()
    connection = object()

    async def connect(info):
        assert info is peer
        started.set()
        await release.wait()
        return connection

    manager.payload_transfer = MagicMock(
        connect=AsyncMock(side_effect=connect),
        request=AsyncMock(return_value=(ReplyHeader({}), {"value": [1]})),
    )
    calls = [asyncio.create_task(manager._request_payload(
        "unit_a", Operation.GET, {"global_indexes": [i]}, recv=None,
    )) for i in range(8)]
    await started.wait()
    release.set()
    await asyncio.wait_for(asyncio.gather(*calls), timeout=2)
    await manager._request_payload("unit_a", Operation.GET, {}, recv=None)
    manager.payload_transfer.connect.assert_awaited_once()
    assert manager.payload_transfer.request.await_count == 9
    assert all(call.args[0] is connection for call in manager.payload_transfer.request.await_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TransferTimeout("lost"), TransferError("broken"),
                                    asyncio.CancelledError(), StorageRejected("denied")])
async def test_payload_connection_eviction_and_storage_rejection(failure):
    manager = _manager()
    manager.payload_transfer = MagicMock(
        connect=AsyncMock(side_effect=[object(), object()]),
        request=AsyncMock(side_effect=[failure, (ReplyHeader({}), {"value": [1]})]),
    )
    expected = asyncio.CancelledError if isinstance(failure, asyncio.CancelledError) else RuntimeError
    with pytest.raises(expected):
        await manager._request_payload("unit_a", Operation.GET, {}, recv=None)
    await manager._request_payload("unit_a", Operation.GET, {}, recv=None)
    assert manager.payload_transfer.connect.await_count == (1 if isinstance(failure, StorageRejected) else 2)


@pytest.mark.asyncio
async def test_payload_connect_timeout_is_bounded_and_can_reconnect(monkeypatch):
    manager = _manager()
    cancelled = asyncio.Event()

    async def stalled_connect(_):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    manager.payload_transfer = MagicMock(
        connect=AsyncMock(side_effect=stalled_connect),
        request=AsyncMock(return_value=(ReplyHeader({}), {})),
    )
    monkeypatch.setattr(ssm, "TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT", 0.01)
    with pytest.raises(StorageUnitTimeout):
        await manager._request_payload("unit_a", Operation.GET, {}, recv=None)
    assert cancelled.is_set()
    manager.payload_transfer.request.assert_not_awaited()
    manager.payload_transfer.connect.side_effect = None
    manager.payload_transfer.connect.return_value = object()
    await manager._request_payload("unit_a", Operation.GET, {}, recv=None)
    assert manager.payload_transfer.connect.await_count == 2


@pytest.mark.asyncio
async def test_late_failure_does_not_evict_replacement_connection():
    manager = _manager()
    old, new = object(), object()
    entered, release = asyncio.Event(), asyncio.Event()

    async def request(connection, header, **_):
        if header.body.get("slow"):
            entered.set()
            await release.wait()
            raise TransferTimeout("late failure")
        if connection is old:
            raise TransferTimeout("first failure")
        return ReplyHeader({}), {}

    manager.payload_transfer = MagicMock(
        connect=AsyncMock(side_effect=[old, new]), request=AsyncMock(side_effect=request),
    )
    late = asyncio.create_task(manager._request_payload("unit_a", Operation.GET, {"slow": True}, recv=None))
    await entered.wait()
    try:
        with pytest.raises(StorageUnitTimeout):
            await manager._request_payload("unit_a", Operation.GET, {}, recv=None)
        await manager._request_payload("unit_a", Operation.GET, {}, recv=None)
    finally:
        release.set()
        with pytest.raises(StorageUnitTimeout):
            await late
    await manager._request_payload("unit_a", Operation.GET, {}, recv=None)
    assert manager.payload_transfer.connect.await_count == 2


def test_payload_connections_do_not_cross_event_loops():
    manager = _manager()
    manager.payload_transfer = MagicMock(
        connect=AsyncMock(side_effect=[object(), object()]),
        request=AsyncMock(return_value=(ReplyHeader({}), {})),
    )
    for _ in range(2):
        asyncio.run(manager._request_payload("unit_a", Operation.GET, {}, recv=None))
    assert manager.payload_transfer.connect.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", list(Operation))
@pytest.mark.parametrize("phase", ["connect", "request"])
@pytest.mark.parametrize("failure", ["transport", "zmq", "decode", "timeout", "rejection", "missing_key", "cancel"])
async def test_payload_failure_preserves_target_and_exception_category(operation, phase, failure):
    manager = _manager()
    manager.config = {"payload_peers": {"unit_a": {"backend": "test"}}}
    manager.payload_transfer = MagicMock()
    manager.payload_transfer.connect = AsyncMock(return_value="connection")
    manager.payload_transfer.request = AsyncMock()
    error = {
        "transport": TransferError("backend failed"),
        "zmq": zmq.ZMQError(errno.EMFILE),
        "decode": ValueError("invalid frame"),
        "timeout": TransferTimeout("timed out", wire_bytes=1024),
        "rejection": StorageRejected("write rejected"),
        "missing_key": StorageRejected("key missing", missing_key=True),
        "cancel": asyncio.CancelledError(),
    }[failure]
    getattr(manager.payload_transfer, phase).side_effect = error
    expected = {
        "timeout": StorageUnitTimeout,
        "missing_key": StorageKeyNotFoundError if operation == Operation.GET else RuntimeError,
        "cancel": asyncio.CancelledError,
    }.get(failure, RuntimeError)
    with pytest.raises(expected) as caught:
        await manager._request_payload(
            "unit_a", operation, {"global_indexes": [0]},
            data=None if operation == Operation.GET else {"value": [1]},
            recv=None if operation == Operation.GET else (),
        )
    assert type(caught.value) is expected
    if failure != "cancel":
        assert "unit_a" in str(caught.value)
        assert "10.0.0.7:5555" in str(caught.value)
        assert operation.name.lower() in str(caught.value)
        assert caught.value.__cause__ is error
        if failure in ("transport", "zmq", "decode"):
            assert type(error).__name__ in str(caught.value)
            assert str(error) in str(caught.value)
    elif phase == "request":
        assert caught.value is error
    if phase == "connect":
        manager.payload_transfer.request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_controller_notification_returns_ack_status(accepted):
    manager = _manager()
    manager.controller_info = MagicMock(id="controller")
    socket = MagicMock()
    socket.send_multipart = AsyncMock()
    socket.recv_multipart = AsyncMock(
        return_value=ZMQMessage.create(
            request_type=ZMQRequestType.NOTIFY_DATA_UPDATE_ACK,
            sender_id="controller",
            body={"success": accepted},
        ).serialize()
    )
    manager.notify_pool = MagicMock()
    manager.notify_pool.alease.return_value = _SocketLease(socket)

    assert await manager._notify_and_wait([b"request"]) is accepted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ack, published, warns",
    [
        (True, True, False),
        (None, False, False),  # no ACK in time: the outcome is unknown, so the put is not failed
        (False, False, True),
    ],
)
async def test_notify_publishes_or_tolerates_ack(ack, published, warns, caplog):
    manager = _manager()
    manager.controller_info = MagicMock(id="controller")
    manager._notify_loop = asyncio.get_running_loop()
    manager._notify_and_wait = AsyncMock(return_value=ack)

    with caplog.at_level(logging.WARNING):
        assert await manager.notify_data_update("p", [0, 1], {}) is published
    assert ("no longer exists" in caplog.text) is warns


@pytest.mark.asyncio
async def test_notify_raises_on_request_error_without_waiting_for_timeout():
    manager = _manager()
    manager.controller_info = MagicMock(id="controller")
    manager._notify_loop = asyncio.get_running_loop()
    socket = MagicMock()
    socket.send_multipart = AsyncMock()
    error = ZMQMessage.create(
        request_type=ZMQRequestType.REQUEST_ERROR,
        sender_id="controller",
        body={"success": False, "message": "Field 'x' dtype mismatch"},
    )
    socket.recv_multipart = AsyncMock(side_effect=[error.serialize(), TimeoutError("no further reply")])
    manager.notify_pool = MagicMock()
    manager.notify_pool.alease.return_value = _SocketLease(socket)

    with pytest.raises(RuntimeError, match="Field 'x' dtype mismatch"):
        await manager.notify_data_update("p", [0], {})
    socket.recv_multipart.assert_awaited_once()
    socket.close.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("data_parser, expected_attempts", [(None, 3), (lambda data: data, 1)])
async def test_put_retries_only_without_a_parser(data_parser, expected_attempts):
    manager = _manager()
    manager.notify_data_update = AsyncMock()
    manager._put_to_single_storage_unit = AsyncMock(side_effect=StorageUnitTimeout("no answer"))
    data, metadata = _single_sample_batch()

    with (
        patch.object(ssm, "TQ_SIMPLE_STORAGE_MAX_ATTEMPTS", 3),
        patch.object(manager, "_diagnose_storage_unit", return_value="diagnosis"),
        pytest.raises(StorageUnitTimeout),
    ):
        await manager.put_data(data, metadata, data_parser=data_parser)

    assert manager._put_to_single_storage_unit.await_count == expected_attempts
    manager.notify_data_update.assert_not_awaited()


_HELD_INPUT_IDS = {"input_ids": {"dtype": torch.int64, "is_non_tensor": False, "held": [True]}}


@pytest.mark.asyncio
@pytest.mark.parametrize("data_parser, expected_attempts", [(None, 3), (lambda data, args: data, 1)])
async def test_get_retries_only_without_a_parser(data_parser, expected_attempts):
    manager = _manager()
    manager._get_from_single_storage_unit = AsyncMock(side_effect=StorageUnitTimeout("no answer"))
    _, metadata = _single_sample_batch()

    with (
        patch.object(ssm, "TQ_SIMPLE_STORAGE_MAX_ATTEMPTS", 3),
        patch.object(manager, "_diagnose_storage_unit", return_value="diagnosis"),
        pytest.raises(StorageUnitTimeout),
    ):
        await manager.get_data(metadata, data_parser=data_parser)

    assert manager._get_from_single_storage_unit.await_count == expected_attempts


@pytest.mark.asyncio
async def test_update_reports_controller_rejection_after_storage_commit():
    manager = _manager()
    manager._update_to_single_storage_unit = AsyncMock(
        return_value={"input_ids": {"dtype": torch.int64, "shapes": [(3,)]}}
    )
    manager.notify_data_update = AsyncMock(return_value=False)
    values, metadata = _single_sample_batch()

    with pytest.raises(RuntimeError, match="Storage update committed.*do not retry"):
        await manager.update_data(metadata, values, lambda old, new: torch.cat([old, new]), _HELD_INPUT_IDS)

    manager._update_to_single_storage_unit.assert_awaited_once()
    manager.notify_data_update.assert_awaited_once()

    manager.notify_data_update = AsyncMock(side_effect=RuntimeError("Controller failed to record written data: dtype"))
    with pytest.raises(RuntimeError, match="Storage update committed.*notification failed.*dtype.*do not retry"):
        await manager.update_data(metadata, values, lambda old, new: torch.cat([old, new]), _HELD_INPUT_IDS)


@pytest.mark.asyncio
async def test_update_timeout_is_not_retried_or_published():
    manager = _manager()
    manager._update_to_single_storage_unit = AsyncMock(side_effect=StorageUnitTimeout("outcome unknown"))
    manager.notify_data_update = AsyncMock()
    values, metadata = _single_sample_batch()

    with patch.object(manager, "_diagnose_storage_unit", return_value="diagnosis"), pytest.raises(StorageUnitTimeout):
        await manager.update_data(metadata, values, lambda old, new: torch.cat([old, new]), _HELD_INPUT_IDS)

    manager._update_to_single_storage_unit.assert_awaited_once()
    manager.notify_data_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_lost_request_recovers_without_diagnosis():
    manager = _manager()
    request = AsyncMock(side_effect=[StorageUnitTimeout("no answer"), "payload"])

    with patch.object(manager, "_diagnose_storage_unit") as diagnose:
        result = await manager._request_with_retry("get", "unit_a", "samples=4", request)

    assert result == "payload"
    assert request.await_count == 2
    diagnose.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, max_attempts, expected_attempts",
    [
        (StorageUnitTimeout("no answer"), 3, 3),
        (RuntimeError("unit rejected request"), 3, 1),
        (StorageUnitTimeout("no answer"), 0, 1),
    ],
)
async def test_retry_attempts_are_bounded_and_only_cover_timeouts(error, max_attempts, expected_attempts):
    manager = _manager()
    request = AsyncMock(side_effect=error)

    with (
        patch.object(manager, "_diagnose_storage_unit", return_value="diagnosis"),
        pytest.raises(type(error)),
    ):
        await manager._request_with_retry("get", "unit_a", "samples=4", request, max_attempts=max_attempts)

    assert request.await_count == expected_attempts


@pytest.mark.asyncio
async def test_failure_log_has_context_without_repeating_the_unit(caplog):
    manager = _manager()
    error = StorageUnitTimeout("no answer in 200s during get from storage unit unit_a at 10.0.0.7:5555")

    with (
        patch.object(manager, "_diagnose_storage_unit", return_value="diagnosis"),
        caplog.at_level(logging.ERROR),
        pytest.raises(StorageUnitTimeout),
    ):
        await manager._request_with_retry(
            "get", "unit_a", "samples=4 fields=['input_ids']", AsyncMock(side_effect=error)
        )

    message = next(record.message for record in caplog.records if "failed after" in record.message)
    assert "samples=4 fields=['input_ids']" in message
    assert message.count("unit_a") == message.count("10.0.0.7:5555") == 1


@pytest.mark.parametrize(
    "elapsed, payload_bytes, should_log",
    [(0.01, 2**20, False), (0.01, 512 * 2**20, True), (6.0, 2**20, True)],
)
def test_log_heavy_operation_thresholds(caplog, elapsed, payload_bytes, should_log):
    with caplog.at_level(logging.WARNING):
        common.log_heavy_operation("TQ_STORAGE_test", "put", elapsed, payload_bytes, "samples=4")

    assert bool(caplog.records) is should_log
    if should_log:
        assert "serialized_mb=" in caplog.records[0].message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tcp_result, probe_result, expected",
    [
        # A probe that answers proves only that the unit is serving again, whatever the
        # cumulative counters happen to say.
        ((None, None), {"active_keys": 4, "arrivals_by_op": {"GET_DATA": 0}}, "unit_serving_again"),
        ((None, None), {"active_keys": 4, "arrivals_by_op": {"GET_DATA": 9}}, "unit_serving_again"),
        ((None, None), {"active_keys": 4}, "unit_serving_again"),
        ((None, None), zmq.error.Again(), "unit_not_serving"),
        (ConnectionRefusedError(), zmq.error.Again(), "tcp=down(ConnectionRefusedError)"),
    ],
)
async def test_diagnosis_classifies_the_failure(tcp_result, probe_result, expected):
    manager = _manager()
    writer = MagicMock()
    tcp = AsyncMock(
        side_effect=tcp_result if isinstance(tcp_result, Exception) else None,
        return_value=(None, writer),
    )
    probe = AsyncMock(
        side_effect=probe_result if isinstance(probe_result, Exception) else None,
        return_value=probe_result,
    )

    with patch.object(ssm.asyncio, "open_connection", tcp), patch.object(manager, "_probe_storage_unit", probe):
        diagnosis = await manager._diagnose_storage_unit("unit_a")

    assert expected in diagnosis


def test_diagnosis_does_not_report_empty_op_stats_as_zero_traffic():
    """op_stats is Prometheus-gated, so an empty dict must not read as 'served nothing'."""
    described = ssm._describe_unit_state({"requests_arrived": 7, "active_keys": 1})

    assert "completed=unavailable(prometheus_disabled)" in described
    assert "completed={}" not in described


@pytest.mark.parametrize(
    "arrivals",
    [
        # Nine historical GETs, all already completed: they say nothing about this request.
        {"GET_DATA": 9},
        # The unit restarted during the timeout window, so its counters began again at zero.
        {"GET_DATA": 0},
        # The request arrived and finished after the caller gave up waiting.
        {"GET_DATA": 10},
    ],
)
def test_cumulative_counters_never_become_a_claim_about_this_request(arrivals):
    """Counters are per-op and cumulative, so no value of them locates one request.

    They carry no request id, never decrease, and reset when the unit restarts, so both
    directions of inference are unsound: a nonzero count may be entirely historical, and a
    zero one may mean the request reached a previous process.
    """
    described = ssm._describe_unit_state({"requests_arrived": sum(arrivals.values()), "arrivals_by_op": arrivals})

    assert "verdict=unit_serving_again" in described
    assert "arrived_but_unfinished" not in described
    assert "request_lost_in_flight" not in described
    # The raw counters stay in the line as triage input, just not as a verdict.
    assert "arrivals_by_op=" in described


@pytest.mark.asyncio
async def test_diagnosis_handles_an_unknown_unit():
    assert "unit_not_registered" in await _manager(with_unit=False)._diagnose_storage_unit("missing")
