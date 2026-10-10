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
from transfer_queue.utils import common
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQServerInfo


def _manager(with_unit: bool = True) -> AsyncSimpleStorageManager:
    manager = AsyncSimpleStorageManager.__new__(AsyncSimpleStorageManager)
    manager.storage_manager_id = "TQ_STORAGE_test"
    manager.storage_unit_infos = {}
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
