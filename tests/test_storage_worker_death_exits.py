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

"""The unit process must exit when the thread that answers requests dies.

The worker runs as a daemon thread, so an unhandled exception in it only unwinds that thread:
the process stays up with its ROUTER still bound, keeps accepting connections and never answers
them, and every caller blocks until its own timeout. Exiting instead turns a silent hang into a
visible actor failure.
"""

import errno
import threading
from unittest.mock import MagicMock, patch

import pytest
import torch
import zmq

from transfer_queue.storage import simple_storage
from transfer_queue.storage.simple_storage import SimpleStorageUnit, _PartialReplySendError
from transfer_queue.utils import common

_StorageUnitClass = SimpleStorageUnit.__ray_metadata__.modified_class


def _unit():
    """A unit with only the attributes the abort path touches; no sockets, no Ray."""
    unit = _StorageUnitClass.__new__(_StorageUnitClass)
    unit.storage_unit_id = "TQ_STORAGE_UNIT_test0001"
    unit._shutdown_event = threading.Event()
    unit._storage_lock = threading.Lock()
    return unit


def test_worker_death_exits_the_process():
    unit = _unit()

    with patch.object(simple_storage.os, "_exit") as exit_call:
        unit._abort_process("the worker thread died")

    exit_call.assert_called_once_with(simple_storage._WORKER_DEATH_EXIT_CODE)


def test_abort_is_silent_during_shutdown():
    """A shutdown races the poller, so an exception on the way out must not look like a crash."""
    unit = _unit()
    unit._shutdown_event.set()

    with patch.object(simple_storage.os, "_exit") as exit_call:
        unit._abort_process("the worker thread died")

    exit_call.assert_not_called()


def test_opt_out_keeps_the_process_alive_for_inspection():
    unit = _unit()

    with (
        patch.object(simple_storage, "TQ_STORAGE_EXIT_ON_WORKER_DEATH", False),
        patch.object(simple_storage.os, "_exit") as exit_call,
    ):
        unit._abort_process("the worker thread died")

    exit_call.assert_not_called()


def test_unhandled_worker_exception_triggers_the_exit():
    """The whole point: a dead worker must take the process down, not linger unreachable."""
    unit = _unit()
    unit._metrics = None
    unit._perf_monitor = MagicMock()
    unit.put_get_socket = MagicMock()

    with (
        patch.object(_StorageUnitClass, "_worker_loop", side_effect=RuntimeError("boom")),
        patch.object(simple_storage.os, "_exit") as exit_call,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            unit._worker_routine()

    exit_call.assert_called_once_with(simple_storage._WORKER_DEATH_EXIT_CODE)


def test_unrecoverable_poll_error_exits_on_first_failure():
    """A broken listening ROUTER (e.g. ENOTSOCK) must abort immediately rather than spin."""
    unit = _unit()
    unit._metrics = None
    unit._perf_monitor = MagicMock()
    unit.put_get_socket = MagicMock()
    poller = MagicMock()
    poller.poll.side_effect = zmq.ZMQError(errno.ENOTSOCK)

    with (
        patch.object(simple_storage.zmq, "Poller", return_value=poller),
        patch.object(simple_storage.os, "_exit") as exit_call,
    ):
        with pytest.raises(zmq.ZMQError):
            unit._worker_routine()

    poller.poll.assert_called_once()
    exit_call.assert_called_once_with(simple_storage._WORKER_DEATH_EXIT_CODE)


def test_poll_error_during_shutdown_exits_cleanly():
    unit = _unit()
    unit._metrics = None
    unit._perf_monitor = MagicMock()
    unit.put_get_socket = MagicMock()

    def poll_then_shutdown(_timeout):
        unit._shutdown_event.set()
        raise zmq.ZMQError(errno.ENOTSOCK)

    poller = MagicMock()
    poller.poll.side_effect = poll_then_shutdown

    with (
        patch.object(simple_storage.zmq, "Poller", return_value=poller),
        patch.object(simple_storage.os, "_exit") as exit_call,
    ):
        unit._worker_routine()

    exit_call.assert_not_called()


@pytest.mark.parametrize("fail_on_frame_idx", [1, 2])
def test_mid_multipart_reply_failure_aborts_process_without_misrouting(fail_on_frame_idx):
    """Once a non-final SNDMORE frame entered ROUTER, a subsequent send failure aborts the unit."""
    unit = _unit()
    unit._metrics = None
    unit._perf_monitor = MagicMock()
    router = MagicMock()
    calls = 0

    def fail_after_sndmore(_frame, flags=0, **_kwargs):
        nonlocal calls
        if calls == fail_on_frame_idx:
            raise MemoryError("simulated mid-multipart OOM")
        calls += 1

    router.send.side_effect = fail_after_sndmore
    unit.put_get_socket = router
    poller = MagicMock()
    poller.poll.return_value = [(router, zmq.POLLIN)]

    def trigger_partial_send(sock, _monitor):
        unit._send_multipart_reply(sock, b"TQ_STORAGE_client", [b"header", b"payload"])

    unit._process_one_worker_request = trigger_partial_send  # type: ignore[method-assign]

    with (
        patch.object(simple_storage.zmq, "Poller", return_value=poller),
        patch.object(simple_storage.os, "_exit") as exit_call,
    ):
        with pytest.raises(_PartialReplySendError):
            unit._worker_routine()

    exit_call.assert_called_once_with(simple_storage._WORKER_DEATH_EXIT_CODE)


def test_set_pytorch_threads_falls_back_to_logical_cores_when_physical_is_none(monkeypatch, caplog):
    """Kunpeng ARM64 / container hosts where cpu_count(logical=False) is None must not clamp to 1 thread."""
    monkeypatch.setattr(
        common.psutil,
        "cpu_count",
        lambda logical=True: None if not logical else 64,
    )
    previous = torch.get_num_threads()
    try:
        chosen = common.set_pytorch_auto_parallel_threads(target_num_threads=8, info="arm64-test")
        assert chosen == 8
        assert torch.get_num_threads() == 8
        assert "should not exceed" not in caplog.text
    finally:
        torch.set_num_threads(previous)
