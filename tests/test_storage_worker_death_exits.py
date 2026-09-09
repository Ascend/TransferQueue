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

Worker and proxy both run as daemon threads, so an exception in either only unwinds that
thread: the process stays up with its ROUTER still bound, keeps accepting connections and
never answers them, and every caller blocks until its own timeout. Exiting instead turns a
silent hang into a visible actor failure.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from transfer_queue.storage import simple_storage
from transfer_queue.storage.simple_storage import SimpleStorageUnit

_StorageUnitClass = SimpleStorageUnit.__ray_metadata__.modified_class


def _unit():
    """A unit with only the attributes the abort path touches; no sockets, no Ray."""
    unit = _StorageUnitClass.__new__(_StorageUnitClass)
    unit.storage_unit_id = "TQ_STORAGE_UNIT_test0001"
    unit._shutdown_event = threading.Event()
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
    unit._node_ip = "127.0.0.1"
    unit._inproc_addr = "inproc://worker_death_test"
    unit.zmq_context = MagicMock()

    with (
        patch.object(simple_storage, "create_zmq_socket", return_value=MagicMock()),
        patch.object(simple_storage.zmq, "Poller", return_value=MagicMock()),
        patch.object(_StorageUnitClass, "_worker_loop", side_effect=RuntimeError("boom")),
        patch.object(simple_storage.os, "_exit") as exit_call,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            unit._worker_routine()

    exit_call.assert_called_once_with(simple_storage._WORKER_DEATH_EXIT_CODE)


def _failing_poller(shutdown_event: threading.Event | None = None) -> MagicMock:
    """A proxy poller whose poll raises, optionally after a shutdown has begun."""

    def poll(timeout):
        if shutdown_event is not None:
            shutdown_event.set()
        raise RuntimeError("proxy boom")

    poller = MagicMock()
    poller.poll.side_effect = poll
    return poller


def test_proxy_death_exits_the_process():
    """Without the proxy nothing reaches the worker, so the unit is equally unreachable."""
    unit = _unit()
    unit.put_get_socket = MagicMock()
    unit.worker_socket = MagicMock()

    with (
        patch.object(simple_storage.zmq, "Poller", return_value=_failing_poller()),
        patch.object(simple_storage.os, "_exit") as exit_call,
    ):
        unit._proxy_routine()

    exit_call.assert_called_once_with(simple_storage._WORKER_DEATH_EXIT_CODE)


def test_proxy_shutdown_does_not_exit():
    unit = _unit()
    unit.put_get_socket = MagicMock()
    unit.worker_socket = MagicMock()

    with (
        patch.object(simple_storage.zmq, "Poller", return_value=_failing_poller(unit._shutdown_event)),
        patch.object(simple_storage.os, "_exit") as exit_call,
    ):
        unit._proxy_routine()

    exit_call.assert_not_called()
