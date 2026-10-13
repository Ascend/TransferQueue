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

"""Tests for the accept-queue probe.

A connection dropped during establishment is invisible to both ends, and the kernel
charges the listening socket's sk_drops on several such paths, of which a full accept
queue is only one. These tests pin the arithmetic the probe reports, the levels it logs
at, and that it never names a cause its counters cannot establish.
"""

import threading

from transfer_queue.utils.accept_probe import (
    AcceptQueueProbe,
    AcceptQueueSample,
    AcceptQueueStats,
    _read_listen_overflows,
    sample_accept_queue,
)


def _sample(
    recv_q: int, backlog: int = 100, sk_drops: int = 0, overflows: int = 0, drops: int | None = None
) -> AcceptQueueSample:
    return AcceptQueueSample(
        timestamp=0.0,
        recv_q=recv_q,
        backlog=backlog,
        sk_drops=sk_drops,
        listen_overflows=overflows,
        listen_drops=overflows if drops is None else drops,
    )


def _probe() -> AcceptQueueProbe:
    return AcceptQueueProbe(port=34513, owner_id="TQ_STORAGE_UNIT_test", interval_s=0.01)


def test_utilization_is_depth_over_backlog():
    assert _sample(50, backlog=100).utilization == 0.5


def test_zero_backlog_reports_zero_utilization_not_zero_division():
    """A socket read before bind reports backlog 0; that must not raise."""
    assert _sample(5, backlog=0).utilization == 0.0


def test_full_queue_reports_full_utilization():
    assert _sample(100, backlog=100).utilization == 1.0


def test_empty_stats_report_zero_deltas():
    stats = AcceptQueueStats(port=1234)

    assert stats.sk_drops_delta == 0
    assert stats.overflow_delta == 0


def test_drop_delta_is_last_minus_first():
    """Deltas are what matter: the absolute counters carry the whole uptime's history."""
    stats = AcceptQueueStats(port=1234)
    stats.first_sample = _sample(0, sk_drops=23, overflows=1775)
    stats.last_sample = _sample(0, sk_drops=31, overflows=1790)

    assert stats.sk_drops_delta == 8
    assert stats.overflow_delta == 15


def test_describe_names_port_and_deltas():
    stats = AcceptQueueStats(port=34513, backlog=100, peak_recv_q=97)
    stats.first_sample = _sample(0, sk_drops=23)
    stats.last_sample = _sample(0, sk_drops=24)

    text = stats.describe()

    assert "port=34513" in text
    assert "peak_recv_q=97" in text
    assert "sk_drops_delta=1" in text


def test_peak_tracks_highest_depth_not_latest():
    """The burst is the signal; sampling after it drains reads zero."""
    probe = _probe()

    probe._record(_sample(10))
    probe._record(_sample(97))
    probe._record(_sample(3))

    assert probe.stats.peak_recv_q == 97


def test_first_sample_is_retained_as_delta_baseline():
    probe = _probe()

    probe._record(_sample(1, sk_drops=23))
    probe._record(_sample(2, sk_drops=25))

    assert probe.stats.first_sample.sk_drops == 23
    assert probe.stats.sk_drops_delta == 2


def test_drop_increase_is_logged_at_error(caplog):
    """A drop is the direct evidence, so it must not be buried at debug level."""
    probe = _probe()
    probe._record(_sample(0, sk_drops=23))

    with caplog.at_level("ERROR"):
        probe._record(_sample(100, sk_drops=24))

    assert "dropped an incoming connection" in caplog.text


def test_steady_drop_count_is_logged_once_not_every_sample(caplog):
    """sk_drops is cumulative, so a past drop must not be re-reported forever.

    Measuring against the probe's first sample made the condition permanently true once
    the socket had ever dropped a connection: at a 0.1s interval that is ten errors a
    second for the life of the process, and it erases when the drop actually happened.
    """
    probe = _probe()
    probe._record(_sample(0, sk_drops=12))

    with caplog.at_level("ERROR"):
        probe._record(_sample(0, sk_drops=13))  # a new drop -- report it
        for _ in range(20):
            probe._record(_sample(0, sk_drops=13))  # unchanged -- stay quiet

    assert caplog.text.count("dropped an incoming connection") == 1


def test_each_new_drop_is_reported(caplog):
    """Quieting the repeat must not swallow genuinely new drops."""
    probe = _probe()
    probe._record(_sample(0, sk_drops=12))

    with caplog.at_level("ERROR"):
        probe._record(_sample(0, sk_drops=13))
        probe._record(_sample(0, sk_drops=13))
        probe._record(_sample(0, sk_drops=14))

    assert caplog.text.count("dropped an incoming connection") == 2


def test_near_full_queue_warns_once(caplog):
    """Repeating the warning every 100ms would flood the log of a 512-node job."""
    probe = _probe()

    with caplog.at_level("WARNING"):
        probe._record(_sample(60))
        probe._record(_sample(70))

    assert caplog.text.count("reached") == 1


def test_no_warning_below_threshold():
    probe = _probe()

    probe._record(_sample(10))

    assert probe._warned is False


def test_listen_overflows_returns_two_non_negative_ints():
    """Reads the real host, so assert shape rather than a specific value."""
    overflows, drops = _read_listen_overflows()

    assert overflows >= 0
    assert drops >= 0


def test_sampling_an_unused_port_returns_none():
    assert sample_accept_queue(1) is None


def test_unit_shutdown_stops_the_probe():
    """Nothing else calls stop(), so the sampling thread would outlive the unit."""
    from unittest.mock import MagicMock

    from transfer_queue.storage.simple_storage import SimpleStorageUnit

    unit_class = SimpleStorageUnit.__ray_metadata__.modified_class
    probe = MagicMock()

    unit_class._shutdown_resources(
        shutdown_event=threading.Event(),
        worker_thread=None,
        proxy_thread=None,
        zmq_context=None,
        put_get_socket=None,
        accept_probe=probe,
    )

    probe.stop.assert_called_once()


def test_shutdown_without_a_probe_is_a_no_op():
    """The probe is opt-in, so the default path must not require one."""
    from transfer_queue.storage.simple_storage import SimpleStorageUnit

    unit_class = SimpleStorageUnit.__ray_metadata__.modified_class

    unit_class._shutdown_resources(
        shutdown_event=threading.Event(),
        worker_thread=None,
        proxy_thread=None,
        zmq_context=None,
        put_get_socket=None,
    )


def test_non_overflow_drops_are_separated_from_overflows():
    """ListenDrops counts every establishment failure; only some are queue overflows.

    The kernel charges a listening socket's sk_drops on several paths -- a full accept
    queue, but also failures to allocate or route the new connection -- so this difference
    is what says whether a bigger backlog could have helped.
    """
    stats = AcceptQueueStats(port=1234)
    stats.first_sample = _sample(0, overflows=10, drops=20)
    stats.last_sample = _sample(0, overflows=11, drops=25)

    assert stats.overflow_delta == 1
    assert stats.non_overflow_drop_delta == 4


def test_pure_overflow_window_reports_no_other_drops():
    stats = AcceptQueueStats(port=1234)
    stats.first_sample = _sample(0, overflows=10, drops=10)
    stats.last_sample = _sample(0, overflows=13, drops=13)

    assert stats.overflow_delta == 3
    assert stats.non_overflow_drop_delta == 0


def test_drop_alert_does_not_assert_the_queue_overflowed(caplog):
    """sk_drops locates the socket, not the cause, so the alert must not name one."""
    probe = _probe()
    probe._record(_sample(0, sk_drops=1))

    with caplog.at_level("ERROR"):
        probe._record(_sample(0, sk_drops=2))

    assert "dropped an incoming connection" in caplog.text
    assert "accept queue dropped" not in caplog.text
