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

"""Accept-queue sampling for the storage-unit ROUTER socket.

A connection dropped while being established is invisible to both ends, so the only
way to see it is to read the kernel's own counters. Two caveats govern how callers
may read what this reports: ``ListenOverflows`` and ``ListenDrops`` are per network
namespace rather than per port, and the kernel charges a socket's ``sk_drops`` for
several establishment failures, of which a full queue is only one. See
``docs/metrics.md`` for the exported metrics.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass

from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)

_RECVQ_RE = re.compile(r"^LISTEN\s+(\d+)\s+(\d+)\s+\S*:(\d+)\s", re.M)
_DROPS_RE = re.compile(r"\bd(\d+)\b")


@dataclass
class AcceptQueueSample:
    """One reading of the accept queue for a single listening port."""

    timestamp: float
    recv_q: int
    backlog: int
    sk_drops: int
    listen_overflows: int
    listen_drops: int

    @property
    def utilization(self) -> float:
        """Queue depth as a fraction of the configured backlog."""
        return self.recv_q / self.backlog if self.backlog else 0.0


@dataclass
class AcceptQueueStats:
    """Peak and delta summary over a sampling window."""

    port: int
    samples: int = 0
    peak_recv_q: int = 0
    peak_utilization: float = 0.0
    backlog: int = 0
    first_sample: AcceptQueueSample | None = None
    last_sample: AcceptQueueSample | None = None

    @property
    def sk_drops_delta(self) -> int:
        """Connections this socket dropped during the window, 0 until two samples exist."""
        if self.first_sample is None or self.last_sample is None:
            return 0
        return self.last_sample.sk_drops - self.first_sample.sk_drops

    @property
    def overflow_delta(self) -> int:
        """Namespace-wide accept-queue overflows during the window, 0 until two samples exist."""
        if self.first_sample is None or self.last_sample is None:
            return 0
        return self.last_sample.listen_overflows - self.first_sample.listen_overflows

    @property
    def non_overflow_drop_delta(self) -> int:
        """Establishment drops in this window that were not accept-queue overflows."""
        if self.first_sample is None or self.last_sample is None:
            return 0
        drops = self.last_sample.listen_drops - self.first_sample.listen_drops
        return drops - self.overflow_delta

    def describe(self) -> str:
        """Return a one-line summary of the window, for the probe's shutdown log."""
        return (
            f"port={self.port} samples={self.samples} backlog={self.backlog} "
            f"peak_recv_q={self.peak_recv_q} peak_util={self.peak_utilization:.1%} "
            f"sk_drops_delta={self.sk_drops_delta} listen_overflow_delta={self.overflow_delta}"
        )


def _read_listen_socket(port: int) -> tuple[int, int, int] | None:
    """Return (recv_q, backlog, sk_drops) for the listening socket on ``port``."""
    try:
        out = subprocess.run(["ss", "-lntm"], capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"accept-probe: ss failed: {exc}")
        return None

    for match in _RECVQ_RE.finditer(out):
        recv_q, backlog, found_port = (int(g) for g in match.groups())
        if found_port != port:
            continue
        # skmem lives on the continuation line right after the match.
        tail = out[match.end() : match.end() + 400]
        drops_match = _DROPS_RE.search(tail.split("\n")[1] if "\n" in tail else "")
        return recv_q, backlog, int(drops_match.group(1)) if drops_match else 0
    return None


def _read_listen_overflows() -> tuple[int, int]:
    """Return this namespace's (ListenOverflows, ListenDrops) from /proc/net/netstat."""
    try:
        with open("/proc/net/netstat") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return 0, 0

    for i, line in enumerate(lines):
        if not line.startswith("TcpExt:") or "ListenOverflows" not in line:
            continue
        keys = line.split()
        values = lines[i + 1].split()
        try:
            return (
                int(values[keys.index("ListenOverflows")]),
                int(values[keys.index("ListenDrops")]),
            )
        except (ValueError, IndexError):
            return 0, 0
    return 0, 0


def sample_accept_queue(port: int) -> AcceptQueueSample | None:
    """Take one accept-queue reading for ``port``, or None if it cannot be read."""
    listen = _read_listen_socket(port)
    if listen is None:
        return None
    recv_q, backlog, sk_drops = listen
    overflows, drops = _read_listen_overflows()
    return AcceptQueueSample(
        timestamp=time.time(),
        recv_q=recv_q,
        backlog=backlog,
        sk_drops=sk_drops,
        listen_overflows=overflows,
        listen_drops=drops,
    )


class AcceptQueueProbe:
    """Sample one listening port's accept queue from a background thread.

    Args:
        port: Listening port to watch.
        owner_id: Identifier used in log lines (the storage unit id).
        interval_s: Seconds between samples. Keep sub-second; the queue drains in
            milliseconds, so a slower cadence misses the burst entirely.
        warn_utilization: Warn once when depth first reaches this fraction of the backlog.
    """

    def __init__(
        self,
        port: int,
        owner_id: str,
        interval_s: float = 0.1,
        warn_utilization: float = 0.5,
    ) -> None:
        self.port = port
        self.owner_id = owner_id
        self.interval_s = interval_s
        self.warn_utilization = warn_utilization
        self.stats = AcceptQueueStats(port=port)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False

    def start(self) -> None:
        """Start the sampling thread; a second call is a no-op."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"AcceptQueueProbe-{self.owner_id}", daemon=True)
        self._thread.start()
        logger.info(f"[{self.owner_id}]: accept-queue probe started on port {self.port} (interval={self.interval_s}s)")

    def stop(self) -> None:
        """Stop the sampling thread and log the window summary."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info(f"[{self.owner_id}]: accept-queue probe stopped. {self.stats.describe()}")

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = sample_accept_queue(self.port)
            if sample is not None:
                self._record(sample)
            self._stop.wait(self.interval_s)

    def _record(self, sample: AcceptQueueSample) -> None:
        stats = self.stats
        stats.samples += 1
        stats.backlog = sample.backlog or stats.backlog
        previous = stats.last_sample
        if stats.first_sample is None:
            stats.first_sample = sample
        stats.last_sample = sample

        if sample.recv_q > stats.peak_recv_q:
            stats.peak_recv_q = sample.recv_q
            stats.peak_utilization = sample.utilization

        # Against the previous sample, not the first: sk_drops is cumulative, so measuring
        # from probe start would re-report one old drop on every sample.
        if previous is not None and sample.sk_drops > previous.sk_drops:
            logger.error(
                f"[{self.owner_id}]: listening socket on port {self.port} dropped an incoming "
                f"connection. recv_q={sample.recv_q}/{sample.backlog} sk_drops={sample.sk_drops} "
                f"(+{sample.sk_drops - previous.sk_drops} since the last sample, "
                f"+{stats.sk_drops_delta} since probe start); netns since probe start: "
                f"listen_overflows +{stats.overflow_delta}, other establishment drops "
                f"+{stats.non_overflow_drop_delta}. The kernel charges sk_drops for several "
                f"establishment failures, so a full queue is only one candidate: raising "
                f"ZMQ_BACKLOG above {sample.backlog} helps only if recv_q above and the "
                f"overflow delta point that way."
            )

        if not self._warned and sample.utilization >= self.warn_utilization:
            self._warned = True
            logger.warning(
                f"[{self.owner_id}]: accept queue on port {self.port} reached "
                f"{sample.recv_q}/{sample.backlog} ({sample.utilization:.0%}); "
                f"connections are queueing faster than they are accepted."
            )
