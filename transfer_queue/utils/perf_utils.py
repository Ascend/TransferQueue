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

import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.INFO))

# Ensure logger has a handler
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

TQ_PERF_LOG_FLUSH_INTERVAL = float(os.environ.get("TQ_PERF_LOG_FLUSH_INTERVAL", 300))  # in seconds


class IntervalPerfMonitor:
    """
    Monitors and logs performance statistics for operations over configurable time intervals.

    This class is designed to be used in contexts where you want to track the number of successful
    operations and their processing times, and periodically log summary statistics such as request
    counts, rates, and timing metrics (average, max, min) per operation type.

    Usage:
        monitor = IntervalPerfMonitor("Your Class")
        with monitor.measure("method_name"):
            # perform upload operation

    At each interval (controlled by TQ_PERF_LOG_FLUSH_INTERVAL), the monitor logs aggregated
    statistics and resets its counters.

    Args:
        caller_name (str): Name of the component or caller using the monitor, included in logs.
    """

    def __init__(self, caller_name: str):
        self.caller_name = caller_name
        self.last_flush_time = time.perf_counter()

        self.success_counts: dict[str, int] = defaultdict(int)
        self.process_time: dict[str, list[float]] = defaultdict(list)

    def _flush_logs(self):
        """
        Internal method to conditionally flush (log) aggregated performance statistics.

        If the configured time interval (TQ_PERF_LOG_FLUSH_INTERVAL) has passed since the last flush,
        this method logs:
          - Total number of successful requests and requests per minute.
          - Average processing time across all operations.
          - For each operation type: request count, requests per minute, average, max, and min processing times.
        After logging, all statistics are reset and the flush timer is updated.
        """
        now = time.perf_counter()

        # only flush if the interval has passed
        if (now - self.last_flush_time) >= TQ_PERF_LOG_FLUSH_INTERVAL:
            minutes = (now - self.last_flush_time) / 60

            total_requests = sum(self.success_counts.values())
            total_process_time = sum(sum(time_list) for time_list in self.process_time.values())
            total_avg_process_time = total_process_time / total_requests if total_requests > 0 else 0.0

            # max/min/avg time for each operation type
            op_detail_stats = []
            for op_type, count in self.success_counts.items():
                times = self.process_time[op_type]
                if not times:
                    op_avg = op_max = op_min = 0.0
                else:
                    op_avg = sum(times) / len(times)
                    op_max = max(times)
                    op_min = min(times)

                op_detail_stats.append(
                    f"{op_type}: req_count={count}, req/min={count / minutes:.2f}, "
                    f"avg_time={op_avg:.6f}s, max_time={op_max:.6f}s, min_time={op_min:.6f}s"
                )

            log_msg = (
                f"{self.caller_name}: [Performance] "
                f"Total success requests: {total_requests}, "
                f"Total req/min: {total_requests / minutes:.2f}, "
                f"Total avg process time: {total_avg_process_time:.4f}s; \n"
                f"Time range: last {minutes:.2f} minutes; \n"
                f"Per-operation statistics: {'; '.join(op_detail_stats)}"
            )

            logger.info(log_msg)

            # reset counts
            self.success_counts.clear()
            self.process_time.clear()
            self.last_flush_time = now

    @contextmanager
    def measure(self, op_type: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            cost = time.perf_counter() - start_time
            self.success_counts[op_type] += 1
            self.process_time[op_type].append(cost)

            # try flush logs
            self._flush_logs()
