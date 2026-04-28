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
from contextlib import contextmanager
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import psutil
import zmq
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from transfer_queue.utils.zmq_utils import (
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    create_zmq_socket,
    format_zmq_address,
)

if TYPE_CHECKING:
    from transfer_queue.controller import TransferQueueController

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", logging.INFO))

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(handler)

TQ_METRICS_PORT = int(os.environ.get("TQ_METRICS_PORT", 0))
TQ_METRICS_COLLECT_INTERVAL = int(os.environ.get("TQ_METRICS_COLLECT_INTERVAL", 10))
TQ_METRICS_STORAGE_TIMEOUT = int(os.environ.get("TQ_METRICS_STORAGE_TIMEOUT", 5))


class TQMetricsExporter:
    """Prometheus metrics exporter embedded in TransferQueueController.

    Exposes an HTTP ``/metrics`` endpoint for Prometheus scraping and periodically
    collects internal statistics from the Controller (in-process) and from
    SimpleStorageUnit instances (via ZMQ ``GET_METRICS`` requests).

    Lifecycle:
        1. Created by ``TransferQueueController.start_metrics()`` when metrics are enabled.
        2. ``start()`` launches the HTTP server and a background collection thread.
        3. The collection thread calls ``collect_controller_metrics`` and
           ``collect_storage_metrics`` every ``TQ_METRICS_COLLECT_INTERVAL`` seconds.

    Environment variables:
        TQ_METRICS_PORT              HTTP port for /metrics (default 0 = auto-assign free port)
        TQ_METRICS_COLLECT_INTERVAL  Collection interval in seconds (default 10)
        TQ_METRICS_STORAGE_TIMEOUT   ZMQ timeout for storage queries (default 5s)
    """

    def __init__(self, controller: "TransferQueueController"):
        self._controller = controller
        self._start_time = time.time()
        self._process = psutil.Process()
        self._storage_unit_infos: dict[str, ZMQServerInfo] = {}
        self._zmq_ctx: zmq.Context | None = None
        self._zmq_sockets: dict[str, zmq.Socket] = {}
        self._known_partition_ids: set[str] = set()
        self._known_consumption_labels: set[tuple[str, str]] = set()
        self._metrics_endpoint: str = ""
        self.registry = CollectorRegistry()
        self._define_metrics()

    def _define_metrics(self) -> None:
        r = self.registry

        # ---- Controller process metrics ----
        self.controller_uptime = Gauge("tq_controller_uptime_seconds", "Controller uptime in seconds", registry=r)
        self.controller_memory_rss = Gauge(
            "tq_controller_memory_rss_bytes", "Controller process RSS memory in bytes", registry=r
        )

        # ---- Partition metrics ----
        self.partitions_total = Gauge("tq_partitions_total", "Total number of active partitions", registry=r)
        self.partition_samples = Gauge(
            "tq_partition_samples_total", "Number of active samples in a partition", ["partition_id"], registry=r
        )
        self.partition_production_progress = Gauge(
            "tq_partition_production_progress",
            "Production progress ratio (0.0-1.0)",
            ["partition_id"],
            registry=r,
        )
        self.partition_consumption_progress = Gauge(
            "tq_partition_consumption_progress",
            "Consumption progress ratio (0.0-1.0)",
            ["partition_id", "task_name"],
            registry=r,
        )

        # ---- Index manager metrics ----
        self.global_index_allocated = Gauge(
            "tq_global_index_allocated_total", "Total allocated global indexes", registry=r
        )
        self.global_index_reusable = Gauge(
            "tq_global_index_reusable_total", "Number of reusable global indexes", registry=r
        )

        # ---- Controller request latency / throughput ----
        self.request_duration = Histogram(
            "tq_controller_request_duration_seconds",
            "Controller request processing duration",
            ["op_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
            registry=r,
        )
        self.request_total = Counter(
            "tq_controller_request_total",
            "Total number of requests processed by the controller",
            ["op_type"],
            registry=r,
        )
        self.request_errors_total = Counter(
            "tq_controller_request_errors_total",
            "Total number of request errors in the controller",
            ["op_type"],
            registry=r,
        )

        # ---- Storage unit metrics ----
        self.storage_capacity = Gauge(
            "tq_storage_capacity_total", "Storage unit capacity (max keys)", ["storage_unit_id"], registry=r
        )
        self.storage_active_keys = Gauge(
            "tq_storage_active_keys_total", "Active keys in storage unit", ["storage_unit_id"], registry=r
        )
        self.storage_utilization = Gauge(
            "tq_storage_utilization_ratio",
            "Storage utilization ratio (active/capacity)",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_memory_rss = Gauge(
            "tq_storage_memory_rss_bytes", "Storage unit process RSS memory", ["storage_unit_id"], registry=r
        )

    @contextmanager
    def measure(self, op_type: str):
        """Context manager that records request count and latency for *op_type*.

        Usage::

            with metrics.measure("GET_META"):
                result = self.get_metadata(...)
        """
        self.request_total.labels(op_type=op_type).inc()
        start = time.perf_counter()
        try:
            yield
        except Exception:
            self.request_errors_total.labels(op_type=op_type).inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.request_duration.labels(op_type=op_type).observe(elapsed)

    def register_storage_units(self, storage_unit_infos: dict[str, ZMQServerInfo]) -> None:
        """Register SimpleStorageUnit ZMQ endpoints for metrics collection."""
        self._storage_unit_infos.update(storage_unit_infos)
        logger.info(f"Metrics exporter registered {len(storage_unit_infos)} storage units")

    def collect_controller_metrics(self) -> None:
        """Collect metrics from the Controller's in-memory state."""
        ctrl = self._controller

        # Process-level
        self.controller_uptime.set(time.time() - self._start_time)
        try:
            self.controller_memory_rss.set(self._process.memory_info().rss)
        except Exception:
            pass

        # Partition-level — iterate over a snapshot to avoid RuntimeError
        # if partitions dict is mutated concurrently.
        partitions_snapshot = list(ctrl.partitions.items())
        current_pids = {pid for pid, _ in partitions_snapshot}
        current_consumption_labels: set[tuple[str, str]] = set()
        self.partitions_total.set(len(current_pids))

        for pid, partition in partitions_snapshot:
            stats = partition.get_statistics()
            self.partition_samples.labels(partition_id=pid).set(stats["total_samples_num"])
            self.partition_production_progress.labels(partition_id=pid).set(stats.get("production_progress", 0))

            for task_name, cstats in stats.get("consumption_statistics", {}).items():
                self.partition_consumption_progress.labels(partition_id=pid, task_name=task_name).set(
                    cstats.get("consumption_progress", 0)
                )
                current_consumption_labels.add((pid, task_name))

        # Prune stale partition labels
        for stale_pid in self._known_partition_ids - current_pids:
            for metric in (
                self.partition_samples,
                self.partition_production_progress,
            ):
                try:
                    metric.remove(stale_pid)
                except (KeyError, ValueError):
                    pass
        for stale_pair in self._known_consumption_labels - current_consumption_labels:
            try:
                self.partition_consumption_progress.remove(*stale_pair)
            except (KeyError, ValueError):
                pass
        self._known_partition_ids = current_pids
        self._known_consumption_labels = current_consumption_labels

        # Index manager
        self.global_index_allocated.set(len(ctrl.index_manager.allocated_indexes))
        self.global_index_reusable.set(len(ctrl.index_manager.reusable_indexes))

    def collect_storage_metrics(self) -> None:
        """Query each registered SimpleStorageUnit for metrics via ZMQ."""
        if not self._storage_unit_infos:
            return

        # Iterate over a snapshot to avoid RuntimeError from concurrent mutation.
        storage_snapshot = list(self._storage_unit_infos.items())
        for su_id, su_info in storage_snapshot:
            try:
                metrics = self._query_storage_unit(su_info, su_id)
                if metrics is None:
                    continue
                # Use the storage unit's own ID from the response as the
                # canonical label to keep dashboard labels consistent with logs.
                label = metrics.get("storage_unit_id", su_id)
                capacity = metrics.get("capacity", 0)
                active = metrics.get("active_keys", 0)
                self.storage_capacity.labels(storage_unit_id=label).set(capacity)
                self.storage_active_keys.labels(storage_unit_id=label).set(active)
                self.storage_utilization.labels(storage_unit_id=label).set(
                    active / capacity if capacity > 0 else 0.0
                )
                self.storage_memory_rss.labels(storage_unit_id=label).set(metrics.get("process_rss_bytes", 0))
            except Exception as e:
                logger.warning(f"Failed to collect metrics from storage unit {su_id}: {e}")

    def _get_or_create_socket(self, su_id: str, su_info: ZMQServerInfo) -> zmq.Socket:
        """Return a cached ZMQ DEALER socket for *su_id*, creating one if needed."""
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()

        sock = self._zmq_sockets.get(su_id)
        if sock is not None and not sock.closed:
            return sock

        identity = f"metrics_collector_{uuid4().hex[:8]}".encode()
        sock = create_zmq_socket(self._zmq_ctx, zmq.DEALER, su_info.ip, identity)
        timeout_ms = TQ_METRICS_STORAGE_TIMEOUT * 1000
        address = format_zmq_address(su_info.ip, su_info.ports.get("put_get_socket"))
        sock.connect(address)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._zmq_sockets[su_id] = sock
        return sock

    def _query_storage_unit(self, su_info: ZMQServerInfo, su_id: str) -> Optional[dict[str, Any]]:
        """Send a synchronous GET_METRICS request to a single storage unit."""
        try:
            sock = self._get_or_create_socket(su_id, su_info)
            request_msg = ZMQMessage.create(
                request_type=ZMQRequestType.GET_METRICS,
                sender_id="metrics_collector",
                body={},
            )
            sock.send_multipart(request_msg.serialize())
            response_frames = sock.recv_multipart(copy=False)
            response_msg = ZMQMessage.deserialize(response_frames)
            if response_msg.request_type == ZMQRequestType.METRICS_RESPONSE:
                return response_msg.body
            return None
        except zmq.error.Again:
            logger.debug(f"Timeout querying metrics from {su_id}")
            return None
        except Exception as e:
            logger.warning(f"Error querying metrics from {su_id}: {e}")
            # Close broken socket so it gets recreated next cycle
            sock = self._zmq_sockets.pop(su_id, None)
            if sock and not sock.closed:
                sock.close(linger=0)
            return None

    def start(self, node_ip: str = "0.0.0.0") -> str:
        """Start the HTTP /metrics server and the background collection thread.

        When ``TQ_METRICS_PORT`` is ``0`` (the default), the OS assigns a free
        port automatically — the actual port is read back from the server socket.

        Args:
            node_ip: The IP address of the node hosting the Controller Actor.

        Returns:
            The metrics endpoint address in ``host:port`` format.
        """
        httpd, _ = start_http_server(TQ_METRICS_PORT, registry=self.registry)
        actual_port = httpd.server_address[1]
        self._metrics_endpoint = f"{node_ip}:{actual_port}"
        logger.info(f"TQ Metrics HTTP server started on {self._metrics_endpoint}")

        self._collect_thread = Thread(
            target=self._collect_loop,
            name="TQMetricsCollectorThread",
            daemon=True,
        )
        self._collect_thread.start()
        return self._metrics_endpoint

    def _collect_loop(self) -> None:
        """Background loop that periodically collects all metrics."""
        while True:
            try:
                self.collect_controller_metrics()
                self.collect_storage_metrics()
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
            time.sleep(TQ_METRICS_COLLECT_INTERVAL)
