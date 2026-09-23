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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from contextlib import contextmanager
from threading import Thread
from typing import Any

import psutil
import zmq
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import (
    METRICS_COLLECTOR_IDENTITY_PREFIX,
    ZMQMessage,
    ZMQRequestType,
    ZMQServerInfo,
    ZMQSocketPool,
)

logger = get_logger(__name__)

TQ_METRICS_COLLECT_INTERVAL = int(os.environ.get("TQ_METRICS_COLLECT_INTERVAL", 10))
TQ_METRICS_STORAGE_TIMEOUT = int(os.environ.get("TQ_METRICS_STORAGE_TIMEOUT", 5))


class TQMetricsExporter:
    """Prometheus metrics exporter for TransferQueue.

    Exposes an HTTP ``/metrics`` endpoint for Prometheus scraping and periodically
    updates Prometheus gauges from a *snapshot* dict pushed by the controller.

    **Decoupling strategy** — the exporter never accesses controller internals
    directly.  Instead, the controller calls ``update_controller_snapshot()``
    periodically (from its own thread) to hand over a plain ``dict`` that the
    background collection thread reads.  This eliminates:

    * Lock contention between the metrics thread and controller request threads.
    * Risk of reading tensors that are being resized concurrently.

    Storage-unit metrics are still collected via ZMQ (already process-isolated).

    Lifecycle:
        1. Created by ``TransferQueueController.start_metrics()`` when metrics are enabled.
        2. ``start()`` launches the HTTP server and a background collection thread.
        3. The controller calls ``update_controller_snapshot()`` on its own cadence.
        4. The collection thread calls ``collect_controller_metrics`` (reads snapshot)
           and ``collect_storage_metrics`` every ``TQ_METRICS_COLLECT_INTERVAL`` seconds.

    Environment variables:
        TQ_METRICS_COLLECT_INTERVAL  Collection interval in seconds (default 10)
        TQ_METRICS_STORAGE_TIMEOUT   ZMQ timeout for storage queries (default 5s)
    """

    def __init__(self, role: str = "controller", zmq_context: zmq.Context | None = None):
        """
        Args:
            role: Which process this exporter runs in; only "controller" collects from
                storage units, so only that role needs a context.
            zmq_context: The owner's long-lived synchronous context, borrowed for
                storage-unit queries and never terminated here. Minting one instead would
                add a second context and its native I/O thread with nobody to close them,
                since the exporter lives as long as its Ray actor.
        """
        self._start_time = time.time()
        self._process = psutil.Process()
        self._role = role
        self._storage_unit_infos: dict[str, ZMQServerInfo] = {}
        self._zmq_ctx = zmq_context
        self._zmq_socket_pool: ZMQSocketPool | None = None
        self._known_partition_ids: set[str] = set()
        self._known_production_labels: set[tuple[str, str]] = set()
        self._known_consumption_labels: set[tuple[str, str]] = set()
        self._metrics_endpoint: str = ""

        # Plain-dict snapshot pushed by the controller via update_controller_snapshot().
        # Dict reference assignment is atomic under the GIL, so no lock is needed.
        self._controller_snapshot: dict[str, Any] = {}

        self.registry = CollectorRegistry()
        self._define_metrics()

    @property
    def endpoint(self) -> str:
        """The metrics HTTP endpoint address in ``host:port`` format."""
        return self._metrics_endpoint

    def _define_metrics(self) -> None:
        r = self.registry
        role = self._role

        # ---- Request latency / throughput (role-prefixed) ----
        self.request_duration = Histogram(
            f"tq_{role}_request_duration_seconds",
            f"{role.capitalize()} request processing duration",
            ["op_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0],
            registry=r,
        )
        self.request_total = Counter(
            f"tq_{role}_request",
            f"Total number of requests processed by the {role}",
            ["op_type"],
            registry=r,
        )
        self.request_errors_total = Counter(
            f"tq_{role}_request_errors",
            f"Total number of request errors in the {role}",
            ["op_type"],
            registry=r,
        )
        self.request_samples_total = Counter(
            f"tq_{role}_request_samples",
            f"Total number of samples processed per operation type in the {role}",
            ["op_type"],
            registry=r,
        )

        if role == "controller":
            self._define_controller_metrics()

    def _define_controller_metrics(self) -> None:
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
            ["partition_id", "task_name"],
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
        self.storage_ssd_offload_enabled = Gauge(
            "tq_storage_ssd_offload_enabled",
            "Whether SSD offload is enabled for the storage unit",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_ssd_active_values = Gauge(
            "tq_storage_ssd_active_values",
            "Active field values stored on SSD",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_ssd_active_bytes = Gauge(
            "tq_storage_ssd_active_bytes",
            "Logical bytes held by active SSD-backed values",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_ssd_fallback_values = Gauge(
            "tq_storage_ssd_fallback_values_total",
            "Values retained in memory because SSD encoding was unavailable",
            ["storage_unit_id"],
            registry=r,
        )

        # ---- Storage-unit request-loss diagnostics ----
        # Read against tq_storage_request_ops, which advances only on completion: a gap
        # between the two is requests arriving and not finishing.
        self.storage_requests_arrived = Gauge(
            "tq_storage_requests_arrived",
            "Requests decoded by the storage unit worker, whether or not they completed",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_arrivals_by_op = Gauge(
            "tq_storage_arrivals_by_op",
            "Requests decoded by the storage unit worker, by operation",
            ["storage_unit_id", "op_type"],
            registry=r,
        )

        # ---- Accept-queue probe (only populated when TQ_ACCEPT_PROBE_INTERVAL > 0) ----
        self.storage_accept_queue_backlog = Gauge(
            "tq_storage_accept_queue_backlog",
            "Configured accept-queue depth of the storage unit's listening socket",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_accept_queue_peak = Gauge(
            "tq_storage_accept_queue_peak",
            "Deepest accept-queue occupancy seen by the probe",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_accept_queue_peak_utilization = Gauge(
            "tq_storage_accept_queue_peak_utilization_ratio",
            "Peak accept-queue occupancy as a fraction of the backlog",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_socket_drops = Gauge(
            "tq_storage_socket_drops",
            "Connections dropped on this listening socket since the probe started",
            ["storage_unit_id"],
            registry=r,
        )
        # Split because sk_drops covers several establishment failures, not only a full
        # queue; the difference is what says whether a bigger backlog would have helped.
        self.storage_listen_overflows = Gauge(
            "tq_storage_listen_overflows",
            "Namespace-wide accept-queue overflows since the probe started",
            ["storage_unit_id"],
            registry=r,
        )
        self.storage_listen_other_drops = Gauge(
            "tq_storage_listen_other_drops",
            "Namespace-wide establishment drops that were not accept-queue overflows",
            ["storage_unit_id"],
            registry=r,
        )

        # ---- Storage request metrics (collected via ZMQ, exposed as gauges) ----
        # P50/P99 are pre-computed on the storage unit side and sent via ZMQ,
        # avoiding the need to replicate histogram bucket structures (which
        # conflict with Prometheus reserved suffixes and label semantics).
        self.storage_request_ops = Gauge(
            "tq_storage_request_ops",
            "Total requests processed by storage unit",
            ["storage_unit_id", "op_type"],
            registry=r,
        )
        self.storage_request_latency_avg = Gauge(
            "tq_storage_request_latency_avg",
            "Average request latency in storage unit (seconds)",
            ["storage_unit_id", "op_type"],
            registry=r,
        )
        self.storage_request_latency_p50 = Gauge(
            "tq_storage_request_latency_p50",
            "P50 request latency in storage unit (seconds)",
            ["storage_unit_id", "op_type"],
            registry=r,
        )
        self.storage_request_latency_p99 = Gauge(
            "tq_storage_request_latency_p99",
            "P99 request latency in storage unit (seconds)",
            ["storage_unit_id", "op_type"],
            registry=r,
        )

    @contextmanager
    def measure(self, op_type: str):
        """Context manager that records request count and latency for *op_type*.

        Prometheus errors are caught and logged so that metrics failures never
        block or crash the controller's request processing loop.

        Usage::

            with metrics.measure("GET_META"):
                result = self.get_metadata(...)
        """
        try:
            self.request_total.labels(op_type=op_type).inc()
        except Exception:
            logger.debug(f"Metrics: failed to increment counter for {op_type}", exc_info=True)
        start = time.perf_counter()
        try:
            yield
        except Exception:
            try:
                self.request_errors_total.labels(op_type=op_type).inc()
            except Exception:
                logger.debug(f"Metrics: failed to increment error counter for {op_type}", exc_info=True)
            raise
        finally:
            elapsed = time.perf_counter() - start
            try:
                self.request_duration.labels(op_type=op_type).observe(elapsed)
            except Exception:
                logger.debug(f"Metrics: failed to observe duration for {op_type}", exc_info=True)

    def record_samples(self, op_type: str, count: int) -> None:
        """Record the number of samples processed in a single operation."""
        try:
            self.request_samples_total.labels(op_type=op_type).inc(count)
        except Exception:
            logger.debug(f"Metrics: failed to record samples for {op_type}", exc_info=True)

    def register_storage_units(self, storage_unit_infos: dict[str, ZMQServerInfo]) -> None:
        """Register SimpleStorageUnit ZMQ endpoints for metrics collection."""
        self._storage_unit_infos.update(storage_unit_infos)
        logger.info(f"Metrics exporter registered {len(storage_unit_infos)} storage units")

    def update_controller_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Replace the controller metrics snapshot (called from the controller thread).

        The snapshot is a plain dict with no references to live controller objects,
        so the metrics thread can read it without risk of concurrent mutation.
        """
        self._controller_snapshot = snapshot

    def collect_controller_metrics(self) -> None:
        """Update Prometheus gauges from the latest controller snapshot."""

        # Process-level
        self.controller_uptime.set(time.time() - self._start_time)
        try:
            self.controller_memory_rss.set(self._process.memory_info().rss)
        except Exception:
            pass

        snapshot = self._controller_snapshot
        partitions = snapshot.get("partitions", {})
        current_pids = set(partitions.keys())
        current_consumption_labels: set[tuple[str, str]] = set()
        self.partitions_total.set(len(current_pids))

        current_production_labels: set[tuple[str, str]] = set()

        for pid, stats in partitions.items():
            self.partition_samples.labels(partition_id=pid).set(stats["total_samples_num"])

            for task_name, pstats in stats.get("production_statistics", {}).items():
                self.partition_production_progress.labels(partition_id=pid, task_name=task_name).set(
                    pstats.get("production_progress", 0)
                )
                current_production_labels.add((pid, task_name))

            for task_name, cstats in stats.get("consumption_statistics", {}).items():
                self.partition_consumption_progress.labels(partition_id=pid, task_name=task_name).set(
                    cstats.get("consumption_progress", 0)
                )
                current_consumption_labels.add((pid, task_name))

        # Prune stale partition labels
        for stale_pid in self._known_partition_ids - current_pids:
            try:
                self.partition_samples.remove(stale_pid)
            except (KeyError, ValueError):
                pass
        for stale_pair in self._known_production_labels - current_production_labels:
            try:
                self.partition_production_progress.remove(*stale_pair)
            except (KeyError, ValueError):
                pass
        for stale_pair in self._known_consumption_labels - current_consumption_labels:
            try:
                self.partition_consumption_progress.remove(*stale_pair)
            except (KeyError, ValueError):
                pass
        self._known_partition_ids = current_pids
        self._known_production_labels = current_production_labels
        self._known_consumption_labels = current_consumption_labels

        # Index manager
        self.global_index_allocated.set(snapshot.get("global_index_allocated", 0))
        self.global_index_reusable.set(snapshot.get("global_index_reusable", 0))

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
                # ``capacity`` is ``None`` when the storage unit is configured with
                # unlimited capacity (``total_storage_size=None``).
                if capacity is not None:
                    self.storage_capacity.labels(storage_unit_id=label).set(capacity)
                    self.storage_utilization.labels(storage_unit_id=label).set(
                        active / capacity if capacity > 0 else 0.0
                    )
                else:
                    # If the storage unit was previously reporting a numeric
                    # capacity and later transitions to unlimited (e.g. after
                    # re-registration), remove the stale series so dashboards
                    # do not keep serving outdated capacity / utilization
                    # values. This mirrors the stale-label pruning applied to
                    # partition-level gauges above.
                    for gauge in (self.storage_capacity, self.storage_utilization):
                        try:
                            gauge.remove(label)
                        except (KeyError, ValueError):
                            pass
                self.storage_active_keys.labels(storage_unit_id=label).set(active)
                self.storage_memory_rss.labels(storage_unit_id=label).set(metrics.get("process_rss_bytes", 0))
                self.storage_ssd_offload_enabled.labels(storage_unit_id=label).set(
                    metrics.get("ssd_offload_enabled", 0)
                )
                self.storage_ssd_active_values.labels(storage_unit_id=label).set(metrics.get("ssd_active_values", 0))
                self.storage_ssd_active_bytes.labels(storage_unit_id=label).set(metrics.get("ssd_active_bytes", 0))
                self.storage_ssd_fallback_values.labels(storage_unit_id=label).set(
                    metrics.get("ssd_fallback_values_total", 0)
                )

                self.storage_requests_arrived.labels(storage_unit_id=label).set(metrics.get("requests_arrived", 0))
                for op_type, arrived in (metrics.get("arrivals_by_op") or {}).items():
                    self.storage_arrivals_by_op.labels(storage_unit_id=label, op_type=op_type).set(arrived)

                # Drop the series rather than report zero when the probe is off, so a
                # disabled probe is not read as "measured, and no drops".
                accept_queue = metrics.get("accept_queue")
                accept_gauges = (
                    (self.storage_accept_queue_backlog, "backlog"),
                    (self.storage_accept_queue_peak, "peak_recv_q"),
                    (self.storage_accept_queue_peak_utilization, "peak_utilization"),
                    (self.storage_socket_drops, "sk_drops_delta"),
                    (self.storage_listen_overflows, "listen_overflow_delta"),
                    (self.storage_listen_other_drops, "non_overflow_drop_delta"),
                )
                for gauge, key in accept_gauges:
                    if accept_queue is None:
                        try:
                            gauge.remove(label)
                        except (KeyError, ValueError):
                            pass
                    else:
                        gauge.labels(storage_unit_id=label).set(accept_queue.get(key, 0))

                # Per-operation request stats
                for op_type, op_data in metrics.get("op_stats", {}).items():
                    self.storage_request_ops.labels(storage_unit_id=label, op_type=op_type).set(
                        op_data.get("request_count", 0)
                    )
                    self.storage_request_latency_avg.labels(storage_unit_id=label, op_type=op_type).set(
                        op_data.get("latency_avg", 0)
                    )
                    self.storage_request_latency_p50.labels(storage_unit_id=label, op_type=op_type).set(
                        op_data.get("latency_p50", 0)
                    )
                    self.storage_request_latency_p99.labels(storage_unit_id=label, op_type=op_type).set(
                        op_data.get("latency_p99", 0)
                    )
            except Exception as e:
                logger.warning(f"Failed to collect metrics from storage unit {su_id}: {e}")

    def _get_socket_pool(self) -> ZMQSocketPool:
        """Return the lazily-created socket pool for storage-unit queries."""
        if self._zmq_socket_pool is None:
            if self._zmq_ctx is None:
                raise RuntimeError(
                    "TQMetricsExporter was built without a ZMQ context, so it cannot query "
                    "storage units; pass zmq_context= from the owning process."
                )
            self._zmq_socket_pool = ZMQSocketPool(
                self._zmq_ctx,
                # The storage proxy drops identities without this prefix, and the pool
                # builds each socket's identity from the owner id.
                METRICS_COLLECTOR_IDENTITY_PREFIX.rstrip("_"),
                "put_get_socket",
                timeout=TQ_METRICS_STORAGE_TIMEOUT,
            )
        return self._zmq_socket_pool

    def _query_storage_unit(self, su_info: ZMQServerInfo, su_id: str) -> dict[str, Any] | None:
        """Send a synchronous GET_METRICS request to a single storage unit."""
        try:
            pool = self._get_socket_pool()
            with pool.lease(su_info) as sock:
                request_msg = ZMQMessage.create(
                    request_type=ZMQRequestType.GET_METRICS,
                    sender_id="metrics_collector",
                    body={},
                )
                sock.send_multipart(request_msg.serialize())
                response_frames = sock.recv_multipart(copy=False)
                response_msg = ZMQMessage.deserialize(response_frames)
                # Closed rather than parked: collection walks every unit once per cycle, so
                # a kept socket holds budget for the whole walk to save one handshake.
                sock.close(linger=0)
                if response_msg.request_type == ZMQRequestType.METRICS_RESPONSE:
                    return response_msg.body
                return None
        except zmq.error.Again:
            # The pool discarded the socket, so the next cycle starts clean. Reusing it
            # would read this reply, if it lands late, as the answer to that cycle's query.
            logger.debug(f"Timeout querying metrics from {su_id}")
            return None
        except Exception as e:
            logger.warning(f"Error querying metrics from {su_id}: {e}")
            return None

    def start(self, node_ip: str = "0.0.0.0", port: int = 0) -> str:
        """Start the HTTP /metrics server and the background collection thread.

        When *port* is ``0`` (the default), the OS assigns a free port
        automatically — the actual port is read back from the server socket.

        Args:
            node_ip: The IP address of the node hosting the process.
            port: HTTP port for the /metrics endpoint.

        Returns:
            The metrics endpoint address in ``host:port`` format.
        """
        from prometheus_client import start_http_server

        httpd, _thread = start_http_server(port=port, addr=node_ip, registry=self.registry)
        actual_port = httpd.server_address[1]
        self._metrics_endpoint = f"{node_ip}:{actual_port}"
        logger.info(f"TQ Metrics HTTP server started on {self._metrics_endpoint}")

        if self._role == "controller":
            self._collect_thread = Thread(
                target=self._collect_loop,
                name="TQMetricsCollectorThread",
                daemon=True,
            )
            self._collect_thread.start()
        return self._metrics_endpoint

    def _collect_loop(self) -> None:
        """Background loop that periodically collects controller and storage metrics."""
        while True:
            try:
                self.collect_controller_metrics()
                self.collect_storage_metrics()
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
            time.sleep(TQ_METRICS_COLLECT_INTERVAL)
