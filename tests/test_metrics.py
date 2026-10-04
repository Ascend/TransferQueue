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

"""Unit tests for the Prometheus metrics exporter (transfer_queue.metrics)."""

import time
from concurrent.futures import Future
from threading import Thread
from unittest.mock import MagicMock, patch

import pytest

try:
    import zmq

    from transfer_queue.metrics import TQMetricsExporter
    from transfer_queue.utils.enum_utils import Role
    from transfer_queue.utils.zmq_utils import ZMQMessage, ZMQRequestType, ZMQServerInfo

    _HAS_DEPS = True
except (ImportError, OSError):
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="prometheus_client / psutil / pyzmq dependencies unavailable")


# ---------------------------------------------------------------------------
# Helpers — build snapshot dicts that TQMetricsExporter.update_controller_snapshot expects
# ---------------------------------------------------------------------------


def _make_partition_snapshot(
    total_samples: int = 10,
    produced_ratio: float = 0.5,
    consumption: dict | None = None,
    tasks: list | None = None,
) -> dict:
    """Return a partition snapshot dict."""
    consumption_stats = {}
    if consumption:
        for task, progress in consumption.items():
            consumption_stats[task] = {"consumption_progress": progress}

    # Build per-task production statistics
    task_list = tasks or list((consumption or {}).keys())
    production_stats = {task: {"production_progress": produced_ratio} for task in task_list}

    return {
        "total_samples_num": total_samples,
        "production_statistics": production_stats,
        "consumption_statistics": consumption_stats,
    }


def _make_snapshot(partitions=None, allocated=10, reusable=2) -> dict:
    """Return a controller metrics snapshot dict."""
    return {
        "partitions": partitions or {},
        "global_index_allocated": allocated,
        "global_index_reusable": reusable,
    }


# ---------------------------------------------------------------------------
# Test: metric definitions
# ---------------------------------------------------------------------------


class TestMetricDefinitions:
    def test_all_metrics_are_registered(self):
        """Verify that all expected metric families exist in the exporter's registry."""
        exporter = TQMetricsExporter()

        expected_prefixes = [
            "tq_controller_uptime_seconds",
            "tq_controller_memory_rss_bytes",
            "tq_partitions_total",
            "tq_partition_samples_total",
            "tq_partition_production_progress",
            "tq_partition_consumption_progress",
            "tq_global_index_allocated_total",
            "tq_global_index_reusable_total",
            "tq_controller_request_duration_seconds",
            "tq_controller_request",
            "tq_controller_request_errors",
            "tq_storage_capacity_total",
            "tq_storage_active_keys_total",
            "tq_storage_utilization_ratio",
            "tq_storage_memory_rss_bytes",
            "tq_storage_ssd_offload_enabled",
            "tq_storage_ssd_active_values",
            "tq_storage_ssd_active_bytes",
            "tq_storage_ssd_fallback_values_total",
            "tq_storage_requests_arrived",
            "tq_storage_arrivals_by_op",
            "tq_storage_accept_queue_backlog",
            "tq_storage_accept_queue_peak",
            "tq_storage_accept_queue_peak_utilization_ratio",
            "tq_storage_socket_drops",
            "tq_storage_listen_overflows",
            "tq_storage_listen_other_drops",
        ]

        registered = {m.name for m in exporter.registry.collect()}
        for prefix in expected_prefixes:
            assert prefix in registered, f"Metric '{prefix}' not found in registry"


# ---------------------------------------------------------------------------
# Test: controller metrics collection
# ---------------------------------------------------------------------------


class TestControllerMetricsCollection:
    def test_collect_empty_controller(self):
        """Collect metrics from an empty snapshot — should not raise."""
        exporter = TQMetricsExporter()
        exporter.update_controller_snapshot(_make_snapshot(partitions={}, allocated=0, reusable=0))
        exporter.collect_controller_metrics()

        assert exporter.partitions_total._value.get() == 0
        assert exporter.global_index_allocated._value.get() == 0
        assert exporter.global_index_reusable._value.get() == 0

    def test_collect_with_partitions(self):
        """Partition-level metrics are populated correctly."""
        p1 = _make_partition_snapshot(total_samples=20, produced_ratio=0.8, consumption={"gen": 0.5})
        p2 = _make_partition_snapshot(total_samples=10, produced_ratio=1.0, consumption={"gen": 1.0, "train": 0.3})
        snapshot = _make_snapshot(partitions={"train_0": p1, "train_1": p2}, allocated=30, reusable=5)

        exporter = TQMetricsExporter()
        exporter.update_controller_snapshot(snapshot)
        exporter.collect_controller_metrics()

        assert exporter.partitions_total._value.get() == 2
        assert exporter.global_index_allocated._value.get() == 30
        assert exporter.global_index_reusable._value.get() == 5

        # Check partition-level gauges
        assert exporter.partition_samples.labels(partition_id="train_0")._value.get() == 20
        assert (
            exporter.partition_production_progress.labels(partition_id="train_0", task_name="gen")._value.get() == 0.8
        )
        assert (
            exporter.partition_consumption_progress.labels(partition_id="train_0", task_name="gen")._value.get() == 0.5
        )

        assert exporter.partition_samples.labels(partition_id="train_1")._value.get() == 10
        assert (
            exporter.partition_production_progress.labels(partition_id="train_1", task_name="gen")._value.get() == 1.0
        )
        assert (
            exporter.partition_consumption_progress.labels(partition_id="train_1", task_name="train")._value.get()
            == 0.3
        )

    def test_uptime_increases(self):
        """Controller uptime should be positive after collection."""
        exporter = TQMetricsExporter()
        exporter.update_controller_snapshot(_make_snapshot())
        time.sleep(0.05)
        exporter.collect_controller_metrics()
        assert exporter.controller_uptime._value.get() > 0


# ---------------------------------------------------------------------------
# Test: measure() context manager
# ---------------------------------------------------------------------------


class TestMeasureContextManager:
    def test_measure_records_count_and_duration(self):
        exporter = TQMetricsExporter()

        with exporter.measure("GET_META"):
            time.sleep(0.01)

        # Counter should have been incremented
        assert exporter.request_total.labels(op_type="GET_META")._value.get() == 1.0

        # Histogram should have at least one observation
        hist = exporter.request_duration.labels(op_type="GET_META")
        # _sum is the sum of observed values
        assert hist._sum.get() > 0

    def test_measure_records_errors(self):
        exporter = TQMetricsExporter()

        with pytest.raises(ValueError):
            with exporter.measure("BAD_OP"):
                raise ValueError("boom")

        assert exporter.request_errors_total.labels(op_type="BAD_OP")._value.get() == 1.0
        # The total counter should also be incremented (inc happens before yield)
        assert exporter.request_total.labels(op_type="BAD_OP")._value.get() == 1.0

    def test_multiple_ops_tracked_independently(self):
        exporter = TQMetricsExporter()

        for _ in range(3):
            with exporter.measure("GET_META"):
                pass
        for _ in range(2):
            with exporter.measure("CLEAR_PARTITION"):
                pass

        assert exporter.request_total.labels(op_type="GET_META")._value.get() == 3.0
        assert exporter.request_total.labels(op_type="CLEAR_PARTITION")._value.get() == 2.0


# ---------------------------------------------------------------------------
# Test: storage unit metrics collection
# ---------------------------------------------------------------------------


class TestStorageQuerySocketPool:
    def test_pool_borrows_the_owner_context(self):
        """The exporter must query storage units over the context it was handed.

        The controller already holds a long-lived synchronous context. A second one would
        add another native I/O thread and leave a context nobody closes, since the exporter
        lives for the whole life of its Ray actor.
        """
        ctx = zmq.Context()
        try:
            exporter = TQMetricsExporter(zmq_context=ctx)
            with patch("zmq.Context") as minted:
                exporter._get_socket_pool()
            minted.assert_not_called()
        finally:
            ctx.destroy(linger=0)

    def test_missing_context_is_reported(self):
        """Without a context there is nothing to query over, so say so rather than mint one."""
        exporter = TQMetricsExporter()
        with pytest.raises(RuntimeError, match="without a ZMQ context"):
            exporter._get_socket_pool()

    def test_collector_parks_no_socket_between_queries(self):
        """A queried unit must leave nothing in the pool.

        Collection walks every unit once per cycle, so a parked socket is reused only a
        cycle later while holding a slot in the controller context's budget for the whole
        walk. At a few thousand units that budget is what runs out first.
        """
        identities: set[bytes] = set()
        ctx_peer = zmq.Context()
        router = ctx_peer.socket(zmq.ROUTER)
        port = router.bind_to_random_port("tcp://127.0.0.1")
        running = True

        def serve():
            poller = zmq.Poller()
            poller.register(router, zmq.POLLIN)
            while running:
                if not dict(poller.poll(50)):
                    continue
                identity, _ = router.recv_multipart()
                identities.add(bytes(identity))
                response = ZMQMessage.create(
                    request_type=ZMQRequestType.METRICS_RESPONSE,
                    sender_id="storage_0",
                    body={},
                )
                router.send_multipart([identity, *response.serialize()])

        server = Thread(target=serve, daemon=True)
        server.start()

        su_info = ZMQServerInfo(role=Role.STORAGE, id="storage_0", ip="127.0.0.1", ports={"put_get_socket": port})
        ctx = zmq.Context()
        try:
            exporter = TQMetricsExporter(zmq_context=ctx)
            for _ in range(3):
                assert exporter._query_storage_unit(su_info, "storage_0") == {}
            # A parked socket would be reused, so a fresh identity per query is the
            # externally visible proof that nothing was kept.
            assert len(identities) == 3, "a queried unit left its socket in the pool"
        finally:
            running = False
            server.join(timeout=2.0)
            ctx.destroy(linger=0)
            router.close(linger=0)
            ctx_peer.term()


class TestStorageMetricsCollection:
    def test_collect_with_no_storage_units(self):
        """No storage units registered — collect should be a no-op."""
        exporter = TQMetricsExporter()
        # Should not raise
        exporter.collect_storage_metrics()

    def test_storage_metrics_populated_on_success(self):
        """Verify storage gauges are set when _query_storage_unit returns data."""
        exporter = TQMetricsExporter()

        fake_su_info = MagicMock()
        fake_su_info.id = "SU_001"
        exporter._storage_unit_infos = {"SU_001": fake_su_info}

        # Mock the ZMQ query to return fake metrics
        exporter._query_storage_unit = MagicMock(
            return_value={
                "storage_unit_id": "SU_001",
                "capacity": 1000,
                "active_keys": 250,
                "process_rss_bytes": 512 * 1024 * 1024,
                "ssd_offload_enabled": 1,
                "ssd_active_values": 120,
                "ssd_active_bytes": 4 * 1024 * 1024 * 1024,
                "ssd_fallback_values_total": 7,
            }
        )

        exporter.collect_storage_metrics()

        assert exporter.storage_capacity.labels(storage_unit_id="SU_001")._value.get() == 1000
        assert exporter.storage_active_keys.labels(storage_unit_id="SU_001")._value.get() == 250
        assert exporter.storage_utilization.labels(storage_unit_id="SU_001")._value.get() == 0.25
        assert exporter.storage_memory_rss.labels(storage_unit_id="SU_001")._value.get() == 512 * 1024 * 1024
        assert exporter.storage_ssd_offload_enabled.labels(storage_unit_id="SU_001")._value.get() == 1
        assert exporter.storage_ssd_active_values.labels(storage_unit_id="SU_001")._value.get() == 120
        assert exporter.storage_ssd_active_bytes.labels(storage_unit_id="SU_001")._value.get() == 4 * 1024 * 1024 * 1024
        assert exporter.storage_ssd_fallback_values.labels(storage_unit_id="SU_001")._value.get() == 7

        exporter._query_storage_unit.return_value["ssd_fallback_values_total"] = 9
        exporter.collect_storage_metrics()
        assert exporter.storage_ssd_fallback_values.labels(storage_unit_id="SU_001")._value.get() == 9

    def test_arrival_counters_are_exported(self):
        """Arrival counts reach Prometheus, so a dashboard can compare them with completions."""
        exporter = TQMetricsExporter()
        fake_su_info = MagicMock()
        fake_su_info.id = "SU_001"
        exporter._storage_unit_infos = {"SU_001": fake_su_info}
        exporter._query_storage_unit = MagicMock(
            return_value={
                "storage_unit_id": "SU_001",
                "capacity": 1000,
                "active_keys": 1,
                "requests_arrived": 42,
                "arrivals_by_op": {"GET_DATA": 30, "PUT_DATA": 12},
            }
        )

        exporter.collect_storage_metrics()

        assert exporter.storage_requests_arrived.labels(storage_unit_id="SU_001")._value.get() == 42
        by_op = exporter.storage_arrivals_by_op
        assert by_op.labels(storage_unit_id="SU_001", op_type="GET_DATA")._value.get() == 30
        assert by_op.labels(storage_unit_id="SU_001", op_type="PUT_DATA")._value.get() == 12

    def test_accept_queue_metrics_are_exported(self):
        """The overflow/non-overflow split is what tells a dashboard if backlog is the issue."""
        exporter = TQMetricsExporter()
        fake_su_info = MagicMock()
        fake_su_info.id = "SU_001"
        exporter._storage_unit_infos = {"SU_001": fake_su_info}
        exporter._query_storage_unit = MagicMock(
            return_value={
                "storage_unit_id": "SU_001",
                "capacity": 1000,
                "active_keys": 1,
                "accept_queue": {
                    "backlog": 4096,
                    "peak_recv_q": 97,
                    "peak_utilization": 0.02,
                    "sk_drops_delta": 5,
                    "listen_overflow_delta": 2,
                    "non_overflow_drop_delta": 3,
                },
            }
        )

        exporter.collect_storage_metrics()

        label = {"storage_unit_id": "SU_001"}
        assert exporter.storage_accept_queue_backlog.labels(**label)._value.get() == 4096
        assert exporter.storage_accept_queue_peak.labels(**label)._value.get() == 97
        assert exporter.storage_socket_drops.labels(**label)._value.get() == 5
        assert exporter.storage_listen_overflows.labels(**label)._value.get() == 2
        assert exporter.storage_listen_other_drops.labels(**label)._value.get() == 3

    def test_accept_queue_series_absent_when_the_probe_is_off(self):
        """The probe is opt-in; reporting 0 drops would read as 'measured, and none'."""
        exporter = TQMetricsExporter()
        fake_su_info = MagicMock()
        fake_su_info.id = "SU_001"
        exporter._storage_unit_infos = {"SU_001": fake_su_info}
        exporter._query_storage_unit = MagicMock(
            return_value={"storage_unit_id": "SU_001", "capacity": 1000, "active_keys": 1}
        )

        exporter.collect_storage_metrics()

        exported = {sample.name for metric in exporter.registry.collect() for sample in metric.samples}
        assert "tq_storage_socket_drops" not in exported
        assert "tq_storage_accept_queue_backlog" not in exported

    def test_storage_metrics_handles_query_failure(self):
        """If a storage unit query fails, other units should still be collected."""
        exporter = TQMetricsExporter()

        su1 = MagicMock()
        su1.id = "SU_001"
        su2 = MagicMock()
        su2.id = "SU_002"
        exporter._storage_unit_infos = {"SU_001": su1, "SU_002": su2}

        call_count = 0

        def mock_query(su_info, su_id):
            nonlocal call_count
            call_count += 1
            if su_id == "SU_001":
                raise ConnectionError("timeout")
            return {
                "storage_unit_id": "SU_002",
                "capacity": 500,
                "active_keys": 100,
                "fields_count": 2,
                "process_rss_bytes": 100 * 1024 * 1024,
            }

        exporter._query_storage_unit = mock_query
        exporter.collect_storage_metrics()

        # SU_002 should still have been collected
        assert exporter.storage_capacity.labels(storage_unit_id="SU_002")._value.get() == 500
        assert call_count == 2

    def test_payload_metrics_collected_and_removed_when_backend_no_longer_reports_them(self):
        exporter = TQMetricsExporter()
        exporter._storage_unit_infos = {"storage": MagicMock()}
        snapshot = {
            "storage_unit_id": "storage",
            "payload_transfer": {"idle_receive_bytes": 128, "queue_wait_seconds": 2.5},
        }
        exporter._query_storage_unit = MagicMock(return_value=snapshot)

        exporter.collect_storage_metrics()
        assert (
            exporter.registry.get_sample_value("tq_storage_payload_idle_receive_bytes", {"storage_unit_id": "storage"})
            == 128
        )
        assert (
            exporter.registry.get_sample_value("tq_storage_payload_queue_wait_seconds", {"storage_unit_id": "storage"})
            == 2.5
        )

        snapshot.pop("payload_transfer")
        exporter.collect_storage_metrics()
        assert (
            exporter.registry.get_sample_value("tq_storage_payload_idle_receive_bytes", {"storage_unit_id": "storage"})
            is None
        )

    def test_storage_metrics_skips_capacity_when_none(self):
        """When capacity is None (unlimited storage), capacity/utilization
        gauges must not be populated, but other metrics still are.

        Regression test guarding against re-introducing ``Gauge.set(None)``,
        which raises ``TypeError`` inside ``float()`` and floods the logs
        with warnings once per collection interval.
        """
        exporter = TQMetricsExporter()

        fake_su_info = MagicMock()
        fake_su_info.id = "SU_UNLIMITED"
        exporter._storage_unit_infos = {"SU_UNLIMITED": fake_su_info}

        exporter._query_storage_unit = MagicMock(
            return_value={
                "storage_unit_id": "SU_UNLIMITED",
                "capacity": None,
                "active_keys": 42,
                "process_rss_bytes": 128 * 1024 * 1024,
            }
        )

        exporter.collect_storage_metrics()

        # capacity / utilization must have no series for this storage unit
        assert ("SU_UNLIMITED",) not in exporter.storage_capacity._metrics
        assert ("SU_UNLIMITED",) not in exporter.storage_utilization._metrics
        # active_keys and memory_rss are still reported
        assert exporter.storage_active_keys.labels(storage_unit_id="SU_UNLIMITED")._value.get() == 42
        assert exporter.storage_memory_rss.labels(storage_unit_id="SU_UNLIMITED")._value.get() == 128 * 1024 * 1024
        assert exporter.storage_ssd_offload_enabled.labels(storage_unit_id="SU_UNLIMITED")._value.get() == 0
        assert exporter.storage_ssd_active_values.labels(storage_unit_id="SU_UNLIMITED")._value.get() == 0
        assert exporter.storage_ssd_active_bytes.labels(storage_unit_id="SU_UNLIMITED")._value.get() == 0
        assert exporter.storage_ssd_fallback_values.labels(storage_unit_id="SU_UNLIMITED")._value.get() == 0

    def test_storage_metrics_prunes_stale_capacity_on_switch_to_unlimited(self):
        """If a storage unit transitions from a numeric capacity to unlimited,
        the previously-set ``storage_capacity`` / ``storage_utilization``
        series must be removed so dashboards do not keep serving stale values.
        """
        exporter = TQMetricsExporter()

        fake_su_info = MagicMock()
        fake_su_info.id = "SU_SWITCH"
        exporter._storage_unit_infos = {"SU_SWITCH": fake_su_info}

        # First collection: numeric capacity → both gauges get populated.
        exporter._query_storage_unit = MagicMock(
            return_value={
                "storage_unit_id": "SU_SWITCH",
                "capacity": 1000,
                "active_keys": 250,
                "process_rss_bytes": 64 * 1024 * 1024,
            }
        )
        exporter.collect_storage_metrics()
        assert exporter.storage_capacity.labels(storage_unit_id="SU_SWITCH")._value.get() == 1000
        assert exporter.storage_utilization.labels(storage_unit_id="SU_SWITCH")._value.get() == 0.25

        # Second collection: unlimited capacity → stale series must be pruned.
        exporter._query_storage_unit = MagicMock(
            return_value={
                "storage_unit_id": "SU_SWITCH",
                "capacity": None,
                "active_keys": 300,
                "process_rss_bytes": 64 * 1024 * 1024,
            }
        )
        exporter.collect_storage_metrics()
        assert ("SU_SWITCH",) not in exporter.storage_capacity._metrics
        assert ("SU_SWITCH",) not in exporter.storage_utilization._metrics
        # active_keys keeps updating with the latest value.
        assert exporter.storage_active_keys.labels(storage_unit_id="SU_SWITCH")._value.get() == 300


# ---------------------------------------------------------------------------
# Test: ZMQ request type registration
# ---------------------------------------------------------------------------


class TestZMQRequestTypes:
    def test_metrics_request_types_exist(self):
        from transfer_queue.utils.zmq_utils import ZMQRequestType

        assert ZMQRequestType.GET_METRICS.value == "GET_METRICS"
        assert ZMQRequestType.METRICS_RESPONSE.value == "METRICS_RESPONSE"


class TestStorageWorkerArrivals:
    @pytest.fixture
    def worker(self):
        from transfer_queue.storage.payload_transfer.zmq import ZmqPayloadTransfer
        from transfer_queue.storage.simple_storage import SimpleStorageUnit, StorageUnitData
        from transfer_queue.utils.perf_utils import IntervalPerfMonitor

        unit_class = SimpleStorageUnit.__ray_metadata__.modified_class
        unit = object.__new__(unit_class)
        unit.storage_unit_id = "unit_a"
        unit.storage_unit_size = 4
        unit.storage_data = StorageUnitData(4)
        unit.payload_transfer = ZmqPayloadTransfer()
        unit._requests_arrived = 0
        unit._arrivals_by_op = {}
        unit._accept_probe = None
        unit._metrics = None
        return unit, IntervalPerfMonitor("unit_a")

    @staticmethod
    def dispatch(worker, operation, body):
        from queue import SimpleQueue

        from transfer_queue.storage.payload_transfer import DeferredResponse
        from transfer_queue.storage.simple_storage import _drain_deferred_responses

        unit, monitor = worker
        request = ZMQMessage.create(request_type=operation, sender_id="manager", body=body)
        socket = MagicMock()
        socket.recv_multipart.return_value = [b"manager"] + request.serialize()
        measurements = {}
        identity, response = unit._process_one_worker_request(socket, monitor, measurements)
        if isinstance(response, DeferredResponse):
            assert response.future.done()
            completions = SimpleQueue()
            completions.put((bytes(identity), response.future))
            _drain_deferred_responses(completions, socket, unit.storage_unit_id, measurements)
        else:
            socket.send_multipart([identity] + response.serialize(), copy=False)
        assert measurements == {}
        sent = socket.send_multipart.call_args.args[0]
        assert sent[0] == b"manager"
        return ZMQMessage.deserialize(sent[1:])

    def test_zmq_metrics_and_checkpoints_keep_their_arrival_labels(self, worker, tmp_path):
        path = str(tmp_path / "storage.pkl")
        requests = [
            (ZMQRequestType.PUT_DATA, {"global_indexes": [0], "data": {"value": [42]}}, "PUT_DATA_RESPONSE"),
            (ZMQRequestType.GET_DATA, {"global_indexes": [0], "fields": ["value"]}, "GET_DATA_RESPONSE"),
            (ZMQRequestType.SAVE_STORAGE_CHECKPOINT, {"path": path}, "SAVE_STORAGE_CHECKPOINT_RESPONSE"),
            (ZMQRequestType.CLEAR_DATA, {"global_indexes": [0]}, "CLEAR_DATA_RESPONSE"),
            (ZMQRequestType.LOAD_STORAGE_CHECKPOINT, {"path": path}, "LOAD_STORAGE_CHECKPOINT_RESPONSE"),
            (ZMQRequestType.GET_METRICS, {}, "METRICS_RESPONSE"),
        ]
        for operation, body, response_type in requests:
            reply = self.dispatch(worker, operation, body)
            assert reply.request_type == ZMQRequestType[response_type]
            if "CHECKPOINT" in operation.name:
                assert reply.body["success"]
        unit, monitor = worker
        expected = {operation.name: 1 for operation, _, _ in requests}
        assert reply.body["requests_arrived"] == len(requests)
        assert reply.body["arrivals_by_op"] == expected
        assert dict(monitor.success_counts) == {"PUT_DATA": 1, "GET_DATA": 1, "CLEAR_DATA": 1}
        assert unit.storage_data.get_data(["value"], [0]) == {"value": [42]}

        exporter = TQMetricsExporter()
        exporter.register_storage_units({"unit_a": MagicMock()})
        exporter._query_storage_unit = MagicMock(return_value=reply.body)
        exporter.collect_storage_metrics()
        for operation in expected:
            assert exporter.storage_arrivals_by_op.labels(storage_unit_id="unit_a", op_type=operation)._value.get() == 1

    @pytest.mark.parametrize("operation", ["put", "get"])
    @pytest.mark.parametrize("finish", ["COMMIT", "CANCEL"])
    def test_nixl_counts_prepare_once_and_skips_continuations(self, worker, operation, finish):
        from transfer_queue.storage.payload_transfer.nixl import NixlPayloadTransfer, PayloadDescriptor
        from transfer_queue.utils.serial_utils import encode

        unit, monitor = worker
        unit.storage_data.put_data({"value": [42]}, [0])
        transfer = object.__new__(NixlPayloadTransfer)
        transfer._pending_puts = {}
        transfer._pending_gets = {}
        transfer._runtime = MagicMock()
        unit.payload_transfer = transfer
        frames = encode({"value": [42]})
        frame_sizes = tuple(memoryview(frame).nbytes for frame in frames)
        descriptor = PayloadDescriptor("payload", sum(frame_sizes), frame_sizes)
        transfer._runtime.receive.return_value = tuple(memoryview(frame) for frame in frames)
        sent = Future()
        sent.set_result(None)
        transfer._runtime.send.return_value = sent
        transfer._runtime.prepare_receive.return_value = {}
        body = {
            "global_indexes": [0],
            "fields": ["value"],
            "transfer_id": "payload",
            "descriptor": descriptor.to_dict(),
        }

        ready = self.dispatch(worker, ZMQRequestType[f"{operation.upper()}_DATA_PREPARE"], body)
        assert ready.request_type == ZMQRequestType[f"{operation.upper()}_DATA_READY"]
        assert unit._requests_arrived == 1
        assert dict(monitor.success_counts) == {}

        reply = self.dispatch(
            worker,
            ZMQRequestType[f"{operation.upper()}_DATA_{finish}"],
            {
                "transfer_id": "payload",
                "receiver_endpoint": {"transport": "nixl-ucx", "data": {"agent_name": "client"}},
                "receive_token": {
                    "data": {"agent_name": "client", "agent_metadata": b"meta", "frame_remote_descs": b"desc"}
                },
            },
        )
        assert reply.request_type == ZMQRequestType[f"{operation.upper()}_DATA_RESPONSE"]
        assert unit._requests_arrived == 1
        assert unit._arrivals_by_op == {f"{operation.upper()}_DATA": 1}
        assert dict(monitor.success_counts) == ({f"{operation.upper()}_DATA": 1} if finish == "COMMIT" else {})

    def test_nixl_prepare_key_miss_is_an_arrival_without_measured_processing(self, worker):
        from transfer_queue.storage.payload_transfer.nixl import NixlPayloadTransfer
        from transfer_queue.storage.simple_storage import KEY_NOT_FOUND_MARKER

        unit, monitor = worker
        unit.storage_data.put_data({"value": [42]}, [0])
        unit.storage_data.clear([0])
        transfer = object.__new__(NixlPayloadTransfer)
        transfer._pending_gets = {}
        unit.payload_transfer = transfer

        reply = self.dispatch(
            worker,
            ZMQRequestType.GET_DATA_PREPARE,
            {"transfer_id": "missing", "global_indexes": [0], "fields": ["value"]},
        )
        assert reply.request_type == ZMQRequestType.PUT_GET_ERROR
        assert KEY_NOT_FOUND_MARKER in reply.body["message"]
        assert unit._requests_arrived == 1
        assert unit._arrivals_by_op == {"GET_DATA": 1}
        assert dict(monitor.success_counts) == {}
