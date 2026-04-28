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

"""Unit tests for the Prometheus metrics exporter (transfer_queue.metrics)."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:
    from transfer_queue.metrics import TQMetricsExporter

    _HAS_DEPS = True
except (ImportError, OSError):
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason="prometheus_client / psutil / pyzmq dependencies unavailable")


# ---------------------------------------------------------------------------
# Helpers — lightweight fakes that satisfy TQMetricsExporter without Ray
# ---------------------------------------------------------------------------


def _make_fake_partition(
    partition_id: str,
    total_samples: int = 10,
    total_fields: int = 2,
    allocated_samples: int = 16,
    produced_ratio: float = 0.5,
    consumption: dict | None = None,
):
    """Return an object that mimics ``DataPartitionStatus.get_statistics()``."""
    consumption_stats = {}
    if consumption:
        for task, progress in consumption.items():
            consumption_stats[task] = {
                "consumed_samples": int(total_samples * progress),
                "consumption_progress": progress,
            }

    stats = {
        "partition_id": partition_id,
        "created_at": time.time(),
        "total_samples_num": total_samples,
        "total_fields_num": total_fields,
        "allocated_samples_num": allocated_samples,
        "allocated_fields_num": total_fields,
        "registered_tasks": list((consumption or {}).keys()),
        "produced_samples": int(total_samples * produced_ratio),
        "production_progress": produced_ratio,
        "field_statistics": {},
        "consumption_statistics": consumption_stats,
    }

    partition = MagicMock()
    partition.get_statistics.return_value = stats
    return partition


def _make_fake_controller(partitions=None, allocated=10, reusable=2):
    """Return a lightweight stand-in for ``TransferQueueController``."""
    ctrl = SimpleNamespace()
    ctrl.partitions = partitions or {}
    ctrl.index_manager = SimpleNamespace()
    ctrl.index_manager.allocated_indexes = set(range(allocated))
    ctrl.index_manager.reusable_indexes = list(range(reusable))
    return ctrl


# ---------------------------------------------------------------------------
# Test: metric definitions
# ---------------------------------------------------------------------------


class TestMetricDefinitions:
    def test_all_metrics_are_registered(self):
        """Verify that all expected metric families exist in the exporter's registry."""
        ctrl = _make_fake_controller()
        exporter = TQMetricsExporter(ctrl)

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
            "tq_controller_request_total",
            "tq_controller_request_errors_total",
            "tq_storage_capacity_total",
            "tq_storage_active_keys_total",
            "tq_storage_utilization_ratio",
            "tq_storage_memory_rss_bytes",
        ]

        registered = {m.name for m in exporter.registry.collect()}
        for prefix in expected_prefixes:
            assert prefix in registered, f"Metric '{prefix}' not found in registry"


# ---------------------------------------------------------------------------
# Test: controller metrics collection
# ---------------------------------------------------------------------------


class TestControllerMetricsCollection:
    def test_collect_empty_controller(self):
        """Collect metrics from a controller with no partitions — should not raise."""
        ctrl = _make_fake_controller(partitions={}, allocated=0, reusable=0)
        exporter = TQMetricsExporter(ctrl)
        exporter.collect_controller_metrics()

        assert exporter.partitions_total._value.get() == 0
        assert exporter.global_index_allocated._value.get() == 0
        assert exporter.global_index_reusable._value.get() == 0

    def test_collect_with_partitions(self):
        """Partition-level metrics are populated correctly."""
        p1 = _make_fake_partition(
            "train_0", total_samples=20, produced_ratio=0.8, consumption={"gen": 0.5}
        )
        p2 = _make_fake_partition(
            "train_1", total_samples=10, produced_ratio=1.0, consumption={"gen": 1.0, "train": 0.3}
        )
        ctrl = _make_fake_controller(partitions={"train_0": p1, "train_1": p2}, allocated=30, reusable=5)

        exporter = TQMetricsExporter(ctrl)
        exporter.collect_controller_metrics()

        assert exporter.partitions_total._value.get() == 2
        assert exporter.global_index_allocated._value.get() == 30
        assert exporter.global_index_reusable._value.get() == 5

        # Check partition-level gauges
        assert exporter.partition_samples.labels(partition_id="train_0")._value.get() == 20
        assert exporter.partition_production_progress.labels(partition_id="train_0")._value.get() == 0.8
        assert exporter.partition_consumption_progress.labels(
            partition_id="train_0", task_name="gen"
        )._value.get() == 0.5

        assert exporter.partition_samples.labels(partition_id="train_1")._value.get() == 10
        assert exporter.partition_consumption_progress.labels(
            partition_id="train_1", task_name="train"
        )._value.get() == 0.3

    def test_uptime_increases(self):
        """Controller uptime should be positive after collection."""
        ctrl = _make_fake_controller()
        exporter = TQMetricsExporter(ctrl)
        time.sleep(0.05)
        exporter.collect_controller_metrics()
        assert exporter.controller_uptime._value.get() > 0


# ---------------------------------------------------------------------------
# Test: measure() context manager
# ---------------------------------------------------------------------------


class TestMeasureContextManager:
    def test_measure_records_count_and_duration(self):
        ctrl = _make_fake_controller()
        exporter = TQMetricsExporter(ctrl)

        with exporter.measure("GET_META"):
            time.sleep(0.01)

        # Counter should have been incremented
        assert exporter.request_total.labels(op_type="GET_META")._value.get() == 1.0

        # Histogram should have at least one observation
        hist = exporter.request_duration.labels(op_type="GET_META")
        # _sum is the sum of observed values
        assert hist._sum.get() > 0

    def test_measure_records_errors(self):
        ctrl = _make_fake_controller()
        exporter = TQMetricsExporter(ctrl)

        with pytest.raises(ValueError):
            with exporter.measure("BAD_OP"):
                raise ValueError("boom")

        assert exporter.request_errors_total.labels(op_type="BAD_OP")._value.get() == 1.0
        # The total counter should also be incremented (inc happens before yield)
        assert exporter.request_total.labels(op_type="BAD_OP")._value.get() == 1.0

    def test_multiple_ops_tracked_independently(self):
        ctrl = _make_fake_controller()
        exporter = TQMetricsExporter(ctrl)

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


class TestStorageMetricsCollection:
    def test_collect_with_no_storage_units(self):
        """No storage units registered — collect should be a no-op."""
        ctrl = _make_fake_controller()
        exporter = TQMetricsExporter(ctrl)
        # Should not raise
        exporter.collect_storage_metrics()

    def test_storage_metrics_populated_on_success(self):
        """Verify storage gauges are set when _query_storage_unit returns data."""
        ctrl = _make_fake_controller()
        exporter = TQMetricsExporter(ctrl)

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
            }
        )

        exporter.collect_storage_metrics()

        assert exporter.storage_capacity.labels(storage_unit_id="SU_001")._value.get() == 1000
        assert exporter.storage_active_keys.labels(storage_unit_id="SU_001")._value.get() == 250
        assert exporter.storage_utilization.labels(storage_unit_id="SU_001")._value.get() == 0.25
        assert exporter.storage_memory_rss.labels(storage_unit_id="SU_001")._value.get() == 512 * 1024 * 1024

    def test_storage_metrics_handles_query_failure(self):
        """If a storage unit query fails, other units should still be collected."""
        ctrl = _make_fake_controller()
        exporter = TQMetricsExporter(ctrl)

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


# ---------------------------------------------------------------------------
# Test: ZMQ request type registration
# ---------------------------------------------------------------------------


class TestZMQRequestTypes:
    def test_metrics_request_types_exist(self):
        from transfer_queue.utils.zmq_utils import ZMQRequestType

        assert ZMQRequestType.GET_METRICS.value == "GET_METRICS"
        assert ZMQRequestType.METRICS_RESPONSE.value == "METRICS_RESPONSE"
