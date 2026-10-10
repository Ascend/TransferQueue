#!/usr/bin/env python3
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

"""Benchmark mixed in-memory and SSD-backed SimpleStorage samples."""

import argparse
import csv
import json
import logging
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ray
import torch
from omegaconf import DictConfig, OmegaConf
from prometheus_client.parser import text_string_to_metric_families
from tensordict import TensorDict

import transfer_queue as tq

LOGGER = logging.getLogger("ssd_offload_benchmark")
DEFAULT_SMALL_SAMPLE_BYTES = 512 * 1024
DEFAULT_LARGE_SAMPLE_BYTES = 2 * 1024 * 1024
BYTES_PER_FLOAT32 = 4
RAY_ADDRESS = "auto"
METRICS_TIMEOUT_SECONDS = 180
RSS_SAMPLE_INTERVAL_SECONDS = 11
RSS_STABLE_SAMPLES = 3
RSS_STABILITY_TOLERANCE_BYTES = 32 * 2**20
MODE_GAP_SECONDS = 10


@dataclass(frozen=True)
class Workload:
    batch_size: int
    small_fields: int
    large_fields: int
    small_sample_bytes: int
    large_sample_bytes: int
    offload_threshold_bytes: int

    def validate(self) -> None:
        sizes = {
            "batch_size": self.batch_size,
            "small_sample_bytes": self.small_sample_bytes,
            "large_sample_bytes": self.large_sample_bytes,
            "offload_threshold_bytes": self.offload_threshold_bytes,
        }
        if any(value <= 0 for value in sizes.values()):
            raise ValueError(f"Workload sizes must be positive: {sizes}")
        if self.small_fields < 0 or self.large_fields < 0:
            raise ValueError("Field counts must be non-negative")
        if self.small_fields + self.large_fields == 0:
            raise ValueError("At least one tensor field is required")
        if self.small_sample_bytes % BYTES_PER_FLOAT32 or self.large_sample_bytes % BYTES_PER_FLOAT32:
            raise ValueError("Sample sizes must be divisible by four for float32 tensors")
        if self.small_sample_bytes >= self.offload_threshold_bytes:
            raise ValueError("small_sample_bytes must be below the SSD offload threshold")
        if self.large_sample_bytes <= self.offload_threshold_bytes:
            raise ValueError("large_sample_bytes must be above the SSD offload threshold")

    @property
    def inline_bytes(self) -> int:
        return self.batch_size * self.small_fields * self.small_sample_bytes

    @property
    def offloaded_bytes(self) -> int:
        return self.batch_size * self.large_fields * self.large_sample_bytes

    @property
    def total_bytes(self) -> int:
        return self.inline_bytes + self.offloaded_bytes


@dataclass(frozen=True)
class StorageMetrics:
    active_keys: int
    rss_bytes: int
    ssd_active_bytes: int


def create_mixed_tensors(workload: Workload) -> TensorDict:
    """Create fields whose individual samples straddle the offload threshold."""
    workload.validate()
    torch.manual_seed(0)
    fields = {
        f"small_{index}": torch.randn(
            workload.batch_size,
            workload.small_sample_bytes // BYTES_PER_FLOAT32,
            dtype=torch.float32,
        )
        for index in range(workload.small_fields)
    }
    fields.update(
        {
            f"large_{index}": torch.randn(
                workload.batch_size,
                workload.large_sample_bytes // BYTES_PER_FLOAT32,
                dtype=torch.float32,
            )
            for index in range(workload.large_fields)
        }
    )
    return TensorDict(fields, batch_size=[workload.batch_size])


def read_storage_metrics(endpoint: str, expected_storage_units: int) -> StorageMetrics:
    """Read one aggregate snapshot from the controller's Prometheus endpoint."""
    with urllib.request.urlopen(f"http://{endpoint}/metrics", timeout=5) as response:
        payload = response.read().decode("utf-8")

    metric_names = {
        "tq_storage_active_keys_total": "active_keys",
        "tq_storage_memory_rss_bytes": "rss_bytes",
        "tq_storage_ssd_active_bytes": "ssd_active_bytes",
    }
    units: dict[str, dict[str, int]] = {}
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            field = metric_names.get(sample.name)
            storage_unit_id = sample.labels.get("storage_unit_id")
            if field is not None and storage_unit_id is not None:
                units.setdefault(storage_unit_id, {})[field] = int(sample.value)

    complete = len(units) == expected_storage_units and all(
        set(values) >= {"active_keys", "rss_bytes", "ssd_active_bytes"} for values in units.values()
    )
    if not complete:
        fields = sorted({tuple(sorted(values)) for values in units.values()})
        raise RuntimeError(f"Incomplete storage metrics: units={len(units)}/{expected_storage_units}, fields={fields}")
    return StorageMetrics(
        active_keys=sum(values["active_keys"] for values in units.values()),
        rss_bytes=sum(values["rss_bytes"] for values in units.values()),
        ssd_active_bytes=sum(values["ssd_active_bytes"] for values in units.values()),
    )


def wait_for_storage_state(
    endpoint: str,
    expected_storage_units: int,
    expected_active_keys: int,
    timeout_seconds: float,
) -> StorageMetrics:
    """Wait until a fresh metrics collection reports the requested key count."""
    deadline = time.monotonic() + timeout_seconds
    last_metrics = None
    last_error = None
    while time.monotonic() < deadline:
        try:
            last_metrics = read_storage_metrics(endpoint, expected_storage_units)
            last_error = None
            if last_metrics.active_keys == expected_active_keys:
                return last_metrics
        except (OSError, RuntimeError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise TimeoutError(
        f"Storage metrics did not reach active_keys={expected_active_keys}: "
        f"last_metrics={last_metrics}, last_error={last_error!r}"
    )


def wait_for_stable_rss(
    endpoint: str,
    expected_storage_units: int,
    expected_active_keys: int,
    timeout_seconds: float,
    sample_interval_seconds: float,
    stable_samples: int,
    tolerance_bytes: int,
) -> StorageMetrics:
    """Return a median RSS after consecutive full metrics collection periods agree."""
    deadline = time.monotonic() + timeout_seconds
    samples: list[StorageMetrics] = []
    while time.monotonic() < deadline:
        remaining = max(0.5, deadline - time.monotonic())
        sample = wait_for_storage_state(
            endpoint,
            expected_storage_units,
            expected_active_keys,
            remaining,
        )
        samples.append(sample)
        window = samples[-stable_samples:]
        if len(window) == stable_samples:
            rss_values = [item.rss_bytes for item in window]
            if max(rss_values) - min(rss_values) <= tolerance_bytes:
                return StorageMetrics(
                    active_keys=expected_active_keys,
                    rss_bytes=int(statistics.median(rss_values)),
                    ssd_active_bytes=int(statistics.median(item.ssd_active_bytes for item in window)),
                )
        time.sleep(sample_interval_seconds)
    rss_values = [sample.rss_bytes for sample in samples[-stable_samples:]]
    raise TimeoutError(
        f"Storage RSS did not stabilize within {timeout_seconds}s: "
        f"last_rss_values={rss_values}, tolerance_bytes={tolerance_bytes}"
    )


@ray.remote
class BenchmarkClient:
    """Own one public TransferQueue client and, for the writer, the test payload."""

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._data: TensorDict | None = None
        self._keys: list[str] = []

    def initialize(self) -> None:
        tq.init(OmegaConf.create(self._config))

    def create_data(self, workload: Workload) -> None:
        self._data = create_mixed_tensors(workload)
        self._keys = [f"mixed_{index}" for index in range(workload.batch_size)]

    def put(self, partition_id: str) -> None:
        if self._data is None:
            raise RuntimeError("Benchmark data has not been created")
        tq.kv_batch_put(keys=self._keys, partition_id=partition_id, fields=self._data)

    def list_keys(self, partition_id: str) -> list[str]:
        partitions = tq.kv_list(partition_id=partition_id)
        return list(partitions.get(partition_id, {}))

    def get(self, partition_id: str, keys: list[str]) -> None:
        tq.kv_batch_get(keys=keys, partition_id=partition_id)

    def clear(self, partition_id: str, keys: list[str]) -> None:
        tq.kv_clear(keys=keys, partition_id=partition_id)

    def metrics_endpoint(self) -> str | None:
        return tq.get_metrics_endpoint()

    def close(self) -> None:
        tq.close()


def build_config(
    base_config: DictConfig,
    workload: Workload,
    ssd_enabled: bool,
    ssd_path: str,
) -> dict[str, Any]:
    """Specialize the benchmark config for one SimpleStorage mode."""
    config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    config.metrics.enabled = True
    config.metrics.port = 0
    config.backend.storage_backend = "SimpleStorage"
    config.backend.SimpleStorage.total_storage_size = workload.batch_size
    config.backend.SimpleStorage.required_node_resource = None
    config.backend.SimpleStorage.ssd_offload.enabled = ssd_enabled
    config.backend.SimpleStorage.ssd_offload.path = ssd_path if ssd_enabled else None
    return OmegaConf.to_container(config, resolve=True)


def validate_cluster(head_node_ip: str, worker_node_ip: str) -> None:
    alive_addresses = {node["NodeManagerAddress"] for node in ray.nodes() if node["Alive"]}
    expected = {head_node_ip, worker_node_ip}
    if alive_addresses != expected:
        raise RuntimeError(f"Expected exactly two Ray nodes {sorted(expected)}, got {sorted(alive_addresses)}")


class BenchmarkRun:
    def __init__(
        self,
        mode: str,
        base_config: DictConfig,
        workload: Workload,
        head_node_ip: str,
        worker_node_ip: str,
        ssd_path: str,
    ) -> None:
        self.mode = mode
        self.workload = workload

        config = build_config(base_config, workload, mode == "ssd", ssd_path)
        self.expected_storage_units = int(config["backend"]["SimpleStorage"]["num_data_storage_units"])
        self.writer = BenchmarkClient.options(
            num_cpus=0.001,
            resources={f"node:{head_node_ip}": 0.001},
        ).remote(config)
        self.reader = BenchmarkClient.options(
            num_cpus=0.001,
            resources={f"node:{worker_node_ip}": 0.001},
        ).remote(config)
        ray.get(self.writer.initialize.remote())
        ray.get(self.reader.initialize.remote())
        ray.get(self.writer.create_data.remote(workload))
        self.endpoint = ray.get(self.writer.metrics_endpoint.remote())
        if not self.endpoint:
            raise RuntimeError("SimpleStorage metrics endpoint is unavailable")

    def stable_metrics(self, active_keys: int) -> StorageMetrics:
        return wait_for_stable_rss(
            self.endpoint,
            self.expected_storage_units,
            active_keys,
            METRICS_TIMEOUT_SECONDS,
            RSS_SAMPLE_INTERVAL_SECONDS,
            RSS_STABLE_SAMPLES,
            RSS_STABILITY_TOLERANCE_BYTES,
        )

    def run(self, iterations: int) -> list[dict[str, Any]]:
        partition_id = "mixed"
        baseline = self.stable_metrics(active_keys=0)
        rows = []

        for iteration in range(1, iterations + 1):
            LOGGER.info("%s iteration %d/%d", self.mode, iteration, iterations)
            put_start = time.perf_counter()
            ray.get(self.writer.put.remote(partition_id))
            put_seconds = time.perf_counter() - put_start
            after_put = self.stable_metrics(active_keys=self.workload.batch_size)

            keys = ray.get(self.reader.list_keys.remote(partition_id))
            if len(keys) != self.workload.batch_size:
                raise AssertionError(f"Listed {len(keys)} keys, expected {self.workload.batch_size}")
            get_start = time.perf_counter()
            ray.get(self.reader.get.remote(partition_id, keys))
            get_seconds = time.perf_counter() - get_start
            time.sleep(RSS_SAMPLE_INTERVAL_SECONDS)
            after_get = self.stable_metrics(active_keys=self.workload.batch_size)

            ray.get(self.writer.clear.remote(partition_id, keys))
            after_clear = self.stable_metrics(active_keys=0)

            retained_bytes = after_clear.rss_bytes - baseline.rss_bytes
            row = {
                "mode": self.mode,
                "iteration": iteration,
                **asdict(self.workload),
                "inline_payload_bytes": self.workload.inline_bytes,
                "offload_candidate_bytes": self.workload.offloaded_bytes,
                "total_payload_bytes": self.workload.total_bytes,
                "put_seconds": put_seconds,
                "get_seconds": get_seconds,
                "put_gbit_per_second": self.workload.total_bytes * 8 / put_seconds / 1e9,
                "get_gbit_per_second": self.workload.total_bytes * 8 / get_seconds / 1e9,
                "storage_rss_before_put_bytes": baseline.rss_bytes,
                "storage_rss_after_put_bytes": after_put.rss_bytes,
                "storage_rss_after_get_bytes": after_get.rss_bytes,
                "storage_rss_after_clear_bytes": after_clear.rss_bytes,
                "storage_rss_retained_after_clear_bytes": retained_bytes,
                "storage_ssd_active_bytes_after_put": after_put.ssd_active_bytes,
                "storage_ssd_active_bytes_after_clear": after_clear.ssd_active_bytes,
            }
            rows.append(row)
            LOGGER.info(
                "%s RSS before/put/get/clear: %.3f/%.3f/%.3f/%.3f GiB; retained: %.1f MiB",
                self.mode,
                baseline.rss_bytes / 2**30,
                after_put.rss_bytes / 2**30,
                after_get.rss_bytes / 2**30,
                after_clear.rss_bytes / 2**30,
                retained_bytes / 2**20,
            )
            baseline = after_clear
        return rows

    def close(self) -> None:
        try:
            ray.get([self.writer.close.remote(), self.reader.close.remote()])
        finally:
            ray.kill(self.writer, no_restart=True)
            ray.kill(self.reader, no_restart=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    ssd_rows: list[dict[str, Any]],
    workload: Workload,
    warmup_iterations: int,
) -> dict[str, Any]:
    ssd_measured = ssd_rows[warmup_iterations:]
    retained_values = [row["storage_rss_retained_after_clear_bytes"] for row in ssd_measured]
    clear_rss_values = [row["storage_rss_after_clear_bytes"] for row in ssd_measured]
    return {
        "workload": asdict(workload),
        "analyzed_iterations": len(ssd_measured),
        "theoretical_offload_bytes": workload.offloaded_bytes,
        "ssd_median_active_bytes_after_put": int(
            statistics.median(row["storage_ssd_active_bytes_after_put"] for row in ssd_measured)
        ),
        "ssd_max_active_bytes_after_clear": max(row["storage_ssd_active_bytes_after_clear"] for row in ssd_measured),
        "ssd_median_rss_retained_after_clear_bytes": int(statistics.median(retained_values)),
        "ssd_max_rss_retained_after_clear_bytes": max(retained_values),
        "ssd_clear_rss_growth_bytes": clear_rss_values[-1] - clear_rss_values[0],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend_config", type=Path, required=True)
    parser.add_argument("--head_node_ip", required=True)
    parser.add_argument("--worker_node_ip", required=True)
    parser.add_argument("--ssd_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--num_test_iterations", type=int, default=4)
    parser.add_argument("--warmup_iterations", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=512)
    parser.add_argument("--small_fields", type=int, default=4)
    parser.add_argument("--large_fields", type=int, default=4)
    parser.add_argument("--small_sample_bytes", type=int, default=DEFAULT_SMALL_SAMPLE_BYTES)
    parser.add_argument("--large_sample_bytes", type=int, default=DEFAULT_LARGE_SAMPLE_BYTES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_config = OmegaConf.load(args.backend_config)
    workload = Workload(
        batch_size=args.global_batch_size,
        small_fields=args.small_fields,
        large_fields=args.large_fields,
        small_sample_bytes=args.small_sample_bytes,
        large_sample_bytes=args.large_sample_bytes,
        offload_threshold_bytes=int(base_config.backend.SimpleStorage.ssd_offload.threshold_bytes),
    )
    workload.validate()
    if args.num_test_iterations <= args.warmup_iterations:
        raise ValueError("num_test_iterations must be greater than warmup_iterations")
    if not args.ssd_path.is_dir():
        raise ValueError(f"SSD path does not exist on the benchmark driver node: {args.ssd_path}")

    ray.init(address=RAY_ADDRESS)
    validate_cluster(args.head_node_ip, args.worker_node_ip)
    results: dict[str, list[dict[str, Any]]] = {}
    try:
        for mode in ("memory", "ssd"):
            benchmark = BenchmarkRun(
                mode=mode,
                base_config=base_config,
                workload=workload,
                head_node_ip=args.head_node_ip,
                worker_node_ip=args.worker_node_ip,
                ssd_path=str(args.ssd_path),
            )
            try:
                results[mode] = benchmark.run(args.num_test_iterations)
            finally:
                benchmark.close()
            write_csv(args.output_dir / f"mixed_{mode}.csv", results[mode])
            time.sleep(MODE_GAP_SECONDS)

        summary = summarize(
            results["ssd"],
            workload,
            args.warmup_iterations,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        LOGGER.info("Mixed SSD offload summary:\n%s", json.dumps(summary, indent=2))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
