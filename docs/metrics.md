
# Prometheus Metrics & Grafana Dashboard

> Last updated: 05/08/2026

## Overview

TransferQueue provides built-in Prometheus metrics exporting for both the **Controller** and **SimpleStorageUnit** processes. When enabled, each process exposes an HTTP `/metrics` endpoint that can be scraped by Prometheus, and a pre-built Grafana dashboard is provided for visualization.

## Quick Start

### 1. Enable Metrics in Config

```yaml
metrics:
  enabled: true
  port: 0  # 0 = auto-assign free port; set a fixed port for production
```

Or pass via `init()`:

```python
import transfer_queue as tq

tq.init({
    "metrics": {
        "enabled": True,
        "port": 9090,
    }
})
```

### 2. Discover the Endpoint

```python
endpoint = tq.get_metrics_endpoint()
print(f"http://{endpoint}/metrics")
```

### 3. Import Grafana Dashboard

Import the pre-built dashboard JSON into your Grafana instance:

**[`scripts/grafana_dashboard.json`](../scripts/grafana_dashboard.json)**

Steps:
1. Open Grafana → Dashboards → Import
2. Upload the JSON file or paste its content
3. Select your Prometheus datasource
4. Done

## Configuration

| Config Key | Default | Description |
|------------|---------|-------------|
| `metrics.enabled` | `false` | Enable/disable the metrics exporter |
| `metrics.port` | `0` | HTTP port for `/metrics` endpoint (0 = OS auto-assign) |

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TQ_METRICS_COLLECT_INTERVAL` | `10` | Background collection interval (seconds) |
| `TQ_METRICS_STORAGE_TIMEOUT` | `5` | ZMQ timeout for storage unit queries (seconds) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Controller Process                                             │
│                                                                 │
│  TransferQueueController                                        │
│       │                                                         │
│       │── snapshot push (every 10s) ──▶ TQMetricsExporter       │
│       │                                  (role="controller")    │
│       │                                    │                    │
│       │                                    ├─ HTTP /metrics ◀── Prometheus
│       │                                    │                    │
│       │                                    └─ ZMQ GET_METRICS   │
│       │                                         │               │
└───────┼─────────────────────────────────────────┼───────────────┘
        │                                         │
        ▼                                         ▼
┌───────────────────┐                   ┌───────────────────┐
│ SimpleStorageUnit │                   │ SimpleStorageUnit │
│                   │                   │                   │
│ TQMetricsExporter │                   │ TQMetricsExporter │
│ (role="storage")  │                   │ (role="storage")  │
│   HTTP /metrics ◀─┼── Prometheus      │   HTTP /metrics   │
└───────────────────┘                   └───────────────────┘
```

- **Controller** (`role="controller"`) pushes plain-dict snapshots to its exporter (no lock contention). Its exporter also queries storage units via ZMQ for capacity/utilization and per-operation request stats.
- **Storage Units** (`role="storage"`) each run their own exporter with native Histogram/Counter metrics for request latency/throughput (PUT_DATA, GET_DATA, CLEAR_DATA).
- **Two scrape paths**: If Prometheus scrapes only the controller endpoint, storage request metrics are available via ZMQ-collected gauges. If Prometheus scrapes each storage unit directly, native histogram data provides more precise quantiles.
- Metrics are **role-prefixed**: controller uses `tq_controller_request_*`, storage uses `tq_storage_request_*` — no naming conflicts.

## Metrics Reference

### Controller Process Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tq_controller_uptime_seconds` | Gauge | — | Controller process uptime |
| `tq_controller_memory_rss_bytes` | Gauge | — | Controller RSS memory |

### Partition Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tq_partitions_total` | Gauge | — | Number of active partitions |
| `tq_partition_samples_total` | Gauge | `partition_id` | Samples per partition |
| `tq_partition_production_progress` | Gauge | `partition_id`, `task_name` | Production progress (0.0–1.0) |
| `tq_partition_consumption_progress` | Gauge | `partition_id`, `task_name` | Consumption progress (0.0–1.0) |

### Index Manager Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tq_global_index_allocated_total` | Gauge | — | Total allocated global indexes |
| `tq_global_index_reusable_total` | Gauge | — | Reusable global indexes |

### Request Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tq_controller_request_total` | Counter | `op_type` | Total requests processed |
| `tq_controller_request_duration_seconds` | Histogram | `op_type` | Request latency (buckets: 1ms–5s) |
| `tq_controller_request_errors_total` | Counter | `op_type` | Total request errors |
| `tq_controller_request_samples_total` | Counter | `op_type` | Total samples processed per operation (for batch-aware accounting) |

### Storage Unit Metrics (collected via ZMQ, exposed on controller)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tq_storage_capacity_total` | Gauge | `storage_unit_id` | Max storage capacity |
| `tq_storage_active_keys_total` | Gauge | `storage_unit_id` | Active keys in storage |
| `tq_storage_utilization_ratio` | Gauge | `storage_unit_id` | Utilization (active/capacity) |
| `tq_storage_memory_rss_bytes` | Gauge | `storage_unit_id` | Storage process RSS memory |
| `tq_storage_request_ops` | Gauge | `storage_unit_id`, `op_type` | Total requests processed by storage unit |
| `tq_storage_request_latency_avg` | Gauge | `storage_unit_id`, `op_type` | Average request latency (seconds) |
| `tq_storage_request_latency_p50` | Gauge | `storage_unit_id`, `op_type` | P50 request latency (seconds) |
| `tq_storage_request_latency_p99` | Gauge | `storage_unit_id`, `op_type` | P99 request latency (seconds) |

### Storage Unit Native Metrics (exposed on each storage unit's own endpoint)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `tq_storage_request_duration_seconds` | Histogram | `op_type` | Request latency (buckets: 1ms–5s) |
| `tq_storage_request_total` | Counter | `op_type` | Total requests processed |
| `tq_storage_request_errors_total` | Counter | `op_type` | Total request errors |
| `tq_storage_request_samples_total` | Counter | `op_type` | Total samples processed per operation |

> **Note on naming**: The ZMQ-collected gauges on the controller avoid all Prometheus reserved suffixes (`_total`, `_bucket`, `_sum`, `_count`, `_info`, `_created`) and the reserved `le` label to prevent type metadata conflicts that break `label_values()` queries. P50/P99 are computed on the storage unit side and sent as pre-calculated values. The storage unit's own endpoint uses standard Counter/Histogram naming conventions.

## Grafana Dashboard

The dashboard ([`scripts/grafana_dashboard.json`](../scripts/grafana_dashboard.json)) includes:

### Panels

| Section | Panels |
|---------|--------|
| **Controller Overview** | Uptime, RSS Memory, Active Partitions, Indexes Allocated, Reusable Indexes |
| **Request Throughput & Latency** | Controller Request Rate (ops/s), Controller Request Latency (repeats per quantile) |
| **Partition Status** | Samples per Partition, Production Progress, Consumption Progress |
| **Storage Units** | Utilization Bar Gauge, Active Keys, Capacity vs Active Keys, RSS Memory, Storage Request Rate, Storage Request Latency (repeats per quantile), Produced vs Cleared Samples/s, Active Keys Delta |

### Template Variables

| Variable | Type | Description |
|----------|------|-------------|
| `datasource` | Datasource | Prometheus datasource selector |
| `task_name` | Query | Filter Production/Consumption Progress panels by task |
| `op_type` | Custom | Filter request panels by operation (PUT_DATA, GET_DATA, CLEAR_DATA, etc.) |
| `quantile` | Custom | Filter latency panels by quantile (p50, p99) |

### Thresholds

- **Storage Utilization**: Green < 70%, Yellow 70–90%, Red > 90%
- **Controller RSS Memory**: Green < 2GB, Yellow 2–4GB, Red > 4GB

## Detecting Leaks: Produced vs Cleared

A common concern is whether consumed samples are being properly cleared from storage. The dashboard provides two panels for this:

### Produced vs Cleared Samples (per second)

Compares the **actual sample count** (not request count) between production and consumption:

- `rate(tq_controller_request_samples_total{op_type="NOTIFY_DATA_UPDATE"})` — samples produced/s
- `rate(tq_controller_request_samples_total{op_type="CLEAR_META"})` — samples cleared/s

> **Why sample count, not request rate?** A single `CLEAR_META` request can batch-clear hundreds of samples. Comparing request rates would be misleading.

| Observation | Meaning |
|-------------|---------|
| Two lines track closely | Production/consumption balanced, no leak |
| Produced consistently > Cleared | Samples accumulating — potential leak |
| Cleared spikes after Produced plateau | Batch consumer pattern (normal) |

### Active Keys Delta

Shows `sum(tq_storage_active_keys_total)` over time:

| Observation | Meaning |
|-------------|---------|
| Stable or oscillating | Healthy steady-state |
| Monotonically increasing | Leak — keys are never freed |
| Approaching capacity | Imminent storage exhaustion |

### Quick Troubleshooting

1. **Active Keys rising?** → Check "Produced vs Cleared Samples" — is CLEAR keeping up?
2. **CLEAR rate is zero?** → Consumer is not calling `clear_samples()` / `clear_partition()`
3. **CLEAR rate > 0 but keys still rising?** → Check Consumption Progress — is the consumer actually finishing before clearing?

## Integration with `IntervalPerfMonitor`

When metrics are **disabled** (default), both the Controller and SimpleStorageUnit use `IntervalPerfMonitor` — a lightweight logger-based fallback that prints aggregated stats every 5 minutes.

When metrics are **enabled**, `TQMetricsExporter` replaces the perf monitor transparently (same `measure(op_type=...)` interface), providing Prometheus-native counters and histograms instead of log-based summaries.
