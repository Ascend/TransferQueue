# TransferQueue SimpleStorage SSD Offload Performance Benchmark

This benchmark is separate from the general backend throughput benchmarks under
`scripts/performance_test`. It runs the same mixed workload with SimpleStorage
host-memory storage and SSD offload, and records throughput, SSD byte accounting,
and storage-unit RSS.

## Prerequisites

1. Start a Ray cluster with exactly two live nodes. Each node must advertise
   the `node:<IP>` resource used by the benchmark for actor placement:

   ```bash
   # On the head node
   ray start --head --resources='{"node:10.0.0.1":1}'

   # On the worker node
   ray start --address=10.0.0.1:6379 --resources='{"node:10.0.0.2":1}'
   ```

2. Create the same node-local SSD path on both nodes. The path must also be
   visible to the benchmark driver and storage actors. When they run in
   containers, bind-mount the path at the same absolute location.
3. Run the benchmark on the Ray head node in an environment containing the
   TransferQueue runtime dependencies. A containerized driver that joins a
   node-local Ray runtime must also bind-mount that runtime's `/tmp` directory
   so it can reach the Ray socket.

## Usage

Use the wrapper to run the default 5 GiB mixed workload:

```bash
HEAD_NODE_IP=10.0.0.1 \
WORKER_NODE_IP=10.0.0.2 \
SSD_OFFLOAD_PATH=/path/to/local/ssd \
RESULTS_DIR=/path/to/results \
bash scripts/performance_test_ssd_offload/run_perf_test.sh
```

### Wrapper configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HEAD_NODE_IP` | Ray head-node IP | - | Yes |
| `WORKER_NODE_IP` | Ray worker-node IP | - | Yes |
| `SSD_OFFLOAD_PATH` | Node-local SSD path on both nodes | - | Yes |
| `RESULTS_DIR` | Output directory | `scripts/performance_test_ssd_offload/results` | No |
| `NUM_TEST_ITERATIONS` | Iterations per storage mode | `4` | No |

## Arguments

Run `perftest.py` directly to change the workload:

```bash
python scripts/performance_test_ssd_offload/perftest.py \
  --backend_config=scripts/performance_test_ssd_offload/perftest_config.yaml \
  --head_node_ip=10.0.0.1 \
  --worker_node_ip=10.0.0.2 \
  --ssd_path=/path/to/local/ssd \
  --output_dir=/path/to/results \
  --global_batch_size=512 \
  --small_fields=4 \
  --small_sample_bytes=524288 \
  --large_fields=4 \
  --large_sample_bytes=2097152 \
  --num_test_iterations=4 \
  --warmup_iterations=1
```

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--backend_config` | Path to the benchmark configuration | - | Yes |
| `--head_node_ip` | Ray head-node IP used for writer placement | - | Yes |
| `--worker_node_ip` | Ray worker-node IP used for reader placement | - | Yes |
| `--ssd_path` | Node-local SSD path | - | Yes |
| `--output_dir` | Output directory | `scripts/performance_test_ssd_offload/results` | No |
| `--num_test_iterations` | Iterations per storage mode | `4` | No |
| `--warmup_iterations` | Leading iterations excluded from the summary | `1` | No |
| `--global_batch_size` | Samples per iteration | `512` | No |
| `--small_fields` | Tensor fields below the offload threshold | `4` | No |
| `--large_fields` | Tensor fields above the offload threshold | `4` | No |
| `--small_sample_bytes` | Bytes per sample in each small field | `524288` | No |
| `--large_sample_bytes` | Bytes per sample in each large field | `2097152` | No |

Sample sizes are bytes per tensor field per sample. Small samples must be
strictly below the configured `ssd_offload.threshold_bytes`, and large samples
must be strictly above it. The SSD threshold and number of storage units come
from `perftest_config.yaml`. The number of iterations must exceed the number of
warm-up iterations.

## Benchmark Scenario

The default batch contains eight float32 tensor fields whose per-sample sizes
straddle the 1 MiB offload threshold:

| Fields | Bytes per sample | Placement |
|--------|-----------------:|-----------|
| 4 small fields | 512 KiB | Host memory |
| 4 large fields | 2 MiB | Local SSD |

With a batch size of 512, the logical payload is 5 GiB: 1 GiB remains in host
memory and 4 GiB is eligible for SSD offload. The benchmark first runs this
workload in pure host-memory mode and then repeats it with SSD offload.

## Benchmark Flow

Each iteration performs a PUT -> LIST -> GET -> CLEAR cycle through the
TransferQueue KV API. Storage-unit Prometheus metrics are sampled before PUT,
after PUT, after GET, and after CLEAR. RSS is accepted only after the configured
number of samples agree within the stability tolerance.

After excluding warm-up iterations, the benchmark reports the theoretical and
actual SSD-active bytes, verifies disk reclamation after CLEAR, and summarizes
SSD-mode RSS before PUT and after PUT, GET, and CLEAR. The post-CLEAR RSS delta
and its growth across iterations show whether memory is progressively retained.

## Distributed Smoke Test

Use a smaller workload to verify node placement, mixed-tier routing, metrics,
and cleanup:

```bash
python scripts/performance_test_ssd_offload/perftest.py \
  --backend_config=scripts/performance_test_ssd_offload/perftest_config.yaml \
  --head_node_ip="$HEAD_NODE_IP" \
  --worker_node_ip="$WORKER_NODE_IP" \
  --ssd_path="$SSD_OFFLOAD_PATH" \
  --output_dir=/tmp/tq-ssd-smoke \
  --global_batch_size=8 \
  --num_test_iterations=2 \
  --warmup_iterations=1
```

This workload keeps 16 MiB in memory and offloads 64 MiB for functional
validation. Use the default 5 GiB workload when collecting post-CLEAR RSS
trends.

## Output Format

The output directory contains:

- `mixed_memory.csv`: memory-mode throughput and diagnostic RSS samples;
- `mixed_ssd.csv`: SSD-mode throughput, RSS samples, and active SSD bytes;
- `summary.json`: warm-up-filtered SSD-byte accounting and post-CLEAR RSS
  retention measurements.

Important `summary.json` fields are:

| Field | Description |
|-------|-------------|
| `theoretical_offload_bytes` | Logical payload expected to be stored on SSD |
| `ssd_median_active_bytes_after_put` | Median bytes reported as active on SSD after PUT |
| `ssd_max_active_bytes_after_clear` | Maximum active SSD bytes after CLEAR; expected to be zero |
| `ssd_median_rss_retained_after_clear_bytes` | Median SSD-mode RSS increase remaining after CLEAR |
| `ssd_max_rss_retained_after_clear_bytes` | Maximum SSD-mode RSS increase remaining after CLEAR |
| `ssd_clear_rss_growth_bytes` | Last post-CLEAR RSS minus the first post-CLEAR RSS |
