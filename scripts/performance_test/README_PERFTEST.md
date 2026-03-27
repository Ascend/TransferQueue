# TransferQueue Throughput Test

This script runs throughput tests for TransferQueue with different backends.

## Prerequisites

1. Start Ray cluster with node resources:
   ```bash
   # On head node
   ray start --head --resources='{"node:192.168.0.1":1}'
   # On worker node
   ray start --address=192.168.0.1:6379 --resources='{"node:192.168.0.2":1}'
   ```

2. Start the backend service (Yuanrong, MooncakeStore, etc.) if testing non-SimpleStorage backends.

## Usage

```bash
python perftest.py \
  --backend_config=perftest_config.yaml \
  --backend=SimpleStorage \
  --device=cpu \
  --global_batch_size=1024 \
  --field_num=9 \
  --seq_len=8192 \
  --head_node_ip=192.168.0.1 \
  --worker_node_ip=192.168.0.2
```

## Arguments

| Argument | Description | Default | Required |
|----------|-------------|---------|----------|
| `--backend_config` | Path to backend config YAML file | - | Yes |
| `--backend` | Override `storage_backend` in config (`SimpleStorage`, `Yuanrong`, `MooncakeStore`) | None | No |
| `--device` | Device: `cpu`, `npu`, `gpu` | `cpu` | No |
| `--global_batch_size` | Global batch size | 1024 | No |
| `--field_num` | Number of fields in the TensorDict | 10 | No |
| `--seq_len` | Sequence length | 8192 | No |
| `--num_test_iterations` | Number of test iterations | 4 | No |
| `--head_node_ip` | Head node IP address | - | Yes |
| `--worker_node_ip` | Worker node IP address (required for Yuanrong) | None | No |
| `--output_csv` | Path to output CSV file | None | No |

## Backend Configuration

The script reads the backend configuration directly from the provided `--backend_config` YAML file. The backend type is determined by `backend.storage_backend` in the config file. When `--backend` is specified, it overrides the value in the config.

For device support of each backend:
- `SimpleStorage`: `cpu`
- `Yuanrong`: `cpu`, `npu`
- `MooncakeStore`: `cpu`, `gpu`

## Test Data Format

The test case creates a `TensorDict` with three types of fields to simulate real training batches:

1. **Regular tensors**: Shape `(batch_size, seq_length)`, float32.
2. **Nested tensors** (non-NPU devices): Variable-length ragged sequences with lengths forming an arithmetic progression from 1 to `seq_length`. Average length ≈ `seq_length / 2`, so each nested field is roughly half the size of a regular field.
3. **NonTensorStack strings**: Each string is `seq_length × 4` bytes, matching the memory footprint of one tensor element.

Fields are distributed evenly across the three types (rounded up). For NPU devices, nested tensors fall back to regular tensors of shape `(batch_size, seq_length // 2)`.

## Test Flow

Each iteration performs a PUT → LIST → GET → DELETE cycle via TransferQueue's KV API:

1. **PUT** (`kv_batch_put`): Writer sends the TensorDict to storage.
2. **LIST** (`kv_list`): Reader queries available keys in the partition.
3. **GET** (`kv_batch_get`): Reader fetches data for those keys.
4. **DELETE** (`kv_clear`): Writer removes the written data.

The test runs `--num_test_iterations` iterations. Data creation only happens in the first iteration; subsequent iterations reuse the same TensorDict to isolate transfer overhead.

## Yuanrong Backend

For Yuanrong backend, writer runs on the head node and reader runs on the worker node. `--worker_node_ip` is required.

## Running Full Test Suite

The `run_perf_test.sh` script automates the full test suite across all backends and data sizes, then generates a comparison chart:

```bash
cd scripts/performance_test
./run_perf_test.sh
```

### Configuration

Configure via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HEAD_NODE_IP` | Head node IP address | `127.0.0.1` |
| `WORKER_NODE_IP` | Worker node IP address | `127.0.0.1` |
| `DEVICE` | Device type (`cpu`, `npu`, `gpu`) | `cpu` |
| `NUM_TEST_ITERATIONS` | Number of iterations per test | `4` |
| `USE_COMPLEX_CASE` | Run with complex test case (nested + nontensor fields) | `false` |

Example:
```bash
# Simple case (default, regular tensors only)
./run_perf_test.sh

# Complex case (nested tensors + nontensor strings)
USE_COMPLEX_CASE=true ./run_perf_test.sh

# With specific node IPs & use NPU
HEAD_NODE_IP=192.168.0.1 WORKER_NODE_IP=192.168.0.2 DEVICE=npu ./run_perf_test.sh
```

### Test Matrix

- **Backends**: SimpleStorage, Yuanrong, MooncakeStore, Ray (baseline)
- **Data sizes**: Small (batch=1024, fields=9, seq=8192), Medium (batch=4096, fields=15, seq=32768), Large (batch=8192, fields=21, seq=128000)

### Output

- CSV results: `results/{backend}_{size}.csv` (e.g., `results/simplestorage_small.csv`, `results/ray_baseline_medium.csv`)
- Performance chart: `results/performance_comparison.pdf`

### Ray Baseline

`ray_perftest_baseline.py` measures raw Ray inter-node transfer throughput without TransferQueue, serving as a baseline. It passes a TensorDict directly to a remote Ray actor (via `ray.get`), using the same test data format. It is automatically included in `run_perf_test.sh`.

### draw_figure.py

After running the tests, `draw_figure.py` reads all CSV files from `results/` and generates a grouped bar chart comparing total throughput (Gbps) across backends and data sizes.

## Examples

### SimpleStorage backend
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=SimpleStorage \
  --head_node_ip=192.168.0.1
```

### Yuanrong backend (inter-node)
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=Yuanrong \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### MooncakeStore backend
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=MooncakeStore \
  --head_node_ip=192.168.0.1
```

### NPU device test (Yuanrong)
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=Yuanrong --device=npu \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### Output to CSV
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=SimpleStorage \
  --head_node_ip=192.168.0.1 --output_csv=results.csv
```

## Output Format

The test prints:
- Total data size
- PUT time and throughput
- GET time and throughput
- Total round-trip throughput

Throughput is shown in both Gb/s (gigabits per second) and GB/s (gigabytes per second).

### CSV Columns

| Column | Description |
|--------|-------------|
| `backend` | Backend name |
| `device` | Device type |
| `total_data_size_gb` | Data size in GB |
| `put_time` | PUT duration (seconds) |
| `get_time` | GET duration (seconds) |
| `put_gbit_per_sec` | PUT throughput (Gbps) |
| `get_gbit_per_sec` | GET throughput (Gbps) |
| `total_gbit_per_sec` | Round-trip throughput (Gbps) |
