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
  --backend_config=../../transfer_queue/config.yaml \
  --device=[cpu|npu|gpu] \
  --global_batch_size=1024 \
  --field_num=10 \
  --seq_len=8192 \
  --head_node_ip=192.168.0.1 \
  --worker_node_ip=192.168.0.2
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--backend_config` | Path to backend config YAML file (required) | -       |
| `--device` | Device: cpu, npu, gpu | cpu     |
| `--global_batch_size` | Global batch size | 1024    |
| `--field_num` | Number of fields | 10      |
| `--seq_len` | Sequence length | 8192    |
| `--num_test_iterations` | Number of test iterations | 4       |
| `--head_node_ip` | Head node IP (required) | -       |
| `--worker_node_ip` | Worker node IP (required for Yuanrong) | None    |
| `--output_csv` | Path to output CSV file (optional) | None    |

## Backend Configuration

The script reads the backend configuration directly from the provided `--backend_config` YAML file. The backend type is determined by `backend.storage_backend` in the config file.

For device support of each backend,
- `SimpleStorage` backend supports `cpu`
- `Yuanrong` supports `cpu` and `npu`
- `MooncakeStore` supports `cpu` and `gpu`

## Test Data Format

The test case creates TensorDict with three types of fields:

1. **Regular tensors**: Shape `(batch_size, seq_length)`, float32
2. **Nested tensors** (non-NPU devices): Variable-length sequences with lengths forming an arithmetic progression from 1 to `seq_length`. For a batch of size N, element j has length `1 + j * (seq_length - 1) / (N - 1)`. This gives an average nested length of approximately `seq_length / 2`, making the nested column size roughly half of a regular tensor column.
3. **NonTensorStack strings**: Each string is `seq_length * 4` bytes to match the memory footprint of one tensor element.

### NPU Fallback

NPU does not support nested tensors. When running with `--device=npu`, the nested tensor fields are replaced with regular tensors of shape `(batch_size, seq_length // 2)` to maintain comparable total data size while avoiding nested tensor operations.

## Yuanrong Backend

For Yuanrong backend, writer runs on head node and reader runs on worker node.

## Running Full Test Suite

The `run_perf_test.sh` script automates the full performance test suite:

```bash
cd scripts/performance_test
./run_perf_test.sh
```

### Configuration

Configure the test environment via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HEAD_NODE_IP` | Head node IP address | 127.0.0.1 |
| `WORKER_NODE_IP` | Worker node IP address | 127.0.0.1 |
| `DEVICE` | Device type (cpu, npu, gpu) | cpu |
| `NUM_TEST_ITERATIONS` | Number of iterations per test | 4 |

Example:
```bash
HEAD_NODE_IP=192.168.0.1 WORKER_NODE_IP=192.168.0.2 DEVICE=npu ./run_perf_test.sh
```

### Test Matrix

The script tests all combinations of:
- **Backends**: SimpleStorage, Yuanrong, MooncakeStore, Ray (baseline)
- **Data sizes**: Small (batch=1024, fields=9, seq=8192), Medium (batch=4096, fields=15, seq=32768), Large (batch=8192, fields=21, seq=128000)

### Output

- CSV results are saved to `results/{backend}_{size}.csv` (e.g., `results/simplestorage_small.csv`)
- A performance comparison chart is generated as `results/performance_comparison.pdf`

### draw_figure.py

After running the tests, `draw_figure.py` reads all CSV files from the `results/` directory and generates a bar chart comparing total throughput (Gbps) across backends and data sizes.

## Examples

Individual test examples using `perftest.py`:

### SimpleStorage backend
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=SimpleStorage \
  --head_node_ip=192.168.0.1
```

### Yuanrong backend
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=Yuanrong \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### MooncakeStore backend
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=MooncakeStore \
  --head_node_ip=192.168.0.1
```

### NPU device test (Yuanrong backend)
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=Yuanrong --device=npu \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### Output to CSV
```bash
python perftest.py --backend_config=perftest_config.yaml --backend=SimpleStorage \
  --head_node_ip=192.168.0.1 --output_csv=results.csv
```

## Output

The test prints:
- Total data size
- PUT time and throughput
- GET time and throughput
- Total round-trip throughput

Throughput is shown in both Gb/s (gigabits per second) and GB/s (gigabytes per second).

### CSV Output

When using `--output_csv`, the test writes results to a CSV file with the following columns:
- backend
- device
- total_data_size_gb
- put_time
- get_time
- put_gbit_per_sec
- get_gbit_per_sec
- total_gbit_per_sec

The test runs `--num_test_iterations` iterations (default: 4) and saves all results to the CSV.
