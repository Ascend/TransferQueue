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
| `--backend_config` | Path to backend config YAML file (required) | - |
| `--device` | Device: cpu, npu, gpu | cpu |
| `--global_batch_size` | Global batch size | 1024 |
| `--field_num` | Number of fields | 10 |
| `--seq_len` | Sequence length | 8192 |
| `--num_test_iterations` | Number of test iterations | 3 |
| `--head_node_ip` | Head node IP (required) | - |
| `--worker_node_ip` | Worker node IP (required for Yuanrong) | None |
| `--ray_address` | Ray cluster address | auto |
| `--output_csv` | Path to output CSV file (optional) | None |

## Backend Configuration

The script reads the backend configuration directly from the provided `--backend_config` YAML file. The backend type is determined by `backend.storage_backend` in the config file.

For device support of each backend,
- `SimpleStorage` backend supports `cpu`
- `Yuanrong` supports `cpu` and `npu`
- `MooncakeStore` supports `cpu` and `gpu`

## Yuanrong Backend

For Yuanrong backend, writer runs on head node and reader runs on worker node.

## Examples

### SimpleStorage/Mooncake backend
```bash
python perftest.py --backend_config=../../transfer_queue/config.yaml \
  --head_node_ip=192.168.0.1
```

### Yuanrong backend
```bash
python perftest.py --backend_config=../../transfer_queue/config.yaml \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### NPU device test
```bash
python perftest.py --backend_config=../../transfer_queue/config.yaml --device=npu \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### Output to CSV
```bash
python perftest.py --backend_config=../../transfer_queue/config.yaml \
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

The test runs `--num_test_iterations` iterations (default: 3) and saves all results to the CSV.
