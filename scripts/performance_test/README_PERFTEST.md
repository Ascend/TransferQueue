# TransferQueue Throughput Test

This script runs throughput tests for TransferQueue with different backends.

## Prerequisites

1. Start Ray cluster with node resources:
   ```bash
   # On head node
   ray start --head --resources='{"node:192.168.0.1":1}'

   # On worker node
   ray start --address=192.168.0.1 --resources='{"node:192.168.0.2":1}'
   ```

2. Start the backend service (Yuanrong, MooncakeStore, etc.) if testing non-SimpleStorage backends.

## Usage

```bash
python perftest.py \
  --backend=[SimpleStorage|Yuanrong|MooncakeStore] \
  --backend_config=xxx.yaml \
  --device=[cpu|npu|gpu] \
  --global_batch_size=1024 \
  --field_num=10 \
  --seq_len=8192 \
  --num_global_batch=1 \
  --head_node_ip=192.168.0.1 \
  --worker_node_ip=192.168.0.2
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--backend` | Backend type: SimpleStorage, Yuanrong, MooncakeStore | SimpleStorage |
| `--backend_config` | Path to YAML config file (optional) | None |
| `--device` | Device: cpu, npu, gpu | cpu |
| `--global_batch_size` | Global batch size | 1024 |
| `--field_num` | Number of fields | 10 |
| `--seq_len` | Sequence length | 8192 |
| `--num_global_batch` | Number of global batches | 1 |
| `--head_node_ip` | Head node IP (required) | - |
| `--worker_node_ip` | Worker node IP (required for Yuanrong inter_node) | None |
| `--ray_address` | Ray cluster address | auto |
| `--output_csv` | Path to output CSV file (optional) | None |

## Backend Configuration

Sample config files are in `configs/`:

- **simple_storage.yaml**: SimpleStorage backend config
  ```yaml
  num_data_storage_units: 1
  ```

- **yuanrong.yaml**: Yuanrong backend config
  ```yaml
  host: 127.0.0.1
  port: 31501
  enable_yr_npu_transport: false
  client_placement: inter_node  # or "intra_node"
  ```

- **mooncake_store.yaml**: MooncakeStore backend config
  ```yaml
  local_hostname: 127.0.0.1
  metadata_server: 127.0.0.1:8080
  master_server_address: 127.0.0.1:8081
  ```

For device support of each backend,
- `SimpleStorage` backend supports `cpu`
- `Yuanrong` supports `cpu` and `npu`
- `MooncakeStore` supports `cpu` and `gpu`

## Yuanrong Client Placement

For Yuanrong backend, since `put` is always local-first, we need to start client actors on different nodes to test cross-node transfer. The client placement is configured in the YAML file:
- `client_placement: intra_node`: Both writer and reader run on head node
- `client_placement: inter_node`: Writer runs on head node, reader runs on worker node (default)

## Examples

### SimpleStorage backend
```bash
python perftest.py --backend=SimpleStorage \
  --head_node_ip=192.168.0.1
```

### Yuanrong backend
```bash
python perftest.py --backend=Yuanrong \
  --backend_config=configs/yuanrong.yaml \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### NPU device test
```bash
python perftest.py --backend=Yuanrong --device=npu \
  --head_node_ip=192.168.0.1
```

### Output to CSV
```bash
python perftest.py --backend=SimpleStorage \
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
- client_placement
- device
- total_data_size_gb
- put_time
- get_time
- put_gbit_per_sec
- put_gbyte_per_sec
- get_gbit_per_sec
- get_gbyte_per_sec
- total_gbit_per_sec
- total_gbyte_per_sec

The test runs 3 iterations and saves all 3 results to the CSV.
