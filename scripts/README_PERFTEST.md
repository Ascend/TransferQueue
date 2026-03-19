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

2. Start the backend service (yuanrong, mooncake, etc.) if testing non-default backends.

## Usage

```bash
python perftest.py \
  --backend=[default|yuanrong|mooncake] \
  --client_placement=[intra_node|inter_node] \
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
| `--backend` | Backend type: default, yuanrong, mooncake | default |
| `--client_placement` | Client placement: intra_node or inter_node | intra_node |
| `--backend_config` | Path to YAML config file (optional) | None |
| `--device` | Device: cpu, npu, gpu | cpu |
| `--global_batch_size` | Global batch size | 1024 |
| `--field_num` | Number of fields | 10 |
| `--seq_len` | Sequence length | 8192 |
| `--num_global_batch` | Number of global batches | 1 |
| `--head_node_ip` | Head node IP (required) | - |
| `--worker_node_ip` | Worker node IP (required for inter_node) | None |
| `--ray_address` | Ray cluster address | auto |

## Backend Configuration

Sample config files are in `configs/`:

- **transferqueue.yaml**: Default backend config
  ```yaml
  num_data_storage_units: 8
  storage_unit_placement: normal  # or "remote"
  ```

- **yuanrong.yaml**: Yuanrong backend config
  ```yaml
  host: 127.0.0.1
  port: 31501
  enable_yr_npu_transport: false
  ```

- **mooncake.yaml**: Mooncake backend config
  ```yaml
  local_hostname: 127.0.0.1
  metadata_server: 127.0.0.1:8080
  master_server_address: 127.0.0.1:8081
  ```

## Examples

### Intra-node test with default backend
```bash
python perftest.py --backend=default --client_placement=intra_node \
  --head_node_ip=192.168.0.1
```

### Inter-node test with yuanrong backend
```bash
python perftest.py --backend=yuanrong --client_placement=inter_node \
  --backend_config=configs/yuanrong.yaml \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### Default backend with remote storage units
```bash
python perftest.py --backend=default --client_placement=intra_node \
  --backend_config=configs/transferqueue.yaml \
  --head_node_ip=192.168.0.1 --worker_node_ip=192.168.0.2
```

### NPU device test
```bash
python perftest.py --backend=mooncake --device=npu \
  --head_node_ip=192.168.0.1
```

## Output

The test prints:
- Total data size
- PUT time and throughput
- GET time and throughput
- Total round-trip throughput

Throughput is shown in both Gb/s (gigabits per second) and GB/s (gigabytes per second).
