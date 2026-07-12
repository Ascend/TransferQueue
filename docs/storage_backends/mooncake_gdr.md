# GPUDirect RDMA (GDR) for MooncakeStore Backend

## Overview

When using the MooncakeStore backend, TransferQueue supports **GPUDirect RDMA (GDR)** to transfer tensor data directly between GPU memory and the network, without going through the CPU. This reduces end-to-end transfer latency and CPU overhead compared to the default path.

**Without GDR** (default): GPU tensor → local CPU → RDMA → remote CPU (storage) → RDMA → local CPU → GPU tensor

**With GDR**: GPU tensor → RDMA → remote CPU (storage) → RDMA → GPU tensor

## Prerequisites

### Hardware

- RDMA-capable NIC (e.g., Mellanox/NVIDIA ConnectX series)
- GPU with GPUDirect RDMA support (query: `nvidia-smi --query-gpu=gdr_supported --format=csv`)
- Both NIC and GPU must be on the same PCIe root complex

### Software

| Dependency | Notes |
|---|---|
| `mooncake-transfer-engine` | MooncakeStore backend |
| `cuda-python` | Required for GDR operations |
| MLNX OFED or inbox RDMA drivers | Driver stack for ConnectX NICs; typically installed by the cluster administrator |

## Quick Start

### Installation

```bash
pip install TransferQueue[mooncake]
pip install cuda-python
```

### Configuration

Set `use_gdr: true` and `protocol: rdma` in the MooncakeStore config:

```python
import transfer_queue as tq
from omegaconf import OmegaConf

conf = OmegaConf.create({
    "backend": {
        "storage_backend": "MooncakeStore",
        "MooncakeStore": {
            "metadata_server": "localhost:50050",
            "master_server_address": "localhost:50051",
            "protocol": "rdma",        # required for GDR
            "device_name": "",         # auto-select RDMA NIC
            "use_gdr": True,
            "gdr_staging_buffer_mb": 1024,
        },
    },
})

tq.init(conf)
```

### Usage

No API changes are required. `kv_put` / `kv_batch_get` behave identically from the caller's perspective:

```python
import torch
from tensordict import TensorDict

data = TensorDict(
    {"logits": torch.randn(8, 128, device="cuda")},
    batch_size=[8],
)
tq.kv_batch_put(keys=[f"s{i}" for i in range(8)], partition_id="train", fields=data)

result = tq.kv_batch_get(keys=[f"s{i}" for i in range(8)], partition_id="train")
```

## What Goes Through GDR

When `use_gdr: true`, TransferQueue routes data as follows:

| Data type | Transfer path |
|---|---|
| GPU tensors | GDR (direct GPU ↔ network) |
| CPU tensors | GDR (GPU memory is used as an intermediate, then transferred via RDMA) |
| Non-tensor values (Python scalars, dicts, etc.) | CPU RDMA, regardless of `use_gdr` |

## When GDR Is Not Active

Even with `use_gdr: true`, GDR will not be used in the following situations:

- **CUDA context not initialized**: If the process has not initialized a CUDA context before calling `tq.init()`, TransferQueue treats this as a signal that the process does not intend to use CUDA, and falls back to CPU RDMA silently. TransferQueue will not initialize CUDA on behalf of the caller. This allows a single cluster-wide `use_gdr: true` config to cover both GPU workers and CPU-only workers (e.g., controller actors) without separate configurations.
- **`gdr_staging_buffer_mb: 0`**: Setting the buffer size to zero disables GDR even if `use_gdr: true`.
- **Hardware does not support GDR**: If the GPU or NIC does not support GPUDirect RDMA, TransferQueue raises an error on the first transfer rather than silently falling back, because this indicates a misconfiguration.

> **Note**: GDR is initialized when `tq.init()` is called. If your process assigns the GPU device (e.g., `torch.cuda.set_device()`) after calling `tq.init()`, GDR will not be active. Always set the GPU device before calling `tq.init()`.

## GDR Buffer

Each process that enables GDR allocates one fixed-size GPU memory buffer on the first GDR transfer, registered with the RDMA NIC once for the lifetime of the process. The buffer lives on the GPU that the process's CUDA context is bound to at that point.

`gdr_staging_buffer_mb` controls the size of this buffer. Every process (i.e., every actor) holds its own independent buffer — there is no sharing across actors.

Only one buffer per process is allocated. All GDR transfers within a process go through this single buffer sequentially. This is sufficient for RL workloads because training and inference alternate in phases: within a single actor, there is no concurrent GDR traffic that would benefit from multiple buffers running in parallel.

## Configuration Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `use_gdr` | bool | `false` | Enable GDR. Requires `protocol: rdma`. |
| `gdr_staging_buffer_mb` | int | `1024` | Amount of GPU memory reserved for GDR transfers, in MB. Set to `0` to disable GDR even if `use_gdr: true`. |
| `protocol` | str | `"tcp"` | Transport protocol. Must be `"rdma"` when `use_gdr: true`. |
| `device_name` | str | `""` | RDMA NIC device name (e.g., `mlx5_0`). Leave empty to let Mooncake auto-select. |
