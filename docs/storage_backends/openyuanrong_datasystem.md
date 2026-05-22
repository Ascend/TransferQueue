# OpenYuanrong-Datasystem Integration for TransferQueue

> Last updated: 05/18/2026 

## Overview

We provide an optional storage backend [**openYuanrong-datasystem**](https://gitcode.com/openeuler/yuanrong-datasystem/blob/master/README.md) for TransferQueue to **deliver better performance on NPU environments**.

OpenYuanrong-datasystem is a **distributed caching system** that utilizes the HBM/DRAM/SSD resources of the computing cluster to build a **near-memory computation multi-level cache**, improving data access performance in model training and inference scenarios.

In TransferQueue, **openYuanrong-datasystem provides high-performance key-value operations for host-to-host transfer via TCP/RDMA, device-to-device transfer via Ascend NPU HCCS, and remote Host-to-Device / Device-to-Host.** 
It manages the mapping between user-defined keys and metadata, and automatically resolves the data source location and builds transport channels.

We have implemented two key components to integrate TransferQueue with **openYuanrong-datasystem**: 

- **`YuanrongStorageClient`**: An adapter layer that encapsulates the functionality of openYuanrong-datasystem and enables efficient read and write operations within TransferQueue.
- **`YuanrongStorageManager`**: The primary storage entry point that manages connections between TransferQueue clients and the underlying data system.

`YuanrongStorageClient` supports `put` and `get` NPU-side tensors and any type of serializable CPU-side data. 
It provides powerful performance, especially for **tensors on the NPU side**.

To use Yuanrong backend, set `storage_backend: Yuanrong` in the configuration. 
TransferQueue's default configuration is located at `transfer_queue/config.yaml`.
When Yuanrong backend is selected, `YuanrongStorageManager` and `YuanrongStorageClient` handle all data storage and retrieval operations.

## Quick Start

### Prerequisites
- **Python Version**: >= 3.10, <= 3.11
- **Architecture**: aarch64 or x86_64

### Installation Steps

Follow these steps to build and install:

#### 1. Install TransferQueue with Yuanrong

Use the `[yuanrong]` extras to install PyTorch, TransferQueue, and openYuanrong-datasystem in one command:

```bash
# Install torch, recommended version: 2.8.0 or higher.
# Version 2.8.0 is used as an example.
pip install torch==2.8.0

# Install from PyPI
pip install TransferQueue[yuanrong]

# Or install from source
git clone https://github.com/Ascend/TransferQueue/
cd TransferQueue
pip install -e ".[yuanrong]"
```

Verify installation:
```bash
dscli -h  # Check datasystem CLI tool
```

#### 2. (Optional for NPU Transfer) Install CANN and torch-npu

If you have NPU devices and want to accelerate the transmission of NPU tensor, you need to install **Ascend-cann-toolkit** and **torch-npu**.

Then check whether CANN is already installed:

```bash
# For root users
ls /usr/local/Ascend/ascend-toolkit/latest

# For non-root users
ls ${HOME}/Ascend/ascend-toolkit/latest
```

If not installed, and you do need to install it, please skip to [Appendix A](#a-install-cann-for-npu-acceleration).

Ensure that CANN is installed, then install torch-npu: 
```bash
# The versions of torch and torch-npu must be the same. 
pip install torch-npu==2.8.0
```

### Single Node Demo

After installation, you can run TransferQueue with Yuanrong backend.

First, start a local Ray cluster. TransferQueue relies on Ray for distributed management:
```bash
ray start --head
```

Then run the simple demo:
```python
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

# Configure Yuanrong backend
# User-provided config will be merged with TransferQueue's default config.yaml.
# Specified fields override defaults; unspecified fields retain default values.
conf = OmegaConf.create({"backend": {"storage_backend": "Yuanrong"}})

# Initialize TransferQueue + Yuanrong
tq.init(conf)

# Put data using kv_put
data = TensorDict({"input": torch.randn(2, 10)}, batch_size=[2])
tq.kv_batch_put(keys=["sample_0", "sample_1"], partition_id="train", fields=data)

# Get data using kv_batch_get
result = tq.kv_batch_get(keys=["sample_0", "sample_1"], partition_id="train")
print("output:", result)

# Cleanup
tq.close()
```

## Deployment

Yuanrong datasystem is deployed **per-host** (one worker per node), managing all TransferQueue clients on the same node. It is not a per-client deployment.

When `auto_init: True` is set in the configuration, TransferQueue automatically initializes the Yuanrong backend during `tq.init()`. The deployment process:

1. **Detects Ray cluster nodes** - identifies all alive nodes in the Ray cluster
2. **Launches YuanrongWorkerActor** - creates one actor per node to manage the datasystem worker
3. **Sets up metastore service** - the head node (driver node) starts the metastore service, other nodes connect as workers

### Configuration

```yaml
backend:
  storage_backend: Yuanrong
  Yuanrong:
    auto_init: True                    # Automatically initialize Yuanrong backend
    worker_port: 31501                 # Port for Yuanrong datasystem worker on each node
    metastore_port: 2379               # Port for metastore service on the head node
    enable_yr_npu_transport: true      # Enable NPU transport for high-performance device-to-device transfer
    worker_args: "--shared_memory_size_mb 8192 --remote_h2d_device_ids 0 --enable_huge_tlb true"
```

**General Options:**
- `auto_init`: Whether to automatically initialize Yuanrong backend. Default is `True`.
- `worker_port`: Port for Yuanrong datasystem worker on each node.
- `metastore_port`: Port for metastore service on the head node.
- `worker_args`: Additional arguments passed to `dscli start` command:
  - `--shared_memory_size_mb`: Shared memory size in MB for datasystem worker.
  - `--enable_huge_tlb`: Configure huge page memory to reduce TLB misses and improve memory access efficiency. Note: may cause system memory shortage, kernel OOM, or system instability.  **Please allocate huge pages before starting datasystem** - refer to [Huge Page Guide](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/appendix/hugepage_guide.html).

**NPU Transfer Options:**
- `enable_yr_npu_transport`: Enable NPU transport for high-performance device-to-device data transfer. Set to `true` when using NPU tensors.
- `worker_args` (**mandatory** when `enable_yr_npu_transport: true`):
  - `--remote_h2d_device_ids`: Enable RH2D (Remote Host-to-Device) for efficient cross-node NPU data transfer. Specify NPU device IDs as comma-separated values (e.g., `0,1,2,3`). Yuanrong manages all specified devices - to put/get tensors on NPU `X`, device ID `X` must be included in this argument.

> More configuration parameters for deploying the datasystem can refer to [dscli config](https://gitcode.com/openeuler/yuanrong-datasystem/blob/master/docs/source_zh_cn/deployment/dscli.md).

### Multi-Node Deployment

TransferQueue automatically deploys Yuanrong datasystem workers across all Ray cluster nodes. Just set `auto_init: True` and TransferQueue will handle the multi-node deployment.

#### Deploy Ray Cluster

```bash
# On head node (assume IP of head_node is 192.168.0.1)
ray start --head --resources='{"node:192.168.0.1": 1}'

# On worker node (assume IP of worker_node is 192.168.0.2)
ray start --address="192.168.0.1:6379" --resources='{"node:192.168.0.2": 1}'
```

The `--resources` parameter defines node-specific resources. It can be used to control Ray actor placement across nodes. For NPU environments, you may also add `--resources='{"NPU": 4}'` or configure `ASCEND_RT_VISIBLE_DEVICES`.

#### Multi-Node Configuration

```yaml
backend:
  storage_backend: Yuanrong
  Yuanrong:
    auto_init: True
    worker_port: 31501
    metastore_port: 2379
    enable_yr_npu_transport: true
    worker_args: "--shared_memory_size_mb 65536 --remote_h2d_device_ids 0 --enable_huge_tlb true"
```

TransferQueue will detect all Ray nodes and deploy datasystem workers automatically.

#### Multi-Node Demo

> **Note**: Before running the demo below, modify `HEAD_NODE_IP` and `WORKER_NODE_IP` to match your actual node IPs.

```python
import torch
import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict

########################################################################
# Please set up Ray cluster before running this script
# e.g., ray start --head --resources='{"node:192.168.0.1": 1}' on head node
#       ray start --address="192.168.0.1:6379" --resources='{"node:192.168.0.2": 1}' on worker node
########################################################################

HEAD_NODE_IP = "192.168.0.1"    # Replace with your head node IP
WORKER_NODE_IP = "192.168.0.2"  # Replace with your worker node IP

# Configure Yuanrong backend
# User-provided config will be merged with TransferQueue's default config.yaml.
# Specified fields override defaults; unspecified fields retain default values.
# For NPU tensor transfer, add enable_yr_npu_transport and --remote_h2d_device_ids.
conf = OmegaConf.create({
    "backend": {
        "storage_backend": "Yuanrong",
        "Yuanrong": {
            "enable_yr_npu_transport": True,
            "worker_args": "--remote_h2d_device_ids 0,1",
        }
    }
})

# Initialize TransferQueue + Yuanrong
# This will deploy Yuanrong workers on all Ray cluster nodes
tq.init(conf)


@ray.remote
class DataActor:
    """Ray actor for put/get data. Actor is persistent, keeping tensor valid during its lifetime."""
    
    def __init__(self, config):
        # Each process must call tq.init() to get a client
        tq.init(config)
        torch.npu.set_device(0)
    
    def put_data(self):
        """Put data on this node."""
        data = TensorDict({"input": torch.ones((3, 512), device="npu")}, batch_size=[3])
        tq.kv_batch_put(keys=["s0", "s1", "s2"], partition_id="train", fields=data)
        print(f"[put] Data put completed")
    
    def get_data(self):
        """Get data on this node."""
        result = tq.kv_batch_get(keys=["s0", "s1", "s2"], partition_id="train")
        print(f"[get] Data get completed: {result['input']}")
        return result


# Create actors on different nodes
put_actor = DataActor.options(resources={f"node:{HEAD_NODE_IP}": 0.001, "NPU": 1}).remote(conf)
get_actor = DataActor.options(resources={f"node:{WORKER_NODE_IP}": 0.001, "NPU": 1}).remote(conf)

# Put data on head node
ray.get(put_actor.put_data.remote())

# Get data on worker node (cross-node transfer)
result = ray.get(get_actor.get_data.remote())

# Cleanup
tq.close()
```

> For more detailed deployment instructions, please refer to [openYuanrong-datasystem documents](https://gitcode.com/openeuler/yuanrong-datasystem/blob/master/README.md).


### Shutdown

TransferQueue automatically handles cleanup when calling `tq.close()`, which stops all Yuanrong datasystem workers gracefully.

## Manual Yuanrong Startup (auto_init=False)

When you need to manually manage Yuanrong datasystem (e.g., independent deployment, integration with other systems), you can use `dscli` command-line tool.

### Start Metastore + Worker on Head Node

```bash
dscli start -w --worker_address <HEAD_IP>:31501 \
    --metastore_address <HEAD_IP>:2379 \
    --start_metastore_service true \
    --arena_per_tenant 1 \
    --enable_worker_worker_batch_get true \
    --shared_memory_size_mb 8192
```

### Start Worker on Worker Nodes

```bash
dscli start -w --worker_address <WORKER_IP>:31501 \
    --metastore_address <HEAD_IP>:2379 \
    --arena_per_tenant 1 \
    --enable_worker_worker_batch_get true \
    --shared_memory_size_mb 8192
```

### Stop Worker

```bash
dscli stop --worker_address <IP>:31501
```

### Connect to Manually Started Yuanrong in TransferQueue

Set `auto_init` to `False` (experimental support):

```yaml
backend:
  storage_backend: Yuanrong
  Yuanrong:
    auto_init: False
    worker_port: 31501
```

Note: In manual startup mode, you need to manage the lifecycle of Yuanrong workers yourself.

## FAQ

### Failed to Start Datasystem Worker

If initialization fails with `RuntimeError: Failed to start datasystem worker...`, check the following possible causes:

**1. Port Conflict**

Check if `worker_port` or `metastore_port` is already in use:
```bash
netstat -tlnp | grep 31501
netstat -tlnp | grep 2379
```
Solution: Change the port or clean up the occupying process.

> If a TransferQueue task terminates abnormally without calling `tq.close()`, the datasystem may become a defunct process and occupy the port.

**2. Shared Memory Allocation Failure**

If you encounter an error like:
```
Runtime error: failed to mmap shared memory: Cannot allocate memory
```
Check the following:
- Docker container shared memory limit (default is 64MB, may need increase)
- System available memory for shared memory allocation
- Huge page configuration if `--enable_huge_tlb true` is enabled

Solution: Increase container shared memory (`--shm-size` flag), or reduce `--shared_memory_size_mb` value.

**3. Proxy Configuration**

HTTP/HTTPS proxy settings may interfere with Yuanrong's internal communication, causing metastore connection timeout errors.

Yuanrong datasystem uses IP addresses directly for internal node communication. If proxy environment variables (`http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY`) are set, they may route internal traffic through the proxy instead of direct connections.

Solution:  unset proxy variables before running:
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```



### Residual Worker Process

If the previous run did not close properly (e.g., task crashed without `tq.close()`), datasystem worker processes may remain:

```bash
# Check residual processes
ps aux | grep datasystem_worker

# Clean up gracefully
dscli stop --worker_address <IP>:<PORT>

# Force cleanup (use with caution)
pkill -f datasystem_worker
```

### Multi-Process Initialization

In multi-process scenarios, each process must call `tq.init()` before using TransferQueue APIs:
- The first process initializes the `TransferQueueController` and Yuanrong backend
- Subsequent processes automatically connect to the existing controller

Best practice: Let the process that initialized the backend (typically the main/driver process) call `tq.close()` for cleanup. Other processes can simply close their clients without affecting the shared backend.


### NPU Transfer Issues

When using `enable_yr_npu_transport: true`, ensure:
- CANN toolkit is properly installed
- `torch-npu` version matches `torch` version
- `--remote_h2d_device_ids` includes all device IDs you intend to use

Common errors and solutions:
- `Device not found`: Check if device ID is included in `--remote_h2d_device_ids`
- `CANN error`: Verify CANN installation path and environment variables

### Out of Memory Error

If Yuanrong throws an OOM error during operation:
```
RuntimeError: code: [Out of memory], msg: [Shared memory no space in arena: ...]
```

Solution: Increase `--shared_memory_size_mb` in `worker_args`, or reduce the data volume being cached.


## Datasystem Logs

If you want to inspect data transmission logs from openYuanrong-Datasystem, set the following environment variable:

```bash
export DATASYSTEM_CLIENT_LOG_DIR="datasystem_logs" # Custom Path
```

## Appendix

### A: Install CANN for NPU Acceleration

> CANN (Compute Architecture for Neural Networks) is a heterogeneous computing architecture launched by Huawei for AI scenarios.

We recommend developing inside a CANN container.

#### Option 1: Docker Image (Recommended)

First, select the appropriate [CANN image](https://hub.docker.com/r/ascendai/cann) aligned with your **CANN version**, **Ascend hardware**, **OS**, and **Python version**. For example:

| CANN Version | Ascend Hardware | OS           | Python Version | Image Name                           |
| ------------ | --------------- | ------------ | -------------- | ------------------------------------ |
| 8.2.rc1      | A3              | Ubuntu 22.04 | 3.11           | cann:8.2.rc1-a3-ubuntu22.04-py3.11   |
| 8.2.rc1      | 910B            | Ubuntu 22.04 | 3.11           | cann:8.2.rc1-910b-ubuntu22.04-py3.11 |
---
Pull the image:

```bash
# For Ascend NPU A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-a3-ubuntu22.04-py3.11

# For Ascend NPU 910B
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-910b-ubuntu22.04-py3.11
```

To run a container based on this image, please refer to [official CANN image documentation](https://github.com/Ascend/cann-container-image?tab=readme-ov-file#usage).


#### Option 2: Manual Installation (.run Package)

If you prefer manual installation, download the appropriate toolkit package from:
[Ascend CANN Downloads](https://www.hiascend.com/developer/download/community/result?cann=8.3.RC1&product=1&model=30).

Please select the appropriate version for your OS and architecture (e.g., Linux + AArch64).

Then install the toolkit:

```bash
# For example, download the aarch64 package, set the execution permission, and install it.
chmod +x Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run
./Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --install

# Dependencies of CANN Installation
pip install scipy psutil tornado decorator ml-dtypes absl-py
```

After installation, confirm the toolkit path exists:

```bash
# Root user
ls /usr/local/Ascend/ascend-toolkit/latest

# Non-root user
ls ${HOME}/Ascend/ascend-toolkit/latest
```

If you need to uninstall, execute:

```bash
./Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --uninstall
```