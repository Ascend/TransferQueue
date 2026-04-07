
# OpenYuanrong-Datasystem Integration for TransferQueue

> Last updated: 04/17/2026 

## 🎉 Overview

We provide an optional storage backend [**openYuanrong-datasystem**](https://gitcode.com/openeuler/yuanrong-datasystem/blob/master/README.md) for TransferQueue to **deliver a better performance on NPU environments**.

OpenYuanrong-datasystem is a **distributed caching system** that utilizes the HBM/DRAM/SSD resources of the computing cluster to build a **near-memory computation multi-level cache**, improving data access performance in model training and inference scenarios.

In TransferQueue, **openYuanrong-datasystem provides high-performance key-value operations for host-to-host transfer via TCP/RDMA, device-to-device transfer via Ascend NPU HCCS, and remote Host-to-Device / Device-to-Host.** 
It manages the mapping between user-defined keys and metadata, and automatically resolves the data source location and builds transport channels.


We have implemented two key components to integrate TransferQueue with **openYuanrong-datasystem**: 

- **`YuanrongStorageClient`**: An adapter layer that encapsulates the functionality of openYuanrong-datasystem enables efficient read and write operations within TransferQueue.
- **`YuanrongStorageManager`**: The primary storage entry point that manages connections between TransferQueue clients and the underlying data system.

`YuanrongStorageClient` supports `put` and `get` NPU-side tensors and any type of serializable CPU-side data. 
It provides powerful performance, especially for **tensors on the NPU side**.
Users can experience its capabilities by filling in the relevant fields in the configuration of TransferQueue.

## 🚀 Quick Start

### Prerequisites
- **Python Version**: $ \geq 3.10~and \leq 3.11 $
- **Architecture**: AArch64 or x86_64

### Installation Steps

Follow these steps to build and install:

#### 1. Install Core Dependencies

Install PyTorch and TransferQueue
```bash
# Install Torch (matching the version specified for your hardware)
pip install torch==2.8.0

# Install TransferQueue from pypi
pip install TransferQueue
# or install from source code
git clone https://github.com/Ascend/TransferQueue/
cd TransferQueue
pip install -r requirements.txt
python -m build --wheel
pip install dist/*.whl
```

#### 2. Install Datasystem :
```bash
# Install the OpenYuanrong Datasystem package
pip install openyuanrong-datasystem

# Verify installation by checking for the dscli command-line tool
dscli -h
```

#### 3. (Optional) Install CANN and torch-npu

If you have NPU devices and want to accelerate the transmission of NPU tensor, 
you can install **Ascend-cann-toolkit** and **torch-npu**.

Then check whether CANN is already installed:

```bash
# For root users
ll /usr/local/Ascend/ascend-toolkit/latest

# For non-root users
ll ${HOME}/Ascend/ascend-toolkit/latest
```

If not installed, and you do need to install it, please skip to [Appendix A](#A-install-cann-for-npu-acceleration).

Ensure that CANN is installed, then install torch-npu: 
```bash
# The versions of torch and torch-npu must be the same. 
pip install torch-npu==2.8.0
```

### Use Case

Next, we will provide deployment and code examples for single-node scenarios.
For multi-node scenarios, please refer to [Appendix B](#B-deploy-multi-node-datasystem-for-multi-node-training-and-inference-scenarios).

#### Deployment

TransferQueue will automatically initialize the Yuanrong backend when `auto_init: True` is set. TransferQueue will:
- Create placement groups to ensure workers are spread across Ray nodes
- Launch YuanrongWorkerActor on each node to start datasystem workers
- Set up metastore service on the head node

Configuration example:
```yaml
backend:
  storage_backend: Yuanrong
  Yuanrong:
    auto_init: True
    worker_port: 31501
    metastore_port: 2379
    enable_yr_npu_transport: true
    worker_args: "--shared_memory_size_mb 8192 --remote_h2d_device_ids 0 --enable_huge_tlb true"
```

Configuration options:
- `auto_init`: Whether to automatically initialize Yuanrong backend. Currently only `True` is supported.
- `worker_port`: Port for Yuanrong datasystem worker on each node.
- `metastore_port`: Port for metastore service on the head node.
- `enable_yr_npu_transport`: Enable NPU transport for high-performance device-to-device data transfer.
- `worker_args`: Additional arguments passed to `dscli start` command:
  - `--shared_memory_size_mb`: Shared memory size in MB for datasystem worker.
  - `--remote_h2d_device_ids`: Enable RH2D (Remote Host-to-Device) for efficient cross-node data transfer. Specify NPU device IDs as comma-separated values (e.g., `0,1,2,3`).
  - `--enable_huge_tlb`: Enable huge page memory, required for >21GB shared memory on Ascend 910B.

Once the configuration is set, you can run your TransferQueue + Datasystem application directly.

#### Demo
You can associate `TransferQueueClient` with `YuanrongStorageManager` through the configuration dictionary when initializing the TransferQueue. 
Then, `YuanrongStorageManager` automatically creates `YuanrongStorageClient` and connects to the datasystem backend.
```python
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import (
    TransferQueueClient,
    TransferQueueController,
    process_zmq_server_info,
)
# port, manager_type and client_name are the config for booting the datasystem.
# host will be auto-detected by checking local IP addresses.
config_str = """
  manager_type: YuanrongStorageManager
  client_name: YuanrongStorageClient
  worker_port: 31501
"""
dict_conf = OmegaConf.create(config_str, flags={"allow_objects": True})
```

We have provided a template method to connect to Yuanrong within TransferQueue, as follows:
```python
class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self._initialize_transferqueue()

    def _initialize_transferqueue(self):
        # 1. Initialize TransferQueueController (single controller only)
        self.tq_controller = TransferQueueController.remote()

        # 2. Prepare necessary information of the controller
        self.tq_controller_info = process_zmq_server_info(self.tq_controller)

        tq_config = OmegaConf.create({}, flags={"allow_objects": True})  # Note: Need to generate a new DictConfig

        # with allow_objects=True to maintain ZMQServerInfo instance. Otherwise it will be flattened to dict
        tq_config.controller_info = self.tq_controller_info
        self.config = OmegaConf.merge(tq_config, self.config)

        # 3. Create TransferQueueClient
        self.tq_client = TransferQueueClient(
            client_id="Trainer",
            controller_info=self.tq_controller_info,
        )

        # 4. Connect to DataSystem
        self.tq_client.initialize_storage_manager(manager_type=self.config["manager_type"], config=self.config)

        return self.tq_client
```
And then you can call user interface of TransferQueue:

```python
# should import tensordict and torch
data = TensorDict({"text": torch.Tensor([[1, 2], [3, 4]]), "prompt": ["5", "6"]}, batch_size=[2])

trainer = Trainer(dict_conf)
trainer.tq_client.put(data=data, partition_id="train_0")

# get_meta before get_data
meta = trainer.tq_client.get_meta(
    data_fields=list(data.keys()),
    batch_size=data.size(0),
    partition_id="train_0",
    task_name="generate_sequences",
)

output = trainer.tq_client.get_data(meta)
print("output: ", output)
```

> The class ```Trainer``` in the above code can also be used as a **ray actor**:


#### Shut down datasystem:
TransferQueue automatically handles cleanup when calling `tq.close()`, which stops all Yuanrong datasystem workers gracefully.

### Datasystem Logs

If you want to inspect data transmission logs from openYuanrong-Datasystem, set the following environment variable:

```bash
export DATASYSTEM_CLIENT_LOG_DIR="datasystem_logs" # Custom Path
```

## 📕 Appendix
### A: Install CANN for NPU Acceleration

> CANN (Compute Architecture for Neural Networks) is a heterogeneous computing architecture launched by Huawei for AI scenarios.



Download the appropriate toolkit package from:
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

### B: Deploy multi-node datasystem for multi-node training and inference scenarios

TransferQueue automatically initializes Yuanrong datasystem workers across all Ray cluster nodes. Just set `auto_init: True` in the configuration and TransferQueue will handle the multi-node deployment.

Let's take two nodes (for instance, 192.168.0.1 and 192.168.0.2) as an example.

#### Deploy ray
```bash
# on head node
ray start --head --resources='{"node:192.168.0.1": 1}'

# on worker node (assume ray port of head_node is 6379)
ray start --address="192.168.0.1:6379" --resources='{"node:192.168.0.2": 1}'
```

#### Configuration

TransferQueue will detect all Ray nodes and deploy datasystem workers automatically:
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

> For more detailed deployment instructions, please refer to [yuanrong documents](https://gitcode.com/openeuler/yuanrong-datasystem/blob/master/README.md#%E9%83%A8%E7%BD%B2-openyuanrong-datasystem).
> The configuration parameters for deploying the data system can refer [dscli config](https://gitcode.com/openeuler/yuanrong-datasystem/blob/master/docs/source_zh_cn/deployment/dscli.md#%E9%85%8D%E7%BD%AE%E9%A1%B9%E8%AF%B4%E6%98%8E).

There is a demo with multi-node scenarios as fellow.

#### Run demo
In the demo below, we use ray actors to implement distributed deployment of processes.
The actor writer writes data to the head node, and the actor reader reads data from the worker nodes.
```python
from omegaconf import OmegaConf
from tensordict import TensorDict
import transfer_queue as tq
from transfer_queue import (
    TransferQueueClient,
    TransferQueueController,
    process_zmq_server_info,
)
import torch
import ray

########################################################################
# Please set up Ray cluster before running this script
# e.g. ray start --head --resources='{"node:192.168.0.1": 1}'
########################################################################
HEAD_NODE_IP = "192.168.0.1"  # Replace with your head node IP
WORKER_NODE_IP = "192.168.0.2"  # Replace with your worker node IP


def initialize_controller():
    tq_controller = TransferQueueController.remote()
    tq_controller_info = process_zmq_server_info(tq_controller)
    return tq_controller, tq_controller_info

@ray.remote
class TransferQueueClientActor:
    def __init__(self, config: dict, client_id: str):
        self.config = config
        self.client_id = client_id
        self._initialize_client()

    def _initialize_client(self):
        # Create TransferQueueClient
        self.tq_client = TransferQueueClient(
            client_id=self.client_id,
            controller_info=self.config.controller_info,
        )
        # Connect to DataSystem
        self.tq_client.initialize_storage_manager(manager_type=self.config["manager_type"], config=self.config)
        return self.tq_client
    
    def put(self, data: TensorDict, partition_id: str):
        self.tq_client.put(data=data, partition_id=partition_id)
        
    def get(self, data_fields, batch_size, partition_id, task_name=None, sampling_config=None):
        # get metadata from tq_controller
        meta = self.tq_client.get_meta(
                data_fields=data_fields,
                batch_size=batch_size,
                partition_id=partition_id,
                task_name=task_name,
                sampling_config=sampling_config,
            )
        # use meta to fetch data
        return self.tq_client.get_data(meta)


def main():
    tq.init()

    config_str = """
        manager_type: YuanrongStorageManager
        client_name: YuanrongStorageClient
        worker_port: 31501
    """
    dict_conf = OmegaConf.create(config_str, flags={"allow_objects": True})
    # It is important to pay attention to the controller's lifecycle.
    controller, dict_conf.controller_info = initialize_controller()

    # Note: host is auto-detected on each node, no need to configure explicitly
    data = TensorDict({ "prompt": torch.ones(3, 512), "big_tensor": torch.randn(3,1024,1024)}, batch_size=[3])
    # you could assign npu or gpu devices by 'resources'
    # resources={f"node:{HEAD_NODE_IP}": 0.001} could Force the actor to run on HEAD_NODE
    writer = TransferQueueClientActor.options(
            resources={f"node:{HEAD_NODE_IP}": 0.001},
    ).remote(dict_conf, "train")
    reader = TransferQueueClientActor.options(
            resources={f"node:{WORKER_NODE_IP}": 0.001}
    ).remote(dict_conf, "rollout")
        
    ray.get(writer.put.remote(data=data, partition_id="train_0"))

    output = reader.get.remote(
        data_fields=list(data.keys()),
        batch_size=data.size(0),
        partition_id="train_0",
        task_name="generate_sequences",
    )
    output = ray.get(output)

    tq.close()

if __name__ == "__main__":
    main()

```
