<div align="center">
  <h2 align="center">
      TransferQueue: An asynchronous streaming data management module for efficient post-training
  </h2>
  <a href="https://arxiv.org/abs/2507.01663" target="_blank"><strong>Paper</strong></a>
| <a href="https://zhuanlan.zhihu.com/p/1930244241625449814" target="_blank"><strong>Zhihu</strong></a>
| <a href="https://github.com/TransferQueue/community_doc/blob/main/other_assets/WeChatGroup.png?raw=true" target="_blank"><strong>WeChat</strong></a>
  <br />
  <br />

  <a href="https://deepwiki.com/Ascend/TransferQueue"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;"></a>
  [![GitHub Repo stars](https://img.shields.io/github/stars/Ascend/TransferQueue)](https://github.com/Ascend/TransferQueue/stargazers/)
  [![GitHub commit activity](https://img.shields.io/github/commit-activity/w/Ascend/TransferQueue)](https://github.com/Ascend/TransferQueue/graphs/commit-activity)

</div>
<br/>

<h2 id="overview">🎉 Overview</h2>

TransferQueue is a high-performance data storage and transfer module with panoramic data visibility and streaming scheduling capabilities, optimized for efficient dataflow in post-training workflows.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/tq_arch.png?raw=true" width="70%">
</p>

TransferQueue offers **fine-grained, sub-sample-level** data management and **load-balancing** (on the way) capabilities, serving as a data gateway that decouples explicit data dependencies across computational tasks. This enables a divide-and-conquer approach, significantly simplifies the algorithm controller design.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/main_func.png?raw=true" width="100%">
</p>

<h2 id="updates">🔄 Updates</h2>

 - **Feb 8, 2026**: 🔥 The initialization and usage is greatly simplified by high-level APIs [PR#26](https://github.com/Ascend/TransferQueue/pull/26), [PR#28](https://github.com/Ascend/TransferQueue/pull/28). You can now use a Redis-style API to take advantage of most of the advanced features provided by TransferQueue!
 - **Jan 28, 2026**: We experimentally introduce `StreamingDataLoader` interface for fully-streamed production-consumption pipeline. Refer to our [tutorials/06_streaming_dataloader.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/06_streaming_dataloader.py) for details.
 - **Dec 30, 2025**: **TransferQueue x verl** integration is tested with the DAPO algorithm at scale **(64 nodes, 1024 cards)**. It significantly optimizes host memory utilization and accelerates data transfers. Stay tuned for more details!
 - **Dec 20, 2025**: 🔥 The official [tutorial](https://github.com/Ascend/TransferQueue/tree/main/tutorial) is released! Feel free to check it out.
 - **Nov 10, 2025**: We disentangle the data retrieval logic from TransferQueueController [PR#101](https://github.com/TransferQueue/TransferQueue/pull/101). Now you can implement your own `Sampler` to control how to consume the data.
 - **Nov 5, 2025**: We provide a `KVStorageManager` that simplifies the integration with KV-based storage backends [PR#96](https://github.com/TransferQueue/TransferQueue/pull/96). The first available KV-based backend is [Yuanrong](https://gitcode.com/openeuler/yuanrong-datasystem).
 - **Nov 4, 2025**: Data partition capability is available in [PR#98](https://github.com/TransferQueue/TransferQueue/pull/98). Now you can define logical data partitions to manage your train/val/test datasets.
 - **Oct 25, 2025**: We make storage backends pluggable in [PR#66](https://github.com/TransferQueue/TransferQueue/pull/66). You can try to integrate your own storage backend with TransferQueue now!
 - **Oct 21, 2025**: Official integration into verl is ready [verl/pulls/3649](https://github.com/volcengine/verl/pull/3649). Following PRs will optimize the single controller architecture by fully decoupling data & control flows.
 - **July 22, 2025**: We present a series of Chinese blogs on <a href="https://zhuanlan.zhihu.com/p/1930244241625449814">Zhihu 1</a>, <a href="https://zhuanlan.zhihu.com/p/1933259599953232589">2</a>.
 - **July 21, 2025**: We started an RFC on verl community [verl/RFC#2662](https://github.com/volcengine/verl/discussions/2662).
 - **July 2, 2025**: We publish the paper [AsyncFlow](https://arxiv.org/abs/2507.01663).

<h2 id="components">🧩 Components</h2>

### Control Plane: Panoramic Data Management

In the control plane, `TransferQueueController` tracks the **production status** and **consumption status** of each training sample as metadata. When all the required data fields are ready (i.e., written to the `TransferQueueStorageManager`), we know that this data sample can be consumed by downstream tasks.

For consumption status, we record the consumption records for each computational task (e.g., `generate_sequences`, `compute_log_prob`, etc.). Therefore, even when different computation tasks require the same data field, they can consume the data independently without interfering with each other.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/control_plane.png?raw=true" width="70%">
</p>

To make the data retrieval process more customizable, we provide a `Sampler` class that allows users to define their own data retrieval and consumption logic. Refer to the [Customize](#customize) section for details.

> In the future, we plan to support **load-balancing** and **dynamic batching** capabilities in the control plane. Additionally, we will support data management for disaggregated frameworks where each rank manages the data retrieval by itself, rather than coordinated by a single controller.

### Data Plane: Distributed Data Storage

In the data plane, we provide a pluggable design that enables TransferQueue to integrate with different storage backends according to user requirements.

Specifically, we provide a `TransferQueueStorageManager` abstraction class that defines the core APIs as follows:

- `async def put_data(self, data: TensorDict, metadata: BatchMeta) -> None`
- `async def get_data(self, metadata: BatchMeta) -> TensorDict`
- `async def clear_data(self, metadata: BatchMeta) -> None`

This class encapsulates the core interaction logic within the TransferQueue system. You only need to write a simple subclass to integrate your own storage backend. Refer to the [Customize](#customize) section for details.

Currently, we support the following storage backends:

- SimpleStorage: A basic CPU memory storage with minimal data format constraints and easy usability.
- [Yuanrong](https://gitee.com/openeuler/yuanrong-datasystem) (beta, [#PR107](https://github.com/TransferQueue/TransferQueue/pull/107), [#PR96](https://github.com/TransferQueue/TransferQueue/pull/96)): An Ascend native data system that provides hierarchical storage interfaces including HBM/DRAM/SSD.
- [MooncakeStore](https://github.com/kvcache-ai/Mooncake) (alpha, [#PR162](https://github.com/TransferQueue/TransferQueue/pull/162)): A high-performance, KV-based hierarchical storage that supports RDMA transport between GPU and DRAM.
- [RayRDT](https://docs.ray.io/en/master/ray-core/direct-transport.html) (alpha, [#PR167](https://github.com/TransferQueue/TransferQueue/pull/167)): Ray's new feature that allows Ray to store and pass objects directly between Ray actors.

Among them, `SimpleStorageUnit` serves as our default storage backend, coordinated by the `AsyncSimpleStorageManager` class. Each storage unit can be deployed on a separate node, allowing for distributed data management.

`SimpleStorageUnit` employs a 2D data structure as follows:

- Each row corresponds to a training sample, assigned a unique index within the corresponding global batch.
- Each column represents the input/output data fields for computational tasks.

This data structure design is motivated by the computational characteristics of the post-training process, where each training sample is generated in a relayed manner across task pipelines. It provides an accurate addressing capability, which allows fine-grained, concurrent data read/write operations in a streaming manner.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/data_plane.png?raw=true" width="70%">
</p>

### User Interface: High-Level & Low-Level APIs

| Level | Tier | Style | Fine-Grained Access | Streaming | Sampler | Multiple-Backends | 
|---|---|---|---|------------------|---|---|
| High | **KV Interface** ([PR#28](https://github.com/Ascend/TransferQueue/pull/28))| Put/Get/List/Clear | ✓ | ○                | ✗ | ✓ | 
| High |  **StreamingDataLoader** ([PR#23](https://github.com/Ascend/TransferQueue/pull/23)) | PyTorch DataLoader | ✓ | ✓                | ✓ | ✓ | 
| Low |  **TransferQueueClient** | Metadata-based | ✓ | ✓                | ✓ | ✓ | 


#### Key-Value based API

To simplify the usage of TransferQueue, we have provided a Redis-style high-level API that can enjoy most of the advanced features provided by TransferQueue ([PR#28](https://github.com/Ascend/TransferQueue/pull/28)).

**Methods**

- **(async_)kv_put**: Insert/Update a multi-column sample by key, with optional metadata tag
- **(async_)kv_batch_put**: Put multiple key-value pairs efficiently in batch
- **(async_)kv_batch_get**: Retrieve samples (by keys), supporting column selection (by fields)
- **(async_)kv_list**: List keys and tags (metadata) in a partition
- **(async_)kv_clear**: Remove key-value pairs from storage

**Key Features**

- **Redis-style Semantics**: Familiar KV interface (Put/Get/List) for zero learning curve
- **Fine-grained Access**: Update or retrieve specific fields (columns) within a key (row) without full op.
- **Partition Isolation**: Logical separation of storage namespaces
- **Metadata Tags**: Lightweight metadata for status tracking
- **Pluggable Backends**: Supports multiple backends

#### StreamingDataLoader API

Designed as a drop-in replacement for the standard PyTorch `DataLoader`, this API allows each rank to automatically consume data without single-controller intervention.

In this scenario, `TransferQueueController` serves as a side-controller for data dispatching, with user-defined `Sampler` class to organize dataflow.
It encapsulates the complex scheduling and data transfer logic required for various parallelism strategies, seamlessly integrating TransferQueue into existing training workflows and simplifying the development of disaggregated frameworks.

See [Roadmap](https://github.com/Ascend/TransferQueue/issues/1) and [tutorials/06_streaming_dataloader.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/06_streaming_dataloader.py) for more details.

#### Low-Level Native API

The native interface of TransferQueue are implemented in `TransferQueueClient`. It offers maximum flexibility through native, atomic operations.

Developers can leverage `TransferQueueClient` directly to implement advanced features that require fine-grained control and fully streamed data scheduling, as illustrated in the following tutorials:
- [tutorial/03_metadata_concepts.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/03_metadata_concepts.py)
- [tutorial/04_understanding_controller.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/04_understanding_controller.py)
- [tutorial/05_custom_sampler.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py)


<h2 id="show-cases">🔥 Showcases</h2>

### Collocated Example

#### verl
The primary motivation for integrating TransferQueue to verl now is to **alleviate the data transfer bottleneck of the single controller `RayPPOTrainer`**. Currently, all `DataProto` objects must be routed through `RayPPOTrainer`, resulting in a single point bottleneck of the whole post-training system.

![verl_dataflow_DataProto](https://github.com/TransferQueue/community_doc/blob/main/docs/verl_workflow.jpeg?raw=true)

Leveraging TransferQueue, we separate experience data transfer from metadata dispatch by

- Replacing `DataProto` with `BatchMeta` (metadata) and `TensorDict` (actual data) structures
- Preserving verl's original Dispatch/Collect logic via BatchMeta (maintaining single-controller debuggability)
- Accelerating data transfer by TransferQueue's distributed storage units

![verl_dataflow_TransferQueue](https://github.com/TransferQueue/community_doc/blob/main/docs/verl_workflow_with_tq.jpeg?raw=true)

You may refer to the [recipe](https://github.com/Ascend/TransferQueue/tree/dev/recipe/simple_use_case), where we mimic the verl usage in both async & sync scenarios. Official integration to verl is also available now at [verl/pulls/3649](https://github.com/volcengine/verl/pull/3649) (with subsequent PRs to further optimize the integration).

### Disaggregated Example

We have experimentally implemented a **standardized, fully-streamed distributed** workflow via TransferQueue. 

By leveraging the `RankAwareSampler` and `StreamingDataLoader` interfaces, we achieve a **streamlined micro-batch-level producer-consumer pipeline**. This design eliminates the need to manually determine data dispatching logic across varying parallelism strategies—a typical complexity in the single-controller paradigm—thereby greatly simplifying framework design. 

Please refer to our [Roadmap](https://github.com/Ascend/TransferQueue/issues/1) and [tutorials/05_streaming_dataloader.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_streaming_dataloader.py) for more details.

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/tq_streaming_dataloader.png?raw=true" width="70%">
</p>

<h2 id="quick-start">🚀 Quick Start</h2>

### Use Python package
```bash
pip install TransferQueue
```

### Install from source code

1. Clone the source code from the GitHub repository
   ```bash
   git clone https://github.com/Ascend/TransferQueue/
   cd TransferQueue
   ```

2. Install from source code
   ```bash
   pip install .
   ```
   
### Build wheel package from source code

1. Clone the source code from the GitHub repository
   ```bash
   git clone https://github.com/Ascend/TransferQueue/
   cd TransferQueue
   ```
   
2. Install dependencies
   ```bash
   pip install build
   ```

3. Build and install
   ```bash
   python -m build --wheel
   pip install dist/*.whl
   ```

<h2 id="performance">📊 Performance</h2>

<p align="center">
  <img src="https://github.com/TransferQueue/community_doc/blob/main/docs/performance_0.1.1.dev2.png?raw=true" width="100%">
</p>

> Note: The above benchmark for TransferQueue is based on our naive `SimpleStorage` backend. By introducing high-performance storage backends and optimizing serialization/deserialization, we expect to achieve even better performance. Warmly welcome contributions from the community!

For detailed performance benchmarks, please refer to [this blog](https://www.yuque.com/haomingzi-lfse7/hlx5g0/tml8ke0zkgn6roey?singleDoc#).

We also provide a [stress test report](https://www.yuque.com/haomingzi-lfse7/hlx5g0/ydbwgo5k2umaag78?singleDoc#) that demonstrates **768 concurrent clients writing 1.4 TB of data** into TransferQueue across 4 nodes. The system remains stable without any crashes or data loss, achieving 80% bandwidth.

<h2 id="customize"> 🛠️ Customize TransferQueue</h2>

### Define your own data retrieval logic
We provide a `BaseSampler` abstraction class, which defines the following interface:

```python3
@abstractmethod
def sample(
    self,
    ready_indexes: list[int],
    batch_size: int,
    *args: Any,
    **kwargs: Any,
) -> tuple[list[int], list[int]]:
    """Sample a batch of indices from the ready indices.

    Args:
        ready_indexes: List of global indices for which all required fields of the
        corresponding samples have been produced, and the samples are not labeled as
        consumed in the corresponding task.
        batch_size: Number of samples to select
        *args: Additional positional arguments for specific sampler implementations
        **kwargs: Additional keyword arguments for specific sampler implementations

    Returns:
        List of sampled global indices of length batch_size
        List of global indices of length batch_size that should be labeled as consumed
        (will never be retrieved in the future)

    Raises:
        ValueError: If batch_size is invalid or ready_indexes is insufficient
    """
    raise NotImplementedError("Subclasses must implement sample")
```

In this design, we separate data retrieval and data consumption through the two return values, which enables us to easily control sample replacement. We have implemented two reference designs: `SequentialSampler` and `GRPOGroupNSampler`.

The `Sampler` class or instance should be passed to the `TransferQueueController` during initialization. During each `get_meta` call, you can provide dynamic sampling parameters to the `Sampler`.

```python3
from transfer_queue import TransferQueueController, TransferQueueClient, GRPOGroupNSampler, process_zmq_server_info

# Option 1: Pass the sampler class to the TransferQueueController
controller = TransferQueueController.remote(GRPOGroupNSampler)

# Option 2: Pass the sampler instance to the TransferQueueController (if you need custom configuration)
your_own_sampler = YourOwnSampler(config)
controller = TransferQueueController.remote(your_own_sampler)

# Use the sampler
batch_meta = client.get_meta(
    data_fields=["input_ids", "attention_mask"],
    batch_size=8,
    partition_id="train_0",
    task_name="generate_sequences",
)
```

<span style="color: #FF0000;">**Refer to [tutorial/05_custom_sampler.py](https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py) for more details.**</span>


### How to integrate a new storage backend

The data plane is organized as follows:
```text
  transfer_queue/
  ├── storage/
  │   ├── __init__.py
  │   │── simple_backend.py             # Default distributed storage backend (SimpleStorageUnit) by TQ 
  │   ├── managers/                     # Managers are upper level interfaces that encapsulate the interaction logic with TQ system.
  │   │   ├── __init__.py
  │   │   ├──base.py                    # TransferQueueStorageManager, KVStorageManager
  │   │   ├──simple_backend_manager.py  # AsyncSimpleStorageManager
  │   │   ├──yuanrong_manager.py        # YuanrongStorageManager
  │   │   ├──mooncake_manager.py        # MooncakeStorageManager
  │   │   └──factory.py                 # TransferQueueStorageManagerFactory
  │   └── clients/                      # Clients are lower level interfaces that directly manipulate the target storage backend.
  │   │   ├── __init__.py
  │   │   ├── base.py                   # TransferQueueStorageKVClient
  │   │   ├── yuanrong_client.py        # YuanrongStorageClient
  │   │   ├── mooncake_client.py        # MooncakeStorageClient
  │   │   ├── ray_storage_client.py     # RayStorageClient
  │   │   └── factory.py                # TransferQueueStorageClientFactory
```

To integrate TransferQueue with a custom storage backend, start by implementing a subclass that inherits from `TransferQueueStorageManager`. This subclass acts as an adapter between the TransferQueue system and the target storage backend. For KV-based storage backends, you can simply inherit from `KVStorageManager`, which can serve as the general manager for all KV-based backends.

Distributed storage backends often come with their own native clients serving as the interface of the storage system. In such cases, a low-level adapter for this client can be written, following the examples provided in the `storage/clients` directory.

Factory classes are provided for both `StorageManager` and `StorageClient` to facilitate easy integration. Adding necessary descriptions of required parameters in the factory class helps enhance the overall user experience.

<h2 id="contribution"> ✏️ Contribution Guide</h2>

<span style="color: #FF0000;">**Contributions are warmly welcome!**</span>

New ideas, feature suggestions, and user experience feedback are all encouraged—feel free to submit issues or PRs. We will respond as soon as possible.

We recommend using pre-commit for better code format.

```bash
# install pre-commit
pip install pre-commit

# run the following command in your repo folder, then fix the check before committing your code
pre-commit install && pre-commit run --all-files --show-diff-on-failure --color=always
```

<h2 id="citation">📑 Citation</h2>
Please kindly cite our paper if you find this repo is useful:

```bibtex
@article{han2025asyncflow,
  title={AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training},
  author={Han, Zhenyu and You, Ansheng and Wang, Haibo and Luo, Kui and Yang, Guang and Shi, Wenqi and Chen, Menglong and Zhang, Sicheng and Lan, Zeshun and Deng, Chunshi and others},
  journal={arXiv preprint arXiv:2507.01663},
  year={2025}
}
```