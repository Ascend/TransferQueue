# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings(
    action="ignore",
    message=r"The PyTorch API of nested tensors is in prototype stage*",
    category=UserWarning,
    module=r"torch\.nested",
)

warnings.filterwarnings(
    action="ignore",
    message=r"Tip: In future versions of Ray, Ray will no longer override accelerator visible "
    r"devices env var if num_gpus=0 or num_gpus=None.*",
    category=FutureWarning,
    module=r"ray\._private\.worker",
)


import ray  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402

# Configure Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"

if not ray.is_initialized():
    ray.init(namespace="TransferQueueTutorial")


def demonstrate_data_workflow():
    """
    Demonstrate basic data workflow: put → get → clear.
    """
    print("=" * 80)
    print("Data Workflow Demo: put → get → clear")
    print("=" * 80)

    # Step 1: Put data
    print("[Step 1] Putting data into TransferQueue...")

    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ]
    )
    attention_mask = torch.ones_like(input_ids)

    data_batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=input_ids.size(0),
    )

    print(f"  Created {data_batch.batch_size[0]} samples")
    partition_id = "tutorial_partition_0"
    tq.put(data=data_batch, partition_id=partition_id)
    print(f"  ✓ Data written to partition: {partition_id}")

    # Step 2: Get metadata
    print("[Step 2] Requesting data metadata...")
    batch_meta = tq.get_meta(
        data_fields=["input_ids", "attention_mask"],
        batch_size=data_batch.batch_size[0],
        partition_id=partition_id,
        task_name="tutorial_task",
    )
    print(f"  ✓ Got metadata: {len(batch_meta)} samples")
    print(f"    Global indexes: {batch_meta.global_indexes}")

    # Step 3: Get actual data
    print("[Step 3] Retrieving actual data...")
    retrieved_data = tq.get_data(batch_meta)
    print("  ✓ Data retrieved successfully")
    print(f"    Keys: {list(retrieved_data.keys())}")

    # Step 4: Verify
    print("[Step 4] Verifying data integrity...")
    assert torch.equal(retrieved_data["input_ids"], input_ids)
    assert torch.equal(retrieved_data["attention_mask"], attention_mask)
    print("  ✓ Data matches original!")

    # Step 5: Clear
    print("[Step 5] Clearing partition... (you may also use clear_samples() to clear specific samples)")
    tq.clear_partition(partition_id=partition_id)
    print("  ✓ Partition cleared")


def demonstrate_storage_backend_options():
    """
    Show different storage backend options.
    """
    print("=" * 80)
    print("Storage Backend Options")
    print("=" * 80)

    print("TransferQueue supports multiple storage backends:")
    print("1. SimpleStorage (default)")
    print("   - In-memory storage, fast and simple")
    print("   - Leveraging ZMQ for communication, with zero-copy serialization and transfer")
    print("   - No extra dependencies, good for development and testing")

    print("2. Yuanrong")
    print("   - Ascend native distributed storage solution")
    print("   - Hierarchical storage interfaces including HBM/DRAM/SSD")

    print("3. MooncakeStore (on the way)")
    print("   - Support multiple transmission protocols")
    print("   - RDMA between DRAM and HBM")

    print("4. Ray RDT (on the way)")
    print("   - Leverage Ray's distributed object store to store data")

    print("5. Custom Storage Backends")
    print("   - Implement your own storage manager by inheriting from `TransferQueueStorageManager` base class")
    print("   - For KV based storage, you only need to provide a storage client and integrate with `KVStorageManager`")


def main():
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 1: Core Components Introduction
    
        This script introduces the three core components of TransferQueue:
        1. TransferQueueController - Manages all the metadata and tracks the production and consumption states
        2. StorageBackend - Pluggable distributed storage backend that holds the actual data
        3. TransferQueueClient - Client interface for reading/writing data (user-facing API)
    
        Key Concepts:
        - Data is organized into logical partitions (e.g., "train", "val")
        - Each sample has multiple fields, with a global index for identification
        - Controller maintains production/consumption state tracking
        - Client is the main interface users interact with
        """
        )
    )
    print("=" * 80)

    try:
        print("Setting up TransferQueue...")
        tq.init()

        print("Demonstrating the user workflow...")
        demonstrate_data_workflow()

        demonstrate_storage_backend_options()

        print("=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("Key Takeaways:")
        print("1. TransferQueue has 3 core components:")
        print("   - Controller: Manages data production/consumption state")
        print("   - StorageBackend: Persists actual data")
        print("   - Client: User-facing API (what you use)")
        print("2. Client is the main interface users interact with")
        print("3. You can swap out different storage backends easily")

        # Cleanup
        tq.close()
        ray.shutdown()
        print("\n✓ Cleanup complete")

    except Exception as e:
        print(f"Error during tutorial: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
