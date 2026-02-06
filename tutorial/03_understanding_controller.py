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


def demonstrate_partition_isolation():
    """Feature 1: Different partitions are isolated - data doesn't interfere."""
    print("=" * 80)
    print("Feature 1: Partition Isolation")
    print("=" * 80)

    print("\nDifferent partitions are completely isolated - data doesn't interfere between partitions")

    if not ray.is_initialized():
        ray.init(namespace="TransferQueueTutorial")

    tq.init()

    # Partition 1: Training data
    print("\n[Partition 1] Putting training data...")
    train_data = TensorDict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([0, 1]),
        },
        batch_size=2,
    )
    tq.put(data=train_data, partition_id="train")
    print("  ✓ Training data added to 'train' partition")

    # Partition 2: Validation data
    print("\n[Partition 2] Putting validation data...")
    val_data = TensorDict(
        {
            "input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([2, 3]),
        },
        batch_size=2,
    )
    tq.put(data=val_data, partition_id="val")
    print("  ✓ Validation data added to 'val' partition")

    # Get from train partition
    print("\n[Retrieving from 'train' partition]")
    train_meta = tq.get_meta(
        data_fields=["input_ids", "labels"], batch_size=2, partition_id="train", task_name="train_task"
    )

    retrieved_train_data = tq.get_data(train_meta)
    print(f"  ✓ Got BatchMeta={train_meta} from partition 'train'")
    print(f"  ✓ Retrieved Data: input_ids={retrieved_train_data['input_ids']}, labels={retrieved_train_data['labels']}")

    # Get from val partition
    print("\n[Retrieving from 'val' partition]")
    val_meta = tq.get_meta(data_fields=["input_ids", "labels"], batch_size=2, partition_id="val", task_name="val_task")
    retrieved_val_data = tq.get_data(val_meta)
    print(f"  ✓ Got BatchMeta={val_meta} from partition 'val'")
    print(f"  ✓ Retrieved Data: input_ids={retrieved_val_data['input_ids']}, labels={retrieved_val_data['labels']}")

    print("\n[Verification]")
    print("  ✓ Data isolation: 'train' and 'val' partitions are completely independent")

    # Cleanup
    tq.clear_partition(partition_id="train")
    tq.clear_partition(partition_id="val")
    tq.close()
    ray.shutdown()


def demonstrate_dynamic_expansion():
    """Feature 2: Dynamic expansion - can add rows and columns anytime."""
    print("\n" + "=" * 80)
    print("Feature 2: Dynamic Expansion - Flexible Row/Column Addition")
    print("=" * 80)

    print("\nPartitions dynamically expand to accommodate new data (rows and columns)")

    if not ray.is_initialized():
        ray.init(namespace="TransferQueueTutorial")

    tq.init()

    # Add first batch with 2 samples, 2 fields
    print("\n[Step 1] Adding initial data (2 samples, 2 fields)...")
    data1 = TensorDict(
        {
            "field1": torch.tensor([[1, 2], [3, 4]]),
            "field2": torch.tensor([[5, 6], [7, 8]]),
        },
        batch_size=2,
    )
    meta1 = tq.put(data=data1, partition_id="dynamic")
    print("  ✓ Added 2 samples")
    print(f"  ✓ Got BatchMeta: {meta1} samples")

    # Add second batch with more samples (expanding rows)
    print("\n[Step 2] Adding more samples (expanding rows)...")
    data2 = TensorDict(
        {
            "field1": torch.tensor([[9, 10], [11, 12], [13, 14]]),
            "field2": torch.tensor([[15, 16], [17, 18], [19, 20]]),
        },
        batch_size=3,
    )
    meta2 = tq.put(data=data2, partition_id="dynamic")

    all_meta = tq.get_meta(
        data_fields=["field1", "field2"], batch_size=5, partition_id="dynamic", task_name="dynamic_task"
    )
    print("  ✓ Added 3 more samples (total: 5)")
    print(f"  ✓ Got BatchMeta {meta2} for newly put data.")
    print(f"  ✓ All BatchMeta in controller is {all_meta}")

    # Add new field (expanding columns)
    print("\n[Step 3] Adding new field (expanding columns)...")
    data3 = TensorDict(
        {
            "field3": torch.tensor([[25, 26], [27, 28]]),
        },
        batch_size=2,
    )
    meta3 = tq.put(data=data3, metadata=meta1)
    print("  ✓ Added 2 samples with new field 'field3'")
    print(f"  ✓ Got BatchMeta: {meta3} for newly put data with new field")

    print("\n[Verification]")
    print("  ✓ Rows auto-expand: Can add more samples anytime")
    print("  ✓ Columns auto-expand: Can add new fields anytime")

    # Cleanup
    tq.clear_partition(partition_id="dynamic")
    tq.close()
    ray.shutdown()


def demonstrate_default_consumption_sample_strategy():
    """Feature 3: Default sequential sampling without replacement."""
    print("\n" + "=" * 80)
    print("Feature 3: Default Sampling Strategy for Controller - No Duplicate, Sequential Samples")
    print("=" * 80)

    if not ray.is_initialized():
        ray.init(namespace="TransferQueueTutorial")

    tq.init()

    # Add 6 samples
    print("\n[Setup] Adding 6 samples...")
    all_data = TensorDict(
        {
            "data": torch.tensor([[i] for i in range(6)]),
        },
        batch_size=6,
    )
    tq.put(data=all_data, partition_id="sampling")
    print("  ✓ 6 samples added")

    # First get - should get samples 0,1,2
    print("\n[Task A, Get 1] Requesting 3 samples...")
    meta1 = tq.get_meta(data_fields=["data"], batch_size=3, partition_id="sampling", task_name="A")
    print(f"  ✓ Got samples: {meta1.global_indexes}")

    # Second get - should get samples 3,4,5 (no duplicates!)
    print("\n[Task A, Get 2] Requesting 3 more samples...")
    meta2 = tq.get_meta(data_fields=["data"], batch_size=3, partition_id="sampling", task_name="A")
    print(f"  ✓ Got samples: {meta2.global_indexes}")

    # Third get - should get samples 0,1
    print("\n[Task B, Get 1] Requesting 2 samples...")
    meta3 = tq.get_meta(data_fields=["data"], batch_size=2, partition_id="sampling", task_name="B")
    print(f"  ✓ Got samples: {meta3.global_indexes}")

    print("\n[Verification]")
    print("  ✓ Same task_name: Sequential sampling, no duplicates")
    print("  ✓ First get (Task A): samples 0,1,2")
    print("  ✓ Second get (Task A): samples 3,4,5")
    print("  ✓ Different task_name: Independent consumption with other tasks")
    print("  ✓ Third get (Task B): samples 0,1")

    # Cleanup
    tq.clear_partition(partition_id="sampling")
    tq.close()
    ray.shutdown()


def main():
    """Main function to run the tutorial."""
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 3: Understanding TransferQueueController

        This script demonstrates TransferQueueController's key features:

        1. Partition Isolation - Different partition_id creates isolated virtual partitions
        2. Dynamic Expansion - Auto-expand rows and columns, get BatchMeta anytime
        3. Sequential Sampling - Same task_name gets non-duplicate samples sequentially by default
        4. Independent Tasks - Different task_name have independent consumption tracking

        Key Concepts:
        - Partition-based organization with complete isolation
        - Dynamic scaling without pre-allocation
        - Sample strategy prevents duplicate consumption
        - Task-specific consumption tracking
        """
        )
    )
    print("=" * 80)

    try:
        demonstrate_partition_isolation()
        demonstrate_dynamic_expansion()
        demonstrate_default_consumption_sample_strategy()

        print("\n" + "=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("Key Takeaways:")
        print("1. Partitions are completely isolated - different partition_id = independent data")
        print("2. Dynamic expansion - add rows/columns anytime, get fresh BatchMeta")
        print("3. Sequential sampling - same task_name gets unique samples in order by default")
        print("4. Independent consumption - different task_name don't interfere")

    except Exception as e:
        print(f"Error during tutorial: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
