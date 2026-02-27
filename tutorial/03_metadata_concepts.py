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
import uuid
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


import numpy as np  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402
from transfer_queue.metadata import BatchMeta  # noqa: E402

# Configure Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"


def demonstrate_batch_meta_schema():
    """
    Demonstrate BatchMeta field_schema - field-level metadata for the batch.
    After the columnar refactoring, field metadata is stored once at the batch level
    (not per-sample). This is the O(F) optimized representation.
    """
    print("=" * 80)
    print("BatchMeta field_schema - Field-Level Metadata (O(F) columnar storage)")
    print("=" * 80)

    print("field_schema stores metadata for each field ONCE per batch (not per sample):")
    print("- dtype: Data type (torch.float32, torch.int64, etc.)")
    print("- shape: Shape of ONE sample's data (NO batch dimension)")
    print("- is_nested: Whether the field uses nested/ragged tensors")
    print("- is_non_tensor: Whether the field is non-tensor data")

    # Example 1: Create a field schema entry for input_ids (analogous to old FieldMeta)
    print("[Example 1] Creating field schema entry for input_ids...")
    batch = BatchMeta(
        global_indexes=[0, 1, 2],
        partition_ids=["train_0"] * 3,
        field_schema={
            "input_ids": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
        },
    )
    print("✓ Created: BatchMeta with field 'input_ids'")
    print(f"  input_ids schema: {batch.field_schema['input_ids']}")
    print(f"  Is ready: {batch.is_ready}")
    print("  Note: Shape (512,) means ONE sample has 512 tokens (no batch dimension)")

    # Example 2: Create a field schema entry for attention_mask
    print("[Example 2] Creating field schema entry for attention_mask...")
    batch2 = BatchMeta(
        global_indexes=[0, 1, 2],
        partition_ids=["train_0"] * 3,
        field_schema={
            "attention_mask": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
        },
    )
    print("✓ Created: BatchMeta with field 'attention_mask'")
    print(f"  attention_mask schema: {batch2.field_schema['attention_mask']}")
    print(f"  Is ready: {batch2.is_ready}")

    # Example 3: Check field readiness via is_ready and production_status
    print("[Example 3] Checking field readiness...")
    ready_batch = BatchMeta(
        global_indexes=[0, 1, 2],
        partition_ids=["train_0"] * 3,
        field_schema={
            "input_ids": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
            "attention_mask": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
        },
        production_status=np.array([1, 1, 1], dtype="int8"),  # 1 = READY_FOR_CONSUME
    )
    print(f"  input_ids field exists: {'input_ids' in ready_batch.field_schema}")
    print(f"  attention_mask field exists: {'attention_mask' in ready_batch.field_schema}")
    print(f"  not-ready batch is_ready: {batch.is_ready}")
    print(f"  ready batch is_ready:     {ready_batch.is_ready}")

    # Example 4: Access per-sample view and individual field schema by key
    print("[Example 4] Accessing sample view and individual field by key...")
    view = ready_batch.samples[0]
    print(f"  batch.samples[0].fields -> {view.fields}")
    print(f"  batch.samples[0].fields['input_ids'] -> {view.fields['input_ids']}")
    print(f"  batch.samples[0].fields['input_ids']['dtype'] -> {view.fields['input_ids']['dtype']}")
    print("  Note: view.fields returns the shared field_schema dict (same for all samples)")
    print(f"  For partition_id: use batch.partition_ids[0] = '{ready_batch.partition_ids[0]}'")
    print(f"  For global_index: use batch.global_indexes[0] = {ready_batch.global_indexes[0]}")


def demonstrate_batch_meta_construction():
    """
    Demonstrate how to construct BatchMeta directly and operate on it
    (analogous to old SampleMeta operations: add_fields, select_fields, union).
    """
    print("=" * 80)
    print("BatchMeta Construction & Operations")
    print("=" * 80)

    print("BatchMeta now uses a columnar layout:")
    print("- global_indexes: list[int] - unique IDs across ALL partitions")
    print("- partition_ids: list[str] - which partition each sample belongs to")
    print("- field_schema: dict[str, dict] - field metadata (stored ONCE, not per-sample)")

    # Example 1: Manually create a BatchMeta (analogous to old SampleMeta construction)
    print("[Example 1] Creating a BatchMeta with input_ids and attention_mask...")
    batch = BatchMeta(
        global_indexes=[0, 1, 2, 3, 4],
        partition_ids=["train_0"] * 5,
        field_schema={
            "input_ids": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
            "attention_mask": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
        },
    )
    print(f"✓ Created: {len(batch)} samples")
    print(f"  Partition IDs: {batch.partition_ids}")
    print(f"  Global indexes: {batch.global_indexes}")
    print(f"  Fields: {batch.field_names}")
    print(f"  Is ready: {batch.is_ready}")

    # Example 2: add_fields - add new fields from real tensor data (analogous to sample.add_fields)
    print("[Example 2] Adding new fields via add_fields(TensorDict)...")
    new_data = TensorDict(
        {
            "responses": torch.randint(0, 1000, (5, 128)),
            "log_probs": torch.randn(5, 128),
        },
        batch_size=5,
    )
    batch.add_fields(new_data)  # infers dtype/shape from actual tensors, sets all ready
    print("✓ Added fields: ['responses', 'log_probs']")
    print(f"  Now has fields: {batch.field_names}")
    print(f"  Is ready: {batch.is_ready}  (add_fields sets all to READY_FOR_CONSUME by default)")

    # Example 3: select_fields - select specific fields (analogous to sample.select_fields)
    print("[Example 3] Selecting specific fields...")
    selected = batch.select_fields(["input_ids", "responses"])
    print(f"✓ Selected fields: {selected.field_names}")
    print(f"  Original fields: {batch.field_names}")

    # Example 4: union - merge two batches with different global_indexes (new columnar semantics)
    print("[Example 4] Unioning two batches with different global_indexes...")
    print("  IMPORTANT: new union semantics = concat unique samples (by global_index)")
    batch_a = BatchMeta(
        global_indexes=[0, 1, 2],
        partition_ids=["train_0"] * 3,
        field_schema={
            "input_ids": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
            "attention_mask": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
        },
    )
    batch_b = BatchMeta(
        global_indexes=[2, 3, 4],  # global_index=2 overlaps with batch_a
        partition_ids=["train_0"] * 3,
        field_schema={
            "input_ids": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
            "attention_mask": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
        },
    )
    print(f"  BatchA indexes: {batch_a.global_indexes}")
    print(f"  BatchB indexes: {batch_b.global_indexes}")
    unioned = batch_a.union(batch_b)
    print(f"✓ Union result indexes: {unioned.global_indexes}  (global_index=2 deduplicated)")

    # Example 5: Empty BatchMeta
    print("[Example 5] Creating an empty BatchMeta (for initializing before data arrives)...")
    empty = BatchMeta.empty()
    print(f"✓ Empty BatchMeta: size={empty.size}, is_ready={empty.is_ready}")


def demonstrate_batch_meta():
    """
    Demonstrate BatchMeta - describes a batch of samples with operations.
    """
    print("=" * 80)
    print("BatchMeta - Operations on Batch")
    print("=" * 80)

    print("BatchMeta represents a collection of data samples with operations:")
    print("- global_indexes: list of global sample indices")
    print("- partition_ids: list of partition IDs per sample")
    print("- field_schema: field-level metadata (stored once at batch level)")
    print("- Operations: chunk, concat, union, select_samples, select_fields, reorder")

    # Helper to create a BatchMeta
    def make_batch(global_indexes, fields=None):
        if fields is None:
            fields = ["input_ids", "attention_mask", "responses"]
        schema = {
            "input_ids": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
            "attention_mask": {"dtype": torch.int64, "shape": (512,), "is_nested": False, "is_non_tensor": False},
            "responses": {"dtype": torch.int64, "shape": (128,), "is_nested": False, "is_non_tensor": False},
        }
        return BatchMeta(
            global_indexes=global_indexes,
            partition_ids=["train_0"] * len(global_indexes),
            field_schema={k: v for k, v in schema.items() if k in fields},
        )

    # Example 1: Create a batch
    print("[Example 1] Creating a BatchMeta...")
    batch = make_batch(list(range(5)))
    print(f"✓ Created batch with {len(batch)} samples")
    print(f"  Global indexes: {batch.global_indexes}")
    print(f"  Field names: {batch.field_names}")
    print(f"  Size: {batch.size}")

    # Example 2: Add extra_info (batch-level)
    print("[Example 2] Adding batch-level information through extra_info...")
    batch.extra_info["epoch"] = 1
    batch.extra_info["batch_idx"] = 0
    print(f"✓ Extra info: {batch.get_all_extra_info()}")

    # Example 3: update_custom_meta (list aligned with global_indexes)
    print("[Example 3] Adding sample-level information through custom_meta...")
    batch.update_custom_meta(
        [
            {"uid": "prompt@0", "session_id": "session@0", "model_version": "epoch@0"},
            {"uid": "prompt@1", "session_id": "session@0", "model_version": "epoch@0"},
            {"uid": "prompt@2", "session_id": "session@0", "model_version": "epoch@0"},
            {"uid": "prompt@3", "session_id": "session@0", "model_version": "epoch@0"},
            {"uid": "prompt@4", "session_id": "session@0", "model_version": "epoch@0"},
        ]
    )
    print(f"✓ Custom meta: {batch.get_all_custom_meta()}")

    # Example 4: Chunk
    print("[Example 4] Chunking a batch into parts...")
    chunks = batch.chunk(3)
    print(f"✓ Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} samples, indexes={chunk.global_indexes}")

    # Example 5: select_fields
    print("[Example 5] Selecting specific fields...")
    selected_batch = batch.select_fields(["input_ids", "responses"])
    print(f"✓ Selected fields: {selected_batch.field_names}")
    print(f"  Original fields: {batch.field_names}")

    # Example 6: select_samples
    print("[Example 6] Selecting specific samples...")
    selected_samples = batch.select_samples([0, 2, 4])
    print(f"✓ Selected samples at batch indices [0,2,4]: global_indexes={selected_samples.global_indexes}")

    # Example 7: reorder
    print("[Example 7] Reordering samples...")
    print(f"  Original order: {batch.global_indexes}")
    batch.reorder([4, 3, 2, 1, 0])
    print(f"  After reorder: {batch.global_indexes}")

    # Example 8: concat
    print("[Example 8] Concatenating batches...")
    batch1 = make_batch(list(range(3)))
    batch2 = make_batch(list(range(3, 6)))
    concatenated = BatchMeta.concat([batch1, batch2])
    print(f"✓ Concatenated {len(batch1)} + {len(batch2)} = {len(concatenated)} samples")
    print(f"  Global indexes: {concatenated.global_indexes}")
    print("  Note: concat combines multiple batches with SAME field structure into one larger batch")

    # Example 9: union (new semantics: concat unique samples, dedup by global_index)
    print("[Example 9] Unioning batches with overlapping global_indexes...")
    batch_a = make_batch(list(range(3)), fields=["input_ids", "attention_mask"])  # indexes [0,1,2]
    batch_b = make_batch(list(range(2, 5)), fields=["input_ids", "attention_mask"])  # indexes [2,3,4] — 2 overlaps!
    print(f"  BatchA fields: {batch_a.field_names}, indexes: {batch_a.global_indexes}")
    print(f"  BatchB fields: {batch_b.field_names}, indexes: {batch_b.global_indexes}")
    unioned = batch_a.union(batch_b)
    print(f"✓ Unioned: {unioned.global_indexes}  (global_index=2 deduplicated, result: [0,1,2,3,4])")
    print("  Note: union keeps self's copy when global_index overlaps")

    print("=" * 80)
    print("concat vs union:")
    print("  - concat: Combines multiple batches with SAME field structure into one larger batch")
    print("    Example: batch[0,1,2] concat batch[3,4,5] = batch[0,1,2,3,4,5]")
    print("  - union:  Merges two batches, deduplicating by global_index (keeps self's copy)")
    print("    Example: batch[0,1,2] union batch[2,3,4] = batch[0,1,2,3,4]")
    print("=" * 80)


def demonstrate_real_workflow():
    """
    Demonstrate a realistic workflow with actual TransferQueue interaction.
    """
    print("=" * 80)
    print("Real Workflow: Interacting with TransferQueue")
    print("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(namespace="TransferQueueTutorial")

    # Initialize TransferQueue
    tq.init()

    tq_client = tq.get_client()

    print("[Step 1] Putting data into TransferQueue...")
    input_ids = torch.randint(0, 1000, (8, 512))
    attention_mask = torch.ones(8, 512)

    data_batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=8,
    )

    partition_id = "demo_partition"
    batch_meta = tq_client.put(data=data_batch, partition_id=partition_id)
    print(f"✓ Put {data_batch.batch_size[0]} samples into partition '{partition_id}', got BatchMeta back {batch_meta}.")

    print("[Step 2] [Optional] Setting sample-level custom_meta...")

    custom_meta = [
        {"uid": uuid.uuid4().hex[:4], "session_id": uuid.uuid4().hex[:4], "model_version": 0}
        for _ in range(batch_meta.size)
    ]
    batch_meta.update_custom_meta(custom_meta)
    print(f"✓ Set custom_meta into BatchMeta: {batch_meta.get_all_custom_meta()}")

    tq_client.set_custom_meta(batch_meta)
    print("✓ Successful to store custom_meta into TQ controller. Now you can retrieve the custom_meta from anywhere.")

    print("[Step 3] Try to get metadata from TransferQueue from other places...")
    batch_meta = tq_client.get_meta(
        data_fields=["input_ids", "attention_mask"],
        batch_size=8,
        partition_id=partition_id,
        task_name="demo_task",  # TransferQueueController prevents same task_name from getting data repeatedly
    )
    print("✓ Got BatchMeta from TransferQueue:")
    print(f"  Number of samples: {len(batch_meta)}")
    print(f"  Global indexes: {batch_meta.global_indexes}")
    print(f"  Field names: {batch_meta.field_names}")
    print(f"  Partition IDs: {batch_meta.partition_ids}")
    print(f"  Sample view: fields={batch_meta.samples[0].fields}")
    print(f"  Custom Meta: {batch_meta.get_all_custom_meta()}")

    print("[Step 4] Retrieve samples with specific fields..")
    selected_meta = batch_meta.select_fields(["input_ids"])
    print("✓ Selected 'input_ids' field only:")
    print(f"  New field names: {selected_meta.field_names}")
    print(f"  Samples still have same global indexes: {selected_meta.global_indexes}")
    retrieved_data = tq_client.get_data(selected_meta)
    print(f"  Retrieved data keys: {list(retrieved_data.keys())}")

    print("[Step 5] Select specific samples from the retrieved BatchMeta...")
    partial_meta = batch_meta.select_samples([0, 2, 4, 6])
    print("✓ Selected samples at indices [0, 2, 4, 6]:")
    print(f"  New global indexes: {partial_meta.global_indexes}")
    print(f"  Number of samples: {len(partial_meta)}")
    retrieved_data = tq_client.get_data(partial_meta)
    print(f"  Retrieved data samples: {retrieved_data}, all the data samples: {data_batch}")

    print("[Step 6] Demonstrate chunk operation...")
    chunks = batch_meta.chunk(2)
    print(f"✓ Chunked into {len(chunks)} parts:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} samples, indexes={chunk.global_indexes}")
        chunk_data = tq_client.get_data(chunk)
        print(f"  Chunk {i}: Retrieved chunk data: {chunk_data}")

    # Cleanup
    tq_client.clear_partition(partition_id=partition_id)
    tq.close()
    ray.shutdown()
    print("✓ Partition cleared and resources cleaned up")


def main():
    """Main function to run the tutorial."""
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 3: Metadata System

        This script introduces the metadata system in TransferQueue, which tracks
        the structure and state of data:

        1. BatchMeta - The central metadata object for a collection of data samples.
           Uses a columnar layout: field metadata is stored ONCE at the batch level (O(F)),
           not per-sample (was O(B×F) in the old design).

        Key Concepts:
        - BatchMeta stores global_indexes, partition_ids, and field_schema directly
        - field_schema: dict[field_name, {dtype, shape, is_nested, is_non_tensor}]
        - custom_meta: list[dict] aligned with global_indexes (one dict per sample)
        - Metadata operations: chunk, concat, union, select_fields, select_samples, reorder
        - batch.samples[i] returns a lazy view with .fields -> field_schema (read-only)
    """
        )
    )
    print("=" * 80)

    try:
        demonstrate_batch_meta_schema()
        demonstrate_batch_meta_construction()
        demonstrate_batch_meta()
        demonstrate_real_workflow()

        print("=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("Key Takeaways:")
        print("1. BatchMeta uses columnar storage: field metadata stored once, not per-sample")
        print("2. Construct BatchMeta with: BatchMeta(global_indexes=[...], partition_ids=[...], field_schema={...})")
        print("3. BatchMeta operations: chunk, concat, union, select_fields, select_samples, reorder")
        print("4. extra_info is batch-level; custom_meta is sample-level (list[dict])")
        print("5. Store custom_meta via TQ controller: tq_client.set_custom_meta(batch_meta)")

        # Cleanup
        ray.shutdown()
        print("\n✓ Cleanup complete")

    except Exception as e:
        print(f"Error during tutorial: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
