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


import ray  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tensordict import TensorDict  # noqa: E402

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from transfer_queue import (  # noqa: E402
    SimpleStorageUnit,
    TransferQueueClient,
    TransferQueueController,
    process_zmq_server_info,
)
from transfer_queue.metadata import BatchMeta, FieldMeta, SampleMeta  # noqa: E402
from transfer_queue.utils.enum_utils import ProductionStatus  # noqa: E402

# Configure Ray
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"


def demonstrate_field_meta():
    """
    Demonstrate FieldMeta - specific data fields of each training sample.
    """
    print("=" * 80)
    print("FieldMeta - Specific data fields of each training sample")
    print("=" * 80)

    print("FieldMeta represents a single field in ONE sample:")
    print("- name: Field identifier ('Prompt', 'Response', etc.)")
    print("- dtype: Data type (torch.float32, torch.int64, etc.)")
    print("- shape: Shape of ONE sample's data (NO batch dimension)")
    print("- production_status: Whether data is ready (has been produced and written to the TQ backend)")

    # Example 1: Create a field for input_ids
    print("[Example 1] Manually creating FieldMeta for input_ids...")
    input_ids_field = FieldMeta(
        name="input_ids",
        dtype=torch.int64,
        shape=(512,),  # Sequence length for ONE sample
        production_status=ProductionStatus.READY_FOR_CONSUME,
    )
    print(f"✓ Created: {input_ids_field}")
    print(f"  Is ready: {input_ids_field.is_ready}")
    print("  Note: Shape (512,) means ONE sample has 512 tokens (no batch dimension)")

    # Example 2: Create a field for attention_mask
    print("[Example 2] Creating FieldMeta for attention_mask...")
    attention_mask_field = FieldMeta(
        name="attention_mask",
        dtype=torch.int64,
        shape=(512,),  # Sequence length for ONE sample
        production_status=ProductionStatus.NOT_PRODUCED,
    )
    print(f"✓ Created: {attention_mask_field}")
    print(f"  Is ready: {attention_mask_field.is_ready}")

    # Example 3: Check field readiness
    print("[Example 3] Checking field readiness...")
    print(f"  input_ids ready: {input_ids_field.is_ready}")
    print(f"  attention_mask ready: {attention_mask_field.is_ready}")


def demonstrate_sample_meta():
    """
    Demonstrate SampleMeta - describes a single data sample.
    """
    print("=" * 80)
    print("SampleMeta - Describing a Single Data Sample")
    print("=" * 80)

    print("SampleMeta represents ONE data sample:")
    print("- partition_id: Which partition the sample belongs to")
    print("- global_index: Unique identifier across ALL partitions")
    print("- fields: Dict of FieldMeta objects (describing each field of THIS sample)")

    # Example 1: Manually create a sample
    print("[Example 1] Creating a SampleMeta...")
    fields = {
        "input_ids": FieldMeta("input_ids", torch.int64, (512,)),
        "attention_mask": FieldMeta("attention_mask", torch.int64, (512,)),
    }
    sample = SampleMeta(partition_id="train_0", global_index=0, fields=fields)
    print(f"✓ Created: {sample}")
    print(f"  Partition: {sample.partition_id}")
    print(f"  Global index: {sample.global_index}")
    print(f"  Fields: {sample.field_names}")
    print(f"  Is ready: {sample.is_ready}")

    # Example 2: Manually add fields to a sample
    print("[Example 2] Adding fields to a sample...")
    new_fields = {
        "responses": FieldMeta("responses", torch.int64, (128,)),
        "log_probs": FieldMeta("log_probs", torch.float32, (128,)),
    }
    sample.add_fields(new_fields)
    print(f"✓ Added fields: {list(new_fields.keys())}")
    print(f"  Now has fields: {sample.field_names}")
    print(f"  Is ready: {sample.is_ready}")

    # Example 3: Select specific fields
    print("[Example 3] Selecting specific fields...")
    selected_sample = sample.select_fields(["input_ids", "responses"])
    print(f"✓ Selected fields: {selected_sample.field_names}")
    print(f"  Original fields: {sample.field_names}")

    # Example 4: Union two samples
    print("[Example 4] Unioning two samples...")
    print("  IMPORTANT: Union requires samples to have IDENTICAL partition_id and global_index!")
    sample1 = SampleMeta(
        partition_id="train_0",
        global_index=5,
        fields={
            "input_ids": FieldMeta("input_ids", torch.int64, (512,)),
            "attention_mask": FieldMeta("attention_mask", torch.int64, (512,)),
        },
    )
    sample2 = SampleMeta(
        partition_id="train_0",
        global_index=5,  # Same global index!
        fields={
            "responses": FieldMeta("responses", torch.int64, (128,)),
            "log_probs": FieldMeta("log_probs", torch.float32, (128,)),
        },
    )
    print(f"  Sample1: partition={sample1.partition_id}, global_index={sample1.global_index}")
    print(f"  Sample2: partition={sample2.partition_id}, global_index={sample2.global_index}")

    try:
        unioned = sample1.union(sample2)
        print("✓ Union successful!")
        print(f"  Unioned fields: {unioned.field_names}")
        print(f"  Global index preserved: {unioned.global_index}")
    except ValueError as e:
        print(f"✗ Union failed: {e}")


def demonstrate_batch_meta():
    """
    Demonstrate BatchMeta - describes a batch of samples with operations.
    """
    print("=" * 80)
    print("BatchMeta - Describing a Batch of Samples")
    print("=" * 80)

    print("BatchMeta represents a collection of samples:")
    print("- samples: List of SampleMeta objects")
    print("- extra_info: Additional batch-level information")
    print("- Provides operations: chunk, concat, union, select, reorder")

    # Example 1: Manually create a batch
    print("[Example 1] Creating a BatchMeta...")
    fields = {
        "input_ids": FieldMeta("input_ids", torch.int64, (512,)),
        "attention_mask": FieldMeta("attention_mask", torch.int64, (512,)),
        "responses": FieldMeta("responses", torch.int64, (128,)),
    }
    samples = [SampleMeta(partition_id="train_0", global_index=i, fields=fields) for i in range(5)]
    batch = BatchMeta(samples=samples)
    print(f"✓ Created batch with {len(batch)} samples")
    print(f"  Global indexes: {batch.global_indexes}")
    print(f"  Field names: {batch.field_names}")
    print(f"  Size: {batch.size}")

    # Example 2: Add extra_info
    print("[Example 2] Adding batch-level information through extra_info...")
    print("Note: The extra info will not be stored into TransferQueueController.")
    batch.extra_info["epoch"] = 1
    batch.extra_info["batch_idx"] = 0
    print(f"✓ Extra info: {batch.get_all_extra_info()}")

    print("[Example 3] Adding sample-level information through custom_meta...")
    batch.set_custom_meta(
        global_index=0, meta_dict={"uid": "prompt@0", "session_id": "session@0", "model_version": "epoch@0"}
    )
    batch.update_custom_meta({1: {"uid": "prompt@1", "session_id": "session@0", "model_version": "epoch@0"}})
    print(f"✓ Extra info: {batch.get_all_custom_meta()}")

    # Example 4: Chunk a batch
    print("[Example 4] Chunking a batch into parts...")
    chunks = batch.chunk(3)
    print(f"✓ Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} samples, indexes={chunk.global_indexes}")

    # Example 5: Select specific fields
    print("[Example 5] Selecting specific fields...")
    selected_batch = batch.select_fields(["input_ids", "responses"])
    print(f"✓ Selected fields: {selected_batch.field_names}")
    print(f"  Original fields: {batch.field_names}")

    # Example 6: Select specific samples
    print("[Example 6] Selecting specific samples...")
    selected_samples = batch.select_samples([0, 2, 4])
    print(f"✓ Selected samples at indexes: {selected_samples.global_indexes}")

    # Example 7: Reorder samples
    print("[Example 7] Reordering samples...")
    print(f"  Original order: {batch.global_indexes}")
    batch.reorder([4, 3, 2, 1, 0])
    print(f"  After reorder: {batch.global_indexes}")

    # Example 8: Concat batches
    print("[Example 8] Concatenating batches...")
    batch1 = BatchMeta(samples=[SampleMeta(partition_id="train_0", global_index=i, fields=fields) for i in range(3)])
    batch2 = BatchMeta(samples=[SampleMeta(partition_id="train_0", global_index=i, fields=fields) for i in range(3, 6)])
    concatenated = BatchMeta.concat([batch1, batch2])
    print(f"✓ Concatenated {len(batch1)} + {len(batch2)} = {len(concatenated)} samples")
    print(f"  Global indexes: {concatenated.global_indexes}")
    print("  Note: concat combines multiple batches into one (same structure)")

    # Example 9: Union batches
    print("[Example 9] Unioning batches (different fields, same samples)...")
    batch_with_input = BatchMeta(
        samples=[
            SampleMeta(
                partition_id="train_0",
                global_index=i,
                fields={
                    "input_ids": FieldMeta("input_ids", torch.int64, (512,)),
                    "attention_mask": FieldMeta("attention_mask", torch.int64, (512,)),
                },
            )
            for i in range(3)
        ]
    )
    batch_with_output = BatchMeta(
        samples=[
            SampleMeta(
                partition_id="train_0",
                global_index=i,
                fields={
                    "responses": FieldMeta("responses", torch.int64, (128,)),
                    "log_probs": FieldMeta("log_probs", torch.float32, (128,)),
                },
            )
            for i in range(3)
        ]
    )
    print(f"  Batch1 has fields: {batch_with_input.field_names}")
    print(f"  Batch2 has fields: {batch_with_output.field_names}")
    print(f"  Both have same samples (global_indexes: {batch_with_input.global_indexes})")

    unioned_batch = batch_with_input.union(batch_with_output)
    print("✓ Union successful!")
    print(f"  Unioned fields: {unioned_batch.field_names}")
    print("  Note: union merges fields from two batches with SAME samples (same global_indexes)")

    print("=" * 80)
    print("concat vs union:")
    print("  - concat: Combines multiple batches with SAME structure into one larger batch")
    print("    Example: batch1[0,1,2] + batch2[3,4,5] = batch[0,1,2,3,4,5]")
    print("  - union: Merges fields from two batches with IDENTICAL samples")
    print("    Example: batch1[0,1] with fields A + batch2[0,1] with fields B = batch[0,1] with fields A+B")
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
        ray.init()

    # Setup TransferQueue
    config = OmegaConf.create(
        {
            "num_data_storage_units": 2,
        }
    )

    storage_units = {}
    for i in range(config["num_data_storage_units"]):
        storage_units[i] = SimpleStorageUnit.remote(storage_unit_size=100)

    controller = TransferQueueController.remote()
    controller_info = process_zmq_server_info(controller)
    storage_unit_infos = process_zmq_server_info(storage_units)

    client = TransferQueueClient(
        client_id="TutorialClient",
        controller_info=controller_info,
    )

    tq_config = OmegaConf.create({}, flags={"allow_objects": True})
    tq_config.controller_info = controller_info
    tq_config.storage_unit_infos = storage_unit_infos
    config = OmegaConf.merge(tq_config, config)

    client.initialize_storage_manager(manager_type="AsyncSimpleStorageManager", config=config)

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
    batch_meta = client.put(data=data_batch, partition_id=partition_id)
    print(f"✓ Put {data_batch.batch_size[0]} samples into partition '{partition_id}', got BatchMeta back {batch_meta}.")

    print("[Step 2] [Optional] Setting sample-level custom_meta...")

    custom_meta = {
        global_index: {"uid": uuid.uuid4().hex[:4], "session_id": uuid.uuid4().hex[:4], "model_version": 0}
        for global_index in batch_meta.global_indexes
    }
    batch_meta.set_custom_meta(custom_meta)
    print(f"✓ Set custom_meta into BatchMeta: {batch_meta.get_all_custom_meta()}")

    client.set_custom_meta(batch_meta)
    print("✓ Successful to store custom_meta into TQ controller. Now you can retrieve the custom_meta from anywhere.")

    print("[Step 2] Try to get metadata from TransferQueue from other places...")
    batch_meta = client.get_meta(
        data_fields=["input_ids", "attention_mask"],
        batch_size=8,
        partition_id=partition_id,
        task_name="demo_task",  # TransferQueueController prevents same task_name from getting data repeatedly
    )
    print("✓ Got BatchMeta from TransferQueue:")
    print(f"  Number of samples: {len(batch_meta)}")
    print(f"  Global indexes: {batch_meta.global_indexes}")
    print(f"  Field names: {batch_meta.field_names}")
    print(f"  Partition ID: {batch_meta.samples[0].partition_id}")
    print(f"  Sample structure: {batch_meta.samples[0]}")
    print(f"  Custom Meta: {batch_meta.get_all_custom_meta()}")

    print("[Step 3] Retrieve samples with specific fields..")
    selected_meta = batch_meta.select_fields(["input_ids"])
    print("✓ Selected 'input_ids' field only:")
    print(f"  New field names: {selected_meta.field_names}")
    print(f"  Samples still have same global indexes: {selected_meta.global_indexes}")
    retrieved_data = client.get_data(selected_meta)
    print(f"  Retrieved data keys: {list(retrieved_data.keys())}")

    print("[Step 4] Select specific samples from the retrieved BatchMeta...")
    partial_meta = batch_meta.select_samples([0, 2, 4, 6])
    print("✓ Selected samples at indices [0, 2, 4, 6]:")
    print(f"  New global indexes: {partial_meta.global_indexes}")
    print(f"  Number of samples: {len(partial_meta)}")
    retrieved_data = client.get_data(partial_meta)
    print(f"  Retrieved data samples: {retrieved_data}, all the data samples: {data_batch}")

    print("[Step 5] Demonstrate chunk operation...")
    chunks = batch_meta.chunk(2)
    print(f"✓ Chunked into {len(chunks)} parts:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} samples, indexes={chunk.global_indexes}")
        chunk_data = client.get_data(chunk)
        print(f"  Chunk {i}: Retrieved chunk data: {chunk_data}")

    # Cleanup
    client.clear_partition(partition_id=partition_id)
    client.close()
    ray.shutdown()
    print("✓ Partition cleared and resources cleaned up")


def main():
    """Main function to run the tutorial."""
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 2: Metadata System

        This script introduces the metadata system in TransferQueue, which tracks
        the structure and state of data:

        1. FieldMeta - Describes a single field (name, dtype, shape, production status)
        2. SampleMeta - Describes a single data sample (partition_id, global_index, fields)
        3. BatchMeta - Describes a batch of samples (collection of SampleMeta with operations)

        Key Concepts:
        - Metadata tracks data structure without storing actual data
        - User can set their own custom metadata into BatchMeta, and use TQ controller to store them.
        - BatchMeta provides operations: chunk, concat, union, select, reorder...
        - Metadata is lightweight and can be passed around efficiently
        - Union requires samples to have identical partition_id and global_index
        """
        )
    )
    print("=" * 80)

    try:
        demonstrate_field_meta()
        demonstrate_sample_meta()
        demonstrate_batch_meta()
        demonstrate_real_workflow()

        print("=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("Key Takeaways:")
        print("1. FieldMeta describes individual data fields (NO batch dimension in shape)")
        print("2. SampleMeta describes a single data sample")
        print("3. BatchMeta manages collections of samples with operations")
        print("4. Metadata operations: chunk, concat, union, select, reorder... You can retrieve subsets easily!")
        print("5. concat combines batches; union merges fields of same samples")

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
