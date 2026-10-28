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


def demonstrate_kv_api():
    """
    Demonstrate the Key-Value (KV) semantic API:
    kv_put & kv_batch_put -> kv_update -> kv_list -> kv_batch_get -> kv_clear
    """
    print("=" * 80)
    print("Key-Value Semantic API Demo: kv_put/kv_batch_put → kv_update → kv_list → kv_batch_get → kv_clear")
    print("=" * 80)

    # Step 1: Put a single key-value pair with kv_put
    print("[Step 1] Putting a single sample with kv_put...")

    # Define the data content (The "Value")
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    single_sample = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        batch_size=input_ids.size(0),
    )

    partition_id = "Train"
    # Use a meaningful string key instead of an auto-increment integer
    key = "0_0"  # User-defined key: "{uid}_{session_id}"
    tag = {"global_steps": 0, "status": "running", "model_version": 0}

    print(f"  Inserting Key: {key}")
    print(f"  Fields (Columns): {list(single_sample.keys())}")
    print(f"  Tag (Metadata): {tag}")

    tq.kv_put(key=key, partition_id=partition_id, fields=single_sample, tag=tag)
    print("  ✓ kv_put success.")

    # Step 2: Put multiple key-value pairs with kv_batch_put
    print("\n[Step 2] Putting batch data with kv_batch_put...")

    batch_input_ids = torch.tensor(
        [
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
        ]
    )
    batch_attention_mask = torch.ones_like(batch_input_ids)

    data_batch = TensorDict(
        {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        },
        batch_size=batch_input_ids.size(0),
    )

    keys = ["1_0", "1_1", "1_2", "2_0"]  # 4 keys for 4 samples
    tags = [{"global_steps": 1, "status": "running", "model_version": 1} for _ in range(len(keys))]

    print(f"  Inserting batch of {len(keys)} samples.")
    print(f"  Fields (Columns): {list(data_batch.keys())}")
    print(f"  Tag (Metadata): {tags}")
    tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=data_batch, tags=tags)
    print("  ✓ kv_batch_put success.")

    # Step 3: Append additional fields to existing samples
    print("\n[Step 3] Appending new fields (Columns) to existing samples...")

    batch_response = torch.tensor(
        [
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    response_batch = TensorDict(
        {
            "response": batch_response,
        },
        batch_size=batch_response.size(0),
    )

    # We only update subset of keys
    append_keys = ["1_1", "2_0"]  # Appending to existing samples
    append_tags = [{"global_steps": 1, "status": "finish", "model_version": 1} for _ in range(len(append_keys))]
    print(f"  Target Keys: {append_keys}")
    print("  New Field to Add is: 'response'")
    print(f"  The updated tags are: {append_tags}")
    tq.kv_batch_put(keys=append_keys, partition_id=partition_id, fields=response_batch, tags=append_tags)
    print("  ✓ Update success: Samples '1_1' and '2_0' now contain {input_ids, attention_mask, response}.")

    # Step 4: Only update tags through kv_put
    print("\n[Step 4] Update existing tags without providing value...")
    print("  Tags hold application metadata; TQ tracks field readiness separately.")
    key_for_update_tags = "0_0"
    tag_update = {"global_steps": 0, "status": "finish", "model_version": 0}
    print(f"  Target Key: {key_for_update_tags}")
    print(f"  The updated tag is: {tag_update}")
    tq.kv_put(key=key_for_update_tags, partition_id=partition_id, fields=None, tag=tag_update)
    print(f"  ✓ Update success: Samples '0_0' now has tag as {tag_update}.")

    response_key = keys[0]
    print(f"\n[Step 5] Building a response for '{response_key}' in chunks...")
    print("  kv_update requires SimpleStorage; use one writer per key or coordinate writers with a KV lock.")
    print("  A timed-out merge may have committed; retrying it blindly can append the same chunk twice.")
    tq.kv_update(
        key=response_key,
        partition_id=partition_id,
        fields={"response": torch.tensor([1, 2])},
        merge_fn=lambda old, new: torch.cat([old, new]),
        finalize=False,
    )
    print("  Stored fields:", tq.kv_describe(response_key, partition_id)[response_key])
    print("  Ready fields:", tq.kv_exists(response_key, partition_id)[response_key])

    tq.kv_update(
        key=response_key,
        partition_id=partition_id,
        fields={"response": torch.tensor([3, 4])},
        merge_fn=lambda old, new: torch.cat([old, new]),
        finalize=True,
    )
    response = tq.kv_batch_get(response_key, partition_id, select_fields="response")
    assert response["response"][0].tolist() == [1, 2, 3, 4]
    print("  Published response:", response["response"][0])
    tq.kv_put(response_key, partition_id, tag={"status": "finish"})

    # Step 6: List all keys and tags in a partition
    print("\n[Step 6] Listing all keys and tags in partition...")

    partition_info = tq.kv_list()
    print(f"  Found {len(partition_info.keys())} partitions: '{list(partition_info.keys())}'")
    for pid, keys_and_tags in partition_info.items():
        for k, t in keys_and_tags.items():
            print(f"Partition: {pid}, - key='{k}' | tag={t}")

    # Step 7: Retrieve specific fields using kv_batch_get
    print("\n[Step 7] Retrieving specific fields (Column) with kv_batch_get...")
    print("  Fetching only 'input_ids' to save bandwidth (ignoring 'attention_mask' and 'response').")

    all_keys = list(partition_info[partition_id].keys())
    retrieved_input_ids = tq.kv_batch_get(keys=all_keys, partition_id=partition_id, select_fields="input_ids")
    print(f"  ✓ Successfully retrieved only {list(retrieved_input_ids.keys())} field for all samples.")

    print("\n[Step 8] Retrieving the fields ready in every requested key...")
    retrieved_all = tq.kv_batch_get(keys=all_keys, partition_id=partition_id)
    print(f"  Requested keys: {all_keys}")
    print(f"  Common ready fields: {list(retrieved_all.keys())}")

    print("\n[Step 9] Clearing selected fields while keeping keys...")
    tq.kv_clear(all_keys, partition_id, fields="attention_mask")
    print("  Keys and their other fields remain:", tq.kv_exists(all_keys, partition_id))

    # Step 10: Clear specific keys
    print("\n[Step 10] Clearing keys from partition...")
    keys_to_clear = all_keys[:2]  # Delete the first 2 keys
    tq.kv_clear(keys=keys_to_clear, partition_id=partition_id)
    print(f"  ✓ Cleared keys: {keys_to_clear}")

    partition_info_after_clear = tq.kv_list(partition_id=partition_id)
    print(f"  Remaining keys in partition: {list(partition_info_after_clear[partition_id].keys())}")


def main():
    print("=" * 80)
    print(
        textwrap.dedent(
            """
        TransferQueue Tutorial 2: Key-Value (KV) Semantic API

        This tutorial demonstrates the KV semantic API, which provides a simple
        interface for data storage and retrieval using user-defined string keys.

        Key Methods:
        1. (async_)kv_put          - Insert/Update a multi-column sample by key, with optional metadata tag
        2. (async_)kv_batch_put    - Put multiple key-value pairs efficiently in batch
        3. (async_)kv_update       - Merge into fields with merge_fn(old, new), adding missing keys/fields
        4. (async_)kv_batch_get    - Retrieve samples (by keys), supporting column selection (by fields)
        5. (async_)kv_list         - List keys and tags (metadata) in a partition
        6. (async_)kv_clear        - Remove key-value pairs, or only selected fields, from storage

        Key Features:
        ✓ Redis-style Semantics  - Familiar KV interface (Put/Get/List) for zero learning curve
        ✓ Fine-grained Access    - Update or retrieve specific fields (columns) within a key (row) without full op.
        ✓ Partition Isolation    - Logical separation of storage namespaces
        ✓ Metadata Tags          - Lightweight metadata for status tracking
        ✓ Pluggable Backends     - Supports multiple backends

        Use Cases:
        - Focusing on fine-grained data access where extreme streaming performance is non-essential
        - Integration with external ReplayBuffer/single-controller that manage sample dispatching
        
        """
        )
    )
    print("=" * 80)

    try:
        print("Setting up TransferQueue...")
        tq.init()

        print("\nDemonstrating the KV semantic API...")
        demonstrate_kv_api()

        print("\n" + "=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. KV API simplifies data access with Redis-style semantics")
        print("  2. Read or write only the fields you need")
        print("  3. Tags store custom metadata such as status and scores")
        print("  4. Use `kv_list` to inspect keys and tags")
        print("  5. Supports incremental updates and publishing completed responses")
        print("  6. Supports field-level clearing while keeping keys")

        # Cleanup
        tq.close()
        ray.shutdown()
        print("\nCleanup complete")

    except Exception as e:
        print(f"Error during tutorial: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
