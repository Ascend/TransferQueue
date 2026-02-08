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

"""
Tutorial 2: High-Level KV Interface

This tutorial demonstrates the Redis-style Key-Value interface for TransferQueue.
The KV API provides a simple Put/Get/List/Clear interface that requires zero
knowledge of TransferQueue's internal metadata concepts (BatchMeta, SampleMeta,
FieldMeta).

TransferQueue offers three API tiers:
  1. KV Interface (this tutorial) — Familiar Redis-style semantics for zero
     learning curve.  Best when you need fine-grained data access and manage
     sample dispatching externally (e.g., via a ReplayBuffer or single-controller).
  2. StreamingDataLoader (tutorial/05_streaming_dataloader.py) — PyTorch-native
     DataLoader with built-in streaming, sampling, and DP-rank coordination.
     Best for fully-streamed training pipelines.
  3. Low-level TransferQueueClient — Full control over BatchMeta, Samplers, and
     production/consumption tracking.  Best when you need maximum flexibility.

Key Methods:
  - kv_put / async_kv_put             — Insert/update a single sample by key
  - kv_batch_put / async_kv_batch_put  — Batch insert multiple key-value pairs
  - kv_batch_get / async_kv_batch_get  — Retrieve samples by keys (with optional
                                         column/field selection)
  - kv_list / async_kv_list            — List all keys and tags in a partition
  - kv_clear / async_kv_clear          — Remove key-value pairs from storage

Key Features:
  ✓ Redis-style Semantics  — Familiar KV interface with zero learning curve
  ✓ Fine-grained Access    — Read/write specific fields (columns) per key (row)
  ✓ Partition Isolation    — Logical separation of storage namespaces
  ✓ Metadata Tags          — Lightweight per-sample metadata for status tracking
  ✓ Pluggable Backends     — Works with SimpleStorage, Yuanrong, MooncakeStore, etc.

Use Cases:
  - Fine-grained data access where extreme streaming performance is non-essential
  - Integration with external ReplayBuffer / single-controller that manages
    sample dispatching

Limitations (vs low-level native APIs):
  - No built-in production/consumption tracking (track status via tags)
  - No built-in Sampler support (dispatch data externally)
  - Not fully streaming (consumers wait for dispatched keys)
"""

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

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import transfer_queue as tq  # noqa: E402

# Configure Ray
os.environ["RAY_DEDUP_LOGS"] = "0"


def setup():
    """Initialize Ray and TransferQueue."""
    if not ray.is_initialized():
        ray.init(namespace="TransferQueueTutorial")

    tq.init()
    print("[Setup]: TransferQueue initialized.\n")


def teardown():
    """Shutdown TransferQueue and Ray."""
    tq.close()
    ray.shutdown()
    print("\n[Teardown]: TransferQueue and Ray shut down.")


# ──────────────────────────────────────────────────────────────────────
# Example 1: Basic Put / Get
# ──────────────────────────────────────────────────────────────────────
def example_basic_put_get():
    """Demonstrate single-key put and batch get."""
    print(
        textwrap.dedent("""
    ┌──────────────────────────────────────────────────────┐
    │ Example 1: Basic Put / Get                           │
    └──────────────────────────────────────────────────────┘
    """)
    )

    partition = "demo_partition"

    # --- Put a single sample ---
    tq.kv_put(
        key="sample_0",
        fields={
            "input_ids": torch.tensor([101, 2003, 1037]),
            "attention_mask": torch.tensor([1, 1, 1]),
        },
        partition_id=partition,
    )
    print("[Put]: Inserted 'sample_0' with fields: input_ids, attention_mask")

    # --- Put another sample with a tag ---
    tq.kv_put(
        key="sample_1",
        fields={
            "input_ids": torch.tensor([101, 4567, 2345]),
            "attention_mask": torch.tensor([1, 1, 0]),
        },
        partition_id=partition,
        tag={"status": "completed", "score": 0.95},
    )
    print("[Put]: Inserted 'sample_1' with tag: {status: completed, score: 0.95}")

    # --- Batch get ---
    results = tq.kv_batch_get(
        keys=["sample_0", "sample_1"],
        partition_id=partition,
    )
    for key, td in results.items():
        print(f"[Get]: {key} → input_ids={td['input_ids'].squeeze()}")

    # --- Get with field selection ---
    results = tq.kv_batch_get(
        keys=["sample_1"],
        fields=["attention_mask"],
        partition_id=partition,
    )
    print(f"[Get]: sample_1 (attention_mask only) → {results['sample_1']['attention_mask'].squeeze()}")

    tq.kv_clear(partition_id=partition)
    print("[Clear]: Partition cleared.\n")


# ──────────────────────────────────────────────────────────────────────
# Example 2: Batch Put
# ──────────────────────────────────────────────────────────────────────
def example_batch_put():
    """Demonstrate efficient batch insertion."""
    print(
        textwrap.dedent("""
    ┌──────────────────────────────────────────────────────┐
    │ Example 2: Batch Put                                 │
    └──────────────────────────────────────────────────────┘
    """)
    )

    partition = "batch_partition"

    tq.kv_batch_put(
        kv_pairs={
            "s0": {"reward": torch.tensor([0.5]), "input_ids": torch.tensor([1, 2])},
            "s1": {"reward": torch.tensor([0.8]), "input_ids": torch.tensor([3, 4])},
            "s2": {"reward": torch.tensor([0.3]), "input_ids": torch.tensor([5, 6])},
        },
        partition_id=partition,
        tags={
            "s0": {"status": "done"},
            "s1": {"status": "done"},
            "s2": {"status": "pending"},
        },
    )
    print("[Batch Put]: Inserted s0, s1, s2 with tags.")

    # List all keys
    entries = tq.kv_list(partition_id=partition)
    for entry in entries:
        print(f"  Key: {entry['key']}, Tag: {entry.get('tag', {})}")

    tq.kv_clear(partition_id=partition)
    print("[Clear]: Partition cleared.\n")


# ──────────────────────────────────────────────────────────────────────
# Example 3: Partition Isolation
# ──────────────────────────────────────────────────────────────────────
def example_partition_isolation():
    """Demonstrate partition-level namespace isolation."""
    print(
        textwrap.dedent("""
    ┌──────────────────────────────────────────────────────┐
    │ Example 3: Partition Isolation                       │
    └──────────────────────────────────────────────────────┘
    """)
    )

    tq.kv_put(key="x", fields={"v": torch.tensor([1.0])}, partition_id="ns_a")
    tq.kv_put(key="x", fields={"v": torch.tensor([2.0])}, partition_id="ns_b")

    res_a = tq.kv_batch_get(keys=["x"], partition_id="ns_a")
    res_b = tq.kv_batch_get(keys=["x"], partition_id="ns_b")

    print(f"[Partition A]: x → {res_a['x']['v'].item()}")
    print(f"[Partition B]: x → {res_b['x']['v'].item()}")

    tq.kv_clear(partition_id="ns_a")
    tq.kv_clear(partition_id="ns_b")
    print("[Clear]: Both partitions cleared.\n")


# ──────────────────────────────────────────────────────────────────────
# Example 4: Selective Clear
# ──────────────────────────────────────────────────────────────────────
def example_selective_clear():
    """Demonstrate clearing specific keys from a partition."""
    print(
        textwrap.dedent("""
    ┌──────────────────────────────────────────────────────┐
    │ Example 4: Selective Clear                           │
    └──────────────────────────────────────────────────────┘
    """)
    )

    partition = "clear_demo"

    tq.kv_batch_put(
        kv_pairs={
            "a": {"val": torch.tensor([10])},
            "b": {"val": torch.tensor([20])},
            "c": {"val": torch.tensor([30])},
        },
        partition_id=partition,
    )
    print("[Put]: Inserted keys a, b, c.")

    tq.kv_clear(keys=["a", "c"], partition_id=partition)
    print("[Clear]: Removed keys a, c.")

    remaining = tq.kv_list(partition_id=partition)
    print(f"[List]: Remaining keys = {[e['key'] for e in remaining]}")

    tq.kv_clear(partition_id=partition)
    print("[Clear]: Partition cleared.\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TransferQueue — KV Interface Tutorial")
    print("=" * 60)

    setup()

    try:
        example_basic_put_get()
        example_batch_put()
        example_partition_isolation()
        example_selective_clear()
    finally:
        teardown()

    print("\nDone! For more details, see tests/e2e/test_kv_interface_e2e.py")
