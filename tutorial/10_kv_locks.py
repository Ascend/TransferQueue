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

"""Tutorial 10: KV Locks

Coordinate operations on keys inside a with block.
Local locks coordinate threads and async tasks within one process. Global locks
coordinate callers across processes and nodes in the same TQ cluster.

Run: python tutorial/10_kv_locks.py

Locks work with every backend. Callers that need coordination must use the same
kind of lock for the same keys; KV APIs do not check these locks automatically.
A key lock does not automatically cover linked public keys or an entire partition.
Exceptions do not roll back completed operations.

Global locks renew a lease automatically. Call lease.check() before writing;
if LockLostError occurs, verify the data before retrying a write that may already
have completed.

Set TQ_ENABLE_KV_GLOBAL_LOCK=1 before the owning tq.init(); global locks are
disabled by default. Attached workers discover the lock actors automatically.

Pass multiple keys in one lock call. Nesting the same kind of lock is rejected;
when combining kinds, acquire global before local. In a running event loop, use
the async context managers instead; see the KV lock examples in basic.ipynb.
"""

import os
import sys
from pathlib import Path

import ray
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import transfer_queue as tq  # noqa: E402


def main():
    """Read, modify and write a value while holding a local or global lock."""
    print("=" * 80)
    print("TransferQueue Tutorial 10: KV Locks")
    print("=" * 80)
    print("The with block can contain several operations on the locked keys.")
    print("Local locks coordinate threads and async tasks within one process.")
    print("Global locks coordinate callers across processes and nodes in the same TQ cluster.")
    print("Callers that need coordination must use the same kind of lock for the same keys.")
    print("\nOperations that can be grouped inside a lock:")
    print("  Read or inspect: kv_batch_get, kv_exists, kv_describe, kv_list")
    print("  Write fields or tags: kv_put, kv_batch_put")
    print("  Merge or finalize fields: kv_update, kv_batch_update")
    print("  Clear keys or fields: kv_clear")
    print("  Link shared fields: kv_link")
    print("  Save or restore selected data: dump_data_by_key, load_data_by_key")
    print("  Application logic: validation, computation and decisions between calls")
    print("A key lock does not automatically lock public dependencies or the whole partition.")

    print("\nSetting up TransferQueue with global locks enabled...")
    os.environ["TQ_ENABLE_KV_GLOBAL_LOCK"] = "1"
    ray.init(namespace="TransferQueueTutorial", include_dashboard=False)
    try:
        tq.init()
        partition_id = "locks"
        print("  TQ_ENABLE_KV_GLOBAL_LOCK=1 was set before tq.init().")

        print("\n[Step 1] Creating counters for the local and global lock examples...")
        tq.kv_put("local", partition_id, fields={"count": torch.tensor(0)})
        tq.kv_put("global", partition_id, fields={"count": torch.tensor(0)})
        print("  Created keys 'local' and 'global', each with count=0.")

        print("\n[Step 2] Updating a counter under a local lock...")
        print("  Other threads using this local lock must wait until the block ends.")
        with tq.kv_local_lock("local", partition_id, timeout=10):
            batch = tq.kv_batch_get("local", partition_id)
            count = int(batch["count"][0])
            print(f"  Acquired the local lock; read count={count}")

            new_count = count + 1
            tq.kv_put("local", partition_id, fields={"count": torch.tensor(new_count)})
            print(f"  Wrote count={new_count} while still holding the lock.")
        print("  Leaving the with block released the local lock.")
        local_count = int(tq.kv_batch_get("local", partition_id)["count"][0])
        print(f"  Stored count after the update: {local_count}")

        print("\n[Step 3] Updating a counter under a global lock...")
        print("  Processes on the same or different nodes use this global lock to coordinate writes.")
        with tq.kv_global_lock("global", partition_id, timeout=10) as lease:
            batch = tq.kv_batch_get("global", partition_id)
            count = int(batch["count"][0])
            print(f"  Acquired the global lock; read count={count}")

            new_count = count + 1
            lease.check()
            tq.kv_put("global", partition_id, fields={"count": torch.tensor(new_count)})
            print(f"  Checked the lease and wrote count={new_count} while holding the lock.")
        print("  Leaving the with block released the global lock.")
        global_count = int(tq.kv_batch_get("global", partition_id)["count"][0])
        print(f"  Stored count after the update: {global_count}")

        print("\n[Step 4] Clearing the example data...")
        tq.kv_clear(["local", "global"], partition_id)
        print("  Cleared both counter keys.")

        print("\n" + "=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Local locks coordinate threads and async tasks within one process")
        print("  2. Global locks coordinate callers across processes and nodes in the same TQ cluster")
        print("  3. A lock can coordinate several operations within one with block")
        print("  4. Global locks use a lease, checked before writing")
        print("  5. Locks are released when the with block ends")
        print("\nSee tutorial/basic.ipynb for async with examples.")
    finally:
        tq.close()
        ray.shutdown()
        print("\nCleanup complete")


if __name__ == "__main__":
    main()
