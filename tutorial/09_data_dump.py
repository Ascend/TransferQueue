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

"""Tutorial 9: Selective Data Dump and Restore

Save selected samples and resume processing after restarting TransferQueue.
The example saves shared fields separately, restores them before the samples,
then continues the unfinished responses.

Run: python tutorial/09_data_dump.py

This single-node SimpleStorage example uses a temporary directory. Multi-node
deployments need a shared filesystem accessible to every storage unit. Pause
writers and clears for the selected keys during dump/load, and load only trusted
dumps. A selective restore preserves unrelated data but does not restore sampler
or consumption state; applications track which keys still need processing.

Restores also support a different number of storage units. For sharing and
reclaiming public data, see tutorial/07_shared_fields.py.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import ray
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

import transfer_queue as tq  # noqa: E402


def main():
    """Save selected samples, restore their data and continue updating them."""
    print("=" * 80)
    print("TransferQueue Tutorial 9: Selective Data Dump and Restore")
    print("=" * 80)
    print("dump_data_by_key saves selected samples; load_data_by_key restores them.")
    print("This example saves two unfinished samples and resumes them after a restart.")
    print("Setting up TransferQueue with the default SimpleStorage backend...")
    ray.init(namespace="TransferQueueTutorial", include_dashboard=False)
    try:
        tq.init()
        partition_id = "resume"

        print("\n[Step 1] Preparing samples with shared fields and unfinished responses...")
        public_key = "tutorial09/shared_0"
        features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        tq.kv_put(public_key, tq.PUBLIC_PARTITION, fields={"features": features})
        print(f"  Stored {public_key}: features shape={tuple(features.shape)}")
        pending_keys = ["pending_0", "pending_1"]
        tq.kv_link(pending_keys, partition_id, public_key=public_key, fields="features")
        print(f"  Linked {pending_keys} to the same public features field.")
        for i, key in enumerate(pending_keys):
            first_chunk = torch.tensor([i, i + 1])
            tq.kv_update(
                key,
                partition_id,
                fields={"response": first_chunk},
                merge_fn=lambda old, new: torch.cat([old, new]),
                finalize=False,
            )
            tq.kv_put(key, partition_id, tag={"status": "pending", "chunks": 1})
            print(f"  {key}: stored response chunk={first_chunk.tolist()}, finalize=False")
        print("  The features field is readable; each response remains unfinished.")

        with TemporaryDirectory(prefix="tq-data-dump-") as dump_root:
            print("\n[Step 2] Saving the selected samples and their shared data...")
            public_dump = Path(dump_root) / "public"
            pending_dump = Path(dump_root) / "pending"
            print("  Sample dumps record links; the referenced public data needs its own dump.")
            print("  All writes have finished for now; the selected keys stay unchanged during the dump.")
            tq.dump_data_by_key(public_dump, [public_key], tq.PUBLIC_PARTITION)
            print(f"  Saved public key {public_key} to {public_dump}")
            report = tq.dump_data_by_key(pending_dump, pending_keys, partition_id)
            print(f"  Saved selected keys {pending_keys} to {pending_dump}")
            print("  Sample dump report:", report)

            print("\n[Step 3] Restarting TransferQueue before restoring the samples...")
            tq.close()
            tq.init()
            print("  Started an empty TransferQueue; both dumps remain on disk.")

            print("\n[Step 4] Restoring public data first, then the samples...")
            tq.load_data_by_key(public_dump)
            print("  Restored public fields:", tq.kv_exists(public_key, tq.PUBLIC_PARTITION))
            print("  Sample links resolve by public key name, so the expected public data must already be ready.")

            tags = tq.load_data_by_key(pending_dump)
            descriptions = tq.kv_describe(pending_keys, partition_id)
            for key in pending_keys:
                row = descriptions[key]
                assert row["unfinalized"] == ["response"]
                print(f"  {key}: tags={tags[key]}")
                print(f"    Stored fields={row['fields']}, unfinalized={row['unfinalized']}")
                print(f"    Shared-field links={row['links']}")
            print("  Restoring data keeps unfinished responses unfinalized.")
            print("  Sampler and consumption state are not restored; the application chooses which keys to resume.")

            print("\n[Step 5] Continuing the responses and publishing the completed fields...")
            for i, key in enumerate(pending_keys):
                final_chunk = torch.tensor([i + 2])
                print(f"  {key}: appending {final_chunk.tolist()} with finalize=True")
                tq.kv_update(
                    key,
                    partition_id,
                    fields={"response": final_chunk},
                    merge_fn=lambda old, new: torch.cat([old, new]),
                    finalize=True,
                )
                tq.kv_put(key, partition_id, tag={"status": "complete", "chunks": 2})
                batch = tq.kv_batch_get(key, partition_id, select_fields=["response", "features"])
                assert batch["response"][0].tolist() == [i, i + 1, i + 2]
                torch.testing.assert_close(batch["features"][0], features)
                print(f"    Readable response={batch['response'][0].tolist()}")
                print(f"    Shared features shape={tuple(batch['features'][0].shape)}")
            print("  Updated tags:", tq.kv_list(partition_id=partition_id)[partition_id])

        print("\n" + "=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Selected samples can be saved and restored by key")
        print("  2. Dumps preserve fields, tags, links and unfinished field state")
        print("  3. Shared data is saved separately and must be ready before restoring linked samples")
        print("  4. Restored unfinished responses can be updated and published later")
        print("\nSee tutorial/07_shared_fields.py for sharing and reclaiming public data.")
    finally:
        tq.close()
        ray.shutdown()
        print("\nCleanup complete")


if __name__ == "__main__":
    main()
