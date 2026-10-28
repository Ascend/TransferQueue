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

"""Tutorial 7: Shared Fields

Share stored fields across samples through PUBLIC_PARTITION and kv_link. This
example uses a tensor field named features and a private score for each sample.

Run: python tutorial/07_shared_fields.py

Requires SimpleStorage. Public keys are shared by applications on the same TQ;
use an application-specific prefix. Linked fields and referenced public rows
cannot be overwritten.
"""

import sys
from pathlib import Path

import ray
import torch
from tensordict import TensorDict

sys.path.append(str(Path(__file__).resolve().parent.parent))

import transfer_queue as tq  # noqa: E402


def main():
    """Demonstrate linking, reading and reclaiming shared fields."""
    print("=" * 80)
    print("TransferQueue Tutorial 7: Shared Fields")
    print("=" * 80)
    print("A field can be stored once and referenced by multiple samples.")
    print("Each sample can also have its own private fields.")
    print("Setting up TransferQueue with the default SimpleStorage backend...")
    ray.init(namespace="TransferQueueTutorial", include_dashboard=False)
    try:
        tq.init()
        partition_id = "train"

        print("\n[Step 1] Storing shared fields in the public partition...")
        public_values = {
            "tutorial07/shared_0": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "tutorial07/shared_1": torch.arange(24, 48, dtype=torch.float32).reshape(6, 4),
        }
        public_keys = list(public_values)
        for public_key, value in public_values.items():
            tq.kv_put(public_key, tq.PUBLIC_PARTITION, fields={"features": value})
            print(f"  Stored {public_key}: features shape={tuple(value.shape)}")

        print("\n[Step 2] Linking sample fields to the shared values...")
        keys = [f"sample_{i}" for i in range(4)]
        sources = [public_keys[0], public_keys[0], public_keys[1], public_keys[1]]
        tq.kv_link(keys, partition_id, public_key=sources, fields="features")
        descriptions = tq.kv_describe(keys, partition_id)
        for key in keys:
            print(f"  {key} reads features from {descriptions[key]['links']['features']}")
        print("  The links reference the public rows; no features payload is copied into the sample rows.")

        print("\n[Step 3] Adding a private field to each sample...")
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
        tq.kv_batch_put(keys, partition_id, fields=TensorDict({"score": scores}, batch_size=len(keys)))
        print("  Added score values:", scores.tolist())
        print("  Each sample now has a shared features field and its own score field.")

        print("\n[Step 4] Reading shared and private fields together...")
        batch = tq.kv_batch_get(keys, partition_id)
        print("  Returned fields:", list(batch.keys()))
        for i, (key, source) in enumerate(zip(keys, sources, strict=True)):
            torch.testing.assert_close(batch["features"][i], public_values[source])
            print(f"  {key}: features shape={tuple(batch['features'][i].shape)}, score={batch['score'][i].item():.1f}")

        print("\n[Step 5] Clearing sample keys and their links...")
        tq.kv_clear(keys, partition_id)
        print("  Cleared sample keys:", keys)
        print("  Public keys still available:", list(tq.kv_exists(public_keys, tq.PUBLIC_PARTITION)))

        print("\n[Step 6] Reclaiming shared data with no remaining links...")
        freed = tq.kv_clear_unlinked_public()
        assert set(freed) == set(public_keys)
        print("  Reclaimed public keys:", freed)

        print("\n" + "=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Multiple samples can share a stored field through links")
        print("  2. Reads combine shared fields with each sample's private fields")
        print("  3. Shared data can be reclaimed after its last link is removed")
    finally:
        tq.close()
        ray.shutdown()
        print("\nCleanup complete")


if __name__ == "__main__":
    main()
