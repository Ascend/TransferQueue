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

"""Tutorial 8: Data Parsers

Load tensors from filesystem paths on put, then select tensor rows on get.
The caller sends file paths; the storage units read the files and store the data.
See https://github.com/Ascend/TransferQueue/pull/82 for the original put-parser design.

Run: python tutorial/08_data_parser.py

This single-node example creates two small .pt files in a temporary directory.
For your own dataset, use existing file paths. In a multi-node deployment, every
storage node must be able to read those paths, for example on a shared filesystem.
Keep the files available and unchanged until the put call returns.

Requires SimpleStorage. Each parser receives the samples assigned to its storage
unit and must preserve field names, sample count and sample order. A get parser
must leave stored values unchanged. It reduces reply traffic; SSD-backed values
are still read and decoded in full before parsing.

Parsers are serialized to storage units; their dependencies must be available
there. Keep them free of side effects: parser-backed requests are not retried
automatically, and retrying a failed call can execute the parser again.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import ray
import torch
from tensordict import NonTensorStack, TensorDict

sys.path.append(str(Path(__file__).resolve().parent.parent))

import transfer_queue as tq  # noqa: E402


def load_features(data):
    """Read each tensor file on the storage unit and replace its path with the data."""
    features = []
    for file_path in data["features"]:
        features.append(torch.load(file_path, map_location="cpu", weights_only=True))
    data["features"] = features
    return data


def select_rows(data, row_indexes):
    """Return the requested rows of each tensor without modifying stored values."""
    selected = []
    for value, indexes in zip(data["features"], row_indexes, strict=True):
        selected.append(value[indexes])
    return {"features": selected}


def main():
    """Parse values on put and get, showing the input and output of each operation."""
    print("=" * 80)
    print("TransferQueue Tutorial 8: Data Parsers")
    print("=" * 80)
    print("A put parser can load data from file paths directly on the storage units.")
    print("A get parser transforms values before they are returned to the caller.")
    print("Both run on the storage units and keep field names and sample count/order unchanged.")
    print("Setting up TransferQueue with the default SimpleStorage backend...")
    ray.init(namespace="TransferQueueTutorial", include_dashboard=False)
    try:
        tq.init()
        partition_id = "parser"

        keys = ["sample_0", "sample_1"]
        source_features = [
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
            torch.arange(12, 24, dtype=torch.float32).reshape(3, 4),
        ]
        with TemporaryDirectory(prefix="tq-parser-") as data_dir:
            print("\n[Step 1] Preparing tensor files on the filesystem...")
            print("  This example creates two .pt files; a dataset can supply existing paths.")
            file_paths = []
            for key, value in zip(keys, source_features, strict=True):
                file_path = Path(data_dir) / f"{key}.pt"
                torch.save(value, file_path)
                file_paths.append(str(file_path))
                print(f"  {key}: saved shape={tuple(value.shape)} to {file_path}")
            print("  Each file contains a tensor with 3 rows and 4 columns.")

            print("\n[Step 2] Passing file paths to the put parser...")
            fields = TensorDict({"features": NonTensorStack(*file_paths)}, batch_size=len(keys))
            print("  NonTensorStack holds one file path per sample.")
            print("  The put request carries these paths; load_features reads the files on the storage units.")
            tq.kv_batch_put(keys, partition_id, fields=fields, data_parser=load_features)
            print("  The put call returned after the files were loaded into storage.")

            print("\n[Step 3] Reading the stored tensors without a get parser...")
            stored = tq.kv_batch_get(keys, partition_id, select_fields="features")
            for i, key in enumerate(keys):
                value = stored["features"][i]
                torch.testing.assert_close(value, source_features[i])
                print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                print(f"    Stored values: {value.tolist()}")
            print("  The stored tensors match the file contents.")

            print("\n[Step 4] Reading selected rows with a get parser...")
            print("  parser_args supplies one argument per key, in the same order as keys.")
            print("  select_rows(data, row_indexes) interprets each argument as row indexes starting at 0.")
            row_indexes = [[0, 2], [1]]
            for key, indexes in zip(keys, row_indexes, strict=True):
                print(f"  {key} requests rows {indexes}")
            selected = tq.kv_batch_get(
                keys,
                partition_id,
                select_fields="features",
                data_parser=select_rows,
                parser_args=row_indexes,
            )
            for i, key in enumerate(keys):
                value = selected["features"][i]
                torch.testing.assert_close(value, stored["features"][i][row_indexes[i]])
                print(f"  {key}: returned shape={tuple(value.shape)}")
                print(f"    Returned values: {value.tolist()}")
            print("  Only the selected rows were sent back from storage.")

            print("\n[Step 5] Reading the full tensors again...")
            full = tq.kv_batch_get(keys, partition_id, select_fields="features")
            for i, key in enumerate(keys):
                torch.testing.assert_close(full["features"][i], stored["features"][i])
                print(f"  {key}: full shape={tuple(full['features'][i].shape)}")
            print("  Both tensors still contain all 3 rows; the get parser left stored values unchanged.")

            print("\n[Step 6] Clearing the example data...")
            tq.kv_clear(keys, partition_id)
            print("  Cleared sample keys:", keys)

        print("\n" + "=" * 80)
        print("Tutorial Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("  1. Put parsers can load actual data from file paths on the storage units")
        print("  2. Get parsers transform returned values while leaving stored data unchanged")
        print("  3. parser_args provides per-sample options for a get parser")
        print("  4. Input files must be accessible from the storage nodes")
    finally:
        tq.close()
        ray.shutdown()
        print("\nCleanup complete")


if __name__ == "__main__":
    main()
