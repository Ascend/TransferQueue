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

"""File format of selective dumps written by ``dump_data_by_key``.

Field schemas are saved alongside independent row records in each shard. The manifest
maps source indexes to byte offsets, so current owner units read only their rows when
restoring into a different topology.

Layout::

    <dump_dir>/
        dump_info.json                # version, partition, counts
        row_index.pt                  # key -> {global_index, fields, tag}
        shards/
            shard_info.json           # unit, row count, source index -> [offset, length]
            shard_<N>_<su_id>.pkl      # independent {global_index, fields} records
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from transfer_queue.utils import compact_pickle
from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)

DUMP_FORMAT_VERSION = 3
SHARD_SUBDIR = "shards"

_DUMP_INFO_FILE = "dump_info.json"
_ROW_INDEX_FILE = "row_index.pt"
_SHARD_INFO_FILE = "shard_info.json"


def _fsync_file(file_object: Any) -> None:
    """Push a just-written file out of page cache before the caller moves on."""
    file_object.flush()
    os.fsync(file_object.fileno())


def _fsync_directory(path: Path) -> None:
    """Make a directory's own entries durable.

    fsync on a file says nothing about the directory entry naming it, so a crash can
    lose a file that was itself fully synced. The rename that publishes the dump needs
    the same treatment.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def dump_field_schema(declared: dict, row_schema: dict[int, dict]) -> dict:
    """Combine each field's declared type with the values its owner units hold.

    A put that wraps rows in ``NonTensorStack`` leaves a tensor field's metadata
    unchanged, so dtypes and shapes come from the stored rows: a field is saved as
    non-tensor unless every row is a tensor of one dtype, and nested if shapes differ.
    """
    rows_by_field = defaultdict(dict)
    for index, fields in row_schema.items():
        for name, meta in fields.items():
            rows_by_field[name][index] = meta
    schema = {}
    for name, rows in rows_by_field.items():
        metas = list(rows.values())
        if declared[name]["is_non_tensor"] or None in metas or len({dtype for dtype, _ in metas}) > 1:
            if not declared[name]["is_non_tensor"]:
                logger.warning("Dump field %r holds rows that are not tensors of one dtype; saving as non-tensor", name)
            schema[name] = {"dtype": None, "shape": None, "is_nested": False, "is_non_tensor": True}
            continue
        shapes = {index: tuple(shape) for index, (_, shape) in rows.items()}
        nested = declared[name]["is_nested"] or len(set(shapes.values())) > 1
        schema[name] = {
            "dtype": metas[0][0],
            # A dense field of scalars is declared with shape (1,), as a put records it.
            "shape": None if nested else shapes[next(iter(shapes))] or (1,),
            "is_nested": nested,
            "is_non_tensor": False,
        }
        if nested:
            schema[name]["per_sample_shapes"] = shapes
    return schema


def publish_dump(tmp_dir: Path, dump_dir: Path, row_index: dict[str, Any], shard_records: list[dict]) -> None:
    """Write the manifests next to the unit-written shards, then rename into place.

    ``dump_info.json`` is written last and the staging directory is renamed only once
    everything is durable, so a published dump is always complete.
    """
    shard_dir = tmp_dir / SHARD_SUBDIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(shard_dir / _SHARD_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(shard_records, f)
        _fsync_file(f)
    # The shards themselves were synced by the units that wrote them, but their
    # directory entries were created here, on this node.
    _fsync_directory(shard_dir)

    # torch.save rather than json: a tag is an arbitrary picklable dict, and this
    # path must not fail on a tag that happens to hold a tensor.
    with open(tmp_dir / _ROW_INDEX_FILE, "wb") as f:
        torch.save(row_index, f, pickle_module=compact_pickle)
        _fsync_file(f)

    rows = row_index["rows"]
    with open(tmp_dir / _DUMP_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "format_version": DUMP_FORMAT_VERSION,
                "partition_id": row_index["partition_id"],
                "num_keys": len(rows),
                "num_rows_with_data": sum(1 for row in rows.values() if row["fields"]),
                "num_shards": len(shard_records),
            },
            f,
            indent=2,
        )
        _fsync_file(f)
    # Everything the dump claims is now durable, so the staging directory can be
    # published. Syncing the parent makes the rename itself survive a crash.
    _fsync_directory(tmp_dir)

    # rename() refuses a non-empty target, so of two dumps racing to one path only
    # the first is published.
    tmp_dir.rename(dump_dir)
    _fsync_directory(dump_dir.parent)


def _read_row_index(dump_dir: Path) -> dict[str, Any]:
    row_index_path = dump_dir / _ROW_INDEX_FILE
    if not row_index_path.exists():
        raise FileNotFoundError(f"{_ROW_INDEX_FILE} not found in {dump_dir}")
    return torch.load(row_index_path, weights_only=False)


def read_dump(dump_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a dump's manifests and return its row index and per-shard records.

    Only manifests are read; each record locates one row's byte range for the unit
    that will load it.

    Raises:
        FileNotFoundError: The dump is incomplete.
        ValueError: The format version is unsupported, or a manifest disagrees with
            the row index.
    """
    info_path = dump_dir / _DUMP_INFO_FILE
    if not info_path.exists():
        raise FileNotFoundError(f"{_DUMP_INFO_FILE} not found in {dump_dir}")
    with open(info_path, encoding="utf-8") as f:
        dump_info = json.load(f)
    if dump_info["format_version"] != DUMP_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported dump format version {dump_info['format_version']} in {dump_dir}; "
            f"this build reads version {DUMP_FORMAT_VERSION}"
        )

    row_index = _read_row_index(dump_dir)
    rows = row_index["rows"]

    shard_dir = dump_dir / SHARD_SUBDIR
    with open(shard_dir / _SHARD_INFO_FILE, encoding="utf-8") as f:
        shard_records = json.load(f)
    if (
        len(rows) != dump_info["num_keys"]
        or len(shard_records) != dump_info["num_shards"]
        or row_index["partition_id"] != dump_info["partition_id"]
    ):
        raise ValueError("Dump manifest disagrees with the row index")
    keys_by_index = {row["global_index"]: key for key, row in rows.items() if row["fields"]}
    if len(keys_by_index) != dump_info["num_rows_with_data"]:
        raise ValueError("Dump row count disagrees with the row index")
    shards = []
    seen = set()
    for record in shard_records:
        path = shard_dir / f"shard_{record['position']}_{record['storage_unit_id']}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing dump shard: {path}")
        records = []
        size = path.stat().st_size
        offsets = record["row_offsets"]
        if len(offsets) != record["rows"]:
            raise ValueError(f"Dump shard row count mismatch: {path}")
        for source_index, (offset, length) in offsets.items():
            source_index = int(source_index)
            if source_index not in keys_by_index or source_index in seen:
                raise ValueError(f"Unexpected or duplicated row {source_index} in {path}")
            if offset < 0 or length <= 0 or offset + length > size:
                raise ValueError(f"Invalid row range for {source_index} in {path}")
            key = keys_by_index[source_index]
            records.append(
                {
                    "key": key,
                    "source_index": source_index,
                    "fields": rows[key]["fields"],
                    "offset": offset,
                    "length": length,
                }
            )
            seen.add(source_index)
        shards.append({"path": str(path), "records": records, "field_schema": row_index["field_schema"]})
    if seen != set(keys_by_index):
        raise ValueError("Dump shards do not contain every produced row")
    return row_index, shards
