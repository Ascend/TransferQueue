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

"""Persist selected rows and restore them by key without checkpointing controller state.

Version 3 preserves field schemas alongside independent row records in each shard.
The manifest maps source indexes to byte offsets, so current owner units read only their rows
when restoring into a different topology. Version 1 remains readable via KV puts.

Layout::

    <dump_dir>/
        dump_info.json                # version, partition, counts
        row_index.pt                  # key -> {global_index, fields, tag}
        shards/
            shard_info.json           # unit, row count, source index -> [offset, length]
            shard_<N>_<su_id>.pkl      # independent {global_index, fields} records
"""

import fcntl
import json
import os
import pickle
import shutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from transfer_queue.storage.dump_io import pack_dump_field, read_dump_row, validate_dump_values
from transfer_queue.utils import compact_pickle
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.tensor_utils import pack_field_values

logger = get_logger(__name__)

DUMP_FORMAT_VERSION = 3

_DUMP_INFO_FILE = "dump_info.json"
_ROW_INDEX_FILE = "row_index.pt"
_SHARD_SUBDIR = "shards"
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


@contextmanager
def _dump_lock(dump_dir: str | Path):
    """Serialize publication and recovery using a stable sibling inode shared by all callers."""
    directory = Path(dump_dir).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.with_name(directory.name + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield directory
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _recover_dump(dump_dir: Path) -> None:
    old_dir = dump_dir.with_name(dump_dir.name + ".old")
    if not dump_dir.exists() and old_dir.exists():
        old_dir.rename(dump_dir)
        _fsync_directory(dump_dir.parent)


def dump_data_by_key(dump_dir: str | Path, keys: list[str], partition_id: str) -> dict[str, int]:
    """Dump the rows addressed by ``keys`` into ``dump_dir``.

    Each storage unit pickles the rows it owns in its own process, so the payload never
    passes through the caller. The caller writes only a small row index.

    The directory is replaced wholesale. The previous dump is retained as ``.old``
    until publication is durable, and recovered on the next access after interruption.
    Access to the same directory is serialized with a sibling file lock.

    .. note::
        **Multi-node limitation**: dump_dir must reside on a shared network filesystem
        (e.g. NFS, GPFS, Lustre) reachable from every storage unit, because each unit
        writes its own shard from its own node.

    Callers must freeze writers for ``keys`` for the duration: TransferQueue has no
    atomic snapshot, so a concurrent put can land between the row index and the shards.

    Args:
        dump_dir: Directory to write the dump into.
        keys: Keys to dump. Duplicates are dropped, first occurrence wins.
        partition_id: Partition that owns ``keys``.

    Returns:
        ``{"keys", "rows_with_data", "shards", "bytes"}``.

    Raises:
        RuntimeError: TransferQueue is not initialized, the partition or a key does not
            exist, or a storage unit holds no data for a row it was asked to dump.
    """
    with _dump_lock(dump_dir) as directory:
        return _dump_data_by_key(directory, keys, partition_id)


def _dump_data_by_key(dump_dir: Path, keys: list[str], partition_id: str) -> dict[str, int]:
    from transfer_queue.interface import _TQ_CONTROLLER, _maybe_create_tq_client

    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    unique_keys = list(dict.fromkeys(keys))
    dump_dir = Path(dump_dir).resolve()
    client = _maybe_create_tq_client()
    _recover_dump(dump_dir)
    tmp_dir = dump_dir.parent / (dump_dir.name + ".tmp")
    old_dir = dump_dir.with_name(dump_dir.name + ".old")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        row_index = (
            client.describe_data_dump(partition_id, unique_keys)
            if unique_keys
            else {
                "partition_id": partition_id,
                "rows": {},
                "field_schema": {},
            }
        )
        rows = row_index["rows"]
        field_schema = row_index["field_schema"]
        missing_shapes = {}
        for row in rows.values():
            index = row["global_index"]
            names = [
                name
                for name in row["fields"]
                if field_schema[name].get("is_nested")
                and not field_schema[name].get("is_non_tensor")
                and field_schema[name].get("per_sample_shapes", {}).get(index) is None
            ]
            if names:
                missing_shapes[index] = names

        # A row whose fields are all still unproduced has nothing for a storage unit to
        # dump, but it keeps its key and tag so the restore can recreate the row.
        indexes_with_data = sorted(row["global_index"] for row in rows.values() if row["fields"])

        shard_records = (
            client.dump_rows_by_index(
                str(tmp_dir / _SHARD_SUBDIR),
                indexes_with_data,
                {row["global_index"]: row["fields"] for row in rows.values() if row["fields"]},
                **({"missing_shapes": missing_shapes} if missing_shapes else {}),
            )
            if indexes_with_data
            else []
        )
        _complete_dump_schema(field_schema, missing_shapes, shard_records)
        shard_dir = tmp_dir / _SHARD_SUBDIR
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

        with open(tmp_dir / _DUMP_INFO_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format_version": DUMP_FORMAT_VERSION,
                    "partition_id": partition_id,
                    "num_keys": len(unique_keys),
                    "num_rows_with_data": len(indexes_with_data),
                    "num_shards": len(shard_records),
                },
                f,
                indent=2,
            )
            _fsync_file(f)
        # Everything the dump claims is now durable, so the staging directory can be
        # published. Syncing the parent makes the rename itself survive a crash.
        _fsync_directory(tmp_dir)

        if dump_dir.exists():
            if old_dir.exists():
                shutil.rmtree(old_dir)
            dump_dir.rename(old_dir)
            _fsync_directory(dump_dir.parent)
        tmp_dir.rename(dump_dir)
        _fsync_directory(dump_dir.parent)
    except Exception:
        _recover_dump(dump_dir)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    # Publication already succeeded; cleanup must not invalidate the new dump.
    if old_dir.exists():
        try:
            shutil.rmtree(old_dir)
            _fsync_directory(dump_dir.parent)
        except OSError:
            logger.warning("Could not remove previous dump at %s", old_dir, exc_info=True)

    total_bytes = sum(path.stat().st_size for path in dump_dir.rglob("*") if path.is_file())
    logger.info(f"Dumped {len(unique_keys)} keys of partition {partition_id} to {dump_dir}")
    return {
        "keys": len(unique_keys),
        "rows_with_data": len(indexes_with_data),
        "shards": len(shard_records),
        "bytes": total_bytes,
    }


def _complete_dump_schema(field_schema: dict, missing_shapes: dict[int, list[str]], shards: list[dict]) -> None:
    """Complete legacy nested metadata before publishing any dump manifest."""
    recovered = {}
    for shard in shards:
        for index, fields in shard.pop("recovered_schema", {}).items():
            if index in recovered:
                raise ValueError(f"Duplicated schema recovery for row {index}")
            recovered[index] = fields
    if set(recovered) != set(missing_shapes):
        raise ValueError("Storage units did not return every requested missing row schema")
    for index, names in missing_shapes.items():
        if set(recovered[index]) != set(names):
            raise ValueError(f"Recovered schema fields disagree for row {index}")

    non_tensor_fields = {name for fields in recovered.values() for name, meta in fields.items() if meta is None}
    for name in non_tensor_fields:
        field_schema[name] = {"is_non_tensor": True, "is_nested": False, "shape": None, "dtype": None}
        logger.warning(
            "Dump field %r contains non-tensor values missing from legacy nested metadata; preserving as non-tensor",
            name,
        )
    for index, fields in recovered.items():
        for name, meta in fields.items():
            if name in non_tensor_fields:
                continue
            field = field_schema[name]
            if meta["dtype"] != field["dtype"]:
                raise ValueError(f"Dump field {name!r} dtype mismatch at row {index}")
            field["per_sample_shapes"][index] = tuple(meta["shape"])
    if recovered:
        logger.info("Recovered missing dump schemas for %s rows", len(recovered))


def read_row_index(dump_dir: str | Path) -> dict[str, Any]:
    """Read a dump's row index without touching its payload.

    Lets a caller answer questions about which keys and fields a dump holds, for
    example whether a row satisfies the current data contract, before paying to
    deserialize any shard.

    Args:
        dump_dir: Directory previously written by ``dump_data_by_key``.

    Returns:
        ``{"partition_id": str, "rows": {key: {"global_index", "fields", "tag"}}}``.

    Raises:
        FileNotFoundError: The row index is missing.
    """
    with _dump_lock(dump_dir) as directory:
        return _read_row_index(directory)


def _read_row_index(dump_dir: Path) -> dict[str, Any]:
    dump_dir = Path(dump_dir)
    _recover_dump(dump_dir)
    row_index_path = dump_dir / _ROW_INDEX_FILE
    if not row_index_path.exists():
        raise FileNotFoundError(f"{_ROW_INDEX_FILE} not found in {dump_dir}")
    return torch.load(row_index_path, weights_only=False)


def load_data_by_key(dump_dir: str | Path) -> dict[str, int]:
    """Merge selected rows into the running system, preserving existing key indexes.

    SimpleStorage units read their assigned indexed records directly and in parallel.
    Version-1 dumps and other backends use the compatible KV put path. New keys receive
    current indexes; unrelated rows and fields remain untouched. Writers and clears for
    these keys must be paused during restore. As with ``kv_batch_put``, a failure may
    leave new keys registered and payload partially written; retrying is idempotent.

    Args:
        dump_dir: Directory previously written by ``dump_data_by_key``. For direct
            distributed loading it must be accessible from every storage unit.

    Returns:
        ``{"keys", "rows_with_data", "shards", "bytes"}``.

    Raises:
        RuntimeError: TransferQueue is not initialized or a storage unit fails.
        FileNotFoundError: The dump is incomplete.
        ValueError: The manifest or a row disagrees with the row index.
    """
    with _dump_lock(dump_dir) as directory:
        return _load_data_by_key(directory)


def _load_data_by_key(dump_dir: Path) -> dict[str, int]:
    from transfer_queue.interface import _TQ_CONTROLLER, _maybe_create_tq_client

    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    dump_dir = Path(dump_dir).resolve()
    _recover_dump(dump_dir)
    info_path = dump_dir / _DUMP_INFO_FILE
    if not info_path.exists():
        raise FileNotFoundError(f"{_DUMP_INFO_FILE} not found in {dump_dir}")
    with open(info_path, encoding="utf-8") as f:
        dump_info = json.load(f)
    if dump_info["format_version"] not in (1, 2, DUMP_FORMAT_VERSION):
        raise ValueError(
            f"Unsupported dump format version {dump_info['format_version']} in {dump_dir}; "
            f"this build reads versions 1 through {DUMP_FORMAT_VERSION}"
        )

    row_index = _read_row_index(dump_dir)
    partition_id = row_index["partition_id"]
    rows = row_index["rows"]

    shard_dir = dump_dir / _SHARD_SUBDIR
    with open(shard_dir / _SHARD_INFO_FILE, encoding="utf-8") as f:
        shard_records = json.load(f)
    if (
        len(rows) != dump_info["num_keys"]
        or len(shard_records) != dump_info["num_shards"]
        or partition_id != dump_info["partition_id"]
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
        if dump_info["format_version"] >= 2:
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
        shard = {"path": str(path), "records": records}
        if dump_info["format_version"] >= 3:
            shard["field_schema"] = row_index["field_schema"]
        shards.append(shard)
    if dump_info["format_version"] >= 2 and seen != set(keys_by_index):
        raise ValueError("Dump shards do not contain every produced row")

    client = _maybe_create_tq_client()
    if dump_info["format_version"] >= 3:
        client.validate_dump_schema(partition_id, row_index["field_schema"])
    if dump_info["format_version"] >= 2 and hasattr(getattr(client, "storage_manager", None), "load_rows_by_index"):
        client.load_rows_by_key(partition_id, rows, shards)
    else:
        _load_via_kv(partition_id, rows, shards, dump_info["format_version"])

    total_bytes = sum(path.stat().st_size for path in dump_dir.rglob("*") if path.is_file())
    logger.info(f"Restored {len(rows)} keys into partition {partition_id} from {dump_dir}")
    return {
        "keys": len(rows),
        "rows_with_data": len(keys_by_index),
        "shards": len(shard_records),
        "bytes": total_bytes,
    }


def _load_via_kv(partition_id: str, rows: dict[str, Any], shards: list[dict], version: int) -> None:
    """Retain v1 and non-SimpleStorage compatibility without changing their put contract."""
    from transfer_queue.interface import kv_batch_put

    keys_by_index = {row["global_index"]: key for key, row in rows.items()}
    restored = set()
    for shard in shards:
        with open(shard["path"], "rb") as f:
            if version == 1:
                saved = pickle.load(f)
                records = [
                    {"key": keys_by_index[index], "source_index": index, "fields": rows[keys_by_index[index]]["fields"]}
                    for index in saved["global_indexes"]
                ]
            else:
                records = shard["records"]
            for start in range(0, len(records), 128):
                groups = defaultdict(list)
                for record in records[start : start + 128]:
                    index = record["source_index"]
                    fields = record["fields"]
                    if version == 1:
                        values = {name: saved["field_data"][name][index] for name in fields}
                    else:
                        values = read_dump_row(f, record["offset"], record["length"], index, fields)
                    if version >= 3:
                        validate_dump_values(values, shard["field_schema"], index)
                    groups[tuple(fields)].append((record["key"], values))
                    restored.add(index)
                for signature, batch in groups.items():
                    keys = [key for key, _ in batch]
                    packed = {
                        name: pack_dump_field([values[name] for _, values in batch], shard["field_schema"][name])
                        if version >= 3
                        else pack_field_values([values[name] for _, values in batch])
                        for name in signature
                    }
                    kv_batch_put(
                        keys,
                        partition_id,
                        TensorDict(packed, batch_size=len(keys)),
                        tags=[rows[key]["tag"] for key in keys],
                    )
    if restored != {row["global_index"] for row in rows.values() if row["fields"]}:
        raise ValueError("Dump restore row count mismatch")
    keys = [key for key, row in rows.items() if not row["fields"]]
    if keys:
        kv_batch_put(keys, partition_id, tags=[rows[key]["tag"] for key in keys])
