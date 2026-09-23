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

"""Selective data dump: persist the rows behind a set of keys, and read them back.

This is deliberately not a checkpoint. It stores payload and the metadata needed to
address it by key, and nothing about the controller: no index manager, no sampler, no
partition snapshot. Restoring therefore goes through the ordinary ``kv_batch_put``
path, which means a dump taken with N storage units restores into a system with M
storage units. Use ``save_checkpoint`` when you need a full system image instead.

Layout::

    <dump_dir>/
        dump_info.json                # partition, counts, shard count
        row_index.pt                  # key -> {global_index, fields, tag}
        shards/
            shard_info.json           # [{position, storage_unit_id, rows}]
            shard_<N>_<su_id>.pkl     # {field_data: {field: {gidx: value}}, global_indexes}
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from transfer_queue.utils import compact_pickle
from transfer_queue.utils.logging_utils import get_logger

logger = get_logger(__name__)

DUMP_FORMAT_VERSION = 1

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


def dump_data_by_key(dump_dir: str | Path, keys: list[str], partition_id: str) -> dict[str, int]:
    """Dump the rows addressed by ``keys`` into ``dump_dir``.

    Each storage unit pickles the rows it owns in its own process, so the payload never
    passes through the caller. The caller writes only a small row index.

    The directory is replaced wholesale: the dump is staged in ``<dump_dir>.tmp`` and
    renamed over ``dump_dir``, so anything already there is destroyed.

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
    from transfer_queue.interface import _TQ_CONTROLLER, _maybe_create_tq_client

    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    unique_keys = list(dict.fromkeys(keys))
    dump_dir = Path(dump_dir)
    tmp_dir = dump_dir.parent / (dump_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        client = _maybe_create_tq_client()
        rows = client.describe_rows_by_key(partition_id, unique_keys) if unique_keys else {}

        # A row whose fields are all still unproduced has nothing for a storage unit to
        # dump, but it keeps its key and tag so the restore can recreate the row.
        indexes_with_data = sorted(row["global_index"] for row in rows.values() if row["fields"])

        shard_records = (
            client.dump_rows_by_index(str(tmp_dir / _SHARD_SUBDIR), indexes_with_data) if indexes_with_data else []
        )
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
            torch.save({"partition_id": partition_id, "rows": rows}, f, pickle_module=compact_pickle)
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
            shutil.rmtree(dump_dir)
        tmp_dir.rename(dump_dir)
        _fsync_directory(dump_dir.parent)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    total_bytes = sum(path.stat().st_size for path in dump_dir.rglob("*") if path.is_file())
    logger.info(f"Dumped {len(unique_keys)} keys of partition {partition_id} to {dump_dir}")
    return {
        "keys": len(unique_keys),
        "rows_with_data": len(indexes_with_data),
        "shards": len(shard_records),
        "bytes": total_bytes,
    }


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
    row_index_path = Path(dump_dir) / _ROW_INDEX_FILE
    if not row_index_path.exists():
        raise FileNotFoundError(f"{_ROW_INDEX_FILE} not found in {dump_dir}")
    return torch.load(row_index_path, weights_only=False)


def load_data_by_key(dump_dir: str | Path) -> dict[str, int]:
    """Restore a dump into the running TransferQueue.

    Rows are written back through ``kv_batch_put``, so restoring merges by key: rows
    outside the dump are untouched, and the number of storage units may differ from
    the one used to write the dump. Global indexes are reallocated, which is safe
    because a dump is addressed by key.

    Shards are processed one at a time. Every field of a row lives in the shard of the
    unit that owned it, so a shard is self-contained and peak memory stays at one shard.

    Args:
        dump_dir: Directory previously written by ``dump_data_by_key``.

    Returns:
        ``{"keys", "rows_with_data", "shards", "bytes"}``.

    Raises:
        RuntimeError: TransferQueue is not initialized.
        FileNotFoundError: The dump is incomplete.
        ValueError: A shard disagrees with the row index.
    """
    from transfer_queue.interface import _TQ_CONTROLLER, kv_batch_put

    if _TQ_CONTROLLER is None:
        raise RuntimeError("TransferQueue is not initialized. Call tq.init() first.")

    dump_dir = Path(dump_dir)
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

    row_index = read_row_index(dump_dir)
    partition_id = row_index["partition_id"]
    rows = row_index["rows"]

    key_by_index = {row["global_index"]: key for key, row in rows.items()}
    fields_by_index = {row["global_index"]: row["fields"] for key, row in rows.items()}

    shard_dir = dump_dir / _SHARD_SUBDIR
    shard_info_path = shard_dir / _SHARD_INFO_FILE
    if not shard_info_path.exists():
        raise FileNotFoundError(f"{_SHARD_INFO_FILE} not found in {shard_dir}")
    with open(shard_info_path, encoding="utf-8") as f:
        shard_records = json.load(f)

    from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager

    restored_indexes: set[int] = set()
    for record in shard_records:
        shard_path = shard_dir / f"shard_{record['position']}_{record['storage_unit_id']}.pkl"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing dump shard: {shard_path}")

        for signature, batch in AsyncSimpleStorageManager.read_shard(str(shard_path), fields_by_index).items():
            global_indexes = batch["global_indexes"]
            batch_keys = [key_by_index[global_index] for global_index in global_indexes]
            kv_batch_put(
                keys=batch_keys,
                partition_id=partition_id,
                fields=batch["fields"],
                tags=[rows[key]["tag"] for key in batch_keys],
            )
            restored_indexes.update(global_indexes)

    # Rows that had no produced field yet were never sent to a storage unit; recreate
    # them from the row index so a caller holding their keys still finds them.
    keys_without_data = [key for key, row in rows.items() if not row["fields"]]
    if keys_without_data:
        kv_batch_put(
            keys=keys_without_data,
            partition_id=partition_id,
            fields=None,
            tags=[rows[key]["tag"] for key in keys_without_data],
        )

    if len(restored_indexes) != dump_info["num_rows_with_data"]:
        raise ValueError(
            f"Dump restore row count mismatch in {dump_dir}: shards yielded {len(restored_indexes)} rows, "
            f"{_DUMP_INFO_FILE} declares {dump_info['num_rows_with_data']}"
        )

    total_bytes = sum(path.stat().st_size for path in dump_dir.rglob("*") if path.is_file())
    logger.info(f"Restored {len(rows)} keys into partition {partition_id} from {dump_dir}")
    return {
        "keys": len(rows),
        "rows_with_data": len(restored_indexes),
        "shards": len(shard_records),
        "bytes": total_bytes,
    }
