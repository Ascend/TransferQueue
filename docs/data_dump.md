# Selective data dump

`dump_data_by_key` persists selected keys, their produced fields and tags.
`load_data_by_key` merges them into a running TransferQueue. It preserves existing
key indexes and unrelated rows, fields and tag entries; new keys receive indexes
from the current controller. It does not restore sampler or consumption state.

```python
import transfer_queue as tq

tq.init()
tq.dump_data_by_key("/shared/dumps/selected", ["sample-1", "sample-2"], "train")
index = tq.read_row_index("/shared/dumps/selected")
tq.load_data_by_key("/shared/dumps/selected")
```

Pause writes and clears for these keys during both operations. A dump is not an
atomic snapshot of concurrent writers. Multiple publishers must not use the same
dump directory concurrently.

## Distributed I/O

On export, the storage manager groups source indexes by their current storage
owner and concurrently asks those units to write their records. Only units holding
selected rows participate. Tensor storage is compacted during serialization, so a
row view cannot include the rest of its original batch, including inside tags.

On version-2 SimpleStorage restore:

1. The caller reads the row index and shard manifest and validates all file ranges.
2. The controller resolves existing keys and allocates indexes for new keys.
3. The storage manager routes records by the **current** indexes, then sends one
   load request to each participating unit concurrently.
4. Each target unit reads only its assigned byte ranges and merges those values
   into local storage. Records are processed in batches of at most 128 rows per
   shard; the caller never reads or forwards their payloads.
5. After all units succeed, the client publishes returned field schemas through
   the controller and merges tags using the normal metadata update path.

The number of source units can differ from the number of destination units.
Even a dump with one source shard can restore across several target units because
records are independently addressable. Empty rows are recreated from metadata.

The dump directory must be on a filesystem accessible to every participating
storage unit. Local temporary storage suffices for single-node deployments.

| Operation | State | Payload I/O | Unit count on restore |
| --- | --- | --- | --- |
| Checkpoint | Entire controller and storage state | Each unit reads/writes its whole file | Must match |
| Selective v2 dump | Selected fields and tags, merged by key | Each owner unit reads/writes its records | May differ |

`DUMP_ROWS` and `LOAD_ROWS` are included in storage operation metrics. Unit logs
record loaded rows and bytes; the manager logs total bytes and participating units.
These count application reads, not filesystem read-ahead or physical disk traffic.

## Format and compatibility

New dumps use `format_version: 2`:

```text
dump_info.json
row_index.pt
shards/
    shard_info.json
    shard_0_<source-unit-id>.pkl
    ...
```

Each shard is a sequence of independent pickle records containing a source global
index and a field/value mapping. `shard_info.json` records each source index's
`[offset, length]`. Source indexes only locate records; they are never reused as
current indexes without controller resolution. `row_index.pt` remains readable
with `read_row_index` without opening payload shards.

Version-1 dumps remain readable using the prior caller-side KV put path. Restoring
to a backend without direct selective loading also uses KV puts. These compatibility
paths do not provide distributed file reads. Old builds that only understand
version 1 cannot read version-2 dumps. Export of nonempty dumps currently requires
SimpleStorage.

## Failure behavior

Publication writes and syncs `.tmp`, moves the old directory to `.old`, publishes
the new directory, and syncs its parent before deleting the backup. If publication
is interrupted while the main directory is absent, the next dump, load or row-index
read recovers `.old`. A backup-cleanup error does not invalidate a published dump.

Restore is not transactional. All unit requests are awaited before returning an
error, and failed storage requests prevent publication of new ready metadata.
Earlier payload writes or earlier metadata updates can remain after a failure;
existing produced rows may already contain restored values. Correct the cause and
retry the same dump with writers paused. No unrelated partition is cleared.

## Tests

Run the selective E2E suite with its default pytest-managed temporary directory:

```bash
python -m pytest -q tests/e2e/test_data_dump_e2e.py tests/e2e/test_data_dump_cross_topology_e2e.py
```

For a multi-node Ray cluster, set `TQ_DUMP_TEST_ROOT` to an existing shared directory.
Tests create and remove only their own child directories beneath that root.
