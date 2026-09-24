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
atomic snapshot of concurrent writers. Calls on the same dump path are serialized
by an exclusive lock in a stable sibling `.lock` file. Different dump paths remain
independent, and storage units within a load still read in parallel.

## Distributed I/O

On export, the storage manager groups source indexes by their current storage
owner and concurrently asks those units to write their records. Only units holding
selected rows participate. Tensor storage is compacted during serialization, so a
row view cannot include the rest of its original batch, including inside tags.

On version-3 SimpleStorage restore:

1. The caller reads the row index and shard manifest and validates all file ranges.
2. The controller resolves existing keys and allocates indexes for new keys.
3. The storage manager routes records by the **current** indexes, then sends one
   load request to each participating unit concurrently.
4. Each target unit reads only its assigned byte ranges and merges those values
   into local storage. Records are processed in batches of at most 128 rows per
   shard; the caller never reads or forwards their payloads.
5. Each unit claims permission from the controller before writing, then reports
   completion directly. The client commits the saved schemas and tags only after
   every unit has completed. The controller reserves the destination partition
   until commit or confirmed cancellation.

The number of source units can differ from the number of destination units.
Even a dump with one source shard can restore across several target units because
records are independently addressable. Empty rows are recreated from metadata.

The dump directory must be on a filesystem accessible to every participating
storage unit. Local temporary storage suffices for single-node deployments.

| Operation | State | Payload I/O | Unit count on restore |
| --- | --- | --- | --- |
| Checkpoint | Entire controller and storage state | Each unit reads/writes its whole file | Must match |
| Selective v3 dump | Selected fields and tags, merged by key | Each owner unit reads/writes its records | May differ |

`DUMP_ROWS` and `LOAD_ROWS` are included in storage operation metrics. Unit logs
record loaded rows and bytes; the manager logs total bytes and participating units.
These count application reads, not filesystem read-ahead or physical disk traffic.

## Format and compatibility

New dumps use `format_version: 3`:

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

Version 3 also saves the original field schemas and only the selected nested row
shapes. Restore uses that schema regardless of target topology or batch boundaries;
non-tensor fields remain non-tensor even when a batch happens to contain only tensors.
Destination type conflicts are rejected before payload writes.

Version-1 dumps remain readable using the prior caller-side KV put path. Version-2
dumps retain direct reads, but lack original schemas and use the older inference
behavior; exact field-type preservation cannot be guaranteed for those files.
Restoring to a backend without direct selective loading uses KV puts. Version-1
and KV fallback restores do not provide distributed file reads. Old builds that only understand
versions 1 or 2 cannot read version-3 dumps. Export of nonempty dumps currently requires
SimpleStorage.

## Failure behavior

Publication writes and syncs `.tmp`, moves the old directory to `.old`, publishes
the new directory, and syncs its parent before deleting the backup. If publication
is interrupted while the main directory is absent, the next dump, load or row-index
read recovers `.old`. Readers perform recovery only while holding the same lock as
publishers, so a healthy rename window is never mistaken for a crashed writer.
The load keeps the lock through all remote reads; a pending load marker continues
to prevent replacement after a timeout or client exit. Do not delete the sibling
lock file: unlinking it can create two independent locks for the same dump.
The shared filesystem must provide cross-node advisory locking (not local-only
locks). A backup-cleanup error does not invalidate a published dump.

Restore is not transactional: payload writes before a failure remain. Every load
has a unique ID. The controller blocks clearing/reusing its destination indexes
and conflicting KV puts while an operation is unresolved. Units must claim that ID
before writing; cancellation rejects requests that have not yet claimed permission.
A receive timeout never releases a writer that has already claimed permission.

`RestorePendingError` means remote work is still running or its outcome is unknown.
The dump also retains a sibling `.restore` marker so an interrupted client cannot
silently allow its files to be replaced. After an interruption, call:

```python
tq.recover_data_load("/shared/dumps/selected")
```

Recovery cancels unclaimed work, asks units to resend terminal results, and releases
indexes only after every claimed worker has finished. It does not publish partial
restores as ready or undo payload writes. Retry recovery while a unit is still busy;
a lost unit requires stopping the old TQ actors and restarting the whole TQ system.
Restarting only the controller while old storage actors run is unsupported. After
recovery succeeds, retry the dump or clear its keys. Unknown operations remain
reserved rather than guessing that a timeout stopped remote execution.

Writers that already hold low-level metadata must remain paused throughout recovery.
The reservation covers the destination partition, so unrelated partitions can proceed.

## Tests

Run the selective E2E suite with its default pytest-managed temporary directory:

```bash
python -m pytest -q tests/e2e/test_data_dump_e2e.py tests/e2e/test_data_dump_cross_topology_e2e.py
```

For a multi-node Ray cluster, set `TQ_DUMP_TEST_ROOT` to an existing shared directory.
Tests create and remove only their own child directories beneath that root.
