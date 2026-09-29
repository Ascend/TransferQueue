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

Legacy imports can leave nested fields with missing row shapes, for example when a
later checkpoint chunk wraps tensor values in `NonTensorStack`. Export asks the owner
units to inspect those values while writing their records. The caller merges only
shape/type metadata before publishing the dump; payloads still stay on the units.
Missing tensor shapes are filled from the actual values and their dtype is checked.
If a missing-shape row contains `None` or an object, the entire selected field is
saved as non-tensor, with a warning, so every unit restores the same field contract.
Fields already declared non-tensor remain non-tensor. This repairs the exported
schema without mutating live controller metadata or inventing shapes for objects.
New puts also carry tensor shape hints for homogeneous tensor values wrapped in
`NonTensorStack`. An existing tensor field uses those hints to keep its shape map
complete; real mixed values make the field non-tensor. A field originally declared
non-tensor stays non-tensor when later batches contain only tensors.

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
Timeouts leave the operation pending, even if every unit later succeeds. They do
not cancel it or require changing the timeout used by ordinary puts and gets.

`RestorePendingError` means remote work is still running or its outcome is unknown.
The dump also retains a sibling `.restore` marker so an interrupted client cannot
silently allow its files to be replaced. After an interruption, call:

```python
committed = tq.recover_data_load("/shared/dumps/selected")
```

Recovery asks units to resend terminal results and commits schemas and tags only
when all units succeeded. It returns `True` for committed loads (or no pending
load), and `False` for a failed or cancelled load after all claimed workers stopped.
Unresolved work raises `RestorePendingError`, which includes `reason`, `unit_states`
and `report_errors`. Reports that fail retain the unit ID and original error message.
The controller remembers terminal outcomes so a lost commit reply can be confirmed
safely by retrying recovery; a redundant report failure cannot reverse that outcome.

- `running` units have claimed permission but have not reported completion. Retry
  recovery later; a reporting timeout does not prove that a worker has stopped.
- `pending` units have not claimed permission. A request may still be queued, so
  recovery does not automatically cancel it. If the initiating client exited before
  sending all requests, use `recover_data_load(dump_dir, cancel=True)` to abandon the
  operation; simply repeating recovery cannot dispatch the missing requests.
- `reason="unknown_restore"` means the controller has no record of the ID in the
  `.restore` marker. If the initiating client has stopped, use explicit cancellation.
  After a whole-system restart, first ensure all old actors have stopped, then cancel
  the stale operation. This marker blocks its dump path, not the new controller's
  unrelated partitions or checkpoints.

A unit whose claim timed out caches a failure confirming that it did not write.
Recovery accepts this failure even when the claim never reached the controller,
cancels unclaimed work and waits for any other claimed workers to finish.

To abandon the load explicitly, use `recover_data_load(dump_dir, cancel=True)`.
This denies unclaimed work and retains the reservation until claimed workers stop.
Cancellation cannot undo a load that has already committed. A failed unit also
cancels remaining unclaimed work; partial payload writes are never rolled back.
After cancellation settles, retry the load or clear its keys. A lost unit requires
stopping the old TQ actors and restarting the whole TQ system; restarting only the
controller while old storage actors run is unsupported.
After restarting, cancel the old marker before loading the dump again:

```python
# Run only after all old TQ actors have stopped and the new system is initialized.
tq.recover_data_load(dump_dir, cancel=True)
tq.load_data_by_key(dump_dir)
```

Writers that already hold low-level metadata must remain paused throughout recovery.
The controller reservation covers only the destination partition. Each storage unit
still serves requests on one worker thread: other partitions using that unit can
wait behind a load. The 128-row batches bound memory, not request latency.

## Tests

Run the selective E2E suite with its default pytest-managed temporary directory:

```bash
python -m pytest -q tests/e2e/test_data_dump_e2e.py tests/e2e/test_data_dump_cross_topology_e2e.py
```

For a multi-node Ray cluster, set `TQ_DUMP_TEST_ROOT` to an existing shared directory.
Tests create and remove only their own child directories beneath that root.
