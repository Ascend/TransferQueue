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
atomic snapshot of concurrent writers. Each dump goes to a new directory: an existing
target is refused, so a published dump never changes and can be loaded from a
read-only location.

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
5. After every unit has succeeded, the storage manager publishes the saved schemas
   to the controller, as an ordinary put does, and the client then writes the tags.

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

A dump is staged in a uniquely named sibling `<dump>.tmp-<id>` directory, with
`dump_info.json` written last, and renamed into place once everything is synced.
A failed dump removes its staging directory; a crash can leave one behind, which is
never read and can be deleted. If two dumps race to one path, only the first rename
succeeds. Readers need no lock and no write access.

Restore has the failure semantics of `kv_batch_put`: it is not transactional, and
payload writes before a failure remain. Metadata is published only after every unit
has succeeded, so a failed load leaves those writes invisible. Existing keys keep
their indexes, so retrying the same load is idempotent; clearing the keys abandons it.
Like an ordinary put, a load does not fence late writes against indexes that are
cleared and reused while it runs, so keep writers and clears for these keys paused.

Each storage unit serves requests on one worker thread: other partitions using that
unit can wait behind a load. The 128-row batches bound memory, not request latency.
Dump and load requests use their own connection pool, whose timeout is
`TQ_SIMPLE_STORAGE_DUMP_TIMEOUT` (3600 seconds by default) rather than the put/get
timeout, because each unit receives a single request covering all of its rows.

## Tests

Run the selective E2E suite with its default pytest-managed temporary directory:

```bash
python -m pytest -q tests/e2e/test_data_dump_e2e.py tests/e2e/test_data_dump_cross_topology_e2e.py
```

For a multi-node Ray cluster, set `TQ_DUMP_TEST_ROOT` to an existing shared directory.
Tests create and remove only their own child directories beneath that root.
