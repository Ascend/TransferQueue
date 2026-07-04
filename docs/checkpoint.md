# Checkpoint: Save and Restore System State

> Last updated: 07/03/2026

## Overview

TransferQueue provides `tq.save_checkpoint` and `tq.load_checkpoint` to persist and restore the full system state — controller metadata and storage data — enabling fault tolerance and training resumption after a restart.

A checkpoint captures two components:

- **Controller state** — all partition metadata, the global index manager, and sampler state.
- **Storage data** — the tensor and non-tensor field data held by each `SimpleStorageUnit`.

## Quick Start

```python
import transfer_queue as tq

tq.init(config)

# ... put/get data during training ...

# Save at a step boundary
tq.save_checkpoint("/shared/fs/checkpoints/step_1000", metadata={"step": 1000})

# --- After restart ---
tq.init(config)
tq.load_checkpoint("/shared/fs/checkpoints/step_1000")
# System state restored; training can resume
```

## API Reference

### `tq.save_checkpoint`

```python
def save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    include_storage: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None
```

| Parameter | Description |
|-----------|-------------|
| `checkpoint_dir` | Directory to write the checkpoint. Created if it does not exist. If a checkpoint already exists at this path it is replaced (best-effort; see [Known Limitations §3](#3-replacing-an-existing-checkpoint-is-not-fully-atomic)). |
| `include_storage` | Whether to save storage unit data. For `SimpleStorage` (in-memory), this is forced to `True` regardless of the value passed — skipping storage would cause complete data loss on restart. For persistent external backends, `False` is valid. |
| `metadata` | Optional user-defined key-value pairs written into `metadata.json`. Useful for recording step number, timestamp, etc. |

**Raises**: `RuntimeError` if `tq.init()` has not been called, or if any write step fails.

---

### `tq.load_checkpoint`

```python
def load_checkpoint(
    checkpoint_dir: str | Path,
) -> None
```

| Parameter | Description |
|-----------|-------------|
| `checkpoint_dir` | Path to a directory previously written by `save_checkpoint`. |

**Raises**: `FileNotFoundError` if the directory or required files are missing; `ValueError` if the number of storage units in the checkpoint does not match the running system; `RuntimeError` if `tq.init()` has not been called or restore fails.

**Prerequisite**: `tq.init()` must have been called and the system must be in a clean state (no prior data operations). `load_checkpoint` restores state into a running system; it does not launch components.

## Checkpoint Directory Layout

```
checkpoint_dir/
├── metadata.json                        # Flags and user metadata
├── controller_state.pkl                 # Controller state (pickle)
└── simple_storage/
    ├── storage_unit_info.json           # Position-to-ID manifest
    ├── su_0_<id>.pkl                    # StorageUnit at position 0
    ├── su_1_<id>.pkl                    # StorageUnit at position 1
    └── ...
```

**`metadata.json`**

```json
{
  "storage_saved": true,
  "user_metadata": {"step": 1000}
}
```

**`storage_unit_info.json`**

```json
[
  {"position": 0, "storage_unit_id": "<id_0>"},
  {"position": 1, "storage_unit_id": "<id_1>"}
]
```

## Architecture

```
tq.save_checkpoint / tq.load_checkpoint
          │
          │  ZMQ RPC
          ├──────────────► TransferQueueController (Ray Actor)
          │                  - partition metadata
          │                  - index_manager state
          │                  - sampler state
          │
          │  ZMQ RPC (concurrent)
          └──────────────► StorageManager
                              ├── SimpleStorageUnit 0  ──► su_0_<id>.pkl
                              ├── SimpleStorageUnit 1  ──► su_1_<id>.pkl
                              └── ...
```

Both the controller and each storage unit write their data directly to disk from within their own processes. The ZMQ RPC carries only the target file path and an ACK, not the payload — this avoids routing large tensors through the Ray object store.

## Save Order and Consistency

The save sequence is: **controller first, storage units second**.

This ordering is intentional. In the normal data flow, a `put` writes to the storage unit before the controller's `production_status` is updated. At any point in time, storage units hold a superset (or equal set) of what the controller considers produced. Snapshotting the controller first therefore guarantees that every index the controller records as produced is present in the subsequently snapshotted storage units.

If storage were snapshotted first, a race could produce a checkpoint where the controller records an index as produced but the corresponding storage data was not yet captured — consumers would read a missing entry on restore.

**Concurrent `clear` during save**: If `clear_partition` or `clear_samples` runs concurrently with `save_checkpoint`, the controller snapshot may reference indexes that are subsequently deleted from storage before the storage snapshot is taken. The resulting checkpoint reflects a mixed view. To avoid this, callers should ensure no concurrent `clear` operations are issued during `save_checkpoint`. In practice, checkpoints are typically taken at step boundaries where no clearing is in progress.

## Multi-Node Requirements

`checkpoint_dir` must reside on a shared filesystem accessible from all nodes (NFS, GPFS, Lustre, etc.), because each StorageUnit writes its file directly to that path. Single-node deployments have no such requirement.

## Storage Unit Count Matching

On load, the number of storage units in the checkpoint must exactly match the running system. The matching is by **position** (index in the ordered list), not by storage unit ID — since IDs are regenerated on each `tq.init()`, position-based matching supports restart with freshly created actors. A count mismatch raises `ValueError` and aborts the restore.

## Known Limitations

### 1. Controller request thread can hang on checkpoint I/O failure

The controller's request loop currently has no error handling around the checkpoint branches. If `save_checkpoint` or `load_checkpoint` raises (e.g., the path is not writable, or the pickle file is corrupt), the exception propagates up and kills the request thread. The client's `recv_multipart` call will block indefinitely.

**Workaround**: Ensure the checkpoint path is writable and the target file is not corrupt before calling. Verify disk space and file system permissions ahead of time.

---

### 2. Save consistency is not guaranteed under concurrent clears

See [Save Order and Consistency](#save-order-and-consistency) above. Concurrent `clear_partition` or `clear_samples` during `save_checkpoint` can produce a checkpoint whose controller view references storage entries that no longer exist.

**Workaround**: Do not issue `clear_partition` or `clear_samples` while `save_checkpoint` is running. This is naturally satisfied when checkpointing at training step boundaries.

---

### 3. Replacing an existing checkpoint is not fully atomic

The current save sequence is:

```python
if checkpoint_dir.exists():
    shutil.rmtree(checkpoint_dir)   # (1) old directory deleted
tmp_dir.rename(checkpoint_dir)      # (2) new directory moved into place
```

If step (2) fails after step (1) (e.g., cross-device rename, disk full), the old checkpoint has already been deleted and the new one is also cleaned up by the exception handler — both copies are lost.

**Workaround**: Maintain an additional copy of the previous checkpoint (e.g., save to `step_N` while keeping `step_N-1`) so a failure at step N leaves `step_N-1` intact.

---

### 4. Load is not transactional — partial restore has no rollback

The load sequence is:

```python
if meta.get("storage_saved"):
    client.load_storage_checkpoint(...)   # (1) storage restored first
client.load_controller_checkpoint(...)    # (2) controller restored second
```

If step (1) partially succeeds and step (2) fails, the system is left in a mixed state: some storage units hold checkpoint data while the controller still reflects its pre-restore state. There is no rollback path.

**Workaround**: If `load_checkpoint` raises, call `tq.init()` again to reset the system to a clean state before retrying.