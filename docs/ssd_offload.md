# SimpleStorage SSD Offload

> Last updated: 09/20/2026

## Overview

SimpleStorage can keep large field values on local SSD instead of keeping them
in host memory for their full lifetime. This reduces long-lived host-memory
use while preserving the existing TransferQueue APIs.

SSD offload is available only with the `SimpleStorage` backend. Placement is
decided for each field value in each sample. Values whose encoded size is at
or above the configured threshold are stored on SSD; smaller values remain in
memory.

SSD offload is temporary storage, not durable storage. `CLEAR` operations and
`tq.close()` remove offloaded data. Use a checkpoint when data must survive a
TransferQueue restart.

## Quick Start

Enable SSD offload under `backend.SimpleStorage`:

```yaml
backend:
  storage_backend: SimpleStorage
  SimpleStorage:
    ssd_offload:
      enabled: true
      path: /local/ssd/transfer-queue-job-a
      threshold_bytes: 1048576
      glibc_mmap_threshold_bytes: 1048576
```

## Configuration

| Config key | Default | Description |
|------------|---------|-------------|
| `backend.SimpleStorage.ssd_offload.enabled` | `false` | Enables SSD offload for SimpleStorage. |
| `backend.SimpleStorage.ssd_offload.path` | `null` | Target directory for SSD offload. TransferQueue creates it, including missing parent directories, if needed. |
| `backend.SimpleStorage.ssd_offload.threshold_bytes` | `1048576` | Minimum encoded size, in bytes, for storing one field value on SSD. The value must be greater than zero. |
| `backend.SimpleStorage.ssd_offload.glibc_mmap_threshold_bytes` | `1048576` | glibc allocation size at which storage actors prefer independently releasable mappings. Set to `null` to leave the process environment unchanged. |

`glibc_mmap_threshold_bytes` is independent of the SSD placement threshold. It
sets `MALLOC_MMAP_THRESHOLD_` only for storage actors and only when SSD offload
is enabled. Lower values can reduce RSS retained after transient allocations
are freed, but can increase mapping system calls and page faults. The setting
is effective only with glibc and disables its dynamic mmap-threshold adjustment.

### Path ownership

TransferQueue stores each run under
`<path>/transfer_queue_ssd_offload/<run_id>/`.

The configured `path` must be a directory, or a path that TransferQueue can
create. The `transfer_queue_ssd_offload` child must be a real directory, not a
symbolic link.

Each storage unit owns only its `<run_id>/<storage_unit_id>` directory. A
successful `tq.close()` removes the current run's unit directories without
deleting sibling runs or unrelated files under `transfer_queue_ssd_offload`.

## Runtime Directory Layout

The current implementation stores active SSD data under this layout:

```text
<path>/
└── transfer_queue_ssd_offload/
    └── <run_id>/
        └── <storage_unit_id>/
            └── <00..ff>/
                ├── <uuid>.bin
                └── .tmp-<uuid>    # Present only while a value is being written
```

One `run_id` is created for each TransferQueue initialization and shared by its
storage units. Each storage unit has its own directory and creates `00` through
`ff` subdirectories.

Each active SSD-backed `(field, global_index)` value is stored in one `.bin`
file. The file contains only the encoded value. Its name does not contain the
field or global index; the storage unit keeps that mapping and the decode
metadata in memory.

TransferQueue first writes a `.tmp-<uuid>` file, then renames it to
`<uuid>.bin` after the full value has been written. If a write fails,
TransferQueue attempts to remove its temporary file.

This directory layout is an internal implementation detail and may change.
Applications should access offloaded values through TransferQueue APIs rather
than reading these files directly. Checkpoint files use the separate layout
described in [Checkpoint Integration](#checkpoint-integration).

## Data Placement

Placement is based on the encoded size of one field value for one sample. The
threshold is inclusive:

```text
encoded size <  threshold_bytes  -> host memory
encoded size >= threshold_bytes  -> local SSD
```

For a batched tensor, each item along the first dimension is considered one
sample. Different fields in the same sample can be placed in different tiers.
Overwriting a key can also move its values between memory and SSD.

The following value types can be stored on SSD:

| Value type | SSD representation |
|------------|--------------------|
| Dense PyTorch tensor | Raw tensor bytes, dtype, and shape |
| NumPy array without object dtype | Raw array bytes, dtype, and shape |
| NumPy array with object dtype | Pickle payload |
| `bytes` | Original bytes |
| Other Python value that `pickle` can serialize | Pickle payload |

Nested or sparse PyTorch tensors and values that cannot be encoded remain in
memory. A non-CPU tensor is copied to CPU before it is written to SSD.

GET reads SSD-backed values and reconstructs their original types. SSD file
references are internal and are not returned to the application.

The storage unit encodes values, performs SSD I/O, and keeps the file
references. The controller continues to manage metadata and scheduling; it
does not move SSD payloads.

## Data Cleanup

TransferQueue removes SSD files at these points:

- `CLEAR` deletes the files for the cleared samples.
- A successful overwrite deletes files that are no longer referenced.
- A successful `tq.close()` stops each SimpleStorage unit and deletes that
  unit's `<run_id>/<storage_unit_id>` directory.

TransferQueue does not sweep sibling or stale run directories. A process that
exits without completing `tq.close()` can leave its run directory behind for
manual cleanup. The configured `path` and `transfer_queue_ssd_offload` child
directory remain in place.

## Multi-Node Requirements

All storage units receive the same `ssd_offload.path`. Each unit uses that path
on its own node, so SSD offload does not require a shared filesystem.

Checkpoint directories must be shared across nodes; see [Checkpoint: Save and
Restore System State](checkpoint.md#multi-node-requirements).

## Checkpoint Integration

SSD offload works with `tq.save_checkpoint` and `tq.load_checkpoint`. Each
storage unit has a `.pkl` file. Its `.pkl.blobs/` directory exists only when
that unit has SSD-backed values:

```text
simple_storage/
├── su_<position>_<id>.pkl          # In-memory values and SSD metadata
└── su_<position>_<id>.pkl.blobs/   # SSD-backed values, when present
```

Saving a checkpoint copies SSD files directly into the `.blobs` directory. It
does not read and decode the full SSD tier into host memory. Loading the
checkpoint copies the blobs into the current SSD offload directory.

An SSD checkpoint must be loaded with SSD offload enabled. A checkpoint made
without SSD offload can be loaded with SSD offload enabled; its values are
placed using the current `threshold_bytes` setting.

See [Checkpoint: Save and Restore System State](checkpoint.md) for checkpoint
consistency, replacement, and multi-node rules.

## Metrics

When metrics are enabled, the controller endpoint exposes these metrics for
each storage unit:

| Metric | Description |
|--------|-------------|
| `tq_storage_ssd_offload_enabled` | `1` when SSD offload is enabled, otherwise `0` |
| `tq_storage_ssd_active_values` | Number of active field values stored on SSD |
| `tq_storage_ssd_active_bytes` | Logical encoded bytes held by active SSD-backed values |
| `tq_storage_ssd_fallback_values_total` | Cumulative number of field values retained in memory because SSD encoding was unavailable |

`tq_storage_ssd_active_bytes` is logical payload size, not filesystem usage.
Filesystem blocks, directories, and temporary files are not included. See
[Prometheus Metrics & Grafana Dashboard](metrics.md) for metrics setup.

`tq_storage_ssd_fallback_values_total` is monotonic for the lifetime of a
storage unit. A non-zero or increasing value means SSD offload is enabled but
some values cannot use any supported SSD encoding and remain in DRAM. Clearing
or overwriting those values does not decrease the counter.

## Choosing a Threshold

A lower threshold moves more values to SSD and can reduce long-lived host
memory use. It also creates more files, consumes more filesystem inodes, and
increases SSD I/O. A higher threshold keeps more values in memory and reduces
SSD work.

Choose the threshold using the real field sizes and access pattern of the
workload. The benchmark under
[`scripts/performance_test_ssd_offload`](../scripts/performance_test_ssd_offload/README_PERFTEST_SSD_OFFLOAD.md)
compares host-memory mode with SSD offload and reports throughput, storage-unit
RSS, and active SSD bytes.

## Known Limitations

### GET loads SSD-backed values into memory

GET reads each requested SSD file and reconstructs the value in host memory.
SSD offload reduces long-lived memory use, but it does not remove temporary
memory use while values are read or encoded.

### `data_parser` outputs should not share backing storage across samples

If a `data_parser` returns differently sized tensor or NumPy views backed by
the same allocation, an in-memory view can keep the entire allocation resident
after another view is offloaded. When parsing URLs or file paths, return an
independently owned value for each sample, or copy retained views before
returning them.
