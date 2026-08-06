---
name: simplicity-first
description: Simplicity and traceability gate for all TransferQueue code written, fixed, or reviewed. Invoke before changing code or reviewing a diff, especially across public APIs, controller and metadata logic, samplers, storage backends, asynchronous execution, or metrics.
---

# Put simplicity first

Keep code readable in one pass. Make each behavior traceable from a public API
or configuration entry through controller, metadata, sampler, and storage
logic to its tests and observability. Treat hidden control flow and unnecessary
indirection as correctness problems, especially across asynchronous or
distributed boundaries.

1. **Keep execution paths explicit.** Make state transitions, partition and
   sample ownership, retry or cleanup behavior, and async boundaries visible at
   the point where they matter. Avoid clever control flow that obscures which
   component mutates metadata or moves data.

2. **Use the smallest readable change.** Delete redundancy before adding new
   machinery. Do not add configuration, return fields, or extension points for
   hypothetical future needs.

3. **Preserve the control-plane/data-plane boundary.** Keep metadata and
   scheduling in the controller layer and payload movement in storage managers,
   clients, and backends. Do not route payloads through the controller or bind a
   public API to one backend merely to shorten a local implementation.

4. **Avoid thin abstractions.** Keep one-off logic inline when it remains clear.
   Add a helper or type only when it removes real nontrivial duplication, names
   an important invariant, or implements an existing interface such as a
   sampler, storage manager, or storage client.

5. **Preserve contracts and capabilities.** Keep public KV, client, and
   dataloader behavior; metadata and sampler semantics; supported storage
   backends; async and distributed behavior; metrics; and existing defaults.
   Simplify implementations without silently narrowing the behavior surface.

6. **Work from reachable behavior.** Prioritize issues reproducible through
   supported APIs, configurations, backends, tutorials, or tests. Still protect
   internal invariants required for concurrency, cleanup, and data consistency;
   do not invent fixes for states the system cannot enter.

## Checklist before finishing any change

- Can a reader trace the change from `transfer_queue.interface`,
  `TransferQueueClient`, or `StreamingDataLoader` through the controller and
  storage path to a focused test?
- Are control-plane metadata and data-plane payload responsibilities still
  separate?
- Are partition isolation, production and consumption status, sampler behavior,
  cleanup, and async failure handling preserved where relevant?
- Could this diff be half the size? If unsure, make it smaller.
- Does every new abstraction remove real complexity or implement an existing
  TransferQueue interface?
- Does any public API, configuration default, metadata shape, or backend
  contract change? If so, verify every affected caller and test.
- Keep comments to concise reasons or invariants; omit run-specific data and
  internal paths, while retaining useful upstream issue or PR links.
- One problem = one minimal diff. Do not batch unrelated "improvements".
