# Working rules for AI assistants

## Code standards (hard rules, not preferences)

The priority order is explicit: **human readability comes first; coding-agent
traceability is the minimum gate.** A human should understand code in one pass,
and an agent must be able to trace a feature from a public API or configuration
entry through controller, metadata, sampler, and storage logic to observability
and tests without reconstructing hidden control flow.

1. Over-complex or hard-to-follow code is a **bug**, not a style issue.
2. Reduce complexity and line count — prefer deleting code over adding it.
3. Preserve the control-plane/data-plane boundary: controllers manage metadata
   and scheduling; storage managers, clients, and backends move payloads.
4. No over-encapsulation: add an abstraction only for real nontrivial reuse, an
   important invariant, or an existing sampler/storage interface.
5. Do NOT remove public APIs, backend support, async/distributed behavior,
   metadata semantics, defaults, or observability in the name of simplicity.
6. Prioritize behavior reachable through supported APIs, configurations,
   backends, tutorials, or tests; do not fix states the system cannot enter.

Details: `.claude/skills/simplicity-first` (invoke before any code change).

## Comments

Concise "why" only, 2-4 lines, written for an external reader: no job ids,
commit hashes, single-run metrics, or internal cluster paths; keep upstream
issue/PR links.

## Workflow

- Reviews report findings only; fixes ship as one minimal diff per issue
  after approval.
- Verify with `python -m compileall -q transfer_queue tutorial tests` and
  `python -m pytest -q`; use the relevant optional-backend environment for
  integration tests and validate shell scripts with `bash -n`.
