# Investigation: `db.flush()` fails when a property named `ext_id` has a scalar index

## Original suspicion (turned out wrong)

uniko2's conv-26 bench produced 415 Facts and 901 SUPPORTED_BY edges that
persist correctly, but the ConsolidationCycle audit node + bench-agent
Participant + Episodes recorded after consolidation all disappear after
shutdown + reopen. The hypothesis was that late writes in a session's
sequence are silently dropped on shutdown.

## Standalone repro result

After building a standalone Cargo project that depends only on
`uni-db = { path = "../../../uni/crates/uni" }` (no uniko-store layer,
no shared workspace), the late-write hypothesis **fails to reproduce**.

Scenarios 1, 2, 3, 4, 5 (single late, bulk-then-late, late-then-bulk,
explicit-flush, MERGE) all PASS — every committed write survives
shutdown when property names follow uniko-store's `<label>_id`
convention.

## What the repro did find

A separate uni-db / Lance interaction bug: **any scalar index on a
property literally named `ext_id` makes `db.flush()` fail** with

    Failed to create table 'vertices_X':
    lance error: LanceError(Schema): Duplicate field name "ext_id"

Truth table (from `repro.rs` scenarios 0–0f):

| Schema                                                | Result            |
|-------------------------------------------------------|-------------------|
| No index                                              | PASS              |
| Hash index on `ext_id` (String)                       | FAIL on flush     |
| BTree index on `ext_id` (String)                      | FAIL on flush     |
| Hash index on `ext_id` (String) + MERGE write         | FAIL on flush     |
| Hash index on `ext_id` (String) + a second property   | FAIL on flush     |
| Hash index on `name` (String, different name)         | PASS              |
| Hash index on `k` (Int64)                             | PASS              |

So the trigger is the combination *(scalar index)* × *(property literally
called `ext_id`)*. uniko-store names its external-id properties
`participant_id`, `fact_id`, `episode_id`, `cycle_id`, etc., so it
sidesteps the bug — but anyone who declares `ext_id` literally is
broken on flush.

## Suspected cause

uni-db's storage layer almost certainly uses an internal `ext_id`
column on every label's Lance table (the column behind
`merge_node`'s ext_id lookup). When the user also declares a
property named `ext_id` *and* adds an index on it, the index
materialization adds another `ext_id` field to the Arrow schema,
collides with the internal one, and Lance rejects the table create
during flush.

## What still needs investigating in uniko2

The original conv-26 persistence regression is **not** explained by
this Lance bug. Possible remaining causes:

- `merge_node("Participant", "participant_id", "bench-agent-...", ...)`
  silently fails inside the question loop without surfacing an error.
- The bench's `Arc::try_unwrap(kb)` is reporting success but `shutdown()`
  isn't propagating to the persistent backend for late-arriving writes
  in some edge case.
- The probe query is wrong (unlikely — it works on the May 12
  `conv-30` KB which has 3 cycles).

This needs a separate, focused investigation — not blocked on the
Lance bug.

## Build & run

```bash
cd bugs/unidb-persistence-loss
cargo run --release
```

Each scenario prints PASS / FAIL.
