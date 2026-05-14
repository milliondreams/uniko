# Investigation: conv-26 bench reveals two distinct post-shutdown bugs

## Setup

- Fresh `rm -rf data/kb_nlp_phaseAB/conv-26` + bench run
- conv-26: 419 turns, 199 questions
- After consolidation: 415 Facts created, 901 Observations processed
- One bench-agent Participant + 199 Episodes recorded during question loop
- Wired `verify_label_visible` / `verify_node_visible_by_label` after every
  cycle / bench-agent / record_episode write — uses a label-anchored MATCH

## Key in-session result

**Zero verify warnings during the entire run.** All label-anchored MATCHes
issued immediately after the writes returned ≥1 row. So the writes were
visible to label-scan at write time, with all expected properties.

## Post-shutdown probe (`probe_kb` + `probe_unconstrained` + `probe_props`)

| Vertex | Label-scan? | Properties? | Notes |
|---|---|---|---|
| Speaker Participants (vid=1, vid=3) | ✓ | full (`participant_id`, `name`, `kind`, `first_seen`, `last_seen`) | created during ingest |
| bench-agent Participant (vid=2153) | ✗ | **NONE** — node has `labels=["Participant"]` but zero property entries | created post-consolidation via `merge_node` |
| ConsolidationCycle (vid=2152) | ✓ | full (`cycle_id`, `agent_id`, …, `facts_created=415`) | created post-consolidation via `create_node` |
| Episode vertices (vid=2154..) | ✗ | full (`episode_id`, `action_type`, `outcome`, `timestamp`, `state`, `embedding`, `importance`) | 199 created during question loop |
| Facts (415) / Observations (901) / SUPPORTED_BY edges (901) | ✓ | full | bulk-written during ingest/consolidation |

## Two distinct bugs

### Bug A: label-scan invisibility post-reopen

Episodes and the bench-agent Participant are both **fully persisted** (reachable
via edges, `labels(n)` returns `["Episode"]` / `["Participant"]`) but invisible
to `MATCH (n:Episode)` / `MATCH (n:Participant)`.

ConsolidationCycle written in the *same session* doesn't have this problem.
The only material difference at the write call-site: ConsolidationCycle is
written ONCE before the question loop and the next operation gives uni-db
~30s of idle time before the next write.  Episodes and bench-agent are
written in rapid succession with reads interleaved.

Hypothesis: under interleaved read/write tx pressure, uni-db's per-label
scalar index (Hash on `episode_id` / `participant_id`) doesn't get its
in-flight L0 entries promoted into a persistent index at flush time.  The
*data* lands in the vertex storage; the *index* state for label-anchored
scan is what gets dropped.

### Bug B: bench-agent Participant lost all properties

vid=2153 has `labels=["Participant"]` but **zero property entries** after
reopen.  Same `merge_node` call site as speakers (which retain everything).
In-session, the label-anchored MATCH on `WHERE n.participant_id = $eid`
succeeded — so the `participant_id` property existed at that point.

Hypothesis candidates (not yet narrowed):

- An auto-flush window crosses during `merge_node`'s internal two-step
  (get_node + create_node), and the create_node's property payload lands
  in a buffer that gets discarded.
- A `update_node` somewhere later writes an empty property set to vid=2153
  (e.g., the `embed_episode` `MATCH (n) WHERE id(n) = $vid SET n.embedding = …`
  picking up the wrong VID).
- Race between the bench-agent insert and the consolidation write phase
  that just completed.

The unlabeled-MATCH probe shows the vertex still has labels but empty
properties — that's a serialization-side anomaly, not just a missing index.

## What this tells us

1. The "vertex loss" framing from the earlier writeup was wrong.  Vertex
   data is persisted; the bugs are at the *index materialization* layer
   (Bug A) and *property serialization* layer (Bug B).
2. The bench mitigation worked: 0 false-negatives in-session, so the
   verifies confirmed in-session integrity.  But they didn't (couldn't)
   detect a post-shutdown regression.
3. ConsolidationCycle survived this time (didn't last time) — likely
   because we added a label-anchored `verify_label_visible` query right
   after its write, which apparently forces uni-db to materialize the
   ConsolidationCycle label index physically.

## Recommended next actions

1. **Add a label-anchored MATCH right after `merge_node` for bench-agent**
   and after every Episode.  We already do this for Episode (and it didn't
   help post-reopen) — but the Cycle case suggests the query needs to run
   on a *different session*, or possibly an explicit `db.flush()` between
   the write and the verify.

2. **Force `db.flush()` after `write_consolidation_cycle` and after the
   bench-agent `merge_node`.**  Cheap, and isolates whether auto-flush
   timing is the culprit.

3. **Reproduce Bug B in `bugs/unidb-persistence-loss/repro.rs`**: a single
   `MERGE` with `kind` + `name` + `participant_id` after ~1000 prior writes,
   shutdown, reopen, check properties.  If properties vanish, that's a
   minimal repro of the bench-agent issue.

4. **Reproduce Bug A in `repro.rs`** by interleaving reads and writes — the
   current scenarios 8 and 9 don't.  Each iteration should do: write,
   read (label-anchored MATCH on a *different* label or by id), write…
   to mimic the bench question loop.

## Standalone reproduction attempts

`bugs/unidb-persistence-loss/repro_conv26.rs` tries to provoke both bugs
using only `uni-db` (no `uniko-store` dependency).  The workload mirrors
the bench:

- 900 Bulk vertices with 768-dim embeddings (Observation analogue)
- 400 Fact vertices with embeddings
- 1 ConsolidationCycle + 400 CREATED + 400 PROCESSED edges
- 1 bench-agent Participant via two-step merge (read-then-create)
- 200 Episodes in a tight interleaved loop: read participant, read prev
  episode, CREATE Episode + RECORDED_BY edge, FOLLOWED_BY edge, labelless
  SET embedding
- 10 dummy labels with vector + FullText indexes matching the bench's
  schema burden

**Neither bug reproduces.**  All 200 Episodes visible to label-scan post
reopen, bench-agent Participant retains all properties, ConsolidationCycle
fine.

That's a strong signal: **the trigger is something in the
`uniko-store`/`uniko-bench`/xervo layer**, not raw uni-db behavior.
Candidates that distinguish the bench from this repro:

- **uniko-store's `KnowledgeBase` wrapper**: holds an `Arc<Uni>`, shares
  it across multiple Arc clones for the xervo prefetch, recall layer,
  embedding pipeline; the shutdown path uses `Arc::try_unwrap` which can
  silently skip if any clone is still alive.
- **xervo `prefetch_all` runs concurrently with first writes** when
  `open_with_xervo` is called, possibly interacting with auto-flush.
- **uniko-store's `merge_node` two-step** has subtleties not captured by
  the raw repro: it calls `validate_label`, `validate_property_name`,
  `build_inline_props` which may interact with the schema state.
- **uniko-bench's ingest path** uses `batch_create_nodes_fast` and
  `batch_create_edges_fast` (bulk_writer), not single `CREATE` per tx.

## Revised assessment (was overstated earlier)

I previously claimed both bugs were "almost certainly uni-db".  The
failure to reproduce in pure uni-db forces a more honest answer:

- **Bug A (label-scan invisibility)**: still likely uni-db, since
  `labels(n)` returns the label but `MATCH (n:Label)` doesn't find the
  row — that's a planner/index disagreement no application code can
  cause directly.  But the trigger requires the uniko-store/xervo layer.
- **Bug B (property loss)**: now genuinely uncertain.  Without a
  standalone repro, I cannot rule out that uniko-store's `merge_node`
  has a subtle race or that something else writes back to that vid.

## Most productive next step

Lift one constraint at a time:

1. **Depend on `uniko-store` in a tiny binary** (not the whole bench) and
   re-run the same workload via `KnowledgeBase::open_with_xervo`,
   `kb.merge_node(...)`, `kb.create_node(...)`, etc.  If the bug
   appears, the issue is in uniko-store (or its interaction with
   xervo/uni-db).  If still no repro, the bench harness itself is
   involved.
2. **Disable xervo prefetch** (`open_with_xervo_no_prefetch`) in the
   bench and re-run — isolates whether xervo concurrency is the trigger.
3. **Add post-write `db.flush()`** after each `record_episode` and after
   the bench-agent merge — checks the auto-flush race hypothesis.

## Artifacts

- `crates/uniko-bench/examples/probe_kb.rs` — label-anchored counts
- `crates/uniko-bench/examples/probe_unconstrained.rs` — edge-traversal counts
- `crates/uniko-bench/examples/probe_props.rs` — raw property dumps
- `crates/uniko-bench/src/main.rs` — `verify_label_visible` /
  `verify_node_visible_by_label` mitigation
- `bugs/unidb-persistence-loss/repro_conv26.rs` — standalone repro
  (does NOT reproduce)
- Bench run log: `/tmp/bench_conv26_verify.log`
