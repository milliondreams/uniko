# uni-db Workarounds & Limitations — Catalog

**Generated:** 2026-05-30 by a per-crate source audit (one agent per crate).
**Upstream-verified:** 2026-06-03 against uni HEAD `3911cbc24` (uni-db **2.0.0**); prior
pass 2026-05-31 against `3d5e849c0` (1.3.0) — see §0 (newest first).
**Scope:** every place in `uniko2` where code is shaped differently than it would
be if uni-db were perfect — query reformulations, retry/fallback wrappers,
defensive schema/type hints, in-Rust post-processing, and serialization for
missing atomicity. Spot-checked against source; each entry marks whether it is
**comment-documented** or **inferred**.

> This is the index. The deep repros live alongside it in `bugs/`:
> [`uni-db-edge-id-vid-planner.md`](uni-db-edge-id-vid-planner.md),
> [`uni-db-rmw-primitives-wishlist.md`](uni-db-rmw-primitives-wishlist.md),
> [`unidb-slow-pattern-in-where/`](unidb-slow-pattern-in-where/),
> [`unidb-persistence-loss/`](unidb-persistence-loss/).

Per memory policy ([never edit `../uni/` directly](../bugs)): all of these are
either filed upstream as `rustic-ai/uni-db#NN`, captured as an in-tree repro, or
flagged below as **not yet filed**.

---

## 0. Upstream verification log

**2026-06-03 — re-verified against uni HEAD (`3911cbc24`, uni-db 2.0.0).** uni jumped
1.3.0→2.0.0 ("upgrade all deps to latest": Lance 6→7, arrow 57→58, datafusion 52→53,
object_store 0.11→0.13). The #39 (`d11ac2cbc`) and #55 (`48f3a4ed3`+`0d2a2cebc`+`e711c87e0`)
fixes remain in history (all ancestors of HEAD). The material change is **#69**:

| Issue | Verdict @ 2.0.0 | Fix / evidence | uniko impact |
|-------|-----------------|----------------|--------------|
| **#69** single-node `MERGE`-in-`UNWIND` | 🟡 **substantially fixed** for the entity-upsert shape | uni `ab405408a` "broaden MERGE-in-UNWIND fast path to un-indexed keys". `execute_merge` (`write.rs:1842`) detects the single-node/single-label shape once, builds **one** batch-level L0 snapshot (`merge_l0_existing`, `:1855`), and resolves each row via `execute_merge_row_indexed` with an O(1) lookup / single filtered scan — **no per-row `LogicalPlan`**. Still a `for row in rows` loop (`:1859`), but the per-row *planning* that made batched MERGE no faster than a loop is gone. Multi-node / relationship / non-scalar-key MERGE still falls to the per-row planning path (`:1905`). | The 4-phase entity upsert (`extract/ner/dedup.rs`) is now a **removal candidate**: its shape — `MERGE (e:Entity {entity_id})`, Hash-indexed key — is exactly what the fast path serves. **Do not remove on faith: benchmark a batched `UNWIND $rows AS r MERGE (e:Entity {entity_id:r.id})` against the 4-phase path first.** Supersedes the 2026-05-31 "❌ NOT fixed" verdict below. |

**uniko2 adoption work for 2.0.0** (in working tree, uncommitted): `object_store` 0.11→0.13
(`Cargo.toml`); `roaring` 0.11.3→0.11.4 in `Cargo.lock` (Lance 7 needs `RoaringBitmap: Eq`,
added in .4); `blob_store/s3.rs` — `use object_store::ObjectStoreExt` (0.13 moved
`get/put/head/delete` to an ext trait) + `ObjectMeta::size as u64` (was `usize`);
`examples/compare_facts.rs` — `#![recursion_limit="256"]` (deeper async-future layout);
`ner/onnx.rs:83` — pre-existing `String + &String` build error fixed (orthogonal to the bump).
`uni-xervo` already 0.13.0. Result: `cargo check --workspace --all-targets --no-default-features`
green (CUDA path + `nextest` not yet run at time of writing).

**2026-05-31 — verified against uni HEAD (`3d5e849c0`, uni-db 1.3.0), by source + behavior:**

| Issue | Verdict | Fix / evidence | uniko impact |
|-------|---------|----------------|--------------|
| **#39** `similar_to()`→0.0 on FTS | ✅ **FIXED** | uni `d11ac2cbc` "Closes #39" — `fts_search_batch` now passes `QueryContext`, so unflushed L0 data is scored (`similar_to_expr.rs:663`). uni's 4 regression tests pass. | **No workaround to remove.** uniko's FTS leg uses `CALL uni.fts.query` (`fulltext.rs:28`), which was *never* broken by #39. The RRF-in-Rust in `hybrid.rs` is a deliberate tiered-ranking design, not a #39 workaround — **RC5 was over-attributed; corrected below.** Repro `unidb-similar-to-fts-zero-score/` **retired 2026-06-03**. |
| **#55** `get_edges` scales w/ graph | ✅ **FIXED** (scaling) | uni `48f3a4ed3` + `0d2a2cebc` + `e711c87e0` short-circuit irrelevant frozen CSR segments. `get_edges_scaling_repro` plateaus (does **not** grow linearly across +3000 nodes). | **No live code to remove.** `store/tests/perf/` repros **retired 2026-06-03** — re-verified on 2.0.0: `get_edges` 4.4× & plateaus, `observed_in_growth` 1.7× (was ~5×). ⚠️ A constant post-first-flush latency step remains (uni's deferred follow-up). |
| **#69** `MERGE` per-row loop | ❌ **NOT fixed** *(@1.3.0 — superseded by 2026-06-03 / 2.0.0 above)* | `execute_merge` (uni `write.rs:1458`) is still `for row in rows` with a per-row plan in `execute_merge_match`. The UNWIND-IN-list/HashJoin commits fix MATCH read legs only, **not** MERGE; no uni test asserts bulk MERGE. | At 1.3.0: **keep** the 4-phase entity upsert. At 2.0.0 (`ab405408a`): now a removal candidate — see top of §0. |

**Adoption prerequisite.** uni HEAD bumped `uni-xervo` 0.12→0.13 (commit `41e4fc263`).
uniko2 will not compile against it until the workspace `Cargo.toml` pin matches
(`uni-xervo = "0.13.0"`) — otherwise two incompatible `uni_xervo::ModelRuntime`
types land in the graph and `storage/mod.rs:279` fails to typecheck. After the
bump: workspace test binaries build clean; `uniko-store` suite 179/179 green.

---

## 1. Root causes (the recurring ones)

These few uni-db gaps generate most of the workarounds below. Fixing them
upstream would let us delete the listed sites.

| # | Root cause | Filed | Workaround sites it drives |
|---|------------|-------|----------------------------|
| RC1 | **`WHERE` on `DateTime`/Temporal properties is unsupported** (only `ORDER BY` on Temporal works); planner returns Temporal as RFC-3339 *strings* | not filed | `memory/episode.rs:175`, `memory/recall/mod.rs:1112` (phase2_temporal), `memory/consolidation.rs:805`, `bench/query.rs:329` |
| RC2 | **No server-side atomic `SET` / no serializable commit (last-writer-wins) / no row lock / no CAS** | [`rmw-primitives-wishlist.md`](uni-db-rmw-primitives-wishlist.md) | entire `store/locks.rs` (`StripedLocks`) + 6 RMW call sites; `store/facts.rs` Fact phase-split + Rust-side count arithmetic; `cortex/topics.rs` BELONGS_TO dup-swallow |
| RC3 | **`MERGE` runs a per-row executor loop** (no bulk fast-path) | `#69` — 🟡 **substantially fixed @ 2.0.0** (uni `ab405408a`; single-node shape skips per-row planning) | `extract/ner/dedup.rs` 4-phase entity upsert — **removal candidate pending bench**, see §0 (2026-06-03) |
| RC4 | **`MERGE`'s internal CREATE checks NOT NULL before `ON CREATE SET`** | not filed | `store/nodes.rs:239` `merge_node` get-then-create split |
| RC5 | **`similar_to()` returns `0.0` on a FullText-indexed field** (BM25 path works) | `#39` — ✅ **FIXED** (uni `d11ac2cbc`) | **none** — over-attributed; `hybrid.rs` uses `CALL uni.fts.query` (never broken), see §0 |
| RC6 | **`get_edges` latency scales with total graph size, not out-degree** (no CSR adjacency) | `#55` — ✅ **FIXED** scaling (uni `48f3a4ed3`); const post-flush step remains | `extract` denormalized ABOUT edges are recall-expressiveness (keep); `store/tests/perf/` repros **retired** (re-verified 2.0.0) |
| RC7 | **`UNWIND … MATCH WHERE id(n)=p` only gets the HashJoin rewrite under specific shapes**; without a label hint the planner does a multi-label scan per row (~18 ms/row) | `rustic-ai/uni-db#53`, `#54` | `extract/ner/dedup.rs` (`:Entity` label hint, `id()`-equality shape); `store/storage/batch.rs` |
| RC8 | **`id(r)` on a relationship lowers to `r._vid`** (planner bug) | [`edge-id-vid-planner.md`](uni-db-edge-id-vid-planner.md) | `store/storage/edges.rs:341` (`delete_edge`), `:398` (`update_edge`) |
| RC9 | **Typed `List<T>` column with no producer → Arrow inference coerces `List<Float32>`→`List<Utf8>`** | not filed (matches known inference fallback) | `store/schema/chunks.rs:30` defers typed columns into a JSON bag |
| RC10 | **Writing `Value::String(rfc3339)` into a `DateTime` column is silently dropped at flush** (row invisible to label-MATCH) | not filed | `store/types.rs:19` mandatory `datetime_value()` helper |
| RC11 | **A scalar index on a property literally named `ext_id` makes `flush()` fail** (Lance duplicate field) | [`unidb-persistence-loss/`](unidb-persistence-loss/) | naming convention: all external ids are `<label>_id` (`fact_id`, `participant_id`, …) |
| RC12 | **Locy rule runtime may not execute stdlib rules** in the pinned uni-db version | not filed | `cortex/procedures.rs` Locy→Cypher fallback; `memory/rules/stdlib.rs:117` best-effort `create_rule` |
| RC13 | **Missing scalar `DataType`s** — `Bytes` ✅ **FIXED** (`#50`, 2.0.0); `Int16` ❌ still missing | `Bytes` = `#50` (done) | `Bytes`: **adopted** — `store/schema/artifact_content.rs` uses `DataType::Bytes` (`bytes`, `audio_fingerprint`); `bench/ingest.rs` TODO de-referenced. `Int16`: `store/schema/artifacts.rs:32` (channels still `Int32`) — **keep** |
| RC14 | **`MERGE`-an-edge between two known ids isn't expressible** (requires both endpoints by `id()` in one pattern) | not filed | `store/migrations.rs:90` manual two-step; `cortex/topics.rs` `create_edge`+dup-swallow |

**Out of scope but logged for completeness** (these are ONNX/ORT embedding-runtime
or LLM-provider limits, *not* graph-DB bugs): embed-batch chunking to dodge ORT
arena OOM (`extract/embedding/mod.rs:97`, `memory/recall:556`), serialized
BGE-small ORT mutex, GPT/o-series temperature handling, LLM retry/backoff,
`#![recursion_limit = "256"]` for uni-db's datafusion/sqlparser type depth.

---

## 2. Filed upstream issues referenced in-tree

| uni-db issue | Symptom | Where it bites |
|--------------|---------|----------------|
| `#39` ✅ FIXED | `similar_to()` → `0.0` on FullText field | RC5; repro `unidb-similar-to-fts-zero-score/` — **retired 2026-06-03** |
| `#40` | `flush()` "non-nullable column" on empty labels | bench repro (now deleted from tree) |
| `#49` | auto-embed insert latency O(n) when label also has a `DateTime` prop | bench repro (deleted) |
| `#50` ✅ FIXED | `DataType::Bytes` unavailable | RC13; **adopted** in `store/schema/artifact_content.rs`; `bench/ingest.rs` TODO de-referenced |
| `#53`/`#54` | `UNWIND…MATCH WHERE id()=p` HashJoin rewrite only under specific shapes; UNWIND-edge ~100× slow | RC3/RC7; `store/tests/bug_repros/unwind_edge_repro.rs` |
| `#55` ✅ FIXED (scaling) | `get_edges` scales with graph size not out-degree | RC6; `store/tests/perf/` — **retired 2026-06-03** (re-verified 2.0.0) |
| `#56` | label disjunction `(n:A|B)` no-op, then `union_schema` panic on differing column counts | `store/tests/bug_repros/label_disjunction*` (uni-db has since fixed) |
| `#69` ❌ STILL OPEN | `MERGE` per-row executor loop (verified uni `write.rs:1458`, 2026-05-31) | RC3; `extract/ner/dedup.rs` — keep workaround |

---

## 3. Full catalog by crate

Legend — **evidence:** `comment` (code says so) · `inferred` (deduced from shape).
**status:** `filed` · `repro` (in-tree) · `not filed` · `design` (borderline:
uni-db-shaped but a defensible design choice).

### uniko-store  (primary uni-db consumer — richest cluster)

| Site | uni-db issue | Workaround | Evidence | Status |
|------|--------------|-----------|----------|--------|
| `src/types.rs:19` `datetime_value()` | RC10: String→DateTime silently dropped at flush | mandatory helper builds `Value::Temporal(DateTime)` for every DateTime write | comment | not filed |
| `src/locks.rs` `StripedLocks` + sites in `storage/mod.rs:80‑287`, `storage/nodes.rs:223` (`merge_node`), `kb_stats.rs:179` (`bump_modality_presence`), `operations/facts.rs:121` (`upsert_fact_by_triple`), `:324` (`batch_upsert_facts`), `:844` (`record_entity_invalidation`) | RC2: last-writer-wins, no atomic SET/row-lock/CAS | in-process striped async mutex serializes every read-modify-write critical section by key | comment | filed ([wishlist](uni-db-rmw-primitives-wishlist.md)) |
| `src/storage/edges.rs:341` `delete_edge`, `:398` `update_edge` | RC8: `id(r)` lowers to `r._vid` | query the internal `r._eid` column directly | comment | filed ([doc](uni-db-edge-id-vid-planner.md)) |
| `src/storage/nodes.rs:239` `merge_node` | RC4: MERGE CREATE checks NOT NULL before `ON CREATE SET` | split into explicit get-then-(update\|create) | comment | not filed |
| `src/storage/migrations.rs:90` | RC14: edge-MERGE between two ids not expressible | perform the MERGE manually (both endpoints by `id()`) | comment | not filed |
| `src/schema/chunks.rs:30` | RC9: `List<Float32>`→`List<Utf8>` inference fallback | defer typed positional columns; stuff modality scalars into one `CypherValue` JSON `metadata` bag | comment | not filed |
| `src/schema/artifacts.rs:32` | RC13: no `Int16` | declare `channels` as `Int32` | comment | not filed |
| `src/search/hybrid.rs:79` + `fulltext.rs`/`vector.rs` | ~~RC5~~ **NOT a #39 workaround** (corrected 2026-05-31) | separate vector leg + `CALL uni.fts.query` leg fused via Rust RRF (`rrf_fuse`) with tier weights — a deliberate ranking design; the FTS leg never used the broken `similar_to`-on-FTS path | design | #39 ✅ fixed, but unrelated |
| `src/operations/facts.rs:282` (within-batch dedup), `:340` (Phase 1/2/3 split) | RC2: no unique constraint on `fact_id`, no atomic SET | dedup by `fact_id` in Rust (first-wins, sum counts); read-match → create-new → update-existing phases; count arithmetic + certainty upgrade computed Rust-side | comment | filed ([wishlist](uni-db-rmw-primitives-wishlist.md)) |
| `src/operations/facts.rs:634` `extract_btic`, `:662` `find_stale_open_facts` | lossy `toString`→RFC3339 roundtrip on BTIC `valid_at`; `Value` has no `FromValue` | bare property projection + structural extraction of `Value::Temporal(Btic)` in Rust | comment | not filed (borderline) |
| naming convention (repo-wide) | RC11: index on prop named `ext_id` breaks `flush()` | name all external ids `<label>_id` | comment | filed ([persistence-loss](unidb-persistence-loss/)) |

**Repros (`tests/bug_repros/`) — documentation, no live workaround:**
`unwind_edge_repro.rs` (#53, UNWIND-edge ~100× slow, assertion disabled pending fix) ·
`label_disjunction_repro.rs` / `label_disjunction_union_schema_panic.rs` (#56, since fixed) ·
`schema_apply_duplicate_index_repro.rs` (`SchemaBuilder::apply` re-appends indexes ~2^N; hit 60,969 in prod).

> The `tests/perf/` group (`get_edges_scaling_repro.rs`, `_autoembed_repro.rs`,
> `observed_in_growth_repro.rs`) was **retired 2026-06-03** — #55/#53/#54 verified
> fixed on 2.0.0 (`get_edges` 4.4×/plateau, `observed_in_growth` 1.7×, was ~5×).

### uniko-extract

| Site | uni-db issue | Workaround | Evidence | Status |
|------|--------------|-----------|----------|--------|
| `src/ner/dedup.rs:38` `upsert_entities` / `prepare_entity_upsert` / `apply_entity_upsert` | RC3 (#69 MERGE loop) + RC7 (#53/#54 HashJoin shape) + multi-label scan ~18 ms/row + writer-lock hold time | **4-phase split:** (1) read-only `UNWIND…MATCH` for existing ids on a *fresh session before the tx*, (2) bulk `UNWIND…CREATE` for new, (3) `UNWIND…MATCH (n:Entity) WHERE id(n)=u.nid SET` — load-bearing `:Entity` **label hint** + `id()`-equality to force HashJoin, (4) `batch_create_edges_fast` for MENTIONS (plain CREATE) | comment (extensive) | **#69 ❌ still open** (verified 2026-05-31 — keep); `#53`/`#54` HashJoin improved; label-hint cost investigated 2026-05-20 |
| `src/ingest/atomic.rs` + `ingest/message.rs:56` | legacy 3-commits-per-message had no atomicity → "Message persisted with no entities" half-states; auto-embed serializes on one ORT mutex | prep-then-single-tx: all CPU/read work before opening one tx, fold Message+edges+chunks+entities+observations into **one commit** | comment + inferred | not filed |
| `src/ingest/session_chunk.rs:280` ABOUT/Participant edge propagation | RC6-adjacent: multi-hop "obs-chunk reachable via its observations' entities" not expressible at query time | post-hoc Rust loop re-queries `Observation-[:ABOUT]->Entity/Participant`, dedups in a `HashSet`, materializes the full Chunk→Entity / Chunk→Participant edge product | inferred | not filed (denormalization-for-recall; attribution uncertain) |
| `src/lib.rs:1` | uni-db `Session`/`Transaction` transitively carry datafusion/sqlparser AST → trait-solver recursion overflow | `#![recursion_limit = "256"]` | comment | not filed (build-time, OOS) |

### uniko-memory

| Site | uni-db issue | Workaround | Evidence | Status |
|------|--------------|-----------|----------|--------|
| `src/episode.rs:175` `find_previous_episode` | RC1: no `WHERE` on DateTime | `ORDER BY e.timestamp DESC LIMIT 1`, then apply the `[earliest, now]` window in Rust | comment | not filed |
| `src/recall/mod.rs:1112` `phase2_temporal` | RC1: no DateTime predicate; can't express proximity-in-window | 3-arm `UNION ALL` with per-arm `WITH…LIMIT` (a single trailing LIMIT would starve an arm); Fact arm via custom `btic_overlaps()` UDF; flat score 1.0 | inferred | not filed |
| `src/recall/mod.rs:1247` `phase2_graph_activation` | spreading activation / weighted PPR not expressible in Cypher; `UNWIND/MATCH` hydration loses order | host-side `kb.personalized_pagerank_weighted(...)`, second `UNWIND $nids` round-trip to hydrate, **re-sort in Rust** | comment + inferred | not filed |
| `src/recall/intent.rs:480` + `recall/mod.rs:1485` `entity_type_match` | exact string equality on `name`; no possessive/punctuation normalization → `{name:"Caroline's"}` silently returns nothing | `normalize_entity_text` strips trailing punct + `'s` in Rust before binding; `entity_type_match` `format!`-inlines `target_type` with manual `'` escaping | comment | not filed |
| `src/consolidation.rs:805` `try_parse_observation` | RC1: Temporal props come back as RFC-3339 strings | read as `String`, re-parse with `DateTime::parse_from_rfc3339`, fall back to `Utc::now()` | inferred | not filed |
| `src/rules/stdlib.rs:117` `register_stdlib_rules` | RC12: Locy runtime may not support the rule syntax | best-effort `kb.create_rule` — log at debug on error and continue (Rule node still merged) | comment | not filed |
| `src/recall/mod.rs:1405` `session_boost_signals` | — | one round-trip per Fact (`id(f)={nid}` inlined); self-noted "could be batched" | comment | design (likely not a uni-db workaround) |
| `src/nl_to_cypher.rs:158`/`:223` | avoid serializing full uni-db schema (vector-index params bloat the LLM prompt); no exposed read-only AST mode | hand-authored compact NODE/EDGE summary tables; regex read-only guard | comment | design (prompt-cost choice) |
| `src/lib.rs:8` | same recursion overflow as extract | `#![recursion_limit = "256"]` | comment | not filed (build-time, OOS) |

> `value_convert.rs` (Temporal↔RFC3339 string handling) is the shared substrate
> these DateTime workarounds lean on, not an independent site.

### uniko-cortex

| Site | uni-db issue | Workaround | Evidence | Status |
|------|--------------|-----------|----------|--------|
| `src/procedures.rs:115` + `:374` `fallback_sequence_query` | RC12: Locy `execute_rule("sequence_detector")` may fail | on `Locy` error, re-run the same detection as hand-written Cypher (`MATCH (e1)-[:FOLLOWED_BY]->(e2) … count(*) … WHERE n>=$thr`) | comment | not filed |
| `src/procedures.rs:126` + `:408` `pick_string`/`pick_i64` | Locy vs Cypher return paths use different/unstable column names | probe a candidate-key list in priority order (`["e1.action_type","key_0","a","action_a"]`); coerce `Float`→`i64` | inferred | not filed |
| `src/topics.rs:330` `upsert_topic` BELONGS_TO | RC14: no idempotent edge-MERGE; `create_edge` is a bare CREATE → duplicate edges on re-run | blindly `create_edge`, catch `UnikoError::Storage` whose message contains `"duplicate"`/`"already"`, treat as success (string-match on error text) | inferred | not filed |
| `src/procedures.rs:222`/`:447` `match_procedures` / `precondition_matches` | can't evaluate a per-row stored predicate string inside the query | Cypher filters only `status='active'`; precondition match done row-by-row in Rust against the in-memory `state` map | inferred | design (MVP evaluator) |
| `src/topics.rs:17`/`:235` `run_lpa` | `CALL uni.algo.louvain` requires materializing a graph projection | reimplement community detection as weighted Label Propagation in Rust over a co-occurrence adjacency list | comment | design (perf/scale choice; swap to Louvain past ~50K entities) |

### uniko-bench  (mostly perf-repro/diagnostic binaries + tooling)

| Site | uni-db issue | Workaround | Evidence | Status |
|------|--------------|-----------|----------|--------|
| `src/cypher_main.rs:212` | Cypher parser rejects a trailing `;` | strip trailing `;` from the REPL buffer | comment | not filed |
| `src/events.rs:10` + `src/ingest.rs:264` | `xervo.embed()` discards the provider `usage` field → no real embedding token counts | estimate embed tokens as `chars/4` | comment (cites `feedback_no_edit_uni`) | not filed (needs uni-db patch) |
| `src/ingest.rs:203` | RC13: `DataType::Bytes` ✅ fixed (`#50`) | caption/query text proxy; raw image bytes unimplemented (needs img_url fetch, not a uni-db limitation) | comment (de-referenced) | `#50` done |
| `src/insert_microbench_main.rs:215` | `SchemaBuilder`↔`LabelBuilder` chained-API type-juggling | apply schema one label at a time (separate `.apply()` per label) | comment | not filed (ergonomics) |
| `src/main.rs:524` `verify_label_visible` etc. (callers `:338`,`:343`,`:487`) | label-scan invisibility: vertices reachable via edges/`id()` but invisible to `MATCH (n:Label)` (conv-26 symptom) | defensive runtime re-query by label + `tracing::warn!` when label-scan returns 0 (detection, not a fix) | comment | repro (examples below) |
| `src/longmemeval_main.rs:34` | default allocator underperforms under concurrent writes | install `uni_db::MiMalloc` global allocator (~3× on uni-db `concurrent_mutations`, commit `65399a2b`) | comment | not filed (perf) |
| `src/longmemeval/query.rs:159` / `src/query.rs:314`,`:350` | no single batched multi-path coalesce over `UNWIND $ids` | per-node-id query in a Rust loop, post-process/dedup in Rust | inferred | not filed (or cost choice) |
| `src/query.rs:329`,`:362` | uni-db formats DateTime as full `YYYY-MM-DDThh:mm:ss±hhmm` | split on `'T'` in Rust to keep the date | comment | not filed (borderline) |

**Repro/diagnostic binaries (documentation only):** `update_microbench_main.rs`
(UNWIND-SET non-monotonic cost 1.9→12 ms, `.profile()` reports `time=0` for
`MutationSetExec`/`GraphScanExec`) · `mutation_set_microbench_main.rs` (~17 ms/row
`MutationSetExec`; Cypher DDL can't declare vector/Hash-scalar indexes) ·
`profile_writes_main.rs` (~600 ms/`batch_create_edges_fast_in_tx` at concurrency
24 regardless of batch size) · `examples/repro_uniko.rs`, `probe_*.rs`,
`tests/diagnostics/graph_debug.rs` (label-scan invisibility + empty-properties-on-reopen).

> **Deleted in the current uncommitted change:** the seven `unidb_*` repro tests
> (`#39`, `#40`, `#49`, plus unnumbered slow-WHERE / insert-latency / edge-latency
> regressions) — once that change is committed those bugs are no longer
> represented in-tree. They are filed upstream; this catalog preserves the record.

### uniko-api & uniko-pipes — **no workarounds**

`uniko-api` is a 31-line `pub use` facade with no uni-db dependency.
`uniko-pipes` is generic Layer-2 infra (Step/circuit-breaker/DLQ/health); its only
uni-db contact is `uni_db::Value` in `dead_letter.rs`, all through the
`uniko_store` abstraction. The `retry`/`fallback`/`CircuitBreaker` hits there wrap
**LLM provider** calls, not uni-db.

---

## 4. Notable items that are NOT uni-db workarounds (audited & discarded)

- LLM-provider retry/backoff, quote-stripping, GPT/o-series temperature handling (`memory/nl_to_cypher.rs`, `memory/llm_triples.rs`, `bench/eval.rs`).
- ONNX/ORT embedding-runtime limits: embed-batch chunking (`extract/embedding/mod.rs:97`, `memory/recall:556`), serialized BGE-small mutex — these are model-runtime, not graph-DB.
- `pdf/extractor.rs` `catch_unwind` — guards the third-party `pdf-extract` crate.
- tree-sitter "unsupported language" fallbacks (`extract/ingest/chunking/`).
- `observations/mod.rs:51` poisoned-`Mutex` recovery — std lib, not uni-db.
- `pipes/step.rs:69` context caching — ordinary app-level memoization with a clean `None` fallback.
- `store/storage/batch.rs` UNWIND batching — *leverages* the fixed HashJoin rewrite; an optimization, not a dodge.
