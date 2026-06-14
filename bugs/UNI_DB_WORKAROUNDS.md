# uni-db Workarounds & Limitations — Catalog

**Generated:** 2026-05-30 by a per-crate source audit (one agent per crate).
**Upstream-verified:** 2026-06-13 against uni HEAD `0a30594bb` (uni-db **2.1.0**); prior
passes 2026-06-12 (`3155a3710`, 2.0.7), 2026-06-03 (`3911cbc24`, 2.0.0) and 2026-05-31
(`3d5e849c0`, 1.3.0) — see §0 (newest first).
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

**2026-06-13 — re-verified against uni HEAD (`0a30594bb`, uni-db 2.1.0) + removed the two
now-dead workarounds in uniko.** uni advanced 2.0.7→2.1.0 (a "consolidation & hardening"
release; `cargo check -p uniko-store -p uniko-memory -p uniko-cortex` green against it).
This pass **acts on** the RC8 + RC1b fixes verified at 2.0.7 — both workarounds are now
deleted from uniko (see table). The rest of 2.1.0 is perf/correctness that does **not**
unlock further removals here:

| Action @ 2.1.0 | Site | What changed |
|----------------|------|--------------|
| **RC8 — REMOVED** | `store/storage/edges.rs` `delete_edge`/`update_edge` | `WHERE r._eid = $eid` → `WHERE id(r) = $eid`; workaround comments dropped. `_eid` now appears nowhere in-tree. Covered by `test_delete_edge`/`test_update_edge`. |
| **RC1b — REMOVED** | `memory/consolidation.rs` `try_parse_observation`; `bench/query.rs` `fetch_session_dates`/`fetch_temporal_anchors` | Manual `get::<String>` + `parse_from_rfc3339` / `split('T')` → read `Value::Temporal` via the existing `value_convert::extract_optional_dt` (memory) and a local `row_date` Temporal reader (bench). **Latent-correctness fix:** post-2.1.0 those columns return `Value::Temporal`, so the old `get::<String>` silently failed → `observed_at` fell back to `Utc::now()` / empty date maps. Both keep a String fallback for legacy rows. Also fixed a stale test assertion (`tests/lifecycle_e2e/consolidation_e2e.rs`) that expected a *stringified* BTIC `valid_at` from a returned Node — now matches `Value::Temporal(Btic{..})` and checks `hi < POS_INF` for closure. |
| **RC4 — comment freshened, KEEP** | `store/nodes.rs` `merge_node` | RC4 (NOT-NULL-before-`ON CREATE SET`) is fixed, but the get-then-create + `rmw_locks` split stays for **RC2** (insert-phantoms). Comment updated to name RC2 as the reason, not RC4. |
| **RC9 — comment freshened, KEEP** | `store/schema/chunks.rs` | Inference fallback fixed; the deferred positioning columns + `metadata` JSON bag are now framed as a deliberate flexible-schema choice, not a uni-db dodge. |

**RC2 still open and still gating.** uni 2.1.0 hardened MERGE matching (schemaless MERGE
matches flushed rows `d5b77ea71`; superseded-MVCC rows dropped `b4bda0793`) and added
retriable `ConstraintConflict` / `LockTimeout` exceptions — but it adds **no atomic CAS**
and still needs a `UNIQUE` constraint to stop concurrent first-upserts double-inserting
(the schema + migration cost uniko declined). So `store/locks.rs` StripedLocks, `merge_node`,
`operations/facts.rs` RMW, and `extract/ner/dedup.rs` 4-phase **all KEEP**. **RC3** got a
large MERGE perf pass (per-statement prefetch / one-scan-per-statement, ~63× on the
SET-residual MERGE bench) but it shares RC2's sites, so it unlocks no removal on its own.

> **2.1.0 behavioral breaks that touch uniko but are NOT workarounds** (build is green, so
> no compile break — flagged for behavioral follow-up only):
> - `db.rules()` mutators (`register`/`remove`/`clear`) are now **async** and **persist** to
>   `catalog/locy_rules.json`, recompiling on open (`2b81bee98`). **Adopted 2026-06-13:**
>   `store/locy/rules.rs` `create_rule`/`delete_rule` are now `async fn` (`.await` the
>   `register`/`remove` calls); `.await` threaded through callers in `memory/rules/stdlib.rs`,
>   `memory/rules/lifecycle.rs`, and `store` locy tests. (Required for uniko to compile
>   against a freshly-built 2.1.0 at all — the prior cached-artifact check masked it.) Rules
>   also survive restart now.
> - `tx.apply(derived)` is **fresh-by-default** — rejects stale derivations with
>   `StaleDerivedFacts` unless `.allow_stale()` / `.max_version_gap(n)` (`1ef384668`). Locy
>   reads now join the SSI read-set, so a `tx.locy()` RMW conflicts correctly.
> - `SerializationConflict`/`ConstraintConflict`/`LockTimeout` now map to distinct retriable
>   exception classes (`ef6195ded`) — switch any generic-`UniError` SSI-contention catch.

**2026-06-12 — re-verified against uni HEAD (`3155a3710`, uni-db 2.0.7) by source audit + running uni's own suite.**
uni advanced 2.0.0→2.0.7 (`3911cbc24`→`2b81bee98` + 2 local fix commits this pass).
**Six gaps closed since the 2026-06-03 pass** — three fixed in uni this session, three found already-fixed upstream:

| Issue | Verdict @ 2.0.7 | Fix / evidence | uniko impact |
|-------|-----------------|----------------|--------------|
| **RC4** MERGE checks NOT NULL before `ON CREATE SET` | ✅ **FIXED** (uni `3155a3710`) | `execute_create_pattern` now gap-fills `ON CREATE SET` props into the new node *before* constraint validation (`on_create_seed_props`; self-referential items skipped to avoid double-apply). Guard `bug_merge_on_create_not_null`; uni suite **1725** + TCK **3925×2** green. | **RC4 gate removed.** `store/nodes.rs merge_node` get-then-create split is now a removal candidate; the `extract/ner/dedup.rs` 4-phase MERGE-collapse is unblocked **on the NOT-NULL axis** — but still gated by **RC2** insert-phantoms (keep StripedLocks / 4-phase until RC2). |
| **RC8** `id(r)` → `_vid` in `WHERE` | ✅ **FIXED** (uni `f99c2dfd7`) | WHERE-clause `id()` rewrite is now edge-aware (`metadata_function_column`/`rewrite_id_to_vid` emit `_eid` for an edge binding via `vars_in_scope`); `RETURN id(r)` was already correct. Guard `bug_edge_id_in_where`. | `store/storage/edges.rs` `delete_edge`/`update_edge` `r._eid`-direct-query workaround is now **removable** (plain `WHERE id(r)=…` works). |
| **RC9** empty typed `List<T>` → `List<Utf8>` | ✅ **FIXED** (confirmed; guard `b25f7fc69`) | uni guard `bug_empty_typed_list_inference` proves an empty `List<Float32>` keeps its element type across flush+reopen; mutation path normalizes `List<T>`→`LargeBinary`. | `store/schema/chunks.rs` JSON-bag deferral no longer *forced* by inference (may stay as design). |
| **RC1b** Temporal returned as RFC-3339 string on read | ✅ **FIXED** (uni `4583ee870`) | property maps encode/decode in `Value` space (no serde_json bridge); `RETURN`/`properties()`/edge maps return `Value::Temporal`. | **Read-side reparse removable**: `consolidation.rs:805` `try_parse_observation`, `bench/query.rs:329/362` date-split. |
| **RC10** `String`→`DateTime` silently dropped at flush | ✅ **FIXED** (uni `68b46bc99`, #68) | write-time `coerce_and_validate_property_value` coerces `String`→Temporal / errors on mismatch (no silent null+drop). | `store/types.rs datetime_value()` is now ergonomics, not mandatory. |
| **RC11** index on `ext_id` breaks `flush()` | ✅ **FIXED** (uni `6c2bc0e3a`, #67) | reserved property names rejected at schema-apply with a clear error (no Lance duplicate-field at flush). | `<label>_id` naming convention now *enforced upstream* (hard error) — keep it, but the silent-loss footgun is gone. |

uni `main` is **unpushed** (local) and uniko2 pins `uni-db` by **path** (`../uni/crates/uni`), so all six fixes are **live for uniko's current builds**. ⚠️ If uniko is ever built against a published crates-io uni-db ≤2.0.0, RC4/RC8/RC1b/RC10/RC11 regress. **Still open:** **RC2** (insert-phantoms / no atomic CAS — deferred, keep locks), **RC13** `Int16`, **RC3** general-path MERGE per-row *planning* (entity-upsert shape already fast-pathed), RC6 const post-flush latency step.

**2026-06-03 — re-verified against uni HEAD (`3911cbc24`, uni-db 2.0.0).** uni jumped
1.3.0→2.0.0 ("upgrade all deps to latest": Lance 6→7, arrow 57→58, datafusion 52→53,
object_store 0.11→0.13). The #39 (`d11ac2cbc`) and #55 (`48f3a4ed3`+`0d2a2cebc`+`e711c87e0`)
fixes remain in history (all ancestors of HEAD). The material change is **#69**:

| Issue | Verdict @ 2.0.0 | Fix / evidence | uniko impact |
|-------|-----------------|----------------|--------------|
| **#69** single-node `MERGE`-in-`UNWIND` | 🟡 **substantially fixed** for the entity-upsert shape | uni `ab405408a` "broaden MERGE-in-UNWIND fast path to un-indexed keys". `execute_merge` (`write.rs:1842`) detects the single-node/single-label shape once, builds **one** batch-level L0 snapshot (`merge_l0_existing`, `:1855`), and resolves each row via `execute_merge_row_indexed` with an O(1) lookup / single filtered scan — **no per-row `LogicalPlan`**. Still a `for row in rows` loop (`:1859`), but the per-row *planning* that made batched MERGE no faster than a loop is gone. Multi-node / relationship / non-scalar-key MERGE still falls to the per-row planning path (`:1905`). | **Benched 2026-06-03 → KEEP the 4-phase.** A single `UNWIND…MERGE…ON CREATE SET` is **not viable** for entity upsert: the real Entity has NOT-NULL `name` (not in the merge key) and MERGE validates NOT NULL *before* `ON CREATE SET` (**RC4, still unfixed** — `entity_upsert_bench` panics "Property 'name' cannot be null"). With `name` forced nullable, MERGE is **3–8× faster** than the 4-phase on existing-heavy batches (Hash-index lookup vs the 4-phase's id()-based update scanning the in-tx L0 buffer) — so RC4 is also a **perf** blocker. #69 fast-path itself is fixed; RC4 is the gate. |

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
| RC1 | **`WHERE` on `DateTime` predicates** ✅ **FIXED in 2.0.0**; Temporal read-as-string ✅ **FIXED 2.0.7** (RC1b, uni `4583ee870`) | not filed | **both workarounds removed**: predicate migrated (`memory/episode.rs`); read-side reparse **removed 2026-06-13** — `consolidation.rs` `try_parse_observation` + `bench/query.rs` now read `Value::Temporal` via `extract_optional_dt`/`row_date`. |
| RC2 | **No server-side atomic `SET`/CAS; SSI does not prevent insert-phantoms** | [`rmw-primitives-wishlist.md`](uni-db-rmw-primitives-wishlist.md) | **KEEP `store/locks.rs` (StripedLocks) + RMW sites** (verdict 2026-06-03). uni 2.0 adds SSI (default-on) + `transact_with_retry`, but SSI tracks **item-level** read/write sets → a "match-else-create" whose match is empty registers no conflict, so concurrent first-upserts of the same key both insert. **Probed: 8 concurrent match-else-create → 8 duplicate nodes with no unique constraint; → 1 with a `UNIQUE` constraint.** Replacing the locks would require adding `UNIQUE` constraints on every merge key (`entity_id`/`fact_id`/…) — a schema + on-open-migration change uniko deliberately avoids. No atomic `SET`/CAS primitive exists. (`cortex/topics.rs` BELONGS_TO dup-swallow already removed via edge-MERGE, RC14.) |
| RC3 | **`MERGE` runs a per-row executor loop** (no bulk fast-path) | `#69` — 🟡 **fixed @ 2.0.0** (uni `ab405408a`) | `extract/ner/dedup.rs` 4-phase entity upsert — RC4 NOT-NULL gate now **FIXED 2.0.7**; MERGE-collapse unblocked on that axis but **still gated by RC2 phantoms → KEEP**, see §0 |
| RC4 | **`MERGE`'s internal CREATE checks NOT NULL before `ON CREATE SET`** ✅ **FIXED 2.0.7** (uni `3155a3710`; guard `bug_merge_on_create_not_null`) | — | `store/nodes.rs` `merge_node` get-then-create split now **removable** (unblocks RC3 MERGE-upsert on the NOT-NULL axis; RC2 still gates) |
| RC5 | **`similar_to()` returns `0.0` on a FullText-indexed field** (BM25 path works) | `#39` — ✅ **FIXED** (uni `d11ac2cbc`) | **none** — over-attributed; `hybrid.rs` uses `CALL uni.fts.query` (never broken), see §0 |
| RC6 | **`get_edges` latency scales with total graph size, not out-degree** (no CSR adjacency) | `#55` — ✅ **FIXED** scaling (uni `48f3a4ed3`); const post-flush step remains | `extract` denormalized ABOUT edges are recall-expressiveness (keep); `store/tests/perf/` repros **retired** (re-verified 2.0.0) |
| RC7 | **`UNWIND … MATCH WHERE id(n)=p` HashJoin rewrite** ✅ **FIXED in 2.0** (`#53`/`#54`: fires without a label hint) | `#53`, `#54` | `extract/ner/dedup.rs` Phase 3 `:Entity` hint **removed** (uniko-extract 222/222). `store/storage/batch.rs` optional caller-supplied label hints **kept** (harmless scan-narrowing, caller-opt-in; removal needs a perf check) |
| RC8 | **`id(r)` on a relationship lowers to `r._vid`** (planner bug) ✅ **FIXED 2.0.7** (uni `f99c2dfd7`; guard `bug_edge_id_in_where`) | [`edge-id-vid-planner.md`](uni-db-edge-id-vid-planner.md) | `store/storage/edges.rs` `delete_edge`/`update_edge` `r._eid`-direct-query **removed 2026-06-13** (now `WHERE id(r)=$eid`) |
| RC9 | **Typed `List<T>` column with no producer → Arrow inference coerces `List<Float32>`→`List<Utf8>`** ✅ **FIXED 2.0.7** (confirmed; uni guard `bug_empty_typed_list_inference`) | not filed | `store/schema/chunks.rs:30` JSON-bag deferral no longer forced by inference (may stay as design) |
| RC10 | **Writing `Value::String(rfc3339)` into a `DateTime` column is silently dropped at flush** ✅ **FIXED 2.0.7** (uni `68b46bc99`, #68 — write-time coerce/reject) | `#68` | `store/types.rs:19` `datetime_value()` now ergonomics, not mandatory |
| RC11 | **A scalar index on a property literally named `ext_id` makes `flush()` fail** ✅ **FIXED 2.0.7** (uni `6c2bc0e3a`, #67 — reserved names rejected at apply) | `#67` / [`unidb-persistence-loss/`](unidb-persistence-loss/) | naming convention now *enforced upstream* (hard error); keep `<label>_id` |
| RC12 | ✅ **RESOLVED uniko-side 2026-06-14** (not a uni-db bug). Two distinct defects: (1) *invocation* — `execute_rule(name)` passed a bare rule name to `session.locy_with`, but a registered rule is invoked via the goal query `QUERY <name> … RETURN …`; and (2) the **`sequence_detector` rule body was itself invalid Locy** and so never registered (which is why the Cypher fallback was always load-bearing): Locy is not Cypher — a second `MATCH` clause is a parse error (use one comma-joined `MATCH`), there is no `VALUE` keyword (aggregate columns are `expr AS name`), and a `$param` in a post-`FOLD` HAVING does not resolve (`Unresolved parameter`). | uniko bug (fixed in-repo; no upstream filing) | **DONE.** Added `KnowledgeBase::query_rule(name, return_cols, params)` (builds `QUERY <name> RETURN …`); fixed the rule (`procedures.rs` `SEQUENCE_DETECTOR_RULE` + `rules/stdlib.rs`); `promote_procedures_once` now registers (idempotent) + `query_rule`s; the Cypher `fallback_sequence_query` + `pick_*` shims are **removed**; gated by `procedures_e2e::query_rule_sequence_detector_returns_rows` + all 4 promote tests green. NB: the other 3 stdlib rules (`relevance_decay`, `episode_pattern_detector`, `contradiction_detector`) likely share the same Locy-validity issues but have no callers — not addressed here. |
| RC13 | **Missing scalar `DataType`s** — `Bytes` ✅ **FIXED** (`#50`, 2.0.0); `Int16` ❌ still missing | `Bytes` = `#50` (done) | `Bytes`: **adopted** — `store/schema/artifact_content.rs` uses `DataType::Bytes` (`bytes`, `audio_fingerprint`); `bench/ingest.rs` TODO de-referenced. `Int16`: `store/schema/artifacts.rs:32` (channels still `Int32`) — **keep** |
| RC14 | **`MERGE`-an-edge between two `id()`-addressed endpoints** ✅ **FIXED in 2.0** | not filed | **both removed 2026-06-03**: `store/migrations.rs` manual two-step → `MERGE (a)-[:HAS_CONTENT…]->(c)`; `cortex/topics.rs` `create_edge`+dup-swallow → `MERGE (m)-[:BELONGS_TO]->(t)`. Hot batched edge paths keep plain CREATE (no dup possible; edge-MERGE has per-row planning cost) |

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
| `src/types.rs:19` `datetime_value()` | RC10: String→DateTime silently dropped at flush | mandatory helper builds `Value::Temporal(DateTime)` for every DateTime write | comment | ✅ fixed 2.0.7 (#68) — helper now optional |
| `src/locks.rs` `StripedLocks` + sites in `storage/mod.rs:80‑287`, `storage/nodes.rs:223` (`merge_node`), `kb_stats.rs:179` (`bump_modality_presence`), `operations/facts.rs:121` (`upsert_fact_by_triple`), `:324` (`batch_upsert_facts`), `:844` (`record_entity_invalidation`) | RC2: last-writer-wins, no atomic SET/row-lock/CAS | in-process striped async mutex serializes every read-modify-write critical section by key | comment | filed ([wishlist](uni-db-rmw-primitives-wishlist.md)) |
| `src/storage/edges.rs` `delete_edge`, `update_edge` | RC8: `id(r)` lowers to `r._vid` | ~~query the internal `r._eid` column directly~~ → `WHERE id(r) = $eid` | comment | ✅ **removed 2026-06-13** (uni `f99c2dfd7`) |
| `src/storage/nodes.rs:239` `merge_node` | RC4: MERGE CREATE checks NOT NULL before `ON CREATE SET` | split into explicit get-then-(update\|create) | comment | ✅ fixed 2.0.7 (uni `3155a3710`) — removable (RC2 still gates upsert-collapse) |
| `src/storage/migrations.rs` | RC14: edge-MERGE ✅ fixed 2.0 | **migrated** to `MERGE (a)-[:HAS_CONTENT…]->(c)` (read kept only for the report counter) | — | resolved |
| `src/schema/chunks.rs:30` | RC9: `List<Float32>`→`List<Utf8>` inference fallback | defer typed positional columns; stuff modality scalars into one `CypherValue` JSON `metadata` bag | comment | ✅ fixed 2.0.7 — no longer forced by inference (may stay as design) |
| `src/schema/artifacts.rs:32` | RC13: no `Int16` | declare `channels` as `Int32` | comment | not filed |
| `src/search/hybrid.rs:79` + `fulltext.rs`/`vector.rs` | ~~RC5~~ **NOT a #39 workaround** (corrected 2026-05-31) | separate vector leg + `CALL uni.fts.query` leg fused via Rust RRF (`rrf_fuse`) with tier weights — a deliberate ranking design; the FTS leg never used the broken `similar_to`-on-FTS path | design | #39 ✅ fixed, but unrelated |
| `src/operations/facts.rs:282` (within-batch dedup), `:340` (Phase 1/2/3 split) | RC2: no unique constraint on `fact_id`, no atomic SET | dedup by `fact_id` in Rust (first-wins, sum counts); read-match → create-new → update-existing phases; count arithmetic + certainty upgrade computed Rust-side | comment | filed ([wishlist](uni-db-rmw-primitives-wishlist.md)) |
| `src/operations/facts.rs:634` `extract_btic`, `:662` `find_stale_open_facts` | lossy `toString`→RFC3339 roundtrip on BTIC `valid_at`; `Value` has no `FromValue` | bare property projection + structural extraction of `Value::Temporal(Btic)` in Rust | comment | not filed (borderline) |
| naming convention (repo-wide) | RC11: index on prop named `ext_id` breaks `flush()` | name all external ids `<label>_id` | comment | ✅ fixed 2.0.7 (#67) — now enforced upstream; convention kept |

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
| `src/ner/dedup.rs:38` `upsert_entities` / `prepare_entity_upsert` / `apply_entity_upsert` | RC3 (#69 MERGE loop) + RC7 (#53/#54 HashJoin shape) + multi-label scan ~18 ms/row + writer-lock hold time | **4-phase split:** (1) read-only `UNWIND…MATCH` for existing ids on a *fresh session before the tx*, (2) bulk `UNWIND…CREATE` for new, (3) `UNWIND…MATCH (n) WHERE id(n)=u.nid SET` — `id()`-equality (HashJoin now fires without a label; `:Entity` hint **removed 2026-06-03**, #53/#54 fixed), (4) `batch_create_edges_fast` for MENTIONS (plain CREATE) | comment (extensive) | **#69 ❌ still open** (verified 2026-05-31 — keep); `#53`/`#54` HashJoin improved; label-hint cost investigated 2026-05-20 |
| `src/ingest/atomic.rs` + `ingest/message.rs:56` | legacy 3-commits-per-message had no atomicity → "Message persisted with no entities" half-states; auto-embed serializes on one ORT mutex | prep-then-single-tx: all CPU/read work before opening one tx, fold Message+edges+chunks+entities+observations into **one commit** | comment + inferred | not filed |
| `src/ingest/session_chunk.rs:280` ABOUT/Participant edge propagation | RC6-adjacent: multi-hop "obs-chunk reachable via its observations' entities" not expressible at query time | post-hoc Rust loop re-queries `Observation-[:ABOUT]->Entity/Participant`, dedups in a `HashSet`, materializes the full Chunk→Entity / Chunk→Participant edge product | inferred | not filed (denormalization-for-recall; attribution uncertain) |
| `src/lib.rs:1` | uni-db `Session`/`Transaction` transitively carry datafusion/sqlparser AST → trait-solver recursion overflow | `#![recursion_limit = "256"]` | comment | not filed (build-time, OOS) |

### uniko-memory

| Site | uni-db issue | Workaround | Evidence | Status |
|------|--------------|-----------|----------|--------|
| `src/episode.rs` `find_previous_episode` | RC1: `WHERE` on DateTime ✅ **fixed 2.0** | **migrated** — `WHERE e.timestamp >= $earliest AND <= $now` + `ORDER BY … LIMIT 1`; Rust window filter removed | — | resolved |
| `src/recall/mod.rs:1112` `phase2_temporal` | RC1: no DateTime predicate; can't express proximity-in-window | 3-arm `UNION ALL` with per-arm `WITH…LIMIT` (a single trailing LIMIT would starve an arm); Fact arm via custom `btic_overlaps()` UDF; flat score 1.0 | inferred | not filed |
| `src/recall/mod.rs:1247` `phase2_graph_activation` | spreading activation / weighted PPR not expressible in Cypher; `UNWIND/MATCH` hydration loses order | host-side `kb.personalized_pagerank_weighted(...)`, second `UNWIND $nids` round-trip to hydrate, **re-sort in Rust** | comment + inferred | not filed |
| `src/recall/intent.rs:480` + `recall/mod.rs:1485` `entity_type_match` | exact string equality on `name`; no possessive/punctuation normalization → `{name:"Caroline's"}` silently returns nothing | `normalize_entity_text` strips trailing punct + `'s` in Rust before binding; `entity_type_match` `format!`-inlines `target_type` with manual `'` escaping | comment | not filed |
| `src/consolidation.rs` `try_parse_observation` | RC1b: Temporal props came back as RFC-3339 strings | ~~read as `String`, re-parse with `DateTime::parse_from_rfc3339`~~ → `value_convert::extract_optional_dt` (reads `Value::Temporal`, String fallback retained) | inferred | ✅ **removed 2026-06-13** (uni `4583ee870`) |
| `src/rules/stdlib.rs:117` `register_stdlib_rules` | RC12: Locy runtime may not support the rule syntax | best-effort `kb.create_rule` — log at debug on error and continue (Rule node still merged). `sequence_detector` source corrected 2026-06-14 so it now registers; the other 3 stdlib rules may still fail to compile (no callers) | comment | ✅ RC12 invocation resolved; remaining-3 rule validity OOS |
| `src/recall/mod.rs:1405` `session_boost_signals` | — | one round-trip per Fact (`id(f)={nid}` inlined); self-noted "could be batched" | comment | design (likely not a uni-db workaround) |
| `src/nl_to_cypher.rs:158`/`:223` | avoid serializing full uni-db schema (vector-index params bloat the LLM prompt); no exposed read-only AST mode | hand-authored compact NODE/EDGE summary tables; regex read-only guard | comment | design (prompt-cost choice) |
| `src/lib.rs:8` | same recursion overflow as extract | `#![recursion_limit = "256"]` | comment | not filed (build-time, OOS) |

> `value_convert.rs` (Temporal↔RFC3339 string handling) is the shared substrate
> these DateTime workarounds lean on, not an independent site.

### uniko-cortex

| Site | uni-db issue | Workaround | Evidence | Status |
|------|--------------|-----------|----------|--------|
| `src/procedures.rs` `SEQUENCE_DETECTOR_RULE` | RC12 (resolved): rule invoked by name via `kb.query_rule` (QUERY goal-query); rule body fixed to valid Locy | register (idempotent) + `query_rule` in `promote_procedures_once` | comment | ✅ **resolved 2026-06-14** — `fallback_sequence_query` + `pick_*` removed |
| `src/topics.rs` `upsert_topic` BELONGS_TO | RC14: idempotent edge-MERGE ✅ fixed 2.0 | **migrated** to `MERGE (m)-[:BELONGS_TO]->(t)`; the string-match-on-error dup-swallow is gone | — | resolved |
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
| `src/query.rs` `fetch_session_dates`/`fetch_temporal_anchors` | RC1b: Temporal read shape | ~~`get::<String>` + split on `'T'`~~ → local `row_date` reads `Value::Temporal` (formats `%Y-%m-%d`), String-split fallback | comment | ✅ **removed 2026-06-13** (uni `4583ee870`) |

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
