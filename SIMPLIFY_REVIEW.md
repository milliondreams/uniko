# Workspace simplification review

Generated 2026-05-28 by running `code-simplifier` against each crate in parallel. Findings are **proposals**, not applied changes. Read the per-crate sections and decide what to act on; some items conflict with intentional design choices and need your judgment.

Severity legend:
- **High** — clear win, likely worth doing
- **Medium** — judgment call, may have tradeoffs
- **Low** — nits / cosmetics

Status legend:
- ✅ **DONE** — addressed in commit `9bf3088` (2026-05-28)
- 🔁 **REFRAMED** — the proposal's premise was wrong; the underlying concern was resolved a different way (see note)

---

## uniko-api

### High
- 🔁 **REFRAMED** `Cargo.toml:10` — declares `uniko-memory` direct dep, but `tests/layering_test.rs:73-75` asserts uniko-api may only depend on `uniko-cortex`. Real layering violation the test will catch. **Fix:** drop `uniko-memory` and re-export through `uniko-cortex` (cortex already declares `uniko-memory` at `crates/uniko-cortex/Cargo.toml:19` but doesn't use it).
  - **Note:** The proposed fix is structurally impossible — `uniko-memory` depends on `uniko-cortex` (`memory/src/pipeline/consolidation_worker.rs:10` imports `uniko_cortex::procedures`), so cortex cannot depend on memory without a cargo cycle. The test in `tests/layering_test.rs` had bitrotted: it described a strict linear chain that the codebase had intentionally departed from, and referenced nonexistent crates (`uniko-fs`/`shell`/`mcp`). It also wasn't wired into any workspace package, so it never actually ran. Resolved by rewriting the test to reflect current architecture (cortex/extract siblings on store; memory consumes both; api consumes cortex+memory). All seven pins verified against current Cargo.toml.
- 🔁 **REFRAMED** `crates/uniko-cortex/Cargo.toml:19` — cortex declares `uniko-memory` but no source file uses it. Either remove, or add the `pub use uniko_memory::{...}` re-exports there so the api crate can drop its direct memory dep.
  - **Note:** The `uniko-memory` line at `cortex/Cargo.toml:19` was always under `[dev-dependencies]` (used by cortex tests), not production. No removal needed.

### Medium
- `src/lib.rs:8-13` vs `src/tools.rs:8-18` — `lib.rs` argues wildcard re-export is intentional; `tools.rs` enumerates specific symbols. Two policies in a 31-line crate. Pick one (wildcard is consistent with the stated rationale).
- `src/tools.rs:8-18` — enumerated list will rot silently when uniko-memory adds/removes/renames tool primitives. Consider per-submodule wildcards (`pub use uniko_memory::rules::*;`).

### Low
- `src/tools.rs:1-6` — module doc explains *what* tools are (obvious from re-exports) rather than *why* this facade module exists separately from `lib.rs` wildcard.

---

## uniko-bench

### Cross-binary duplication
- `update_microbench_main.rs:28-33` & `mutation_set_microbench_main.rs:150-155` — identical `UPDATE_CYPHER`. Move to `lib.rs` as `pub const ENTITY_UPDATE_CYPHER`.
- `update_microbench_main.rs:130-133` & `mutation_set_microbench_main.rs:195-198` — identical `med` closure. Extract `pub fn median(v: &mut [f64]) -> f64`.
- `cypher_main.rs:264-274, 305-315`, `profile_writes_main.rs:206-215`, `update_microbench_main.rs:171-177` — same per-operator runtime stats print loop. Extract `print_runtime_stats(prof)`.
- `cypher_main.rs:108-113`, `insert_microbench_main.rs:74-79`, `profile_writes_main.rs:40-45`, `nlp_compare_main.rs:109-117`, `main.rs:166-174`, `longmemeval_main.rs:87-95` — duplicated `tracing_subscriber::fmt()...` init. Add `pub fn init_tracing(default_filter: &str)`.
- `update_microbench_main.rs:37-40`, `insert_microbench_main.rs:203-207`, `mutation_set_microbench_main.rs:67-70` — repeated `tempdir + Uni::open + to_string_lossy`. Extract `open_tmp_uni()`.
- `main.rs:529-560` and `main.rs:564-594` — `verify_label_visible` / `verify_node_visible_by_label` differ only in WHERE clause. Fold into one helper.
- `main.rs:331-336` and `longmemeval_main.rs:271-276` — identical `BenchConfig → TripleSource` mapping. Add `BenchConfig::triple_source()`.
- `cypher_main.rs:286-325` and `:239-284` — `run_profile` and `run_profile_write` share ~80% of body. Collapse with `ProfileMode` enum.

### High
- `lib.rs:13-19` `now_value` — fallback comment is misleading (`timestamp_nanos_opt()` only fails post-2262).
- `insert_microbench_main.rs:380-389` `run_worker_edges` — `_label` and `_worker_id` unused params with `#[allow(clippy::too_many_arguments)]`. Drop both.
- `bench_config.rs:340 build_catalog_specs` + 3 callers — `build_catalog_specs` then immediate per-spec `tracing::info!` log loop duplicated in `main.rs:213-222` and `longmemeval_main.rs:164-173`. Move the log loop into `build_catalog_specs`.

### Medium
- `cypher_main.rs:117-128` — `UnikoConfig { .., ..Default::default() }` then mutated. Restructure as a single struct literal.
- `cypher_main.rs:340-344` — string-match dispatch on `&cli.format`. Promote to `clap::ValueEnum`.
- `cypher_main.rs:213-218` — `if !q.is_empty() && let Err(e) = run_one(...)` reads cleaner split into nested `if`.
- `insert_microbench_main.rs:93-98` — `--op` parsed by hand. Use `clap::ValueEnum` + `value_delimiter = ','`.
- `insert_microbench_main.rs:81-85` — `.expect("--sess parses as u32")` for a `usize` parse — wrong type in message; use clap value_parser.
- `nlp_compare_main.rs:208-211` — same default-then-mutate pattern; use struct-update syntax.
- `profile_writes_main.rs:77-135` — seven near-identical `if let (Some(a), Some(b)) = (...)` blocks. Drive from a table.

### Low
- `lib.rs:148-167 format_context` — unify on `writeln!`/`write!`.
- `lib.rs:263-295 string_or_number` — visitor's `visit_str` always returns `Some`; `visit_none/visit_unit` cover null.
- `cypher_main.rs:357-383 print_pretty` — comment restates next line; replace with threshold rationale.
- `update_microbench_main.rs:1-19` — top-of-file comment says "drop into `uni-db/crates/uni/examples/`" — stale.
- `insert_microbench_main.rs:434-439` `label_for`/`edge_for` — one-liners; inline at call sites.
- `longmemeval_main.rs:34-37` — `#[global_allocator] static GLOBAL: MiMalloc` only here; promote to `lib.rs` or document why others don't get it.
- `longmemeval_main.rs:228-236` — extract `EvidenceMap::reuse_only(item)`.
- `main.rs:89-140 RETIRED_FLAGS` — could become a typed `enum Migration { Replaced(&str), SetFalse(&str) }`.

---

## uniko-cortex

### High
- `procedures.rs:158-172` — `record_procedure_use` does two lookups (`get_node_by_ext_id` then `read_procedure`) for the same Procedure. Drop the first and reuse the snapshot's node id, or fetch `id(p)` inside `read_procedure`.
- ✅ **DONE** `procedures.rs:434-445` & `topics.rs:289-301` — identical SHA256→first-8-bytes-hex helper, also duplicated in `uniko-memory/src/action.rs:321-334`. Promote `uniko_store::ids::stable_hex64(parts: &[&[u8]])` and call from all three. Drops `sha2` dep from this crate.
  - **Note:** Landed as `uniko_store::id::stable_hex64(prefix, closure)`. The three sites use **different separators** (`\x00` vs `|`), so a `parts: &[&[u8]]` signature would have changed persisted IDs — closure-based design preserves byte-identical output. `sha2` dep dropped from cortex.
- `Cargo.toml:11,13` — `tokio` and `serde_json` are in `[dependencies]` but neither is referenced from `src/` (`serde_json` is only used in tests; tokio macros come transitively).

### Medium
- `topics.rs:334-349` — duplicate-edge handling string-matches `"duplicate"`/`"already"` in `UnikoError::Storage(msg)`. Replace with `kb.merge_edge(...)` or a typed `UnikoError::DuplicateEdge`.
- `procedures.rs:108-122` — ad-hoc `HashMap<String, Value>` for every Locy call; reuse the existing params builder.
- `procedures.rs:290-304 next_status` — nested `if/else if` chain over `(new, snapshot.status, support_count)`. A `match (new, snapshot.status.as_str())` plus threshold compare is clearer.
- `procedures.rs:408-426 pick_string`/`pick_i64` — exist only because rule and fallback emit different column names. Alias in the fallback Cypher and delete both helpers.
- `topics.rs:275-281 group_by_label` — clones every `EntityRow` (including embeddings) into groups. Pass indices into `entities` instead.

### Low
- `procedures.rs:107` — explicit `let detection_threshold: i64 = 1;` annotation is noise.
- `procedures.rs:298-304` — most doc-comments restate code (`pick_string`, `bucket_by_type`).
- `topics.rs:419-422` — single-use `GenerationOptions { ... }`; inline.
- `topics.rs:493-514 pool_embedding` — could be `fold`-based but the loop is arguably clearer; leave unless adding dim-mismatch warning.
- `topics.rs:516-647` test helpers `ent` and `ent_with_type` — collapse to one with `""` at call sites.
- `lib.rs:6-7` — module doc promises MCTS/rule induction/working memory/NL2Cypher but only `procedures` and `topics` ship.

---

## uniko-extract

### High
- `src/ingest/message.rs:131-145 ensure_session_and_sender` — duplicates participant-ensure/link/register across `Some(0)` and `None` arms. Collapse into a single create path.
- `src/ingest/message.rs:196-233` — standalone `create_chunks` is a `kb.tx + create_chunks_in_tx + commit` wrapper, only called from 4 sites while others manage their own tx. Inline or use a `kb.with_tx(...)` helper.
- `src/ner/dedup.rs:70-97 upsert_entities` — dead (no in-tree callers; the atomic path uses `prepare_entity_upsert + apply_entity_upsert`).

### Medium
- `src/ingest/atomic.rs:198-234` — giant `tracing::info!` mirrors every `AtomicTimings` field; use `?timings`. Follow-up `tx_perf` log duplicates totals already emitted.
- `src/ingest/atomic.rs:118-122, 283-287` — `cfg(feature = "onnx")` rebind of `nlp_ms` only because the var is reassigned. Restructure so it's assigned once inside the cfg block; drops the `cfg_attr` allow.
- `src/ingest/artifact.rs:42-62` — two near-identical idempotency lookups (by hash, by artifact_id). Extract `find_existing(kb, hash, artifact_id) -> Option<NodeId>`. (Bonus: comment numbering goes `1,2,3,4,5,6,5`.)
- `src/ingest/message.rs:255-281` — `props.insert(...) ×9` building chunk props. Extract `fn chunk_props(chunk, parent_ext_id) -> HashMap`.
- `src/ingest/message.rs:316-340 json_to_uni_value` — likely duplicates a helper in `uniko-store` / `uni-db`. Grep before redefining.
- `src/observations/mod.rs:319-330 combine_entity_refs` — called twice (one inside the rule fallback, again at 339). Compute once before the model branch.
- `src/observations/mod.rs:436-455` — `raw.subject.to_lowercase()` allocated per entity per observation. Lowercase `entity_refs` once outside the loop.
- `src/ingest/session.rs:38-47, 67-79, 96-112` — three `tx_perf` blocks reuse `start.elapsed()` for both `total_ms` and `commit_ms`. The `commit_ms` field lies — drop or compute separately.

### Low
- `src/ner/dedup.rs:208-213` — `new_conf = if ... { entity.confidence } else { old_conf }` is `entity.confidence.max(old_conf)`.
- `src/ner/onnx.rs:31-42` — turn the conversion into `impl NerEntityType { fn to_extract_type(&self) -> EntityType }`.
- `src/ner/code.rs:65-92` — consolidate universal node kinds into one arm above the per-language match.
- `src/ner/code.rs:107-120` — `extract_name` recurses into `function_definition`/`class_definition` children but the outer `walk_node` already does; may double-extract.
- `src/observations/filter.rs:147-152 is_informative_by_cls` — dead (replaced by `cls_gate_admits`).
- `src/observations/mod.rs:236-238` — bind `sender_name` inside the `cfg(onnx)` block instead of allowing `unused_variables`.
- `src/ingest/chunking/text.rs:247-272 apply_overlap` — mixes char/byte arithmetic; the `unwrap_or(0)` word-boundary fallback yields a leading-space overlap.
- `src/embedding/mod.rs:56-71/78-95` — `embed_raw`/`embed_batch` repeat the `xervo.is_available()` guard. Extract `require_xervo(kb)`.

---

## uniko-memory

### High
- ✅ **DONE** `recall/mod.rs` (10+ sites: `:563-567, 599-603, 627-631, 655-659, 674-678, 691-695, 1048-1052, 1384-1388, 1720-1724`, `working_memory.rs:165-169`) — score-desc `partial_cmp(...).unwrap_or(Equal)` sort repeated everywhere. Extract `sort_by_score_desc<T>(items, key)` or `cmp_score_desc` helper.
  - **Note:** Added `pub(crate) fn sort_by_score_desc` to `uniko_memory::lib`. Replaced 13 call sites (10 RecallItem + 2 RankedHit + 1 tuple in rrf_tests).
- `consolidation.rs:172-205` — `LLM` arm builds `inputs`, reads `inputs.len()` into a `(inputs, requested)` tuple, then `drop(inputs)` explicitly. Drop the tuple and explicit drop: `let requested = inputs.len();` then move `inputs` in.
- `recall/mod.rs:1525-1551 entity_type_match` — Cypher built via string interpolation of `node_id`/`target_type` with hand-rolled `'` escaping. Switch to `$nid`/`$t` params (matches the rest of the file).
- `recall/mod.rs:1571-1726 run_recall_for_variant` — legacy `HashMap<NodeId, (String, String, f64)>` accumulator + post-conversion (`:1710-1719`) to `RankedHit` exists because the migration is incomplete. Make `scored` a `HashMap<NodeId, RankedHit>` directly.

### Medium
- `recall/mod.rs:1770-1778` and `working_memory.rs:532-540` — two near-identical `empty_bundle()` helpers. Merge as `ContextBundle::empty()`.
- `recall/mod.rs:1735-1750 rrf_fuse` — still returns `HashMap<NodeId, (String, String, f64)>`. Return `HashMap<NodeId, RankedHit>` and remove the tuple destructuring in callers.
- `episode.rs:104-121` — `state_value`/`delta_value` cloned into props before passing on. Extract an `insert_some(&mut props, "state", state_value.as_ref())` shared with `action.rs`.
- `action.rs:172-226` — three near-identical optional-edge blocks (`TRIGGERED_BY`/`IN_SESSION`/`NEXT_ACTION`). Extract `wire_optional_edge(kb, edge_name, from, label, key, value, direction)`.
- `consolidation.rs:254-279, 295-302, 282-294 unique_objects` — collected via two near-identical loops; fold into one iterator chain.
- `recall/mod.rs:651-708` — `Phase1Strategy::Merge` duplicates the sort+truncate done at `:463-471`. Factor `merge_phase1_into(items, phase1_items, cap)`.
- `consolidation.rs:476-519` — `_stale_obj` destructured but unused. Change `prior_stale` to `Vec<(NodeId, Btic)>` and drop the now-dead `object` column.
- `recall/intent.rs:512-516` — `if keywords.is_empty() { String::new() } else { keywords.join(" ") }` — `join` already returns `""` for an empty slice.
- `value_convert.rs:49-62` vs `:77-96` — `extract_optional_dt` and `require_datetime` duplicate Temporal/String branches. Express `require_datetime` in terms of the optional helper.
- `pipeline/consolidation_worker.rs:131-143` — collecting `agents: Vec<String>` then reset+run; the existing comment flags the awkwardness. Either collect `(agent_id, &mut count)` or drain.
- `recall/mod.rs:758-812 phase1_compact` — keep-max-score `entry/and_modify/or_insert` pattern appears in five places. Extract `keep_max_score<K, V: ScoreCarrier>`.

### Low
- `recall/mod.rs:843-844, 866-867` — tuple-destructure of `Phase2Activation` right after construction. Use `let Phase2Activation { qvec, qtxt, has_vec, .. } = act;` or access via fields.
- `recall/mod.rs:1062` — `#[allow(clippy::too_many_arguments)]` on `run_phase2_source`. Bundle into a `Phase2Probe<'a>` struct.
- `nl_to_cypher.rs:100-122` — two identical `cache_mutex(...).lock().expect(...)` calls; store once at fn entry.
- `nl_to_cypher.rs:151-156 normalise` — `split_whitespace().collect::<Vec<_>>().join(" ")` — drop the intermediate `Vec`.
- `rules/lifecycle.rs:170-195` — bool-flag ladder; compute `new_status` as a single `match (prune, demote, repromote || promote_candidate)`.
- `consolidation.rs:584-589 min_or` — one-line wrapper used twice; inline.
- `llm_triples.rs:155-156` — `parts.first().filter(...)?.clone()` pattern; destructure with `let [subject, predicate, rest @ ..] = parts.as_slice()`.
- `recall/mod.rs:1294` — magic `15` and `2`; promote to a named const with rationale.
- `recall/intent.rs:88-99 intent_vec()/keywords()` — admitted legacy back-compat; mark `#[cfg(test)]` or drop.

---

## uniko-pipes

### High
- `circuit_breaker.rs:85-111 CircuitBreaker::call()` — dead (callers only use `state()`/`is_available()`/`record_*`). Remove or document why it's public.
- `cancel.rs:67-102 ShutdownCoordinator::shutdown()` / `is_cancelled()` — dead. The pipeline creates the coordinator and hands out child tokens but never calls graceful shutdown. Wire it up in `uniko-memory::pipeline` or delete. (Also: the 5s/10s sleep ladder unconditionally sleeps the full window.)
- `dead_letter.rs` — module doc already admits "Retry/list/clear surfaces have no production callers". 46-line `DeadLetterQueue` is a one-method `create_node` wrapper around `Arc<KnowledgeBase>`. Inline at the single ingest_worker call site.

### Medium
- `circuit_breaker.rs:114-137 record_success/record_failure` — `swap(STATE_CLOSED, …)` unconditionally on the hot success path. Use `load → compare_exchange` only on transition.
- `circuit_breaker.rs:147-152 now_ms()` — duplicates time-now logic; consider using `Instant::now()` + stored `Instant` instead of millis-since-epoch (avoids `SystemTime` skew).
- `types.rs:52-77 PdfInput`/`IngestPdf` — only matched in `ingest_worker` with `(String::new(), …)`; never produced. Dead arm unless planned.
- `types.rs:165-174 ItemResult::new` — trivial; `#[derive(Default)]` + struct-update syntax instead.
- `health.rs:73-81 HealthTracker::new()` — `#[allow(clippy::new_without_default)]` + derive `Default` (matches recent `Default for ShutdownCoordinator` style).
- `metrics.rs:77-130` — 9 emission helpers are one-liners around `counter!`/`histogram!` with single callers each. Keep only the ones that add a label; drop the no-label trivials.

### Low
- `step.rs:69-83` — `sender` field carries a 15-line comment duplicating `MessageIngestResult.sender` in `uniko-extract`. Reference the source of truth.
- `types.rs:1-4` — module doc says "carry no logic — only data" yet `ItemResult::new` lives here.
- `health.rs:120-146 classify` — `queue_capacity > 0` guard could be `queue_capacity.max(1)`.
- `circuit_breaker.rs:14-16` — `STATE_*` consts plus `CircuitState` enum duplicate the same three values; `#[repr(u8)]` on the enum eliminates them.
- `config.rs:9-44` — 12 flat fields. Grouping into `IngestConfig`/`ConsolidationConfig`/etc. if more land. No behavior change.

---

## uniko-store

### High
- ✅ **DONE** `storage/nodes.rs:56-107 create_node_in_tx` — 5 per-section `Instant::now()` timers (validate/build/bind/fetch/extract) + total on the per-node hot path; emits 10+ structured fields per CREATE. uni-db `QueryMetrics` already exposes parse/plan/exec. Keep only `t_total` + metrics.
  - **Note:** Confirmed zero `query_metrics` consumers in workspace before trimming. Now emits `site`, `label`, `total_us`, `cache_hit` (4 fields).
- ✅ **DONE** `storage/edges.rs:204-286 create_message_edges_in_tx` — four sub-bulk timers + a synthetic `query_metrics` event. Collapse to one timing + one log line.
  - **Note:** Also dropped the `msg_edges_breakdown` event. Single `query_metrics` emission with `site`, `total_us`, `recipients`, `has_prev`, `bulk`.
- ✅ **DONE** `storage/batch.rs:230-233` — `let _ = validate_property_name(...)` discards the error on the bulk fast path even though the comment 4 lines above says property names need validation. Real correctness gap — change to `?`.
  - **Note:** Hoisted validation out of the `.map(...)` closure so `?` can propagate. Regression test `test_batch_create_edges_fast_rejects_invalid_property_name` added.
- `operations/facts.rs:595-617 parse_btic_lo_millis`/`parse_granularity`/`parse_certainty` — exist because `find_stale_open_facts` returns `toString(f.valid_at)` + name strings instead of the raw BTIC. `batch_upsert_facts` at line 318 already reads `f.valid_at` directly with `extract_btic(...)`. Switch the stale-fact query to the same shape and delete all three parsers.
- ✅ **DONE** `storage/kb_stats.rs:160-213 bump_modality_presence` — reads modality via a fresh session inside an open tx (`self.read_modality_presence()` opens its own session), creating a read-then-write race outside the tx. Move the read inside the same tx.
  - **Note:** In-tx read alone was insufficient — uni-db's tx commit is last-writer-wins, so two concurrent txs both observe the same pre-image. Two-part fix: (a) added `read_modality_presence_in_tx`, (b) added `kb_stats_lock: Arc<tokio::sync::Mutex<()>>` on `KnowledgeBase` to serialize the RMW. Concurrency regression test `test_bump_modality_presence_concurrent_no_lost_update` added (fails without mutex, passes with).

### Medium
- `storage/edges.rs:144-166 create_edges_in_tx` — always returns `Ok(Vec::new())`; the `Result<Vec<EdgeId>>` signature lies. Return `()` or actually collect EIDs (same for the bulk arm at `:265`).
- `storage/edges.rs:71-73` and other `qb = qb.param(...)` loops in `nodes.rs`/`edges.rs` — a tiny `bind_params(qb, &params)` would replace 5+ copies.
- `storage/batch.rs:81-106`, `nodes.rs:88-107`, `edges.rs:262-286` — three near-identical bulk `tracing::info!(target: "query_metrics", ...)` blocks. Extract `emit_bulk_metrics(site, label_or_type, count, elapsed)`.
- `operations/facts.rs:227-456 batch_upsert_facts` — 230 lines, splits cleanly into `dedup_inputs`, `fetch_existing`, `partition_into_creates_and_updates`, `apply`.
- ✅ **DONE (partial)** `storage/kb_stats.rs:181-199` — builds a map of false flags then overwrites with current state. Just clone existing and insert one key. The 4-element `["image","audio","video","multimodal"]` is duplicated at `:140-143, 182, 250` — lift to a `MODALITIES` const.
  - **Note:** `MODALITIES` const added and consumed at the two array-literal sites (`bump_modality_presence` + `empty_modality_presence`). The 4-field `bool_flag(m, "image"|"audio"|...)` block was factored into `extract_modality_presence(Option<&Row>)`. The "clone existing and insert one key" micro-rewrite was not done — current shape is clearer and not on a hot path.
- `operations/facts.rs:910 count_recent_invalidations` — name says "recent" but implementation is cumulative; comment apologizes. Rename or implement the 30-day window.
- `search/hybrid.rs:47-56 TIER_WEIGHT_*` consts — duplicate `Tier::*.weight()`; only one is grepped anywhere. Delete.
- `storage/blob.rs:165-172` — `row.value(...)` pattern match is verbose; `row.get::<Option<Vec<u8>>>("bytes")?` is the idiomatic uni-db shape used elsewhere.
- `storage/mod.rs:25-48 apply_perf_knobs_from_env` — reads env at every KB open. Cache via `LazyLock<Option<UniConfig>>`.
- `search/fulltext.rs:50-88` — `multi_field_fulltext_search` dedup-by-max + sort+truncate; `multi_type_vector_search` only sort+truncate (no dedup). Undocumented asymmetry.

### Low
- `search/traversal.rs:71-73, 112` — depth hardcoded to 0; `depth` field in `TraversalResult` is dead.
- `storage/edges.rs:471-480 edge_type_label_hints` — 12-line what-comment for a 6-arm match.
- `error.rs:60-124 test_error_display` — 9 near-identical cases that test thiserror's own `Display`.
- `types.rs:19-34 datetime_value` doc — can shorten ~50% without losing the load-bearing warning.
- `storage/nodes.rs:140-173 get_node_by_ext_id` — could be `.map(...).transpose()` but explicit form is fine.
- `locy/rules.rs:50` — trailing "rows() returns Option<&Vec<Record>>" what-comment.
- `storage/migrations.rs:59-70` — three near-identical `match row.value(...) { Some(Value::String(s)) => ..., _ => ... }`. Extract `string_or<T>(row, field, default)`.
- `storage/edges.rs:147-149` — `type EdgeTriple = (...)` used exactly once; inline.

### uni-db usage flags
- `operations/facts.rs:649-655 find_stale_open_facts` — `RETURN toString(f.valid_at), btic_lo_granularity, btic_lo_certainty` then string-parses BTIC. `batch_upsert_facts` shows uni-db returns `Value::Temporal(Btic{...})` raw when the column is returned bare. Switch to `RETURN f.valid_at AS valid_at, f.object AS obj` and extract structurally. Removes the three parser helpers.
- `storage/edges.rs:362-368, 419-421 delete_edge`/`update_edge` — read `r._eid` directly because of a uni-db planner bug (`bugs/uni-db-edge-id-vid-planner.md`). Only one site references the bug doc. Re-check against current uni-db; swap back to `id(r) = $eid` if fixed.
- `storage/batch.rs:268-269` — comment notes uni-db `bulk_insert_edges` doesn't return EIDs at the public API. Worth a uni-db feature request rather than maintaining the Cypher-with-RETURN fallback.
- `storage/blob.rs:88-105` and `nodes.rs::merge_node:245-261` — both forced into two-step upserts by uni-db's MERGE evaluating NOT NULL before `ON CREATE SET`. If filed upstream and fixed, both collapse to a proper `MERGE ... ON CREATE SET ... ON MATCH SET ...`.

---

## Cross-cutting themes worth noting

1. ✅ **DONE** **Score-desc sort** is duplicated 10+ times in `uniko-memory` and likely in `uniko-store` ranking paths. A workspace-level `cmp_score_desc`/`sort_by_score_desc` helper would be the single biggest dedup win. — *Landed as `uniko_memory::sort_by_score_desc`; 13 sites collapsed. uniko-store ranking paths not yet audited.*
2. ✅ **DONE** **SHA256 → first 8 bytes → hex** stable-id helper is duplicated 3× (cortex procedures.rs, cortex topics.rs, memory action.rs). One `uniko_store::ids::stable_hex64` kills it. — *Landed as `uniko_store::id::stable_hex64`.*
3. ✅ **DONE (uniko-store half)** **Per-stage `Instant::now()` instrumentation** on hot paths in `uniko-store` (nodes/edges create) and `uniko-extract` ingest produces 10+ structured fields per write while uni-db already exposes parse/plan/exec via `QueryMetrics`. Trim aggressively. — *uniko-store nodes/edges trimmed (verified zero consumers). uniko-extract ingest still pending.*
4. **`tracing_subscriber::fmt + EnvFilter::try_from_default_env()` init** boilerplate appears in 6+ bench binaries. `init_tracing(default_filter)` once.
5. **Tuple-typed accumulators** (`HashMap<NodeId, (String, String, f64)>`) in `uniko-memory` recall paths exist mid-migration to typed structs like `RankedHit`. Finish the migration.
6. **Stringly-typed dispatch** (clap `--format` strings, `--op` parsing, error message substring matching for duplicates) shows up in bench + cortex. Several `ValueEnum` and typed-error wins are easy.
7. **`new()` constructors that should be `#[derive(Default)]`** appear in pipes (`HealthTracker`, `ItemResult`) — matches the recent `Default for ShutdownCoordinator` cleanup.

## Risks / things NOT to act on blindly

- ~~The `uniko-api` layering fix is the only finding that's likely a bug (test will fail). Everything else is a design call.~~ — *The test was dead (not wired into any cargo target) and described an aspirational architecture the code had departed from. Resolved by realigning the test with reality, not by changing dep graph.*
- Several "dead code" findings (`CircuitBreaker::call`, `ShutdownCoordinator::shutdown`, `DeadLetterQueue`) are infrastructure stubs that may be on roadmap. Confirm before deleting.
- ~~`uniko-store` hot-path instrumentation trims save Rust-side overhead but lose visibility — confirm with the `query_metrics` consumer (bench? grafana?) before pruning.~~ — *Audited: no consumers exist in workspace (no bench parser, no observability config, no grep matches outside emitter sites). Safe to trim; done for nodes/edges create paths.*
- uni-db workaround consolidations need to be re-verified against current uni-db (per the verify-memory-before-acting rule) before assuming the workarounds are still load-bearing.

### Surprise findings from the work itself

- The `bump_modality_presence` race was deeper than the review suggested. Moving the read inside the open tx (the review's fix) was insufficient because uni-db's tx commit is last-writer-wins, not serializable — two concurrent txs both observe the same pre-image. Required an in-process `tokio::sync::Mutex` on the KB. **Implication for elsewhere:** any other read-modify-write pattern against the same row needs the same treatment (or a server-side merge expression). Worth a workspace-wide audit.
- The three "identical" SHA-id helpers had **different separators** (`\x00` vs `|`). A `parts: &[&[u8]]` signature would have changed persisted IDs and broken stored graphs. Closure-based `stable_hex64(prefix, |h| h.update(...))` preserves byte-identical output. **Implication for elsewhere:** when consolidating "duplicated" hash helpers, diff the actual byte streams, not just the surrounding boilerplate.
