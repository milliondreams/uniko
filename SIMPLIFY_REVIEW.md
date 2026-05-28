# Code-Simplifier Review — uniko2 workspace

Read-only review produced by 10 parallel `code-simplifier` subagents, one per
crate. Each agent surveyed its full crate (src/ + tests/ + examples/) and
returned concrete simplification opportunities with `file:line` citations.

**Date:** 2026-05-27
**Mode:** report-only (no edits)
**Scope per agent:** entire crate

---

## Table of contents

- [Cross-crate roll-up](#cross-crate-roll-up)
- [uniko-api](#uniko-api)
- [uniko-bench](#uniko-bench)
- [uniko-cortex](#uniko-cortex)
- [uniko-extract](#uniko-extract)
- [uniko-fs](#uniko-fs)
- [uniko-mcp](#uniko-mcp)
- [uniko-memory](#uniko-memory)
- [uniko-pipes](#uniko-pipes)
- [uniko-shell](#uniko-shell)
- [uniko-store](#uniko-store)

---

## Cross-crate roll-up

**Stub crates (recommend deletion or workspace removal until their phase lands):**
- `uniko-fs` — 3 TODO modules + 3 unused deps; zero code
- `uniko-mcp` — 2 TODO modules + 4 unused deps; zero code
- `uniko-shell` — `todo!()` main + 2 unused deps
- `uniko-api` — thin facade, only nit is the `pub use uniko_cortex::*;` wildcard
- `uniko-cortex/src/reasoning.rs` — single TODO line

**Dormant scaffolding to delete (real crates):**
- `uniko-memory`: `IngestWorker.consolidation_tx`, `ConsolidationWorker.semaphore`, video-channel methods on `RecallCounters`
- `uniko-extract`: `observations/llm.rs`, `observations/contradiction.rs`, `ner/llm.rs`, `ingest/overflow.rs`, embedding-dedup block in `ner/dedup.rs`
- `uniko-pipes`: entire `retry.rs` module + `rand` dep, most of `dead_letter.rs` (list/retry/clear), `MetricsSnapshot`, several `PipelineConfig` fields
- `uniko-store`: `vid_to_node_id`/`node_id_to_vid`/`eid_to_edge_id`, `_ensure_default_used_via_kbconfig`, dead `earliest_ms` in `facts.rs`

**Highest-leverage refactors:**
- `uniko-store`: ~100+ sites of `.map_err(|e| UnikoError::Storage(e.to_string()))` collapse to `?` via the existing `From` impl
- `uniko-extract/nlp/mod.rs`: 6 near-identical `extract_f32_*` tensor extractors → 1 generic
- `uniko-memory/recall/mod.rs`: 4 copies of sort-desc/truncate/token-budget → one `finalize_bundle()` helper (~80 LOC)
- `uniko-bench`: port LongMemEval to `bench_config.rs` (kills `build_llm_catalog` + ~120 lines of duplicated CLI flags); consolidate the 5 ad-hoc `Arc::try_unwrap(kb).shutdown()` copies and the 5 `load_kb()` test helpers
- `uniko-extract`: remove the legacy `ingest_message` / `EntityExtractionStep` / `ObservationExtractionStep` paths now that atomic ingest is the supported one
- `uniko-cortex`: replace `DefaultHasher` in `stable_procedure_id`/`stable_topic_id` (correctness — not stable across rustc versions)

**Note:** The cross-cutting "`// Rust guideline compliant` marker comment" item from the original roll-up has already been actioned (commit `823311a`).

---

## uniko-api

**Crate summary.** The crate is a 26-line pure facade with no `tests/` directory and no logic, only `pub use` re-exports from `uniko-cortex` and `uniko-memory`. There is essentially nothing to simplify — the crate is already at minimum viable size. The only opportunities are minor: a redundant doc comment, a stale "Rust guideline compliant" marker, and a partial overlap between the wildcard re-export in `lib.rs` and the explicit `uniko-memory` re-exports in `tools.rs` worth verifying.

### `crates/uniko-api/src/lib.rs`
- **L8** — `pub use uniko_cortex::*;` wildcard re-export hides the public surface and can silently leak/change with cortex edits; consider naming the items actually consumed downstream, or at least document why a wildcard is intentional.
- **L2-4** — Module doc says "Contains no logic — just builders and re-exports," but there are no builders; trim to "Re-exports from the cognitive stack."

### `crates/uniko-api/src/tools.rs`
- **L8** — `// Rust guideline compliant` is scaffolding noise that restates a policy, not code intent; delete. *(Actioned in commit `823311a`.)*
- **L10-18** — Three separate `pub use uniko_memory::...` blocks could collapse into one grouped `pub use uniko_memory::{nl_to_cypher::{...}, policy::{...}, rules::{...}, ...};` for a single source of the memory surface.
- **General** — Worth confirming none of these `uniko_memory` items are also re-exported transitively via `uniko_cortex::*` in `lib.rs`. If `uniko-cortex` re-exports `uniko-memory`, this whole module may be redundant.

### Cross-cutting
- No `tests/` directory exists.
- No dead code, no traits/generics, no error handling, no control flow to simplify; the crate is genuinely a thin facade.
- Cargo.toml is minimal and correct.

---

## uniko-bench

**Crate summary.** `uniko-bench` (~10.2 kLOC across 39 .rs files) hosts the LoCoMo + LongMemEval harnesses plus microbenches, examples, and standalone uni-db bug repros. The biggest wins are (a) collapsing the legacy `build_llm_catalog` flag surface in `lib.rs` that has been superseded by `bench_config.rs`, (b) deduping the four KB-counts probe binaries into one parameterised tool, (c) shedding the no-judge-only `llm_judge` wrapper plus `aggregate` / `write_json` zero-pricing trampolines kept "for compat" with no real callers, and (d) dropping the unused `_breaker` / `_cancel` and never-populated atomic counters in `longmemeval/ingest.rs`. The harness binaries (`main.rs`, `longmemeval_main.rs`) share substantial post-ingest-sweep + Participant-creation + checkpoint-write boilerplate that `lib.rs::run_post_ingest_sweep` already factors out for one but not the other.

### `src/lib.rs`
- L46–177: `build_llm_catalog` + `provider_options` — **dead-ish.** Since `bench_config.rs` ships its own `llm_alias_to_spec` + identical `provider_options`, only `longmemeval_main.rs` still uses `build_llm_catalog`; migrate LME to bench-config and delete this whole block.
- L88–145: 8-arg `build_llm_catalog` — over-engineered; if kept, take a small struct instead.
- L156–177: `provider_options` is verbatim-duplicated at `bench_config.rs:370–391` — extract to one `pub(crate) fn`.
- L213–221: `retrieval_answer` builds a `Vec` then `join`s — can drop the intermediate `Vec`.
- L226–297: `run_post_ingest_sweep` doc references stale line numbers; replace with a symbol reference.
- L306–337: `string_or_number` custom Visitor — `serde_with::PickFirst<(_, DisplayFromStr)>` would be ~10 lines.

### `src/main.rs`
- L91–142: `RETIRED_FLAGS` migration table — consider time-boxing; decide a removal date.
- L226–238: `llm_alias` / `judge_alias` derivation — 12 lines that could be `bench_cfg.effective_gen_alias()` methods.
- L344–361: ad-hoc bench-agent Participant creation duplicated nearly verbatim in `longmemeval_main.rs:531–546`; lift into `lib.rs`.
- L432–475: events-cost calculation duplicates `report::answer_cost`/`judge_cost` logic.
- L513–525: `Arc::try_unwrap` + shutdown pattern repeated in every example; extract a `shutdown_kb(Arc<KB>)` helper.
- L559–624: `verify_label_visible` + `verify_node_visible_by_label` differ only by the WHERE clause; merge.

### `src/bench_config.rs`
- L370–391: `provider_options` — duplicate of `lib.rs::provider_options`.
- L182–195: `RecallSettings::Default` redefines what the per-field `serde(default)` already provides.
- L213–221: same pattern for `CostSettings::Default`.

### `src/data.rs`
- L191–211: speaker-prefix-text accumulation mirrors `ingest.rs:171–187`; extract `fn enriched_text`.
- L97: `_ => Self::SingleHop` silently maps unknown categories — log a warning or return `Option<Self>`.

### `src/eval.rs`
- L84–115: 27-entry `negation_phrases` overlaps semantically with `longmemeval/eval.rs:92–107` abstention; unify.
- L296–306: `llm_judge` is a 4-line wrapper around `llm_judge_with_usage` discarding usage — delete if unused.
- L375–384: usage destructure appears at `query.rs:287–296` and `eval.rs:375–384`; extract helper.

### `src/events.rs`
- L31–94: `BenchEvent::Query` has 16 fields; `IngestTurn` has 14. Group token-related fields under a `TokenUsage` substruct.
- L156–164: `now_unix_ms` candidate for a shared `lib.rs` helper.

### `src/ingest.rs`
- L31–46: `IngestObserver` plus the `*_with_observer` variants — over-engineered for two call sites. Collapse to two functions taking `Option<&IngestObserver>`.
- L83, L122: doc-comments re-state the obvious "Like X but with observer".
- L356–395: `parse_session_datetime` tries 8 formats; some fallbacks may be dead — measure and drop.

### `src/query.rs`
- L196–206: `generate_answer` is a 4-line wrapper around `generate_answer_with_usage` discarding usage.
- L240–242, L304–339 vs L341–371: per-item Cypher `fetch_session_dates` and `fetch_temporal_anchors` are nearly identical; collapse.

### `src/report.rs`
- L148–170: `Accum::new()` initialises 18 fields to zero — replace with `#[derive(Default)]`.
- L249–289: nine near-identical `avg_*` helpers — define one `fn mean(sum: f64, count: usize) -> f64`.
- L356–361 / L573–579: `aggregate` / `write_json` one-line trampolines supplying empty pricing.
- L406, L425: `total_query_cost_usd` documented "Deprecated" — add a removal milestone.

### `src/longmemeval_main.rs`
- L156–242: ~80 lines of CLI flags exactly equivalent to the LoCoMo retired-flag set — port LME to `bench_config.rs`.
- L685–724: `run_cortex_sweep` is a strict subset of `lib.rs::run_post_ingest_sweep`; merge.
- L497–525: consolidation step duplicated verbatim.
- L444–494: KB open/reuse branch structurally identical to `main.rs:306–330`; extract `open_or_ingest_kb`.

### `src/longmemeval/ingest.rs`
- L72–73: `_breaker`, `_cancel` — explicitly unused with "kept for when we plumb them through" comment; delete.
- L92–112: 14 `Arc<AtomicU64>` counters of which 4 documented to "stay at zero"; drop.
- L222: `t_ctx_setup_ms.fetch_add(0, Ordering::Relaxed)` — dead.
- L287–288: 14-counter splatter could be one `IngestTimings` struct.

### `src/longmemeval/query.rs`
- L159–189: `extract_session_ids` has same N+1 pattern + Cypher shape as `query.rs::fetch_session_dates`; unify.

### `src/longmemeval/report.rs`
- L140–185: five separate `if total > 0 { sum / total } else { 0.0 }` blocks — extract `fn mean_or_zero`.
- L70–104: `or_insert(...)` → `or_default()` after `#[derive(Default)]`.

### `src/longmemeval/eval.rs`
- L92–107: abstention list overlaps with LoCoMo's `negation_phrases`; consolidate.
- L60–85: `ndcg_at_k` builds an inline `HashSet`; `recall_at_k` (L35–53) builds the same; extract.

### `src/longmemeval/data.rs`
- L112–123 + L125–135: `parse` and `from_shorthand` are sibling string-table matches; collapse with `phf` or single function.

### `src/cypher_main.rs`
- L241–286 vs L288–327: `run_profile_write` and `run_profile` differ only in tx vs session + one println; merge with `RunMode` enum.
- L329–425: three print functions duplicate cell-extraction loop; extract `cell_to_string`.

### `src/insert_microbench_main.rs`
- L62–73, L169–173, L176–193: three small enums + `CypherMetrics` could colocate.
- L168 declares `OpKind` after L95 already uses it; clap reads as `String` instead of `#[arg(value_enum)]`.
- L381–391: `#[allow(clippy::too_many_arguments)]` + two `_`-prefixed unused params; drop both.

### `src/mutation_set_microbench_main.rs` + `src/update_microbench_main.rs`
- `now_value` (L22–28 / L27–33) and median closure (L202–205 / L137–140) duplicated verbatim; extract to shared module.

### `src/profile_writes_main.rs`
- L173–185: `profile_edge` could accept the assembled Cypher rather than re-building from `prop_pairs`.

### `src/nlp_compare_main.rs`
- L181–204: catalog JSON literal could live in a fixture file.

### Examples — large duplication
- `examples/probe_full.rs`, `probe_kb.rs`, `probe_persist.rs`, `probe_unconstrained.rs`, `probe_props.rs`, `compare_facts.rs` all share open_kb → session → loop-and-print-count pattern, `Arc::try_unwrap` + shutdown epilogue (5 copies), `row.get::<i64>("n").ok().unwrap_or(-1)` extraction. **Suggested:** add `lib.rs::scalar_count(...)` and `shutdown_kb(...)`; merge probes into one binary.
- `examples/compare_facts.rs:119`: `let _ = HashMap::<(), ()>::new();` with comment about silencing unused-import — `HashMap` is no longer imported; remove.

### Tests — shared scaffolding
- 5 copies of `load_kb()` helper across test files; move to `tests/common/mod.rs`.
- `tests/graph_debug.rs:34–55`: `load_kb_no_schema_json` is one-flag variant; add a `WithSchemaJson(bool)` arg.
- `tests/unidb_flush_bug.rs:33`: `panic!` after `eprintln!` of the same message — drop the eprintln.

### Cross-cutting
1. **Catalog/config drift:** LoCoMo uses `bench_config.rs`; LongMemEval still uses 25+ CLI flags. Porting removes ~120 lines + one code path.
2. **Per-item Cypher loops:** 3 helpers do N×roundtrips; one batched `UNWIND` query replaces all.
3. **Two judge prompts, two abstention lists:** consolidate.
4. **Shutdown ritual:** one helper, used everywhere.
5. **`#[derive(Default)]` opportunities** for `Accum`, `CatAccum`, `CypherMetrics`.

---

## uniko-cortex

**Crate summary.** Small crate (~1.5K LoC) with two real modules (`procedures`, `topics`) and one stub. Both are reasonably tight but carry some scaffolding-for-the-future (multi-key fallback decoders, a Locy→Cypher fallback, a thin `UpsertOutcome` wrapper struct), a few comments that restate code, and one piece of dead test scaffolding. `reasoning.rs` and the `ProcedureNodeId` alias are dead. The `llm` feature path in `topics.rs` is the densest area: name-resolution flow has cfg-gate gymnastics that can collapse.

### `src/lib.rs`
- **L10, L13–14, L18–19**: `pub mod reasoning` plus inline re-exports are aspirational. `reasoning` is a TODO stub — drop until real content arrives.

### `src/reasoning.rs`
- **L1 (entire file)**: Single TODO comment. Delete the file and the `pub mod reasoning;`.

### `src/procedures.rs`
- **L104–108**: detection_threshold=1 comment is mostly self-evident; trim.
- **L128–130**: `pick_string`/`pick_i64` walk a 3–4 candidate-key list. Scaffolding for hypothetical rule output shapes; the only producers emit `"a"/"b"/"n"`. Either confirm and reduce or document which producers each alias is for.
- **L117–124, L385–417**: Locy→Cypher fallback (`fallback_sequence_query`) duplicates rule semantics. If the rule is stdlib and always loadable, this is dead defensive code.
- **L262–267, L347–353**: `enum UpsertOutcome { Created, Reinforced, Promoted }` used in exactly one place to bump report counters; returning `(bool, bool)` would remove the enum + match.
- **L292–306**: `if new { } else if … is_empty() { }` has an unreachable arm given `existing.is_none()` branch above.
- **L289–290**: `let existing = read_procedure(...).await.ok();` silently swallows storage errors as "missing." Distinguish.
- **L361–382**: `read_procedure` issues raw Cypher to read four scalars after `record_procedure_use` already fetched the node.
- **L440–447**: `stable_procedure_id` uses `DefaultHasher` — not stable across Rust releases despite "deterministic" doc.
- **L475**: `pub type ProcedureNodeId = NodeId;` has no users. Drop.

### `src/topics.rs`
- **L82–87**: `detect_topics_once` is a one-liner forwarder to `detect_topics_once_with_llm(.., None)`.
- **L188–194**: missing-id rows silently dropped while missing-name rows kept; inconsistent.
- **L264–268**: LPA tiebreaker correct but dense; needs short comment.
- **L305–307, L356–357**: `struct UpsertOutcome { created: bool }` is a single-field wrapper. Return `bool`.
- **L341–355**: "swallow duplicate edge errors by string-matching the message" — fragile; belongs in `kb.merge_edge`.
- **L366–383**: `resolve_topic_name` has cfg gymnastics (`let _ = kb; let _ = llm_alias;` in `not(feature = "llm")` branch). Split into two cfg-conditional definitions.
- **L395–409 vs L467–472**: type-bucket aggregation duplicated; extract `bucket_by_type` helper.
- **L450–460**: `if top.is_empty() { "topic" }` arm unreachable given upstream `min_community_size` filter.

### `tests/procedures_e2e.rs`
- **L33–37**: `embedding_must_work` is `#[allow(dead_code)] fn` with empty body — delete.
- **L73–132**: two tests share `seed_participant + record_seq + promote + assert status` shape; helper `assert_status(&kb, name, expected)`.
- **L200–209**: `read_status` duplicates inline status-query.

### `tests/topics_e2e.rs`
- **L86–87**: assert on exact entity-count + created delta; current asserts loose.

### Highest-leverage cleanups
1. Delete `src/reasoning.rs` + its mod line.
2. Delete `ProcedureNodeId` alias and `embedding_must_work` test stub.
3. Collapse `UpsertOutcome` wrapper in `topics.rs` to a plain `bool`.
4. Factor duplicate "bucket by entity_type" logic.
5. Decide fate of `pick_string`/`pick_i64` scaffolding and the Locy→Cypher fallback.
6. Replace `DefaultHasher` in `stable_*_id` with a cross-version-stable hash.

---

## uniko-extract

**Crate summary.** The crate is well-organised but carries substantial scaffolding for unbuilt futures (LLM stubs, contradiction detection, action-output overflow), legacy code paths that duplicate the atomic ingest flow, and a fan-out of near-identical tensor-shape extractors in `nlp/mod.rs`. The biggest concentrated wins are in `nlp/mod.rs` (six near-duplicate `extract_f32_*` helpers), the parallel legacy/atomic ingest paths (`message.rs::ingest_message` vs `atomic.rs::ingest_message_atomic`), and the legacy `ObservationExtractionStep`/`EntityExtractionStep` that exist only because the legacy `IngestStep` is still wired. Many module-level docstrings restate "what the code does" rather than "why."

### `src/observations/llm.rs` (entire file, 20 lines)
- Pure stub returning `Vec::new()`; only consumer is dead `EntityExtractionStep`/orchestrator helper. **Delete.**

### `src/ner/llm.rs` (entire file, 16 lines)
- Same pattern — dead stub returning empty vec. **Delete.**

### `src/observations/contradiction.rs` (74 lines)
- 50-line essay docstring followed by a 6-line stub never called anywhere. **Delete.**

### `src/ingest/overflow.rs` (entire file, 52 lines)
- L24: `#[expect(dead_code, reason = "used when record_action tool ships in Phase 2")]`. **Delete.**

### `src/ner/dedup.rs`
- L19–23, L357–452: `find_similar_entity` + similarity constants + `types_compatible` all dead (`#[expect(dead_code)]`). Drop the whole embedding-dedup block.
- L351–352: `debug_assert_eq!(labels::ENTITY, "Entity")` — trivial tautology.
- L218–272: new-vs-update partition could be `partition_map` over `Either`.
- L229–235: `let new_conf = if x > old { x } else { old };` → `.max(old_conf)`.

### `src/ner/onnx.rs`
- L52–64: manual title-casing — extract or use `heck`.
- L9–15: three `#[cfg]` toggles for the same `RawEntity` import — collapse.

### `src/ner/mod.rs`
- L36–177: `EntityExtractionStep` largely duplicates `ingest/atomic.rs::extract_entities_and_nlp`. Deprecate/remove the `Step` impl.
- L73, L100–106: `#[cfg(not(feature = "onnx"))]` branch calls a hard-error stub.

### `src/ner/rules.rs`
- L29–52: `Patterns` struct + `static PATTERNS` + `fn patterns()` triple — replace with `static PATTERNS: LazyLock<Patterns>`.

### `src/observations/mod.rs`
- L36–163: `ObservationExtractionStep` parallels atomic path. Deprecate.
- L142–156: `let _ = obs_count;` after final use is dead.
- L171–204: `expect("observation rules cache poisoned")` aborts on lock-poisoning — recoverable via `into_inner`.
- L376: confusing `#[allow(unused_variables)]` on `sender_name` which IS used inside `#[cfg(feature = "onnx")]`.
- L380–388: cfg-shadowed `has_nlp` — `let has_nlp = cfg!(feature = "onnx") && input.nlp_results.is_some();`.
- L626–639: `load_sender_ref_by_lookup` becomes dead after removing legacy step.

### `src/observations/rules.rs`
- L217–260: `extract_observations_from_dep_tree` (cfg `onnx`) may not be called.
- L141–180: hand-curated ~80 English verbs to detect "SVO" — duplicates NLP work. Move to asset.
- L186–209: `make_self_contained` pronoun rewrite is brittle.

### `src/observations/cleanup.rs`
- L110–139: `SUFFIX_RULES` table; doubled-consonant rules per consonant could collapse via a single detector.
- L162–241: 80-entry irregular verb table — move to asset.
- L308–376: `LEADING_STRIP_WORDS` and `REJECT_OBJECT_PHRASES` — also asset candidates.

### `src/observations/rules_engine/matcher.rs`
- L495–548: `collect_children` constructs three `Box<dyn Fn>` closures cloning the entire `words` / `dep_arcs` vectors per match. Plain `match` calling helpers with borrowed slices avoids the heap.
- L550–563: `lookup_collector` builds a global `OnceLock` for `"subtree"` collector — early `if name == "subtree"` branch is cleaner.
- L91–95, L565–571: duplicated `noun_phrase_relations` lookup.
- L437–443: `captures.get("modifier").or_else(|| captures.get("modifiers"))` — standardize on one name in YAML.
- L469–472, L481–484: same pluralisation issue (object/obj/target/complement, time/temporal/when).

### `src/observations/temporal.rs`
- L242–271: `parse_n_ago_with_gran` and `parse_in_n_with_gran` identical apart from one regex token; factor shared builder.
- L321–330: `duration_for_unit` re-implements unit→duration mapping.
- L434–466: `parse_in_month` does 24 substring scans per call; single compiled regex.
- L137–237: long `if … return Some(...)` ladder; table-drive literal anchors.

### `src/nlp/mod.rs`
- **L597–834**: six near-identical tensor extractors (`extract_f32_1d`, `_2d`, `_2d_batch`, `_2d_squeeze_batch`, `_3d`, `_3d_square`, `_3d_squeeze_batch`, `_4d`). Each repeats `outputs.get(name).ok_or_else(...)`, `TensorValue::F32` match, shape check, `into_dimensionality`. **Collapse to one generic `extract_f32::<Dim>(outputs, name, &expected_shape_pattern)`.**
- L370–398: `RowMeta` struct then immediately moved into separate vectors at L459–510 — direct push avoids intermediate struct.
- L171–193: `analyze` slices to single-row to call `compute_srl_frames_batched` then `pop().unwrap_or_default()` — awkward.
- L228: `#[expect(clippy::too_many_arguments)]` — `RowBundle<'a>` struct would simplify (same fix at matcher.rs L31, L332, L495).
- L102–128: three near-identical `TensorValue::I64(ArrayD::from_shape_vec(...).map_err(...))` — one local helper.

### `src/nlp/decode.rs`
- L53–71: `argmax_with_confidence` collects exponentials to a Vec only to sum; drop the `collect`.
- L257–275: `flush` closure inside `decode_srl_frame` could be a private `fn flush_arg`.
- L424–473, L536–689: `extract_svo_triples` and `extract_dep_observations` largely superseded by rules-engine.
- L692–713: `is_unresolvable_pronoun` reimplements vocabulary the rules-engine YAML now owns.
- L719–769: `resolve_subject` (decode) parallels `rules_engine::resolver::resolve_subject`.

### `src/ingest/message.rs`
- L42–126: `ingest_message` is legacy; `atomic.rs::ingest_message_atomic` is supported. Remove legacy + the `IngestStep::execute "message"` arm.
- L246–261: long `if … else …` chain on participant cache — collapse with `match`.
- L432–456: `json_to_uni_value` belongs in `uniko_store::types`.

### `src/ingest/session_chunk.rs`
- L34–148 and L165–389: `chunk_session` and `chunk_session_observations` share ~80% structure; extract `ChunkSessionFlow` helper.
- L286–331 and L336–377: two `ABOUT` propagation blocks differ only in target label + cypher; one helper.

### `src/ingest/atomic.rs`
- L46–72 + L249–256: `AtomicTimings` carries 11 ms fields only consumed by trace log; over-engineered if not read by bench.
- L283–310: `nlp_results` computed via nested `match` with outer `#[allow(unused_assignments)]` — refactor as tuple return.
- L261–332: `extract_entities_and_nlp` has many `#[cfg]` arms + `EntityExtractionOutput` whose only consumer reads 3 fields once. Inline.

### `src/ingest/chunking/text.rs`
- L88–106: `split_paragraphs` recomputes `content[offset..].find(trimmed)` per part — O(n²). Track offset directly.
- L109–126: `split_sentences` doesn't check next-char-is-whitespace despite comment claiming it should.
- L228–240: `merge_small_chunks` calls `count_tokens` repeatedly; cache in `RawChunk`.

### `src/ingest/chunking/html.rs`
- L257–263: parallel arrays for skip tags; extract const `&["script", "style", "noscript"]`.
- L297–354: `decode_entities` named-entity table — extract to asset.

### `src/ingest/chunking/structured.rs`
- L256–283: `format_header_row` and `format_data_row` differ by 2 lines; single `format_row(cols, with_separator: bool)`.
- L36–55: detection cascade does double JSON probe.

### `src/ingest/chunking/code.rs`
- L118–130 and `src/ner/code.rs` L34–47: identical `parser_for_language`; share via common module.
- L132–180 (`classify_node`) and `src/ner/code.rs` L67–105: same kind→type tables.

### `src/embedding/mod.rs`
- L80–97 and L116–132: `embed_batch` and `embed_batch_chunked` overlap.

### `src/ingest/pdf/mod.rs`
- L87–108: two identical idempotency early-returns (by hash, by artifact_id); `dedup_check` helper.

### `src/ingest/pdf/chunker.rs`
- L40–73: two near-identical `ChunkData` constructions; `tag_as_page_chunk(...)` helper.

### `src/ingest/context.rs`
- L52–65: `set_current_speaker` keeps `sentence_ctx.other_speakers` and `other_speakers` in sync manually; derive from `participants` + `current_speaker`.

### `src/ingest/mod.rs`
- L107–127: `deserialize_message` and `deserialize_artifact` differ only by type — generic `deserialize_payload::<T>`.

### Files to consider for outright deletion
- `crates/uniko-extract/src/observations/llm.rs`
- `crates/uniko-extract/src/observations/contradiction.rs`
- `crates/uniko-extract/src/ner/llm.rs`
- `crates/uniko-extract/src/ingest/overflow.rs`

### Highest concentrated wins
- `nlp/mod.rs` (collapse 6 tensor extractors → 1 generic)
- `ingest/message.rs` + `ingest/atomic.rs` (remove legacy ingest path)
- `observations/mod.rs` + `ner/mod.rs` (drop legacy `Step` impls)
- `ingest/session_chunk.rs` (collapse two parallel chunkers)
- `observations/rules_engine/matcher.rs` (drop closure allocations in `collect_children`)

### Verification gaps (caller analysis needed before deleting)
- Whether `ObservationExtractionStep` / `EntityExtractionStep` / `IngestStep` message arm have live callers outside this crate.
- Whether `extract_observations_from_dep_tree`, `extract_svo_triples`, decode-side `resolve_subject` still feed any path.
- Whether the bench reads `AtomicTimings` fields beyond tracing.

---

## uniko-fs

**Crate summary.** The entire `uniko-fs` crate is a placeholder: 3 source modules each contain only `// TODO: implementation in Phase 5`, plus a 7-line `lib.rs` re-exporting them. It declares dependencies (`uniko-api`, `tokio`, `tracing`) that are not used. There is no `tests/` directory. The crate itself is dead scaffolding.

### `crates/uniko-fs/Cargo.toml`
- L9: `uniko-api` dependency unused.
- L10: `tokio` unused.
- L11: `tracing` unused.
- L3: package description advertises capabilities that don't exist.

### `crates/uniko-fs/src/lib.rs`
- L1–7: three `pub mod` declarations expose empty namespaces — premature public API surface.

### `crates/uniko-fs/src/git.rs`, `shadow.rs`, `watcher.rs`
- L1: each is a single TODO comment. Delete the files until Phase 5 begins.

### Recommended consolidation (single action)
Either:
1. Delete `crates/uniko-fs` from the workspace entirely until Phase 5 starts, or
2. Reduce it to `lib.rs` with only the crate doc-comment and drop all submodules + all dependencies.

---

## uniko-mcp

**Crate summary.** The entire `uniko-mcp` crate is scaffolding. `src/server.rs` and `src/tools.rs` each contain a single `// TODO: implementation in Phase 4` line, and `src/lib.rs` only re-exports the empty modules. There are no `tests/` and no real code (8 LOC total). Meanwhile `Cargo.toml` declares non-trivial dependencies (`uniko-api`, `tokio`, `serde_json`, `tracing`) that are unused.

### `crates/uniko-mcp/Cargo.toml`
- L8–13: deps unused. Drop until Phase 4 needs them (or delete the crate from the workspace until then).

### `crates/uniko-mcp/src/lib.rs`
- L5–6: empty modules published — scaffolding for hypothetical futures.

### `crates/uniko-mcp/src/server.rs`, `tools.rs`
- L1: placeholder-only files. Delete.

### Recommendation
Either remove `uniko-mcp` from the workspace (and re-add when Phase 4 lands), or at minimum strip the unused deps and empty submodules.

---

## uniko-memory

**Crate summary.** ~8.4k LoC source, ~3.5k LoC tests. Largest hotspots: `recall/mod.rs` (1866 LoC) and `consolidation.rs` (1150) — both repeat the same sort/truncate/budget-cap dance multiple times. Strongest recurring smell: the `sort_by(|a,b| b.score.partial_cmp(&a.score).unwrap_or(Equal))` idiom appears ~12× across recall + working-memory; same with "iterate items, sum tokens, break on budget". Two dead-code `#[expect(dead_code)]` fields still wired through constructors (`IngestWorker.consolidation_tx`, `ConsolidationWorker.semaphore`) — scaffolding for not-yet-implemented features. Two near-identical `json_to_value` helpers (episode.rs, action.rs). `RecallCounters::bump_video` + `video_channel_active` exist but no recall code actually fires the video channel.

### `src/pipeline/mod.rs`
- L52–62 / L68–127: `PipelineSystem::new` takes 3 args but stores 9 fields; collapse spawn-time wiring into a `PipelineParts` struct.
- L198–204: `Debug` impl hand-written; consider field selection or remove.

### `src/pipeline/ingest_worker.rs`
- L29–34: `consolidation_tx` field `#[expect(dead_code)]` — pure scaffolding. Drop field + ctor arg.
- L38–53: `#[expect(clippy::too_many_arguments)]` 10-arg ctor; collapse via `Config`/`Deps` struct.
- L153–221: `run_step_chain` duplicates failure-handling tail twice.
- L107–110: `match permit.acquire().await { ... }` — `.ok()?` or `let Ok(_permit) = ... else { return };` clearer.

### `src/pipeline/consolidation_worker.rs`
- L24–28: `semaphore` field dead. Remove.
- L139–149: timer-tick branch clones `agent_id` twice.
- L166–201 / L239–275 / L282–317: three near-identical "start clock → emit metric → run op → record duration → log" blocks; factor `instrument_async!` macro.
- L29–82: 7 separate `HashMap`/`Option` fields for cortex bookkeeping; group into `CortexSchedule`.

### `src/recall/mod.rs` (largest opportunity surface)
- **L407–429, L458–491, L542–565, L735–743**: four copies of `items.sort_by(score-desc); items.truncate(limit); accumulate tokens; break on budget` — extract `finalize_bundle(...)` helper. Removes ~80 lines.
- L52–73 / L62–73: `RecallTier::from_label` lists `"Chunk" | "Artifact"` twice.
- L224–266 vs L290–337: `RecallConfig::default` and `from_uniko_config` duplicate the 30-line comment verbatim.
- L862–1037 `phase2_expand`: 175-line function; split into `phase2_active_channels` + `build_phase2_futures`.
- L1131–1249 `phase2_temporal`: comment-vs-code mismatch on scoring.
- L1554–1758 `run_recall_for_variant`: 200 LoC; `hybrid_targets` array has 2 entries differing only by `chunk_type` filter.
- L1567 `#[allow(clippy::type_complexity)]` on 5-tuple — replace with named local struct.
- L1685–1707: vec vs no-vec branching inside loop; lift cypher template build out.

### `src/recall/intent.rs`
- L86–100: `intent_vec`/`keywords` flagged "back-compat with legacy tests"; inline at call sites or remove wrappers.
- L243–251: `join_all` `unwrap_or_default()` swallows errors silently; log on error.
- L438–523: `analyze_query` `cfg(feature = "onnx")` branch is 70 lines; no-onnx fallback is 12. Extract onnx branch.

### `src/recall/mmr.rs`
- L51–106: dual-loop best-selection — `let _ = best_pos;` at L103 admits the first loop's `best_pos` is unused after pruning. Collapse to single pass.
- L118–124: `tokenize` returns `HashSet<String>` per item; consider `HashSet<&str>` over pre-lowercased owned strings.

### `src/recall/modality.rs`
- L37–101: `RecallCounters` exposes `bump_video` and `video()` never called from production code.
- L121–127: `video_channel_active` defined but never called.

### `src/consolidation.rs`
- L228–249: nested `Some(prev) => prev.min(obs...)` ladder appears twice; helper `fn min_opt`.
- L1–35: 35-line module doc duplicates per-function comments.
- L439–469: fact-embedding fallback same pattern as L313–345 object embeddings; extract `embed_or_warn`.
- L803–838: per-row parsing chain; helper `try_parse_observation(row) -> Option<UnprocessedObs>`.

### `src/episode.rs`
- L233–260: `json_to_value` duplicated verbatim in action.rs (L462–485). Move to shared `crate::json_value` module.
- L209–225: `Value::Temporal` / `Value::String` parsing same pattern as rules/lifecycle.rs L356–369 — centralise.
- L153–155: `HashMap::from([("gap_ms".into(), Value::Int(gap_ms))])` is one line.

### `src/action.rs`
- L266–272, L316–320, L331–345: three `#[cfg(feature = "onnx")]` helpers; verify no duplicate with extract crate.
- L322–327: `stable_entity_id` uses `DefaultHasher` — not stable across Rust versions. Correctness risk.
- L348–377: `split_overflow` calls `count_tokens(s)` twice; cache.
- L412–429: `build_embed_text` — `MAX_PAYLOAD_CHARS = 400` possibly redefined in extract crate.
- L439–446: `truncate_chars` likely duplicated in extract crate.
- L451–456: `simple_hash` same `DefaultHasher` concern.

### `src/query.rs`
- L116–193: `record_query_episode` — 80 lines of `state.insert(...)`; switch to `serde_json::json!({...})` macro literal then merge `extra_state`.
- L142–158: `recall_coverage` finite-check — encapsulate as `fn finite_or_zero`.

### `src/working_memory.rs`
- L116: `#[allow(clippy::too_many_lines)]` on a 65-line function — stale.
- L140–147: 6 fetchers share IDs param + lim param + `"WHERE goal_id IN $ids"` prefix; define `goal_scope_clause()` const + `fetch_with(...)` helper.
- L156–168: sort/dedup/truncate — same `finalize_bundle` opportunity.

### `src/nl_to_cypher.rs`
- L66–124: hand-rolled LRU on `HashMap<String, CacheEntry>` with monotonic clock — replace with `lru` crate.
- L188–234: `is_safe_read_only` + `contains_word` 50 lines; one OnceLock regex is 5 lines.
- L292–298: no-onnx `kniv_ner_hints` fallback — combine via cfg inside body.
- L466–482: `clean_response` 4 sequential `strip_prefix`/`strip_suffix` chains; combine.

### `src/rules/lifecycle.rs`
- L158–206: `apply_decay_cycle` outcome cascade — collapse into `decide_new_status(...)` returning `Transition` enum.
- L273–312 / L314–347: `fetch_lifecycle_rules` and `fetch_rule_by_name` share row parsing; extract `row_to_snapshot`.
- L356–369: `extract_optional_dt` same as episode.rs Temporal parsing.

### `src/llm_triples.rs`
- L82–108: `extract_one` returns `Option`; `.ok()?` swallows LLM error silently despite docstring claim.

### tests/
- tests/common/mod.rs (87 LoC): shared test setup. Many `_e2e.rs` files near 200+ LoC likely have repeated KB-bootstrap code that could move into `common`.
- recall_modality_lazy_e2e.rs + recall_cross_modal_fire_e2e.rs are tightly coupled to `RecallCounters` — update if video channel is removed.

### Top 5 highest-leverage cleanups
1. Extract `finalize_bundle()` for sort/truncate/budget pattern (~80 LOC removed).
2. Delete `IngestWorker.consolidation_tx` and `ConsolidationWorker.semaphore` scaffolding.
3. Move `json_to_value` + `extract_optional_dt` + Temporal-to-DateTime parsing into one shared `crate::value_convert` module.
4. Replace hand-rolled LRU with `lru` crate; replace `is_safe_read_only` keyword check with one OnceLock regex.
5. Collapse `mmr_dedup`'s double-best loop into one pass.

---

## uniko-pipes

**Crate summary.** Small, mostly-thin pipeline scaffolding crate (~1.5k LOC across 9 modules). Most code is reasonable, but several public items are unused outside the crate (DLQ retry/list/clear surface, `RetryPolicy` + `retry_with_policy`, `MetricsSnapshot`, parts of `ShutdownCoordinator`, several `PipelineConfig` knobs). Biggest wins: deleting the unused DLQ query/retry methods and the entire `retry` module, plus dropping `item_token`/`Default` plumbing that no caller exercises.

### `Cargo.toml`
- L22: `rand` only used by `retry.rs`; drop if retry is removed.

### `src/lib.rs`
- L20–35: re-exports include `RetryPolicy` (unused externally). Trim re-exports to match real consumers. Verify `IngestPdf`/`PdfInput`/`ObservationsReady` have external consumers before exporting.

### `src/retry.rs` (entire file — 187 lines)
- L1–101: `retry_with_policy` and `RetryPolicy` not referenced anywhere outside this crate. **Delete** the module and the `rand` dependency.
- L75–80: `if let Some(e) = last_err` branch after cancellation unreachable on first attempt.
- L100: `expect("at least one attempt must have run")` panics if `max_attempts == 0`.

### `src/circuit_breaker.rs`
- L62–65: `_ => CircuitState::Closed` defends against impossible u8 value. `unreachable!()` or store enum directly.
- L101–103: second `_ => {}` arm dead.
- L92–103: pre-call match could collapse to `if st == STATE_OPEN { … }`.
- L20–29 + L16–18: hand-rolled `u8` constants + `CircuitState` enum duplicate state representation.

### `src/cancel.rs`
- L51–53: `item_token()` never called outside the crate.
- L56–58: `is_cancelled()` never called externally.
- L106–110: `impl Default` unused.
- L73–86: hardcoded 5s/10s drain phases — pull into `PipelineConfig` or document.
- L13–19: ASCII tree comment restates fields below.

### `src/dead_letter.rs`
- L70–121: `list_pending` — no external caller. **Dead.**
- L128–140: `increment_retry` — **dead.**
- L147–149: `clear` (single item) — **dead.**
- L156–170: `clear_all` — **dead.**
- L16–30: `DeadLetterInfo` only returned by `list_pending`; goes away with it.
- Net effect: module collapses to `DeadLetterQueue::new` + `store`.

### `src/health.rs`
- L121–125: `impl Default` unused.
- L42–52: `PipelineHealth` consumed only by `uniko-memory::pipeline::mod.rs::health()` — consider moving the aggregate struct there.

### `src/metrics.rs`
- L184–200: `MetricsSnapshot` never instantiated. **Delete.**
- L48–55: `uniko.recall.phase1_only_pct` / `uniko.recall.assembly_ms` have no `emit_*` helper or direct macro call — recall doesn't go through this crate.
- L98–181: most `emit_*` helpers are one-line wrappers; doc comments restate function names.

### `src/config.rs`
- L51, L77: `retry: RetryPolicy` field not read by any consumer. **Dead.**
- L59: `dead_letter_check_interval_secs` — no reader. **Dead.**
- L49: `extract_triples_via_llm` — no reader. Probably dead; verify.
- L57–59: `dead_letter_max_retries` is stored on node but never consulted.

### `src/step.rs`
- L36–50: `Step::error_policy()` method redundant — `StepOutcome::Failed` already carries a `policy`.
- L70–84: long doc comments restate field types.
- L37: `should_run` — trait could provide default `fn should_run(_) -> bool { true }`.

### `src/types.rs`
- L60–66, L69–79: `PdfInput` and `IngestPdf` — verify no external readers.
- L93–103: `ObservationsReady` — used only as a variant payload; inline fields into the enum variant.
- L167–177: `ItemResult::new` is trivial; `#[derive(Default)]` + helper.
- L82–90, L106–120: `IngestTask::RunCycle` and `ConsolidationTask::ForceConsolidate` have identical payload; consider collapsing.

### Cross-cutting observations
- "Unused public surface" is the dominant issue. Removing dead items would drop ~300-400 LOC and one dependency (`rand`) without changing observed behavior.
- Several modules carry "future-proof" scaffolding (DLQ retry loop, recall metrics descriptions, `retry_with_policy`) that no consumer reaches.

---

## uniko-shell

**Crate summary.** Effectively empty — a single `main.rs` containing `todo!("Semantic shell — Phase 5")` and a `Cargo.toml` with two dependencies (`uniko-api`, `tokio`) that are never used. No `src/lib.rs`, no `tests/`, no integration code.

### `crates/uniko-shell/Cargo.toml`
- L9–10: `uniko-api` and `tokio` declared but never used. Drop both until Phase 5 needs them.

### `crates/uniko-shell/src/main.rs`
- L1–3: entire binary is `todo!()`. Either remove the crate from the workspace until Phase 5, or replace with a clean `eprintln!` + `std::process::exit(1)` so accidental invocation doesn't panic.

### Cross-cutting
- If Phase 5 work is imminent, keep as-is; if not, removing the crate from the workspace is the highest-value simplification.

---

## uniko-store

**Crate summary.** ~13.7K LOC, 36 src files + 21 test files. The crate is generally well-structured but suffers from three pervasive patterns: (1) `.map_err(|e| UnikoError::Storage(e.to_string()))` boilerplate appearing ~100+ times that could collapse to `?` via the `From` impl; (2) heavy per-call tracing/instrumentation in hot paths (notably `nodes.rs::create_node_in_tx` and `edges.rs::create_message_edges_in_tx`); (3) several near-duplicate helpers (`build_inline_props` vs `build_set_clause`, vector/fulltext search bodies, multi-type/multi-field search). A few dead-code items, dormant scaffolding, and over-eager presets in `config.rs` round out the list.

### `src/error.rs`
- L51–55: `From<uni_db::UniError>` wraps every uni-db error into `Storage` — every storage-layer call site then writes its own `.map_err(|e| UnikoError::Storage(e.to_string()))` instead of `?`. **Drop the explicit map_err sprinkled across `storage/*.rs`, `search/*.rs`, etc.** This alone removes ~150 lines.

### `src/lib.rs`
- L25–31: "uni-db wraps `ModelRuntime` internally but does not re-export it" comment justifies a re-export that may be obsolete.

### `src/id.rs`
- L99–101: `proptest_id_always_valid` uses `_seed` only as loop counter; a plain `for _ in 0..N` test would be clearer.

### `src/config.rs`
- L74–220: 7 named `EmbeddingConfig::*` constructors (Nomic, NomicQ, MiniLM, BGE-small, BGE-large, EmbeddingGemma ONNX, EmbeddingGemma mistralrs). Audit for unused presets — likely belong in `uniko-bench`.
- L310–365: 8 `fn default_*() -> T` helpers for `#[serde(default = "…")]` — replace with `#[serde(default)]` + `Default` impl.
- L724–774: `Default for UnikoConfig` repeats every default already encoded in `default_*()`.
- L777–844: `validate()` does ad-hoc range checks — one helper `check_range(name, value, lo, hi)`.

### `src/types.rs`
- L17–26: `AgentId`, `SessionId`, `GoalId`, `TaskId` — verify usage; delete if unreferenced outside the file.

### `src/storage/mod.rs`
- L96–228: 5 public constructors duplicate the same `validate → load_catalog → Uni::… → apply_schema → prefetch → finalize_init` skeleton. Extend `open_with_xervo_inner` collapse to cover all variants.
- L27–50: `apply_perf_knobs_from_env` triplet — share `parse_bool_env(name)` helper.
- L558–582: 3 `#[expect(dead_code)]` helpers (`vid_to_node_id`, `node_id_to_vid`, `eid_to_edge_id`) — speculative scaffolding. **Delete.**
- L605–620: `validate_property_name` walks chars manually; `name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')` plus first-char check is shorter.
- L625–665: `build_inline_props` and `build_set_clause` are 90% identical; single `build_kv_pairs(fmt_fn, …)` helper.

### `src/storage/nodes.rs`
- L63–119: `create_node_in_tx` is 80% timing instrumentation. Gate behind `tracing::enabled!(Level::DEBUG)` or extract `#[macro] timed_step!`.
- L155–193: `get_node_by_ext_id` matches shape of `get_node` (L131–153). Extract `row_to_node_props` helper.

### `src/storage/edges.rs`
- L210–310: `create_message_edges_in_tx` — ~100 lines, ~50% timing scaffolding. Same gate.
- L297–308: duplicated `tracing::info!(target: "query_metrics", …)` block.
- L361–385: `get_all_edges` and `get_edges` (L318–349) differ only in `:{edge_type}` filter; share body.
- L491–519: `rows_to_edge_records` — every `.get("…")` repeats `.map_err(...)` boilerplate.

### `src/storage/batch.rs`
- L189–212: `batch_create_edges_inner` is a 1-tx wrapper around `batch_create_edges_inner_in_tx`. Extract generic `fn with_tx<F, R>` helper.
- L246–248: `for k in props.keys() { let _ = validate_property_name(k); }` — silently discards validation errors.
- L97–115: two `tracing` emissions with overlapping fields.

### `src/storage/migrations.rs`
- L64–75: 3 near-identical `match row.value(...) { Some(Value::String(s)) => …, _ => … }` — `fn opt_string(row, key) -> Option<String>` helper.
- L99–148: `migrate_artifact_content` MATCH-or-CREATE pattern mirrors `merge_artifact_content` (blob.rs).
- L260–277: `mean_pool` validation + accumulation could fuse via `try_fold`.

### `src/storage/blob.rs`
- L207–218: `hex_encode` reimplements `hex::encode` — drop if `hex` is in tree.
- L220–221: `_ensure_default_used_via_kbconfig` is dead code with unexplained `#[allow(dead_code)]`. **Delete.**

### `src/storage/kb_stats.rs`
- L195–213: `bump_modality_presence` rebuilds entire 4-key map per call. Inline `if cur.has_X { ... }` quartet begs for loop over array.

### `src/storage/filter.rs`
- L65–70: 6 near-identical `Self::Eq(...) => scalar_op(...)` arms — already factored; could table-drive.
- L144–148: special-case for `fragments.len() == 1` is harmless wrapping; remove.

### `src/search/vector.rs`
- L40–72: body nearly identical to `fulltext.rs::fulltext_search`. Extract shared `call_yield_node_score(...)` helper.
- L32–95: `_filter` parameter unused. Implement or drop.

### `src/search/fulltext.rs`
- L17–63: see vector.rs — shared helper.
- L72–109: `multi_field_fulltext_search` (sort-by-score + truncate) identical in shape to `vector.rs::multi_type_vector_search`; extract `merge_top_k(hits, k)`.

### `src/search/hybrid.rs`
- L15–26: 5 `pub const TIER_WEIGHT_*: f64` — prefer `enum Tier { ... }` with `weight()` method.
- L132–167: `rrf_fuse` `.map(|(score, mut hit)| { hit.score = score; hit })` could be clearer `for h in &mut results`.

### `src/search/traversal.rs`
- L141–178: `shortest_path` returns `Ok(None)` on **any** Cypher error — swallows real errors as "no path found".
- L107–122: `depth: 0` populated even though comment says "Depth not computed".
- L233–265: weighted adjacency: `if w == 0.0 { insert nodes; continue }` branch is asymmetric.

### `src/locy/rules.rs`
- L52–58: row-collection idiom repeated in `assume.rs::run` (L82–87) and `abduce.rs::abduce` (L67–76). Extract `fn collect_locy_records(result) -> Vec<Record>`.
- L92–102: `explain_rule` returns `format!("{:?}", result.stats())` — fragile debug-formatted output.

### `src/locy/abduce.rs`
- L78–83: `confidence: 1.0` hardcoded and placeholder explanation. If abduction isn't wired yet, mark `unimplemented!()`.

### `src/locy/assume.rs`
- L65–89: body nearly identical to `rules.rs::execute_rule`; share helper.

### `src/blob_store/mod.rs`
- L156–172: `fs_relative_path` panics on `len < 4`; return `Option<PathBuf>` instead.

### `src/operations/facts.rs`
- L623–651: `parse_btic_lo_millis`, `parse_granularity`, `parse_certainty` are uni-db serialization workarounds. File uni-db issue (per uni-db bug policy) and mark with TODO.
- L959–981: `count_recent_invalidations` computes `earliest_ms` then writes `let _ = earliest_ms;` — dead computation.
- L491–498: `#[expect(clippy::too_many_arguments, …)]` on `write_consolidation_cycle` — `ConsolidationCycleInput` struct.
- L544–550: 5 `if !x.is_empty() { batch_create_edges_fast(...) }` blocks — `for (edge_type, label, targets) in [...]` loop.

### tests/
- Most test files (esp. `unwind_batch_test.rs`, `get_edges_scaling_*`, `profile_*`, `*_repro.rs`) are reproductions for specific bugs. Audit `*_repro.rs` post-fix to confirm underlying uni-db issues are still open; archive/delete those whose bugs are fixed.

### Top wins by impact
1. Replace `.map_err(|e| UnikoError::Storage(e.to_string()))` boilerplate via the existing `From` impl (~100+ sites).
2. Delete dormant scaffolding: `vid_to_node_id` family + `_ensure_default_used_via_kbconfig`.
3. Collapse the 5 KB constructors in `storage/mod.rs` through a single private builder.
4. Gate `tracing::info!(target: "query_metrics", …)` in hot paths.
5. Extract `collect_locy_records` once, used by `rules.rs`/`assume.rs`/`abduce.rs`.
