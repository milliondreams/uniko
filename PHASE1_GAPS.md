# Phase 1 Implementation Gaps

Tracked shortcuts, stubs, and unimplemented features from sub-phases 01–08.
Each item must be resolved before Phase 1 can be considered complete.

Last audited: after A3 (entity embedding dedup) resolution.

## Status Legend

- **RESOLVED** — Fully implemented and tested.
- **STUB** — Function exists but returns empty/error. Zero functionality.
- **MISSING** — Feature not implemented at all. No code exists.
- **DEGRADED** — Implemented with a naive approach that doesn't meet spec quality.
- **PARTIAL** — Core logic exists but not wired into the pipeline where needed.

---

## A. NER Pipeline (sub-06)

| # | Item | File | Status | Spec Requirement | Notes |
|---|------|------|--------|-----------------|-------|
| A1 | ONNX NER model | `uniko-extract/src/ner/onnx.rs` | STUB | Distilled spaCy NER via `ort` crate for PERSON/ORG/GPE/EVENT. Always runs when available. ~60% recall. | Returns `Err("not available")`. Need to pick a model, bundle weights, implement tokenizer + BIO tag alignment. `ort` dep already feature-gated. |
| A2 | LLM NER enhancement | `uniko-extract/src/ner/llm.rs` | STUB | Async LLM call for type refinement, coreference resolution, missed entity extraction. ~90% recall. | Returns `Vec::new()`. Need LLM provider integration (Xervo generate API). Should respect circuit breaker. |
| A3 | Entity embedding dedup | `uniko-extract/src/ner/dedup.rs` | RESOLVED | Dedup cascade: exact → case-insensitive → embedding similarity (0.85 same-type, 0.92 cross-type). Type conflict guard. | Three-tier cascade implemented: (1) exact entity_id match, (2) vector similarity via `find_similar_entity()` with `SIMILARITY_SAME_TYPE=0.85` / `SIMILARITY_CROSS_TYPE=0.92`, (3) create new. Type compatibility guard in `types_compatible()`. Embeddings computed via Xervo and stored on new entities. Graceful fallback when Xervo unavailable. |
| A4 | Coreference resolution | Not implemented | MISSING | "he/she" → most recent person entity in session. "it/this" → most recent non-person entity. Session-scoped. | No pronoun resolution anywhere. `observations/rules.rs` has `make_self_contained()` for leading-pronoun replacement in one sentence, but no cross-sentence coreference. |

## B. Observation Pipeline (sub-07)

| # | Item | File | Status | Spec Requirement | Notes |
|---|------|------|--------|-----------------|-------|
| B1 | LLM observation extraction | `uniko-extract/src/observations/llm.rs` | STUB | LLM extracts implicit facts, complex sentences, paraphrases. ~80% recall. | Returns `Vec::new()`. Same LLM provider dependency as A2. |
| B2 | Contradiction detection | `uniko-extract/src/observations/contradiction.rs` | STUB (deferred by design) | Same-subject + same-relation-slot + different-value → flag for P4. | Returns `Vec::new()`. Embedding similarity alone is insufficient — low similarity means "unrelated", not "contradictory." Proper detection requires predicate extraction (ADR-3) which ships with P4 consolidation. See detailed rationale in `contradiction.rs` module docs. |
| B3 | Question detection | `uniko-extract/src/observations/filter.rs` | DEGRADED | Reject pure questions, keep questions with embedded facts. | Uses `?` check + 4 hardcoded embedded-fact patterns + word-count threshold (<8). Should use interrogative-word detection (who/what/where/when/why/how/is/are/do/does/did/can/could/will/would). |
| B4 | SVO detection | `uniko-extract/src/observations/rules.rs` | DEGRADED | Identify subject-verb-object structure in sentences. | Massive verb regex (~100+ verb forms). No POS tagging or parse-based analysis. Works for common English but brittle. |
| B5 | Consolidation notification | `uniko-extract/src/observations/mod.rs` | MISSING | After creating observations, send `ObservationsReady` to consolidation channel with count + source node IDs. | Not wired. `ObservationExtractionStep` has no access to the consolidation channel. Needs either: (a) add consolidation_tx to PipelineContext, or (b) store notification in ctx.metadata for the worker to read. |
| B6 | Sentence splitting | `uniko-extract/src/observations/rules.rs` | DEGRADED | Split at sentence boundaries with abbreviation awareness (Mr., Dr., etc.). | Simple regex `r"[.!?;]\s+\|\n+"`. No abbreviation handling — "Dr. Smith" splits incorrectly. |

## C. Ingest Pipeline (sub-05)

| # | Item | File | Status | Spec Requirement | Notes |
|---|------|------|--------|-----------------|-------|
| C1 | HTML chunking | `uniko-extract/src/ingest/chunking/html.rs` | STUB | Strip nav/footer/script, split at section/heading boundaries, extract tables separately. | Delegates to `TextChunker`. Needs `scraper` or `lol_html` crate. |
| C2 | PDF chunking | `uniko-extract/src/ingest/chunking/pdf.rs` | STUB | Page extraction, multi-column merge, table detection. | Delegates to `TextChunker`. Needs `pdf-extract` or `lopdf` crate. |
| C3 | Structured data chunking | `uniko-extract/src/ingest/chunking/structured.rs` | STUB | CSV: header + row grouping by token budget. JSON: per-object or per-key splitting. | Delegates to `TextChunker`. Straightforward with `serde_json` (already a dep) + `csv` crate. |
| C4 | Session inactivity timeout | Not implemented | MISSING | Background task checks active sessions every 5 min. End sessions with no message for 30 min. | No background task spawned. Spec: configurable timeout, triggers P7d summarization + re-embedding. |
| C5 | ADDRESSED_TO edges | `uniko-extract/src/ingest/message.rs` | RESOLVED | Message → Participant edges for recipients. Infer from session participants if not provided. | `addressed_to: Option<Vec<String>>` added to `IngestMessage`. `create_addressed_to_edges()` creates edges from explicit list or infers from PARTICIPATED_IN on the session. |
| C6 | PARTICIPATED_IN edges | `uniko-extract/src/ingest/session.rs` | RESOLVED | Participant → Session with role ("initiator", "responder", "observer"). | `ensure_participated_in()` creates edge with role "initiator" (first) or "responder" (subsequent). Called during message ingest after session + participant exist. Idempotent. |
| C7 | FOR_TASK / FOR_GOAL edges | `uniko-extract/src/ingest/session.rs` | MISSING | Session → Task or Session → Goal edges for goal-oriented sessions. | `get_or_create_session` only takes session_id + timestamp. No goal/task reference parameters. |

## D. Pipeline Infrastructure (sub-04)

| # | Item | File | Status | Spec Requirement | Notes |
|---|------|------|--------|-----------------|-------|
| D1 | Dead-letter auto-retry task | `uniko-pipes/src/dead_letter.rs` | MISSING | Background task every 5 min retries eligible DLQ items. | `DeadLetterQueue` has full CRUD. Config has `dead_letter_check_interval_secs: 300` and `dead_letter_max_retries: 3`. But no background sweep task spawned by `PipelineSystem`. |
| D2 | Interactive query preemption | `uniko-memory/src/pipeline/ingest_worker.rs` | MISSING | Biased select with interactive_rx channel for query preemption over background work. | Worker only has cancel + task channels. No interactive query channel. |

## E. Embedding (sub-08)

| # | Item | File | Status | Spec Requirement | Notes |
|---|------|------|--------|-----------------|-------|
| E1 | Auto-embed for Message/Chunk/Observation/Summary | `uniko-store/src/schema/{messages,chunks,observations,summaries}.rs` | RESOLVED | uni-db auto-embeds from content/text field on node creation. | `hnsw_auto_embed_index()` configured. Xervo catalog with fastembed (`AllMiniLML6V2`, 384d) wired in `KnowledgeBase` constructors. |
| E2 | Computed embed wiring | `uniko-extract/src/ner/dedup.rs` | RESOLVED | Entity embeddings computed and stored during NER dedup. | `embed_text()` called in `find_similar_entity()` for similarity search AND in the create-new-entity path to store embedding on the node. Wired as part of A3 resolution. |

## F. Recall & Rules (sub-08)

| # | Item | File | Status | Spec Requirement | Notes |
|---|------|------|--------|-----------------|-------|
| F1 | Basic recall API | `uniko-memory/src/recall/mod.rs` | RESOLVED | Phase 3 (Broaden) search: fulltext + vector + graph traversal, RRF fusion, tier weighting, token budget. | ~330 lines. Searches Message/Chunk/Observation/Entity. RRF k=60, tier weights applied. Coverage scoring. |
| F2 | Stdlib Locy rules | `uniko-memory/src/rules/stdlib.rs` | RESOLVED | 4 rules: relevance_decay, episode_pattern_detector, sequence_detector, contradiction_detector. | Rules registered as Rule nodes + Locy runtime. Idempotent. `is_stdlib_rule()` helper for lifecycle protection. |
| F3 | Consolidation logic | `uniko-memory/src/consolidation.rs` | STUB | Derive Facts from Observations, BTIC invalidation, drift detection. | Empty TODO file. Consolidation worker runs but cycle is a no-op placeholder. Phase 2 work. |

---

## Resolution Priority

### Must fix before Phase 1 sign-off:
1. **B5** (consolidation notification) — wire `ObservationsReady` to consolidation channel

### Should fix before Phase 1 sign-off:
4. **A4** (coreference) — basic pronoun resolution heuristic
5. **B3** (question detection) — add interrogative-word check
6. **B6** (sentence splitting) — add abbreviation list
7. **C7** (FOR_TASK/FOR_GOAL) — needs `get_or_create_session` API change
8. **D1** (DLQ auto-retry) — spawn background task in `PipelineSystem`

### Can defer to Phase 2:
- **A1** (ONNX NER) — requires model selection and bundling
- **A2, B1** (LLM paths) — need LLM provider integration
- **B2** (contradiction detection) — requires predicate extraction (ADR-3) which ships with P4 consolidation; embedding similarity alone is insufficient (see `contradiction.rs` docs)
- **B4** (SVO detection quality) — functional with regex, NLP library is enhancement
- **C1, C2, C3** (HTML/PDF/structured chunking) — functional with text fallback
- **C4** (session inactivity timeout) — needs background task design
- **D2** (interactive preemption) — nice-to-have
- **F3** (consolidation logic) — Phase 2 scope per spec

---

## Resolved Items Log

| Item | Resolved In | What Changed |
|------|-------------|-------------|
| E1 | Auto-embed wiring | `hnsw_auto_embed_index()` on Message/Chunk/Observation/Summary; Xervo catalog in KnowledgeBase constructors |
| F1 | Sub-08 | `recall/mod.rs` — fulltext + vector + graph traversal, RRF fusion |
| F2 | Sub-08 | `rules/stdlib.rs` — 4 Locy rules as Rule nodes |
| A3 | Post sub-08 | Three-tier dedup: exact → embedding similarity → create new. `find_similar_entity()` + `types_compatible()` + embedding stored on new entities |
| E2 | Post sub-08 (with A3) | `embed_text()` called in `dedup.rs` for similarity search and entity creation |
| C5 | Post sub-08 | `addressed_to: Option<Vec<String>>` added to `IngestMessage`. `create_addressed_to_edges()` creates explicit or inferred ADDRESSED_TO edges. |
| C6 | Post sub-08 | `ensure_participated_in()` creates Participant→Session edge with role "initiator"/"responder". Called during message ingest. Idempotent. |

---

## How This Document Should Be Used

1. Work through "must fix" item B5 — the single remaining blocking gap.
2. Each fix should include tests that validate the spec requirement.
3. Do NOT mark Phase 1 as complete until all "must fix" items are resolved.
4. "Should fix" items are quality improvements that strengthen Phase 1 but aren't blocking.
5. "Can defer" items are tracked for Phase 2.
