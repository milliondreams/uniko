# uniko — Implementation Addendum to Spec v6.0

**Enhancements, Alternative Approaches, and Design Decisions**

April 2026 · Companion to `uniko-spec-v6.md`

---

## Purpose

This document records where the uniko implementation diverges from, extends, or takes alternative approaches relative to Specification v6.0. It serves three purposes:

1. **Traceability** — future contributors can understand *why* the code differs from the spec without archaeology.
2. **Spec evolution** — items here feed back into the next spec revision.
3. **Benchmark interpretation** — some divergences directly affect retrieval quality and latency; this document helps diagnose score deltas.

Each section references the relevant spec requirement IDs and code locations.

---

## Part I: Major Architectural Extensions

### 1. Multi-Task NLP Pipeline (replaces ADR-2 "spaCy NER exported to ONNX")

**Spec reference:** ADR-2, F31, NF5

**What the spec says:** "Local NER runs via ONNX Runtime with a lightweight NER model (e.g., distilled spaCy NER exported to ONNX)."

**What we built:** A custom multi-head DeBERTa model (`dragonscale-ai/kniv-deberta-v3-nlp-en`, INT8 quantized) that produces four outputs in a single forward pass:

| Output Head | Spec Expected | Purpose |
|---|---|---|
| NER logits | Yes | 10 entity types (Person, Org, Location, Date, Numeric, Event, Product, WorkOfArt, Group, Misc) |
| POS logits | No | 17 Universal Dependency tags — drives keyword extraction in intent profiles |
| DEP logits | No | dep2label dependency trees — enables structured observation extraction |
| CLS logits | No | 9 sentence classes (inform, question, correction, plan_commit, etc.) — gates P3 extraction |

**Why:** A single multi-task model eliminates 3-4 separate inference calls while providing richer linguistic structure. The dependency trees are the foundation for observation extraction (Section 2). The sentence classifier gates extraction quality (Section 3). POS tags improve recall intent construction (Section 10).

**Trade-off:** Model is larger than a simple NER tagger (~95MB INT8 vs ~15MB for spaCy-distilled). Inference latency is comparable because one forward pass replaces multiple.

**Code:**
- Model configuration: `uniko-store/src/storage/mod.rs` (schema registration)
- NLP pipeline: `uniko-extract/src/nlp/mod.rs` (orchestration), `decode.rs` (post-processing)
- Compile-time assets: `uniko-extract/src/nlp/assets.rs` (tokenizer + label maps)

**Impact on spec requirements:**
- NF5 (NER < 100ms): Met — single forward pass + decode
- F31 (local NER primary): Met — ONNX, no Python dependency
- F72 (operate without LLM): Enhanced — richer local extraction reduces LLM dependency

---

### 2. Dependency-Tree Observation Extraction (alternative to ADR-3 verb-frame patterns)

**Spec reference:** ADR-3, F34, F35

**What the spec says:** "Predicates are extracted from observations using verb-frame patterns: 'X is Y' → predicate: 'is', 'X attended Y' → predicate: 'attended'. Rule-based extraction covers ~60% of common frames."

**What we built:** Observation extraction by walking dependency tree arcs from the NLP pipeline (Section 1). The system finds VERB tokens or copular predicates (ADJ/NOUN with `cop` dependent), collects their `nsubj`/`obj`/`obl`/`xcomp` dependents, and reconstructs grammatically complete statements.

**Example:**

```
Input sentence:  "I'm starting a dance studio in Brooklyn"
Speaker (SENT_BY): Jon
Dependency arcs:  nsubj(starting, I), obj(starting, studio), obl(starting, Brooklyn)
Pronoun resolution: "I" → "Jon" (first-person → speaker)
Output observation: "Jon is starting a dance studio in Brooklyn"
Confidence: 0.85
```

**Key differences from spec:**

| Aspect | Spec (ADR-3) | Implementation |
|---|---|---|
| Method | Regex verb-frame patterns | Dependency tree traversal |
| Input | Raw text | Parsed dep arcs from NLP pipeline |
| Output | `(subject, predicate, object)` triples | Self-contained natural-language sentences |
| Coverage | ~60% of common frames | Broader — handles subordination, copular predicates, modifiers |
| Predicate storage | Structured field on Observation | Deferred to P4 consolidation |

**Why:** Dependency trees provide syntactically grounded extraction that handles complex sentence structures (relative clauses, passives, embedded complements) that regex patterns miss. The cost is requiring the multi-task NLP model from Section 1.

**Consequence for P4:** Because observations are stored as sentences rather than `(S, P, O)` triples, predicate normalization and structured fact derivation are the responsibility of P4 consolidation, not P3 extraction. This is an intentional separation of concerns: P3 extracts; P4 reasons.

**Code:**
- DEP extraction: `uniko-extract/src/nlp/decode.rs` → `extract_dep_observations()`
- Rule-based fallback: `uniko-extract/src/observations/rules.rs`
- Orchestration: `uniko-extract/src/observations/mod.rs`

---

### 3. Sentence Classification Gate (extension to F34)

**Spec reference:** F34

**What the spec says:** "Extract observations (factual statements) from messages — not questions, greetings, or reactions."

**What we built:** A hard per-sentence gate using the CLS head of the DeBERTa model. Sentences are classified before any extraction attempt. Only informative classes proceed:

**Informative (extraction proceeds):**
- `inform` — declarative statements
- `correction` — factual corrections
- `plan_commit` — stated intentions
- `request` — requests containing implicit facts

**Filtered (skipped entirely):**
- `question` — pure questions without embedded facts
- `social` — greetings, farewells
- `filler` — acknowledgments ("okay", "right")
- `agreement` — confirmations without new information
- `feedback` — reactions ("that's great")

**Rule-based fallback** (when ONNX unavailable): A static HashSet of 48+ greeting/filler phrases, plus heuristics for pure questions (< 8 words ending in `?`), system messages, and too-short content (< 5 words).

**Why:** The spec implies post-extraction filtering ("not questions, greetings, or reactions"). Pre-extraction filtering is more efficient — it avoids running dependency analysis on sentences that won't produce observations — and more accurate, because the CLS head is trained specifically for this classification.

**Design note:** Even when a sentence is filtered, its nouns are still recorded as antecedents for pronoun resolution in subsequent sentences (see Section 4).

**Code:**
- CLS gate: `uniko-extract/src/observations/mod.rs` (per-sentence loop)
- Informativeness check: `uniko-extract/src/observations/filter.rs` → `is_informative_by_cls()`
- Rule-based filter: `uniko-extract/src/observations/filter.rs` → `is_informative()`

---

## Part II: Linguistic Extensions (Not in Spec)

### 4. Session-Scoped Pronoun Resolution

**Spec reference:** F2 (speaker attribution via SENT_BY), F34 (observations). Spec is silent on pronoun handling.

**What we built:** A `SentenceContext` that maintains a rolling window of noun antecedents across sentences within a message, enabling pronoun-to-referent resolution:

| Pronoun Category | Resolution Target |
|---|---|
| First-person (`I`, `we`, `me`, `my`, `myself`) | Speaker name (resolved via SENT_BY → Participant) |
| Second-person (`you`, `your`) | First entry in `other_speakers` (other session participants) |
| Third-person (`it`, `this`, `that`, `they`) | Last NOUN/PROPN seen in subject or object position |

**State management:** After each sentence, `update_sentence_context()` records the most recent NOUN/PROPN tokens in subject (`nsubj`, `nsubj:pass`, `csubj`) and object (`obj`, `dobj`, `iobj`) dependency positions — excluding pronouns — for use as antecedents in the next sentence.

**Example chain:**

```
Sentence 1: "Caroline is researching adoption agencies"
  → SentenceContext.last_noun_subject = "Caroline"
  → SentenceContext.last_noun_object = "adoption agencies"

Sentence 2: "She wants to adopt a child from overseas"
  → "She" resolves to "Caroline" (last noun subject)
  → Observation: "Caroline wants to adopt a child from overseas"
```

**Limitation:** Resolution is within a single message's sentences. Cross-message coreference (e.g., "she" in message N referring to an entity in message N-1) is not implemented.

**Code:**
- Context struct: `uniko-extract/src/ingest/context.rs` → `SentenceContext`
- Resolution logic: `uniko-extract/src/nlp/decode.rs` → `resolve_subject()`
- Context updates: `uniko-extract/src/nlp/decode.rs` → `update_sentence_context()`

---

### 5. Temporal Expression Parsing for `observed_at`

**Spec reference:** F34 (observations include timestamp). Spec does not describe computation logic.

**What we built:** A temporal resolution pipeline that interprets relative time expressions in observation text and adjusts the `observed_at` timestamp accordingly:

| Pattern | Example | Resolution |
|---|---|---|
| `"yesterday"` | "I attended a workshop yesterday" | reference − 1 day |
| `"last {period}"` | "Last month I visited Paris" | reference − period duration |
| `"N {units} ago"` | "3 days ago I changed jobs" | reference − parsed duration |
| `"in {month}"` | "In March I started the project" | nearest past occurrence of month |
| No temporal expression | "I work at a hospital" | message timestamp (fallback) |

**Reference timestamp** is the message's own timestamp (from `PipelineContext.metadata["timestamp"]`), falling back to `Utc::now()`.

**Why:** Many conversational statements reference past events ("I moved here last year"). Without temporal parsing, all observations from a single message get the same timestamp, losing the temporal structure that BTIC fact derivation in P4 depends on.

**Code:** `uniko-extract/src/observations/temporal.rs`

---

### 6. Entity Deduplication: 3-Tier Cascade (extends F32)

**Spec reference:** F32 (track entity frequency, first_seen, last_seen)

**What the spec implies:** Basic entity tracking with frequency counting.

**What we built:** A 3-tier deduplication cascade:

1. **In-batch exact match** — entities with the same canonical name within a single extraction batch are merged (highest confidence wins, mention counts accumulate).

2. **Graph-level lookup** — check if entity already exists by deterministic ID `"{type}:{canonical_name}"`. If found: increment frequency, update `last_seen` and confidence. If new: create Entity node.

3. **Embedding-based similarity search** (infrastructure ready, not yet in hot path) — cosine similarity with type-aware thresholds:
   - Same entity type: 0.85
   - Cross-type: 0.92 (stricter to avoid false merges)
   - Blocked pairs: Person ↔ Organization, Person ↔ Location, CodeSymbol ↔ CodeImport, Date ↔ anything

**Canonicalization:** Title-case the surface form (word-by-word capitalization). E.g., "new york city" → "New York City".

**Code:** `uniko-extract/src/ner/dedup.rs`

---

## Part III: Alternative Approaches

### 7. Observations as Chunks for Retrieval (alternative to spec auto-embed)

**Spec reference:** Section 8 (Embedding Strategy) lists Observation under "Auto-embed (uni-db handles)".

**What we do:** Observation nodes are **trace-only** — stored in the graph for provenance and debugging but NOT indexed for vector or fulltext search. Instead, observations from a session are batched into Chunk nodes with `chunk_type = "observation"`, and those chunks are auto-embedded and searched.

**Rationale (from `uniko-store/src/schema/observations.rs`):**

> "Observation nodes are trace-only: stored in the graph for debugging but NOT indexed for search. Session-level observation Chunks (chunk_type='observation') handle retrieval instead."

**Why:** Individual observations are short (often 5-15 words). Embedding each one separately produces sparse, low-context vectors. Batching observations into session-level chunks produces denser embeddings with more context per vector, improving retrieval quality.

**Impact:** Recall Phase 2 (Expand) searches observation chunks rather than observation nodes directly. The spec's "vector search on Observation.embedding" becomes "vector search on Chunk.embedding WHERE chunk_type = 'observation'".

---

### 8. uni-db Native Fusion (alternative to application-level RRF)

**Spec reference:** Section 9 (Recall Cascade) — "Results from different search methods are fused via Reciprocal Rank Fusion (RRF): score(item) = Σ 1/(k + rank_i), k = 60"

**What we do:** Delegate fusion to uni-db's native `similar_to()` function with weighted fusion at the query level:

```cypher
MATCH (m:Chunk)
WHERE similar_to([m.embedding, m.text], [$qvec, $qtxt],
  {method: 'weighted', weights: [0.5, 0.5]})
RETURN m, score()
```

| Aspect | Spec | Implementation |
|---|---|---|
| Fusion location | Application code (Rust) | Inside uni-db query engine |
| Fusion method | Multi-list RRF with k=60 | Weighted vector + BM25 per query |
| Configurability | Hardcoded k=60 | `RecallConfig.vector_weight` / `.bm25_weight` |
| Per-source normalization | Min-max per source | uni-db internal normalization |

**Tier weights** are applied in a second pass in application code, post-search:

```rust
score: raw_score * tier.weight()
```

**Why:** uni-db's `similar_to()` handles vector + fulltext fusion natively and efficiently within the query engine. Application-level RRF would require separate queries, separate result lists, and rank computation — more code, more round-trips, no quality benefit at current scale.

**Trade-off:** Less control over per-source ranking behavior. If future benchmarks show that explicit RRF outperforms weighted fusion, the application-level approach can be restored without schema changes.

**Code:**
- Recall queries: `uniko-memory/src/recall/mod.rs` (hybrid search loop)
- Store-level hybrid: `uniko-store/src/search/hybrid.rs` (RRF implementation for direct API use)

---

### 9. Entity-Scoped Search (alternative to Personalized PageRank)

**Spec reference:** Section 9 Phase 3 — "Personalized PageRank from query entities across the knowledge graph (HippoRAG-inspired, NeurIPS 2024)"

**What we do:** Fixed-pattern Cypher traversal scoped to entities mentioned in the query:

```cypher
MATCH (m:Chunk)<-[:HAS_CHUNK]-(:Session)<-[:PARTICIPATED_IN]-(:Participant {name: $entity_name})
WHERE similar_to(m.embedding, $query_vec)
RETURN m, score()
```

**What's missing vs. spec:**
- Free-form graph traversal via Entity → MENTIONS → Chunk/Message
- PPR with damping=0.85, max_iter=20, top-20 activation spread
- Multi-hop discovery of connections beyond direct participation

**Why deferred:** PPR requires iterative computation across the full graph neighborhood. At cold start with few entities and no facts, the fixed traversal pattern covers the same ground. PPR becomes valuable when the knowledge graph is dense enough for multi-hop connections to matter — likely after P4 consolidation is running.

**Code:** `uniko-memory/src/recall/mod.rs` (entity-scoped search section)

---

### 10. Contradiction Detection Deferred to P4 (spec assigns to P3)

**Spec reference:** F38 — "Detect contradictions: when contradicting observations exceed 40% of total, invalidate the fact by closing its BTIC interval."

**What we do:** `contradiction.rs` returns an empty vector with an explicit explanation:

> "Returns an empty vector. Proper contradiction detection requires predicate-aware comparison (same subject + same relation slot + different value), which ships with P4 consolidation."

**Rationale:** Low embedding similarity does not imply contradiction:

| Pair | Similarity | Relationship |
|---|---|---|
| "Caroline works at hospital" vs. "Caroline likes jazz" | Low | Unrelated (not a contradiction) |
| "Caroline works at hospital" vs. "Caroline works night shifts" | Moderate | Compatible (not a contradiction) |
| "Caroline works at hospital" vs. "Caroline works at law firm" | Moderate-high | **Real contradiction** (same predicate slot, different value) |

Detecting contradictions requires structured `(S, P, O)` comparison — same subject, same predicate slot, different object. Since P3 stores observations as natural-language sentences (Section 2), predicate extraction is a P4 responsibility.

**Impact:** Contradiction detection and BTIC invalidation will activate when P4 consolidation ships. The graph structure (SUPPORTED_BY, INVALIDATES edges) and BTIC type on Facts are already in place.

**Code:** `uniko-extract/src/observations/contradiction.rs`

---

## Part IV: Configuration Divergences

### 11. Chunk Size Default

**Spec reference:** Section 7 — "Recursive 400-512 tokens, sentence-boundary aligned, 10-20% overlap"

| Parameter | Spec | Implementation Default |
|---|---|---|
| `max_chunk_tokens` | 400-512 | **256** |
| `min_chunk_tokens` | (not specified) | 32 |
| Overlap | 10-20% | ~10% (auto: `max/10`, capped at 50) |

**Why 256:** Smaller chunks produce more granular retrieval — each chunk is more likely to be precisely relevant rather than containing a relevant sentence buried in irrelevant context. At 256 tokens with Nomic v1.5 (8192-token context), the model still has ample context for embedding quality. The parameter is fully configurable via `UnikoConfig.max_chunk_tokens`.

**Benchmark note:** If retrieval precision is high but recall is low, increasing to 400-512 may help by providing more context per chunk. This is a tunable parameter for benchmark optimization.

**Code:** `uniko-store/src/config.rs` → `UnikoConfig::default()`

---

### 12. Recall Tier Weights: Two-Layer System

**Spec reference:** Section 9 — single tier weight table (Semantic=1.0, Procedural=0.9, Episodic=0.7, KB=0.5, Provenance=0.4)

**What we have:** Two separate weight systems applied at different layers:

| Tier | Store Layer (`hybrid.rs`) | Recall Layer (`recall/mod.rs`) |
|---|---|---|
| Semantic | 1.0 | 0.9 |
| Procedural | 0.9 | 0.8 |
| Episodic | 0.7 | 0.7 |
| KB/Store | 0.5 | 0.5 |
| Provenance | 0.4 | 0.4 |

The store-layer weights apply during `hybrid_search_weighted()` (direct API). The recall-layer weights apply post-search in the recall cascade. They are not stacked — recall uses its own weights, not the store-layer ones.

**Why the difference:** The recall layer slightly discounts Semantic and Procedural tiers (0.9/0.8 vs 1.0/0.9) because at cold start these tiers have no content; the discount prevents empty-tier bias when Facts and Procedures don't yet exist. The store-layer weights match the spec exactly for direct API callers.

**Code:**
- Store weights: `uniko-store/src/search/hybrid.rs` (constants)
- Recall weights: `uniko-memory/src/recall/mod.rs` → `RecallTier::weight()`

---

### 13. Coverage Scoring: Simplified Formula

**Spec reference:** Section 9 — `coverage = 0.4 * facet_coverage + 0.3 * mean_score + 0.3 * diversity`

**What we compute:**

```rust
coverage = 0.3 * mean_score + 0.3 * diversity
// facet_coverage omitted: always 0 at cold start (no Facts)
```

The `facet_coverage` term (`semantic_items / max(semantic_items, 3)`) evaluates to 0 when no Facts, Procedures, or Topics exist. Rather than computing a term that's guaranteed to be zero, the implementation omits it with a comment noting it activates when consolidation creates semantic items.

**When to restore:** When P4 consolidation ships and Facts begin populating the graph, the full formula should be restored to enable Phase 1 early exit.

**Code:** `uniko-memory/src/recall/mod.rs` (coverage scoring section)

---

## Part V: Partial Implementations

### 14. NEXT Edge gap_ms

**Spec reference:** F6 — "Maintain message ordering via NEXT edges with gap_ms"

**Status:** NEXT edges are created between consecutive messages in a session, but `gap_ms` is hardcoded to 0. No time delta calculation between messages is performed.

**Impact:** Temporal gap analysis (e.g., "messages arrived 3 hours apart, suggesting session boundary") is not yet functional.

**Code:** `uniko-extract/src/ingest/message.rs` (NEXT edge creation)

---

### 15. SENT_BY Role

**Spec reference:** F8 — "Track participant roles per session (initiator, responder, observer)"

**Status:** `PARTICIPATED_IN` edges correctly assign `"initiator"` / `"responder"` roles. However, `SENT_BY` edges are hardcoded with role `"user"` regardless of participant type (human, agent, service).

**Impact:** Queries filtering by sender role (e.g., "messages sent by agents") won't differentiate. Participant type is available on the Participant node itself, so the information exists — it's just not propagated to the edge.

**Code:** `uniko-extract/src/ingest/message.rs` (SENT_BY edge creation)

---

### 16. Session Inactivity Close

**Spec reference:** F14 — "Auto-create sessions for participant+goal combinations; close on inactivity timeout"

**Status:** Session auto-creation works via `get_or_create_session()`. Session inactivity timeout and auto-close are **not implemented**. Sessions persist indefinitely.

**Impact:** Long-running sessions accumulate unbounded message chains. Session-level summaries (P7d) and session-scoped retrieval may become less precise as sessions grow.

**Code:** `uniko-extract/src/ingest/session.rs`

---

### 17. Recall Phases 1 and 2

**Spec reference:** Section 9 — Phase 1 (Compact), Phase 2 (Expand), Phase 3 (Broaden) with coverage-gated early exit.

**Status:** Only Phase 3 (Broaden) is implemented. This is intentional and correct per the spec's own cold-start description:

> "At cold start (no Facts, no Procedures, no Topics), all recalls cascade to Phase 3. phase1_only_pct begins at 0%."

**What activates Phases 1 and 2:**
- Phase 1: Requires Facts from P4 consolidation (vector search on Fact.embedding)
- Phase 2: Requires Episodes from agent recording + Observations (vector search on Episode.embedding, Observation.embedding)
- Coverage-gated early exit: Requires the full coverage formula (Section 13)

**Also not yet implemented:**
- MMR deduplication (spec: lambda=0.7, cosine > 0.85)
- Contrastive retrieval (retrieve failure episodes alongside successes)
- Drift override (force Phase 2+ when drift facts match query entities)
- Abstention logic (return empty bundle when all phases produce < 3 items with max score < 0.15)

**Code:** `uniko-memory/src/recall/mod.rs`

---

### 18. Intent Profile: Keywords over Facet Vecs

**Spec reference:** Section 9 — `IntentProfile` with `facet_vecs: Vec<Vec<f32>>` (per-entity sub-query embeddings)

**What we have:**

```rust
pub struct IntentProfile {
    pub intent_vec: Vec<f32>,       // embedding of keywords (or raw query)
    pub keywords: String,           // extracted keywords for fulltext (not in spec)
    pub entity_refs: Vec<String>,   // extracted entity names
    pub facet_count: usize,         // max(entity_refs.len(), 1)
}
```

**Differences from spec:**
- `facet_vecs` (per-entity embeddings) not implemented — only whole-query `intent_vec`
- `keywords` field added (not in spec) — used for fulltext search parameter
- When ONNX is available, keywords are extracted by POS filtering (NOUN, VERB, PROPN, ADJ, NUM), and the **keywords** — not the raw question — are embedded. This produces vectors closer to statement-form content in the index.

**Why POS-filtered keywords:** Questions like "What did Caroline research?" contain interrogative structure that doesn't appear in indexed content. Stripping to content words ("Caroline research") produces embeddings closer to the statements being searched ("Caroline is researching adoption").

**Code:** `uniko-memory/src/recall/intent.rs`

---

### 19. Content-Type Chunkers: Placeholders

**Spec reference:** Section 7 — chunking strategies per content type (HTML → DOM sections, PDF → page extraction, CSV/JSON → schema-aware row grouping)

**Status:**

| Content Type | Chunker Module Exists | Actual Implementation |
|---|---|---|
| Text (plain, markdown) | Yes | Full recursive splitting with overlap |
| Code (Python, Rust, JS, TS, TSX) | Yes | Full tree-sitter AST chunking |
| HTML | Yes | **Placeholder — falls back to TextChunker** |
| PDF | Yes | **Placeholder — falls back to TextChunker** |
| CSV/JSON/Structured | Yes | **Placeholder — falls back to TextChunker** |
| Audio/Video | No | Not started (Phase 6) |

HTML, PDF, and structured chunkers have module files and dispatch routing but delegate to `TextChunker` internally. The infrastructure for content-type dispatch is complete; the specialized logic is deferred.

**Code:** `uniko-extract/src/ingest/chunking/` (mod.rs for dispatch, individual files per type)

---

## Part VI: Infrastructure Extensions Beyond Spec

### 20. Health Tracking with EMA

**Spec reference:** Pipeline management section mentions "health endpoint with per-worker status and circuit breaker state" but doesn't detail the implementation.

**What we built:**
- Exponential Moving Average latency tracking (α = 0.1)
- Four health status levels: `Healthy`, `Degraded`, `Backpressured`, `Stalled`
- Stall detection: no processing for 300+ seconds with non-empty queue
- Backpressure detection: queue depth / capacity > 0.8
- Circuit breaker integration: open circuit → `Degraded`

**Code:** `uniko-pipes/src/health.rs`

---

### 21. Retry Jitter

**Spec reference:** Pipeline management mentions "3 attempts, exponential backoff (500ms → 30s)" but not jitter.

**What we built:** ±25% uniform jitter applied to each backoff delay to prevent thundering herd when multiple pipeline items fail simultaneously against the same LLM provider.

**Code:** `uniko-pipes/src/retry.rs`

---

### 22. Extended Entity Types

**Spec reference:** F31 mentions "entities" without enumerating types.

**What we extract:** 18+ entity types from three sources:

**From NER model (10 types):** Person, Organization, Location, Date, Numeric, Event, Product, WorkOfArt, Group, Misc

**From rule-based extraction (6 types):** Url, Email, Measurement, Preference, QuotedString, Person (proper-noun fallback)

**From tree-sitter code analysis (2 types):** CodeSymbol (functions, classes, structs), CodeImport (import statements)

**Code:** `uniko-extract/src/nlp/types.rs` (NER types), `uniko-extract/src/ner/rules.rs` (rule types), `uniko-extract/src/ner/code.rs` (code types)

---

### 23. Observation Confidence Scoring

**Spec reference:** F34 mentions observations but not confidence levels.

**What we assign:**

| Extraction Method | Confidence | Rationale |
|---|---|---|
| DEP-tree extraction | 0.85 | Syntactically grounded, high reliability |
| Rule-based: fact/preference pattern | 0.70 | Pattern-matched, moderate reliability |
| Rule-based: SVO structure only | 0.60 | Structural match without semantic confirmation |

**Quality gate:** Observations with fewer than 3 content words are discarded regardless of confidence.

**Code:** `uniko-extract/src/observations/mod.rs`, `rules.rs`

---

## Appendix: Quick Reference

### Divergences by Severity

**Alternative approaches (different method, same goal):**
- Section 2: DEP-tree extraction vs. verb-frame patterns
- Section 7: Observations as chunks vs. direct auto-embed
- Section 8: uni-db native fusion vs. application-level RRF
- Section 9: Entity-scoped search vs. Personalized PageRank

**Extensions (capabilities not in spec):**
- Section 1: Multi-task NLP pipeline (POS, DEP, CLS beyond NER)
- Section 3: Sentence classification gate
- Section 4: Pronoun resolution
- Section 5: Temporal expression parsing
- Section 6: 3-tier entity deduplication
- Section 10: POS-filtered keyword embedding in intent profiles
- Sections 20-23: Health EMA, retry jitter, extended entity types, confidence scoring

**Configuration divergences (tunable):**
- Section 11: Chunk size 256 vs. 400-512
- Section 12: Two-layer tier weights
- Section 13: Simplified coverage formula

**Deferred implementations (infrastructure ready, logic pending):**
- Section 10: Contradiction detection → P4
- Section 14: NEXT edge gap_ms
- Section 15: SENT_BY role propagation
- Section 16: Session inactivity close
- Section 17: Recall Phases 1-2, MMR, contrastive, drift override
- Section 18: Facet vecs in intent profile
- Section 19: HTML/PDF/structured chunkers

---

*This document should be updated as the implementation evolves. Items in "Deferred implementations" should be removed as they ship. Items in "Alternative approaches" should be validated against benchmark results — if a spec approach would measurably improve scores, it should be reconsidered.*
