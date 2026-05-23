# Phase 10: Recall Cascade & Meta-Memory

## Context

This phase implements the 3-phase recall cascade -- the Meta-Memory retrieval engine. Given a query and token budget, it searches across all memory layers with coverage-gated early exit, MMR deduplication, drift override, RRF hybrid scoring, and token budget enforcement.

The recall cascade is the primary read path for all agent queries. It is the mechanism by which uniko's "compile once, query forever" principle pays off: Phase 1 searches compiled knowledge (Facts, Procedures, Topics), and only cascades to raw content (Phase 2/3) when compiled knowledge is insufficient. The `phase1_only_pct` metric tracks how often Phase 1 alone satisfies queries -- this is the primary signal that consolidation is working and the system is improving over time.

The recall cascade lives in `uniko-memory` and calls into `uniko-extract` for embedding and NER, and `uniko-store` (KnowledgeBase) for vector search, fulltext search, and graph traversal.

**Key differentiator:** No competitor offers formal coverage scoring with early exit. All competitors run the same retrieval pipeline regardless of query complexity. uniko's cascade means simple queries about well-known facts complete in < 30ms (Phase 1 only), while novel or ambiguous queries get progressively deeper search (Phase 2/3) up to < 100ms.

## Prerequisites

- **Phase 8 (Embedding Pipeline P7)** -- all node types must have embeddings computed so vector search returns results. Fact.embedding, Procedure.embedding, Topic.embedding, Episode.embedding, Observation.embedding, Message.embedding, Chunk.embedding must all be populated.
- **Phase 9 (Consolidation Pipeline P4)** -- consolidation must be functional so that Facts exist for Phase 1 to find. Without consolidation, Phase 1 returns nothing and every query cascades to Phase 3 (valid but defeats the purpose of the cascade).
- **Phase 2 (NER Pipeline P2)** -- entity extraction must be functional for IntentProfile construction (entity_refs extraction from queries).
- **uniko-store (KnowledgeBase)** -- vector search, fulltext search, and graph traversal APIs must be operational.
- **uniko-extract** -- embedding model must be loaded and callable for intent_vec computation.

## Sub-phases

---

### 10.1 -- IntentProfile Construction

**Objective:** Convert a raw query string into a structured IntentProfile that drives all subsequent search operations.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/mod.rs` | Rust | Recall module root with submodule declarations |
| `crates/uniko-memory/src/recall/intent.rs` | Rust | IntentProfile construction logic |

#### Types

```rust
pub struct IntentProfile {
    pub intent_vec: Vec<f32>,           // embedding of full query text
    pub facet_vecs: Vec<Vec<f32>>,      // per-entity sub-query embeddings
    pub entity_refs: Vec<String>,       // extracted entity names
    pub facet_count: usize,             // max(entity_refs.len(), 1)
}
```

#### Functions

| Function | Signature | Description |
|---|---|---|
| `build_intent_profile` | `async fn build_intent_profile(query: &str, ner: &EntityExtractor, embedder: &EmbedModel) -> Result<IntentProfile>` | Orchestrates the full intent construction pipeline |

#### Implementation Steps

1. **Embed full query** -- call `embedder.embed(query)` to produce `intent_vec`. This vector drives all vector searches across all phases.
2. **Extract entities** -- call `ner.extract(query)` using the P2 NER path (local, < 100ms). Extract named entities from the query text (e.g., "Caroline" from "What did Caroline research?"). Store as `entity_refs`.
3. **Embed each entity** -- for each entity in `entity_refs`, call `embedder.embed(entity_name)` to produce individual facet vectors. Store as `facet_vecs`. These enable entity-boosted search in later phases.
4. **Compute facet_count** -- `max(entity_refs.len(), 1)`. This feeds coverage scoring (facet_coverage denominator).

#### Performance Target

- Total IntentProfile construction: < 100ms (NF5 entity extraction is the bottleneck)
- Entity extraction must use local NER only, not LLM path

#### Edge Cases

- Query with no extractable entities: `entity_refs = []`, `facet_vecs = []`, `facet_count = 1`
- Very long query (> 512 tokens): truncate to embedding model's context window before embedding
- Non-English query: NER may extract fewer entities; system proceeds with `facet_count = 1`

---

### 10.2 -- Phase 1: Compact

**Objective:** Search compiled knowledge (Facts, Procedures, Topics) for high-quality semantic matches. This is the cheapest path -- if coverage is sufficient, return immediately without Phase 2/3.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/phase1.rs` | Rust | Phase 1 compact retrieval logic |
| `crates/uniko-memory/src/recall/types.rs` | Rust | RecallItem, Tier, and scoring types |

#### Types

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tier {
    Semantic,       // Facts, Topics         -- weight 1.0
    Procedural,     // Procedures            -- weight 0.9
    Episodic,       // Episodes, Observations -- weight 0.7
    KnowledgeBase,  // Chunks, Artifacts     -- weight 0.5
    Provenance,     // Actions, Messages     -- weight 0.4
}

#[derive(Debug, Clone)]
pub struct RecallItem {
    pub node_id: NodeId,
    pub node_type: String,
    pub score: f64,
    pub tier: Tier,
    pub content: String,
    pub embedding: Option<Vec<f32>>,  // cached for MMR computation
}
```

#### Functions

| Function | Signature | Description |
|---|---|---|
| `execute_phase1` | `async fn execute_phase1(intent: &IntentProfile, kb: &KnowledgeBase) -> Result<Vec<RecallItem>>` | Runs vector search on Phase 1 node types |
| `tier_weight` | `fn tier_weight(tier: Tier) -> f64` | Returns the scoring weight for a tier |

#### Implementation Steps

1. **Vector search on Facts** -- `kb.vector_search("Fact", "embedding", &intent.intent_vec, top_k=20)`. Score each result: `cosine_similarity * 1.0` (Semantic tier weight).
2. **Vector search on Procedures** -- `kb.vector_search("Procedure", "embedding", &intent.intent_vec, top_k=10)`. Score: `cosine_similarity * 0.9` (Procedural tier weight).
3. **Vector search on Topics** -- `kb.vector_search("Topic", "embedding", &intent.intent_vec, top_k=5)`. Score: `cosine_similarity * 1.0` (Semantic tier weight).
4. **Min-max normalization** -- normalize all cosine scores to [0,1] across the combined result set. Formula: `(score - min_score) / (max_score - min_score)`. If all scores identical, set all to 1.0.
5. **Abstention check** -- if `max(score) < 0.3 AND coverage < 0.2`, flag "low confidence" on the result set. Do not abort; proceed to coverage check.
6. **Return** -- combined `Vec<RecallItem>` sorted by normalized score descending.

#### Scoring Contract

| Source | Tier | Weight | Top-K |
|---|---|---|---|
| Fact.embedding | Semantic | 1.0 | 20 |
| Procedure.embedding | Procedural | 0.9 | 10 |
| Topic.embedding | Semantic | 1.0 | 5 |

#### Performance Target

- Phase 1 total: < 30ms (NF7 -- compact-only assembly)
- Three vector searches can run concurrently (futures joined)

---

### 10.3 -- Phase 2: Expand

**Objective:** Search episodic memory (Episodes, Observations, Messages) when Phase 1 coverage is insufficient. Adds recency-boosted scoring, RRF fusion across multiple vector sources, and optional contrastive retrieval.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/phase2.rs` | Rust | Phase 2 expand retrieval logic |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `execute_phase2` | `async fn execute_phase2(intent: &IntentProfile, kb: &KnowledgeBase, phase1_items: &[RecallItem], contrastive: bool) -> Result<Vec<RecallItem>>` | Runs vector + fulltext search on Phase 2 node types |
| `recency_boost` | `fn recency_boost(recency_rank: usize) -> f64` | Computes `1.0 + 0.1 * recency_rank` |
| `rrf_fuse` | `fn rrf_fuse(ranked_lists: &[Vec<RecallItem>], k: u32) -> Vec<RecallItem>` | Reciprocal Rank Fusion across multiple source lists |

#### Implementation Steps

1. **Vector search on Episodes** -- `kb.vector_search("Episode", "embedding", &intent.intent_vec, top_k=20)`. Score: `cosine * 0.7 * recency_boost`.
2. **Vector search on Observations** -- `kb.vector_search("Observation", "embedding", &intent.intent_vec, top_k=20)`. Score: `cosine * 0.7 * recency_boost`.
3. **Vector search on Messages** -- `kb.vector_search("Message", "embedding", &intent.intent_vec, top_k=10)`. Score: `cosine * 0.4 * recency_boost`. (Provenance tier weight = 0.4.)
4. **Fulltext search on Messages** -- `kb.fulltext_search("Message", "content", query_text, top_k=10)`.
5. **Fulltext search on Observations** -- `kb.fulltext_search("Observation", "content", query_text, top_k=10)`.
6. **RRF fusion** -- apply Reciprocal Rank Fusion across all 5 source lists with k=60. Formula: `score(item) = sum(1/(60 + rank_i))` for each retrieval method where the item appears.
7. **Contrastive mode** (if enabled) -- additionally retrieve Episodes where `outcome = 'failure'` for entities matching `intent.entity_refs`. These provide negative examples alongside positive results.
8. **Merge** -- combine Phase 1 items + Phase 2 items, re-sort by fused score descending.

#### RRF Formula

```
score(item) = Σ 1/(k + rank_i)  for each retrieval method i where item appears
k = 60 (standard RRF constant)
```

Items appearing in multiple retrieval methods get higher fused scores (reward for multi-signal agreement).

#### Recency Boost

```
recency_boost = 1.0 + 0.1 * recency_rank
```

Where `recency_rank` is the item's position when sorted by timestamp descending (most recent = highest rank). This gently favors recent content without overwhelming relevance scoring.

#### Scoring Contract

| Source | Type | Tier | Weight | Top-K |
|---|---|---|---|---|
| Episode.embedding | Vector | Episodic | 0.7 | 20 |
| Observation.embedding | Vector | Episodic | 0.7 | 20 |
| Message.embedding | Vector | Provenance | 0.4 | 10 |
| Message.content | Fulltext | Provenance | 0.4 | 10 |
| Observation.content | Fulltext | Episodic | 0.7 | 10 |

---

### 10.4 -- Phase 3: Broaden

**Objective:** Search raw content (Chunks, Messages) with fulltext BM25, vector search, graph traversal via entity links, and Personalized PageRank. This is the broadest and most expensive phase. Always completes (no early exit).

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/phase3.rs` | Rust | Phase 3 broaden retrieval logic |
| `crates/uniko-memory/src/recall/ppr.rs` | Rust | Personalized PageRank implementation |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `execute_phase3` | `async fn execute_phase3(intent: &IntentProfile, kb: &KnowledgeBase, prior_items: &[RecallItem]) -> Result<Vec<RecallItem>>` | Runs fulltext + vector + graph + PPR search |
| `graph_traverse_entities` | `async fn graph_traverse_entities(entity_refs: &[String], kb: &KnowledgeBase) -> Result<Vec<RecallItem>>` | Follows Entity -> MENTIONS -> Chunk/Message edges |
| `personalized_pagerank` | `fn personalized_pagerank(seed_nodes: &[NodeId], kb: &KnowledgeBase, damping: f64, max_iter: usize, top_k: usize) -> Result<Vec<(NodeId, f64)>>` | PPR from query entity nodes |

#### Implementation Steps

1. **Fulltext BM25 on Chunks** -- `kb.fulltext_search("Chunk", "text", query_text, top_k=20)`. Tier: KnowledgeBase (weight 0.5).
2. **Fulltext BM25 on Messages** -- `kb.fulltext_search("Message", "content", query_text, top_k=10)`. Tier: Provenance (weight 0.4).
3. **Vector search on Chunks** -- `kb.vector_search("Chunk", "embedding", &intent.intent_vec, top_k=10)`. Tier: KnowledgeBase (weight 0.5).
4. **Graph traversal** -- for each entity in `intent.entity_refs`, find the Entity node, then traverse `MENTIONS` edges to collect connected Chunk and Message nodes. Tier determined by node type.
5. **Personalized PageRank** -- seed PPR with Entity nodes matching `intent.entity_refs`. Parameters: `damping=0.85`, `max_iter=20`, `top_k=20`. PPR discovers multi-hop connections that direct graph traversal misses (HippoRAG-inspired, NeurIPS 2024). Collect top-scoring nodes as RecallItems.
6. **Per-source normalization** -- min-max normalize scores to [0,1] within each source before RRF.
7. **RRF fusion** -- apply RRF (k=60) across all 5+ source lists (fulltext Chunks, fulltext Messages, vector Chunks, graph traversal, PPR).
8. **Merge all phases** -- combine Phase 1 + Phase 2 + Phase 3 items, re-sort by `fused_score * tier_weight`.
9. **Final MMR pass** -- apply MMR deduplication over the full merged bundle (see 10.6).
10. **Abstention check** -- if `max(score) < 0.15` across ALL phases AND total items < 3, return empty bundle with `abstention: true`.

#### PPR Algorithm

```
Input: seed_nodes (entity NodeIds), damping=0.85, max_iter=20
Initialize: scores[seed] = 1/|seeds|, all others = 0
For each iteration:
    new_scores = (1 - damping) * seed_distribution
    For each node n with score > 0:
        For each neighbor m of n:
            new_scores[m] += damping * scores[n] / out_degree(n)
    scores = new_scores
Return top-20 nodes by score (excluding seed nodes)
```

#### Scoring Contract

| Source | Type | Tier | Weight | Top-K |
|---|---|---|---|---|
| Chunk.text | Fulltext BM25 | KnowledgeBase | 0.5 | 20 |
| Message.content | Fulltext BM25 | Provenance | 0.4 | 10 |
| Chunk.embedding | Vector | KnowledgeBase | 0.5 | 10 |
| Entity -> MENTIONS | Graph traversal | varies | varies | all reachable |
| PPR from entities | Graph | varies | varies | 20 |

#### Performance Target

- Phase 3 total (including PPR): should keep all-phases total < 100ms (NF8)
- PPR with 20 iterations over a 10K-node graph should complete in < 30ms

---

### 10.5 -- Coverage Scoring & Early Exit

**Objective:** Compute a coverage score after each phase to determine whether sufficient information has been retrieved, enabling early exit from the cascade.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/coverage.rs` | Rust | Coverage scoring and early exit logic |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `compute_coverage` | `fn compute_coverage(items: &[RecallItem], facet_count: usize) -> f64` | Computes the coverage score for a result set |
| `should_exit_early` | `fn should_exit_early(coverage: f64, phase: u8, config: &UnikoConfig) -> bool` | Determines whether to skip remaining phases |

#### Coverage Formula

```
semantic_items = count of items where tier in {Semantic, Procedural}
facet_coverage = semantic_items / max(semantic_items, 3)
mean_score     = mean(item.score for all items)
diversity      = distinct_tier_count / 5     (5 tiers max)

coverage = 0.4 * facet_coverage + 0.3 * mean_score + 0.3 * diversity
```

#### Component Breakdown

| Component | Weight | What It Measures | Range |
|---|---|---|---|
| `facet_coverage` | 0.4 | Proportion of compiled knowledge found (Facts, Procedures) | [0.0, 1.0] |
| `mean_score` | 0.3 | Average relevance of all retrieved items | [0.0, 1.0] |
| `diversity` | 0.3 | How many distinct memory tiers are represented | [0.0, 1.0] |

#### Thresholds (Configurable via UnikoConfig)

| Phase | Threshold | Config Field | Meaning |
|---|---|---|---|
| Phase 1 | 0.75 | `phase1_coverage_threshold` | High bar -- only exit early if compiled knowledge strongly covers the query |
| Phase 2 | 0.65 | `phase2_coverage_threshold` | Lower bar -- episodic memory supplements facts |
| Phase 3 | N/A | N/A | Always completes (no early exit) |

#### Decision Logic

```
After Phase 1:
    coverage = compute_coverage(phase1_items, facet_count)
    if coverage >= 0.75 AND no drift override:
        return phase1_items  (early exit)
    else:
        proceed to Phase 2

After Phase 2:
    coverage = compute_coverage(merged_items, facet_count)
    if coverage >= 0.65:
        return merged_items  (early exit)
    else:
        proceed to Phase 3

After Phase 3:
    always return full merged bundle
```

---

### 10.6 -- MMR Deduplication

**Objective:** Remove near-duplicate items from the result set using Maximal Marginal Relevance, balancing relevance against diversity.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/mmr.rs` | Rust | MMR deduplication algorithm |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `apply_mmr` | `fn apply_mmr(items: &mut Vec<RecallItem>, lambda: f64, dup_threshold: f64)` | Removes near-duplicates from the item list in-place |
| `cosine_similarity` | `fn cosine_similarity(a: &[f32], b: &[f32]) -> f64` | Cosine similarity between two embedding vectors |
| `jaccard_similarity` | `fn jaccard_similarity(a: &str, b: &str) -> f64` | Word-level Jaccard similarity as fallback |

#### Parameters

| Parameter | Value | Description |
|---|---|---|
| `lambda` | 0.7 | Balance between relevance (1.0) and diversity (0.0). 0.7 favors relevance. |
| `dup_threshold` | 0.85 | Cosine similarity above this threshold = duplicate |

#### Algorithm

```
1. Sort items by score descending
2. selected = [items[0]]  (highest-scored item always selected)
3. For each remaining item candidate:
    max_sim = max cosine_similarity(candidate.embedding, s.embedding) for s in selected
    if max_sim > 0.85:
        skip candidate (duplicate)
    else:
        mmr_score = lambda * candidate.score - (1 - lambda) * max_sim
        if mmr_score > 0:
            add candidate to selected
4. items = selected
```

#### Fallback

When `embedding` is `None` on a RecallItem (embedding not cached), fall back to Jaccard word overlap on `content` field:

```
jaccard(a, b) = |words(a) ∩ words(b)| / |words(a) ∪ words(b)|
```

Use same threshold (0.85) for Jaccard-based deduplication.

#### Application Points

- **Phase 2:** MMR applied after Phase 2 merge (Phase 1 + Phase 2 items)
- **Phase 3:** Final MMR pass over full bundle (all phases merged)

---

### 10.7 -- Drift Override

**Objective:** When a query references entities flagged as "drifting" (unstable knowledge from P4 drift detection), force the cascade past Phase 1 early exit to ensure recent episodic evidence is always consulted.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/drift.rs` | Rust | Drift detection and override logic |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `check_drift_override` | `async fn check_drift_override(entity_refs: &[String], kb: &KnowledgeBase) -> Result<bool>` | Returns true if any referenced entity has a drift flag |

#### Implementation Steps

1. For each entity name in `intent.entity_refs`, look up the Entity node in the graph.
2. Check if the entity has `drift_flagged = true` (set by P4 consolidation when an entity accumulates > 4 invalidations in 30 days).
3. If ANY referenced entity is drift-flagged, return `true`.
4. When drift override is active:
   - Phase 1 early exit is **skipped** regardless of coverage score
   - Phase 2 (and Phase 3 if needed) execute unconditionally
   - This ensures queries about entities with recently-changing facts always check the latest episodic evidence

#### Rationale

Drift flags indicate that the compiled knowledge (Facts) about an entity is unstable -- recent contradictions have been detected. Phase 1 might return a Fact that was true last week but has since been invalidated. By forcing Phase 2+, the cascade picks up recent Episodes and Observations that reflect the current state.

#### Performance Target

- Drift check: < 100ms (NF12)
- Entity lookup + drift flag check should be fast indexed queries

---

### 10.8 -- Token Budget Enforcement & ContextBundle

**Objective:** Package the final recall results into a ContextBundle, enforcing the caller's token budget by truncating lower-scored items.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/bundle.rs` | Rust | ContextBundle assembly and token budget enforcement |

#### Types

```rust
pub struct ContextBundle {
    pub items: Vec<RecallItem>,
    pub coverage: f64,
    pub phases_executed: Vec<u8>,       // e.g., [1] or [1, 2] or [1, 2, 3]
    pub abstention: bool,               // true if no meaningful results found
    pub assembly_latency_ms: u64,       // total time to assemble this bundle
    pub low_confidence: bool,           // true if Phase 1 abstention check triggered
}
```

#### Functions

| Function | Signature | Description |
|---|---|---|
| `assemble_bundle` | `fn assemble_bundle(items: Vec<RecallItem>, coverage: f64, phases: Vec<u8>, budget: usize, latency_ms: u64) -> ContextBundle` | Packages items into a budget-constrained bundle |
| `estimate_tokens` | `fn estimate_tokens(items: &[RecallItem]) -> usize` | Estimates total token count for a set of items |

#### Token Budget Enforcement

1. **Rank items** by `score * tier_weight` descending (already done by merge step).
2. **Estimate tokens** -- approximately 50 tokens per item (spec-defined constant). This is a rough estimate; actual token count depends on content length.
3. **Truncate** -- remove items from the tail until `estimated_tokens <= budget`.
4. **Default budget** -- 8192 tokens if caller does not specify.

#### Budget Calculation

```
max_items = budget / 50  (tokens_per_item estimate)
items = items[..min(items.len(), max_items)]
```

For budget=8192, max_items = 163. In practice, limit (from RecallContextBuilder) will cap results well below this.

---

### 10.9 -- RecallContextBuilder API

**Objective:** Provide an ergonomic builder-pattern API for configuring and executing recall queries. This is the primary entry point for all recall operations.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/builder.rs` | Rust | RecallContextBuilder implementation |

#### Types

```rust
pub struct RecallContextBuilder<'a> {
    cortex: &'a Cortex,
    intent: IntentProfile,
    limit: usize,                       // default: 15
    tier_weights: Option<HashMap<Tier, f64>>,
    recency_window: Option<Duration>,   // filter by time window
    min_reliability: f64,               // default: 0.4
    include_procedures: bool,           // default: true
    include_kb: bool,                   // default: true
    contrastive: bool,                  // default: false
    budget: usize,                      // default: 8192 tokens
}
```

#### Builder Methods

| Method | Signature | Description |
|---|---|---|
| `new` | `fn new(cortex: &'a Cortex, intent: IntentProfile) -> Self` | Creates builder with defaults |
| `limit` | `fn limit(mut self, n: usize) -> Self` | Max items in result bundle |
| `weights` | `fn weights(mut self, w: HashMap<Tier, f64>) -> Self` | Override per-tier scoring weights |
| `recency_window` | `fn recency_window(mut self, days: u64) -> Self` | Filter items to within N days |
| `min_reliability` | `fn min_reliability(mut self, r: f64) -> Self` | Minimum item reliability score |
| `include_procedures` | `fn include_procedures(mut self, b: bool) -> Self` | Include/exclude procedural tier |
| `include_kb` | `fn include_kb(mut self, b: bool) -> Self` | Include/exclude KB/provenance tiers |
| `contrastive` | `fn contrastive(mut self, b: bool) -> Self` | Enable contrastive retrieval (success + failure episodes) |
| `assemble` | `async fn assemble(self) -> Result<ContextBundle>` | Terminal: executes the full cascade and returns results |

#### Cascade Orchestration (inside `assemble`)

```
1. Start timer
2. Check drift override for intent.entity_refs
3. Execute Phase 1 (compact)
4. Compute coverage
5. If coverage >= phase1_coverage_threshold AND no drift override:
       → assemble_bundle(phase1_items) → return
6. Execute Phase 2 (expand, with contrastive flag)
7. Merge Phase 1 + Phase 2
8. Apply MMR
9. Compute coverage
10. If coverage >= phase2_coverage_threshold:
        → assemble_bundle(merged_items) → return
11. Execute Phase 3 (broaden)
12. Merge all phases
13. Apply final MMR
14. Compute final coverage
15. Check abstention (max_score < 0.15 AND items < 3)
16. assemble_bundle(all_items) → return
17. Record assembly_latency_ms
18. Emit phase1_only_pct metric
```

#### Metrics

| Metric | Type | Description |
|---|---|---|
| `recall_phase1_only_pct` | Gauge | Percentage of recall queries satisfied by Phase 1 alone. Should trend upward as consolidation improves. |
| `recall_assembly_latency_ms` | Histogram | Time to assemble a ContextBundle |
| `recall_phases_executed` | Histogram | Number of phases executed per query (1, 2, or 3) |
| `recall_items_returned` | Histogram | Number of items in the returned bundle |
| `recall_coverage` | Histogram | Final coverage score |
| `recall_abstention_count` | Counter | Number of queries that resulted in abstention |

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_intent_profile_basic` | `recall/intent.rs` | Query "What did Caroline research?" produces intent_vec (non-empty), entity_refs=["Caroline"], facet_count=1 |
| `test_intent_profile_no_entities` | `recall/intent.rs` | Query "tell me something interesting" produces entity_refs=[], facet_count=1 |
| `test_intent_profile_multiple_entities` | `recall/intent.rs` | Query "Did Alice meet Bob?" produces entity_refs=["Alice", "Bob"], facet_count=2 |
| `test_phase1_fact_retrieval` | `recall/phase1.rs` | Store 5 Facts with embeddings, query related topic, top result is the matching Fact |
| `test_phase1_procedure_retrieval` | `recall/phase1.rs` | Store Procedures with embeddings, query matches correct Procedure |
| `test_phase1_scoring` | `recall/phase1.rs` | Fact scores higher than Procedure at same cosine similarity (1.0 vs 0.9 weight) |
| `test_phase1_min_max_normalization` | `recall/phase1.rs` | Scores normalized to [0,1] range correctly |
| `test_phase1_abstention` | `recall/phase1.rs` | All scores < 0.3 and coverage < 0.2 triggers low_confidence flag |
| `test_phase2_episode_retrieval` | `recall/phase2.rs` | Store Episodes, query retrieves matching episodes |
| `test_phase2_recency_boost` | `recall/phase2.rs` | More recent items score higher than older items at same cosine similarity |
| `test_phase2_rrf_fusion` | `recall/phase2.rs` | Item appearing in multiple sources scores higher than item in one source |
| `test_phase2_contrastive` | `recall/phase2.rs` | Contrastive mode retrieves failure episodes alongside successes |
| `test_phase3_fulltext_chunks` | `recall/phase3.rs` | BM25 search on Chunk.text returns keyword-matching chunks |
| `test_phase3_graph_traversal` | `recall/phase3.rs` | Entity -> MENTIONS -> Chunk/Message edges followed correctly |
| `test_phase3_ppr` | `recall/ppr.rs` | PPR from seed entities discovers multi-hop connections |
| `test_phase3_ppr_convergence` | `recall/ppr.rs` | PPR converges within 20 iterations on test graph |
| `test_phase3_abstention` | `recall/phase3.rs` | max_score < 0.15 and items < 3 across all phases triggers abstention=true |
| `test_coverage_formula` | `recall/coverage.rs` | Known inputs produce expected coverage score |
| `test_coverage_all_semantic` | `recall/coverage.rs` | All Semantic tier items, high scores, multiple tiers = high coverage |
| `test_coverage_no_semantic` | `recall/coverage.rs` | No Semantic/Procedural items = low facet_coverage component |
| `test_early_exit_phase1` | `recall/coverage.rs` | Coverage >= 0.75 returns true for Phase 1 exit |
| `test_no_early_exit_low_coverage` | `recall/coverage.rs` | Coverage < 0.75 returns false for Phase 1 exit |
| `test_mmr_duplicate_removal` | `recall/mmr.rs` | Two items with cosine > 0.85 -- lower-scored one removed |
| `test_mmr_preserves_diverse` | `recall/mmr.rs` | Two items with cosine < 0.85 -- both preserved |
| `test_mmr_jaccard_fallback` | `recall/mmr.rs` | Items without embeddings use Jaccard word overlap for dedup |
| `test_drift_override_active` | `recall/drift.rs` | Entity with drift_flagged=true triggers drift override |
| `test_drift_override_inactive` | `recall/drift.rs` | Entity without drift flag does not trigger override |
| `test_token_budget_truncation` | `recall/bundle.rs` | Items exceeding budget are truncated from tail |
| `test_token_budget_default` | `recall/bundle.rs` | Default budget of 8192 tokens |
| `test_bundle_metadata` | `recall/bundle.rs` | phases_executed, coverage, abstention, latency_ms populated correctly |
| `test_rrf_ordering` | `recall/phase2.rs` | RRF produces correct ordering: items in multiple lists ranked highest |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_cascade_phase1_sufficient` | `tests/recall_integration.rs` | Store well-covered Facts, query returns Phase 1 only (phases_executed=[1]) |
| `test_cascade_phase2_needed` | `tests/recall_integration.rs` | Insufficient Facts, cascade reaches Phase 2 (phases_executed=[1,2]) |
| `test_cascade_phase3_full` | `tests/recall_integration.rs` | No Facts or Episodes, cascade reaches Phase 3 (phases_executed=[1,2,3]) |
| `test_cascade_drift_forces_phase2` | `tests/recall_integration.rs` | Query entity has drift flag, Phase 1 coverage sufficient but Phase 2 forced |
| `test_cascade_cold_start` | `tests/recall_integration.rs` | Empty graph (no facts), cascade goes to Phase 3 directly |
| `test_cascade_abstention` | `tests/recall_integration.rs` | Query about non-existent topic returns abstention=true |
| `test_cascade_builder_api` | `tests/recall_integration.rs` | RecallContextBuilder with custom limit, weights, contrastive works |
| `test_cascade_mmr_dedup` | `tests/recall_integration.rs` | Near-duplicate items across phases deduplicated correctly |
| `test_cascade_end_to_end` | `tests/recall_integration.rs` | Full pipeline: ingest messages -> NER -> observations -> consolidation -> facts -> recall returns correct facts |

### Performance Tests

| Test | What It Validates | Target |
|---|---|---|
| `bench_phase1_compact` | Phase 1 alone with 1K facts, 100 procedures | < 30ms (NF7) |
| `bench_all_phases` | Full cascade Phase 1+2+3 with 10K nodes | < 100ms (NF8) |
| `bench_ppr_10k_nodes` | PPR convergence on 10K-node graph | < 30ms |
| `bench_mmr_100_items` | MMR on 100 items with embeddings | < 5ms |
| `bench_intent_profile` | IntentProfile construction | < 100ms |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_coverage_bounded` | Coverage score is always in [0.0, 1.0] for any valid inputs |
| `proptest_mmr_no_grow` | MMR never increases the number of items |
| `proptest_rrf_monotone` | Items in more lists always score >= items in fewer lists |
| `proptest_budget_respected` | Token estimate after truncation never exceeds budget |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module-level doc comment | `recall/mod.rs` | Overview of 3-phase cascade architecture, when each phase triggers, coverage scoring explained |
| IntentProfile docs | `recall/intent.rs` | What each field means, how entity extraction works, performance expectations |
| RecallItem/Tier docs | `recall/types.rs` | Tier weights, scoring contracts, what each tier represents |
| Coverage formula docs | `recall/coverage.rs` | Full formula with component explanations, threshold rationale |
| MMR docs | `recall/mmr.rs` | Algorithm explanation, lambda/threshold parameter rationale |
| PPR docs | `recall/ppr.rs` | Algorithm explanation, damping/iteration parameters, HippoRAG citation |
| RecallContextBuilder docs | `recall/builder.rs` | Builder method docs with usage examples, default values |
| ContextBundle docs | `recall/bundle.rs` | Field explanations, how to interpret phases_executed/abstention/coverage |

---

## Review Checklist

- [ ] `recall/mod.rs` declares all submodules: intent, phase1, phase2, phase3, coverage, mmr, drift, bundle, builder, types, ppr
- [ ] `IntentProfile` struct matches spec: intent_vec, facet_vecs, entity_refs, facet_count
- [ ] Entity extraction uses local NER only (< 100ms), not LLM path
- [ ] `RecallItem` struct has node_id, node_type, score, tier, content, optional embedding
- [ ] `Tier` enum has all 5 variants: Semantic, Procedural, Episodic, KnowledgeBase, Provenance
- [ ] Tier weights match spec: 1.0, 0.9, 0.7, 0.5, 0.4
- [ ] Phase 1 searches Fact (top-20), Procedure (top-10), Topic (top-5)
- [ ] Phase 1 applies min-max normalization to [0,1]
- [ ] Phase 1 abstention check: max_score < 0.3 AND coverage < 0.2
- [ ] Phase 2 searches Episode (top-20), Observation (top-20), Message (top-10) by vector
- [ ] Phase 2 includes fulltext search on Message.content and Observation.content
- [ ] Phase 2 applies recency boost: `1.0 + 0.1 * recency_rank`
- [ ] Phase 2 RRF fusion with k=60
- [ ] Phase 2 contrastive mode retrieves failure episodes when enabled
- [ ] Phase 3 fulltext BM25 on Chunk.text (top-20) and Message.content (top-10)
- [ ] Phase 3 vector search on Chunk.embedding (top-10)
- [ ] Phase 3 graph traversal: Entity -> MENTIONS -> Chunk/Message
- [ ] Phase 3 PPR: damping=0.85, max_iter=20, top-20 results
- [ ] Phase 3 always completes (no early exit)
- [ ] Phase 3 final abstention: max_score < 0.15 AND items < 3 across ALL phases
- [ ] Coverage formula: 0.4 * facet_coverage + 0.3 * mean_score + 0.3 * diversity
- [ ] facet_coverage = semantic_items / max(semantic_items, 3)
- [ ] Phase 1 threshold: 0.75 (configurable)
- [ ] Phase 2 threshold: 0.65 (configurable)
- [ ] MMR lambda=0.7, duplicate threshold cosine > 0.85
- [ ] MMR Jaccard fallback when embeddings unavailable
- [ ] Drift override checks entity drift flags and forces Phase 2+
- [ ] Token budget default: 8192, estimate ~50 tokens per item
- [ ] ContextBundle has items, coverage, phases_executed, abstention, assembly_latency_ms
- [ ] RecallContextBuilder supports: limit, weights, recency_window, min_reliability, include_procedures, include_kb, contrastive, assemble
- [ ] phase1_only_pct metric is emitted and trackable
- [ ] Phase 1 only: < 30ms (NF7)
- [ ] All phases: < 100ms (NF8)
- [ ] Drift check: < 100ms (NF12)
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Performance benchmarks meet targets in --release

---

## Definition of Done

1. **IntentProfile construction works:** Query text is converted into IntentProfile with correct intent_vec, entity_refs, facet_vecs, and facet_count. Entity extraction uses local NER in < 100ms.
2. **Phase 1 (Compact) operational:** Vector search on Facts, Procedures, Topics returns scored RecallItems with correct tier weights and min-max normalization.
3. **Phase 2 (Expand) operational:** Vector + fulltext search on Episodes, Observations, Messages with recency boost, RRF fusion, and contrastive mode.
4. **Phase 3 (Broaden) operational:** Fulltext BM25, vector search, graph traversal, and PPR all contribute results with proper RRF fusion.
5. **Coverage scoring correct:** Formula produces expected values for known inputs. Early exit triggers at correct thresholds (0.75 Phase 1, 0.65 Phase 2).
6. **MMR deduplication works:** Near-duplicate items (cosine > 0.85) removed. Jaccard fallback functional when embeddings unavailable.
7. **Drift override works:** Queries referencing drift-flagged entities force Phase 2+ execution regardless of Phase 1 coverage.
8. **Token budget enforced:** Items truncated to fit budget (default 8192 tokens, ~50 tokens/item estimate).
9. **RecallContextBuilder API complete:** All builder methods functional. Terminal `assemble()` executes the full cascade and returns a ContextBundle.
10. **Metrics emitted:** phase1_only_pct, assembly_latency_ms, phases_executed, items_returned, coverage, abstention_count all tracked.
11. **Cold start handled:** Empty graph cascades to Phase 3 without errors.
12. **Abstention handled:** Queries with no meaningful results return `abstention: true`.
13. **Performance targets met:** Phase 1 only < 30ms (NF7), all phases < 100ms (NF8), drift check < 100ms (NF12) on --release build.
14. **End-to-end validation:** Full pipeline (ingest -> NER -> observations -> consolidation -> facts -> recall) returns correct items for known queries.
