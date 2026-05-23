# Sub-Phase 8: Recall, Stdlib Rules & Minimal Embedding

## Context

This phase closes three gaps in the Phase 1 execution plan:

1. **Basic recall API** -- no sub-doc existed. The README's Definition of Done requires "Basic recall returns relevant results from the graph given a query."
2. **Stdlib Locy rule registration** -- no sub-doc existed. The README requires "Stdlib Locy rules execute correctly within the KnowledgeBase runtime."
3. **Minimal embedding support** -- implicitly needed by P2 entity deduplication, P3 contradiction detection, and recall, but never specified. The README requires "Offline mode operates without external service dependencies."

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). Phase 1's key milestone is "Messages -> Entities -> Observations searchable." The pipeline chain is P1 (Ingest) -> P2 (NER) -> P3 (Observations) -> P4 (Consolidation). P7 (Embedding/Summary) runs alongside. Recall is the read-side counterpart of the write pipeline -- it queries the graph structures created by P1-P3 and returns ranked results.

The crate structure:
```
uniko-store    -> graph CRUD, search, Locy runtime
uniko-pipes    -> Step trait, circuit breaker, retry, DLQ, metrics
uniko-extract  -> NER, observations, chunking, ingest steps, embedding computation
uniko-memory   -> PipelineSystem, workers, recall, consolidation, rules mgmt
uniko-cortex   -> procedures, topics, MCTS, rule induction
uniko-api      -> facade
```

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Sub-phase 3: KnowledgeBase (L1) | Complete | Graph CRUD, vector/fulltext/hybrid search, Locy runtime |
| Sub-phase 4: Pipeline Infrastructure | Complete | Step trait, PipelineSystem, PipelineContext, circuit breaker |
| Sub-phase 5: Ingest Pipeline (P1) | Complete | Message, Session, Participant nodes in graph |
| Sub-phase 6: NER Pipeline (P2) | Complete | Entity nodes, MENTIONS edges, entity deduplication |
| Sub-phase 7: Observation Pipeline (P3) | Complete | Observation nodes, ABOUT/OBSERVED_IN edges, contradiction flags |
| `fastembed` crate | Available | Local embedding model (all-MiniLM-L6-v2 or similar, 384d) |

## Sub-phases

---

### 8.1 -- Minimal Embedding Support

**Objective:** Provide the minimum embedding infrastructure needed for Phase 1: auto-embed configuration for uni-db and computed entity embeddings for NER deduplication.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/embedding/mod.rs` | Rust | EmbedModel struct, embed_entity(), embed_computed() |

#### EmbedModel Struct

```rust
pub struct EmbedModel {
    model: fastembed::TextEmbedding,
}

impl EmbedModel {
    /// Load the default embedding model (all-MiniLM-L6-v2 or similar, 384d).
    /// This is a local ONNX model bundled with fastembed -- no network required.
    pub fn new() -> Result<Self>;

    /// Embed a single text string into a 384-dimensional vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of texts. More efficient than calling embed() in a loop
    /// because the model processes the batch in a single forward pass.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}
```

#### Helper Functions

```rust
/// Compute embedding for an Entity node.
/// Formula: name + " (" + entity_type + ")" or just name if type unknown.
///
/// Examples:
///   embed_entity(model, "Caroline", Some("person")) -> embeds "Caroline (person)"
///   embed_entity(model, "Paris", None) -> embeds "Paris"
pub fn embed_entity(model: &EmbedModel, name: &str, entity_type: Option<&str>) -> Result<Vec<f32>>;

/// Compute embedding for any text. Generic helper used by all computed embeddings.
/// Thin wrapper around model.embed() for API consistency.
pub fn embed_computed(model: &EmbedModel, text: &str) -> Result<Vec<f32>>;
```

#### Auto-Embed Configuration

During `register_schema()` in `uniko-store`, configure uni-db to auto-embed the following fields:

| Node Type | Source Field | Embedding Field | Notes |
|---|---|---|---|
| Message | content | embedding | Every ingested message gets an embedding |
| Chunk | text | embedding | Artifact chunks for semantic search |
| Observation | content | embedding | Observations for contradiction detection and recall |
| Summary | text | embedding | Summaries for recall (Phase 2+ creation, but field ready now) |

```rust
/// Configure uni-db auto-embed for all node types that need it.
/// Called during register_schema() in uniko-store.
///
/// Auto-embed means uni-db automatically computes and stores the embedding
/// when a node is created or the source field is updated. No explicit
/// embedding call is needed in application code.
pub fn configure_auto_embed(db: &Database, model_name: &str) -> Result<()>;
```

#### EmbedModel Initialization

The `EmbedModel` is created in `PipelineSystem::new()` (in `uniko-memory`) and distributed to consumers via:

1. **`Arc<EmbedModel>`** -- passed to step constructors that need embedding at initialization time (e.g., `EntityExtractionStep` for dedup, `ObservationExtractionStep` for contradiction detection).
2. **`PipelineContext`** -- stored in the context so steps can access it at execution time without requiring it at construction.

```rust
// In PipelineSystem::new():
let embed_model = Arc::new(EmbedModel::new()?);

// Passed to step constructors:
let ner_step = EntityExtractionStep::new(Arc::clone(&embed_model), ...);
let obs_step = ObservationExtractionStep::new(Arc::clone(&embed_model), ...);

// Also stored in PipelineContext for ad-hoc use:
ctx.embed_model = Some(Arc::clone(&embed_model));
```

#### Phase 1 Scope Note

Only auto-embed (Message, Chunk, Observation, Summary) and Entity computed embedding are needed in Phase 1. The full P7 (all computed types, artifact pooling, multimodal embeddings) ships in Phase 2.

#### Tests

| Test | What It Validates |
|---|---|
| `test_embed_model_loads` | Model loads without error, no network required |
| `test_embed_produces_vector` | `embed("hello")` returns `Vec<f32>` of expected dimensions (384) |
| `test_embed_entity_with_type` | `embed_entity(model, "Caroline", Some("person"))` produces a 384d vector |
| `test_embed_entity_without_type` | `embed_entity(model, "Caroline", None)` produces a 384d vector |
| `test_embed_batch` | Batch embedding of 10 texts produces 10 vectors of 384 dimensions each |
| `test_auto_embed_message` | Create Message node, verify embedding field is populated by uni-db |
| `test_auto_embed_observation` | Create Observation node, verify embedding field is populated by uni-db |

---

### 8.2 -- Stdlib Rule Registration

**Objective:** Register the 4 stdlib Locy rules from the spec as Rule nodes in the graph at system initialization.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/rules/mod.rs` | Rust | Module root, re-exports |
| `crates/uniko-memory/src/rules/stdlib.rs` | Rust | `register_stdlib_rules()` function, rule definitions |

#### Function

```rust
/// Register the 4 stdlib Locy rules in the graph.
/// Called during PipelineSystem::new().
/// Idempotent -- safe to call multiple times (uses deterministic rule_ids).
///
/// Rules registered:
/// 1. relevance_decay -- memory decay on episodes
/// 2. episode_pattern_detector -- count episode patterns
/// 3. sequence_detector -- detect recurring action sequences
/// 4. contradiction_detector -- find contradicting observations
pub fn register_stdlib_rules(kb: &KnowledgeBase) -> Result<()>;
```

#### Rule Node Schema

For each rule, create a Rule node with the following fields:

| Field | Value | Notes |
|---|---|---|
| `rule_id` | `"stdlib_{name}"` | Deterministic for idempotency -- upsert by rule_id |
| `name` | Rule name | Human-readable identifier |
| `source` | Full Locy source code | From spec lines 665-718 |
| `natural_language` | Human-readable description | What the rule does in plain English |
| `source_type` | `"stdlib"` | Distinguishes from learned/user rules |
| `status` | `"active"` | Direct activation, no candidate phase |
| `version` | `1` | Initial version |
| `confidence` | `1.0` | Stdlib rules start with full confidence |
| `created_at` | `now` | Timestamp of creation |

#### The 4 Stdlib Rules

**Rule 1: relevance_decay**

Natural language: "Compute decayed relevance for episodes based on age. Older episodes lose relevance exponentially. Episodes below 0.05 relevance are effectively forgotten."

```cypher
CREATE RULE relevance_decay AS
    MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id})
    WITH e,
         duration.inDays(e.timestamp, datetime()) AS age_days,
         e.importance AS base_importance
    WITH e,
         base_importance * exp(-0.05 * age_days) AS decayed
    WHERE decayed > 0.05
    YIELD KEY e, VALUE decayed AS relevance
```

**Rule 2: episode_pattern_detector**

Natural language: "Detect recurring episode patterns by counting episodes of the same action type and outcome. Patterns with at least 3 occurrences and average importance above 0.3 are surfaced for potential fact creation."

```cypher
CREATE RULE episode_pattern_detector AS
    MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id})
    FOLD n = COUNT(*)
    FOLD avg_importance = AVG(e.importance)
    WHERE n >= 3 AND avg_importance > 0.3
    YIELD KEY e.action_type, KEY e.outcome,
          VALUE n AS support,
          VALUE avg_importance AS mean_importance
```

**Rule 3: sequence_detector**

Natural language: "Detect recurring successful action sequences. When two actions consistently follow each other with successful outcomes, surface the pattern for procedural knowledge extraction."

```cypher
CREATE RULE sequence_detector AS
    MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode)
    MATCH (e1)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id})
    WHERE e1.outcome = 'success'
      AND e2.outcome = 'success'
    FOLD n = COUNT(*)
    WHERE n >= $promotion_threshold
    YIELD KEY e1.action_type, KEY e2.action_type,
          VALUE n AS success_count
```

**Rule 4: contradiction_detector**

Natural language: "Find episodes whose outcomes contradict established facts. When an episode's outcome differs from the recorded outcome pattern for that action type, flag it for fact revision."

```cypher
CREATE RULE contradiction_detector AS
    MATCH (e:Episode)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id})
    MATCH (f:Fact)
    WHERE f.subject = e.action_type
      AND f.predicate = 'outcome_pattern'
      AND btic.contains(f.valid_at, datetime())
      AND e.outcome <> f.object
    FOLD n = COUNT(e)
    WHERE n >= $contradiction_threshold
    YIELD KEY f.fact_id AS stale_fact,
          KEY e.action_type AS action,
          VALUE n AS contradicting_count,
          VALUE f.object AS old_outcome,
          VALUE e.outcome AS new_outcome
```

#### Parameter Defaults

| Parameter | Default | Source |
|---|---|---|
| `$agent_id` | Injected from context at execution time | Session/participant context |
| `$promotion_threshold` | `3` | `PipelineConfig::promotion_threshold` |
| `$contradiction_threshold` | `3` | `PipelineConfig::contradiction_threshold` |

#### Stdlib Rule Lifecycle Protection

Stdlib rules are EXEMPT from demotion, pruning, and supersession. Any lifecycle code that applies confidence decay, demotion to candidate status, or deletion must check `source_type == "stdlib"` before proceeding.

```rust
/// Check whether a rule is protected from lifecycle operations.
/// Stdlib rules cannot be demoted, pruned, or superseded.
fn is_stdlib_rule(rule: &Rule) -> bool {
    rule.source_type == "stdlib"
}

// In demotion logic:
if is_stdlib_rule(&rule) {
    // Skip demotion -- stdlib rules are permanent
    continue;
}
```

#### Tests

| Test | What It Validates |
|---|---|
| `test_register_stdlib_rules` | All 4 rules created as Rule nodes in the graph |
| `test_register_idempotent` | Calling `register_stdlib_rules()` twice creates no duplicates (4 rules, not 8) |
| `test_rule_source_type` | All 4 rules have `source_type = "stdlib"` |
| `test_rule_status_active` | All 4 rules have `status = "active"` |
| `test_relevance_decay_execution` | Insert episodes with known timestamps, execute rule, verify decayed relevance values match `importance * exp(-0.05 * age_days)` |
| `test_episode_pattern_detector_execution` | Insert 5 episodes of same action type, execute rule, verify pattern detected with `support = 5` |
| `test_sequence_detector_execution` | Insert FOLLOWED_BY chain of 3+ successful episodes, execute rule, verify sequence detected |
| `test_contradiction_detector_execution` | Insert episode whose outcome contradicts an established fact, execute rule, verify contradiction flagged |
| `test_stdlib_exempt_from_demotion` | Verify demotion logic skips rules with `source_type = "stdlib"` |

---

### 8.3 -- Basic Recall API

**Objective:** Implement a working `recall()` function that searches across the Phase 1 graph (Messages, Chunks, Observations, Entities) and returns ranked results.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/mod.rs` | Rust | `recall()` function, types, RRF fusion |
| `crates/uniko-memory/src/recall/intent.rs` | Rust | `IntentProfile` construction |

#### Types

```rust
pub struct IntentProfile {
    /// Embedding of the full query text.
    pub intent_vec: Vec<f32>,
    /// Entity names extracted from the query (rule-based NER).
    pub entity_refs: Vec<String>,
    /// Number of facets: max(entity_refs.len(), 1).
    pub facet_count: usize,
}

pub struct RecallItem {
    /// Node ID of the recalled item.
    pub node_id: NodeId,
    /// Node type label: "Message", "Chunk", "Observation", "Entity", etc.
    pub node_type: String,
    /// Fused score after RRF and tier weighting.
    pub score: f64,
    /// Display text for the recalled item.
    pub content: String,
    /// Tier classification for weighting.
    pub tier: RecallTier,
}

pub enum RecallTier {
    /// Facts, Topics (Phase 2+)
    Semantic,
    /// Procedures (Phase 3+)
    Procedural,
    /// Episodes, Observations
    Episodic,
    /// Chunks, Artifacts
    KnowledgeBase,
    /// Actions, Messages
    Provenance,
}

pub struct ContextBundle {
    /// Ranked recalled items.
    pub items: Vec<RecallItem>,
    /// Estimated total tokens across all items.
    pub total_tokens: usize,
    /// Whether Compact phase (Phase 1 of recall) was sufficient.
    /// Always false in Phase 1 (no Facts exist yet).
    pub phase1_only: bool,
    /// Coverage score (0.0-1.0).
    pub coverage: f64,
}

pub struct RecallConfig {
    /// Maximum number of items to return. Default: 15.
    pub limit: usize,
    /// Maximum total tokens across all returned items. Default: 8192.
    pub token_budget: usize,
    /// Minimum fused score for inclusion. Default: 0.1.
    pub min_score: f64,
}
```

#### Main Function

```rust
/// Recall relevant context from the memory graph.
///
/// Phase 1 implementation: executes Phase 3 (Broaden) only.
/// At cold start (no Facts, no Procedures), this is the expected behavior
/// per spec line 559. Phase 1 (Compact) and Phase 2 (Expand) activate
/// in execution Phase 2 when consolidation creates Facts.
///
/// Search strategy:
/// 1. Construct IntentProfile (embed query, extract entities)
/// 2. Fulltext search: Message.content, Chunk.text, Observation.content
/// 3. Vector search: Message.embedding, Chunk.embedding,
///    Observation.embedding, Entity.embedding
/// 4. Graph traversal: Entity -> MENTIONS -> Message/Chunk
/// 5. Fuse via RRF with tier weights
/// 6. Truncate to token budget
pub async fn recall(
    kb: &KnowledgeBase,
    embed_model: &EmbedModel,
    query: &str,
    config: RecallConfig,
) -> Result<ContextBundle>;
```

#### IntentProfile Construction

```rust
/// Build an IntentProfile from a query string.
///
/// Steps:
/// 1. Embed the full query -> intent_vec (via EmbedModel)
/// 2. Extract entities via rule-based NER (same as P2, < 100ms)
/// 3. Count entity_refs -> facet_count = max(entity_refs.len(), 1)
pub fn build_intent(
    embed_model: &EmbedModel,
    query: &str,
) -> Result<IntentProfile>;
```

#### Phase 3 (Broaden) Search -- The Only Phase Active in Phase 1

At cold start (no Facts, no Procedures), the recall system skips Compact (Phase 1) and Expand (Phase 2) and executes only Broaden (Phase 3). This is correct behavior per spec line 559.

**Step 1: Fulltext BM25 search**

| Source | Field | Top-K |
|---|---|---|
| Message | content | 20 |
| Chunk | text | 20 |
| Observation | content | 10 |

**Step 2: Vector search**

| Source | Field | Top-K |
|---|---|---|
| Message | embedding | 10 |
| Chunk | embedding | 10 |
| Observation | embedding | 10 |
| Entity | embedding | 10 |

**Step 3: Graph traversal (if entity_refs non-empty)**

For each entity name in `entity_refs`:
1. Lookup Entity node by name (Hash index)
2. Traverse MENTIONS edges to collect neighbor Messages and Chunks
3. Add traversal results to the candidate set

**Step 4: Score normalization**

Per-source min-max normalization to [0, 1]. Each search source (fulltext-Message, fulltext-Chunk, vector-Message, etc.) is normalized independently so that scores are comparable across sources.

**Step 5: RRF fusion**

For each candidate item appearing in one or more ranked lists:
```
rrf_score = sum(1 / (60 + rank_i))
```
Where `rank_i` is the rank of the item in list `i`. Items appearing in more lists accumulate higher scores. The constant 60 is the standard RRF dampening factor.

**Step 6: Tier weighting**

| Tier | Weight | Active Node Types (Phase 1) |
|---|---|---|
| Semantic | 0.9 | (none -- no Facts yet) |
| Procedural | 0.8 | (none -- no Procedures yet) |
| Episodic | 0.7 | Observations |
| KnowledgeBase | 0.5 | Chunks |
| Provenance | 0.4 | Messages |

```
final_score = rrf_score * tier_weight
```

**Step 7: Sort and filter**

Sort by `final_score` descending. Filter out items below `config.min_score`.

**Step 8: Token budget truncation**

Estimate ~50 tokens per item. Accumulate items until `total_tokens` exceeds `config.token_budget`, then stop. Set `total_tokens` to the accumulated count.

#### Coverage Scoring (Simplified for Phase 1)

```
semantic_items = 0          // no Facts yet
facet_coverage = 0          // no semantic facets to cover
mean_score = mean(item.score for item in items)
diversity = distinct_tier_count / 5

coverage = 0.4 * facet_coverage + 0.3 * mean_score + 0.3 * diversity
```

Coverage is always < 0.75 at cold start, which means a multi-phase recall system would cascade past Phase 1 (Compact). This is correct behavior -- at cold start, Broaden is always needed.

#### phase1_only_pct Tracking

- Always 0% in Phase 1 (no Facts, so the Compact phase is always empty)
- Metric: `uniko.recall.phase1_only_pct` gauge = 0.0
- This becomes meaningful in execution Phase 2 when Facts exist and some queries can be answered purely from consolidated knowledge

#### Tests

| Test | What It Validates |
|---|---|
| `test_recall_empty_graph` | Empty graph returns empty ContextBundle with 0 items, 0 tokens |
| `test_recall_finds_message` | Ingest a message, recall finds it via fulltext search |
| `test_recall_finds_entity` | Ingest message with entity, recall finds the entity via vector search |
| `test_recall_finds_observation` | Ingest message, P2+P3 run, recall finds the resulting observation |
| `test_recall_rrf_fusion` | Item appearing in both vector and fulltext result lists ranks higher than items in only one |
| `test_recall_tier_weighting` | Observation (Episodic, weight 0.7) outranks Message (Provenance, weight 0.4) at equivalent RRF scores |
| `test_recall_token_budget` | Large result set truncated to fit within token budget |
| `test_recall_graph_traversal` | Entity MENTIONS edges improve recall by pulling in connected Messages/Chunks |
| `test_recall_intent_profile` | Query containing entity name correctly extracts entity_refs |
| `test_recall_coverage_cold_start` | Coverage score < 0.75, `phase1_only = false` |
| `test_recall_offline` | Recall works without LLM -- embedding model is local, no network calls |

---

### 8.4 -- Offline End-to-End Integration Test

**Objective:** Validate the complete Phase 1 pipeline chain works without external dependencies (no ONNX NER model, no LLM, no network). This is the capstone test for Phase 1.

#### File

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/tests/offline_e2e.rs` | Rust | End-to-end integration test |

#### Test

```rust
#[tokio::test]
async fn test_offline_e2e() {
    // 1. Initialize system: no ONNX, no LLM
    //    - KnowledgeBase (in-memory)
    //    - EmbedModel (fastembed, local)
    //    - PipelineSystem with IngestWorker
    //    - Stdlib rules registered

    // 2. Create Participant "Caroline" and "Melanie"

    // 3. Ingest 10 messages from a sample conversation
    //    - Session auto-created
    //    - Messages get SENT_BY, IN_SESSION, NEXT edges
    //    - P2 (rule-based NER only) extracts entities
    //    - P3 (rule-based only) extracts observations

    // 4. Verify graph state
    //    - Entity nodes exist (at least "Caroline", "Melanie")
    //    - MENTIONS edges connect Messages to Entities
    //    - Observation nodes exist
    //    - OBSERVED_IN edges connect Observations to Messages
    //    - ABOUT edges connect Observations to Entities

    // 5. Run recall("What did Caroline do?")
    //    - Verify non-empty ContextBundle
    //    - Verify items include Messages and/or Observations about Caroline

    // 6. Verify system health
    //    - Zero dead-letter items
    //    - All pipeline steps completed
    //    - No errors in pipeline health

    // 7. Verify stdlib rules registered
    //    - 4 Rule nodes with source_type "stdlib" and status "active"
}
```

#### Sample Messages (LoCoMo-style conversation)

```
Caroline: "I went to an LGBTQ support group yesterday. It was really helpful."
Melanie: "That's great! I've been painting a lot lately. I finished a sunrise painting."
Caroline: "I'm also looking into adoption agencies. It's been a dream of mine."
Melanie: "How exciting! I ran a charity race last weekend."
Caroline: "I started a new job at the social work agency downtown."
Melanie: "I prefer oil paints over watercolors for landscapes."
Caroline: "The adoption process requires a home study first."
Melanie: "I've been training for another race in March."
Caroline: "Work has been busy but fulfilling. I love helping families."
Melanie: "My painting of the sunset sold at the gallery!"
```

#### Verification Steps

| Step | Expected Outcome |
|---|---|
| System initialization | KnowledgeBase, EmbedModel, PipelineSystem all initialized without network calls |
| Participant creation | 2 Participant nodes ("Caroline", "Melanie") exist in graph |
| Message ingest | 10 Message nodes with SENT_BY, IN_SESSION, NEXT edges |
| P2 entity extraction | Entity nodes for at least "Caroline", "Melanie" (rule-based NER) |
| P2 edge wiring | MENTIONS edges from Messages to Entities |
| P3 observation extraction | Observation nodes for factual statements (e.g., "Caroline went to an LGBTQ support group") |
| P3 edge wiring | OBSERVED_IN (Observation -> Message), ABOUT (Observation -> Entity) |
| Recall | `recall("What did Caroline do?")` returns non-empty ContextBundle containing Caroline-related items |
| System health | Zero DLQ items, all steps completed, no errors |
| Stdlib rules | 4 Rule nodes with `source_type = "stdlib"` and `status = "active"` |

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_embed_model_loads` | `embedding/mod.rs` | Model loads without error |
| `test_embed_produces_vector` | `embedding/mod.rs` | embed("hello") returns 384d Vec<f32> |
| `test_embed_entity_with_type` | `embedding/mod.rs` | "Caroline (person)" embeds correctly |
| `test_embed_entity_without_type` | `embedding/mod.rs` | "Caroline" embeds correctly |
| `test_embed_batch` | `embedding/mod.rs` | Batch of 10 texts produces 10 x 384d vectors |
| `test_register_stdlib_rules` | `rules/stdlib.rs` | All 4 rules created as Rule nodes |
| `test_register_idempotent` | `rules/stdlib.rs` | Calling twice creates no duplicates |
| `test_rule_source_type` | `rules/stdlib.rs` | All have source_type "stdlib" |
| `test_rule_status_active` | `rules/stdlib.rs` | All have status "active" |
| `test_recall_empty_graph` | `recall/mod.rs` | Empty graph returns empty ContextBundle |
| `test_recall_intent_profile` | `recall/intent.rs` | Query with entity name extracts entity_refs |
| `test_recall_rrf_fusion` | `recall/mod.rs` | Multi-source items rank higher |
| `test_recall_tier_weighting` | `recall/mod.rs` | Episodic outranks Provenance at same RRF score |
| `test_recall_token_budget` | `recall/mod.rs` | Results truncated to budget |
| `test_recall_coverage_cold_start` | `recall/mod.rs` | Coverage < 0.75, phase1_only = false |
| `test_stdlib_exempt_from_demotion` | `rules/stdlib.rs` | Demotion logic skips stdlib rules |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_auto_embed_message` | `embedding/mod.rs` | Message node gets embedding populated by uni-db |
| `test_auto_embed_observation` | `embedding/mod.rs` | Observation node gets embedding populated by uni-db |
| `test_relevance_decay_execution` | `rules/stdlib.rs` | Insert episodes, execute rule, verify decay math |
| `test_episode_pattern_detector_execution` | `rules/stdlib.rs` | 5 same-type episodes detected as pattern |
| `test_sequence_detector_execution` | `rules/stdlib.rs` | FOLLOWED_BY chain detected |
| `test_contradiction_detector_execution` | `rules/stdlib.rs` | Contradicting episode vs fact flagged |
| `test_recall_finds_message` | `recall/mod.rs` | Ingest message, recall finds it via fulltext |
| `test_recall_finds_entity` | `recall/mod.rs` | Ingest message with entity, recall finds entity |
| `test_recall_finds_observation` | `recall/mod.rs` | P2+P3 run, recall finds observation |
| `test_recall_graph_traversal` | `recall/mod.rs` | Entity MENTIONS edges improve recall |
| `test_recall_offline` | `recall/mod.rs` | Works without LLM (local embedding model) |
| `test_offline_e2e` | `tests/offline_e2e.rs` | Full Phase 1 pipeline without external dependencies |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module-level doc comment | `embedding/mod.rs` | EmbedModel overview, auto-embed configuration, Phase 1 vs Phase 2 scope |
| `EmbedModel` doc comment | `embedding/mod.rs` | Thread safety (Arc-safe), model dimensions, offline operation |
| Module-level doc comment | `rules/mod.rs` | Rule management overview, stdlib vs learned rules |
| `register_stdlib_rules` doc comment | `rules/stdlib.rs` | Idempotency guarantee, rule listing, lifecycle protection |
| Module-level doc comment | `recall/mod.rs` | Recall architecture, Phase 1/2/3 search phases, RRF fusion |
| `recall()` doc comment | `recall/mod.rs` | Search strategy, token budget, coverage scoring |
| `IntentProfile` doc comment | `recall/intent.rs` | Construction steps, entity extraction reuse |

---

## Review Checklist

- [ ] `embedding/mod.rs` exists and `EmbedModel::new()` loads the local model without network calls
- [ ] `embed()` produces 384-dimensional vectors
- [ ] `embed_batch()` processes multiple texts efficiently
- [ ] `embed_entity()` computes correct text formula: "name (type)" or "name"
- [ ] `configure_auto_embed()` configures Message, Chunk, Observation, Summary auto-embed
- [ ] Auto-embed triggers on node creation (verified by integration tests)
- [ ] `EmbedModel` is `Arc`-shared and stored in `PipelineContext`
- [ ] `rules/stdlib.rs` exists and `register_stdlib_rules()` creates all 4 Rule nodes
- [ ] Each Rule has correct: rule_id, name, source (full Locy), natural_language, source_type, status, version, confidence
- [ ] Registration is idempotent -- calling twice creates no duplicates
- [ ] Stdlib rules have `source_type = "stdlib"` and `status = "active"`
- [ ] Stdlib rules are exempt from demotion/pruning/supersession
- [ ] `recall/mod.rs` exists and `recall()` returns a `ContextBundle`
- [ ] IntentProfile constructed: embed query, extract entities, count facets
- [ ] Fulltext BM25 search runs on Message.content, Chunk.text, Observation.content
- [ ] Vector search runs on Message.embedding, Chunk.embedding, Observation.embedding, Entity.embedding
- [ ] Graph traversal follows Entity -> MENTIONS -> Message/Chunk when entity_refs present
- [ ] Per-source min-max normalization applied
- [ ] RRF fusion: `score = sum(1 / (60 + rank_i))`
- [ ] Tier weights applied: Semantic=0.9, Procedural=0.8, Episodic=0.7, KB=0.5, Provenance=0.4
- [ ] Results sorted by final score descending
- [ ] Token budget enforced (estimated ~50 tokens per item)
- [ ] Coverage score computed correctly, < 0.75 at cold start
- [ ] `phase1_only = false` always in Phase 1
- [ ] `phase1_only_pct` metric emitted as gauge = 0.0
- [ ] Offline e2e test passes: full pipeline chain without ONNX, LLM, or network
- [ ] All unit tests pass
- [ ] All integration tests pass

---

## Definition of Done

1. **EmbedModel functional:** Loads local model, embeds text, produces 384-dimensional vectors. No network required.
2. **Auto-embed configured:** Message, Chunk, Observation, Summary auto-embed on insert via uni-db configuration.
3. **Entity embedding works:** `embed_entity()` produces vectors usable for dedup (name + type formula).
4. **Stdlib rules registered:** 4 Rule nodes exist with correct Locy source, `status = "active"`, `source_type = "stdlib"`.
5. **Stdlib rules execute:** Each rule runs against sample data and produces expected output (decay values, pattern counts, sequence detection, contradiction flagging).
6. **Stdlib rules protected:** Demotion/pruning/supersession logic skips rules with `source_type = "stdlib"`.
7. **Basic recall works:** `recall(query)` returns relevant items from ingested messages, chunks, observations, and entities.
8. **RRF fusion correct:** Items appearing in multiple search sources rank higher than items in a single source.
9. **Token budget enforced:** Results truncated to specified budget (~50 tokens/item estimate).
10. **Offline e2e passes:** Full pipeline chain (ingest -> NER -> observations -> recall) works without ONNX NER model, LLM, or network. fastembed runs locally.
11. **All tests pass:** `cargo nextest run -p uniko-extract --lib embedding` and `cargo nextest run -p uniko-memory --lib recall` and `cargo nextest run -p uniko-memory --lib rules` all green.
