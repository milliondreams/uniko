# Sub-Phase 7: Observation Extraction Pipeline (P3)

## Context

This phase implements Pipeline 3 — observation extraction. Observations are the bridge between raw communication (Messages, Chunks) and consolidated knowledge (Facts). P3 takes text that has already been ingested (P1) and entity-tagged (P2), identifies factual statements within it, creates Observation nodes, wires all required edges, flags potential contradictions for P4, and notifies the consolidation worker when enough observations accumulate.

The pipeline runs asynchronously with a latency target of < 5s per message/chunk (NF6). It operates in two modes: rule-based (always available, ~40% recall) and LLM-enhanced (optional, ~80% recall). The system must function fully without LLM — degraded observation quality is acceptable, non-functional observation extraction is not.

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). The pipeline chain is P1 (Ingest) -> P2 (NER) -> P3 (Observations) -> P4 (Consolidation). P7 (Embedding/Summary) runs alongside. Consolidation derives Facts from Observations using BTIC temporal intervals for validity tracking.

**Key principle:** Observations are direct statements or perceived facts from messages — not derived knowledge. "Caroline attended an LGBTQ support group" is an observation. "Caroline is pursuing adoption" is a Fact derived later by P4 from multiple observations. P3 must never create Facts.

## Prerequisites

- **Sub-phase 6 (NER P2) complete** — P3 runs after P2 extracts entities. Observations reference entities via ABOUT edges.
- **Sub-phase 2 (Schema) complete** — Observation, Entity, Message, Chunk, Episode node types defined in `uniko-store` schema.
- **Sub-phase 3 (KnowledgeBase) complete** — graph storage, search, and CRUD operations available via KnowledgeBase API in `uniko-store`.
- **Sub-phase 4 (Pipeline infrastructure) complete** — `Step` trait from `uniko-pipes`, error policies, circuit breaker for LLM.
- **Sub-phase 8 (Embedding) complete** — auto-embed configured so Observation.embedding auto-embeds from content field on node creation.

## Sub-phases

---

### 7.1 — Content Filtering

**Objective:** Filter out non-informative content before observation extraction, avoiding wasted processing and noise in the knowledge graph.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/observations/filter.rs` | Rust | Content filtering logic |
| `crates/uniko-extract/src/observations/filter_tests.rs` | Rust | Unit tests for filtering |

#### Functions

```rust
/// Determines whether a text block contains potentially informative content
/// worth extracting observations from. Returns false for greetings, reactions,
/// pure questions, system messages, and very short content.
pub fn is_informative(text: &str) -> bool;

/// Checks if text is a greeting or social filler.
/// Matches: "Hey!", "Hi there", "Wow!", "Thanks!", "lol", "haha",
/// "Sure thing", "No worries", "You're welcome", "Sounds good"
fn is_greeting_or_filler(text: &str) -> bool;

/// Checks if text is a pure question with no embedded facts.
/// A pure question ends with '?' and contains no declarative clauses.
/// "Where is the store?" → true (pure question)
/// "I went to Paris, did you?" → false (contains embedded fact)
fn is_pure_question(text: &str) -> bool;

/// Checks if content_type indicates a system/error message.
/// Matches content_type: "system", "error", "tool_result" (when no facts)
fn is_system_message(content_type: Option<&str>) -> bool;

/// Checks if text is too short to contain a meaningful observation.
/// Threshold: < 5 words after stripping punctuation and whitespace.
fn is_too_short(text: &str) -> bool;
```

#### Filter Rules

| Category | Examples | Action |
|---|---|---|
| Greetings/filler | "Hey!", "Wow!", "Thanks!", "lol", "Sure thing", "No worries" | Skip |
| Pure questions | "Where is the store?", "What time is it?" | Skip |
| Questions with facts | "Did you know Caroline went to Paris?" | Keep |
| System messages | content_type = "system" or "error" | Skip |
| Very short | < 5 words after normalization | Skip |
| Tool results | content_type = "tool_result" with no entity mentions | Skip |
| Everything else | Declarative sentences, descriptions, narratives | Keep |

#### Implementation Notes

- Greeting detection uses a static hashset of normalized lowercased patterns, not regex (faster, deterministic).
- Pure question detection: split into clauses by commas/semicolons. If all clauses end with `?` or are subordinate to a question, it's pure. If any clause is declarative ("I went to Paris"), the text is informative.
- Word count uses `split_whitespace().count()` after stripping leading/trailing punctuation.
- The filter runs before any LLM or embedding calls — it must be fast (< 1ms).

---

### 7.2 — Rule-Based Observation Extraction

**Objective:** Extract observations using deterministic rules — no LLM required. This is the primary extraction path and must always be available (F72 offline mode).

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/observations/rules.rs` | Rust | Rule-based extraction logic |
| `crates/uniko-extract/src/observations/temporal.rs` | Rust | Temporal expression resolution |

#### Core Struct

```rust
/// A raw observation extracted from text, before node creation.
pub struct RawObservation {
    /// The full self-contained observation statement.
    /// Must be understandable without surrounding context.
    /// Not a fragment — "attended LGBTQ group" is wrong,
    /// "Caroline attended an LGBTQ support group" is correct.
    pub content: String,

    /// The primary subject entity name (matches Entity.name from P2).
    pub subject: String,

    /// When the observation was true in the real world.
    /// Computed from message timestamp + temporal context clues.
    pub observed_at: DateTime<Utc>,

    /// Extraction confidence (0.0-1.0).
    /// Rule-based: 0.5-0.7 depending on pattern strength.
    /// LLM-enhanced: 0.7-0.9 depending on model confidence.
    pub confidence: f64,
}
```

#### Functions

```rust
/// Extract observations from text using rule-based patterns.
/// Requires entities extracted by P2 to anchor observations to known subjects.
///
/// Steps:
///   1. Sentence-split the text
///   2. For each sentence:
///      a. Check if it contains a named entity from the entity list
///      b. Check for SVO (subject-verb-object) structure
///      c. Check for preference patterns
///      d. Check for fact patterns
///   3. For each candidate, construct a self-contained observation
///   4. Compute observed_at from message timestamp + context clues
pub fn extract_observations_rule_based(
    text: &str,
    entities: &[Entity],
    timestamp: DateTime<Utc>,
) -> Vec<RawObservation>;

/// Split text into sentences. Handles:
/// - Period-space boundaries (with abbreviation awareness: Mr., Dr., etc.)
/// - Newline boundaries
/// - Semicolons as sentence separators
/// Does NOT split on commas (clause boundary, not sentence).
fn sentence_split(text: &str) -> Vec<&str>;

/// Check if a sentence has SVO structure (contains subject + verb + object).
/// Uses simple heuristic: at least 3 words, contains a verb-like word
/// (not in stop-verb list: is, am, are, was, were, be, been, being
/// are valid when followed by a complement).
fn has_svo_structure(sentence: &str) -> bool;

/// Check for preference patterns.
/// Matches: "I prefer X", "I like X", "I don't like X", "I love X",
/// "I enjoy X", "X is my favorite", "I'd rather X", "I'm interested in X"
fn is_preference_pattern(sentence: &str) -> bool;

/// Check for factual patterns.
/// Matches: "X is Y", "X has Y", "X went to Y", "X works at Y",
/// "X lives in Y", "X attended Y", "X started Y", "X completed Y"
fn is_fact_pattern(sentence: &str) -> bool;

/// Extract the subject from a sentence given known entities.
/// Priority:
///   1. Named entity that appears in the sentence
///   2. Pronoun resolved to most recent entity of matching type
///   3. Grammatical subject (first noun phrase before main verb)
fn extract_subject(sentence: &str, entities: &[Entity]) -> Option<String>;

/// Construct a self-contained observation from a sentence.
/// If the sentence uses pronouns, resolve them using entities.
/// "She went to Paris" + entity "Caroline" → "Caroline went to Paris"
fn make_self_contained(sentence: &str, subject: &str, entities: &[Entity]) -> String;
```

#### Temporal Resolution

```rust
/// Resolve temporal expressions relative to a reference timestamp.
///
/// "yesterday" → timestamp - 1 day
/// "last week" → timestamp - 7 days
/// "last month" → timestamp - 30 days (approx)
/// "last year" → timestamp.year() - 1, same month/day
/// "in March" → nearest March (past if current month > March, else current year)
/// "in 2022" → 2022-01-01 (year granularity)
/// "on Tuesday" → most recent Tuesday before timestamp
/// "two days ago" → timestamp - 2 days
/// "a few months ago" → timestamp - 3 months (approximate)
///
/// Returns the resolved timestamp. If no temporal expression found,
/// returns the original message timestamp.
pub fn resolve_temporal(text: &str, reference: DateTime<Utc>) -> DateTime<Utc>;

/// Extract temporal expressions from text.
/// Returns (expression, position) pairs.
fn find_temporal_expressions(text: &str) -> Vec<(String, usize)>;
```

#### Extraction Pipeline (within `extract_observations_rule_based`)

```
Input: "I went to a LGBTQ support group yesterday. It was really helpful."
Entities: [Entity { name: "LGBTQ support group", entity_type: "organization" }]
Timestamp: 2023-06-15T10:00:00Z

Step 1: Sentence split
  → ["I went to a LGBTQ support group yesterday.", "It was really helpful."]

Step 2a: Sentence 1 contains entity "LGBTQ support group" → candidate
Step 2b: SVO: subject="I", verb="went to", object="LGBTQ support group" → yes
Step 2d: Fact pattern: "X went to Y" → yes

Step 3: Construct observation
  → Subject: sender name (from SENT_BY) or "I" resolved to participant
  → Content: "Caroline went to an LGBTQ support group"
  → Confidence: 0.7 (entity match + SVO + fact pattern)

Step 4: Temporal resolution
  → "yesterday" → 2023-06-14T10:00:00Z

Step 2a: Sentence 2 "It was really helpful" — no named entity → skip
  (pronouns referencing non-entity subjects are not extracted)

Output: [RawObservation {
    content: "Caroline went to an LGBTQ support group",
    subject: "Caroline",
    observed_at: 2023-06-14T10:00:00Z,
    confidence: 0.7,
}]
```

---

### 7.3 — LLM-Enhanced Extraction (Async, Optional)

**Objective:** Use LLM to extract observations that rule-based patterns miss — implicit statements, complex sentence structures, paraphrases. This path is optional and gated by the circuit breaker.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/observations/llm.rs` | Rust | LLM-enhanced extraction |

#### Functions

```rust
/// Extract observations using LLM. Runs only when the LLM circuit breaker
/// is closed. Returns observations that supplement rule-based extraction.
///
/// Gated by: `#[cfg(feature = "llm")]` and circuit breaker state.
/// error_policy: Skip (LLM failure never blocks the pipeline).
pub async fn extract_observations_llm(
    text: &str,
    entities: &[Entity],
    provider: &LlmProvider,
) -> Result<Vec<RawObservation>>;

/// Merge LLM-extracted observations with rule-based observations.
/// Deduplication: if cosine similarity between two observation embeddings
/// is > 0.9, keep the one with higher confidence.
/// LLM observations that don't reference any P2 entity are discarded
/// (prevents hallucinated entities).
pub async fn merge_observations(
    rule_based: Vec<RawObservation>,
    llm_extracted: Vec<RawObservation>,
    embed_model: &EmbedModel,
) -> Vec<RawObservation>;
```

#### LLM Prompt

```
Extract factual observations from this message. For each observation, provide:
- content: a self-contained factual statement (not a question or greeting)
- subject: the primary entity this observation is about
- observed_at: when this was true (use ISO 8601, or "now" if present tense)

Rules:
- Skip greetings, questions, reactions, and filler
- Each observation must be understandable without the original message
- The subject must be a specific named entity, not a pronoun
- Include temporal context clues in observed_at, not in content

Known entities in this message: {entity_names}
Message timestamp: {timestamp}

Message:
{text}

Return JSON array:
[{"content": "...", "subject": "...", "observed_at": "..."}]
```

#### Validation Rules

| Rule | Action |
|---|---|
| Observation subject not in P2 entity list | Discard (no hallucinated entities) |
| Observation content is a question | Discard |
| Observation content < 3 words | Discard |
| observed_at is unparseable | Fall back to message timestamp |
| Duplicate of rule-based observation (cosine > 0.9) | Keep higher confidence |

#### Error Handling

- LLM timeout (> 5s): skip LLM path, use rule-based only
- LLM response not valid JSON: skip, log warning
- Circuit breaker open: skip entirely, no retry
- Partial JSON (some entries valid, some not): keep valid entries

---

### 7.4 — Observation Node Creation & Edge Wiring

**Objective:** Create Observation nodes in the graph and wire all required edges. This is the core step that materializes extracted observations into the knowledge graph.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/observations/mod.rs` | Rust | ObservationExtractionStep + orchestration |

#### Step Implementation

```rust
/// The main pipeline step for observation extraction.
/// Implements the Step trait from the pipeline management system.
pub struct ObservationExtractionStep {
    embed_model: Arc<EmbedModel>,
    llm_provider: Option<Arc<LlmProvider>>,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl Step for ObservationExtractionStep {
    fn name(&self) -> &str { "observation_extraction" }

    fn error_policy(&self) -> StepErrorPolicy {
        StepErrorPolicy::Skip // individual observation failures don't block the pipeline
    }

    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepOutcome> {
        // 1. Get source text and entities from context
        // 2. Filter (7.1)
        // 3. Rule-based extraction (7.2)
        // 4. LLM extraction if available (7.3)
        // 5. Merge results
        // 6. Create nodes and wire edges (this sub-phase)
        // 7. Check contradictions (7.5)
        // 8. Notify consolidation (7.6)
    }
}
```

#### Node Creation

For each `RawObservation` after filtering and merging:

```rust
/// Create an Observation node in the graph.
///
/// Fields:
///   observation_id: new_id() (UUID v7)
///   content: raw_obs.content
///   subject: raw_obs.subject
///   observed_at: raw_obs.observed_at
///   confidence: raw_obs.confidence
///   embedding: auto-embedded by uni-db from content field (P7a)
async fn create_observation_node(
    kb: &KnowledgeBase,
    raw_obs: &RawObservation,
) -> Result<NodeId>;
```

#### Edge Wiring

| Edge | Direction | Target | Condition | Properties |
|---|---|---|---|---|
| OBSERVED_IN | Observation -> Message | Source is a Message | — |
| OBSERVED_IN | Observation -> Chunk | Source is an Artifact Chunk | — |
| ABOUT | Observation -> Entity | For each P2 entity mentioned in subject or content | — |
| OBSERVED_DURING | Observation -> Episode | Most recent Episode where: RECORDED_BY same participant AND IN_SESSION same session AND episode.timestamp within 5 min of message.timestamp | — |

```rust
/// Wire all edges from an Observation to its related nodes.
///
/// OBSERVED_IN: always created — links to source Message or Chunk.
/// ABOUT: created for each entity referenced by subject or content match.
/// OBSERVED_DURING: created only if a qualifying Episode exists
///   (same participant, same session, within 5 minutes).
async fn wire_observation_edges(
    kb: &KnowledgeBase,
    obs_node_id: NodeId,
    source_node_id: NodeId,   // Message or Chunk
    source_type: SourceType,  // Message or Chunk enum
    entities: &[Entity],
    raw_obs: &RawObservation,
    session_id: &str,
    participant_id: &str,
    message_timestamp: DateTime<Utc>,
) -> Result<()>;
```

#### OBSERVED_DURING Logic

```
Query: Find the most recent Episode where:
  1. RECORDED_BY → same Participant (participant_id match)
  2. IN_SESSION → same Session (session_id match)
  3. episode.timestamp >= message.timestamp - 5 minutes
  4. episode.timestamp <= message.timestamp + 5 minutes

If found: create OBSERVED_DURING edge from Observation to Episode
If not found: no OBSERVED_DURING edge (observation linked only via OBSERVED_IN → Message)
```

#### Auto-Embed Integration

Observation.embedding is auto-embedded by uni-db from the `content` field (P7a configuration). No explicit embedding call is needed in this step — uni-db handles it on node creation. Verify that the auto-embed configuration from Phase 8.1 is active.

---

### 7.5 — Contradiction Flagging

**Objective:** Perform a lightweight inline check for potential contradictions between new observations and existing ones. Flag contradictions for P4 (consolidation) — do NOT invalidate anything here.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/observations/contradiction.rs` | Rust | Contradiction detection |

#### Types

```rust
/// A potential contradiction between two observations about the same subject.
/// Flagged for Pipeline 4 attention — not resolved here.
pub struct ContradictionFlag {
    /// The new observation that may contradict existing knowledge.
    pub new_observation_id: NodeId,
    /// The existing observation that may be contradicted.
    pub existing_observation_id: NodeId,
    /// The subject entity they share.
    pub subject: String,
    /// Cosine similarity between the two observation embeddings.
    /// Low similarity (< 0.3) with same subject = likely contradiction.
    pub similarity: f64,
}
```

#### Functions

```rust
/// Check a new observation against existing observations with the same subject.
///
/// Algorithm:
///   1. Query existing observations: WHERE subject = new_obs.subject
///   2. For each existing observation:
///      a. Compute cosine similarity between embeddings
///      b. If similarity < 0.3 AND same subject → flag as contradiction
///   3. Return all flags (may be empty)
///
/// This is a lightweight check — it doesn't consider temporal context,
/// BTIC intervals, or fact confidence. P4 does the real analysis.
pub fn check_contradictions(
    kb: &KnowledgeBase,
    new_obs: &Observation,
) -> Vec<ContradictionFlag>;
```

#### Contradiction Criteria

| Condition | Interpretation | Action |
|---|---|---|
| Same subject, cosine similarity < 0.3 | Semantically divergent statements about the same entity | Flag for P4 |
| Same subject, cosine similarity 0.3-0.7 | Possibly related but different aspects | No flag |
| Same subject, cosine similarity > 0.7 | Reinforcing or restating | No flag |
| Different subjects | Unrelated | No flag |

#### Storage of Flags

Contradiction flags are not stored as graph nodes — they are passed to the consolidation notification (7.6) as part of the `ObservationsReady` message. P4 re-analyzes all observations during consolidation and makes the real determination.

---

### 7.6 — Artifact Chunk Observation Path & Consolidation Notification

**Objective:** Handle observations extracted from Artifact chunks (not Messages) and notify the consolidation worker when observations are ready for processing.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/observations/mod.rs` | Rust | Chunk path + notification (same file as 7.4) |

#### Artifact Chunk Path

When an Artifact is ingested, the complete pipeline path is:

```
Artifact → P1 chunking → Chunk nodes created
  → each Chunk → P2 NER → Entity + MENTIONS edges
  → each Chunk → P3 → Observation nodes
  → Observations linked via OBSERVED_IN → Chunk (not Message)
  → Observations linked via ABOUT → Entity
```

Key differences from the Message path:

| Aspect | Message Path | Artifact Chunk Path |
|---|---|---|
| OBSERVED_IN target | Message node | Chunk node |
| OBSERVED_DURING | May link to Episode | Typically no Episode (artifacts are ingested, not conversational) |
| Temporal resolution | Message timestamp + context clues | Chunk has no timestamp; use Artifact.created_at |
| Subject extraction | Sender participant + entities | Entities only (no sender context) |

#### Consolidation Notification

```rust
/// Notification sent to the consolidation worker when observations
/// from a processing batch are ready.
pub struct ObservationsReady {
    /// The agent whose knowledge base was updated.
    pub agent_id: String,
    /// Number of new observations created in this batch.
    pub observation_count: u32,
    /// Node IDs of the source Messages/Chunks that were processed.
    pub source_node_ids: Vec<NodeId>,
    /// Any contradiction flags detected during extraction.
    pub contradiction_flags: Vec<ContradictionFlag>,
}
```

#### Consolidation Trigger Logic

The consolidation worker maintains per-agent observation counts and triggers consolidation (P4) based on:

| Trigger | Threshold | Source |
|---|---|---|
| Observation count | >= 20 new observations since last cycle | `UnikoConfig::consolidation_threshold` |
| Timer | 15 minutes since last cycle | `UnikoConfig::consolidation_interval_secs` |

```rust
/// Send notification to the consolidation worker.
/// The worker aggregates counts per agent and triggers P4
/// when the threshold or timer is reached.
async fn notify_consolidation(
    consolidation_tx: &mpsc::Sender<ConsolidationTask>,
    notification: ObservationsReady,
) -> Result<()>;
```

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_filter_greetings` | `observations/filter_tests.rs` | "Hey!", "Thanks!", "lol", "Wow!" all return `is_informative = false` |
| `test_filter_pure_questions` | `observations/filter_tests.rs` | "Where is the store?" → false; "Did you know Caroline went to Paris?" → true |
| `test_filter_short_content` | `observations/filter_tests.rs` | "OK sure" (< 5 words) → false; "Caroline went to Paris yesterday" → true |
| `test_filter_system_messages` | `observations/filter_tests.rs` | content_type = "system" → false; content_type = "text" → true |
| `test_filter_informative_kept` | `observations/filter_tests.rs` | Normal declarative sentences pass the filter |
| `test_svo_extraction` | `observations/rules.rs` | "Caroline attended the LGBTQ support group" → SVO detected |
| `test_preference_detection` | `observations/rules.rs` | "I prefer dark chocolate" → preference pattern detected |
| `test_fact_pattern_detection` | `observations/rules.rs` | "Caroline is a social worker" → fact pattern "X is Y" detected |
| `test_subject_from_entity` | `observations/rules.rs` | Subject extracted from known entity in sentence |
| `test_self_contained_construction` | `observations/rules.rs` | "She went to Paris" + entity "Caroline" → "Caroline went to Paris" |
| `test_temporal_yesterday` | `observations/temporal.rs` | "yesterday" relative to 2023-06-15 → 2023-06-14 |
| `test_temporal_last_year` | `observations/temporal.rs` | "last year" relative to 2023-06-15 → 2022-06-15 |
| `test_temporal_in_march` | `observations/temporal.rs` | "in March" relative to 2023-06-15 → 2023-03-01 (past March) |
| `test_temporal_two_days_ago` | `observations/temporal.rs` | "two days ago" relative to ref → ref - 2 days |
| `test_temporal_no_expression` | `observations/temporal.rs` | No temporal expression → returns original timestamp |
| `test_observation_from_simple_message` | `observations/rules.rs` | Full extraction pipeline on a simple declarative message |
| `test_observation_from_multi_sentence` | `observations/rules.rs` | Multiple observations from multi-sentence text |
| `test_no_observations_from_greeting` | `observations/rules.rs` | "Hey! How are you?" → zero observations |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_observation_node_creation` | `observations/mod.rs` | Observation node created with correct fields in uni-db |
| `test_observed_in_edge_message` | `observations/mod.rs` | OBSERVED_IN edge wired to source Message |
| `test_observed_in_edge_chunk` | `observations/mod.rs` | OBSERVED_IN edge wired to source Chunk (artifact path) |
| `test_about_edge_wiring` | `observations/mod.rs` | ABOUT edges created for each referenced entity |
| `test_observed_during_edge` | `observations/mod.rs` | OBSERVED_DURING edge created when qualifying Episode exists |
| `test_observed_during_no_episode` | `observations/mod.rs` | No OBSERVED_DURING edge when no qualifying Episode |
| `test_observed_during_time_window` | `observations/mod.rs` | Episode outside 5-min window → no OBSERVED_DURING |
| `test_contradiction_flagging` | `observations/contradiction.rs` | Low-similarity same-subject observations flagged |
| `test_no_contradiction_different_subject` | `observations/contradiction.rs` | Different subjects → no flag even if low similarity |
| `test_no_contradiction_high_similarity` | `observations/contradiction.rs` | Same subject, high similarity → no flag (reinforcing) |
| `test_artifact_chunk_path` | `observations/mod.rs` | Complete path: Chunk → P3 → Observation → OBSERVED_IN → Chunk |
| `test_consolidation_notification` | `observations/mod.rs` | ObservationsReady sent to consolidation channel |
| `test_llm_extraction_mock` | `observations/llm.rs` | LLM path with mock provider returns valid observations |
| `test_llm_merge_dedup` | `observations/llm.rs` | Duplicate observations (cosine > 0.9) merged, higher confidence kept |
| `test_llm_validation_rejects_unknown_entity` | `observations/llm.rs` | LLM observation referencing non-P2 entity is discarded |
| `test_offline_mode` | `observations/mod.rs` | With circuit breaker open: rule-based runs, LLM skipped, no error |
| `test_auto_embed_populated` | `observations/mod.rs` | Observation.embedding is populated after node creation (via P7a) |

### Performance Tests

| Test | Target | What It Validates |
|---|---|---|
| `bench_filter_throughput` | < 0.1ms per call | Filtering is negligible cost |
| `bench_rule_extraction` | < 100ms per message | Rule-based extraction is fast enough for sync-adjacent use |
| `bench_end_to_end_latency` | < 5s per message (NF6) | Complete P3 pipeline within latency target |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_filter_deterministic` | Same input always produces same filter result |
| `proptest_temporal_resolution_bounded` | Resolved timestamp is never more than 2 years from reference |
| `proptest_observation_has_subject` | Every extracted observation has a non-empty subject |
| `proptest_self_contained_no_pronouns` | Constructed observations contain no unresolved pronouns ("she", "he", "they") |

### Validation Criteria

| Metric | Offline Target | Online Target | Source |
|---|---|---|---|
| Observation recall | > 40% | > 80% | Spec: Offline Mode table |
| No observations from greetings | 0 observations from non-informative content | 0 | F34 |
| Contradictions flagged correctly | All same-subject, low-similarity pairs flagged | All | F38 prerequisite |
| Latency | < 5s per message | < 5s per message | NF6 |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module-level doc comment | `observations/mod.rs` | P3 pipeline overview, data flow, online vs offline modes |
| `is_informative` doc comment | `observations/filter.rs` | Complete list of filtered categories with examples |
| `RawObservation` doc comment | `observations/rules.rs` | Field semantics, confidence ranges, self-contained requirement |
| `resolve_temporal` doc comment | `observations/temporal.rs` | Supported temporal expressions with examples |
| `ContradictionFlag` doc comment | `observations/contradiction.rs` | When flags are created, what P4 does with them |
| `ObservationsReady` doc comment | `observations/mod.rs` | Consolidation trigger mechanism |

---

## Review Checklist

- [ ] `observations/filter.rs` exists and `is_informative` correctly rejects all non-informative categories
- [ ] Greetings: "Hey!", "Wow!", "Thanks!", "lol" all filtered
- [ ] Pure questions: "Where is the store?" filtered; "Did you know X went to Y?" kept
- [ ] Short content: < 5 words filtered
- [ ] System messages: content_type "system" and "error" filtered
- [ ] `observations/rules.rs` exists and `extract_observations_rule_based` produces observations
- [ ] Sentence splitting handles abbreviations (Mr., Dr., etc.) without false splits
- [ ] SVO structure detection works for common English patterns
- [ ] Preference patterns detected: "I prefer", "I like", "I don't like"
- [ ] Fact patterns detected: "X is Y", "X has Y", "X went to Y"
- [ ] Subject extraction prioritizes named entities from P2
- [ ] Self-contained construction resolves pronouns to entity names
- [ ] `observations/temporal.rs` resolves: "yesterday", "last year", "in March", "two days ago"
- [ ] No temporal expression → returns original message timestamp
- [ ] `observations/llm.rs` exists and is gated by `#[cfg(feature = "llm")]`
- [ ] LLM observations validated against P2 entity list (no hallucinated entities)
- [ ] Merge deduplicates by embedding similarity > 0.9
- [ ] Circuit breaker open → LLM path skipped without error
- [ ] Observation nodes created with: observation_id, content, subject, observed_at, confidence
- [ ] OBSERVED_IN edge created for every observation (→ Message or → Chunk)
- [ ] ABOUT edges created for each referenced entity
- [ ] OBSERVED_DURING edge created only when qualifying Episode exists within 5-min window
- [ ] Artifact chunk path: OBSERVED_IN → Chunk (not Message)
- [ ] Contradiction flags created when: same subject AND cosine similarity < 0.3
- [ ] Contradictions are flagged only — no invalidation in P3
- [ ] ObservationsReady notification sent to consolidation worker
- [ ] Consolidation triggers at threshold (20 observations) or timer (15 min)
- [ ] Auto-embed triggers on Observation creation (P7a)
- [ ] error_policy is Skip (individual failures don't block pipeline)
- [ ] Offline mode: rule-based extraction works without LLM, no errors
- [ ] Latency < 5s per message (NF6)
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All property-based tests pass

---

## Definition of Done

1. **Filtering works:** Non-informative content (greetings, pure questions, system messages, short text) is filtered before extraction. Zero observations produced from filtered content.
2. **Rule-based extraction functional:** Observations extracted from declarative sentences containing P2 entities, with SVO structure, preference patterns, and fact patterns. Works without LLM (F72).
3. **Temporal resolution correct:** "yesterday", "last year", "in March", "two days ago" all resolve to correct timestamps relative to message timestamp.
4. **Self-contained observations:** Every observation is understandable without surrounding context. No sentence fragments. No unresolved pronouns.
5. **LLM path optional and gated:** LLM extraction runs only when feature flag enabled and circuit breaker closed. Failures skip gracefully.
6. **Edge wiring complete:** OBSERVED_IN (always), ABOUT (for each entity), OBSERVED_DURING (when qualifying Episode exists within 5-min window) all wired correctly.
7. **Artifact chunk path functional:** Observations from Artifact chunks link via OBSERVED_IN → Chunk, not Message. Full path tested.
8. **Contradictions flagged, not resolved:** Same-subject observations with embedding similarity < 0.3 flagged for P4. No invalidation in P3.
9. **Consolidation notified:** ObservationsReady sent to consolidation worker with observation count and source node IDs. Triggers at 20 observations or 15-min timer.
10. **Auto-embed active:** Observation.embedding populated on creation via P7a auto-embed.
11. **Latency within target:** End-to-end P3 latency < 5s per message (NF6).
12. **Recall targets met:** Offline recall > 40%, online recall > 80% on validation set.
13. **All tests pass:** Unit, integration, property-based, and performance tests green.
