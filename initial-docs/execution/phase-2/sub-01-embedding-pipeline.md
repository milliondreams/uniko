# Phase 8: Embedding & Summary Pipeline (P7)

## Context

This phase implements all 4 sub-pipelines of Pipeline 7: auto-embed (P7a), computed embedding (P7b), artifact pooling (P7c), and summarization (P7d). Every node type in the system gets the correct embedding through this pipeline, enabling the vector search that powers the recall cascade and all semantic matching operations.

The embedding pipeline is not a single sequential process — it is four independently triggered sub-pipelines that run alongside the main P1-P4 chain. P7a fires on node creation. P7b fires on node creation or update. P7c fires after all chunks of an artifact are embedded. P7d fires on lifecycle events (session end, task completion, etc.).

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). The pipeline chain is P1 (Ingest) -> P2 (NER) -> P3 (Observations) -> P4 (Consolidation). P7 (Embedding/Summary) runs alongside. Consolidation derives Facts from Observations using BTIC temporal intervals for validity tracking.

**Key principle:** Every node type that participates in semantic search must have an embedding. The embedding strategy varies by node type: auto-embed (uni-db handles it), computed (application constructs the embed string), pooled (aggregated from children), or none (queried by indexed fields only). Getting the embed string wrong — especially for Episodes — makes the recall cascade non-functional.

**Critical fix from v5:** Episode embeds the topic extracted from state JSON (e.g., "LGBTQ support group, career plans"), NOT `"conversation success"`. This makes Phase 2 of the recall cascade functional. Episode embeddings must be diverse and reflect what was actually discussed/done.

## Prerequisites

- **Phase 7 (Observation P3) complete** — P7 needs nodes created by P1-P3. Phases 5-7 create the nodes; Phase 8 ensures they're all embedded.
- **Phase 3 (Schema) complete** — All 16 node types defined with embedding vector fields.
- **Phase 4 (KnowledgeBase L1) complete** — uni-db auto-embed configuration, vector index management, node CRUD.
- **Embedding model available** — fastembed or Xervo embedding model configured (384d default dimension for text).
- **Pipeline management infrastructure** — Step trait, error policies, channel-based actor model.

## Sub-phases

---

### 8.1 — Auto-Embed Configuration (P7a)

**Objective:** Configure uni-db to automatically embed specific node types on creation. These are nodes where the embedding source is a single text field — no construction needed.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/embedding/auto_embed.rs` | Rust | Auto-embed configuration and verification |

#### Auto-Embed Mapping

| Node Type | Embedding Field | Source Field | Trigger |
|---|---|---|---|
| Message | `embedding` | `content` | Node creation in P1 |
| Chunk | `embedding` | `text` | Node creation in P1 |
| Observation | `embedding` | `content` | Node creation in P3 |
| Summary | `embedding` | `text` | Node creation in P7d |

#### Functions

```rust
/// Configure uni-db auto-embed for all node types that use single-field embedding.
/// Called once during KnowledgeBase initialization.
///
/// Auto-embed means uni-db embeds the node automatically when the source field
/// is set during node creation or update. No application code needed per node.
pub fn configure_auto_embed(
    kb: &KnowledgeBase,
    model: &EmbedModel,
) -> Result<()>;

/// Verify that auto-embed is functioning correctly.
/// Creates a test node, checks that the embedding field is populated,
/// then removes the test node.
pub async fn verify_auto_embed(
    kb: &KnowledgeBase,
    node_type: &str,
    source_field: &str,
) -> Result<bool>;
```

#### Embedding Model Configuration

```rust
/// Configuration for the embedding model used across P7.
pub struct EmbedModelConfig {
    /// Model name or path (e.g., "all-MiniLM-L6-v2" for fastembed)
    pub model_name: String,
    /// Embedding dimensions (default: 384 for text models)
    pub dimensions: usize,
    /// Maximum input tokens before truncation
    pub max_tokens: usize,
    /// Batch size for bulk embedding operations
    pub batch_size: usize,
}

impl Default for EmbedModelConfig {
    fn default() -> Self {
        Self {
            model_name: "all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
            max_tokens: 512,
            batch_size: 32,
        }
    }
}
```

#### Latency Target

- < 50ms per node (batch-embedded by uni-db)
- Auto-embed is synchronous with node creation — the node is not "complete" until embedded

---

### 8.2 — Computed Embedding (P7b)

**Objective:** For node types where the embedding is derived from multiple fields or requires domain-specific construction, compute the embed string at the application level and then call the embedding model.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/embedding/computed.rs` | Rust | Computed embedding logic for all node types |

#### Embed String Construction

```rust
/// Construct the text string that will be embedded for a given node type.
/// This is the most critical function in P7 — getting these strings wrong
/// makes semantic search non-functional for the affected node types.
///
/// Each node type has a different construction strategy based on which
/// fields carry the most semantic signal.
pub fn compute_embed_string(node_type: &str, node: &dyn NodeData) -> String;
```

#### Per-Node-Type Embed String Rules

| Node Type | Construction | Example | Rationale |
|---|---|---|---|
| Entity | `name + " (" + entity_type + ")"` | `"Java (programming_language)"` | Disambiguates entities with same name but different types. Falls back to just `name` if `entity_type` is null. |
| Goal | `title` | `"Reduce refund cycle time by 40%"` | Title only. Description may be paragraphs that dilute the embedding signal. |
| Task | `title` | `"Investigate auth module dependencies"` | Same reasoning as Goal. |
| Session | `topic` (initially); `topic + " " + summary` (after session ends) | `"LGBTQ support group, career plans"` | Re-embed when summary is generated (P7d). Summary adds semantic context after session ends. |
| Topic | `name + " " + summary` | `"Caroline's adoption journey ..."` | Both name and summary carry signal. |
| Fact | `subject + " " + predicate + " " + object` | `"Caroline pursuing adoption"` | Always short text. Concatenation captures the triple structure. |
| Procedure | `name + ": " + description` (truncate description to 200 chars) | `"revenue-dip analysis: 7-step diagnostic..."` | Name anchors, description provides detail. Truncation prevents long descriptions from dominating. |
| Episode | See detailed Episode section below | `"LGBTQ support group, career plans -- conversation complete"` | Extract topic from state JSON, NOT just action_type + outcome. |
| Action | See detailed Action section below | `"file_read /src/auth.rs"` | action_type + key input identifier. |

#### Episode Embed String (Critical)

```rust
/// Construct the embed string for an Episode node.
/// This is the hardest embedding to get right because state is domain-specific JSON.
///
/// Strategy: Extract text from state JSON using a priority list of keys,
/// then append action_type and outcome.
///
/// Priority order for state key extraction:
///   1. state.topic       → "LGBTQ support group, career plans"
///   2. state.question    → "What does RAII stand for?"
///   3. state.description → "Investigating auth module dependencies"
///   4. state.summary     → "User reported login failures on mobile"
///   5. state.input       → (first 200 chars of input text)
///   6. (none found)      → fall back to action_type alone
///
/// Final embed string: `{extracted_text} — {action_type} {outcome}`
/// Example: "LGBTQ support group, career plans — conversation complete"
///
/// CRITICAL: Never embed as just "conversation success" — this makes all
/// conversation episodes identical in vector space, destroying Phase 2
/// recall cascade functionality.
fn episode_embed_string(episode: &Episode) -> String;
```

```rust
/// Extract the most semantically relevant text from an Episode's state JSON.
/// Tries keys in priority order: topic > question > description > summary > input.
/// Returns the first non-empty value found, truncated to 200 chars.
fn extract_topic_from_state(state: &serde_json::Value) -> Option<String>;
```

#### Action Embed String

```rust
/// Construct the embed string for an Action node.
/// Extracts the key identifier from the input JSON based on action_type.
///
/// action_type = "file_read"     → input.path         → "file_read /src/auth.rs"
/// action_type = "command_run"   → input.command       → "command_run cargo test --lib"
/// action_type = "search"        → input.query         → "search authentication flow"
/// action_type = "delegate"      → input.task          → "delegate review PR #42"
/// action_type = "file_write"    → input.path          → "file_write /src/config.rs"
/// action_type = "api_call"      → input.endpoint      → "api_call /v1/users"
/// (general)                     → first string value  → "action_type {first_string_value}"
fn action_embed_string(action: &Action) -> String;

/// Extract the key input summary from an Action's input JSON.
/// Looks for type-specific keys first, then falls back to the first
/// string value in the JSON object.
fn extract_key_input(action_type: &str, input: &serde_json::Value) -> Option<String>;
```

#### Embed and Set

```rust
/// Embed a text string and set the resulting vector on a node.
/// Used for all computed embeddings (P7b).
///
/// Steps:
///   1. Construct embed string via compute_embed_string()
///   2. Call embed model to get vector
///   3. Set embedding field on the node
///   4. uni-db updates the vector index automatically
pub async fn embed_and_set(
    kb: &KnowledgeBase,
    node_id: NodeId,
    embed_string: &str,
    model: &EmbedModel,
) -> Result<()>;
```

#### Trigger Points

P7b runs on node creation or update from:

| Source | Node Types Affected |
|---|---|
| P1 (Ingest) | Session (topic set) |
| P2 (NER) | Entity (created or merged) |
| P3 (Observations) | — (Observations use auto-embed P7a) |
| P4 (Consolidation) | Fact (created), Entity (updated) |
| P5 (Procedure Promotion) | Procedure (created or updated) |
| P6 (Topic Detection) | Topic (created or updated) |
| Agent tools: record_episode | Episode |
| Agent tools: record_action | Action |
| Agent tools: create_goal | Goal |
| Agent tools: create_task | Task |
| Agent tools: end_session | Session (re-embed with summary) |

#### Latency Target

- < 100ms per node

---

### 8.3 — Artifact Embedding Pooling (P7c)

**Objective:** Compute the Artifact-level text embedding by mean-pooling all chunk embeddings. This handles large artifacts that cannot be embedded as a single unit.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/embedding/pooling.rs` | Rust | Chunk embedding aggregation |

#### Functions

```rust
/// Pool chunk embeddings to produce the Artifact-level text_embedding.
///
/// Algorithm:
///   1. Query all Chunks: MATCH (a:Artifact {artifact_id: $id})-[:HAS_CHUNK]->(c:Chunk)
///   2. Collect all chunk embedding vectors
///   3. Compute element-wise mean: artifact_text_embedding[i] = mean(chunk_embeddings[*][i])
///   4. Set Artifact.text_embedding to the pooled vector
///
/// Waits for all chunk embeddings to be present before pooling.
/// If any chunk is missing its embedding, waits up to 30s with polling.
pub async fn pool_chunk_embeddings(
    kb: &KnowledgeBase,
    artifact_id: &str,
) -> Result<()>;

/// Compute the element-wise mean of a collection of embedding vectors.
/// All vectors must have the same dimensionality.
/// Returns an error if the input is empty or dimensions mismatch.
fn mean_pool(embeddings: &[Vec<f32>]) -> Result<Vec<f32>>;

/// Check if all chunks of an artifact have their embeddings computed.
/// Returns (ready_count, total_count).
async fn check_chunk_embedding_completeness(
    kb: &KnowledgeBase,
    artifact_id: &str,
) -> Result<(usize, usize)>;
```

#### Polling Strategy

```
1. After P1 creates all Chunk nodes for an Artifact:
   → Queue P7c task for this Artifact
2. P7c checks if all chunks have embeddings:
   → All ready: proceed with mean-pool
   → Not all ready: wait 1s, retry (up to 30 attempts = 30s max)
   → Timeout: log warning, pool with available embeddings
3. Set Artifact.text_embedding
```

#### Scope

This sub-phase handles text/code artifacts only. Multimodal embeddings (image, audio, video) are research-tier (F25/RES) and deferred:

| Artifact Type | P7c Scope | Status |
|---|---|---|
| Text/code | text_embedding via mean-pool | This phase |
| Image | image_embedding via CLIP/SigLIP | Deferred (RES) |
| Audio | audio_embedding via CLAP | Deferred (RES) |
| Video | video_embedding via LanguageBind | Deferred (RES) |
| Any | multimodal_embedding via ImageBind | Deferred (RES) |

#### Latency Target

- < 500ms for text pooling (mean-pool is O(n * d) where n = chunks, d = dimensions)

---

### 8.4 — Summarization (P7d)

**Objective:** Generate Summary nodes at various levels (session, task, goal, entity, topic, artifact) triggered by lifecycle events. Summaries are LLM-dependent — when LLM is unavailable, summarization skips entirely.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/embedding/summarization.rs` | Rust | Summary generation and edge wiring |

#### Summary Triggers

| Level | Trigger Event | Input | Output |
|---|---|---|---|
| Session | Session ends (inactivity timeout or explicit `end_session`) | All Messages in session | Summary (level: "session") + SUMMARIZES -> Session |
| Task | Task status -> "completed" or "failed" | All Episodes and Sessions FOR_TASK | Summary (level: "task") + SUMMARIZES -> Task |
| Goal | Goal status change, or periodic (daily) | All Tasks PART_OF goal + their summaries | Summary (level: "goal") + SUMMARIZES -> Goal |
| Entity | Entity frequency crosses threshold (10, 50, 100) | All Observations ABOUT entity + all Facts ABOUT entity | Summary (level: "entity") + SUMMARIZES -> Entity |
| Topic | P6 creates/updates topic | Member entities + their facts + their observations | Summary (level: "topic") + SUMMARIZES -> Topic |
| Artifact | After chunking + embedding for text artifacts > 2000 tokens | All chunks of the artifact | Summary (level: "artifact") + SUMMARIZES -> Artifact |

#### Functions

```rust
/// Generate a session summary when a session ends.
/// Collects all messages in the session, sends to LLM for summarization,
/// creates Summary node + SUMMARIZES edge, then re-embeds the Session
/// (which now has topic + summary for a richer embedding).
pub async fn summarize_session(
    kb: &KnowledgeBase,
    session_id: &str,
    provider: &LlmProvider,
) -> Result<Option<NodeId>>;

/// Generate a task summary when a task completes or fails.
pub async fn summarize_task(
    kb: &KnowledgeBase,
    task_id: &str,
    provider: &LlmProvider,
) -> Result<Option<NodeId>>;

/// Generate a goal summary on status change or periodic trigger.
pub async fn summarize_goal(
    kb: &KnowledgeBase,
    goal_id: &str,
    provider: &LlmProvider,
) -> Result<Option<NodeId>>;

/// Generate an entity summary when frequency crosses a threshold.
/// Example output: "Caroline is a transgender woman pursuing adoption.
/// She attended LGBTQ support groups, chose an inclusive agency,
/// and is interested in counseling as a career."
pub async fn summarize_entity(
    kb: &KnowledgeBase,
    entity_id: &str,
    provider: &LlmProvider,
) -> Result<Option<NodeId>>;

/// Generate a topic summary when P6 creates or updates a topic.
pub async fn summarize_topic(
    kb: &KnowledgeBase,
    topic_id: &str,
    provider: &LlmProvider,
) -> Result<Option<NodeId>>;

/// Generate an artifact summary for text artifacts > 2000 tokens.
pub async fn summarize_artifact(
    kb: &KnowledgeBase,
    artifact_id: &str,
    provider: &LlmProvider,
) -> Result<Option<NodeId>>;
```

#### Summary Node Creation

```rust
/// Create a Summary node and wire the SUMMARIZES edge.
///
/// Fields:
///   summary_id: new_id() (UUID v7)
///   text: LLM-generated summary text
///   level: "session" | "task" | "goal" | "entity" | "topic" | "artifact"
///   generated_at: now()
///   embedding: auto-embedded by uni-db from text field (P7a)
///
/// Edge: SUMMARIZES → target node (Session, Task, Goal, Entity, Topic, or Artifact)
async fn create_summary_node(
    kb: &KnowledgeBase,
    text: &str,
    level: &str,
    target_node_id: NodeId,
) -> Result<NodeId>;
```

#### Re-Embedding on Summary

When a Summary is generated for a Session or Topic, the parent node must be re-embedded because its embed string now includes the summary:

| Node Type | Before Summary | After Summary | Action |
|---|---|---|---|
| Session | embed = `topic` | embed = `topic + " " + summary` | Re-embed via P7b |
| Topic | embed = `name` | embed = `name + " " + summary` | Re-embed via P7b |

```rust
/// Re-embed a node after its summary is generated.
/// Called after summarize_session and summarize_topic to update the
/// parent node's embedding with the new summary content.
async fn re_embed_after_summary(
    kb: &KnowledgeBase,
    node_id: NodeId,
    node_type: &str,
    model: &EmbedModel,
) -> Result<()>;
```

#### LLM Prompts

**Session summary:**
```
Summarize this conversation session in 2-3 sentences. Focus on key topics discussed,
decisions made, and any facts established. Do not include greetings or filler.

Messages:
{messages_chronological}
```

**Entity summary:**
```
Summarize everything known about {entity_name} based on these observations and facts.
Write 2-4 sentences covering key attributes, relationships, and activities.

Observations:
{observations}

Facts:
{facts}
```

**Artifact summary:**
```
Summarize this document in 2-3 sentences. Focus on the main topic, key points,
and any conclusions or recommendations.

Chunks:
{chunks}
```

#### Error Handling & Offline Mode

| Condition | Behavior |
|---|---|
| LLM circuit breaker open | Skip summarization entirely, log warning. No Summary node created. |
| LLM timeout | Skip this summary, do not retry (error_policy: Skip). |
| LLM returns invalid response | Skip, log warning. |
| LLM feature flag disabled | Compile-time exclusion of all summarization code. |
| Summary already exists for target | Create new Summary (multiple summaries per target are valid — entity summaries at 10, 50, 100). |

**error_policy: Skip** — Summaries are nice-to-have. A missing summary degrades the Session/Topic embedding quality but does not break any functional path.

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_entity_embed_string` | `embedding/computed.rs` | Entity: `"Java (programming_language)"` |
| `test_entity_embed_string_no_type` | `embedding/computed.rs` | Entity without type: `"Java"` (fallback to name only) |
| `test_goal_embed_string` | `embedding/computed.rs` | Goal: title only, description excluded |
| `test_task_embed_string` | `embedding/computed.rs` | Task: title only |
| `test_session_embed_string_initial` | `embedding/computed.rs` | Session before summary: topic only |
| `test_session_embed_string_with_summary` | `embedding/computed.rs` | Session after summary: `"topic summary_text"` |
| `test_topic_embed_string` | `embedding/computed.rs` | Topic: `"name summary"` |
| `test_fact_embed_string` | `embedding/computed.rs` | Fact: `"Caroline pursuing adoption"` |
| `test_procedure_embed_string` | `embedding/computed.rs` | Procedure: `"name: description..."` (truncated at 200 chars) |
| `test_procedure_embed_string_truncation` | `embedding/computed.rs` | Description longer than 200 chars is truncated |
| `test_episode_embed_string_topic` | `embedding/computed.rs` | Episode with state.topic: `"LGBTQ support group — conversation complete"` |
| `test_episode_embed_string_question` | `embedding/computed.rs` | Episode with state.question (no topic): `"What does RAII stand for? — investigate success"` |
| `test_episode_embed_string_description` | `embedding/computed.rs` | Episode with state.description: description used |
| `test_episode_embed_string_summary` | `embedding/computed.rs` | Episode with state.summary: summary used |
| `test_episode_embed_string_input` | `embedding/computed.rs` | Episode with state.input only: first 200 chars used |
| `test_episode_embed_string_fallback` | `embedding/computed.rs` | Episode with empty state: `"conversation"` (action_type only) |
| `test_episode_embed_not_conversation_success` | `embedding/computed.rs` | **Critical**: Episode embedding is NOT `"conversation success"` — it includes topic content |
| `test_action_embed_file_read` | `embedding/computed.rs` | Action file_read: `"file_read /src/auth.rs"` |
| `test_action_embed_command_run` | `embedding/computed.rs` | Action command_run: `"command_run cargo test --lib"` |
| `test_action_embed_search` | `embedding/computed.rs` | Action search: `"search authentication flow"` |
| `test_action_embed_delegate` | `embedding/computed.rs` | Action delegate: `"delegate review PR #42"` |
| `test_action_embed_general` | `embedding/computed.rs` | Action with unknown type: `"custom_type {first_string_value}"` |
| `test_mean_pool_basic` | `embedding/pooling.rs` | Mean of [[1,2,3], [3,4,5]] = [2,3,4] |
| `test_mean_pool_single` | `embedding/pooling.rs` | Mean of single vector = that vector |
| `test_mean_pool_empty` | `embedding/pooling.rs` | Empty input → error |
| `test_mean_pool_dimension_mismatch` | `embedding/pooling.rs` | Mismatched dimensions → error |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_auto_embed_message` | `embedding/auto_embed.rs` | Message.embedding populated on creation from content |
| `test_auto_embed_chunk` | `embedding/auto_embed.rs` | Chunk.embedding populated on creation from text |
| `test_auto_embed_observation` | `embedding/auto_embed.rs` | Observation.embedding populated on creation from content |
| `test_auto_embed_summary` | `embedding/auto_embed.rs` | Summary.embedding populated on creation from text |
| `test_computed_embed_entity` | `embedding/computed.rs` | Entity.embedding set after compute + embed_and_set |
| `test_computed_embed_episode` | `embedding/computed.rs` | Episode.embedding set with topic-based embed string |
| `test_computed_embed_action` | `embedding/computed.rs` | Action.embedding set with type+input embed string |
| `test_pooling_matches_manual` | `embedding/pooling.rs` | Artifact.text_embedding = manually computed mean of chunk embeddings |
| `test_pooling_waits_for_chunks` | `embedding/pooling.rs` | Pooling waits for all chunk embeddings before computing |
| `test_pooling_timeout` | `embedding/pooling.rs` | Missing chunk embedding after 30s → pool with available, log warning |
| `test_summarize_session_mock` | `embedding/summarization.rs` | Session summary created with mock LLM, SUMMARIZES edge wired |
| `test_summarize_entity_mock` | `embedding/summarization.rs` | Entity summary created at frequency threshold, SUMMARIZES edge wired |
| `test_summarize_artifact_mock` | `embedding/summarization.rs` | Artifact summary created for > 2000 token artifact |
| `test_session_re_embed_after_summary` | `embedding/summarization.rs` | Session.embedding updated to include summary after summarization |
| `test_topic_re_embed_after_summary` | `embedding/summarization.rs` | Topic.embedding updated to include summary after summarization |
| `test_offline_mode_summarization_skips` | `embedding/summarization.rs` | Circuit breaker open → no Summary nodes created, no errors |
| `test_offline_mode_auto_embed_works` | `embedding/auto_embed.rs` | Auto-embed functions without LLM (embedding model is local) |
| `test_every_node_type_has_embedding` | `embedding/computed.rs` | Create one of each node type → all have non-null embedding |

### Performance Tests

| Test | Target | What It Validates |
|---|---|---|
| `bench_auto_embed_latency` | < 50ms per node | Auto-embed within latency target |
| `bench_computed_embed_latency` | < 100ms per node | Computed embedding within latency target |
| `bench_pooling_latency` | < 500ms for 50 chunks | Pooling within latency target |
| `bench_embed_string_construction` | < 1ms per node | String construction is negligible overhead |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_embed_string_non_empty` | Every node type produces a non-empty embed string |
| `proptest_episode_embeds_diverse` | Episodes with different state.topic values produce different embed strings |
| `proptest_mean_pool_dimensions_preserved` | Output dimensions always match input dimensions |
| `proptest_mean_pool_commutative` | Order of input vectors doesn't affect result |

### Validation Criteria

| Metric | Target | How Measured |
|---|---|---|
| Every node type has embedding | 100% of created nodes (except Participant, Rule, ConsolidationCycle) | Query for null embedding fields |
| Episode embeddings are diverse | Pairwise cosine similarity < 0.9 for episodes with different topics | Compute similarity matrix |
| Pooled embeddings correlate with content | Artifact text_embedding nearest neighbors match content | Search quality test |
| Auto-embed latency | < 50ms per node | Instrumented timing |
| Computed embed latency | < 100ms per node | Instrumented timing |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module-level doc comment | `embedding/mod.rs` | P7 overview, 4 sub-pipeline descriptions, which nodes use which strategy |
| `compute_embed_string` doc comment | `embedding/computed.rs` | Complete mapping of node type to embed string construction |
| `episode_embed_string` doc comment | `embedding/computed.rs` | State key priority list, critical warning about "conversation success" anti-pattern |
| `action_embed_string` doc comment | `embedding/computed.rs` | Action type to input key mapping |
| `mean_pool` doc comment | `embedding/pooling.rs` | Algorithm, edge cases (empty, dimension mismatch) |
| `EmbedModelConfig` doc comment | `embedding/auto_embed.rs` | Model selection rationale, dimension choices |
| Summarization triggers table | `embedding/summarization.rs` | When each summary level is triggered, what input is used |

---

## Review Checklist

### P7a — Auto-Embed
- [ ] Auto-embed configured for Message.embedding <- content
- [ ] Auto-embed configured for Chunk.embedding <- text
- [ ] Auto-embed configured for Observation.embedding <- content
- [ ] Auto-embed configured for Summary.embedding <- text
- [ ] Auto-embed verified: creating a Message populates embedding field
- [ ] Embedding model configured (fastembed or Xervo, 384d default)
- [ ] Auto-embed latency < 50ms per node verified

### P7b — Computed Embedding
- [ ] `compute_embed_string` handles all 9 node types: Entity, Goal, Task, Session, Topic, Fact, Procedure, Episode, Action
- [ ] Entity embed: `"name (entity_type)"`, fallback to `"name"` if no type
- [ ] Goal embed: title only
- [ ] Task embed: title only
- [ ] Session embed: topic initially, topic + summary after session ends
- [ ] Topic embed: name + summary
- [ ] Fact embed: subject + predicate + object
- [ ] Procedure embed: name + ": " + description[:200]
- [ ] Episode embed extracts topic from state JSON using priority: topic > question > description > summary > input
- [ ] Episode embed is NOT "conversation success" — includes topic content
- [ ] Episode embed appends action_type and outcome after topic: `"{topic} — {action_type} {outcome}"`
- [ ] Action embed: action_type + key input (path for file_read, command for command_run, query for search, etc.)
- [ ] `embed_and_set` calls embedding model and updates node
- [ ] P7b triggers on node creation and update
- [ ] Computed embed latency < 100ms per node verified

### P7c — Artifact Pooling
- [ ] `pool_chunk_embeddings` queries all chunks via HAS_CHUNK edges
- [ ] `mean_pool` computes correct element-wise mean
- [ ] Handles empty chunk list (error)
- [ ] Handles dimension mismatch (error)
- [ ] Waits for all chunk embeddings before pooling (polling with timeout)
- [ ] Sets Artifact.text_embedding after pooling
- [ ] Pooling latency < 500ms for text artifacts

### P7d — Summarization
- [ ] Session summary: triggers on session end, collects all messages, creates Summary + SUMMARIZES edge
- [ ] Task summary: triggers on task completion/failure
- [ ] Goal summary: triggers on goal status change or periodic
- [ ] Entity summary: triggers when frequency crosses 10/50/100
- [ ] Topic summary: triggers when P6 creates/updates topic
- [ ] Artifact summary: triggers for text artifacts > 2000 tokens
- [ ] Summary node has: summary_id, text, level, generated_at, embedding (auto-embedded)
- [ ] SUMMARIZES edge created from Summary to target node
- [ ] Session re-embedded after summary (embed includes summary now)
- [ ] Topic re-embedded after summary (embed includes summary now)
- [ ] Circuit breaker open → summarization skips entirely, no errors
- [ ] error_policy is Skip (summaries are nice-to-have)
- [ ] LLM feature flag gates all summarization code

### General
- [ ] No node type participates in semantic search without an embedding
- [ ] Participant, Rule, ConsolidationCycle correctly have no embedding
- [ ] Offline mode: auto-embed works (local model), summarization skips
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All property-based tests pass

---

## Definition of Done

1. **Auto-embed functional (P7a):** Message, Chunk, Observation, and Summary nodes are automatically embedded on creation. Verified by integration test.
2. **Computed embeddings correct (P7b):** All 9 computed node types produce semantically meaningful embed strings. Entity disambiguates by type. Episode embeds topic from state JSON, not "conversation success". Action embeds key input identifier.
3. **Episode embeddings are diverse:** Episodes with different topics produce different embeddings. Pairwise cosine similarity < 0.9 for distinct topics. The recall cascade Phase 2 can distinguish episodes by content.
4. **Artifact pooling works (P7c):** Artifact.text_embedding is the mean-pool of all chunk embeddings. Pooling waits for chunk embeddings, handles timeouts gracefully.
5. **Summarization functional (P7d):** Session, task, goal, entity, topic, and artifact summaries generated on their respective triggers. Summary nodes created with correct level and SUMMARIZES edges.
6. **Re-embedding on summary:** Session and Topic nodes are re-embedded when their summaries are generated, incorporating the summary into the embed string.
7. **Offline mode works:** Auto-embed and computed embed function without LLM (using local fastembed model). Summarization skips cleanly when circuit breaker is open.
8. **Every searchable node has embedding:** Query across all node types confirms non-null embeddings (except Participant, Rule, ConsolidationCycle).
9. **Latency targets met:** Auto-embed < 50ms, computed embed < 100ms, pooling < 500ms.
10. **All tests pass:** Unit, integration, property-based, and performance tests green.
