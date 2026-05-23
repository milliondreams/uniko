# Phase 12: Public API Facade & MVP Integration Testing

## Context

This phase creates the `uniko-api` crate as the public-facing facade with ergonomic builder APIs and re-exports, then runs the comprehensive integration test suite validating the full MVP pipeline end-to-end. This phase makes the MVP shippable.

The `uniko-api` crate is the only crate that external consumers interact with. It re-exports key types and wraps all tool functions as methods on a single `Uniko` entry point. `uniko-api` depends on `uniko-cortex` and `uniko-memory` and exposes no internal types from lower layers directly.

The integration test suite validates 10 scenarios that together prove the full pipeline works: Message -> NER -> Observation -> Consolidation -> Fact -> Recall. These tests are the final gate before shipping. If all 10 pass, the MVP is validated.

**Key principle:** The API must be ergonomic for the common case and powerful for the advanced case. Creating an agent, ingesting messages, and recalling context should be 3-5 lines of code. Advanced features (custom tier weights, contrastive recall, ASSUME/ABDUCE) are accessible through builder APIs.

## Prerequisites

- **Phase 11 (Agent Tools)** -- all tool functions must be implemented and tested. The API facade wraps these functions.
- **Phase 10 (Recall Cascade)** -- RecallContextBuilder must be fully functional for recall operations.
- **Phase 9 (Consolidation P4)** -- consolidation must derive Facts from Observations for the consolidation improvement test.
- **Phase 8 (Embedding P7)** -- embeddings must be computed for vector search to work in integration tests.
- **Phases 2-7** -- all pipelines (ingest, NER, observations, chunking, embedding, summarization) must be operational.
- **Phase 1 (Foundation)** -- workspace, schema, types, error handling all established.

## Sub-phases

---

### 12.1 -- uniko-api Facade Crate

**Objective:** Create the top-level `Uniko` struct as the single entry point for all operations. Re-export all public types. Initialize all internal layers on construction.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-api/src/lib.rs` | Rust | Uniko struct, re-exports, module declarations |
| `crates/uniko-api/src/config.rs` | Rust | UnikoConfig re-export + API-level config extensions |
| `crates/uniko-api/Cargo.toml` | Config | Dependencies on uniko-cortex and uniko-memory |

#### Types

```rust
/// Top-level entry point for the uniko cognitive memory system.
///
/// # Example
/// ```rust
/// let uniko = Uniko::in_memory(UnikoConfig::default()).await?;
/// uniko.create_goal("Ship v1", "Launch the MVP", json!({}), json!({}), None).await?;
/// let bundle = uniko.recall("what are our goals?", 8192, None).await?;
/// ```
pub struct Uniko {
    cortex: Cortex,
}
```

#### Constructor Methods

| Method | Signature | Description |
|---|---|---|
| `new` | `pub async fn new(config: UnikoConfig) -> Result<Self>` | Initializes all layers with default storage backend |
| `from_path` | `pub async fn from_path(path: &Path, config: UnikoConfig) -> Result<Self>` | Initializes with disk-backed storage at the given path |
| `in_memory` | `pub async fn in_memory(config: UnikoConfig) -> Result<Self>` | Initializes with in-memory storage (for tests) |

#### Initialization Sequence (inside constructors)

```
1. Create/open uni-db instance (disk-backed or in-memory)
2. Initialize uniko-store (KnowledgeBase): register schema, create indexes
3. Initialize uniko-extract: load NER model, load embedding model
4. Initialize uniko-pipes: Step trait, circuit breaker, retry, DLQ
5. Initialize uniko-memory: create pipeline workers, register pipelines, recall cascade
6. Initialize uniko-cortex: procedures, topics, MCTS
7. Register stdlib rules (4 Locy rules)
8. Start pipeline system (background workers)
9. Return Uniko { memory, cortex }
```

#### Re-exported Methods (delegate to uniko-memory / uniko-cortex tools)

**Lifecycle:**
| Method | Delegates To |
|---|---|
| `create_goal` | `tools::lifecycle::create_goal` |
| `create_task` | `tools::lifecycle::create_task` |
| `start_session` | `tools::lifecycle::start_session` |
| `end_session` | `tools::lifecycle::end_session` |
| `update_goal` | `tools::lifecycle::update_goal` |
| `update_task` | `tools::lifecycle::update_task` |
| `create_organization` | `tools::lifecycle::create_organization` |
| `create_team` | `tools::lifecycle::create_team` |
| `add_member` | `tools::lifecycle::add_member` |

**Knowledge:**
| Method | Delegates To |
|---|---|
| `record_episode` | `tools::knowledge::record_episode` |
| `record_action` | `tools::knowledge::record_action` |
| `add_observation` | `tools::knowledge::add_observation` |
| `assert_fact` | `tools::knowledge::assert_fact` |
| `invalidate_fact` | `tools::knowledge::invalidate_fact` |
| `add_rule` | `tools::knowledge::add_rule` |
| `author_rule` | `tools::knowledge::author_rule` |
| `share_fact` | `tools::knowledge::share_fact` |
| `shared_facts` | `tools::knowledge::shared_facts` |

**Query:**
| Method | Delegates To |
|---|---|
| `recall` | `tools::query::recall` |
| `search_entities` | `tools::query::search_entities` |
| `search_facts` | `tools::query::search_facts` |
| `search_messages` | `tools::query::search_messages` |
| `assume` | `tools::query::assume` |
| `abduce` | `tools::query::abduce` |
| `working_memory` | `tools::working_memory::working_memory` |

**Builders:**
| Method | Returns |
|---|---|
| `recall_context` | `RecallContextBuilder` (build and customize recall queries) |
| `ingest` | `IngestBuilder` (chainable message/artifact ingestion) |
| `episode` | `EpisodeBuilder` (chainable episode recording) |

#### Re-exported Types

```rust
// Node types & IDs (from uniko-store)
pub use uniko_store::{
    Message, Entity, Observation, Fact, Episode, Goal, Task, Session,
    Procedure, Rule, Action, Artifact, Chunk, Topic, Participant,
    Organization, Team, Summary, ConsolidationCycle,
    NodeId, GoalId, TaskId, SessionId, EpisodeId, FactId, RuleId,
    ActionId, ObservationId,
    UnikoConfig, UnikoError, Result,
};

// Recall & memory types (from uniko-memory)
pub use uniko_memory::{
    ContextBundle, WorkingMemoryBundle, IntentProfile,
    RecallItem, Tier, RecallContextBuilder, RecallFilters,
    PipelineHealth, PipelineStatus,
};
```

#### Shutdown

```rust
impl Uniko {
    /// Gracefully shuts down all background workers and flushes pending operations.
    /// Follows cancellation hierarchy: stop ingest (5s) -> stop consolidation (10s) -> force (30s).
    pub async fn shutdown(self) -> Result<()>;
}
```

---

### 12.2 -- Builder Pattern APIs

**Objective:** Provide ergonomic builder APIs for common multi-step operations: recall configuration, message ingestion, and episode recording.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-api/src/builders.rs` | Rust | IngestBuilder, EpisodeBuilder definitions |

#### IngestBuilder

```rust
pub struct IngestBuilder<'a> {
    uniko: &'a Uniko,
    messages: Vec<PendingMessage>,
    artifacts: Vec<PendingArtifact>,
    session_id: Option<SessionId>,
    participant_id: Option<String>,
}
```

| Method | Signature | Description |
|---|---|---|
| `message` | `fn message(mut self, content: &str, participant_id: &str) -> Self` | Add a message to ingest |
| `message_with_timestamp` | `fn message_with_timestamp(mut self, content: &str, participant_id: &str, timestamp: Timestamp) -> Self` | Add a message with explicit timestamp |
| `artifact` | `fn artifact(mut self, content: &str, content_type: &str, name: &str) -> Self` | Add an artifact to ingest |
| `in_session` | `fn in_session(mut self, session_id: &SessionId) -> Self` | Set the session for all messages |
| `as_participant` | `fn as_participant(mut self, participant_id: &str) -> Self` | Set default participant |
| `run` | `async fn run(self) -> Result<Vec<MessageId>>` | Execute ingestion, returns IDs of created messages |

**Usage example:**
```rust
let ids = uniko.ingest()
    .in_session(&session_id)
    .message("Hello, how are you?", "alice")
    .message("I'm doing well, thanks!", "bob")
    .message("Let's discuss the project plan.", "alice")
    .run()
    .await?;
```

#### EpisodeBuilder

```rust
pub struct EpisodeBuilder<'a> {
    uniko: &'a Uniko,
    action_type: String,
    outcome: String,
    state: Value,
    delta: Value,
    importance: f64,
    entity_refs: Vec<String>,
    session_id: Option<SessionId>,
    task_id: Option<TaskId>,
}
```

| Method | Signature | Description |
|---|---|---|
| `action` | `fn action(mut self, action_type: &str) -> Self` | Set action type |
| `outcome` | `fn outcome(mut self, outcome: &str) -> Self` | Set outcome |
| `state` | `fn state(mut self, state: Value) -> Self` | Set state snapshot |
| `delta` | `fn delta(mut self, delta: Value) -> Self` | Set state delta |
| `importance` | `fn importance(mut self, importance: f64) -> Self` | Set importance score |
| `entity` | `fn entity(mut self, name: &str) -> Self` | Add an entity reference |
| `entities` | `fn entities(mut self, names: Vec<String>) -> Self` | Add multiple entity references |
| `in_session` | `fn in_session(mut self, session_id: &SessionId) -> Self` | Set session context |
| `for_task` | `fn for_task(mut self, task_id: &TaskId) -> Self` | Set task context |
| `record` | `async fn record(self) -> Result<EpisodeId>` | Execute episode recording |

**Usage example:**
```rust
let episode_id = uniko.episode()
    .action("code_review")
    .outcome("success")
    .state(json!({"file": "main.rs", "changes": 15}))
    .delta(json!({"bugs_found": 2}))
    .importance(0.8)
    .entity("main.rs")
    .entity("authentication module")
    .in_session(&session_id)
    .for_task(&task_id)
    .record()
    .await?;
```

#### RecallContextBuilder (re-exported from Phase 10)

Already built in Phase 10 (`uniko-memory/src/recall/builder.rs`). Re-exported here and accessible via `uniko.recall_context(intent)`.

---

### 12.3 -- MVP Integration Test Suite

**Objective:** Validate the full MVP pipeline end-to-end with 10 comprehensive test scenarios. These tests prove that the system works as designed and is ready to ship.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `tests/integration/mvp_tests.rs` | Rust | All 10 integration test scenarios |
| `tests/integration/mod.rs` | Rust | Integration test module root |
| `tests/integration/helpers.rs` | Rust | Shared test utilities (create_test_uniko, seed_data, etc.) |

#### Test Helpers

```rust
/// Creates an in-memory Uniko instance for testing.
async fn create_test_uniko() -> Uniko {
    Uniko::in_memory(UnikoConfig::default()).await.unwrap()
}

/// Seeds a conversation: creates participant, session, and ingests messages.
async fn seed_conversation(uniko: &Uniko, messages: &[(&str, &str)]) -> SessionId {
    // Creates participants, starts session, ingests messages in order
}

/// Waits for all background pipelines to complete (NER, observations, embedding).
async fn wait_for_pipelines(uniko: &Uniko) {
    // Polls pipeline health until all queues drained
}
```

#### Test 1: Single-Hop Recall

```rust
#[tokio::test]
async fn test_single_hop_recall() {
    // Setup: Ingest 10 messages, one containing "Caroline is researching adoption"
    // Action: recall("What did Caroline research?", 8192, None)
    // Assert: ContextBundle contains a Fact or Observation with subject="Caroline",
    //         predicate contains "research", object contains "adoption"
    // Validates: Full pipeline Message -> NER -> Observation -> (optional) Consolidation -> Recall
}
```

**What it proves:** The most basic recall path works -- a single message contains the answer, and the system finds it.

#### Test 2: Temporal Reasoning

```rust
#[tokio::test]
async fn test_temporal_reasoning() {
    // Setup: Messages with date references:
    //   - "We met yesterday to discuss the project" (sent 2026-04-14)
    //   - "The deadline is in March" (sent 2026-01-15)
    //   - "I started learning Rust last week" (sent 2026-04-10)
    // Action: recall("When did we meet to discuss the project?", 8192, None)
    // Assert: Observation.observed_at reflects the correct date (2026-04-13 for "yesterday")
    // Validates: Temporal extraction from relative date references, observed_at computation
}
```

**What it proves:** The system correctly resolves temporal references relative to message timestamps and stores them as queryable dates.

#### Test 3: Adversarial Attribution

```rust
#[tokio::test]
async fn test_adversarial_attribution() {
    // Setup: Messages from two participants:
    //   - Alice: "I love hiking in the mountains"
    //   - Bob: "I ran the charity marathon last weekend"
    // Action: recall("Did Alice run the charity marathon?", 8192, None)
    // Assert: The system identifies that Bob, not Alice, ran the marathon.
    //         SENT_BY edges correctly attribute messages to participants.
    //         Abstention or correct negative for Alice.
    // Validates: Speaker attribution via SENT_BY -> Participant graph structure
}
```

**What it proves:** The system does not confuse who said what. Graph structure (SENT_BY edges) prevents false attribution -- the core advantage over flat vector stores.

#### Test 4: Multi-Hop Aggregation

```rust
#[tokio::test]
async fn test_multi_hop_aggregation() {
    // Setup: Multiple sessions with mentions of entity "Alice":
    //   - Session 1: "Alice went hiking this weekend"
    //   - Session 2: "Alice started a pottery class"
    //   - Session 3: "Alice is training for a triathlon"
    // Action: recall("What activities does Alice do?", 8192, None)
    // Assert: ContextBundle contains items from all 3 sessions.
    //         Entity "Alice" linked via MENTIONS to messages/observations across sessions.
    //         Results aggregate: hiking, pottery, triathlon all present.
    // Validates: Cross-session entity aggregation via MENTIONS edges
}
```

**What it proves:** The system aggregates information about an entity across multiple sessions, which flat retrieval systems cannot do reliably.

#### Test 5: Knowledge Update

```rust
#[tokio::test]
async fn test_knowledge_update() {
    // Setup: Contradicting messages in sequence:
    //   - Message 1: "The server runs on port 8080"
    //   - (consolidation runs, derives Fact: server port = 8080)
    //   - Message 2: "We switched the server to port 9090"
    //   - (consolidation runs, detects contradiction)
    // Assert:
    //   - Old Fact (port 8080) has BTIC hi closed (no longer valid)
    //   - New Fact (port 9090) has BTIC [now, ∞) (currently valid)
    //   - recall("what port does the server run on?") returns port 9090
    // Validates: BTIC temporal validity, contradiction detection, fact invalidation
}
```

**What it proves:** The system handles knowledge updates correctly -- when facts change, old facts are invalidated and new facts take precedence. This is a key differentiator over systems that accumulate contradictory information.

#### Test 6: Abstention

```rust
#[tokio::test]
async fn test_abstention() {
    // Setup: Ingest messages about cooking and gardening only
    // Action: recall("What is the capital of France?", 8192, None)
    // Assert:
    //   - ContextBundle.abstention == true
    //   - ContextBundle.items is empty or near-empty
    //   - No Entity, Observation, or Fact about France/capitals exists
    // Validates: Abstention detection, no hallucinated results
}
```

**What it proves:** The system correctly abstains when asked about topics not in its memory, rather than returning irrelevant results. The abstention flag (max_score < 0.15 across all phases AND items < 3) is correctly triggered.

#### Test 7: Consolidation Improvement

```rust
#[tokio::test]
async fn test_consolidation_improvement() {
    // Setup:
    //   - Record N episodes with consistent patterns
    //   - Run consolidation (trigger manually or wait for threshold)
    //   - Verify Facts derived from observations
    // Assert:
    //   - Facts exist that were derived from repeated observations
    //   - recall() returns Facts from Phase 1 (not just raw content from Phase 3)
    //   - phase1_only_pct > 0 (some recalls satisfied by Phase 1 alone)
    //   - After more consolidation, phase1_only_pct increases
    // Validates: Consolidation pipeline, phase1_only_pct as scaling signal
}
```

**What it proves:** The system improves over time. Consolidation converts raw observations into compiled facts, and the recall cascade progressively relies more on Phase 1 (fast, compiled knowledge) instead of Phase 3 (slow, raw content). This is the "compile once, query forever" principle in action.

#### Test 8: Working Memory

```rust
#[tokio::test]
async fn test_working_memory() {
    // Setup:
    //   - Create Goal "Ship MVP"
    //   - Create Task "Implement recall" under Goal
    //   - Start Session for Task
    //   - Ingest messages in session
    //   - Record episodes for task
    //   - Assert facts
    // Action: working_memory(goal_id, 50)
    // Assert:
    //   - WorkingMemoryBundle.goal matches
    //   - WorkingMemoryBundle.tasks contains "Implement recall"
    //   - WorkingMemoryBundle.sessions contains the session
    //   - WorkingMemoryBundle.messages contains ingested messages
    //   - WorkingMemoryBundle.facts contains asserted facts
    //   - WorkingMemoryBundle.entities populated from message NER
    //   - All items correctly linked via graph edges
    // Validates: Working memory traversal returns complete goal-scoped context
}
```

**What it proves:** Goal-scoped working memory assembles all relevant context by traversing the graph from Goal through Tasks, Sessions, Messages, Facts, and Entities. This is the cognitive "workspace" that no competitor offers.

#### Test 9: Offline Mode

```rust
#[tokio::test]
async fn test_offline_mode() {
    // Setup: Create Uniko with LLM disabled (feature flag or config)
    //   - Ingest messages
    //   - Wait for NER (local only, no LLM enhancement)
    //   - Wait for observations (rule-based only)
    //   - Run consolidation (Locy rules)
    // Assert:
    //   - Entities extracted (lower recall than with LLM, but functional)
    //   - Observations generated (rule-based extraction)
    //   - Facts derived via Locy stdlib rules
    //   - recall() returns results from all phases
    //   - No LLM API calls made (verify via mock or feature gate)
    // Validates: System functions without LLM (F72)
}
```

**What it proves:** The system is fully functional without an LLM. Local NER extracts entities, rule-based observation extraction finds factual statements, Locy stdlib rules execute consolidation, and recall works across all phases. Quality is lower than with LLM, but the system is operational.

#### Test 10: Scale

```rust
#[tokio::test]
async fn test_scale() {
    // Setup: Ingest 10K messages, 1K entities
    // Assert:
    //   - Message storage < 10ms per message (NF1)
    //   - Vector search < 20ms (NF2)
    //   - Hybrid search < 50ms (NF3)
    //   - Graph traversal < 5ms (NF4)
    //   - Entity extraction < 100ms (NF5)
    //   - Context bundle (compact) < 30ms (NF7)
    //   - Context bundle (all phases) < 100ms (NF8)
    //   - Episode recording < 30ms (NF10)
    //   - Tier queries < 20ms (NF11)
    //   - Drift detection < 100ms (NF12)
    //   - Working memory < 200ms (NF17)
    //   - All operations complete without errors
    // Validates: System meets all latency targets at scale (NF1-NF19 applicable)
}
```

**What it proves:** The system meets all non-functional requirements at the target scale (10K messages, 1K entities) on commodity hardware. This test must run with `--release` profile for valid timing.

---

### 12.4 -- Documentation

**Objective:** Complete Rustdoc documentation on all public types and methods in uniko-api, with module-level architecture docs and usage examples.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-api/src/lib.rs` | Rust | Crate-level doc comment with architecture overview |
| `crates/uniko-api/src/builders.rs` | Rust | Builder doc comments with usage examples |
| `ARCHITECTURE.md` | Markdown | 4-layer stack explanation for contributors |

#### Crate-Level Documentation (`lib.rs`)

```rust
//! # uniko - Cognitive Memory for AI Agents
//!
//! uniko is a cognitive memory system that gives AI agents the ability to
//! remember conversations, learn from experience, reason over accumulated
//! knowledge, and improve over time.
//!
//! ## Architecture
//!
//! uniko follows a 4-layer cognitive stack:
//!
//! - **uniko-store**: Graph CRUD, search, Locy runtime
//! - **uniko-pipes**: Step trait, circuit breaker, retry, DLQ, metrics
//! - **uniko-extract**: NER, observations, chunking, ingest steps, embedding
//! - **uniko-memory**: PipelineSystem, workers, recall cascade, consolidation, working memory, rules mgmt
//! - **uniko-cortex**: Procedures, topics, MCTS, rule induction
//! - **Layer 4 (Integration)**: FS, Shell, MCP server (uniko-fs, uniko-shell, uniko-mcp)
//!
//! This crate (`uniko-api`) is the public facade. All operations go through
//! the [`Uniko`] entry point.
//!
//! ## Quick Start
//!
//! ```rust
//! use uniko_api::{Uniko, UnikoConfig};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> uniko_api::Result<()> {
//!     // Create an in-memory instance
//!     let uniko = Uniko::in_memory(UnikoConfig::default()).await?;
//!
//!     // Create a goal
//!     let goal_id = uniko.create_goal(
//!         "Ship MVP", "Launch the first version",
//!         json!({}), json!({}), None
//!     ).await?;
//!
//!     // Ingest messages
//!     let session_id = uniko.start_session(&goal_id, "planning").await?;
//!     uniko.ingest()
//!         .in_session(&session_id)
//!         .message("Let's plan the architecture", "alice")
//!         .message("I suggest starting with the data layer", "bob")
//!         .run().await?;
//!
//!     // Recall relevant context
//!     let bundle = uniko.recall("what was discussed about architecture?", 8192, None).await?;
//!     println!("Found {} items, coverage: {}", bundle.items.len(), bundle.coverage);
//!
//!     // Get working memory for the goal
//!     let wm = uniko.working_memory(&goal_id, 50).await?;
//!     println!("Goal has {} tasks, {} messages", wm.tasks.len(), wm.messages.len());
//!
//!     uniko.shutdown().await?;
//!     Ok(())
//! }
//! ```
```

#### ARCHITECTURE.md Contents

1. **4-Layer Stack** -- diagram and description of each layer, what it does, what it depends on
2. **Strict Linear Dependency** -- why L3 calls L2 only, L2 calls L1 only, enforcement via Cargo
3. **Pipeline System** -- 8 pipelines (P1-P7d), how they chain, backpressure, retry, circuit breaker
4. **Recall Cascade** -- 3-phase architecture, coverage scoring, early exit, drift override
5. **Locy Reasoning** -- database-native logic programming, stdlib rules, rule lifecycle
6. **Memory Types** -- working, episodic, semantic, procedural, meta-memory mapped to graph nodes
7. **Data Flow** -- Message -> NER -> Observation -> Consolidation -> Fact, and how recall queries this

#### Per-Method Documentation Requirements

Every public method on `Uniko` must have:
- One-line summary
- Parameters documented with types and semantics
- Return type documented
- Error conditions listed
- Performance expectation (latency target reference)
- Example in doc comment (at least for core methods: create_goal, ingest, recall, working_memory)

---

### 12.5 -- Performance Baseline & Optimization

**Objective:** Establish baseline performance numbers for all NF targets, profile hot paths, fix bottlenecks, and document final numbers.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `tests/benchmarks/nf_benchmarks.rs` | Rust | Criterion benchmarks for all NF targets |
| `tests/benchmarks/mod.rs` | Rust | Benchmark module root |

#### NF Target Benchmarks

| Benchmark | NF | Operation | Target |
|---|---|---|---|
| `bench_nf1_store_message` | NF1 | Store message (create node + edges) | < 10ms |
| `bench_nf2_vector_search` | NF2 | Vector search (top-10) | < 20ms |
| `bench_nf3_hybrid_search` | NF3 | Hybrid search (vector + FTS + graph) | < 50ms |
| `bench_nf4_graph_traversal` | NF4 | Graph traversal (3-hop) | < 5ms |
| `bench_nf5_ner` | NF5 | Entity extraction (local NER) | < 100ms |
| `bench_nf7_compact_assembly` | NF7 | Context bundle assembly (compact-only) | < 30ms |
| `bench_nf8_full_assembly` | NF8 | Context bundle assembly (all phases) | < 100ms |
| `bench_nf9_assume` | NF9 | Single ASSUME (hypothetical reasoning) | < 200ms |
| `bench_nf10_episode_record` | NF10 | Episode recording | < 30ms |
| `bench_nf11_tier_query` | NF11 | Tier-specific queries | < 20ms |
| `bench_nf12_drift_detection` | NF12 | Drift detection step | < 100ms |
| `bench_nf17_working_memory` | NF17 | Working memory traversal | < 200ms |

#### Profiling Plan

1. **Run all benchmarks** with `cargo bench --release`
2. **Generate flamegraph** for the slowest operations using `cargo flamegraph`
3. **Identify hot paths** -- common bottlenecks:
   - Embedding computation (if not cached)
   - Graph serialization/deserialization
   - Vector distance computation
   - PPR convergence
4. **Optimize** -- potential fixes:
   - Batch embedding calls instead of per-item
   - Pre-compute and cache embeddings
   - Use SIMD for vector distance if available
   - Reduce PPR iterations when convergence detected early
   - Connection pooling for graph queries
5. **Re-run benchmarks** and verify targets met
6. **Document final numbers** -- actual measured latencies vs targets

#### Performance Measurement Context

- All benchmarks run on warm in-memory stores with < 10K nodes per label
- Commodity hardware (M-series Mac or 8-core Linux)
- `--release` profile with optimizations
- Cold-start may exceed targets by 2-5x (documented, not a failure condition)
- Each benchmark runs 100+ iterations for statistical significance

---

## Test Plan

### Integration Tests (MVP Suite)

| Test | File | What It Validates |
|---|---|---|
| `test_single_hop_recall` | `tests/integration/mvp_tests.rs` | Message -> NER -> Observation -> recall -> correct Fact/Observation found |
| `test_temporal_reasoning` | `tests/integration/mvp_tests.rs` | Date references resolved to correct timestamps, temporal queries work |
| `test_adversarial_attribution` | `tests/integration/mvp_tests.rs` | SENT_BY edges prevent false speaker attribution |
| `test_multi_hop_aggregation` | `tests/integration/mvp_tests.rs` | Cross-session entity mentions aggregated correctly |
| `test_knowledge_update` | `tests/integration/mvp_tests.rs` | BTIC invalidation on contradiction, new fact supersedes old |
| `test_abstention` | `tests/integration/mvp_tests.rs` | Query about unknown topic returns abstention=true |
| `test_consolidation_improvement` | `tests/integration/mvp_tests.rs` | phase1_only_pct > 0 and trends upward after consolidation |
| `test_working_memory` | `tests/integration/mvp_tests.rs` | Goal -> Task -> Session -> Message traversal returns complete context |
| `test_offline_mode` | `tests/integration/mvp_tests.rs` | All operations functional without LLM |
| `test_scale` | `tests/integration/mvp_tests.rs` | 10K messages, 1K entities, all NF targets met |

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_uniko_in_memory` | `uniko-api/src/lib.rs` | In-memory construction succeeds, all layers initialized |
| `test_uniko_from_path` | `uniko-api/src/lib.rs` | Disk-backed construction succeeds, data persists across restarts |
| `test_uniko_shutdown` | `uniko-api/src/lib.rs` | Graceful shutdown stops all workers without data loss |
| `test_ingest_builder` | `uniko-api/src/builders.rs` | IngestBuilder chains messages and executes ingestion |
| `test_episode_builder` | `uniko-api/src/builders.rs` | EpisodeBuilder chains fields and records episode |
| `test_reexports` | `uniko-api/src/lib.rs` | All expected types are accessible from uniko_api |
| `test_stdlib_auto_registration` | `uniko-api/src/lib.rs` | 4 stdlib rules exist after Uniko construction |

### API Ergonomics Tests

| Test | What It Validates |
|---|---|
| `test_3_line_quickstart` | Create Uniko + ingest message + recall in 3 method calls |
| `test_builder_chaining` | All builders chain without intermediate variables |
| `test_error_messages` | Error types produce human-readable messages |
| `test_default_config` | UnikoConfig::default() produces working configuration |

### Performance Benchmarks

| Benchmark | Target |
|---|---|
| All NF1-NF17 benchmarks | Meet specified targets on --release |
| API overhead measurement | `Uniko.method()` overhead vs direct `cortex.tool()` < 1ms |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Crate-level docs | `uniko-api/src/lib.rs` | Architecture overview, quick start, usage examples |
| `Uniko` struct docs | `uniko-api/src/lib.rs` | Constructor options, method summary, lifecycle |
| Builder docs | `uniko-api/src/builders.rs` | IngestBuilder, EpisodeBuilder with chaining examples |
| Re-export docs | `uniko-api/src/lib.rs` | What each re-exported type represents |
| `ARCHITECTURE.md` | Project root | 4-layer stack, pipeline system, recall cascade, Locy reasoning |
| Performance results | `ARCHITECTURE.md` or separate file | Final NF benchmark numbers vs targets |

---

## Review Checklist

- [ ] `Uniko` struct exists with `cortex` field
- [ ] `Uniko::new`, `Uniko::from_path`, `Uniko::in_memory` constructors work
- [ ] Initialization sequence: uni-db -> uniko-store -> uniko-extract -> uniko-pipes -> uniko-memory -> uniko-cortex -> stdlib rules -> pipelines
- [ ] `Uniko::shutdown` follows cancellation hierarchy (5s -> 10s -> 30s)
- [ ] All lifecycle tools re-exported as Uniko methods (9 methods)
- [ ] All knowledge tools re-exported as Uniko methods (9 methods)
- [ ] All query tools re-exported as Uniko methods (7 methods)
- [ ] RecallContextBuilder accessible via `uniko.recall_context(intent)`
- [ ] IngestBuilder supports: message, message_with_timestamp, artifact, in_session, as_participant, run
- [ ] EpisodeBuilder supports: action, outcome, state, delta, importance, entity, entities, in_session, for_task, record
- [ ] All key types re-exported from uniko-api (Message, Entity, Fact, Episode, ContextBundle, etc.)
- [ ] uniko-api depends on uniko-cortex and uniko-memory (no direct uniko-store/uniko-extract/uniko-pipes deps)
- [ ] Crate-level doc comment with architecture overview and quick start example
- [ ] ARCHITECTURE.md covers all 7 sections (layers, deps, pipelines, recall, Locy, memory types, data flow)
- [ ] Every public method on Uniko has doc comment with parameters, return, errors, performance target
- [ ] Test 1 (single-hop recall) passes
- [ ] Test 2 (temporal reasoning) passes
- [ ] Test 3 (adversarial attribution) passes
- [ ] Test 4 (multi-hop aggregation) passes
- [ ] Test 5 (knowledge update) passes
- [ ] Test 6 (abstention) passes
- [ ] Test 7 (consolidation improvement) passes
- [ ] Test 8 (working memory) passes
- [ ] Test 9 (offline mode) passes
- [ ] Test 10 (scale) passes with --release
- [ ] All NF benchmarks meet targets on --release
- [ ] Flamegraph generated for slowest operations
- [ ] Bottlenecks identified and addressed
- [ ] Final performance numbers documented
- [ ] API overhead (Uniko wrapper vs direct cortex call) < 1ms
- [ ] `cargo doc --no-deps` generates clean documentation
- [ ] No broken doc links

---

## Definition of Done

1. **uniko-api facade complete:** `Uniko` struct with constructors (`new`, `from_path`, `in_memory`), all 25 tool methods re-exported, `shutdown` implemented. Depends on `uniko-cortex` and `uniko-memory`.
2. **Builder APIs ergonomic:** IngestBuilder and EpisodeBuilder provide chainable APIs. RecallContextBuilder re-exported from Phase 10. All builders return Result types and produce correct graph state.
3. **All 10 integration tests pass:**
   - Single-hop recall finds correct Fact/Observation
   - Temporal reasoning resolves date references
   - Adversarial attribution uses SENT_BY edges correctly
   - Multi-hop aggregation crosses sessions via MENTIONS
   - Knowledge update invalidates old facts via BTIC
   - Abstention flags queries about unknown topics
   - Consolidation improvement shows phase1_only_pct trending upward
   - Working memory returns complete goal-scoped context
   - Offline mode functions without LLM
   - Scale test meets all NF targets with 10K messages
4. **Documentation complete:** Crate-level docs, per-method docs, ARCHITECTURE.md, builder examples, all doc comments present and accurate.
5. **Performance validated:** All NF benchmarks meet targets on `--release` build. Flamegraph generated. Bottlenecks addressed. Final numbers documented.
6. **API overhead minimal:** Uniko wrapper adds < 1ms overhead over direct Cortex calls.
7. **Full pipeline validated:** Message -> NER -> Observation -> Consolidation -> Fact -> Recall works end-to-end, verified by integration tests.
8. **Offline mode validated:** System operates without LLM -- local NER, rule-based observations, Locy rules, recall all functional.
9. **Scale validated:** 10K messages, 1K entities processed within latency budgets on commodity hardware.
10. **MVP is shippable:** All tests pass, all benchmarks meet targets, documentation complete, API ergonomic. The system is ready for Phase 13+ (benchmarks, MCP, Python binding) or external consumption.
