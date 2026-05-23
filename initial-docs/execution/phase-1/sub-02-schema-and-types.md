# Sub-Phase 2: Schema Definition & Test Suite

## Context

This phase defines the complete schema -- ALL 16+ node types, 35+ edge types, BTIC temporal integration, all indexes (Hash, BTree, Fulltext, Vector), and a comprehensive test suite. The schema is the single source of truth for the entire project. Every pipeline, every query, every tool interacts with these types. Getting this right means everything downstream has a solid foundation; getting it wrong means cascading rework across every crate.

The schema is organized in 8 logical layers (0-7) for grouping related node types by purpose. These are distinct from the architecture crate layers (Store, Pipes, Extract, Memory, Cortex, API, Integration). Schema layers group nodes; architecture layers group crates and capabilities.

All schema definition code lives in `uniko-store` (Layer 1) because the graph storage layer owns the schema registration. Higher layers use these types but do not define them.

## Prerequisites

- **Sub-phase 1 complete:** Workspace compiles, all crate skeletons exist, shared types/errors/config are implemented.
- uni-db API available: `Database`, `Node`, `Edge`, `Index`, `Btic`, `Vector` types accessible from `uniko-store`.
- `uniko-store/src/schema/` module directory exists (stub from Phase 1).

## Sub-phases

---

### 2.1 -- Layer 0-1 Node Types (Participants, Goals, Tasks, Sessions)

**Objective:** Define the foundational nodes for who communicates, why things happen, and when.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/schema/participants.rs` | Rust | Participant node type definition |
| `crates/uniko-store/src/schema/goals.rs` | Rust | Goal and Task node type definitions |
| `crates/uniko-store/src/schema/sessions.rs` | Rust | Session node type definition |

#### `participants.rs` -- Participant Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `participant_id` | `String` | not null | ext_id for MERGE |
| `kind` | `String` | not null | "human", "agent", "service" |
| `name` | `String` | nullable | Display name |
| `capabilities` | `Json` | nullable | Tools/skills this participant has |
| `metadata` | `Json` | nullable | Arbitrary metadata |
| `first_seen` | `DateTime` | nullable | Auto-set on creation |
| `last_seen` | `DateTime` | nullable | Updated on activity |

**Indexes:**
- `participant_id` -- Hash (unique lookup)
- `kind` -- Hash (filter by type)
- `name` -- Fulltext (search by name)

#### `goals.rs` -- Goal Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `goal_id` | `String` | not null | ext_id for MERGE |
| `title` | `String` | not null | e.g., "reduce refund cycle time by 40%" |
| `description` | `String` | nullable | Detailed specification |
| `status` | `String` | nullable | "active", "achieved", "failed", "paused" |
| `metrics` | `Json` | nullable | Target metrics and current values |
| `guardrails` | `Json` | nullable | Constraints: budget, compliance, risk |
| `owner_id` | `String` | nullable | Participant responsible |
| `created_at` | `DateTime` | nullable | |
| `deadline` | `DateTime` | nullable | |
| `completed_at` | `DateTime` | nullable | |
| `embedding` | `Vector` | nullable | Computed from title |

**Indexes:**
- `goal_id` -- Hash
- `status` -- Hash
- `title` -- Fulltext
- `embedding` -- Vector (HNSW)

**Edges:**
- `OWNED_BY`: Goal -> Participant
- `PARENT_GOAL`: Goal -> Goal (goal decomposition hierarchy)

#### `goals.rs` -- Task Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `task_id` | `String` | not null | ext_id for MERGE |
| `title` | `String` | not null | |
| `description` | `String` | nullable | |
| `status` | `String` | nullable | "pending", "in_progress", "completed", "failed", "blocked" |
| `priority` | `Float64` | nullable | |
| `created_at` | `DateTime` | nullable | |
| `completed_at` | `DateTime` | nullable | |
| `embedding` | `Vector` | nullable | Computed from title |

**Indexes:**
- `task_id` -- Hash
- `status` -- Hash
- `title` -- Fulltext
- `embedding` -- Vector (HNSW)

**Edges:**
- `PART_OF`: Task -> Goal
- `ASSIGNED_TO`: Task -> Participant
- `DEPENDS_ON`: Task -> Task
- `SUBTASK_OF`: Task -> Task

#### `sessions.rs` -- Session Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `session_id` | `String` | not null | ext_id for MERGE |
| `topic` | `String` | nullable | |
| `summary` | `String` | nullable | Generated after session ends |
| `started_at` | `DateTime` | not null | |
| `ended_at` | `DateTime` | nullable | |
| `embedding` | `Vector` | nullable | Computed: topic initially, topic + summary after end |

**Indexes:**
- `session_id` -- Hash
- `topic` -- Fulltext
- `started_at` -- BTree (range queries on time)
- `embedding` -- Vector (HNSW)

**Edges:**
- `FOR_TASK`: Session -> Task
- `FOR_GOAL`: Session -> Goal (direct link if no task)
- `PARTICIPATED_IN`: Participant -> Session (with `role: String` property -- "initiator", "responder", "observer")

---

### 2.2 -- Layer 2 Node Types (Episodic: Messages, Actions, Episodes)

**Objective:** Define the episodic memory layer -- what happened, who said/did what, when.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/schema/messages.rs` | Rust | Message node type |
| `crates/uniko-store/src/schema/actions.rs` | Rust | Action node type |
| `crates/uniko-store/src/schema/episodes.rs` | Rust | Episode node type |

#### `messages.rs` -- Message Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `message_id` | `String` | not null | ext_id for MERGE |
| `content` | `String` | not null | Message text |
| `content_type` | `String` | nullable | "text", "code", "image", "tool_result", "error", "system" |
| `timestamp` | `DateTime` | not null | When sent |
| `embedding` | `Vector` | nullable | Auto-embed from content |

**Indexes:**
- `message_id` -- Hash
- `timestamp` -- BTree
- `content` -- Fulltext
- `embedding` -- Vector (HNSW, auto-embed from content)

**Edges:**
- `SENT_BY`: Message -> Participant (with `role: String` -- "user", "assistant", "system", "tool")
- `ADDRESSED_TO`: Message -> Participant
- `IN_SESSION`: Message -> Session
- `NEXT`: Message -> Message (with `gap_ms: Int64` -- milliseconds between messages)

#### `actions.rs` -- Action Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `action_id` | `String` | not null | ext_id for MERGE |
| `action_type` | `String` | not null | "tool_call", "file_read", "file_write", "command_run", "search", "delegate", "api_call" |
| `input` | `Json` | nullable | |
| `output` | `Json` | nullable | |
| `status` | `String` | nullable | "success", "failure", "partial", "timeout" |
| `started_at` | `DateTime` | nullable | |
| `ended_at` | `DateTime` | nullable | |
| `duration_ms` | `Int64` | nullable | |
| `error` | `String` | nullable | |
| `embedding` | `Vector` | nullable | Computed: action_type + key_input_summary |

**Indexes:**
- `action_id` -- Hash
- `action_type` -- Hash
- `status` -- Hash
- `started_at` -- BTree
- `embedding` -- Vector (HNSW)

**Edges:**
- `PERFORMED_BY`: Action -> Participant
- `TRIGGERED_BY`: Action -> Message (the message that caused this action)
- `IN_SESSION`: Action -> Session
- `PRODUCED`: Action -> Artifact
- `NEXT_ACTION`: Action -> Action (action sequences within a session; provenance tracing only, not used for procedure promotion)

#### `episodes.rs` -- Episode Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `episode_id` | `String` | not null | ext_id for MERGE |
| `action_type` | `String` | not null | "investigate", "implement", "review", "conversation", "memorize", "diagnose" |
| `outcome` | `String` | nullable | "success", "failure", "partial", "inconclusive" |
| `state` | `Json` | nullable | World context at time of episode |
| `delta` | `Json` | nullable | What changed as a result |
| `importance` | `Float64` | nullable | 0.0-1.0, higher = more significant |
| `timestamp` | `DateTime` | not null | |
| `embedding` | `Vector` | nullable | Computed: extracted topic from state Json + " -- " + action_type |

**Indexes:**
- `episode_id` -- Hash
- `action_type` -- Hash
- `outcome` -- Hash
- `timestamp` -- BTree
- `embedding` -- Vector (HNSW)

**Edges:**
- `RECORDED_BY`: Episode -> Participant
- `TRIGGERED_BY`: Episode -> Message (what message/request started this)
- `FOR_TASK`: Episode -> Task
- `IN_SESSION`: Episode -> Session
- `INVOLVES`: Episode -> Action (the actions taken during this episode)
- `MENTIONS`: Episode -> Entity (entities referenced)
- `FOLLOWED_BY`: Episode -> Episode (temporal chain, with `gap_ms: Int64`)

---

### 2.3 -- Layer 3 Node Types (Artifacts & Chunks)

**Objective:** Define artifact storage with multimodal embedding support and structured chunking.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/schema/artifacts.rs` | Rust | Artifact node type (5 embedding fields) |
| `crates/uniko-store/src/schema/chunks.rs` | Rust | Chunk node type with rich metadata |

#### `artifacts.rs` -- Artifact Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `artifact_id` | `String` | not null | ext_id for MERGE |
| `kind` | `String` | not null | "file", "document", "url", "snippet", "config", "image", "audio", "video", "dataset" |
| `path` | `String` | nullable | Filesystem path, URL, or identifier |
| `content` | `String` | nullable | Text content (null for binary artifacts) |
| `mime_type` | `String` | nullable | |
| `hash` | `String` | nullable | Content hash for dedup |
| `size` | `Int64` | nullable | Bytes |
| `language` | `String` | nullable | For code: "rust", "python", etc. |
| `created_at` | `DateTime` | nullable | |
| `updated_at` | `DateTime` | nullable | |
| `text_embedding` | `Vector` | nullable | Pooled from chunk embeddings |
| `image_embedding` | `Vector` | nullable | CLIP/SigLIP |
| `audio_embedding` | `Vector` | nullable | CLAP |
| `video_embedding` | `Vector` | nullable | LanguageBind/InternVideo |
| `multimodal_embedding` | `Vector` | nullable | ImageBind/ONE-PEACE |

**Indexes:**
- `artifact_id` -- Hash
- `path` -- BTree (unique)
- `kind` -- Hash
- `content` -- Fulltext (only covers text artifacts; null for binary)
- `language` -- Hash
- `mime_type` -- Hash

**Vector Indexes (each HNSW):**
- `text_embedding` -- pooled from chunks, not auto-embed
- `image_embedding` -- computed by vision model
- `audio_embedding` -- computed by audio model
- `video_embedding` -- computed by video model
- `multimodal_embedding` -- computed by unified model

**Edges:**
- `CREATED_BY`: Artifact -> Action (which action produced this artifact)
- `MODIFIED_BY`: Artifact -> Action (with `diff_summary: String`)

#### `chunks.rs` -- Chunk Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `chunk_id` | `String` | not null | Deterministic: `{parent_id}:{index}` |
| `text` | `String` | not null | Chunk text content |
| `index` | `Int64` | nullable | Position within parent |
| `start` | `Int64` | nullable | Byte/char offset in parent |
| `end` | `Int64` | nullable | Byte/char offset end |
| `token_count` | `Int64` | nullable | |
| `chunk_type` | `String` | nullable | "paragraph", "sentence_group", "function", "class", "table", "speaker_turn", "scene", "heading_section", "block" |
| `language` | `String` | nullable | For code: "rust", "python", etc. |
| `symbol_name` | `String` | nullable | For code: function/class name |
| `speaker` | `String` | nullable | For audio: speaker attribution |
| `heading` | `String` | nullable | For docs: section heading context |
| `mime_type` | `String` | nullable | Source content type |
| `embedding` | `Vector` | nullable | Auto-embed from text |

**Indexes:**
- `text` -- Fulltext
- `chunk_type` -- Hash
- `language` -- Hash
- `symbol_name` -- Hash
- `speaker` -- Hash
- `embedding` -- Vector (HNSW, auto-embed from text)

**Edges:**
- `HAS_CHUNK`: Artifact -> Chunk (with `index: Int64` property)
- `HAS_CHUNK`: Message -> Chunk (for long messages > 1024 tokens, with `index: Int64` property)

---

### 2.4 -- Layer 4 Node Types (Semantic: Entities, Observations, Facts, Topics, Summaries)

**Objective:** Define the semantic memory layer -- extracted and consolidated knowledge.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/schema/entities.rs` | Rust | Entity node type |
| `crates/uniko-store/src/schema/observations.rs` | Rust | Observation node type |
| `crates/uniko-store/src/schema/facts.rs` | Rust | Fact node type with BTIC valid_at |
| `crates/uniko-store/src/schema/topics.rs` | Rust | Topic node type |
| `crates/uniko-store/src/schema/summaries.rs` | Rust | Summary node type |

#### `entities.rs` -- Entity Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `entity_id` | `String` | not null | ext_id for MERGE |
| `name` | `String` | not null | Canonical name |
| `entity_type` | `String` | nullable | "person", "place", "org", "concept", "project", "tool", "event", "date" |
| `first_seen` | `DateTime` | nullable | |
| `last_seen` | `DateTime` | nullable | |
| `frequency` | `Int64` | nullable | Mention count |
| `confidence` | `Float64` | nullable | |
| `embedding` | `Vector` | nullable | Computed: `name + " (" + entity_type + ")"` |

**Indexes:**
- `entity_id` -- Hash
- `name` -- Hash + Fulltext (dual index for exact and fuzzy lookup)
- `entity_type` -- Hash
- `embedding` -- Vector (HNSW)

**Edges (MENTIONS from 4 source types):**
- `MENTIONS`: Message -> Entity (with `count: Int64`)
- `MENTIONS`: Chunk -> Entity (with `count: Int64`)
- `MENTIONS`: Action -> Entity
- `MENTIONS`: Artifact -> Entity (with `count: Int64`)

#### `observations.rs` -- Observation Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `observation_id` | `String` | not null | ext_id for MERGE |
| `content` | `String` | not null | The observation text |
| `subject` | `String` | nullable | Who/what it's about |
| `observed_at` | `DateTime` | nullable | When in the real world |
| `confidence` | `Float64` | nullable | |
| `embedding` | `Vector` | nullable | Auto-embed from content |

**Indexes:**
- `observation_id` -- Hash
- `content` -- Fulltext
- `subject` -- Hash + Fulltext
- `embedding` -- Vector (HNSW, auto-embed from content)

**Edges:**
- `OBSERVED_IN`: Observation -> Message (source message)
- `OBSERVED_IN`: Observation -> Chunk (observations from artifact chunks)
- `OBSERVED_DURING`: Observation -> Episode (which episode this was observed in)
- `ABOUT`: Observation -> Entity

#### `facts.rs` -- Fact Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `fact_id` | `String` | not null | ext_id for MERGE |
| `subject` | `String` | not null | |
| `predicate` | `String` | not null | |
| `object` | `String` | nullable | |
| `confidence` | `Float64` | nullable | Laplace smoothing: `count / (count + 2)` |
| `observation_count` | `Int64` | nullable | |
| `valid_at` | `Btic` | nullable | Temporal validity interval `[lo, hi)` |
| `source_rule` | `String` | nullable | Which Rule derived this fact |
| `visibility` | `String` | nullable | "agent" (default, scoped) or "global" (shared) |
| `embedding` | `Vector` | nullable | Computed: `subject + " " + predicate + " " + object` |

**Indexes:**
- `fact_id` -- Hash
- `subject` -- Hash + Fulltext
- `predicate` -- Hash
- `confidence` -- BTree (range queries on reliability)
- `embedding` -- Vector (HNSW)

**Edges:**
- `SUPPORTED_BY`: Fact -> Observation (with `weight: Float64`)
- `DERIVED_BY`: Fact -> Rule (which consolidation rule created this)
- `DERIVED_FROM`: Fact -> Episode (episodes whose observations provided evidence)
- `INVALIDATES`: Fact -> Fact (with `reason: String`)
- `ABOUT`: Fact -> Entity
- `SHARED_FROM`: Fact -> Fact (with `shared_by: String` participant_id, `shared_at: DateTime`)

#### `topics.rs` -- Topic Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `topic_id` | `String` | not null | ext_id for MERGE |
| `name` | `String` | not null | |
| `summary` | `String` | nullable | Generated summary of the cluster |
| `entity_count` | `Int64` | nullable | |
| `fact_count` | `Int64` | nullable | |
| `embedding` | `Vector` | nullable | Auto-embed from `name + summary` |

**Indexes:**
- `topic_id` -- Hash
- `name` -- Fulltext
- `embedding` -- Vector (HNSW)

**Edges:**
- `BELONGS_TO`: Entity -> Topic
- `BELONGS_TO`: Fact -> Topic

#### `summaries.rs` -- Summary Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `summary_id` | `String` | not null | ext_id for MERGE |
| `text` | `String` | not null | Summary content |
| `level` | `String` | nullable | "session", "task", "goal", "artifact", "entity", "topic" |
| `generated_at` | `DateTime` | nullable | |
| `embedding` | `Vector` | nullable | Auto-embed from text |

**Indexes:**
- `embedding` -- Vector (HNSW, auto-embed from text)

**Edges (SUMMARIZES to 6 targets):**
- `SUMMARIZES`: Summary -> Session
- `SUMMARIZES`: Summary -> Task
- `SUMMARIZES`: Summary -> Goal
- `SUMMARIZES`: Summary -> Artifact
- `SUMMARIZES`: Summary -> Entity
- `SUMMARIZES`: Summary -> Topic

---

### 2.5 -- Layer 5-7 Node Types (Procedural, Meta, Organization)

**Objective:** Define procedural memory, meta-memory tracking, and organizational grouping.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/schema/procedures.rs` | Rust | Procedure node type |
| `crates/uniko-store/src/schema/rules.rs` | Rust | Rule node type with lifecycle |
| `crates/uniko-store/src/schema/consolidation.rs` | Rust | ConsolidationCycle and DeadLetter nodes |
| `crates/uniko-store/src/schema/organization.rs` | Rust | Organization and Team nodes |

#### `procedures.rs` -- Procedure Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `procedure_id` | `String` | not null | ext_id for MERGE |
| `name` | `String` | not null | |
| `description` | `String` | nullable | |
| `steps` | `Json` | nullable | Ordered list of action_types with parameters |
| `preconditions` | `Json` | nullable | State conditions for applicability |
| `precondition_rule` | `String` | nullable | Locy WHERE fragment for matching |
| `parameters` | `Json` | nullable | Configurable inputs |
| `effectiveness` | `Float64` | nullable | |
| `use_count` | `Int64` | nullable | |
| `success_count` | `Int64` | nullable | |
| `failure_count` | `Int64` | nullable | |
| `avg_outcome_delta` | `Json` | nullable | Average outcome changes |
| `status` | `String` | nullable | "candidate", "active", "deprecated" |
| `created_at` | `DateTime` | nullable | |
| `last_used_at` | `DateTime` | nullable | |
| `embedding` | `Vector` | nullable | Computed: `name + ": " + description[:200]` |

**Indexes:**
- `procedure_id` -- Hash
- `name` -- Fulltext
- `status` -- Hash
- `embedding` -- Vector (HNSW)

**Edges:**
- `DERIVED_FROM`: Procedure -> Action (specific actions forming steps)
- `DERIVED_FROM`: Procedure -> Episode (episodes where pattern was observed)
- `OPERATES_ON`: Procedure -> Entity
- `USED_IN`: Procedure -> Task (tasks where procedure was applied)

#### `rules.rs` -- Rule Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `rule_id` | `String` | not null | ext_id for MERGE |
| `name` | `String` | not null | |
| `source` | `String` | nullable | Locy source code |
| `natural_language` | `String` | nullable | Human-readable description |
| `source_type` | `String` | nullable | "stdlib", "authored", "induced" |
| `status` | `String` | nullable | "active", "demoted", "pruned", "superseded" |
| `version` | `Int64` | nullable | |
| `confidence` | `Float64` | nullable | precision * 0.4 + recall * 0.3 + novelty * 0.3 |
| `precision` | `Float64` | nullable | |
| `recall` | `Float64` | nullable | |
| `coverage` | `Int64` | nullable | Episodes covered by this rule |
| `created_at` | `DateTime` | nullable | |
| `validated_at` | `DateTime` | nullable | |
| `last_scored_at` | `DateTime` | nullable | |

**Lifecycle:** `Created -> Active` (direct for stdlib/authored), `Created -> Candidate -> Active` (for induced, after validation). `Active -> Demoted` (confidence < 0.40) `-> Pruned` (90 days inactive, terminal). `Active -> Superseded` (terminal, replaced by newer rule). Stdlib rules exempt from demotion/pruning/supersession. Confidence decay: `stored_confidence * (0.95^missed_cycles)`. Re-promotion: confidence > 0.60 (hysteresis).

**Indexes:**
- `rule_id` -- Hash
- `name` -- Hash
- `status` -- Hash
- `source_type` -- Hash

**Edges:**
- `SUPERSEDES`: Rule -> Rule (newer rule replaces older)
- `DERIVED_BY`: Fact -> Rule (which rule derived this fact -- defined on Fact side)
- `COVERS`: Rule -> Episode (with `correct: Int64` -- 1=true, 0=false)

#### `consolidation.rs` -- ConsolidationCycle Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `cycle_id` | `String` | not null | ext_id for MERGE |
| `agent_id` | `String` | not null | |
| `started_at` | `DateTime` | nullable | |
| `completed_at` | `DateTime` | nullable | |
| `observations_processed` | `Int64` | nullable | |
| `episodes_involved` | `Int64` | nullable | |
| `facts_created` | `Int64` | nullable | |
| `facts_reinforced` | `Int64` | nullable | |
| `facts_invalidated` | `Int64` | nullable | |
| `procedures_promoted` | `Int64` | nullable | |
| `drift_alerts` | `Int64` | nullable | |

**Indexes:**
- `cycle_id` -- Hash
- `agent_id` -- Hash
- `started_at` -- BTree

**Edges:**
- `PROCESSED`: ConsolidationCycle -> Observation
- `INVOLVED`: ConsolidationCycle -> Episode
- `CREATED`: ConsolidationCycle -> Fact
- `INVALIDATED`: ConsolidationCycle -> Fact
- `PROMOTED`: ConsolidationCycle -> Procedure
- `APPLIED_RULE`: ConsolidationCycle -> Rule

#### `consolidation.rs` -- DeadLetter Node

**Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `step` | `String` | nullable | Which pipeline step failed |
| `error` | `String` | nullable | Error message |
| `node_ref` | `Int64` | nullable | The node that couldn't be processed |
| `retry_count` | `Int64` | nullable | How many times retried |
| `max_retries` | `Int64` | nullable | Default: 3 |
| `next_retry_at` | `DateTime` | nullable | Computed from backoff |
| `created_at` | `DateTime` | nullable | |

#### `organization.rs` -- Organization & Team Nodes

**Organization Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `org_id` | `String` | not null | ext_id for MERGE |
| `name` | `String` | nullable | |

**Team Properties:**

| Property | Type | Constraints | Notes |
|---|---|---|---|
| `team_id` | `String` | not null | ext_id for MERGE |
| `name` | `String` | nullable | |
| `purpose` | `String` | nullable | |

**Edges:**
- `MEMBER_OF`: Participant -> Organization (with `role: String`, `joined_at: DateTime`)
- `PART_OF_TEAM`: Participant -> Team
- `TEAM_IN_ORG`: Team -> Organization

---

### 2.6 -- BTIC Integration & Temporal Helpers

**Objective:** Provide Rust helper functions for working with uni-db's native BTIC (Binary Temporal Interval Composite) type on Fact nodes.

#### File to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/schema/btic.rs` | Rust | BTIC helper functions |

#### Functions

```rust
/// Create an active BTIC interval: [observed_at, infinity).
/// Certainty is "approximate" by default (< 10 observations).
pub fn btic_active(observed_at: DateTime<Utc>) -> Btic;

/// Invalidate a fact by closing the hi bound of its BTIC interval.
/// Sets hi = now, making the fact valid only in [lo, now).
pub fn btic_invalidate(fact_valid_at: &mut Btic, now: DateTime<Utc>);

/// Check if a point in time falls within a BTIC interval.
/// Returns true if lo <= point < hi.
pub fn btic_contains(interval: &Btic, point: DateTime<Utc>) -> bool;

/// Check if two BTIC intervals overlap (Allen's algebra).
/// Two intervals overlap if their intersection is non-empty.
pub fn btic_overlaps(a: &Btic, b: &Btic) -> bool;

/// Check if interval a ends before interval b begins (Allen's algebra).
pub fn btic_before(a: &Btic, b: &Btic) -> bool;

/// Upgrade certainty from approximate to definite.
/// Called when observation_count crosses the 10-observation threshold.
pub fn btic_upgrade_certainty(interval: &mut Btic);

/// Get the granularity of an interval's bounds (day, month, year, etc.).
pub fn btic_granularity(interval: &Btic) -> (Granularity, Granularity);
```

**Certainty rules:**
- `approximate` if < 10 supporting observations
- `definite` if >= 10 supporting observations

**Granularity:**
- Per-bound granularity: "sometime in 2022" (year) vs "on 7 May 2023" (day)
- Finest available granularity from observation timestamps

---

### 2.7 -- Schema Registration Module

**Objective:** Provide a single entry point that registers all node types, edge types, and indexes with the uni-db database instance.

#### File to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/schema/mod.rs` | Rust | Module root + `register_schema()` function |

#### Function Signature

```rust
/// Register the complete uniko schema with the database.
///
/// This function is idempotent -- safe to call multiple times.
/// It registers all 16+ node types, 35+ edge types, and all indexes.
///
/// Vector index dimensions validated:
///   - 384 (lightweight models)
///   - 512 (mid-range models)
///   - 768 (standard models like all-MiniLM-L6-v2)
///   - 1024 (larger models)
///
/// # Errors
/// Returns `UnikoError::Schema` if registration fails.
pub fn register_schema(db: &Database) -> Result<()>;
```

#### Module Declarations

```rust
pub mod participants;
pub mod goals;
pub mod sessions;
pub mod messages;
pub mod actions;
pub mod episodes;
pub mod artifacts;
pub mod chunks;
pub mod entities;
pub mod observations;
pub mod facts;
pub mod topics;
pub mod summaries;
pub mod procedures;
pub mod rules;
pub mod consolidation;
pub mod organization;
pub mod btic;
```

**Requirements:**
- Idempotent: calling `register_schema()` twice has no effect the second time
- Validates vector index dimensions match expected model output sizes
- Registers indexes in correct order (node types first, then indexes)
- Returns clear error messages if any registration fails

---

### 2.8 -- Comprehensive Test Suite

**Objective:** Build a test suite that validates every node type, edge type, index, and BTIC operation against the schema specification.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/tests/schema_tests.rs` | Rust | Integration tests for schema |
| `crates/uniko-store/tests/btic_tests.rs` | Rust | BTIC-specific tests |
| `crates/uniko-store/tests/schema_completeness.rs` | Rust | Programmatic spec cross-reference |

#### CRUD Tests (per node type -- 16 test groups)

For each of the 16+ node types (Participant, Goal, Task, Session, Message, Action, Episode, Artifact, Chunk, Entity, Observation, Fact, Topic, Summary, Procedure, Rule, ConsolidationCycle, DeadLetter, Organization, Team):

1. **Create** with all properties set, verify node exists
2. **Read** back all properties, verify values match
3. **Update** mutable properties, verify changes persisted
4. **Delete** node, verify removal
5. **MERGE/upsert** via ext_id, verify idempotency

#### Edge Tests (per edge type -- 35+ test groups)

For each edge type:

1. **Create** edge between correct source/target node types
2. **Verify** source and target nodes are correct types
3. **Test** edge properties (role, gap_ms, count, weight, reason, etc.)
4. **Delete** edge, verify removal
5. **Query** edges by direction and type

**Edge types to test:**
- OWNED_BY, PARENT_GOAL, PART_OF, ASSIGNED_TO, DEPENDS_ON, SUBTASK_OF
- FOR_TASK, FOR_GOAL, PARTICIPATED_IN (role)
- SENT_BY (role), ADDRESSED_TO, IN_SESSION, NEXT (gap_ms)
- PERFORMED_BY, TRIGGERED_BY, PRODUCED, NEXT_ACTION
- RECORDED_BY, INVOLVES, FOLLOWED_BY (gap_ms), MENTIONS (count from Episode)
- HAS_CHUNK (index, from both Artifact and Message)
- CREATED_BY, MODIFIED_BY (diff_summary)
- MENTIONS (count, from Message, Chunk, Action, Artifact)
- OBSERVED_IN, OBSERVED_DURING, ABOUT (from Observation)
- SUPPORTED_BY (weight), DERIVED_BY, DERIVED_FROM, INVALIDATES (reason), SHARED_FROM (shared_by, shared_at)
- BELONGS_TO
- SUMMARIZES (to 6 targets)
- DERIVED_FROM (Procedure -> Action, Procedure -> Episode), OPERATES_ON, USED_IN
- SUPERSEDES, COVERS (correct)
- PROCESSED, INVOLVED, CREATED, INVALIDATED, PROMOTED, APPLIED_RULE
- MEMBER_OF (role, joined_at), PART_OF_TEAM, TEAM_IN_ORG

#### Index Tests

| Index Type | Test | Expected Behavior |
|---|---|---|
| Hash | Lookup by exact value | Returns matching nodes in O(1) |
| BTree | Range query (e.g., started_at between dates) | Returns ordered results |
| Fulltext | Search by keyword/phrase | Returns BM25-ranked results |
| Vector | KNN query with embedding | Returns top-k by cosine similarity |

**Specific index tests:**
- Hash: `participant_id`, `goal_id`, `entity_id`, `fact_id`, etc.
- BTree: `started_at` on Session, `timestamp` on Message/Episode, `confidence` on Fact
- Fulltext: `content` on Message, `text` on Chunk, `name` on Entity, `title` on Goal/Task
- Vector: `embedding` on Message, Chunk, Entity, Fact, Episode, Observation, Goal, Task, Session, Topic, Summary, Procedure, plus 5 embedding fields on Artifact

#### BTIC Tests

| Test | What It Validates |
|---|---|
| `test_btic_active_creation` | `btic_active(date)` creates `[date, infinity)` with approximate certainty |
| `test_btic_invalidation` | `btic_invalidate()` closes hi bound to now |
| `test_btic_contains_active` | Active fact contains current time |
| `test_btic_contains_invalidated` | Invalidated fact contains time within `[lo, hi)` but not after hi |
| `test_btic_contains_before_start` | Fact does not contain time before lo |
| `test_btic_overlaps_same_period` | Two concurrent facts overlap |
| `test_btic_overlaps_sequential` | Two non-overlapping facts do not overlap |
| `test_btic_before` | Older invalidated fact is before newer fact |
| `test_btic_certainty_upgrade` | Certainty changes from approximate to definite at 10 observations |
| `test_btic_granularity` | Granularity correctly reflects observation precision |

#### Property-Based Tests (proptest)

| Test | What It Validates |
|---|---|
| `proptest_valid_properties` | Random valid property combinations create valid nodes |
| `proptest_invalid_properties` | Missing required properties (e.g., null message_id) are rejected |
| `proptest_btic_containment` | For any valid interval and point, containment is correct |
| `proptest_btic_allen_operators` | Allen's operators are mutually consistent |
| `proptest_roundtrip` | Create node -> read back -> all properties match |

#### Schema Completeness Test

| Test | What It Validates |
|---|---|
| `test_all_node_types_registered` | Programmatic check: all 16+ node types exist in registered schema |
| `test_all_edge_types_registered` | Programmatic check: all 35+ edge types exist |
| `test_all_indexes_registered` | Programmatic check: all Hash, BTree, Fulltext, Vector indexes exist |
| `test_schema_matches_spec` | Cross-reference against schema-v3.md constants |

#### Cross-Layer Edge Tests

| Test | What It Validates |
|---|---|
| `test_episode_mentions_entity` | L3 Episode -> L2 Entity edge works across architecture layers |
| `test_observation_observed_during_episode` | L2 Observation -> L3 Episode edge works |
| `test_fact_derived_from_episode` | L4 Fact -> L3 Episode edge works |
| `test_summary_summarizes_session` | Summary -> Session edge works |

---

## Test Plan

### Coverage Targets

- **Node types:** 100% coverage (every node type has CRUD + property tests)
- **Edge types:** 100% coverage (every edge type has creation + property tests)
- **Indexes:** 100% coverage (every index type tested for its operation)
- **BTIC operations:** All 6 helper functions tested
- **Property-based:** At least 5 proptest scenarios
- **Schema completeness:** Programmatic cross-reference passes

### Test Execution

```bash
# Run all schema tests
poetry run -- cargo nextest run -n auto -p uniko-store --test schema_tests
poetry run -- cargo nextest run -n auto -p uniko-store --test btic_tests
poetry run -- cargo nextest run -n auto -p uniko-store --test schema_completeness
```

### Test Database

Each test function creates a fresh in-memory uni-db instance, registers the schema, performs operations, and drops the instance. No shared state between tests.

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Schema module doc | `schema/mod.rs` | Overview of all layers, how to use `register_schema()` |
| Per-node-type docs | Each `schema/*.rs` file | Properties table, index list, edge list, usage examples |
| BTIC module doc | `schema/btic.rs` | BTIC concept explanation, Allen's operators, examples |
| Property constraints | Each struct definition | Which fields are required vs optional, valid values |
| Edge relationship docs | Each edge definition | Source type, target type, properties, cardinality |

---

## Review Checklist

- [ ] All 16+ node types defined with properties matching schema-v3.md exactly
- [ ] All 35+ edge types defined with correct source/target types and properties
- [ ] Participant: participant_id (Hash), kind (Hash), name (Fulltext)
- [ ] Goal: goal_id (Hash), status (Hash), title (Fulltext), embedding (Vector)
- [ ] Task: task_id (Hash), status (Hash), title (Fulltext), embedding (Vector)
- [ ] Session: session_id (Hash), topic (Fulltext), started_at (BTree), embedding (Vector)
- [ ] Message: message_id (Hash), timestamp (BTree), content (Fulltext), embedding (Vector, auto-embed)
- [ ] Action: action_id (Hash), action_type (Hash), status (Hash), started_at (BTree), embedding (Vector)
- [ ] Episode: episode_id (Hash), action_type (Hash), outcome (Hash), timestamp (BTree), embedding (Vector)
- [ ] Artifact: artifact_id (Hash), path (BTree unique), kind (Hash), content (Fulltext), language (Hash), mime_type (Hash), plus 5 vector indexes
- [ ] Chunk: text (Fulltext), chunk_type (Hash), language (Hash), symbol_name (Hash), speaker (Hash), embedding (Vector, auto-embed)
- [ ] Entity: entity_id (Hash), name (Hash + Fulltext), entity_type (Hash), embedding (Vector)
- [ ] Observation: observation_id (Hash), content (Fulltext), subject (Hash + Fulltext), embedding (Vector, auto-embed)
- [ ] Fact: fact_id (Hash), subject (Hash + Fulltext), predicate (Hash), confidence (BTree), embedding (Vector); valid_at is Btic type
- [ ] Topic: topic_id (Hash), name (Fulltext), embedding (Vector)
- [ ] Summary: embedding (Vector, auto-embed from text)
- [ ] Procedure: procedure_id (Hash), name (Fulltext), status (Hash), embedding (Vector)
- [ ] Rule: rule_id (Hash), name (Hash), status (Hash), source_type (Hash)
- [ ] ConsolidationCycle: cycle_id (Hash), agent_id (Hash), started_at (BTree)
- [ ] Organization: org_id (Hash)
- [ ] Team: team_id (Hash)
- [ ] BTIC helpers: btic_active, btic_invalidate, btic_contains, btic_overlaps, btic_before, btic_upgrade_certainty
- [ ] `register_schema()` is idempotent
- [ ] Vector index dimensions validated (384, 512, 768, 1024)
- [ ] CRUD tests pass for every node type
- [ ] Edge tests pass for every edge type
- [ ] Index tests pass for every index (Hash, BTree, Fulltext, Vector)
- [ ] BTIC tests pass for all helper functions
- [ ] Property-based tests pass
- [ ] Schema completeness test passes (programmatic cross-reference)
- [ ] Cross-layer edge tests pass

---

## Definition of Done

1. **Schema complete:** All 16+ node types, 35+ edge types, and all indexes are defined in code and match `schema-v3.md` exactly.
2. **BTIC functional:** All temporal helper functions work correctly. Active facts contain current time. Invalidated facts have closed intervals. Allen's operators produce correct results.
3. **Registration idempotent:** Calling `register_schema()` twice on the same database produces no errors and no duplicate types/indexes.
4. **CRUD validated:** Every node type can be created, read, updated, deleted, and upserted via MERGE semantics.
5. **Edges validated:** Every edge type connects the correct source and target node types with correct properties.
6. **Indexes validated:** Hash indexes support O(1) lookup. BTree indexes support range queries. Fulltext indexes support keyword search. Vector indexes support KNN queries.
7. **Property-based tests pass:** Random valid inputs produce valid nodes; random invalid inputs are rejected.
8. **Schema completeness test passes:** Programmatic check confirms all types and indexes from the spec are registered.
9. **Zero test failures:** All tests pass with `cargo nextest run -n auto`.
10. **Documentation complete:** Every node type, edge type, index, and BTIC helper has doc comments explaining purpose and constraints.
