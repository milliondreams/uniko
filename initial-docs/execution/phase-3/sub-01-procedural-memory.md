# Phase 13: Procedural Memory -- P5 (Procedure Promotion) & P6 (Topic Detection)

## Context

This phase implements the procedural memory layer -- the system's ability to learn reusable playbooks from experience and organize knowledge into coherent clusters. It comprises two consolidation pipelines:

**Pipeline 5 (Procedure Promotion)** analyzes Episode FOLLOWED_BY chains to detect recurring action sequences that consistently succeed. When a sequence appears frequently enough with sufficient effectiveness, it is promoted to a Procedure node -- a reusable playbook that agents can reference and apply. Procedures have a full lifecycle: candidate (detected but unproven) -> active (validated through subsequent use) -> deprecated (stale or ineffective).

**Pipeline 6 (Topic Detection)** runs community detection on an entity co-occurrence graph to discover natural knowledge clusters. Entities that frequently appear together in messages, observations, episodes, and facts are grouped into Topic nodes. Topics organize the semantic memory layer into navigable clusters, improving recall quality and enabling agents to reason about domains rather than individual facts.

Together, P5 and P6 implement requirements F40-F43 from the spec:
- F40 (DIF): Cluster entities into Topics via community detection on co-occurrence graph
- F41 (DIF): Promote recurring action sequences into Procedure nodes (candidate -> active -> deprecated)
- F42 (DIF): Track procedure effectiveness (success/failure counts, use_count)
- F43 (DIF): Support procedure precondition matching via Locy WHERE fragments

These pipelines run as part of the ConsolidationWorker's step chain (Phase 4 infrastructure). They execute after P4 (fact consolidation) in each consolidation cycle, using the same trigger logic (threshold or timer).

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 12 (MVP complete) | Complete | All MVP pipelines operational, schema registered, agent tools functional |
| Episodes (from agent tools) | Available | `record_episode` tool creates Episode nodes with FOLLOWED_BY chains, action_type, outcome, state, delta |
| Entities (from P2 NER) | Available | Entity nodes with MENTIONS edges from Messages, Chunks, Actions, Artifacts |
| Facts (from P4 Consolidation) | Available | Fact nodes with subject/predicate/object, linked to Entities via ABOUT edges |
| Procedure schema (Phase 2) | Defined | Procedure node type with all properties, indexes, and edges already registered |
| Topic schema (Phase 2) | Defined | Topic node type with all properties, indexes, and edges already registered |
| ConsolidationWorker (Phase 4) | Operational | Step chain executor, consolidation triggers, circuit breaker, retry policy |
| Locy runtime (Phase 3) | Operational | Rule execution, MATCH/FOLD/YIELD queries for sequence detection |
| LLM provider (optional) | Available | Used for Procedure naming and Topic naming/summarization; falls back to derived names |

## Sub-phases

---

### 13.1 -- P5: Sequence Detection

**Objective:** Query the Episode FOLLOWED_BY graph to identify recurring action sequences that represent candidate procedures.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/consolidation/procedures.rs` | New | Sequence detection logic, `ActionSequence` struct |

#### Structs and Functions

```rust
/// A detected recurring action sequence from Episode FOLLOWED_BY chains.
pub struct ActionSequence {
    /// Ordered list of action_types forming the sequence.
    pub action_types: Vec<String>,
    /// How many times this exact sequence was observed.
    pub frequency: u32,
    /// Success rate: success_count / total_count.
    pub effectiveness: f64,
    /// Source episode IDs where this sequence was observed.
    pub episodes: Vec<EpisodeId>,
    /// Common entities across the episodes in this sequence.
    pub common_entities: Vec<EntityId>,
}
```

- `async fn detect_sequences(kb: &KnowledgeBase, agent_id: &str) -> Result<Vec<ActionSequence>>` -- Main entry point. Queries the Episode graph for FOLLOWED_BY chains, groups by action_type sequence, computes frequency and effectiveness, filters by thresholds.

#### Query Logic

The sequence detector uses the `sequence_detector` stdlib Locy rule (Rule 3 from the spec) as the foundation, extended to handle chains of length >= 2:

**2-step sequences:**
```cypher
MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode)
MATCH (e1)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id})
MATCH (e2)-[:RECORDED_BY]->(p)
WITH e1.action_type AS step1, e2.action_type AS step2,
     collect(e1.episode_id) AS ep1_ids, collect(e2.episode_id) AS ep2_ids,
     count(*) AS freq,
     sum(CASE WHEN e1.outcome = 'success' AND e2.outcome = 'success' THEN 1 ELSE 0 END) AS success_count
WHERE freq >= 5
RETURN step1, step2, freq, toFloat(success_count) / freq AS effectiveness, ep1_ids, ep2_ids
```

**3-step sequences:**
```cypher
MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode)-[:FOLLOWED_BY]->(e3:Episode)
MATCH (e1)-[:RECORDED_BY]->(p:Participant {participant_id: $agent_id})
MATCH (e2)-[:RECORDED_BY]->(p)
MATCH (e3)-[:RECORDED_BY]->(p)
WITH e1.action_type AS step1, e2.action_type AS step2, e3.action_type AS step3,
     count(*) AS freq,
     sum(CASE WHEN e1.outcome = 'success' AND e2.outcome = 'success' AND e3.outcome = 'success' THEN 1 ELSE 0 END) AS success_count
WHERE freq >= 5
RETURN step1, step2, step3, freq, toFloat(success_count) / freq AS effectiveness
```

#### Filtering Thresholds

| Criterion | Threshold | Rationale |
|---|---|---|
| Minimum frequency | >= 5 occurrences | Avoid promoting one-off sequences |
| Minimum effectiveness | > 0.7 (70% success rate) | Only promote patterns that actually work |
| Minimum distinct action_types | >= 2 | Single-step sequences are not procedures |
| Exclude existing procedures | Skip if active Procedure already exists for this sequence | Avoid duplicates |

#### Common Entity Extraction

For each detected sequence, query the MENTIONS edges from the source Episodes to find entities that appear in 60%+ of the instances:

```cypher
MATCH (e:Episode)-[:MENTIONS]->(ent:Entity)
WHERE e.episode_id IN $episode_ids
WITH ent, count(DISTINCT e) AS mention_count
WHERE mention_count >= $threshold  // 60% of episode count
RETURN ent.entity_id, ent.name
```

These common entities become the OPERATES_ON targets for the created Procedure.

---

### 13.2 -- P5: Candidate Creation & Precondition Extraction

**Objective:** Create Procedure nodes from detected sequences, link them to source Episodes and common Entities, and extract preconditions from Episode state fields.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/consolidation/procedure_lifecycle.rs` | New | Procedure creation, precondition extraction, lifecycle management |

#### Structs and Functions

```rust
/// Create a candidate Procedure from a detected ActionSequence.
pub async fn create_candidate_procedure(
    kb: &KnowledgeBase,
    sequence: &ActionSequence,
    agent_id: &str,
    llm: Option<&LlmProvider>,
) -> Result<ProcedureId>;

/// Extract preconditions from the state fields of successful Episode instances.
pub fn extract_preconditions(
    episodes: &[Episode],
    min_presence_ratio: f64,  // default: 0.8
) -> Option<PreconditionRule>;

/// A Locy WHERE fragment representing preconditions for a procedure.
pub struct PreconditionRule {
    /// The Locy WHERE clause fragment.
    pub locy_fragment: String,
    /// Human-readable description.
    pub description: String,
    /// State keys and their expected values.
    pub conditions: Vec<(String, serde_json::Value)>,
}
```

#### Procedure Node Creation

When `create_candidate_procedure` is called:

1. **Generate name:** If LLM available, prompt: "Given action sequence [step1, step2, step3], generate a concise kebab-case name (e.g., 'investigate-then-implement')." Fallback: join action_types with hyphens, e.g., `"investigate-implement-review"`.

2. **Generate description:** If LLM available, prompt for a one-sentence description. Fallback: "Procedure derived from {frequency} occurrences of {step1} -> {step2} [-> {step3}] with {effectiveness*100}% effectiveness."

3. **Create Procedure node:**
   - `procedure_id`: generated UUID v7
   - `name`: LLM-generated or derived
   - `description`: LLM-generated or derived
   - `steps`: JSON array of `[{"action_type": "investigate", "order": 0}, {"action_type": "implement", "order": 1}, ...]`
   - `preconditions`: JSON from precondition extraction (or null)
   - `precondition_rule`: Locy WHERE fragment (or null)
   - `effectiveness`: from ActionSequence
   - `use_count`: 0
   - `success_count`: 0
   - `failure_count`: 0
   - `status`: `"candidate"`
   - `created_at`: now
   - `embedding`: computed from `name + ": " + description[:200]`

4. **Create edges:**
   - `DERIVED_FROM`: Procedure -> each source Episode
   - `OPERATES_ON`: Procedure -> each common Entity

#### Precondition Extraction Algorithm

1. Collect `state` JSON from all successful instances of the sequence.
2. For each key `k` in the state dictionaries:
   - Compute the set of values `V_k` across all instances.
   - If a single value `v` appears in >= 80% of instances:
     - Add condition: `state.{k} = '{v}'`
3. If any conditions found, generate a Locy WHERE fragment:
   ```
   WHERE e.state.{k1} = '{v1}' AND e.state.{k2} = '{v2}'
   ```
4. Return `PreconditionRule` with the fragment, or `None` if no clear preconditions.

**Example:** If 85% of successful "investigate-then-implement" sequences have `state.language = "rust"`, the precondition becomes `WHERE e.state.language = 'rust'`.

---

### 13.3 -- P5: Promotion & Deprecation Lifecycle

**Objective:** Implement the full Procedure lifecycle: candidate -> active promotion when validated through use, and active -> deprecated when stale or ineffective.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/consolidation/procedure_lifecycle.rs` | Modified | Add promotion and deprecation logic |

#### Lifecycle State Machine

```
CANDIDATE --> (validate) --> ACTIVE --> (deprecate) --> DEPRECATED
                 ^                          |
                 |                          |
                 +--- (re-promote) ---------+
```

#### Promotion Criteria

A candidate procedure is promoted to active when ALL of the following are met:

| Criterion | Threshold | Field |
|---|---|---|
| Additional successful uses after creation | >= 3 | `use_count` (only counts uses after Procedure was created) |
| Sustained effectiveness | > 0.7 | `effectiveness` = `success_count / (success_count + failure_count)` |

```rust
pub async fn check_promotion(kb: &KnowledgeBase, procedure_id: &str) -> Result<bool>;
pub async fn promote_procedure(kb: &KnowledgeBase, procedure_id: &str) -> Result<()>;
```

#### Deprecation Criteria

An active procedure is deprecated when ANY of the following are met:

| Criterion | Threshold | Field |
|---|---|---|
| No recent use | Last used > 30 days ago | `last_used_at` |
| Poor recent effectiveness | < 0.5 | Recent `effectiveness` (last 10 uses) |

```rust
pub async fn check_deprecation(kb: &KnowledgeBase, procedure_id: &str) -> Result<bool>;
pub async fn deprecate_procedure(kb: &KnowledgeBase, procedure_id: &str) -> Result<()>;
```

#### Tracking Fields (updated on each use)

| Field | Update Logic |
|---|---|
| `use_count` | Increment by 1 on each invocation |
| `success_count` | Increment by 1 if outcome = "success" |
| `failure_count` | Increment by 1 if outcome = "failure" |
| `effectiveness` | Recompute: `success_count / (success_count + failure_count)` |
| `avg_outcome_delta` | Running average of `delta` JSON from episodes using this procedure |
| `last_used_at` | Set to current timestamp |

```rust
pub async fn record_procedure_use(
    kb: &KnowledgeBase,
    procedure_id: &str,
    outcome: &str,
    delta: Option<&serde_json::Value>,
) -> Result<()>;
```

#### Lifecycle Checks in Consolidation

During each consolidation cycle, after sequence detection and candidate creation:

1. For each candidate Procedure: `check_promotion()` -> if true, `promote_procedure()`
2. For each active Procedure: `check_deprecation()` -> if true, `deprecate_procedure()`

Log lifecycle transitions:
```
INFO  procedure "investigate-then-implement" promoted: candidate -> active (use_count=5, effectiveness=0.82)
INFO  procedure "manual-review-loop" deprecated: active -> deprecated (last_used=32d ago)
```

---

### 13.4 -- P6: Co-occurrence Graph & Community Detection

**Objective:** Build a weighted entity co-occurrence graph from multiple evidence sources and run community detection to discover natural knowledge clusters.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/consolidation/topics.rs` | New | Co-occurrence graph construction, community detection, Topic creation |

#### Structs and Functions

```rust
/// A weighted edge in the entity co-occurrence graph.
pub struct CoOccurrence {
    pub entity_a: EntityId,
    pub entity_b: EntityId,
    pub weight: f64,
}

/// A detected community of co-occurring entities.
pub struct Community {
    pub entity_ids: Vec<EntityId>,
    pub entity_names: Vec<String>,
    pub total_weight: f64,
}

/// Build the entity co-occurrence graph from all evidence sources.
pub async fn build_cooccurrence_graph(
    kb: &KnowledgeBase,
    agent_id: &str,
) -> Result<Vec<CoOccurrence>>;

/// Run community detection on the co-occurrence graph.
pub fn detect_communities(
    edges: &[CoOccurrence],
    min_community_size: usize,  // default: 3
) -> Vec<Community>;
```

#### Co-occurrence Sources

Entities co-occur when they appear together in the same context. Weight = number of co-occurrences across all sources:

| Source | How Co-occurrence Is Detected | Query Pattern |
|---|---|---|
| Same Message | Both entities have MENTIONS edges from the same Message | `MATCH (m:Message)-[:MENTIONS]->(e1:Entity), (m)-[:MENTIONS]->(e2:Entity) WHERE e1 <> e2` |
| Same Observation | Both entities have ABOUT edges from the same Observation | `MATCH (o:Observation)-[:ABOUT]->(e1:Entity), (o)-[:ABOUT]->(e2:Entity) WHERE e1 <> e2` |
| Same Episode | Both entities have MENTIONS edges from the same Episode | `MATCH (ep:Episode)-[:MENTIONS]->(e1:Entity), (ep)-[:MENTIONS]->(e2:Entity) WHERE e1 <> e2` |
| Same Fact | One entity is the subject and another is the object of the same Fact | `MATCH (f:Fact)-[:ABOUT]->(e1:Entity), (f)-[:ABOUT]->(e2:Entity) WHERE e1 <> e2` |

Aggregate query returns `(entity_a_id, entity_b_id, total_weight)` tuples.

#### Community Detection Algorithm

Use label propagation (simpler, O(n) per iteration) as the primary algorithm:

1. Assign each entity a unique label.
2. Iterate: each entity adopts the label most common among its neighbors (weighted by edge weight).
3. Converge: stop when no label changes (typically 5-15 iterations).
4. Communities = groups of entities sharing the same label.
5. Filter: remove communities with < 3 entities.

**Alternative:** Louvain algorithm if label propagation produces too many small clusters. Louvain optimizes modularity and produces hierarchical communities.

Implementation note: the co-occurrence graph is typically small (< 10K entities per agent), so either algorithm runs in milliseconds.

```rust
/// Label propagation community detection.
pub fn label_propagation(
    edges: &[CoOccurrence],
    max_iterations: usize,  // default: 20
) -> HashMap<EntityId, usize>;  // entity -> community label
```

#### Trigger Conditions

P6 does not run on every consolidation cycle -- it runs when:

| Trigger | Condition | Rationale |
|---|---|---|
| Periodic | Every N consolidation cycles (default: 5) | Avoid excessive computation |
| Growth-based | Entity count increased > 10% since last P6 run | New entities may form new clusters |

Track: `last_p6_cycle_number: u64` and `last_p6_entity_count: u64` in the ConsolidationWorker state.

---

### 13.5 -- P6: Topic Lifecycle (Create, Merge, Split, Dissolve)

**Objective:** Manage Topic node lifecycle based on community detection results. Create new Topics, merge overlapping ones, split oversized ones, and dissolve topics that lose their members.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/consolidation/topics.rs` | Modified | Add Topic lifecycle operations |

#### Structs and Functions

```rust
/// Create a Topic from a detected community.
pub async fn create_topic(
    kb: &KnowledgeBase,
    community: &Community,
    agent_id: &str,
    llm: Option<&LlmProvider>,
) -> Result<TopicId>;

/// Merge two Topics with significant entity overlap.
pub async fn merge_topics(
    kb: &KnowledgeBase,
    larger_topic_id: &str,
    smaller_topic_id: &str,
) -> Result<()>;

/// Split a Topic into child Topics based on subclusters.
pub async fn split_topic(
    kb: &KnowledgeBase,
    topic_id: &str,
    subclusters: &[Vec<EntityId>],
    llm: Option<&LlmProvider>,
) -> Result<Vec<TopicId>>;

/// Dissolve a Topic that has too few remaining entities.
pub async fn dissolve_topic(
    kb: &KnowledgeBase,
    topic_id: &str,
) -> Result<()>;

/// Reconcile detected communities with existing Topics.
pub async fn reconcile_topics(
    kb: &KnowledgeBase,
    communities: &[Community],
    agent_id: &str,
    llm: Option<&LlmProvider>,
) -> Result<TopicReconciliationResult>;

pub struct TopicReconciliationResult {
    pub created: Vec<TopicId>,
    pub merged: Vec<(TopicId, TopicId)>,  // (kept, absorbed)
    pub split: Vec<(TopicId, Vec<TopicId>)>,  // (original, children)
    pub dissolved: Vec<TopicId>,
    pub unchanged: Vec<TopicId>,
}
```

#### Topic Creation

For each new community (not matching an existing Topic):

1. **Generate name:** If LLM available, prompt with entity names: "These entities form a knowledge cluster: [entity1, entity2, entity3, ...]. Suggest a concise topic name (2-4 words)." Fallback: most frequent entity type + top entity name, e.g., "Rust Projects" or "Auth Services".

2. **Generate summary:** If LLM available, prompt for a one-sentence summary. Fallback: "Topic containing {count} entities: {top_3_entity_names}."

3. **Create Topic node:**
   - `topic_id`: generated UUID v7
   - `name`: LLM-generated or derived
   - `summary`: LLM-generated or derived
   - `entity_count`: count of member entities
   - `fact_count`: count of facts linked to member entities
   - `embedding`: computed from `name + " " + summary`

4. **Create BELONGS_TO edges:**
   - Entity -> Topic (for each member entity)
   - Fact -> Topic (for facts whose subject or object entities belong to this topic)

#### Merge Logic

Two Topics are merge candidates when:

| Criterion | Threshold |
|---|---|
| Entity overlap | > 60% of the smaller topic's entities also belong to the larger topic |

Merge process:
1. Identify the larger topic (by entity_count) -- this one survives.
2. Move all BELONGS_TO edges from the smaller topic to the larger topic.
3. Update the surviving topic: recalculate entity_count, fact_count, regenerate name/summary if LLM available.
4. Delete the smaller Topic node.

#### Split Logic

A Topic is a split candidate when:

| Criterion | Threshold |
|---|---|
| Entity count | > 15 entities |
| Subclusters detected | Re-running community detection at higher resolution on the topic's entities reveals >= 2 subclusters with >= 3 entities each |

Split process:
1. Extract the entity subgraph for this Topic.
2. Run community detection at higher resolution (lower weight threshold or more iterations).
3. If subclusters found: create child Topics for each subcluster.
4. Move BELONGS_TO edges to the appropriate child Topics.
5. Delete the parent Topic node.

#### Dissolve Logic

A Topic is dissolved when:

| Criterion | Threshold |
|---|---|
| Remaining entity count | < 3 entities (after entities were deleted or reassigned) |

Dissolve process:
1. Remove all BELONGS_TO edges from entities to this Topic.
2. Remove all BELONGS_TO edges from facts to this Topic.
3. Delete the Topic node.

#### Reconciliation Algorithm

The `reconcile_topics` function orchestrates the full lifecycle:

1. **Match communities to existing Topics:** Compute Jaccard similarity between each community's entity set and each existing Topic's entity set. Match if Jaccard > 0.5.
2. **Create:** Communities with no matching Topic -> `create_topic()`.
3. **Merge:** Pairs of existing Topics with > 60% overlap -> `merge_topics()`.
4. **Split:** Topics with > 15 entities and detectable subclusters -> `split_topic()`.
5. **Dissolve:** Topics with < 3 remaining entities -> `dissolve_topic()`.
6. **Update:** Matched Topics: update entity_count, fact_count, refresh BELONGS_TO edges.

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_detect_sequences_basic` | `consolidation/procedures.rs` | 5+ identical 2-step sequences detected with correct frequency |
| `test_detect_sequences_three_step` | `consolidation/procedures.rs` | 3-step sequences detected from FOLLOWED_BY chains |
| `test_detect_sequences_filters_low_frequency` | `consolidation/procedures.rs` | Sequences with frequency < 5 are excluded |
| `test_detect_sequences_filters_low_effectiveness` | `consolidation/procedures.rs` | Sequences with effectiveness <= 0.7 are excluded |
| `test_detect_sequences_filters_single_step` | `consolidation/procedures.rs` | Single-step sequences are excluded |
| `test_detect_sequences_excludes_existing` | `consolidation/procedures.rs` | Sequences matching active Procedures are skipped |
| `test_common_entity_extraction` | `consolidation/procedures.rs` | Entities mentioned in 60%+ of episodes are identified |
| `test_create_candidate_procedure` | `consolidation/procedure_lifecycle.rs` | Procedure node created with status "candidate" and correct properties |
| `test_create_candidate_procedure_with_llm` | `consolidation/procedure_lifecycle.rs` | LLM-generated name and description applied |
| `test_create_candidate_procedure_fallback` | `consolidation/procedure_lifecycle.rs` | Derived name and description when LLM unavailable |
| `test_derived_from_edges` | `consolidation/procedure_lifecycle.rs` | DERIVED_FROM edges link Procedure to source Episodes |
| `test_operates_on_edges` | `consolidation/procedure_lifecycle.rs` | OPERATES_ON edges link Procedure to common Entities |
| `test_precondition_extraction_found` | `consolidation/procedure_lifecycle.rs` | State key present in 80%+ of successes generates WHERE fragment |
| `test_precondition_extraction_none` | `consolidation/procedure_lifecycle.rs` | No clear preconditions returns None |
| `test_precondition_locy_fragment` | `consolidation/procedure_lifecycle.rs` | Generated Locy WHERE fragment is syntactically valid |
| `test_promotion_criteria_met` | `consolidation/procedure_lifecycle.rs` | Candidate promoted when use_count >= 3 and effectiveness > 0.7 |
| `test_promotion_criteria_not_met_low_uses` | `consolidation/procedure_lifecycle.rs` | Candidate not promoted with use_count < 3 |
| `test_promotion_criteria_not_met_low_effectiveness` | `consolidation/procedure_lifecycle.rs` | Candidate not promoted with effectiveness <= 0.7 |
| `test_deprecation_no_recent_use` | `consolidation/procedure_lifecycle.rs` | Active deprecated when last_used > 30 days |
| `test_deprecation_low_effectiveness` | `consolidation/procedure_lifecycle.rs` | Active deprecated when recent effectiveness < 0.5 |
| `test_record_procedure_use_success` | `consolidation/procedure_lifecycle.rs` | use_count, success_count, last_used_at updated |
| `test_record_procedure_use_failure` | `consolidation/procedure_lifecycle.rs` | failure_count updated, effectiveness recalculated |
| `test_build_cooccurrence_graph` | `consolidation/topics.rs` | Co-occurrences from all 4 sources aggregated correctly |
| `test_cooccurrence_from_messages` | `consolidation/topics.rs` | Two entities in same Message produce co-occurrence |
| `test_cooccurrence_from_observations` | `consolidation/topics.rs` | Two entities ABOUT same Observation produce co-occurrence |
| `test_cooccurrence_from_episodes` | `consolidation/topics.rs` | Two entities MENTIONS from same Episode produce co-occurrence |
| `test_cooccurrence_from_facts` | `consolidation/topics.rs` | Subject and object entities of same Fact produce co-occurrence |
| `test_cooccurrence_weight_accumulates` | `consolidation/topics.rs` | Multiple co-occurrence sources increase weight |
| `test_label_propagation_basic` | `consolidation/topics.rs` | 3 connected entities form a single community |
| `test_label_propagation_two_clusters` | `consolidation/topics.rs` | Two disconnected groups form two communities |
| `test_label_propagation_min_size` | `consolidation/topics.rs` | Communities with < 3 entities are filtered out |
| `test_create_topic` | `consolidation/topics.rs` | Topic node created with correct properties and BELONGS_TO edges |
| `test_merge_topics` | `consolidation/topics.rs` | Two overlapping topics merged, smaller deleted |
| `test_split_topic` | `consolidation/topics.rs` | Large topic split into subclusters |
| `test_dissolve_topic` | `consolidation/topics.rs` | Topic with < 3 entities dissolved, edges removed |
| `test_reconcile_new_community` | `consolidation/topics.rs` | New community creates new Topic |
| `test_reconcile_merge` | `consolidation/topics.rs` | Overlapping existing Topics merged |
| `test_reconcile_unchanged` | `consolidation/topics.rs` | Stable community matches existing Topic, no changes |
| `test_p6_trigger_periodic` | `consolidation/topics.rs` | P6 runs every 5th consolidation cycle |
| `test_p6_trigger_growth` | `consolidation/topics.rs` | P6 runs when entity count increases > 10% |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_episode_to_procedure_pipeline` | `tests/procedural_integration.rs` | Full pipeline: create 5+ episodes with FOLLOWED_BY -> detect sequence -> create candidate Procedure -> verify properties |
| `test_procedure_promotion_lifecycle` | `tests/procedural_integration.rs` | Candidate -> 3 successful uses -> promoted to active -> verify status change |
| `test_procedure_deprecation_lifecycle` | `tests/procedural_integration.rs` | Active -> 31 days idle -> deprecated -> verify status change |
| `test_entity_to_topic_pipeline` | `tests/procedural_integration.rs` | Full pipeline: create entities with co-occurrences -> detect communities -> create Topics -> verify BELONGS_TO edges |
| `test_topic_merge_integration` | `tests/procedural_integration.rs` | Create two overlapping topics -> run reconciliation -> verify merge |
| `test_consolidation_cycle_with_p5_p6` | `tests/procedural_integration.rs` | Full consolidation cycle runs P4 -> P5 -> P6 in sequence, ConsolidationCycle node records procedures_promoted count |
| `test_offline_procedure_naming` | `tests/procedural_integration.rs` | Procedure creation works without LLM, using derived names |
| `test_offline_topic_naming` | `tests/procedural_integration.rs` | Topic creation works without LLM, using derived names |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_sequence_frequency` | Any random episode graph with N identical sequences of length M produces correct frequency N |
| `proptest_effectiveness_range` | Effectiveness is always in [0.0, 1.0] for any combination of success/failure outcomes |
| `proptest_community_stability` | Running community detection twice on the same graph produces the same communities |
| `proptest_merge_idempotent` | Merging a topic with itself is a no-op |

### Validation Criteria

- Procedures are derived from episode patterns (FOLLOWED_BY chains), not fabricated
- Topics cluster related entities meaningfully (co-occurrence based, not random)
- Procedure preconditions match when applicable (Locy WHERE fragment is valid)
- P5 and P6 run as ConsolidationWorker steps after P4
- LLM fallback works: all paths functional without LLM
- ConsolidationCycle node records P5/P6 activity (procedures_promoted, etc.)

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module doc | `consolidation/procedures.rs` | Overview of P5 pipeline, sequence detection algorithm, thresholds |
| Module doc | `consolidation/procedure_lifecycle.rs` | Procedure lifecycle state machine, promotion/deprecation criteria, precondition extraction |
| Module doc | `consolidation/topics.rs` | Overview of P6 pipeline, co-occurrence sources, community detection, topic lifecycle |
| Inline rustdoc on `ActionSequence` | `procedures.rs` | What each field means, filtering criteria |
| Inline rustdoc on `PreconditionRule` | `procedure_lifecycle.rs` | Locy WHERE fragment format, extraction algorithm |
| Inline rustdoc on `Community` | `topics.rs` | Community detection output format |
| Inline rustdoc on `TopicReconciliationResult` | `topics.rs` | What each lifecycle operation means |

---

## Review Checklist

- [ ] `detect_sequences` queries FOLLOWED_BY chains of length 2 and 3
- [ ] Frequency threshold >= 5 enforced
- [ ] Effectiveness threshold > 0.7 enforced
- [ ] Single-step sequences excluded (>= 2 distinct action_types)
- [ ] Existing active Procedures are not duplicated
- [ ] Common entities extracted with 60%+ presence threshold
- [ ] Candidate Procedure created with status `"candidate"` and correct properties
- [ ] DERIVED_FROM edges link Procedure to source Episodes
- [ ] OPERATES_ON edges link Procedure to common Entities
- [ ] Precondition extraction examines state fields of successful instances
- [ ] Precondition Locy WHERE fragment is syntactically valid
- [ ] Null precondition returned when no clear pattern exists
- [ ] Promotion requires use_count >= 3 AND effectiveness > 0.7
- [ ] Deprecation triggers on last_used > 30 days OR effectiveness < 0.5
- [ ] `record_procedure_use` updates all tracking fields correctly
- [ ] Co-occurrence graph built from all 4 sources (Message, Observation, Episode, Fact)
- [ ] Weights accumulate across sources
- [ ] Label propagation converges in < 20 iterations
- [ ] Communities with < 3 entities filtered out
- [ ] Topic nodes created with BELONGS_TO edges to Entities and Facts
- [ ] Topic embedding computed from `name + " " + summary`
- [ ] Merge threshold: > 60% entity overlap
- [ ] Split threshold: > 15 entities with detectable subclusters
- [ ] Dissolve threshold: < 3 remaining entities
- [ ] P6 trigger: every 5th cycle OR entity count increase > 10%
- [ ] LLM fallback: derived names when LLM unavailable (P5 and P6)
- [ ] ConsolidationCycle node updated with `procedures_promoted` count
- [ ] All lifecycle transitions logged at INFO level
- [ ] No unwrap() or expect() on LLM calls (always handle unavailability)

---

## Definition of Done

1. **P5 sequence detection works:** Querying an Episode graph with 5+ identical FOLLOWED_BY sequences produces ActionSequence candidates with correct frequency and effectiveness.
2. **P5 candidate creation works:** ActionSequences are converted to Procedure nodes with status "candidate", DERIVED_FROM edges to Episodes, and OPERATES_ON edges to Entities.
3. **P5 precondition extraction works:** State fields of successful episodes are analyzed; clear patterns produce valid Locy WHERE fragments; ambiguous patterns produce None.
4. **P5 lifecycle functional:** Candidates promote to active after 3+ successful uses with > 0.7 effectiveness. Active procedures deprecate after 30+ days idle or < 0.5 effectiveness. All tracking fields update correctly on each use.
5. **P6 co-occurrence graph works:** Entity co-occurrences are aggregated from Messages, Observations, Episodes, and Facts with correct weights.
6. **P6 community detection works:** Label propagation produces stable communities of >= 3 entities. Disconnected entity groups form separate communities.
7. **P6 topic lifecycle functional:** Topics are created from new communities, merged when > 60% overlap, split when > 15 entities with subclusters, and dissolved when < 3 entities remain.
8. **P6 trigger logic works:** P6 runs every 5th consolidation cycle or when entity count increases > 10%.
9. **LLM fallback works:** All P5 and P6 operations function without LLM, using derived names and descriptions.
10. **Integration validated:** Full pipeline from Episodes -> P5 -> Procedure and Entities -> P6 -> Topic works end-to-end within a consolidation cycle.
11. **All unit tests pass:** `cargo nextest run -n auto -p uniko-cortex` passes with zero failures for all procedure and topic tests.
12. **Clippy clean:** `cargo clippy -p uniko-cortex -- -D warnings` passes.
13. **Documented:** All public types, functions, and lifecycle rules have rustdoc with examples.
