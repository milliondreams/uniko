# Phase 11: Agent Tools & Working Memory

## Context

This phase implements all agent-facing tools (lifecycle, knowledge, query), the working memory traversal, and the Locy stdlib rules. These tools are the primary API surface that agents interact with through the Cortex layer.

Agent tools are divided into three categories:
- **Lifecycle tools** -- creating and managing goals, tasks, sessions, organizations, and teams. These define the structural skeleton of an agent's work.
- **Knowledge tools** -- recording episodes, actions, observations, facts, and rules. These are how agents contribute knowledge that pipelines cannot capture automatically.
- **Query tools** -- recall, search, hypothetical reasoning (ASSUME/ABDUCE), and working memory traversal. These are how agents retrieve information.

Agent tools supplement the automatic pipelines (P1-P7). Pipelines process raw content automatically; agent tools let agents contribute structured knowledge that only they can provide (subjective episode assessments, explicit fact assertions, goal definitions).

**Key design principle:** Every tool call creates well-formed graph state -- correct nodes, edges, embeddings triggered, timestamps set. Tools are the only way agents interact with the memory system; they must be reliable, fast, and correct.

## Prerequisites

- **Phase 10 (Recall Cascade)** -- query tools delegate to RecallContextBuilder. The recall cascade must be fully functional.
- **Phase 9 (Consolidation P4)** -- fact assertion/invalidation relies on BTIC intervals. Consolidation must handle contradiction detection.
- **Phase 8 (Embedding P7)** -- tools trigger P7b embedding for newly created nodes. The embedding pipeline must be operational.
- **Phase 2 (NER P2)** -- entity extraction used by IntentProfile and entity_refs resolution.
- **Phase 3 (Schema)** -- all node types and edge types must be registered in the schema.
- **Layer 1 (KnowledgeBase)** -- Locy runtime must be operational for stdlib rules and ASSUME/ABDUCE.

## Sub-phases

---

### 11.1 -- Lifecycle Tools

**Objective:** Implement tools for creating and managing the organizational hierarchy: Goals, Tasks, Sessions, Organizations, Teams, and Participants.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/tools/mod.rs` | Rust | Tools module root with submodule declarations |
| `crates/uniko-memory/src/tools/lifecycle.rs` | Rust | Goal, Task, Session, Org, Team lifecycle management |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `create_goal` | `async fn create_goal(cortex: &Cortex, title: &str, description: &str, metrics: Value, guardrails: Value, deadline: Option<Timestamp>) -> Result<GoalId>` | Creates a Goal node + OWNED_BY edge to the agent's Participant |
| `create_task` | `async fn create_task(cortex: &Cortex, title: &str, description: &str, goal_id: &GoalId, priority: u8) -> Result<TaskId>` | Creates a Task node + PART_OF -> Goal + ASSIGNED_TO -> Participant |
| `start_session` | `async fn start_session(cortex: &Cortex, task_id_or_goal_id: &str, topic: &str) -> Result<SessionId>` | Creates a Session + FOR_TASK/FOR_GOAL + PARTICIPATED_IN edges |
| `end_session` | `async fn end_session(cortex: &Cortex, session_id: &SessionId) -> Result<()>` | Sets ended_at timestamp, triggers P7d summarization |
| `update_goal` | `async fn update_goal(cortex: &Cortex, goal_id: &GoalId, status: Option<&str>, metrics: Option<Value>, description: Option<&str>) -> Result<()>` | Updates Goal node properties |
| `update_task` | `async fn update_task(cortex: &Cortex, task_id: &TaskId, status: Option<&str>, priority: Option<u8>, description: Option<&str>) -> Result<()>` | Updates Task node properties |
| `create_organization` | `async fn create_organization(cortex: &Cortex, name: &str) -> Result<OrgId>` | Creates an Organization node |
| `create_team` | `async fn create_team(cortex: &Cortex, name: &str, purpose: &str, org_id: &OrgId) -> Result<TeamId>` | Creates a Team node + PART_OF -> Organization |
| `add_member` | `async fn add_member(cortex: &Cortex, participant_id: &str, team_or_org_id: &str, role: &str) -> Result<()>` | Creates MEMBER_OF edge with role property |

#### Implementation Details

**`create_goal`:**
1. Generate `goal_id` via `new_id()`
2. Create Goal node: `{ goal_id, title, description, status: "active", metrics, guardrails, deadline, created_at: now() }`
3. Create OWNED_BY edge from Goal to the calling agent's Participant node
4. Trigger P7b embedding (embed `"{title} {description}"`)
5. Return `goal_id`

**`create_task`:**
1. Generate `task_id` via `new_id()`
2. Verify `goal_id` references an existing Goal
3. Create Task node: `{ task_id, title, description, status: "pending", priority, created_at: now() }`
4. Create PART_OF edge from Task to Goal
5. Create ASSIGNED_TO edge from Task to the calling agent's Participant
6. Trigger P7b embedding (embed `"{title} {description}"`)
7. Return `task_id`

**`start_session`:**
1. Generate `session_id` via `new_id()`
2. Determine whether the target is a Task or Goal by looking up the ID
3. Create Session node: `{ session_id, topic, started_at: now(), ended_at: null }`
4. Create FOR_TASK or FOR_GOAL edge as appropriate
5. Create PARTICIPATED_IN edge from Participant to Session
6. Trigger P7b embedding (embed `"{topic}"`)
7. Return `session_id`

**`end_session`:**
1. Set `ended_at = now()` on the Session node
2. Trigger P7d summarization pipeline for the session (generates Session.summary)
3. Re-trigger P7b embedding with updated embed string

---

### 11.2 -- Knowledge Tools

**Objective:** Implement tools for agents to record episodes, actions, observations, facts, and rules. These tools create the raw material that consolidation processes into derived knowledge.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/tools/knowledge.rs` | Rust | Episode, Action, Observation, Fact, Rule recording tools |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `record_episode` | `async fn record_episode(cortex: &Cortex, action_type: &str, outcome: &str, state: Value, delta: Value, importance: f64, entity_refs: Vec<String>) -> Result<EpisodeId>` | Records an agent's learning experience |
| `record_action` | `async fn record_action(cortex: &Cortex, action_type: &str, input: Value, output: Value, status: &str, triggered_by_message: Option<MessageId>) -> Result<ActionId>` | Records a tool call or operation |
| `add_observation` | `async fn add_observation(cortex: &Cortex, content: &str, subject: &str, source_message_id: Option<MessageId>) -> Result<ObservationId>` | Records a factual statement the agent noticed |
| `assert_fact` | `async fn assert_fact(cortex: &Cortex, subject: &str, predicate: &str, object: &str, confidence: f64, source: &str) -> Result<FactId>` | Creates a Fact with BTIC [now, infinity) |
| `invalidate_fact` | `async fn invalidate_fact(cortex: &Cortex, fact_id: &FactId, reason: &str) -> Result<()>` | Closes BTIC interval + creates INVALIDATES edge |
| `add_rule` | `async fn add_rule(cortex: &Cortex, name: &str, locy_source: &str, natural_language: &str) -> Result<RuleId>` | Validates Locy syntax and creates a Rule node |
| `author_rule` | `async fn author_rule(cortex: &Cortex, description: &str) -> Result<RuleId>` | LLM generates Locy from description, validates, creates Rule |
| `share_fact` | `async fn share_fact(cortex: &Cortex, fact_id: &FactId) -> Result<()>` | Promotes a fact to global visibility |
| `shared_facts` | `async fn shared_facts(cortex: &Cortex, agent_id: Option<&str>) -> Result<Vec<Fact>>` | Retrieves all globally-visible facts |

#### Implementation Details

**`record_episode`:**
1. Generate `episode_id` via `new_id()`
2. Create Episode node: `{ episode_id, action_type, outcome, state, delta, importance, timestamp: now() }`
3. Create RECORDED_BY edge from Episode to the calling agent's Participant
4. Create FOR_TASK edge from Episode to the active task (if any)
5. Create IN_SESSION edge from Episode to the active session (if any)
6. **MENTIONS edges** -- for each entity name in `entity_refs`, find or create the Entity node, then create a MENTIONS edge from Episode to Entity
7. **FOLLOWED_BY chain** -- find the previous Episode for this agent in the current session (by timestamp). If found, create FOLLOWED_BY edge from previous to this episode with `gap_ms = this.timestamp - prev.timestamp`
8. Trigger P7b embedding -- embed from topic extracted from state JSON (not just action+outcome). This is the critical v5 fix that makes Phase 2 recall functional.
9. Return `episode_id`

**`record_action`:**
1. Generate `action_id` via `new_id()`
2. Create Action node: `{ action_id, action_type, input, output, status, started_at: now(), duration_ms }`
3. Create PERFORMED_BY edge from Action to calling agent's Participant
4. Create IN_SESSION edge from Action to active session
5. **TRIGGERED_BY** -- if `triggered_by_message` provided, create TRIGGERED_BY edge from Action to Message
6. **NEXT_ACTION chain** -- find previous Action for this agent in session, create NEXT_ACTION edge
7. **Output overflow** -- if serialized `output` > 256 tokens, create Artifact node with the output content, create PRODUCED edge from Action to Artifact, truncate Action.output to summary, and chunk the Artifact via P1 chunking
8. Return `action_id`

**`assert_fact`:**
1. Generate `fact_id` via `new_id()`
2. Create Fact node: `{ fact_id, subject, predicate, object, confidence, source, valid_at: BTIC [now(), ∞) }`
3. Create ABOUT edge from Fact to Entity matching `subject` (find or create)
4. Trigger P7b embedding (embed `"{subject} {predicate} {object}"`)
5. Return `fact_id`

**`invalidate_fact`:**
1. Look up Fact by `fact_id`
2. Close BTIC interval: set `valid_at.hi = now()` (changes from [lo, ∞) to [lo, now))
3. Create INVALIDATES edge from the invalidating action/fact to the invalidated Fact, with `reason` property
4. No re-embedding needed (invalidated facts are excluded from search)

**`add_rule`:**
1. **Validate Locy syntax** -- parse `locy_source` using the Locy parser. If parse fails, return `UnikoError::Locy` with the parse error.
2. Generate `rule_id` via `new_id()`
3. Create Rule node: `{ rule_id, name, locy_source, natural_language, status: "active", source_type: "authored", confidence: 1.0, created_at: now() }`
4. Return `rule_id`

**`author_rule`:**
1. Call LLM with `description` to generate Locy source code
2. Validate generated Locy via parsing
3. If validation fails, retry once with error feedback
4. Create Rule node with `status: "candidate"`, `source_type: "authored"`
5. Return `rule_id`

**`share_fact` / `shared_facts`:**
- `share_fact`: set `visibility = "global"` on the Fact node + create SHARED_FROM edge to the agent's Participant
- `shared_facts`: query all Fact nodes where `visibility = "global"`, optionally filtered by agent_id via SHARED_FROM edge

#### Performance Targets

| Operation | Target | Reference |
|---|---|---|
| Episode recording | < 30ms | NF10 |
| Action recording | < 30ms | -- |
| Fact assertion | < 10ms | NF1 (node creation) |
| Fact invalidation | < 10ms | -- |
| Rule addition | < 50ms | (includes Locy parsing) |

---

### 11.3 -- Query Tools

**Objective:** Implement tools for agents to query memory: recall (cascade), entity search, fact search, message search, hypothetical reasoning (ASSUME), and abductive reasoning (ABDUCE).

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/tools/query.rs` | Rust | Query and retrieval tools |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `recall` | `async fn recall(cortex: &Cortex, query: &str, budget: usize, filters: Option<RecallFilters>) -> Result<ContextBundle>` | Full recall cascade via RecallContextBuilder |
| `search_entities` | `async fn search_entities(cortex: &Cortex, query: &str, entity_type: Option<&str>, limit: Option<usize>) -> Result<Vec<Entity>>` | Search entities by name/type |
| `search_facts` | `async fn search_facts(cortex: &Cortex, subject: Option<&str>, predicate: Option<&str>, active_only: Option<bool>, valid_at: Option<Timestamp>, limit: Option<usize>) -> Result<Vec<Fact>>` | Search facts by subject/predicate with temporal filter |
| `search_messages` | `async fn search_messages(cortex: &Cortex, query: &str, session_id: Option<&SessionId>, participant_id: Option<&str>, time_range: Option<(Timestamp, Timestamp)>, limit: Option<usize>) -> Result<Vec<Message>>` | Search messages with filters |
| `assume` | `async fn assume(cortex: &Cortex, mutations: Vec<Mutation>, query: &str) -> Result<Vec<Record>>` | Hypothetical reasoning: fork state, apply mutations, query, rollback |
| `abduce` | `async fn abduce(cortex: &Cortex, conclusion: &str) -> Result<Vec<Fact>>` | Abductive reasoning: find minimal set of facts supporting a conclusion |

#### Implementation Details

**`recall`:**
1. Build IntentProfile from `query` via `build_intent_profile`
2. Create `RecallContextBuilder` with intent
3. Apply any filters from `RecallFilters` (recency_window, min_reliability, contrastive, etc.)
4. Apply budget
5. Call `.assemble()` and return the ContextBundle

```rust
pub struct RecallFilters {
    pub recency_window_days: Option<u64>,
    pub min_reliability: Option<f64>,
    pub include_procedures: Option<bool>,
    pub include_kb: Option<bool>,
    pub contrastive: Option<bool>,
    pub limit: Option<usize>,
}
```

**`search_entities`:**
1. If `query` provided, vector search on Entity.embedding + fulltext on Entity.name
2. Filter by `entity_type` if specified
3. Limit results (default 20)
4. Return Entity nodes with properties

**`search_facts`:**
1. Build Cypher query filtering by subject, predicate, active_only (BTIC hi = ∞), valid_at (BTIC contains timestamp)
2. Limit results (default 20)
3. Return Fact nodes with BTIC information

**`search_messages`:**
1. Fulltext search on Message.content
2. Filter by session_id (IN_SESSION edge), participant_id (SENT_BY edge), time_range (timestamp)
3. Limit results (default 20)
4. Return Message nodes with properties

**`assume`:**
1. Delegate to KnowledgeBase ASSUME builder
2. Each `Mutation` is a graph mutation (create node, create edge, set property)
3. KnowledgeBase forks state, applies mutations, executes `query` (Locy or Cypher), rolls back
4. Return query results from the hypothetical state
5. Target: < 200ms (NF9)

**`abduce`:**
1. Delegate to KnowledgeBase ABDUCE
2. Parse `conclusion` into a target fact pattern
3. KnowledgeBase backward-infers through Locy rules to find minimal supporting facts
4. Return the supporting fact set

#### Performance Targets

| Operation | Target | Reference |
|---|---|---|
| Recall (compact only) | < 30ms | NF7 |
| Recall (all phases) | < 100ms | NF8 |
| Entity search | < 20ms | NF11 |
| Fact search | < 20ms | NF11 |
| Message search | < 20ms | NF11 |
| ASSUME | < 200ms | NF9 |

---

### 11.4 -- Working Memory Traversal

**Objective:** Implement goal-scoped working memory that assembles all relevant context for an active goal by traversing the graph from Goal through Tasks, Sessions, Messages, Facts, Entities, Observations, and Procedures.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/tools/working_memory.rs` | Rust | Working memory traversal logic |

#### Types

```rust
pub struct WorkingMemoryBundle {
    pub goal: Goal,
    pub tasks: Vec<Task>,
    pub sessions: Vec<Session>,
    pub messages: Vec<Message>,
    pub facts: Vec<Fact>,
    pub entities: Vec<Entity>,
    pub observations: Vec<Observation>,
    pub procedures: Vec<Procedure>,
}
```

#### Functions

| Function | Signature | Description |
|---|---|---|
| `working_memory` | `async fn working_memory(cortex: &Cortex, goal_id: &str, budget: usize) -> Result<WorkingMemoryBundle>` | Executes the full working memory traversal |

#### Traversal Pattern (Cypher)

```cypher
MATCH (g:Goal {goal_id: $goal_id})
OPTIONAL MATCH (t:Task)-[:PART_OF]->(g)
OPTIONAL MATCH (s:Session)-[:FOR_TASK]->(t)
OPTIONAL MATCH (m:Message)-[:IN_SESSION]->(s)
OPTIONAL MATCH (e:Episode)-[:FOR_TASK]->(t)
OPTIONAL MATCH (f:Fact)-[:DERIVED_FROM]->(e)
OPTIONAL MATCH (ent:Entity)<-[:MENTIONS]-(m)
OPTIONAL MATCH (o:Observation)-[:OBSERVED_IN]->(m)
OPTIONAL MATCH (p:Procedure)-[:USED_IN]->(t)
ORDER BY m.timestamp DESC
LIMIT $budget
RETURN g, collect(DISTINCT t) AS tasks,
       collect(DISTINCT s) AS sessions,
       collect(DISTINCT m) AS messages,
       collect(DISTINCT f) AS facts,
       collect(DISTINCT ent) AS entities,
       collect(DISTINCT o) AS observations,
       collect(DISTINCT p) AS procedures
```

#### Implementation Details

1. **Start at Goal** -- look up Goal node by `goal_id`
2. **Traverse to Tasks** -- follow PART_OF edges to find all Tasks belonging to this Goal
3. **Traverse to Sessions** -- follow FOR_TASK edges from Tasks to Sessions (+ FOR_GOAL edges from Goal to Sessions)
4. **Traverse to Messages** -- follow IN_SESSION edges from Sessions to Messages, ordered by timestamp DESC
5. **Traverse to Facts** -- follow DERIVED_FROM edges from Episodes (which are linked to Tasks via FOR_TASK) to Facts
6. **Traverse to Entities** -- follow MENTIONS edges from Messages (and Episodes) to Entities
7. **Traverse to Observations** -- follow OBSERVED_IN edges from Messages to Observations
8. **Traverse to Procedures** -- follow USED_IN edges from Tasks to Procedures
9. **Budget enforcement** -- budget is a node count limit (not token count). Default: 50 items. The ORDER BY timestamp DESC ensures the most recent information is prioritized when budget forces truncation.
10. **Deduplication** -- ensure each node appears only once in the result (DISTINCT in Cypher)

#### Edge Cases

- Goal with no Tasks: return just the Goal with empty collections
- Task with no Sessions: Tasks included but no Messages/Observations
- Session with no Messages: Session included (may have been just started)
- Recursive goal hierarchy: for MVP, only traverse one level (no PARENT_GOAL recursion). Can be extended later.

#### Performance Target

- Working memory traversal: < 200ms (NF17)
- The traversal is a multi-hop graph query (3-4 hops). NF4 says graph traversal (3-hop) < 5ms at Layer 1, but working memory involves collecting and deduplicating across many paths, hence the more generous 200ms target.

---

### 11.5 -- Locy Stdlib Rules

**Objective:** Create and register the four standard library rules that ship with uniko. These rules are the foundation of automated reasoning -- they detect patterns in episodes, identify contradictions, and compute relevance decay.

#### Files to Create/Modify

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/rules/mod.rs` | Rust | Rules module root |
| `crates/uniko-memory/src/rules/stdlib.rs` | Rust | Stdlib rule definitions, registration, and lifecycle management |

#### Stdlib Rules

**Rule 1: `relevance_decay`** (runs first every cycle -- other rules depend on it)

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

Purpose: Applies exponential decay to episode importance. Episodes older than ~60 days with low base importance fall below the 0.05 threshold and are effectively forgotten. This is the memory decay mechanism (F50).

**Rule 2: `episode_pattern_detector`**

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

Purpose: Detects recurring action+outcome patterns. When the same (action_type, outcome) pair occurs >= 3 times with mean importance > 0.3, it signals a pattern worth promoting to a Procedure.

**Rule 3: `sequence_detector`**

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

Purpose: Detects successful action sequences (A followed by B, both successful). When a sequence occurs >= promotion_threshold times, it is a candidate for Procedure promotion.

**Rule 4: `contradiction_detector`**

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

Purpose: Detects when recent episodes contradict existing facts. When contradicting episodes reach the threshold, the fact is flagged for invalidation.

#### Rule Registration

```rust
pub async fn register_stdlib_rules(cortex: &Cortex, agent_id: &str) -> Result<()>
```

1. For each of the 4 rules, create a Rule node with:
   - `source_type: "stdlib"`
   - `status: "active"`
   - `confidence: 1.0`
   - `created_at: now()`
2. Register the Locy source with the KnowledgeBase runtime
3. Bind parameters: `$agent_id`, `$promotion_threshold`, `$contradiction_threshold`

#### Rule Lifecycle Management

| State | Description | Transitions |
|---|---|---|
| Created | Rule just defined | -> Active (after validation) |
| Candidate | LLM-generated, needs validation | -> Active (promote if confidence > 0.60) |
| Active | Executing in consolidation cycles | -> Demoted (confidence < 0.40) |
| Demoted | Temporarily disabled due to low confidence | -> Active (re-promote if confidence > 0.60), -> Pruned (90 days inactive) |
| Pruned | Permanently removed (terminal) | -- |
| Superseded | Replaced by a better rule (terminal) | -- |

#### Lifecycle Functions

| Function | Signature | Description |
|---|---|---|
| `register_stdlib_rules` | `async fn register_stdlib_rules(cortex: &Cortex, agent_id: &str) -> Result<()>` | Creates and registers all 4 stdlib rules |
| `evaluate_rule_confidence` | `fn evaluate_rule_confidence(rule: &Rule, missed_cycles: u32) -> f64` | Computes decayed confidence |
| `apply_lifecycle_transitions` | `async fn apply_lifecycle_transitions(cortex: &Cortex) -> Result<()>` | Demotes, re-promotes, and prunes rules as needed |

#### Confidence Decay

```
decayed_confidence = stored_confidence * (0.95 ^ missed_cycles)
```

- Each consolidation cycle where a rule produces no matches increments `missed_cycles`
- When `decayed_confidence < 0.40`: demote rule (set status = "demoted")
- When `decayed_confidence > 0.60`: re-promote (set status = "active") -- hysteresis prevents oscillation
- After 90 days with status = "demoted" and no matches: prune (set status = "pruned")

#### Stdlib Exemptions

- Stdlib rules (source_type = "stdlib") are **exempt** from demotion, pruning, and supersession
- Their confidence always remains 1.0
- Agent-scoped rules with the same name as a stdlib rule **shadow** the global stdlib (agent's version takes precedence)

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_create_goal` | `tools/lifecycle.rs` | Goal node created with correct properties, OWNED_BY edge exists, P7b triggered |
| `test_create_task` | `tools/lifecycle.rs` | Task node created, PART_OF -> Goal edge, ASSIGNED_TO -> Participant edge |
| `test_create_task_invalid_goal` | `tools/lifecycle.rs` | Error returned when goal_id does not exist |
| `test_start_session_for_task` | `tools/lifecycle.rs` | Session created with FOR_TASK edge, PARTICIPATED_IN edge |
| `test_start_session_for_goal` | `tools/lifecycle.rs` | Session created with FOR_GOAL edge when goal_id passed |
| `test_end_session` | `tools/lifecycle.rs` | ended_at set, P7d summarization triggered |
| `test_update_goal` | `tools/lifecycle.rs` | Goal properties updated, unchanged fields preserved |
| `test_update_task` | `tools/lifecycle.rs` | Task properties updated correctly |
| `test_create_organization` | `tools/lifecycle.rs` | Organization node created |
| `test_create_team` | `tools/lifecycle.rs` | Team node created with PART_OF -> Organization edge |
| `test_add_member` | `tools/lifecycle.rs` | MEMBER_OF edge created with role property |
| `test_record_episode` | `tools/knowledge.rs` | Episode node created with all properties |
| `test_episode_recorded_by` | `tools/knowledge.rs` | RECORDED_BY edge links Episode to Participant |
| `test_episode_followed_by` | `tools/knowledge.rs` | Second episode in session creates FOLLOWED_BY edge from first |
| `test_episode_followed_by_gap_ms` | `tools/knowledge.rs` | gap_ms computed correctly between consecutive episodes |
| `test_episode_mentions` | `tools/knowledge.rs` | MENTIONS edges created for each entity_ref |
| `test_episode_embedding_source` | `tools/knowledge.rs` | P7b triggered with state topic, not just action+outcome |
| `test_record_action` | `tools/knowledge.rs` | Action node created with correct properties |
| `test_action_overflow` | `tools/knowledge.rs` | Output > 256 tokens creates Artifact + PRODUCED edge |
| `test_action_next_chain` | `tools/knowledge.rs` | NEXT_ACTION edge links sequential actions in session |
| `test_action_triggered_by` | `tools/knowledge.rs` | TRIGGERED_BY edge created when source message provided |
| `test_add_observation` | `tools/knowledge.rs` | Observation node with OBSERVED_IN and ABOUT edges |
| `test_assert_fact` | `tools/knowledge.rs` | Fact node with BTIC [now, ∞), ABOUT edge, P7b triggered |
| `test_assert_fact_confidence` | `tools/knowledge.rs` | Confidence value stored correctly |
| `test_invalidate_fact` | `tools/knowledge.rs` | BTIC hi closed to now(), INVALIDATES edge created |
| `test_invalidate_fact_not_found` | `tools/knowledge.rs` | Error returned when fact_id does not exist |
| `test_add_rule_valid` | `tools/knowledge.rs` | Rule node created with status "active", Locy parsed |
| `test_add_rule_invalid_locy` | `tools/knowledge.rs` | UnikoError::Locy returned for unparseable Locy source |
| `test_share_fact` | `tools/knowledge.rs` | visibility set to "global", SHARED_FROM edge created |
| `test_shared_facts` | `tools/knowledge.rs` | Returns only visibility="global" facts |
| `test_shared_facts_by_agent` | `tools/knowledge.rs` | Filters by agent_id via SHARED_FROM edge |
| `test_recall_delegates` | `tools/query.rs` | recall() builds IntentProfile and delegates to RecallContextBuilder |
| `test_search_entities` | `tools/query.rs` | Entities found by name and type filter |
| `test_search_facts_active` | `tools/query.rs` | active_only=true returns only BTIC [lo, ∞) facts |
| `test_search_facts_temporal` | `tools/query.rs` | valid_at filter returns facts valid at that timestamp |
| `test_search_messages` | `tools/query.rs` | Messages found by content, filtered by session and participant |
| `test_assume` | `tools/query.rs` | Mutations applied, query runs, state rolled back |
| `test_abduce` | `tools/query.rs` | Minimal supporting facts returned for conclusion |
| `test_working_memory_basic` | `tools/working_memory.rs` | Goal with Tasks, Sessions, Messages returns correct bundle |
| `test_working_memory_complete` | `tools/working_memory.rs` | All 8 collections populated (goal, tasks, sessions, messages, facts, entities, observations, procedures) |
| `test_working_memory_budget` | `tools/working_memory.rs` | Budget=10 limits total node count |
| `test_working_memory_ordering` | `tools/working_memory.rs` | Messages ordered by timestamp DESC |
| `test_working_memory_empty_goal` | `tools/working_memory.rs` | Goal with no tasks returns goal only |
| `test_working_memory_dedup` | `tools/working_memory.rs` | Same entity referenced from multiple messages appears once |
| `test_stdlib_registration` | `rules/stdlib.rs` | All 4 rules created as Rule nodes with source_type "stdlib" |
| `test_stdlib_relevance_decay` | `rules/stdlib.rs` | Old episodes get lower relevance, very old episodes pruned |
| `test_stdlib_pattern_detector` | `rules/stdlib.rs` | 3+ occurrences of (action_type, outcome) detected |
| `test_stdlib_sequence_detector` | `rules/stdlib.rs` | Successful A->B chains detected above threshold |
| `test_stdlib_contradiction_detector` | `rules/stdlib.rs` | Episodes contradicting facts detected |
| `test_confidence_decay` | `rules/stdlib.rs` | `1.0 * 0.95^5 = 0.7738` computed correctly |
| `test_demotion_threshold` | `rules/stdlib.rs` | confidence < 0.40 triggers demotion |
| `test_repromotion_threshold` | `rules/stdlib.rs` | confidence > 0.60 triggers re-promotion |
| `test_stdlib_exempt_demotion` | `rules/stdlib.rs` | Stdlib rules never demoted even at low confidence |
| `test_agent_rule_shadows_stdlib` | `rules/stdlib.rs` | Agent rule with same name as stdlib takes precedence |
| `test_pruning_90_days` | `rules/stdlib.rs` | Demoted rule with 90+ days inactivity gets pruned |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_full_episode_chain` | `tests/tools_integration.rs` | Record 5 episodes in sequence -> FOLLOWED_BY chain complete, gap_ms correct |
| `test_goal_task_session_lifecycle` | `tests/tools_integration.rs` | Create goal -> task -> session -> end session -> all edges correct |
| `test_fact_lifecycle` | `tests/tools_integration.rs` | Assert fact -> verify BTIC -> invalidate -> BTIC closed -> new fact asserted |
| `test_recall_end_to_end` | `tests/tools_integration.rs` | Record knowledge -> recall returns it via cascade |
| `test_working_memory_end_to_end` | `tests/tools_integration.rs` | Full lifecycle -> working_memory returns complete context |
| `test_stdlib_in_consolidation` | `tests/tools_integration.rs` | Register stdlib -> record episodes -> run consolidation -> stdlib rules fire and produce output |
| `test_rule_lifecycle_transitions` | `tests/tools_integration.rs` | Rule goes Created -> Active -> Demoted -> Pruned correctly |
| `test_share_and_retrieve` | `tests/tools_integration.rs` | Agent A shares fact -> Agent B retrieves via shared_facts |
| `test_offline_mode_tools` | `tests/tools_integration.rs` | All tools work without LLM (author_rule degrades gracefully) |

### Performance Tests

| Test | What It Validates | Target |
|---|---|---|
| `bench_record_episode` | Episode recording latency | < 30ms (NF10) |
| `bench_working_memory` | Working memory traversal with 50-item budget | < 200ms (NF17) |
| `bench_search_facts` | Fact search by subject/predicate | < 20ms (NF11) |
| `bench_search_entities` | Entity search by name | < 20ms (NF11) |
| `bench_search_messages` | Message search by content | < 20ms (NF11) |
| `bench_assert_fact` | Fact assertion | < 10ms |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module-level doc comment | `tools/mod.rs` | Overview of tool categories (lifecycle, knowledge, query), design rationale |
| Lifecycle tools docs | `tools/lifecycle.rs` | Each function documented with parameters, graph state created, edges formed |
| Knowledge tools docs | `tools/knowledge.rs` | Each function documented with parameter semantics, BTIC behavior, embedding triggers |
| Query tools docs | `tools/query.rs` | Each function documented with filter options, delegation to RecallContextBuilder |
| Working memory docs | `tools/working_memory.rs` | Traversal pattern explained, budget semantics, WorkingMemoryBundle field explanations |
| Stdlib rules docs | `rules/stdlib.rs` | Each rule's purpose, parameters, expected output, when it fires during consolidation |
| Rule lifecycle docs | `rules/stdlib.rs` | State diagram, confidence decay formula, threshold values, stdlib exemptions |
| RecallFilters docs | `tools/query.rs` | Each filter field documented with defaults and valid ranges |

---

## Review Checklist

- [ ] `tools/mod.rs` declares submodules: lifecycle, knowledge, query, working_memory
- [ ] `rules/mod.rs` declares submodule: stdlib
- [ ] `create_goal` creates Goal node + OWNED_BY edge + triggers P7b
- [ ] `create_task` creates Task node + PART_OF -> Goal + ASSIGNED_TO -> Participant + triggers P7b
- [ ] `start_session` creates Session + FOR_TASK or FOR_GOAL + PARTICIPATED_IN
- [ ] `end_session` sets ended_at + triggers P7d summarization
- [ ] `record_episode` creates Episode + RECORDED_BY + FOR_TASK + IN_SESSION + MENTIONS + FOLLOWED_BY + triggers P7b
- [ ] Episode embedding uses state topic (not action+outcome) per v5 critical fix
- [ ] `record_action` creates Action + PERFORMED_BY + IN_SESSION + TRIGGERED_BY + NEXT_ACTION
- [ ] Action output overflow at 256 tokens creates Artifact + PRODUCED edge
- [ ] `assert_fact` creates Fact with BTIC [now, ∞) + ABOUT edge
- [ ] `invalidate_fact` closes BTIC hi to now() + creates INVALIDATES edge
- [ ] `add_rule` validates Locy syntax before creating Rule node
- [ ] `share_fact` sets visibility="global" + SHARED_FROM edge
- [ ] `recall` delegates to RecallContextBuilder correctly
- [ ] `search_entities`, `search_facts`, `search_messages` all support filtering and limits
- [ ] `assume` delegates to KnowledgeBase ASSUME with fork/rollback
- [ ] `abduce` delegates to KnowledgeBase ABDUCE
- [ ] Working memory traversal follows correct path: Goal -> Task -> Session -> Message -> Fact -> Entity -> Observation -> Procedure
- [ ] Working memory budget is node count, default 50
- [ ] Working memory ordered by message.timestamp DESC
- [ ] All 4 stdlib rules registered with correct Locy source
- [ ] Stdlib rules have source_type="stdlib", status="active", confidence=1.0
- [ ] Stdlib rules exempt from demotion/pruning/supersession
- [ ] Confidence decay formula: stored_confidence * (0.95 ^ missed_cycles)
- [ ] Demotion threshold: confidence < 0.40
- [ ] Re-promotion threshold: confidence > 0.60 (hysteresis)
- [ ] Pruning after 90 days in "demoted" state with no matches
- [ ] Agent-scoped rules shadow global stdlib (same name precedence)
- [ ] Episode recording < 30ms (NF10)
- [ ] Working memory < 200ms (NF17)
- [ ] Tier queries < 20ms (NF11)
- [ ] ASSUME < 200ms (NF9)
- [ ] All unit tests pass
- [ ] All integration tests pass

---

## Definition of Done

1. **Lifecycle tools complete:** create_goal, create_task, start_session, end_session, update_goal, update_task, create_organization, create_team, add_member all create correct graph state with proper edges and triggered embeddings.
2. **Knowledge tools complete:** record_episode, record_action, add_observation, assert_fact, invalidate_fact, add_rule, author_rule, share_fact, shared_facts all create correct graph state. Episode FOLLOWED_BY chains maintain integrity. BTIC intervals are correct.
3. **Query tools complete:** recall delegates to RecallContextBuilder and returns ContextBundle. search_entities, search_facts, search_messages support all documented filters. assume and abduce delegate to KnowledgeBase correctly.
4. **Working memory functional:** working_memory(goal_id) traverses Goal -> Tasks -> Sessions -> Messages -> Facts -> Entities -> Observations -> Procedures and returns a complete WorkingMemoryBundle within 200ms (NF17).
5. **Stdlib rules registered:** All 4 rules (relevance_decay, episode_pattern_detector, sequence_detector, contradiction_detector) created as Rule nodes with correct Locy source, source_type="stdlib", status="active".
6. **Stdlib rules execute:** Each rule produces expected output when consolidation runs with matching data.
7. **Rule lifecycle works:** Confidence decay computed correctly. Demotion at < 0.40, re-promotion at > 0.60. Pruning after 90 days. Stdlib exempt.
8. **Cross-agent sharing works:** share_fact promotes visibility, shared_facts retrieves global facts filtered by agent.
9. **Offline mode:** All tools function without LLM. author_rule degrades gracefully (returns error or skips LLM step).
10. **Performance targets met:** Episode recording < 30ms (NF10), working memory < 200ms (NF17), tier queries < 20ms (NF11), ASSUME < 200ms (NF9) on --release build.
11. **All tests pass:** Unit tests for every tool, integration tests for lifecycle flows and rule execution, performance benchmarks meet targets.
