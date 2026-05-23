# Pipelines & Tools Design (v2)

## Overview

Data enters the system as Messages and Artifacts. Pipelines process
them through 8 stages, populating the graph from raw content to
derived knowledge. Tools let agents add knowledge the pipelines
can't infer.

```
Data In                  Immediate              Near-Realtime           Background
────────                 ─────────              ─────────────           ──────────

Message ──┬─ P1: Ingest ─┬─ P2: NER ──────────── P3: Observations ─┐
          │   (sync)      │   (sync, <100ms)      (async, <5s)       │
          │               │                                          │
Artifact ─┘               └─ P7a: Auto-Embed                        ├─ P4: Consolidation
                              (async)                                │     (periodic)
                                                                     │
Episode ──── (via tool) ─── P7b: Computed Embed ─────────────────────├─ P5: Procedure Promotion
                                                                     │     (periodic)
Action ───── (via tool) ─── P7b: Computed Embed                      │
                                                                     ├─ P6: Topic Detection
                                                                     │     (low frequency)
                                                                     │
                                                                     ├─ P7c: Artifact Pooling
                                                                     │     (after chunking)
                                                                     │
                                                                     ├─ P7d: Summaries
                                                                     │     (on session end, etc.)
                                                                     │
                                                                     └─ P8: Rule Induction
                                                                           (low frequency)
```


## Pipeline 1: Ingest (synchronous)

**Trigger**: Message received or Artifact ingested.
**Latency target**: < 10ms for messages, < 100ms for text artifacts,
variable for audio/video.

### Message path

```
Input:  content, sender_id, session_id, timestamp, content_type, addressed_to: Option<Vec<String>>
Output: Message node + edges

Steps:
  1. Create Message node
  2. Create SENT_BY → Participant edge (with role)
  3. Create ADDRESSED_TO edges: use provided addressed_to list, or infer from session PARTICIPATED_IN edges (all participants except sender)
  4. Create IN_SESSION → Session edge
  5. Find previous Message in session → create NEXT edge (with gap_ms)
  6. If content length > message_chunk_threshold (1024 tokens):
     a. Chunk using text document strategy
     b. Create Chunk nodes + HAS_CHUNK edges
  7. Queue for Pipeline 2 (NER)
  8. Queue for Pipeline 3 (Observation extraction)
  9. Queue for Pipeline 7a (auto-embed on Message.content)
```

### Artifact path

```
Input:  content/bytes, path, mime_type, metadata
Output: Artifact node + Chunk nodes + HAS_CHUNK edges

Steps:
  1. Compute hash, detect mime_type if not provided
  1b. Check if an Artifact with the same hash already exists. If yes, skip chunking and embedding — create a reference edge instead of a duplicate node.
  2. Create Artifact node (path, content, mime_type, hash, size, kind, language)
  3. Select chunker by mime_type:
     ├─ text/plain, text/markdown   → recursive splitter (400-512 tokens, sentence-boundary aligned, 10-20% overlap)
     ├─ text/x-python, text/x-rust  → tree-sitter AST chunker
     ├─ text/html, application/xml  → DOM section chunker
     ├─ application/pdf             → page extractor + text chunker
     ├─ text/csv, application/json  → schema-aware row grouper
     ├─ audio/*                     → transcribe + speaker-turn chunker
     ├─ video/*                     → scene detector + transcript aligner
     └─ image/*                     → no chunking (atomic)
  4. Create Chunk nodes with metadata:
     chunk_id: deterministic "{artifact_id}:{index}" (enables dedup on re-chunk)
     (chunk_type, language, symbol_name, speaker, heading, mime_type)
  5. Create HAS_CHUNK edges (with index)
  6. Queue each Chunk for Pipeline 2 (NER)
  7. Queue for Pipeline 7a (auto-embed on Chunks)
  8. Queue for Pipeline 7c (pool chunk embeddings → Artifact.text_embedding)
  9. If image/audio/video: queue for Pipeline 7b (multimodal embeddings)
```

### Action output overflow

When an Action's output exceeds action_output_artifact_threshold
(256 tokens):

```
  1. Create Artifact (kind: "snippet", content: full output)
  2. Create PRODUCED edge: Action → Artifact
  3. Store summary in Action.output instead of full content
  4. Artifact follows normal Artifact path (chunk, embed)
```

### Session lifecycle

Sessions are created on first message if no active session exists
for the participant+goal/task combination. Sessions end when:
- Explicit end signal (tool call)
- Inactivity timeout (configurable, default 30 min)
- Goal/Task status changes to terminal

```
On session end:
  1. Set Session.ended_at
  2. Queue for Pipeline 7d (generate session summary)
  3. Re-embed Session (now has topic + summary)
```


## Pipeline 2: Entity Extraction (synchronous/near-sync)

**Trigger**: Message or Chunk created by Pipeline 1.
**Latency target**: < 100ms (local NER), < 3s (with LLM enhancement).

```
Input:  Message or Chunk node with text content
Output: Entity nodes + MENTIONS edges

Steps:
  1. LOCAL NER (primary, always runs, < 100ms):

     For prose (text/plain, text/markdown, messages):
       a. Rule-based extraction:
          - Capitalized proper nouns
          - Quoted strings
          - Date/time expressions → entity_type: "date"
          - Numbers with units → entity_type: "measurement"
          - Email addresses, URLs → entity_type: "reference"
       b. Lightweight NER model (spaCy-equivalent, runs locally):
          - PERSON, ORG, GPE, EVENT, PRODUCT, WORK_OF_ART
          - No network calls, no LLM dependency

     For code (text/x-python, text/x-rust, etc.):
       a. tree-sitter extraction:
          - Function/method names → entity_type: "function"
          - Class/struct names → entity_type: "type"
          - Module/package names → entity_type: "module"
          - Import targets → entity_type: "dependency"

     For audio/video chunks:
       a. Use transcript text → same as prose extraction
       b. Speaker name from chunk.speaker → entity_type: "person"

  2. DEDUPLICATION (< 10ms):
     a. Exact match on Entity.name (Hash index)
     b. Case-insensitive match (normalize to lowercase for comparison)
     c. Fuzzy match with embedding similarity:
        - Compute cosine similarity between new entity embedding and existing entities
        - Merge threshold: cosine > 0.85 when entity_type matches
        - Merge threshold: cosine > 0.92 when entity_types differ or are unknown
        - On merge: keep the longest form as canonical name, shorter forms become aliases
          Example: "LGBTQ support group" (canonical) ← "support group" (alias)
        - Never merge across incompatible types (e.g., "person" and "org")
     d. Basic coreference within session: resolve pronouns to the most recently
        mentioned entity of matching type. This is a heuristic with limited
        accuracy — complex coreference chains require the LLM enhancement path.

  3. UPSERT:
     a. Existing entity → increment frequency, update last_seen
     b. New entity → create node with entity_type, first_seen, frequency: 1
     c. Create MENTIONS edge (source → Entity, with count)

  4. LLM ENHANCEMENT (async, optional, queued):
     a. Refine entity_type ("LGBTQ support group" → "organization" not "concept")
     b. Resolve complex coreferences ("the agency Caroline chose" → "LGBTQ-inclusive adoption agency")
     c. Extract entities missed by local NER (implicit references, paraphrases)
     d. Only runs if LLM provider is available
     e. Results merged back: update entity_type, create additional MENTIONS
```


## Pipeline 3: Observation Extraction (async, near-real-time)

**Trigger**: Message or Chunk created + entities extracted by Pipeline 2.
**Latency target**: < 5s.
**Requires**: LLM or rule-based statement classifier.

```
Input:  Message/Chunk node + MENTIONS edges to entities
Output: Observation nodes + OBSERVED_IN, OBSERVED_DURING, ABOUT edges

Steps:
  1. FILTER — skip non-informative content:
     - Greetings, reactions, filler ("Hey!", "Wow!", "Thanks!")
     - Pure questions with no embedded facts
     - System messages, error messages
     - Very short content (< 5 words)

  2. EXTRACT — identify observable statements:

     Rule-based path (no LLM, always available):
       a. Sentence-split the content
       b. For each sentence:
          - Contains a named entity from Pipeline 2? → candidate
          - Has subject-verb-object structure? → candidate
          - Expresses preference ("I prefer", "I like", "I don't")? → candidate
          - States a fact ("X is Y", "X has Y", "X went to Y")? → candidate
       c. For each candidate:
          - Extract subject (from entity mentions or grammatical subject)
          - Construct observation as self-contained statement
          - Set observed_at from message timestamp + context clues
            ("yesterday" → timestamp - 1 day, "last year" → year - 1)

     LLM-enhanced path (optional, better quality):
       a. Prompt: "Extract factual observations from this message.
          For each observation, provide: content, subject, observed_at.
          Skip greetings, questions, and reactions."
       b. Validate: each observation must reference an entity from Pipeline 2
       c. Merge with rule-based results

  3. CREATE — for each observation:
     a. Create Observation node (content, subject, observed_at, confidence)
     b. Create OBSERVED_IN → source Message/Chunk
     c. Create ABOUT → relevant Entity nodes
     d. If the most recent Episode RECORDED_BY the same participant IN_SESSION this session was recorded within the last 5 minutes → create OBSERVED_DURING edge. Otherwise, observations are linked only via OBSERVED_IN → Message.
     e. Auto-embed observation.content

  4. CONTRADICTION CHECK — lightweight, inline:
     a. For each new observation, find existing observations with same subject
     b. If semantically conflicting (embed similarity < 0.3 but same subject):
        Flag pair for Pipeline 4 (consolidation) attention
     c. Do NOT invalidate anything here — just flag. Pipeline 4 decides.

  5. OBSERVATION FROM ARTIFACTS:
     When an Artifact is ingested (not a message), observations are
     extracted from Chunks instead. The complete Artifact path is:
       Artifact → P1 chunking → Chunk nodes created
       → each Chunk → P2 NER → Entity + MENTIONS edges
       → each Chunk → P3 → Observation nodes
       → Observations linked via OBSERVED_IN → Chunk (not Message)
       → Observations linked via ABOUT → Entity
     This path is critical for MemoryAgentBench (memorize phase):
        document chunks → observations → consolidation → facts
```


## Pipeline 4: Consolidation (background, periodic)

**Trigger**: Timer (configurable, default 15 min) or threshold
(N new observations since last cycle, default 20).
**Non-blocking**: Never interrupts agent operations.

```
Input:  New Observations + existing Facts + active Rules + recent Episodes
Output: New/updated/invalidated Facts + ConsolidationCycle record

Steps:

  1. LOAD
     a. Observations created since last consolidation cycle
     b. Existing active Facts (WHERE btic.contains(valid_at, now()))
     c. Active Rules (status = "active")
     d. Recent Episodes (for procedure promotion in Pipeline 5)

  2. GROUP BY SUBJECT ENTITY
     For each entity with new observations:

  3. PATTERN DETECTION
     a. Cluster new observations by semantic similarity
        (embedding cosine > 0.7 = same pattern)
     b. Match against existing Facts by subject + embedding similarity
     c. Classify each observation as:
        - REINFORCING: supports existing fact
        - NOVEL: no matching fact exists
        - CONTRADICTING: conflicts with existing fact

  4. FACT DERIVATION (from NOVEL observations)
     When N observations (>= 3) cluster together, from at least 2 distinct sessions
     (to avoid premature crystallization from a single conversational tangent):
     a. Derive Fact:
        subject = common entity
        predicate = extracted from observation content (LLM or pattern)
        object = extracted from observation content
        confidence = observation_count / (observation_count + 2)  (Laplace smoothing)
     b. Set valid_at BTIC:
        lo = earliest observation.observed_at
        hi = ∞ (open, active)
        certainty = "approximate" if count < 10, "definite" if >= 10
        granularity = finest granularity from observations
     c. Create edges:
        SUPPORTED_BY → each supporting Observation (with weight)
        DERIVED_FROM → Episodes that contain these observations
        ABOUT → subject Entity
        DERIVED_BY → Rule (if a Locy rule triggered this derivation)

  5. FACT REINFORCEMENT (from REINFORCING observations)
     a. Increase confidence: new_conf = old_conf + (1 - old_conf) * 0.1 * count, capped at 0.95
     b. Increment observation_count
     c. Add SUPPORTED_BY edges to new observations
     d. Update certainty if crossing the 10-observation threshold

  6. CONTRADICTION DETECTION
     a. For each CONTRADICTING observation:
        - Count reinforcing vs contradicting observations
        - If contradicting > 40% of total → invalidate
     b. Invalidation:
        - Close BTIC interval: set hi = now()
        - Create INVALIDATES edge with reason
        - Derive new Fact from contradicting observations (repeat step 4)
     c. Oscillation detection:
        - If same subject+predicate invalidated >= 3 times:
          Create special "unstable" observation, flag for human review

  7. DRIFT DETECTION
     a. For each entity, count invalidations in last 30 days
     b. If invalidation_count >= threshold (default 4):
        - Create drift alert (stored as Observation with subject=entity,
          content="Entity shows systematic drift")
        - Flag entity for recall cascade override:
          when this entity appears in a query, force Phase 2+
          even if Phase 1 coverage is sufficient

  8. APPLY LOCY RULES
     a. Execute each active Rule against the current graph state
     b. Rules can:
        - Derive additional Facts (risk propagation, transitive relationships)
        - Detect patterns the clustering missed
        - Validate existing Facts against new evidence
     c. Track: DERIVED_BY edge from Fact to Rule

  9. RECORD CYCLE
     Create ConsolidationCycle node:
       episodes_processed, observations_processed,
       facts_created, facts_reinforced, facts_invalidated,
       drift_alerts, rules_applied
     Create edges:
       PROCESSED → Observations consumed
       CREATED → Facts derived
       INVALIDATED → Facts closed
       APPLIED_RULE → Rules executed
```

### Consolidation Implementation Details

**Step trait:**
```rust
pub trait ConsolidationStep: Send + Sync {
    fn name(&self) -> &str;
    fn should_run(&self, ctx: &ConsolidationContext) -> bool;
    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
    fn error_policy(&self) -> StepErrorPolicy { StepErrorPolicy::Skip }
}
```

**Scheduling:**
- Observation count threshold: default 20 new observations since last cycle
- Time interval: default 15 min
- Semaphore: max 4 concurrent agent consolidations

**Confidence reinforcement formula:**
When new observations support existing fact:
`new_conf = old_conf + (1 - old_conf) * 0.1 * count` capped at 0.95

**Memory decay (applied in step 1 before pattern detection):**
`importance * exp(-ln(2) / half_life * age_days)`
Default half_life_days: 30. Episodes below prune_below (default 0.05) are excluded from pattern detection.

**Dead-letter queue:**
Failed consolidation or enrichment tasks are captured as DeadLetter nodes:
```
DeadLetter
  step              String      -- which pipeline step failed
  error             String      -- error message
  node_ref          Int64       -- the node that couldn't be processed
  retry_count       Int64       -- how many times retried so far
  max_retries       Int64       -- default: 3
  next_retry_at     DateTime    -- computed from backoff
  created_at        DateTime
```
Operations: retry(id), retry_all_pending(), clear(id), clear_all(), list_pending()


## Pipeline 5: Procedure Promotion (background, periodic)

**Trigger**: Runs after Pipeline 4, or separately on timer.
**Input**: Episodes linked via FOLLOWED_BY chains.

```
Steps:

  1. FIND ACTION SEQUENCES
     Query FOLLOWED_BY chains on Episodes:
     MATCH (e1:Episode)-[:FOLLOWED_BY]->(e2:Episode)-[:FOLLOWED_BY]->(e3:Episode)
     WHERE e1.outcome = "success" AND e2.outcome = "success" AND e3.outcome = "success"
     RETURN e1.action_type, e2.action_type, e3.action_type, count(*) AS freq

  2. DETECT PATTERNS
     Group by (action_type sequence) where:
     - Frequency >= 5
     - Overall effectiveness > 0.7 (success_count / total_count)
     - Involves at least 2 distinct action_types

  3. CHECK FOR EXISTING PROCEDURE
     Does a Procedure with matching action sequence already exist?
     - Yes → update use_count, recalculate effectiveness
     - No → create candidate

  4. CREATE CANDIDATE PROCEDURE
     a. Create Procedure node:
        name: LLM-generated if available, otherwise derived from action_type sequence (e.g., "investigate-then-implement")
        description: LLM-generated if available, otherwise "Procedure from {action_types} with {effectiveness}% effectiveness"
        steps: ordered list of action_types with parameters
        status: "candidate"
        effectiveness: success_rate from episodes
     b. Create edges:
        DERIVED_FROM → source Episodes
        OPERATES_ON → common Entities across episodes

  4b. EXTRACT PRECONDITIONS
      Examine Episode.state fields of successful instances in the sequence.
      If a state key appears with value above a threshold in 80%+ of successes:
        Generate Locy WHERE fragment as precondition_rule.
      If no clear precondition extractable:
        Leave precondition_rule null (procedure applies unconditionally).

  5. PROMOTE CANDIDATES
     Candidates with:
     - use_count >= 3 additional successful uses after creation
     - effectiveness > 0.7 sustained
     → Promote to status: "active"

  6. DEPRECATE STALE PROCEDURES
     Active procedures with:
     - No use in last 30 days
     - Recent effectiveness < 0.5
     → Set status: "deprecated"

Note: The 5× frequency threshold and 3× post-creation validation are intentionally
conservative. Promoting premature procedures is worse than not promoting — a bad
procedure executed repeatedly compounds errors. For domains with sparse episodes,
lower thresholds via PipelineParams configuration.
```


## Pipeline 6: Topic Detection (background, low frequency)

**Trigger**: After consolidation when new entities or facts appear.
**Frequency**: Every N consolidation cycles (default 5), or when
entity count increases by > 10%.

```
Steps:

  1. BUILD CO-OCCURRENCE GRAPH
     For each pair of entities that co-occur in:
     - Same Message (both MENTIONS edges from same Message)
     - Same Observation (both ABOUT edges from same Observation)
     - Same Episode (both MENTIONS edges from same Episode)
     - Same Fact (subject + object entities)
     Weight = co-occurrence count

  2. COMMUNITY DETECTION
     Run Louvain or label propagation on the co-occurrence graph.
     Each community = a Topic candidate.

  3. CREATE/UPDATE TOPICS
     For each community with >= 3 entities:
     a. Create or find existing Topic node
     b. Generate name: most frequent entity names or LLM-generated
     c. Generate summary: LLM summarizes member entities and their facts
        "Caroline's adoption journey: Caroline has been researching adoption
         agencies since May 2023, chose an LGBTQ-inclusive agency, applied
         and passed interviews by October 2023."
     d. Create BELONGS_TO edges from member Entities and related Facts
     e. Compute embedding from name + summary

  4. MERGE/SPLIT TOPICS
     - Two topics with > 60% entity overlap → merge
     - One topic with > 15 entities and subclusters detected by running community detection again within the topic's subgraph at higher resolution → split
     - Topics with < 3 remaining entities → dissolve
```


## Pipeline 7: Embedding & Summary (async, continuous)

Split into 4 sub-pipelines that run independently:

### 7a: Auto-Embed (triggered by node creation)

```
Nodes that auto-embed (uni-db handles via source field):
  Message.embedding     ← content
  Chunk.embedding       ← text
  Observation.embedding ← content
  Summary.embedding     ← text

Trigger: node creation in Pipeline 1 or Pipeline 3
Latency: < 50ms per node (batch-embedded by uni-db)
```

### 7b: Computed Embedding (triggered by node creation/update)

```
Nodes that need application-computed embed strings:

  Entity.embedding      ← name + " (" + entity_type + ")"
  Goal.embedding        ← title
  Task.embedding        ← title
  Session.embedding     ← topic (initially), topic + summary (on session end)
  Topic.embedding       ← name + summary
  Fact.embedding        ← subject + " " + predicate + " " + object
  Procedure.embedding   ← name + ": " + description[:200]
  Episode.embedding     ← extracted topic from state Json + " — " + action_type
  Action.embedding      ← action_type + " " + key_input_summary

Process:
  1. Construct embed string from node fields
  2. Call embed model
  3. Set embedding on node
  4. Update vector index

Trigger: node creation (Pipeline 1, 2, 3, 4, 5, or agent tools)
Latency: < 100ms per node
```

### 7c: Artifact Embedding Pooling (triggered after chunking)

```
For text/code Artifacts:
  1. Wait for all Chunk embeddings to be computed (7a)
  2. Mean-pool all chunk embedding vectors
  3. Set Artifact.text_embedding

For image Artifacts:
  1. Run vision model (CLIP/SigLIP) → Artifact.image_embedding
  2. Run unified model (ImageBind) → Artifact.multimodal_embedding

For audio Artifacts:
  1. Run audio model (CLAP) → Artifact.audio_embedding
  2. Run unified model (ImageBind) → Artifact.multimodal_embedding
  3. Pool transcript chunk embeddings → Artifact.text_embedding

For video Artifacts:
  1. Run video model (LanguageBind/InternVideo) → Artifact.video_embedding
  2. Sample frames → vision model → pool → Artifact.image_embedding
  3. Extract audio → audio model → Artifact.audio_embedding
  4. Pool transcript chunks → Artifact.text_embedding
  5. Run unified model → Artifact.multimodal_embedding

Trigger: all Chunks embedded (for text) or Artifact created (for media)
Latency: < 500ms for text pooling, 1-10s for model inference on media
```

### 7d: Summarization (triggered by lifecycle events)

```
Session summary:
  Trigger: session ends (inactivity timeout or explicit end)
  Input: all Messages in session
  Output: Summary node (level: "session") + SUMMARIZES → Session
  Method: LLM summarizes message sequence
  Side-effect: re-embed Session (now has summary)

Task summary:
  Trigger: task status → "completed" or "failed"
  Input: all Episodes and Sessions FOR_TASK
  Output: Summary node (level: "task") + SUMMARIZES → Task

Goal summary:
  Trigger: goal status change, or periodic (daily)
  Input: all Tasks PART_OF goal, their summaries
  Output: Summary node (level: "goal") + SUMMARIZES → Goal

Entity summary:
  Trigger: entity frequency crosses threshold (10, 50, 100)
  Input: all Observations ABOUT entity, all Facts ABOUT entity
  Output: Summary node (level: "entity") + SUMMARIZES → Entity
  Example: "Caroline is a transgender woman pursuing adoption. She
           attended LGBTQ support groups, chose an inclusive agency,
           and is interested in counseling as a career."

Topic summary:
  Trigger: Pipeline 6 creates/updates topic
  Input: member entities, their facts, their observations
  Output: Summary node (level: "topic") + SUMMARIZES → Topic

Artifact summary:
  Trigger: after chunking + embedding completes for text artifacts > 2000 tokens
  Input: all chunks of the artifact
  Output: Summary node (level: "artifact") + SUMMARIZES → Artifact
```


## Pipeline 8: Rule Induction (background, low frequency)

**Maturity: Research/Experimental.** This pipeline involves LLM generation of Locy rules with holdout validation — a compound system with many failure modes. Ship authored rules (via `add_rule`/`author_rule` tools) before investing in automatic induction. The `add_rule` tool is the recommended path for Phases 1-3; Pipeline 8 is deferred to Phase 6.

**Trigger**: After N consolidation cycles (default 10), or when
fact count grows by > 20% since last induction.
**This is the ILP pipeline from the spec.**

```
Steps:

  1. MINE — statistical pattern queries
     Run Cypher/Locy queries over the graph to find patterns:
     - Correlations between entity types and outcomes
     - Temporal patterns (X always happens before Y)
     - Conditional patterns (when state.X > threshold, outcome = failure)
     Example finding: "backlog > 200 + evening + batch_approve → failure (8/10)"

  2. GENERATE — LLM creates candidate Locy rules
     One LLM call per pattern:
     Input: pattern description + existing rules + schema
     Output: Locy rule source code + natural language description
     Example:
       CREATE RULE high_backlog_batch_risk AS
         MATCH (e:Episode)
         WHERE e.action_type = "batch_approve"
           AND e.state.backlog > 200
         YIELD e.outcome AS outcome
       → "Batch approvals with backlog > 200 have 80% failure rate"

  3. VALIDATE — test via ASSUME/ABDUCE
     a. ASSUME: apply rule to holdout episodes, check predictions
     b. ABDUCE: check if the rule's conclusions can be falsified
     c. Compute precision, recall, coverage

  4. PERSIST — store qualifying rules
     Acceptance: score = precision×0.4 + recall×0.3 + novelty×0.3, threshold >= 0.65
     a. Create Rule node (source, natural_language, source_type: "induced",
        status: "candidate", confidence, precision, recall, coverage)
     b. Mark as available for Pipeline 4 (consolidation)

  4b. PROMOTE CANDIDATE RULES
     After 3 successful validations against new episodes in subsequent
     consolidation cycles, promote to status "active".

  5. MONITOR — re-score existing induced rules
     For each active induced Rule:
     a. Evaluate against episodes since last scoring
     b. Update precision, recall, coverage
     c. If precision < 0.5 → demote (status: "demoted")
     d. If no coverage in 30 days → prune (status: "pruned")
```


## Agent Tools

### Lifecycle tools (create graph structure)

```
create_goal(title, description, metrics, guardrails, deadline)
  → creates Goal node, OWNED_BY edge

create_task(title, description, goal_id, priority)
  → creates Task node, PART_OF → Goal, ASSIGNED_TO → Participant

start_session(task_id_or_goal_id, topic)
  → creates Session node, FOR_TASK/FOR_GOAL edge, PARTICIPATED_IN edge

end_session(session_id)
  → sets ended_at, triggers summarization (Pipeline 7d)

create_organization(name)
  → creates Organization node

create_team(name, purpose, org_id)
  → creates Team node + TEAM_IN_ORG → Organization edge

add_member(participant_id, team_or_org_id, role)
  → creates MEMBER_OF or PART_OF_TEAM edge

update_goal(goal_id, status?, metrics?, description?)
  → updates Goal node fields

update_task(task_id, status?, priority?, description?)
  → updates Task node fields
```

### Knowledge tools (add explicit knowledge)

```
record_episode(action_type, outcome, state, delta, importance, entity_refs)
  → creates Episode node + edges (RECORDED_BY, FOR_TASK, IN_SESSION,
    MENTIONS, FOLLOWED_BY)
  → triggers Pipeline 7b (computed embedding)
  → Episode is ONLY created via this tool, never by a pipeline

add_observation(content, subject, source_message_id?)
  → creates Observation node + OBSERVED_IN, ABOUT edges
  → for things the pipeline missed

assert_fact(subject, predicate, object, confidence, source)
  → creates Fact node + ABOUT edges, sets BTIC valid_at
  → for definitive statements that shouldn't wait for consolidation

invalidate_fact(fact_id, reason)
  → closes BTIC interval, creates INVALIDATES edge
  → for explicit corrections

record_action(action_type, input, output, status, triggered_by_message?)
  → creates Action node + PERFORMED_BY, TRIGGERED_BY, IN_SESSION edges
  → Find previous Action in same session → create NEXT_ACTION edge with gap
  → if output is large, creates Artifact via overflow path

add_rule(name, locy_source, natural_language)
  → creates Rule node with status "active", source_type "authored"
  → validates Locy source syntax before persisting

author_rule(description)
  → LLM generates Locy source from natural language description
  → validates via ASSUME on sample data
  → creates Rule node with status "candidate", source_type "authored"

share_fact(fact_id)
  → sets Fact.visibility to "global", creates SHARED_FROM edge
  → makes the fact visible to all agents

shared_facts(agent_id?)
  → queries Facts where visibility = "global"
  → optionally filters by originating agent
```

### Query tools (retrieve from memory)

```
recall(query, budget, filters?)
  → runs recall cascade (Compact → Expand → Broaden)
  → returns ContextBundle with ranked items from all memory layers
  → when procedures are returned, those with precondition_rule are evaluated against the query context. Only procedures whose preconditions match are included.

search_entities(query, entity_type?, limit?)
  → hybrid search: vector + fulltext on Entity nodes

search_facts(subject?, predicate?, active_only?, valid_at?, limit?)
  → filtered query on Fact nodes, BTIC-aware

search_messages(query, session_id?, participant_id?, time_range?, limit?)
  → hybrid search on Message nodes

working_memory(goal_id)
  → traverses Goal → Tasks → Sessions → Messages → Facts → Entities
  → returns the full active context for a goal

assume(mutations, query)
  → ASSUME: fork graph state, apply mutations, execute query, rollback
  → returns query results from hypothetical state

abduce(conclusion)
  → ABDUCE: given a conclusion, find minimal set of facts that support it
  → returns supporting facts with explanations
```


## Pipeline vs Tool Boundary

```
                        Pipeline (automatic)          Tool (agent-initiated)
                        ────────────────────          ──────────────────────
Messages              → P1 creates Message           (messages arrive externally)
Artifacts             → P1 creates Artifact           (artifacts arrive externally)
Entities              → P2 extracts from text         search_entities (query)
Observations          → P3 extracts from messages     add_observation (implicit things)
Facts                 → P4 consolidates               assert_fact (definitive statements)
Contradictions        → P4 detects                    invalidate_fact (corrections)
Procedures            → P5 promotes from episodes     (always derived, never manual)
Topics                → P6 detects from clusters      (always derived, never manual)
Embeddings            → P7 computes                   (always automatic)
Summaries             → P7d generates                 (always automatic)
Rules                 → P8 induces from patterns      add_rule / author_rule (authored Locy)
Episodes              → (never automatic)             record_episode (always manual)
Actions               → (never automatic)             record_action (always manual)
Goals                 → (never automatic)             create_goal (always manual)
Tasks                 → (never automatic)             create_task (always manual)
Sessions              → P1 auto-creates if needed     start_session / end_session
```

Episodes, Actions, Goals, and Tasks are ONLY created by agents.
They represent intentional acts, not derivable observations.


## Benchmark Data Flow

### LoCoMo

```
For each session (19 sessions, 419 turns):
  For each turn:
    P1: Create Message + edges                    ← 419 Messages
    P2: Extract entities (Caroline, LGBTQ group)  ← ~50 unique Entities
    P3: Extract observations                      ← ~100 Observations
    P7a: Auto-embed Messages, Observations

  On session end:
    P7d: Generate session summary                 ← 19 Summaries

After all sessions:
  P4: Consolidation                               ← ~30 Facts
  P6: Topic detection                             ← ~5 Topics
  P8: Rule induction (if enough patterns)         ← ~2 Rules

Query: "What did Caroline research?"
  recall() → Phase 1: Fact(Caroline, pursuing, adoption) → done

Query: "What did Caroline realize after her charity race?" (adversarial)
  recall() → "charity race" → Entity → MENTIONS → Message → SENT_BY → Melanie
  → adversarial: question attributes to Caroline but Melanie said it

Query: "What activities does Melanie partake in?" (multi-hop)
  recall() → Entity(Melanie) → MENTIONS ← Messages → MENTIONS → Entities
  → collect: pottery, camping, painting, swimming, running, violin
```

### Evo-Memory

```
For each task (sequential):
  1. recall(task.question)                 → context from prior tasks
  2. Generate answer
  3. record_episode(outcome, state, delta) → Episode node
  4. Every 10 tasks:
     P4: Consolidation → derive Facts from episode patterns
     → Future recall() finds Facts in Phase 1 (Compact)
     → phase1_only_pct increases over time

Improvement signal:
  accuracy[late tasks] - accuracy[early tasks] > 0
  → proves consolidation makes the agent smarter
```

### MemoryAgentBench

```
Memorize phase:
  Ingest context as Artifact → P1 chunks → P2 NER → P3 observations
  → P4 consolidation → derive Facts

Query phase:
  recall(question) → finds Facts + Observations → answers

Conflict resolution:
  New chunk contradicts earlier one:
  P3: observation flagged as contradicting
  P4: invalidates old Fact via BTIC, creates new Fact
  → query returns updated information
```
