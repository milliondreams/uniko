# Schema v3: Goal-Oriented, Interaction-First

## Design Principles

1. Start from what's observable — communication and actions
2. Organize around goals — the reason anything happens
3. Knowledge is derived, never directly stored — it traces back to messages and actions
4. Participants are uniform — humans and agents are the same type
5. Every node type maps to a cognitive memory layer
6. Everything is traceable — facts link to rules, rules link to episodes, episodes link to messages


## Cognitive Memory Mapping

| Memory Layer | Graph Nodes | Purpose |
|---|---|---|
| Working Memory | Goal, Task + edges to relevant context | Active goal context, persists across sessions |
| Episodic Memory | Message, Action, Episode, Session | What happened, who said/did what, when, and what was the outcome |
| Semantic Memory | Entity, Observation, Fact | What we know — extracted and consolidated |
| Procedural Memory | Procedure, Rule | What works — proven patterns and formal logic |
| Meta-Memory | ConsolidationCycle + recall cascade + indexes | How we find, derive, and maintain knowledge |


## Layer 0: Participants

```
Participant
  participant_id    String      not null
  kind              String      not null    -- "human", "agent", "service"
  name              String
  capabilities      Json                    -- what tools/skills this participant has
  metadata          Json
  first_seen        DateTime
  last_seen         DateTime
```

Index: participant_id (Hash), kind (Hash), name (Fulltext)


## Layer 1: Goals, Tasks, Sessions

### Goal — the long-running objective

```
Goal
  goal_id           String      not null
  title             String      not null    -- "reduce refund cycle time by 40%"
  description       String                  -- detailed specification
  status            String                  -- "active", "achieved", "failed", "paused"
  metrics           Json                    -- target metrics and current values
  guardrails        Json                    -- constraints: budget, compliance, risk
  owner_id          String                  -- participant responsible
  created_at        DateTime
  deadline          DateTime
  completed_at      DateTime
  embedding         Vector
```

Index: goal_id (Hash), status (Hash), title (Fulltext)
Vector index: embedding

```
OWNED_BY            Goal → Participant
PARENT_GOAL         Goal → Goal            -- goal decomposition
```


### Task — a unit of work toward a goal

```
Task
  task_id           String      not null
  title             String      not null
  description       String
  status            String                  -- "pending", "in_progress", "completed",
                                            --  "failed", "blocked"
  priority          Float64
  created_at        DateTime
  completed_at      DateTime
  embedding         Vector
```

Index: task_id (Hash), status (Hash), title (Fulltext)
Vector index: embedding

```
PART_OF             Task → Goal
ASSIGNED_TO         Task → Participant
DEPENDS_ON          Task → Task
SUBTASK_OF          Task → Task
```


### Session — a bounded interaction

```
Session
  session_id        String      not null
  topic             String
  summary           String                  -- generated after session ends
  started_at        DateTime    not null
  ended_at          DateTime
  embedding         Vector
```

Index: session_id (Hash), topic (Fulltext), started_at (BTree)
Vector index: embedding (computed: topic initially, topic + summary after session ends)

```
FOR_TASK            Session → Task          -- which task this session works on
FOR_GOAL            Session → Goal          -- direct link if no task
PARTICIPATED_IN     Participant → Session
  role              String                  -- "initiator", "responder", "observer"
```


## Layer 2: Episodic — Messages, Actions, Episodes

### Message — atomic unit of communication

```
Message
  message_id        String      not null
  content           String      not null
  content_type      String                  -- "text", "code", "image", "tool_result",
                                            --  "error", "system"
  timestamp         DateTime    not null
  embedding         Vector
```

Index: message_id (Hash), timestamp (BTree), content (Fulltext)
Vector index: embedding (auto-embed from content)

```
SENT_BY             Message → Participant
  role              String                  -- "user", "assistant", "system", "tool"

ADDRESSED_TO        Message → Participant

IN_SESSION          Message → Session

NEXT                Message → Message
  gap_ms            Int64
```


### Action — something a participant did beyond talking

```
Action
  action_id         String      not null
  action_type       String      not null    -- "tool_call", "file_read", "file_write",
                                            --  "command_run", "search", "delegate",
                                            --  "api_call"
  input             Json
  output            Json
  status            String                  -- "success", "failure", "partial", "timeout"
  started_at        DateTime
  ended_at          DateTime
  duration_ms       Int64
  error             String
  embedding         Vector
```

Index: action_id (Hash), action_type (Hash), status (Hash),
       started_at (BTree)
Vector index: embedding

```
PERFORMED_BY        Action → Participant
TRIGGERED_BY        Action → Message       -- the message that caused this action
IN_SESSION          Action → Session
PRODUCED            Action → Artifact
NEXT_ACTION         Action → Action        -- action sequences within a session (provenance tracing, not used for procedure promotion — procedures are promoted from Episode FOLLOWED_BY chains)
```


### Episode — a structured learning experience

An episode records the outcome of working on something. It's the
unit that consolidation operates on to derive facts and procedures.

An Action is "I called grep" (low-level operation).
An Episode is "I investigated the auth bug and found the root cause"
(higher-level experience with outcome and learning).

```
Episode
  episode_id        String      not null
  action_type       String      not null    -- "investigate", "implement", "review",
                                            --  "conversation", "memorize", "diagnose"
  outcome           String                  -- "success", "failure", "partial", "inconclusive"
  state             Json                    -- world context at time of episode
  delta             Json                    -- what changed as a result
  importance        Float64                 -- 0.0-1.0, higher = more significant
  timestamp         DateTime    not null
  embedding         Vector                  -- embeds state + action_type + outcome
```

Index: episode_id (Hash), action_type (Hash), outcome (Hash),
       timestamp (BTree)
Vector index: embedding

```
RECORDED_BY         Episode → Participant
TRIGGERED_BY        Episode → Message      -- what message/request started this
FOR_TASK            Episode → Task         -- which task this episode belongs to
IN_SESSION          Episode → Session
INVOLVES            Episode → Action       -- the actions taken during this episode
MENTIONS            Episode → Entity       -- entities referenced
FOLLOWED_BY         Episode → Episode      -- temporal chain of experiences
  gap_ms            Int64
```


## Layer 3: Artifacts — things in the world

```
Artifact
  artifact_id       String      not null
  kind              String      not null    -- "file", "document", "url", "snippet",
                                            --  "config", "image", "audio", "video",
                                            --  "dataset"
  path              String                  -- filesystem path, URL, or identifier
  content           String                  -- text content (null for binary artifacts)
  mime_type         String
  hash              String
  size              Int64
  language          String                  -- for code: "rust", "python", etc.
  created_at        DateTime
  updated_at        DateTime
  text_embedding         Vector      nullable    -- pooled from chunk embeddings (text/code)
  image_embedding        Vector      nullable    -- CLIP/SigLIP (images, video frames)
  audio_embedding        Vector      nullable    -- CLAP (audio, video audio track)
  video_embedding        Vector      nullable    -- LanguageBind/InternVideo (video)
  multimodal_embedding   Vector      nullable    -- ImageBind/ONE-PEACE (unified cross-modal)
```

Index: artifact_id (Hash), path (BTree, Unique), kind (Hash),
       content (Fulltext), language (Hash), mime_type (Hash)
Note: content is nullable for binary artifacts (image, audio, video).
Fulltext index on content only covers text artifacts. Binary artifacts
are searchable exclusively through multimodal embeddings.
Vector indexes:
  text_embedding (HNSW)         — pooled from chunks, not auto-embed
  image_embedding (HNSW)        — computed by vision model
  audio_embedding (HNSW)        — computed by audio model
  video_embedding (HNSW)        — computed by video model
  multimodal_embedding (HNSW)   — computed by unified model


```
Chunk
  chunk_id          String      not null
  text              String      not null
  index             Int64                   -- position within parent
  start             Int64                   -- byte/char offset in parent
  end               Int64                   -- byte/char offset end
  token_count       Int64
  chunk_type        String                  -- "paragraph", "sentence_group", "function",
                                            --  "class", "table", "speaker_turn", "scene",
                                            --  "heading_section", "block"
  language          String                  -- for code: "rust", "python", etc.
  symbol_name       String                  -- for code: function/class name
  speaker           String                  -- for audio: speaker attribution
  heading           String                  -- for docs: section heading context
  mime_type         String                  -- source content type
  embedding         Vector
```

Index: text (Fulltext), chunk_type (Hash), language (Hash),
       symbol_name (Hash), speaker (Hash)
Vector index: embedding (auto-embed from text)

```
HAS_CHUNK           Artifact → Chunk
  index             Int64

HAS_CHUNK           Message → Chunk        -- long messages get chunked too
  index             Int64

CREATED_BY          Artifact → Action      -- which action produced this artifact
MODIFIED_BY         Artifact → Action
  diff_summary      String
```


## Layer 4: Semantic — extracted and consolidated knowledge

### Entity — named things mentioned in messages, artifacts, actions

```
Entity
  entity_id         String      not null
  name              String      not null
  entity_type       String                  -- "person", "place", "org", "concept",
                                            --  "project", "tool", "event", "date"
  first_seen        DateTime
  last_seen         DateTime
  frequency         Int64
  confidence        Float64
  embedding         Vector
```

Index: entity_id (Hash), name (Hash, Fulltext),
       entity_type (Hash)
Vector index: embedding (computed: name + " (" + entity_type + ")" for disambiguation)

```
MENTIONS            Message → Entity
  count             Int64

MENTIONS            Chunk → Entity
  count             Int64

MENTIONS            Action → Entity

MENTIONS            Artifact → Entity
  count             Int64
```


### Observation — a direct statement or perceived fact from a message

Not a derived fact — something someone actually said or the system
directly observed. "Caroline attended an LGBTQ support group."

```
Observation
  observation_id    String      not null
  content           String      not null
  subject           String                  -- who/what it's about
  observed_at       DateTime               -- when in the real world
  confidence        Float64
  embedding         Vector
```

Index: observation_id (Hash), content (Fulltext),
       subject (Hash, Fulltext)
Vector index: embedding (auto-embed from content)

```
OBSERVED_IN         Observation → Message   -- source message
OBSERVED_IN         Observation → Chunk     -- observations from artifact chunks
OBSERVED_DURING     Observation → Episode   -- which episode this was observed in
ABOUT               Observation → Entity
```


### Fact — consolidated from multiple observations over time

"Caroline is pursuing adoption" — derived from observations across
sessions 2, 5, 13, 17, 19.

```
Fact
  fact_id           String      not null
  subject           String      not null
  predicate         String      not null
  object            String
  confidence        Float64
  observation_count  Int64
  valid_at          Btic                    -- temporal validity interval [lo, hi)
  source_rule       String                  -- which Rule derived this fact
  visibility        String                  -- "agent" (default, scoped) or "global" (shared across agents)
  embedding         Vector
```

**Temporal model (BTIC):**

`valid_at` is a single BTIC property — a half-open interval `[lo, hi)`
with per-bound certainty and granularity. This replaces separate
`valid_from`/`valid_until` DateTime fields and eliminates the need
for Graphiti's `expired_at` (uni-db's automatic `_updated_at`
transaction time covers that).

- Active fact: `[2023-05-25, ∞)` — true since the evidence began, still true
- Invalidated fact: `[2023-05-25, 2023-10-15)` — was true during this window only
- Certainty: `approximate` if < 10 supporting observations, `definite` if >= 10
- Granularity: per-bound (day, month, year, etc.) — "sometime in 2022" vs "on 7 May 2023"

**Bitemporal queries:**
```cypher
-- "What is true now?"
MATCH (f:Fact) WHERE btic.contains(f.valid_at, now())

-- "What was true on March 15?"
MATCH (f:Fact) WHERE btic.contains(f.valid_at, datetime('2026-03-15'))

-- "What did we know last week about what was true in January?"
TIMESTAMP AS OF '2026-04-08'
MATCH (f:Fact) WHERE btic.contains(f.valid_at, datetime('2026-01-15'))

-- "When did a fact about Caroline's adoption become known?"
MATCH (f:Fact) WHERE f.subject = 'Caroline' AND f.predicate = 'pursuing'
RETURN f.valid_at, f._created_at   -- valid time vs transaction time
```

**Why BTIC instead of two DateTimes:**
1. Single property, atomic semantics — no split `valid_from`/`valid_until` to keep in sync
2. Certainty metadata — `approximate` vs `definite` based on evidence count
3. Granularity per bound — "2022" (year granularity) vs "7 May 2023" (day granularity)
4. Allen's interval algebra — `btic.overlaps()`, `btic.before()`, `btic.during()` for
   temporal reasoning across facts
5. Transaction time via uni-db `_updated_at` gives us Graphiti's `expired_at` for free

Index: fact_id (Hash), subject (Hash, Fulltext),
       predicate (Hash), confidence (BTree)
Vector index: embedding

```
SUPPORTED_BY        Fact → Observation
  weight            Float64

DERIVED_BY          Fact → Rule            -- which consolidation rule created this
DERIVED_FROM        Fact → Episode         -- episodes whose observations provided evidence for this fact

INVALIDATES         Fact → Fact
  reason            String                  -- why this fact was invalidated
                                            -- invalidation time = hi bound of the
                                            -- invalidated fact's BTIC interval

ABOUT               Fact → Entity

SHARED_FROM         Fact → Fact
  shared_by         String                  -- participant_id who shared it
  shared_at         DateTime
```


### Topic — aggregated knowledge cluster (from Graphiti's Communities)

A cluster of related entities and facts. "Caroline's adoption journey"
spans entities (Caroline, adoption agencies, LGBTQ-inclusive agency),
facts (pursuing adoption, chose inclusive agency, single parent),
and observations across many sessions.

Topics are derived from entity co-occurrence and graph structure,
similar to Graphiti's community detection.

```
Topic
  topic_id          String      not null
  name              String      not null
  summary           String                  -- generated summary of the cluster
  entity_count      Int64
  fact_count         Int64
  embedding         Vector
```

Index: topic_id (Hash), name (Fulltext)
Vector index: embedding (auto-embed from name + summary)

```
BELONGS_TO          Entity → Topic
BELONGS_TO          Fact → Topic
```


### Summary — generated at any level

```
Summary
  summary_id        String      not null
  text              String      not null
  level             String                  -- "session", "task", "goal",
                                            --  "artifact", "entity", "topic"
  generated_at      DateTime
  embedding         Vector
```

Vector index: embedding (auto-embed from text)

```
SUMMARIZES          Summary → Session
SUMMARIZES          Summary → Task
SUMMARIZES          Summary → Goal
SUMMARIZES          Summary → Artifact
SUMMARIZES          Summary → Entity
SUMMARIZES          Summary → Topic
```


## Layer 5: Procedural — what works

### Procedure — proven action sequences

```
Procedure
  procedure_id      String      not null
  name              String      not null
  description       String
  steps             Json
  preconditions     Json                    -- state conditions for applicability
  precondition_rule String                  -- Locy WHERE fragment for matching
  parameters        Json                    -- configurable inputs
  effectiveness     Float64
  use_count         Int64
  success_count     Int64
  failure_count     Int64
  avg_outcome_delta Json                    -- average outcome changes
  status            String                  -- "candidate", "active", "deprecated"
  created_at        DateTime
  last_used_at      DateTime
  embedding         Vector
```

Index: procedure_id (Hash), name (Fulltext), status (Hash)
Vector index: embedding

```
DERIVED_FROM        Procedure → Action     -- the specific actions that form this procedure's steps
DERIVED_FROM        Procedure → Episode    -- the episodes where this action pattern was observed succeeding
OPERATES_ON         Procedure → Entity
USED_IN             Procedure → Task       -- tasks where this procedure was applied
```


### Rule — Locy rules for formal reasoning

```
Rule
  rule_id           String      not null
  name              String      not null
  source            String                  -- Locy source code
  natural_language  String                  -- human-readable description
  source_type       String                  -- "stdlib", "authored", "induced"
  status            String                  -- "active", "demoted", "pruned", "superseded"
  version           Int64
  confidence        Float64                 -- precision×0.4 + recall×0.3 + novelty×0.3
  precision         Float64
  recall            Float64
  coverage          Int64                   -- episodes covered by this rule
  created_at        DateTime
  validated_at      DateTime
  last_scored_at    DateTime
```

Index: rule_id (Hash), name (Hash), status (Hash),
       source_type (Hash)

Lifecycle: Created → Active (direct for stdlib/authored),
Created → Candidate → Active (for induced, after validation).
Active → Demoted (confidence < 0.40) → Pruned (90 days inactive, terminal).
Active → Superseded (terminal, replaced by newer rule).
Stdlib rules exempt from demotion/pruning/supersession.
Confidence decay: `stored_confidence * (0.95^missed_cycles)`.
Re-promotion: confidence > 0.60 (hysteresis).

```
SUPERSEDES          Rule → Rule            -- newer rule replaces older
DERIVED_BY          Fact → Rule            -- which rule derived this fact
COVERS              Rule → Episode         -- episodes this rule applies to
  correct           Int64                  -- 1=true, 0=false
```


## Layer 6: Meta-Memory — tracking how knowledge is managed

The consolidation process itself is observable and should be tracked.
This lets you query "why does this fact exist?" and "what happened in
the last consolidation cycle?"

```
ConsolidationCycle
  cycle_id          String      not null
  agent_id          String      not null
  started_at        DateTime
  completed_at      DateTime
  observations_processed Int64
  episodes_involved Int64
  facts_created     Int64
  facts_reinforced  Int64
  facts_invalidated Int64
  procedures_promoted Int64
  drift_alerts      Int64
```

Index: cycle_id (Hash), agent_id (Hash), started_at (BTree)

```
PROCESSED           ConsolidationCycle → Observation -- observations consumed
INVOLVED            ConsolidationCycle → Episode    -- episodes involved
CREATED             ConsolidationCycle → Fact       -- facts derived
INVALIDATED         ConsolidationCycle → Fact       -- facts closed
PROMOTED            ConsolidationCycle → Procedure  -- procedures created
APPLIED_RULE        ConsolidationCycle → Rule       -- rules evaluated
```


### DeadLetter — failed pipeline tasks

```
DeadLetter
  step              String                  -- which pipeline step failed
  error             String                  -- error message
  node_ref          Int64                   -- the node that couldn't be processed
  retry_count       Int64                   -- how many times retried
  max_retries       Int64                   -- default: 3
  next_retry_at     DateTime                -- computed from backoff
  created_at        DateTime
```

Operations: retry(id), retry_all_pending(), clear(id), clear_all(), list_pending()


## Layer 7: Organization

```
Organization
  org_id            String      not null
  name              String

MEMBER_OF           Participant → Organization
  role              String
  joined_at         DateTime

Team
  team_id           String      not null
  name              String
  purpose           String

PART_OF_TEAM        Participant → Team
TEAM_IN_ORG         Team → Organization
```


## Working Memory — how it works

Working memory is not a node type. It's a **live view** computed for
a specific goal by traversing the graph:

```cypher
MATCH (g:Goal {goal_id: $goal_id})

// Tasks for this goal
OPTIONAL MATCH (t:Task)-[:PART_OF]->(g)

// Episodes (learning experiences) for those tasks
OPTIONAL MATCH (ep:Episode)-[:FOR_TASK]->(t)

// Recent sessions
OPTIONAL MATCH (s:Session)-[:FOR_TASK]->(t)

// Recent messages in those sessions
OPTIONAL MATCH (m:Message)-[:IN_SESSION]->(s)

// Facts derived from episodes
OPTIONAL MATCH (f:Fact)-[:DERIVED_FROM]->(ep)

// Observations from messages
OPTIONAL MATCH (o:Observation)-[:OBSERVED_IN]->(m)

// Observations from artifact chunks (via entity links)
OPTIONAL MATCH (o2:Observation)-[:ABOUT]->(e2:Entity)<-[:MENTIONS]-(m)

// Entities involved
OPTIONAL MATCH (e:Entity)<-[:MENTIONS]-(m)

// Proven procedures
OPTIONAL MATCH (p:Procedure)-[:USED_IN]->(t)

RETURN g, t, ep, s, m, f, o, o2, e, e2, p
ORDER BY m.timestamp DESC
LIMIT $budget
```

Default budget: 50 items. Configurable via the `working_memory(goal_id, budget)` tool. Budget is in node count, not token count.

This gives the agent everything it needs for the goal — recent
messages, learning episodes, derived facts, involved entities,
proven procedures — all connected through the graph.


## Benchmark Mappings

### LoCoMo (conversation recall)

```
Participant (kind: "human", name: "Caroline")
Participant (kind: "human", name: "Melanie")

Session (topic: "LGBTQ support group, career plans",
         started_at: "2023-05-08")
  ←PARTICIPATED_IN── Caroline
  ←PARTICIPATED_IN── Melanie

Message (content: "I went to a LGBTQ support group yesterday...")
  ─SENT_BY→ Caroline
  ─IN_SESSION→ Session 1
  ─MENTIONS→ Entity("LGBTQ support group")
  ─MENTIONS→ Entity("Caroline")
  ─NEXT→ Message(D1:4)

Episode (action_type: "conversation", outcome: "complete",
         state: {"topic": "LGBTQ group, career plans, painting"},
         importance: 0.7)
  ─RECORDED_BY→ Agent("locomo-bench")
  ─IN_SESSION→ Session 1
  ─MENTIONS→ Entity("Caroline")
  ─MENTIONS→ Entity("LGBTQ support group")
  ─FOLLOWED_BY→ Episode(Session 2)

Observation (content: "Caroline attended LGBTQ support group",
             subject: "Caroline", observed_at: 2023-05-07)
  ─OBSERVED_IN→ Message(D1:3)
  ─OBSERVED_DURING→ Episode(Session 1)
  ─ABOUT→ Entity("Caroline")
  ─ABOUT→ Entity("LGBTQ support group")
```

Answering questions:

```cypher
-- "What did Caroline realize after her charity race?" (adversarial)
MATCH (m:Message)-[:MENTIONS]->(e:Entity {name: "charity race"})
MATCH (m)-[:SENT_BY]->(p:Participant)
RETURN p.name, m.content
-- Result: Melanie said it, not Caroline → adversarial detected

-- "When did Caroline go to the LGBTQ support group?" (temporal)
MATCH (o:Observation)-[:ABOUT]->(e:Entity {name: "LGBTQ support group"})
WHERE o.subject = "Caroline"
RETURN o.observed_at
-- Result: 2023-05-07

-- "What activities does Melanie partake in?" (multi-hop)
MATCH (m:Message)-[:SENT_BY]->(p:Participant {name: "Melanie"})
MATCH (m)-[:MENTIONS]->(e:Entity)
WHERE e.entity_type IN ["activity", "event", "hobby"]
RETURN DISTINCT e.name
-- Result: pottery, camping, painting, swimming, running, violin
```


### MemoryAgentBench (memorize → consolidate → query)

```
-- Memorize phase: ingest context chunks
Artifact (kind: "document", content: "Project Aurora uses Rust...")
  ─HAS_CHUNK→ Chunk(text: "The project lead is Sarah Chen...")
  ─HAS_CHUNK→ Chunk(text: "Target p99 latency is 50ms...")

Episode (action_type: "memorize", outcome: "stored",
         state: {"chunk_index": 0, "topic": "project overview"})
  ─MENTIONS→ Entity("Sarah Chen")
  ─MENTIONS→ Entity("Project Aurora")

-- After consolidation:
Fact (subject: "Sarah Chen", predicate: "leads",
      object: "Project Aurora", confidence: 0.9)
  ─DERIVED_FROM→ Episode(memorize)
  ─ABOUT→ Entity("Sarah Chen")

-- Conflict resolution: new chunk says port changed
Fact (subject: "server", predicate: "port", object: "9090",
      confidence: 0.85, valid_from: "2024-02-12")
  ─INVALIDATES→ Fact(port: "8080")
```


### Evo-Memory (sequential learning)

```
-- Task 1: answer question
Episode (action_type: "answer_question", outcome: "failure",
         state: {"question": "What does RAII stand for?"},
         delta: {"correct": false, "expected": "Resource Acquisition..."},
         importance: 0.9)     -- failures get high importance
  ─RECORDED_BY→ Agent
  ─MENTIONS→ Entity("RAII")

-- Task 5: related question (after consolidation)
Episode (action_type: "answer_question", outcome: "success",
         state: {"question": "How does unique_ptr relate to RAII?"},
         delta: {"correct": true},
         importance: 0.6)     -- successes get lower importance
  ─RECORDED_BY→ Agent
  ─MENTIONS→ Entity("RAII")
  ─MENTIONS→ Entity("unique_ptr")

-- Consolidation derived fact from failures + successes:
Fact (subject: "RAII", predicate: "stands_for",
      object: "Resource Acquisition Is Initialization")
  ─DERIVED_FROM→ Episode(Task 1)
  ─DERIVED_FROM→ Episode(Task 5)

-- phase1_only_pct should increase as more facts are derived
-- because recall_context finds facts in Phase 1 (Compact)
-- without needing to search episodes in Phase 2 (Expand)
```


### BEAM (extreme scale)

At 10M tokens, the schema must support:
- 33K+ Messages with fulltext + vector indexes
- Entity deduplication via name Hash index
- Fact retrieval via subject/predicate Hash indexes
- Phase 1 (Compact): Facts + Procedures → fast, indexed
- Phase 2 (Expand): Episodes by entity + time range → indexed
- Phase 3 (Broaden): Fulltext on Chunk.text + Message.content

The scale test is whether Phase 1 covers enough queries
(phase1_only_pct trending up) so most questions never hit
the expensive Phase 3 fulltext scan over 33K messages.


### LongMemEval (knowledge updates + abstention)

```
-- Knowledge update: user changed preference
Message (content: "Actually I switched to VSCode last month")
  ─SENT_BY→ User
  ─MENTIONS→ Entity("VSCode")

Observation (content: "User switched to VSCode", observed_at: March 2024)
  ─OBSERVED_IN→ Message

-- Consolidation detects contradiction with earlier fact:
Fact (subject: "User", predicate: "uses_editor", object: "VSCode",
      valid_from: "2024-03")
  ─INVALIDATES→ Fact(object: "Vim", valid_until: "2024-03")

-- Abstention: question about something never discussed
-- "What color is the user's car?"
-- Graph traversal finds: no Message mentioning "car",
-- no Entity("car"), no Observation about cars
-- → system can definitively say "no information"
```
