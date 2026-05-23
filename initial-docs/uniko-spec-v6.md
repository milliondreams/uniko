# uniko — Specification v6.0

**Cognitive Memory for AI Agents**

April 2026 · Rust-First, Uni-Native

---

## Part I: Vision

### 1. Executive Brief

uniko is a cognitive memory system for AI agents. Built in Rust on Uni — an embedded graph database with OpenCypher queries, vector search, full-text indexing, and Locy logic programming — it gives agents the ability to remember conversations, learn from experience, reason over accumulated knowledge, and improve over time.

**The problem.** AI agents today are stateless. They can retrieve text snippets from vector stores, but they cannot track who said what across sessions, detect when facts change, learn reusable procedures from repeated experience, or explain why they believe something. The memory systems available to them — Mem0, Zep/Graphiti, Cognee, Letta — each solve a piece of this but none provide the complete cognitive stack.

**The solution.** uniko models memory as a typed knowledge graph organized around communication and goals. Messages between participants are the atomic unit. Entities, observations, and facts are extracted and consolidated automatically. Locy rules execute formal reasoning inside the database — no LLM calls at inference time. The LLM pays the cost once (writing rules, extracting entities); Locy pays nothing executing them at query time. Rules persist as long as they remain effective — confidence decays with disuse, and stale rules are demoted and eventually pruned.

**Key differentiators:**

| Capability | uniko | Nearest Competitor |
|---|---|---|
| Embedded, zero infrastructure | Single in-process database | Graphiti needs Neo4j; Mem0 needs Qdrant |
| Formal reasoning (Locy) | Database-native rule execution | All competitors use LLM at query time |
| Hypothetical reasoning | ASSUME/ABDUCE inside the DB | No competitor offers this |
| Bitemporal knowledge (BTIC) | Native temporal intervals with certainty | Graphiti has valid_at/invalid_at but no certainty metadata |
| Conversation-native schema | Message → Observation → Fact pipeline | Mem0/Letta bolt conversation onto flat memory |
| Goal-oriented working memory | Graph traversal across Goal → Task → Session | No competitor organizes memory around goals |

**The cognitive model.** uniko implements five memory types from cognitive science:

| Memory Type | What It Stores | Graph Nodes |
|---|---|---|
| Working Memory | Active goal context (live traversal, not stored) | Goal, Task, Session → Messages, Facts, Entities |
| Episodic Memory | What happened: communications, actions, experiences | Message, Action, Episode |
| Semantic Memory | What we know: extracted and consolidated knowledge | Entity, Observation, Fact, Topic |
| Procedural Memory | What works: proven patterns and formal logic | Procedure, Rule |
| Meta-Memory | How knowledge is managed: consolidation tracking | ConsolidationCycle, recall cascade |


### 2. Purpose & Vision

AI agents need memory that understands communication, not just stores files.

Every meaningful interaction an agent has — with humans, with other agents, with tools — is a communication event. A user says "switch to port 9090." An agent reports "build failed with 3 errors." A tool returns 500 lines of output. These are messages, and they are the ground truth from which all knowledge derives.

**Interaction-first.** Everything in the graph traces back to a message or an action. Entities are extracted from messages. Observations are statements found in messages. Facts are consolidated from observations. Procedures are promoted from repeated episodes. The provenance chain is always: who said what → what was observed → what was learned → what works.

**Goal-oriented.** Agents don't just remember — they remember *for a purpose*. Working memory is not a chat history; it's everything relevant to an active goal, assembled by traversing the graph from Goal → Tasks → Sessions → Messages → Facts → Entities. When a goal changes, working memory recomputes instantly.

**Compile once, query forever.** This is the LLM Wiki insight applied to a graph database. Raw messages are "source code." Consolidation "compiles" them into facts and procedures. The recall cascade queries the compiled knowledge, not the raw messages. The LLM pays the extraction cost once; every subsequent query benefits for free.

**GoalOS alignment.** uniko serves as the internal memory for GoalOS — a platform for goal-native multi-agent systems. GoalOS defines goals and guardrails; agents pursue them collaboratively. uniko provides the memory substrate: working memory per goal, episodic records of what happened, semantic knowledge of the domain, procedural playbooks for what works, and meta-memory that governs retrieval and consolidation.


### 3. Use Cases

**Conversation Memory** (LoCoMo scenario)
Two friends talk across 19 sessions over 6 months — about careers, adoption, hobbies, personal struggles. Later, questions test whether the system can recall specific details: "What did Caroline research?" "When did Melanie paint a sunrise?" "Did Caroline run the charity race?" (trick question — it was Melanie). This requires: message storage with speaker attribution, entity extraction, temporal tracking, multi-session aggregation, and adversarial detection via graph structure (SENT_BY edges).

**Multi-Agent Collaboration** (GoalOS scenario)
A team of agents pursues the goal "Reduce refund cycle time by 40% in 90 days." A DataOps agent queries metrics, an Analyst agent diagnoses root causes, a Strategy agent proposes interventions. Each records episodes and shares observations. Consolidation derives domain facts ("Supplier X has 62% late-delivery probability in Q3-Q4"). Working memory for the goal assembles relevant context from all agents' contributions.

**Codebase Knowledge** (developer agent scenario)
An agent operating on a code repository reads files, runs tests, records what succeeded and failed. Entity extraction identifies functions, classes, modules. Observations capture code relationships. Consolidation derives facts about the codebase ("the auth module depends on the session service"). Procedures capture proven workflows ("investigate → implement → test" as a reusable playbook).

**Enterprise Knowledge Management** (Evo-Memory scenario)
An agent processes a stream of business tasks sequentially. Early on, it has no domain knowledge and fails often. Each attempt is recorded as an episode. Consolidation derives facts from failures and successes. By the 50th task, the agent draws on accumulated knowledge and performs measurably better. The improvement delta (accuracy[late] - accuracy[early]) proves the system adds value.

**Document Comprehension** (MemoryAgentBench scenario)
A system ingests a long document in chunks, then answers questions about it. Chunks become Artifacts with Chunk nodes. NER extracts entities. Observations capture key statements. Consolidation derives facts. When contradicting information appears later in the document, the old fact is invalidated via BTIC and the new fact takes precedence. The system can answer "what port does the server run on?" correctly even when the answer changed mid-document.


---

## Part II: Competitive Landscape

### 4. Market Overview

Six systems define the current agent memory landscape. Each solves a subset of the problem.

| System | Architecture | Strength | Weakness | LoCoMo |
|---|---|---|---|---|
| **Mem0** | Vector + SQLite + optional entity boost | Production simplicity, hybrid scoring, 1.4s latency | No graph, no temporal reasoning | 91.6% |
| **Graphiti** (Zep) | Temporal knowledge graph (Neo4j/FalkorDB) | Best temporal model (valid_at/invalid_at/expired_at), rich query | High LLM cost per ingestion, requires external graph DB | 75-84% |
| **Letta** (MemGPT) | PostgreSQL + agent-managed memory blocks | Best multi-agent support, agent decides what to remember | No structured extraction, no knowledge graph | 74.0% |
| **Cognee** | Graph + Vector + Relational + Cache | Content diversity, feedback-weighted learning | No temporal validity, weak benchmark position | — |
| **LangMem** | LangGraph BaseStore | Prompt optimization (unique feature) | Very slow (59.8s p95), vector-only retrieval | 58.1% |
| **LLM Wiki** | Markdown files on disk | Compile-once insight, human-readable, git-native | Not an agent memory system, no formal data model | — |

**What we learn from each:**
- **Mem0**: Non-LLM NER (spaCy) as primary entity extractor, not LLM. Hybrid scoring (vector + BM25 + entity boost). Inline consolidation during add().
- **Graphiti**: Bi-temporal fact validity with automatic invalidation. Full provenance chain (Episode → Entity/Edge). Prescribed + learned ontology via Pydantic.
- **Letta**: Agent autonomy over memory management outperforms extraction pipelines (74% on LoCoMo with just file tools). Multi-agent groups with shared memory blocks.
- **Cognee**: Feedback-weighted learning — user feedback adjusts memory importance. Two-tier memory (fast cache + permanent graph).
- **LangMem**: Prompt optimization from interaction history. Hot path + background separation.
- **LLM Wiki**: Knowledge should be compiled at ingest, not re-derived per query. Human-readable artifacts have value.

**Where uniko differentiates:**
1. **Embedded, zero infrastructure** — all competitors require external databases (Neo4j, Qdrant, PostgreSQL). uniko runs in-process on uni-db.
2. **Formal reasoning via Locy** — no competitor has database-internal logic programming. All use LLM for inference.
3. **Hypothetical reasoning** — ASSUME creates temporary graph mutations, evaluates, rolls back. Unique to uniko.
4. **BTIC temporality** — native bitemporal intervals with per-bound certainty and granularity. Richer than Graphiti's three-timestamp model.
5. **Conversation-native schema** — Messages, Sessions, Participants as first-class nodes. Not bolted onto a flat memory store.
6. **Goal-oriented working memory** — computed via graph traversal from Goal → Tasks → Sessions. No competitor organizes memory around goals.


---

## Part III: Requirements

### 5. Functional Requirements

**Tier legend:** MVP = ship in Phase 1-2, required for benchmark validation. DIF = competitive differentiator, Phase 3. RES = research track, experimental.

#### Communication (F1-F8)
| ID | Tier | Requirement |
|---|---|---|
| F1 | MVP | Store messages as graph nodes with content, timestamp, content_type, and embedding |
| F2 | MVP | Track speaker attribution via SENT_BY edges from Message to Participant |
| F3 | MVP | Support multiple participants (human, agent, service) with unified Participant type |
| F4 | MVP | Group messages into Sessions with topic, summary, start/end times |
| F5 | MVP | Link sessions to Goals and Tasks via FOR_GOAL / FOR_TASK edges |
| F6 | MVP | Maintain message ordering via NEXT edges with gap_ms |
| F7 | MVP | Chunk long messages (> 1024 tokens) into Chunk nodes for precise retrieval |
| F8 | MVP | Track participant roles per session (initiator, responder, observer) |

#### Goals & Tasks (F9-F14)
| ID | Tier | Requirement |
|---|---|---|
| F9 | MVP | Define goals with title, description, metrics, guardrails, deadline |
| F10 | MVP | Decompose goals into tasks with priority, status, assignment |
| F11 | MVP | Support goal hierarchy via PARENT_GOAL edges |
| F12 | MVP | Support task dependencies and subtasks via DEPENDS_ON / SUBTASK_OF edges |
| F13 | DIF | Compute working memory by traversing Goal → Tasks → Sessions → Messages → Facts → Entities |
| F14 | MVP | Auto-create sessions for participant+goal combinations; close on inactivity timeout |

#### Episodic (F15-F22)
| ID | Tier | Requirement |
|---|---|---|
| F15 | MVP | Record agent episodes as structured tuples: (action_type, outcome, state, delta, importance) |
| F16 | MVP | Track episode temporal chains via FOLLOWED_BY edges with gap_ms |
| F17 | MVP | Record actions (tool calls, file operations) with input, output, status, duration |
| F18 | MVP | Link actions to producing artifacts via PRODUCED edges |
| F19 | MVP | Link episodes and actions to triggering messages via TRIGGERED_BY edges |
| F20 | MVP | Overflow large action outputs to Artifact nodes (> 256 tokens) |
| F21 | MVP | Embed episodes from state topic content, not just action+outcome |
| F22 | DIF | Track full provenance: who did what, when, triggered by which message, producing which artifacts |

#### Content (F23-F30)
| ID | Tier | Requirement |
|---|---|---|
| F23 | MVP | Ingest artifacts (files, documents, URLs, images, audio, video) as graph nodes |
| F24 | MVP | Chunk artifacts by content type: recursive splitting (text), tree-sitter (code), DOM sections (HTML), speaker turns (audio), scenes (video) |
| F25 | RES | Support multimodal embedding: text_embedding (pooled from chunks), image_embedding (CLIP), audio_embedding (CLAP), video_embedding (LanguageBind), multimodal_embedding (ImageBind) |
| F26 | MVP | Track chunk metadata: chunk_type, language, symbol_name, speaker, heading |
| F27 | MVP | Support fulltext search on Chunk.text and Message.content |
| F28 | MVP | Support vector search on all embedding fields |
| F29 | MVP | Deduplicate artifacts by content hash |
| F30 | DIF | Track artifact provenance via CREATED_BY and MODIFIED_BY edges to Actions |

#### Knowledge (F31-F40)
| ID | Tier | Requirement |
|---|---|---|
| F31 | MVP | Extract entities from messages and chunks using local NER (primary) with optional LLM enhancement |
| F32 | MVP | Track entity frequency, first_seen, last_seen, entity_type, embedding |
| F33 | MVP | Link entities to mentions via MENTIONS edges with count |
| F34 | MVP | Extract observations (factual statements) from messages — not questions, greetings, or reactions |
| F35 | MVP | Link observations to source messages (OBSERVED_IN) and entities (ABOUT) |
| F36 | MVP | Consolidate observations into facts with BTIC temporal validity: active facts have [lo, ∞), invalidated facts have [lo, hi) |
| F37 | MVP | Track fact confidence with Laplace smoothing based on observation count |
| F38 | MVP | Detect contradictions: when contradicting observations exceed 40% of total, invalidate the fact by closing its BTIC interval |
| F39 | MVP | Detect entity drift: when an entity accumulates > 4 invalidations in 30 days, flag as unstable |
| F40 | DIF | Cluster entities into Topics via community detection on co-occurrence graph |

#### Procedural (F41-F52)
| ID | Tier | Requirement |
|---|---|---|
| F41 | DIF | Promote recurring action sequences into Procedure nodes (candidate → active → deprecated) |
| F42 | DIF | Track procedure effectiveness (success/failure counts, use_count) |
| F43 | DIF | Support procedure precondition matching via Locy WHERE fragments — procedures only execute when preconditions match the current state |
| F44 | MVP | Store Locy rules as first-class Rule nodes with full lifecycle: Created → Active → Demoted → Pruned/Superseded. Stdlib rules exempt from demotion. |
| F45 | MVP | Ship stdlib rules with full Locy source: `episode_pattern_detector`, `sequence_detector`, `contradiction_detector`, `relevance_decay` |
| F46 | DIF | Support hypothetical reasoning via ASSUME builder: `.assume_fact(s,p,v).then_query(locy).run()` — fork state, apply mutations, query, rollback |
| F47 | DIF | Support abductive reasoning via ABDUCE builder: given conclusion, find minimal set of facts that would make it true |
| F48 | RES | Support MCTS planning via PlanBuilder: tree search over nested ASSUME with UCB1 selection, configurable depth and simulations, Locy rules as evaluation function |
| F49 | RES | Induce new Locy rules automatically: MINE → GENERATE → VALIDATE → PERSIST → MONITOR |
| F50 | MVP | Apply memory decay: `importance * exp(-ln(2) / half_life * age_days)`, configurable half_life_days, prune below threshold |
| F51 | DIF | Support cross-agent knowledge sharing: `share_fact()` promotes facts to global scope, `shared_facts()` retrieves shared facts |
| F52 | DIF | Maintain rule confidence with decay: `stored_confidence * (0.95^missed_cycles)`, demotion/re-promotion hysteresis, pruning after 90 days |

#### Meta-Memory (F53-F62)
| ID | Tier | Requirement |
|---|---|---|
| F53 | MVP | Record consolidation cycles as ConsolidationCycle nodes with counters and edges to affected nodes |
| F54 | MVP | Implement 3-phase recall cascade with IntentProfile and CoverageScore |
| F55 | MVP | Enforce coverage-gated early exit: Phase 1 threshold 0.75, Phase 2 threshold 0.65, configurable |
| F56 | DIF | Support contrastive retrieval mode: Phase 2 retrieves both success AND failure episodes |
| F57 | MVP | Apply MMR deduplication in Phase 2: lambda=0.7, cosine > 0.85 (Jaccard fallback) |
| F58 | MVP | Override Phase 1 early exit when drift detected on referenced entities (force Phase 2+) |
| F59 | MVP | Generate summaries at session, task, goal, entity, and topic levels |
| F60 | MVP | Track phase1_only_pct as the primary scaling signal |
| F61 | DIF | Support NL-to-Cypher with LRU cache, retry, schema introspection, mutation blocking |
| F62 | MVP | Maintain dead-letter queue: failed tasks stored as DeadLetter nodes with retry/clear |

#### Organization (F63-F66)
| ID | Tier | Requirement |
|---|---|---|
| F63 | DIF | Support Organization and Team nodes for multi-tenant grouping |
| F64 | DIF | Link participants to organizations via MEMBER_OF edges with role |
| F65 | DIF | Support cross-agent knowledge sharing through shared sessions and goals |
| F66 | DIF | Enforce basic access control on facts and observations via Meta-Memory policy (Phase 3) |

#### Integration (F67-F72)
| ID | Tier | Requirement |
|---|---|---|
| F67 | DIF | Expose all operations as MCP tools for external LLM agents |
| F68 | RES | Shadow filesystem: sync a directory tree with the graph, watch for changes |
| F69 | RES | Git integration: map commits to provenance, enhanced blame |
| F70 | RES | Semantic shell: enhanced builtins (grep, find, cat, diff, blame, ls, log) using graph search and provenance |
| F71 | DIF | Python binding via PyO3 exposing all layers |
| F72 | MVP | Operate without LLM — degraded but functional: storage, retrieval, Locy reasoning, local NER |

**Summary:** MVP: 47 requirements. DIF: 19 requirements. RES: 6 requirements. Total: 72.


### 6. Non-Functional Requirements

#### Latency Targets

| ID | Operation | Target | Layer |
|---|---|---|---|
| NF1 | Store message (create node + edges) | < 10ms | 1 |
| NF2 | Vector search (top-10) | < 20ms | 1 |
| NF3 | Hybrid search (vector + FTS + graph) | < 50ms | 1 |
| NF4 | Graph traversal (3-hop) | < 5ms | 1 |
| NF5 | Entity extraction (local NER) | < 100ms | 2 |
| NF6 | Observation extraction | < 5s | 2 |
| NF7 | Context bundle assembly (compact-only) | < 30ms | 3 |
| NF8 | Context bundle assembly (all phases) | < 100ms | 3 |
| NF9 | Single ASSUME (hypothetical reasoning) | < 200ms | 1 |
| NF10 | Episode recording | < 30ms | 3 |
| NF11 | Tier-specific queries (episodes, facts, procedures) | < 20ms | 3 |
| NF12 | Drift detection step | < 100ms | 3 |
| NF13 | NL-to-Cypher (API backend) | 200-500ms | 2 |
| NF14 | Entity extraction with LLM | 1-3s | 2 |
| NF15 | Consolidation cycle | < 5s per agent | 3 |
| NF16 | Rule induction cycle | < 30s | 3 |
| NF17 | Working memory traversal | < 200ms | 3 |
| NF18 | File change detection + re-index | < 100ms | 4 |
| NF19 | MCP tool call round-trip | < 50ms overhead | 4 |

#### Scale Targets
- 10K episodes per agent
- 100K messages per session group
- 10M token conversations (BEAM scale)
- 100 active rules per agent
- Commodity hardware (M-series Mac or 8-core Linux)

**Measurement context:** Latency targets are for warm in-memory stores with <10K nodes per label on commodity hardware. Cold-start (first query after process start) and disk-backed stores may exceed these targets by 2-5x. Targets are initial goals validated against NF5-NF8 test runs (13-54ms assembly latency in debug mode on in-memory stores), not SLA guarantees. All benchmarks should be run with `--release` profile. Targets will be revised as system-level measurement matures.

#### Reliability
- Offline operation: NER, observation extraction, consolidation, and recall all function without LLM
- Graceful degradation: LLM-dependent features (NL-to-Cypher, rule induction, LLM entity enhancement) skip cleanly with warnings
- Single-writer, multi-reader via uni-db snapshot isolation

#### Offline Mode

The system functions without an LLM, but at reduced extraction quality:

| Pipeline | Online (with LLM) | Offline (local only) | Quality Impact |
|---|---|---|---|
| P1 Ingest | Fully functional | Fully functional | None |
| P2 NER | Local + LLM enhancement | Local only (spaCy/rules) | Entity recall ~60% vs ~90% |
| P3 Observations | Rule-based + LLM | Rule-based only | Observation recall ~40% vs ~80% |
| P4 Consolidation | Fully functional | Fully functional | Depends on P2/P3 quality |
| P5 Procedures | Fully functional | Fully functional | Depends on episode quality |
| P6 Topics | LLM naming + local detection | Local detection, generic names | Cosmetic only |
| P7 Embedding | Local fastembed | Local fastembed | None |
| P7d Summaries | LLM-generated | Skipped | No summaries generated |
| P8 Rule Induction | LLM generates rules | **Non-functional** | No automatic rule induction |
| NL-to-Cypher | LLM translates | **Non-functional** | Must use structured queries |

Offline benchmark target: LoCoMo retrieval score > 50% (context_contains_answer metric). This validates that the graph structure and local NER produce usable memory without LLM. Answer generation requires an LLM and is not tested offline.


---

## Part IV: Solution Design

### 7. Architecture

#### Four-Layer Cognitive Stack

```
┌──────────────────────────────────────────────────────────────┐
│               Layer 3: Cognitive                               │
│  uniko-memory: PipelineSystem, workers, recall, consolidation  │
│  uniko-cortex: procedures, topics, MCTS, rule induction        │
│  Memory management, reasoning, drift detection                 │
└─────────────────────────┬────────────────────────────────────┘
                          │ uses
┌─────────────────────────▼────────────────────────────────────┐
│               Layer 2: Processing                              │
│  uniko-pipes: Step trait, circuit breaker, retry, DLQ, metrics │
│  uniko-extract: NER, observations, chunking, ingest, embedding │
│  Pipeline machinery, content processing                        │
└─────────────────────────┬────────────────────────────────────┘
                          │ uses
┌─────────────────────────▼────────────────────────────────────┐
│               Layer 1: Store                                   │
│  uniko-store: graph storage, search, Locy runtime              │
│  Hybrid search: vector + full-text + graph traversal           │
│  Locy runtime: rule execution, ASSUME/ABDUCE                   │
│  Auto-embed: uni-db embeds on insert for configured fields     │
└──────────────────────────────────────────────────────────────┘

Layer 4: Integration (uniko-fs, uniko-shell, uniko-mcp)
```

**The layer distinction.** Store (uniko-store) is a graph database with search — it knows nodes, edges, and embeddings but nothing about what they mean. Processing (uniko-pipes, uniko-extract) adds pipeline machinery and content intelligence — step execution, circuit breakers, NER, observation extraction, chunking, embedding. Cognitive (uniko-memory, uniko-cortex) adds memory management and reasoning — consolidation, recall cascade, procedures, rule induction, MCTS planning.

**Design principles:**
1. Strict linear dependency — Layer 3 calls Layer 2 only. Layer 2 calls Layer 1 only.
2. Each layer adds a category of capability — independently valuable.
3. Agents talk to Layer 3 — the Cognitive layer is the agent-facing API.
4. Data lives in one graph — all layers share a single uni-db instance.
5. LLM is optional at every layer — both Layers 2 and 3 degrade gracefully.
6. Rules are data, not code — Locy rules are graph nodes, authored and versioned.
7. No LLM in the hot path — LLM calls happen in background or on explicit user action.
8. Pipelines do the heavy lifting — 8 automated pipelines process data from ingestion to knowledge.
9. Tools supplement pipelines — agents add what pipelines can't infer (episodes, explicit facts).

#### Architectural Decisions

**ADR-1: ID generation.** All `*_id` fields (message_id, entity_id, fact_id, observation_id, etc.) use UUID v7 (time-sortable, monotonically increasing) when not caller-provided. IDs serve as ext_id for uni-db's MERGE/upsert semantics. Exception: `chunk_id` uses deterministic `{parent_id}:{index}` to enable idempotent re-chunking. Caller-provided IDs are accepted but not required — this allows external systems to maintain their own ID space.

**ADR-2: NER runtime.** Local NER runs via ONNX Runtime (`ort` crate) with a lightweight NER model (e.g., distilled spaCy NER exported to ONNX). This keeps the dependency tree pure Rust — no PyO3, no Python runtime. Tree-sitter for code NER is native Rust. LLM-based NER enhancement runs through Xervo (same provider as embeddings). If ONNX is unavailable, falls back to rule-based extraction (regex patterns for proper nouns, dates, numbers).

**ADR-3: Predicate extraction.** For MVP, predicates are extracted from observations using verb-frame patterns:
- "X is Y" → predicate: "is"
- "X attended Y" → predicate: "attended"
- "X wants to Y" → predicate: "wants_to"
- "X prefers Y" → predicate: "prefers"

Rule-based extraction covers ~60% of common frames. LLM enhancement (when available) uses a one-shot prompt: "Given the observation '{content}', extract: subject, predicate, object." Results are normalized to snake_case verb forms.

#### Crate Structure

```
uniko/
├── crates/
│   ├── uniko-store/        # Layer 1: graph storage, search, Locy runtime
│   ├── uniko-pipes/        # Layer 2: pipeline infrastructure — Step trait, circuit breaker, retry, DLQ, metrics
│   ├── uniko-extract/      # Layer 2: content processing — NER, observations, chunking, ingest, embedding
│   ├── uniko-memory/       # Layer 3: memory management — PipelineSystem, workers, recall, consolidation, rules
│   ├── uniko-cortex/       # Layer 3: higher reasoning — procedures, topics, MCTS, rule induction
│   ├── uniko-api/          # Public facade: builders, re-exports (no logic)
│   ├── uniko-fs/           # Layer 4: shadow FS, file watching, git integration
│   ├── uniko-shell/        # Layer 4: semantic shell binary
│   └── uniko-mcp/          # Layer 4: MCP server for external agents
└── bindings/
    └── uniko-py/           # Python binding (PyO3)
```

#### Schema Ownership by Layer

| Layer | Crate(s) | Nodes Owned | Edges Owned |
|---|---|---|---|
| Store (L1) | uniko-store | Participant, Artifact, Chunk | HAS_CHUNK |
| Processing (L2) | uniko-pipes, uniko-extract | Entity, Observation, Summary, Action | MENTIONS, OBSERVED_IN, ABOUT, SUMMARIZES, PERFORMED_BY, TRIGGERED_BY, PRODUCED, CREATED_BY, MODIFIED_BY |
| Cognitive (L3) | uniko-memory, uniko-cortex | Goal, Task, Session, Message, Episode, Fact, Topic, Procedure, Rule, ConsolidationCycle | OWNED_BY, PARENT_GOAL, PART_OF, ASSIGNED_TO, DEPENDS_ON, SUBTASK_OF, FOR_TASK, FOR_GOAL, PARTICIPATED_IN, SENT_BY, ADDRESSED_TO, IN_SESSION, NEXT, RECORDED_BY, INVOLVES, FOLLOWED_BY, OBSERVED_DURING, SUPPORTED_BY, DERIVED_BY, DERIVED_FROM, INVALIDATES, BELONGS_TO, USED_IN, all ConsolidationCycle edges |

**Note on cross-layer edges:** Some edges connect nodes from different architecture layers (e.g., MENTIONS from Episode [L3] to Entity [L2], OBSERVED_DURING from Observation [L2] to Episode [L3]). This is by design — the strict linear dependency applies to *crate calls* (L3 crates call L2 crates), not to graph edges. The graph is a single unified schema; edges cross layers freely.


---

## Part V: Schema

See the full schema specification in the companion document: `schema-v3.md`

The schema defines 16 node types and 35+ edge types organized in 8 layers:

| Layer | Nodes | Purpose |
|---|---|---|
| 0: Participants | Participant | Who communicates and acts |
| 1: Goals & Sessions | Goal, Task, Session | Why things happen and when |
| 2: Episodic | Message, Action, Episode | What happened — communications, operations, experiences |
| 3: Artifacts | Artifact, Chunk | Things in the world — files, documents, media |
| 4: Semantic | Entity, Observation, Fact, Topic, Summary | What we know — extracted and consolidated knowledge |
| 5: Procedural | Procedure, Rule | What works — proven patterns and formal logic |
| 6: Meta-Memory | ConsolidationCycle | How knowledge is derived and maintained |
| 7: Organization | Organization, Team | Multi-tenant grouping |

**Note on layer numbering:** The schema is organized in 8 layers (0-7) for logical grouping of related node types. These are NOT the same as the 4 architecture layers (Store, Processing, Cognitive, Integration). Schema layers group nodes by purpose; architecture layers group crates and capabilities.

**Key design decisions:**
- **BTIC on Facts**: BTIC (Binary Temporal Interval Composite) is a native uni-db data type — a half-open interval `[lo, hi)` in milliseconds with per-bound granularity (ms to millennium) and certainty (definite, approximate, uncertain, unknown). Supports Allen's interval algebra operators (`btic.contains`, `btic.overlaps`, `btic.before`, etc.). See the Uni Black Book for formal type definition. Active facts: `[observed_at, ∞)`. Invalidated facts: `[observed_at, contradiction_time)`. Transaction time via uni-db `_updated_at` replaces Graphiti's `expired_at`.
- **Multimodal Artifacts**: 5 embedding fields per Artifact — text_embedding (pooled), image_embedding (CLIP), audio_embedding (CLAP), video_embedding (LanguageBind), multimodal_embedding (ImageBind). Each with its own HNSW index.
- **Rich Chunks**: Metadata fields for filtered retrieval — chunk_type, language, symbol_name, speaker, heading.
- **Working Memory as traversal**: Not a stored node — computed by traversing Goal → Task → Session → Message → Fact → Entity.
- **HAS_CHUNK from Message**: Long messages (> 1024 tokens) get chunked, not just Artifacts.


---

## Part VI: Pipelines

See the full pipeline specification in the companion document: `pipelines-design.md`

Eight pipelines process data from ingestion to knowledge:

| Pipeline | Timing | What It Does | Maturity |
|---|---|---|---|
| **P1: Ingest** | Sync, < 10ms (msg), < 100ms (artifact) | Store Message/Artifact nodes, chunk by content type, create edges | Production |
| **P2: NER** | Sync/near-sync, < 100ms | Extract entities using local NER (spaCy, rules, tree-sitter). LLM optional. | Production |
| **P3: Observations** | Async, < 5s | Extract factual statements from messages. Flag contradictions. | Production |
| **P4: Consolidation** | Background, periodic | Derive facts from observations, reinforce/invalidate, detect drift, apply Locy rules | Production |
| **P5: Procedure Promotion** | Background, periodic | Detect recurring action sequences → promote to Procedures | Differentiator |
| **P6: Topic Detection** | Background, low frequency | Community detection on entity co-occurrence → create Topics | Differentiator |
| **P7: Embedding & Summary** | Async, continuous | 4 sub-pipelines: auto-embed, computed embed, artifact pooling/multimodal, summarization | Production |
| **P8: Rule Induction** | Background, low frequency | MINE → GENERATE → VALIDATE → PERSIST → MONITOR | **Research** |

### Pipeline Management

See the full pipeline management specification in the companion document: `pipeline-management.md`

The pipeline management system provides:
- **Error isolation**: per-item failure handling with Skip/DeadLetter/Abort policies
- **Retry with backoff**: 3 attempts, exponential backoff (500ms → 30s) for LLM-dependent operations
- **Circuit breaker**: LLM provider protection (5 failures → open 60s → probe → recovery). When open, all LLM-dependent steps fall back to local alternatives (rule-based NER, rule-based observation extraction)
- **Backpressure**: bounded channels (200 ingest, 32 consolidation). Interactive queries preempt background via `biased` select
- **Coordination**: P3 completion notifies consolidation worker. Consolidation triggers on 20 observations OR 15 min timer
- **Cancellation**: CancellationToken hierarchy for graceful shutdown (stop ingest 5s → stop consolidation 10s → force 30s)
- **Observability**: 14 metrics (via `metrics` crate), structured tracing (via `tracing`), health endpoint with per-worker status and circuit breaker state
- **Dead-letter queue**: failed items stored as DeadLetter nodes with automatic retry every 5 minutes

**Agent Tools** supplement pipelines for knowledge only agents can provide:

**Note on episode recording:** Procedural memory (Procedures, Rules) depends on agents recording Episodes via the `record_episode` tool. Agents that don't record episodes will have no procedure promotion and limited rule induction. The system's improvement over time is proportional to the richness of episode recording.

| Tool | What It Does | Why Not a Pipeline |
|---|---|---|
| record_episode | Record a learning experience (action, outcome, state, delta) | Episodes are subjective — the agent decides what's worth recording |
| record_action | Record a tool call or operation | Actions are explicit agent operations |
| add_observation | Record something the pipeline missed | Implicit preferences, behavioral patterns |
| assert_fact | Create a fact directly | User stated definitively — no need to wait for consolidation |
| invalidate_fact | Mark a fact as no longer true | Explicit correction |
| create_goal / create_task | Define objectives | Goals are human/agent-initiated |
| recall | Query memory across all layers | The Meta-Memory entry point |
| working_memory | Get all context for a goal | Goal-scoped traversal |


---

## Part VII: Chunking Strategy

See the full chunking specification in the companion document: `chunking-analysis.md`

Only two node types produce Chunks: **Artifacts** (always) and **long Messages** (> 1024 tokens).

| Content Type | Chunking Strategy | Chunk Metadata |
|---|---|---|
| Text (plain, markdown) | Recursive 400-512 tokens, sentence-boundary aligned, 10-20% overlap | chunk_type, heading |
| Code (Python, Rust, JS) | tree-sitter AST split-then-merge by function/class/struct (cAST approach, +11% over naive) | chunk_type, language, symbol_name |
| HTML/XML | DOM section extraction | chunk_type, heading |
| PDF | Page extraction + recursive text chunking | chunk_type, heading |
| CSV/JSON | Schema-aware row grouping | chunk_type, heading (column names) |
| Audio | Transcribe + speaker-turn chunking | chunk_type, speaker, start, end |
| Video | Scene boundary detection + transcript alignment | chunk_type, speaker, start, end |
| Image | No chunking (atomic) | — |

**Note on chunking strategy (NAACL 2025 finding):** Chunking configuration has as much influence on retrieval quality as embedding model choice. Fixed recursive splitting at 400-512 tokens with sentence-boundary alignment (69% accuracy) outperforms semantic chunking (54%) in benchmarks. Semantic chunking produces inconsistently small fragments (~43 tokens) that lack context. We use recursive splitting as the default, with AST-based splitting for code where structure matters.

Large Action outputs (> 256 tokens) overflow to Artifact nodes via PRODUCED edges, then chunk normally.


---

## Part VIII: Embedding Strategy

See the full embedding specification in the companion document: `embedding-analysis.md`

| Strategy | Nodes | How |
|---|---|---|
| **Auto-embed** (uni-db handles) | Message, Chunk, Observation, Summary | Single source field → uni-db embeds on insert |
| **Computed** (application code) | Entity, Goal, Task, Session, Topic, Fact, Procedure, Episode, Action | Construct embed string from multiple fields, call embed model |
| **Pooled** | Artifact.text_embedding | Mean-pool all chunk embeddings |
| **Multimodal** | Artifact.image/audio/video/multimodal_embedding | Modality-specific models (CLIP, CLAP, LanguageBind, ImageBind) |
| **No embedding** | Participant, Rule, ConsolidationCycle | Queried by indexed fields, not semantic search |

**Critical fix from v5**: Episode embeds the topic extracted from state JSON (e.g., "LGBTQ support group, career plans"), not `"conversation success"`. This makes Phase 2 of the recall cascade functional.

**SOTA context (April 2026)**: Decoder-only LLM backbones (Gemma3, Qwen3, LLaMA) have overtaken BERT-family encoders on MTEB. Top models: Gemini Embedding 001 (68.32 MTEB, 3072d), Qwen3-Embedding-8B (70.58 multilingual). For code: Voyage-code-3 outperforms all alternatives by 13-16% across 32 code retrieval datasets. Matryoshka representation learning is now standard — embed at full dimensions, search at truncated dimensions for speed, with minimal accuracy loss. For multimodal: Gemini Embedding 2 natively maps text/image/video/audio into a single 3072d space, scoring 68.8 on video retrieval.


---

## Part IX: Recall Cascade

The recall cascade is the Meta-Memory retrieval engine. Given a query and token budget, it searches across all memory layers with coverage-gated early exit.

### Three Phases

**Phase 1: Compact** (semantic + procedural facts)
- Vector search on Fact.embedding, Procedure.embedding, Topic.embedding
- Query: `"What did Caroline research?"` → finds Fact(subject=Caroline, predicate=pursuing, object=adoption)
- Coverage threshold: 0.75 — if coverage met, stop here (cheapest path)

**Phase 2: Expand** (episodic recall)
- Vector search on Episode.embedding, Observation.embedding, Session.embedding
- Fulltext search on Message.content, Observation.content
- MMR deduplication (cosine similarity > 0.85 = duplicate; Jaccard word overlap as fallback)
- Contrastive mode: include failure episodes alongside successes
- Coverage threshold: 0.65

**Phase 3: Broaden** (raw content + graph traversal)
- Fulltext search on Chunk.text, Message.content
- Vector search on Chunk.embedding, Message.embedding, Artifact.text_embedding
- Graph traversal: Entity → MENTIONS → Chunk/Message (follow entity links)
- Personalized PageRank from query entities across the knowledge graph (HippoRAG-inspired, NeurIPS 2024) — spreads activation from seed entities to discover multi-hop connections without expensive community summarization
- Always completes (no early exit)

### Coverage Scoring

```
semantic_items = count of items from Semantic + Procedural tiers (Facts, Procedures, Topics)
facet_coverage = semantic_items / max(semantic_items, 3)
mean_score     = mean(item.score for item in results)
diversity      = distinct_tier_count / 5   (max 5 tiers: Semantic, Procedural, Episodic, KB, Provenance)

coverage = 0.4 * facet_coverage + 0.3 * mean_score + 0.3 * diversity
```

### Drift Override

When drift facts (from Pipeline 4 drift detection) match the query's entity references, force Phase 2+ execution even if Phase 1 coverage is sufficient. This ensures queries about unstable entities always check recent episodic evidence.

### Hybrid Scoring

Results from different search methods (vector, fulltext, graph) are fused via Reciprocal Rank Fusion (RRF):

```
score(item) = Σ 1/(k + rank_i) for each retrieval method i
```

where k = 60 (standard RRF constant). Items are then weighted by tier:

| Tier | Weight |
|---|---|
| Semantic (Facts) | 1.0 |
| Procedural (Procedures) | 0.9 |
| Episodic (Episodes, Observations) | 0.7 |
| Store (Chunks, Artifacts) | 0.5 |
| Provenance (Actions, Messages) | 0.4 |

### Cold Start Behavior

At cold start (no Facts, no Procedures, no Topics), all recalls cascade to Phase 3. `phase1_only_pct` begins at 0% and becomes meaningful only after consolidation has run enough cycles to derive Facts. This is expected — the system progressively shifts from raw content retrieval (Phase 3) to compiled knowledge retrieval (Phase 1) as consolidation accumulates evidence.

### Phase 2 vs Phase 3 Message Search

Phase 2 does **vector search** on Message.embedding (semantic similarity). Phase 3 does **fulltext search** on Message.content (BM25 keyword matching). These are different retrieval methods producing different results — a semantic match ("adoption agencies" matches "researching ways to adopt") differs from a keyword match ("adoption" matches "adoption"). Not redundant.

### Token Budget Enforcement

Items are ranked by score × tier_weight, then truncated to fit the token budget. The budget is specified by the caller (default 8192 tokens). Item token count is estimated at ~50 tokens per item.

### IntentProfile Construction

When `recall(query)` is called, the query text is converted into an IntentProfile before search begins:

1. **Embed the full query** → `intent_vec` (vector used for all vector searches)
2. **Extract entity sub-queries** via the same P2 NER path (local, < 100ms):
   - Named entities in the query → entity_refs (e.g., "Caroline" from "What did Caroline research?")
   - Each entity name is embedded separately → facet_vecs (for entity-boosted search)
3. **Count facets** → facet_count = max(len(entity_refs), 1)

For MVP, no facet decomposition beyond entity extraction. The intent_vec drives vector search; entity_refs drive graph traversal in Phase 3; facet_count feeds coverage scoring.

```rust
pub struct IntentProfile {
    pub intent_vec: Vec<f32>,           // embedding of full query text
    pub facet_vecs: Vec<Vec<f32>>,      // per-entity sub-query embeddings
    pub entity_refs: Vec<String>,       // extracted entity names
    pub facet_count: usize,             // max(entity_refs.len(), 1)
}
```

### Retrieval Contract

Per-phase operational definition for debugging and evaluation:

**Phase 1 (Compact):**
- Candidates: vector search on Fact.embedding (top-20), Procedure.embedding (top-10), Topic.embedding (top-5)
- Score: cosine_similarity × tier_weight (Semantic=1.0, Procedural=0.9)
- Normalization: min-max normalize cosine scores to [0,1]
- Fusion: N/A (single source per candidate type)
- Reranking: none
- Abstention: if max(item.score) < 0.3 AND coverage < 0.2 → flag "low confidence"
- Eval metric: Fact-precision@5

**Phase 2 (Expand):**
- Candidates: vector search on Episode.embedding (top-20), Observation.embedding (top-20), Message.embedding (top-10)
- Score: cosine_similarity × tier_weight (Episodic=0.7) × recency_boost (1.0 + 0.1 × recency_rank)
- Normalization: RRF across vector sources with k=60
- Fusion: merge Phase 1 + Phase 2, re-sort by fused score
- Reranking: MMR with lambda=0.7, skip cosine > 0.85
- Contrastive (optional): also retrieve failure-outcome episodes for same entities
- Abstention: if coverage < 0.3 after Phase 2 → proceed to Phase 3
- Eval metric: Observation-recall@20

**Phase 3 (Broaden):**
- Candidates: fulltext BM25 on Chunk.text (top-20) + Message.content (top-10), vector on Chunk.embedding (top-10), graph traversal via Entity → MENTIONS (collect neighbors), PPR from seed entities (damping=0.85, max_iter=20, top-20)
- Score: RRF across all sources with k=60
- Normalization: per-source min-max to [0,1] before RRF
- Fusion: merge all phases, re-sort by fused score × tier_weight
- Reranking: final MMR pass over full bundle
- Abstention: if max(item.score) < 0.15 across ALL phases AND items < 3 → return empty bundle with `abstention: true`
- Eval metric: End-to-end recall@k at budget

**Debugging retrieval misses:**
- Phase 1 recall low but Phase 3 high → consolidation needs improvement (not enough facts derived)
- Phase 1 recall high but answer F1 low → answer synthesis problem, not retrieval
- All phases recall low → content not in graph (ingestion issue) or NER missed entities

### RecallContextBuilder API

```rust
agent.recall_context(intent)
    .limit(15)                    // max items in bundle
    .weights(tier_weights)        // per-tier scoring multipliers
    .recency_window(days)         // filter by time window
    .min_reliability(0.4)         // reliability threshold
    .include_procedures(true)     // include procedural tier
    .include_kb(true)             // include KB/provenance tiers
    .contrastive(false)           // retrieve success+failure episodes
    .assemble()                   // terminal: returns ContextBundle
```


---

## Part IX-B: Locy Reasoning

### Locy Language Features

Uni's Locy framework provides database-native logic programming. These features are available to all layers:

| Feature | What It Does | Example |
|---|---|---|
| `CREATE RULE` | Define reusable logic with MATCH/WHERE/FOLD/YIELD | Pattern detection rules |
| `FOLD` | Aggregation within rules: COUNT, SUM, AVG, MIN, MAX, MNOR, MPROD | Count episode patterns |
| `ALONG/BEST BY` | Multi-hop value propagation across graph edges | Risk propagation |
| `ASSUME { } THEN { }` | Hypothetical reasoning: fork state, mutate, query, rollback | What-if simulation |
| `ABDUCE` | Backward inference: given conclusion, find minimal supporting facts | Explain a decision |
| `EXPLAIN RULE` | Produce auditable derivation trees for any derived fact | Proof traces |
| `similar_to()` | Vector similarity within Locy rules | Semantic matching in rules |

### Standard Rule Library

Four rules ship with uniko-memory. Parameters (`$agent_id`, `$promotion_threshold`, `$contradiction_threshold`) are injected by Layer 3 at execution time.

**Rule 1: relevance_decay** (runs first every cycle — other rules depend on it)
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

### MCTS Planning

The PlanBuilder enables consequence simulation via tree search over nested ASSUME:

```rust
agent.plan("minimize refund cycle time")
    .actions(["batch_approve", "parallel_process", "escalate"])
    .depth(3)                     // max lookahead
    .simulations(50)              // rollouts per node
    .score_rule("risk_propagation")
    .exploration(1.41)            // UCB1 constant
    .state(json!({"backlog": 247, "hour": 18}))
    .run()
```

**Algorithm:**
1. **Selection:** UCB1: `score + exploration * sqrt(ln(parent_visits) / visits)`
2. **Expansion:** Generate children from actions list (or applicable Procedures with matching preconditions)
3. **Simulation:** Nested ASSUME via Processing → Store. Fork store → Locy eval → score → restore.
4. **Backpropagation:** Update visit counts and mean scores up tree.

Returns `PlanResult` with best action path, score, alternatives, tree stats, and proof traces.

### Rule Lifecycle

```
CREATED → (validate) → CANDIDATE → (promote) → ACTIVE
CREATED (stdlib/authored) → ACTIVE (direct)
ACTIVE → (demote) → DEMOTED → (prune) → PRUNED (terminal)
ACTIVE → (supersede) → SUPERSEDED (terminal)
```

- Stdlib rules: exempt from demotion/pruning/supersession
- Confidence decay: `stored_confidence * (0.95^missed_cycles)` per cycle
- Demotion threshold: confidence < 0.40
- Re-promotion threshold: confidence > 0.60 (hysteresis prevents oscillation)
- Pruning: no matches for 90 days
- Agent-scoped rules shadow global stdlib (same name takes precedence)


---

## Part X: Testing Strategy

### 15. Testing Approach

#### Unit Tests (per crate)

**uniko-store:** Storage operations (node/edge CRUD), search (vector, fulltext, hybrid), Locy runtime (rule execution, ASSUME, ABDUCE), schema registration (every property, every index).

**uniko-pipes:** Step trait execution, circuit breaker (open/close/probe), retry with backoff, dead-letter queue, metrics emission.

**uniko-extract:** Entity extraction (local NER accuracy, code entity detection via tree-sitter, LLM enhancement path), observation extraction (statement classification, subject extraction, observed_at computation), chunking (recursive, tree-sitter, DOM), ingest pipeline, embedding computation (correct source field per node type).

**uniko-memory:** Consolidation (pattern detection, fact derivation, reinforcement, contradiction, drift, oscillation), recall cascade (Phase 1/2/3 execution, coverage scoring, early exit, drift override), PipelineSystem workers, episode recording (FOLLOWED_BY chain, entity linking, embedding computation), stdlib rules.

**uniko-cortex:** Procedure promotion (sequence detection, threshold, lifecycle), topic detection, MCTS planning, rule induction (MINE/GENERATE/VALIDATE/PERSIST), working memory traversal, NL-to-Cypher (schema awareness, mutation blocking, retry logic).

#### Integration Tests

| Test | What It Validates |
|---|---|
| Message → NER → Observation → Fact | Full pipeline from raw input to derived knowledge |
| Contradiction detection | Conflicting messages → BTIC invalidation of old fact |
| Multi-session aggregation | Messages across sessions → correct temporal queries |
| Speaker attribution | Messages from different participants → SENT_BY edges correctly distinguish who said what |
| Working memory assembly | Goal with tasks → traversal returns relevant context from all layers |
| Offline mode | No LLM → local NER works, rule-based observations work, Locy consolidation works |

#### Major Test Scenarios

1. **Single-hop recall** — Ingest 10 messages, ask a question answerable from one → correct answer found via Observation or Fact
2. **Temporal reasoning** — Ingest messages with date references, ask "when did X happen?" → correct date derived from Observation.observed_at and session timestamps
3. **Adversarial attribution** — Ingest messages from A and B, ask "did A say X?" when B said it → system correctly identifies B via SENT_BY → Participant edge
4. **Multi-hop aggregation** — Ask "what activities does X do?" → aggregate Entity mentions across all sessions via MENTIONS edges
5. **Knowledge update** — Ingest contradicting messages → old Fact invalidated (BTIC hi closed), new Fact created with updated information
6. **Abstention** — Ask about something never mentioned → no Entity, no Observation, no Fact exists → system reports "no information"
7. **Consolidation improvement** — Record N episodes → run consolidation → verify Facts derived → recall cascade Phase 1 coverage increases → phase1_only_pct trends upward
8. **Procedure promotion** — Record 5+ episodes with same action sequence → Procedure created (candidate) → 3 more successes → promoted to active
9. **Working memory** — Create Goal with Tasks → Sessions with Messages → working_memory(goal_id) returns relevant Messages, Facts, Entities, Procedures
10. **Multimodal** — Ingest image + text document → both searchable → text query finds image via multimodal_embedding
11. **Scale** — 10K messages, 1K entities → all latency targets met (NF1-NF19)
12. **Offline** — No LLM available → NER via spaCy/rules, observations via rule-based extraction, consolidation via Locy stdlib rules → system produces valid Facts


---

## Part XI: Benchmarking Strategy & Targets

### 16. Benchmark Suite

#### Why We Benchmark

The purpose is to prove three claims:
1. Agents make better decisions with uniko's multi-layer memory than with flat retrieval.
2. Agents improve over time as memory accumulates.
3. uniko provides capabilities flat systems cannot: speaker attribution, temporal reasoning, causal chains, contradiction detection.

All benchmarks run at **fixed token budgets** (8K, 16K, 32K) to force memory architecture to matter.

#### Benchmark Inventory

| Benchmark | What It Tests | Questions | Scale |
|---|---|---|---|
| **LoCoMo** | Cross-session conversation recall: 5 question types (single-hop, temporal, multi-hop, open-ended, adversarial) | 1,986 | 10 conversations, 5,882 turns |
| **LongMemEval** | 5 memory abilities: extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention | 500 | ~115K-500K tokens per instance |
| **MemoryAgentBench** | 4 competencies: accurate retrieval, test-time learning, long-range understanding, conflict resolution | 2,071 | 100K-1.4M tokens |
| **BEAM** | Extreme scale: 10 memory abilities at 128K, 500K, 1M, 10M tokens | 400 | Up to 10M tokens |
| **Evo-Memory** | Self-evolving memory: agent improves over sequential task stream | 10 datasets | 8 diverse task types |

#### Targets

| Benchmark | Current (v5, broken) | Target (v6) | Competitor Best |
|---|---|---|---|
| **LoCoMo** | 22.2% (gpt-4o-mini) | 75%+ | Mem0: 91.6% |
| **LongMemEval** | 33.3% (partial) | 70%+ | Zep: ~75% |
| **MemoryAgentBench** | 0% (broken schema) | 60%+ | — |
| **BEAM (1M)** | 0% (broken schema) | 50%+ | Hindsight: 64.1% |
| **Evo-Memory** | 0% (broken schema) | delta > 0 | — |

#### Key Metrics

| Metric | What It Measures | Target |
|---|---|---|
| **phase1_only_pct** | % of recalls satisfied by Phase 1 (Compact) alone | Must trend upward as knowledge accumulates |
| **compression_ratio** | Facts derived / observations processed | > 0.1 |
| **assembly_latency** | Context bundle assembly time | < 100ms |
| **generation_latency** | LLM answer synthesis time | < 2s with API model |
| **improvement_delta** | accuracy[late tasks] - accuracy[early tasks] | > 0 (Evo-Memory) |
| **causal_chain_score** | Fraction of expected provenance links in context | > 0.5 |


---

## Part XII: Shipping Phases

### 17. Implementation Order

| Phase | Tier | What Ships | Key Milestone |
|---|---|---|---|
| **1** | MVP | Schema (all node types, all indexes) + P1 (ingest) + P2 (NER) + P3 (observations) + basic recall + stdlib rules + offline mode | Messages → Entities → Observations searchable |
| **2** | MVP | P4 (consolidation) + Facts with BTIC + P7a/7b (embedding) + P7d (summaries) + memory decay + contradiction/drift detection | Facts derived; phase1_only_pct > 0 |
| **3** | DIF | Episodes + Actions + Procedures + P5 + P6 + authored rules (via `add_rule`) + basic access control + working memory traversal + NL-to-Cypher | Procedural memory; access control; benchmark validation |
| **4** | DIF | Benchmark harness + full LoCoMo/LongMemEval run + ASSUME/ABDUCE builders + contrastive retrieval + MCP + Python binding | Published benchmark numbers; prove LoCoMo uplift |
| **5** | RES | FS/Shell/Git integration + cross-agent sharing + organization/team support | Layer 4 integration surfaces |
| **6** | RES | P8 (rule induction) + MCTS planning + multimodal embedding + audio/video chunking | Research extensions |

Each phase is independently shippable. **Phases 1-2 are the MVP.** Prove LoCoMo/LongMemEval uplift before investing in Phase 3+. Ship authored rules before automatic rule induction. Make hybrid retrieval concrete before MCTS planning.
