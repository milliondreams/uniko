# Competitor Analysis: AI Agent Memory Systems

**Note:** This analysis was conducted during the v5→v6 redesign and reflects the state of uniko before schema-v3, the interaction-first architecture, and the 8-pipeline design. The "What's Blocking uniko" section at the bottom describes issues that have been addressed in the v6 spec — the schema is now complete, NER has a local fallback, Episode embedding is fixed, and Message/Session/Participant are first-class nodes. This document is retained as historical context for design decisions.

## Benchmark Leaderboard (LoCoMo)

| System | LoCoMo Score | Approach |
|--------|-------------|----------|
| Mem0 (token-efficient) | 91.6% | Vector + BM25 + entity boost |
| Zep/Graphiti | 75-84% (disputed) | Temporal knowledge graph |
| Letta/MemGPT | 74.0% | Agent-directed file-based memory |
| Mem0 (standard) | 66.9-68.5% | Vector + optional graph |
| LangMem | 58.1% | Vector similarity only |
| **uniko (gpt-4o-mini)** | **22.2%** | NL-to-Cypher on broken schema |


## Architecture Comparison

| System | Storage | Knowledge Model | Temporal | Multi-Agent |
|--------|---------|----------------|----------|-------------|
| **Mem0** | Vector store + SQLite | Flat text memories | None | Metadata scoping |
| **Graphiti** | Neo4j/FalkorDB/Kuzu | Entities + Fact edges + Episodes + Communities + Sagas | **Best**: valid_at/invalid_at/expired_at | group_id only |
| **Cognee** | Graph + Vector + Relational + Cache | DataPoints with versioning | Version tracking only | Multi-tenant |
| **Letta** | PostgreSQL | Core memory blocks + archival + recall | None | **Best**: Groups, shared blocks, sleeptime |
| **LangMem** | LangGraph BaseStore | Namespaced key-value | None | Namespace sharing |
| **uniko** | uni-db (graph + vector + fulltext + Locy) | Graph nodes + vector + fulltext | BTIC temporal intervals | Single agent only |


## What Each Does Best

### Graphiti — Temporal fact management
- Every fact has `valid_at`, `invalid_at`, `expired_at`
- Automatic invalidation when contradictions detected
- Full provenance: Episode → Entity/Edge lineage
- Richest query model: semantic + BM25 + BFS, 5 rerankers
- Saga nodes for session-level summarization

### Mem0 — Production simplicity + hybrid scoring
- Semantic + BM25 + entity boost scoring
- 20+ vector store backends
- spaCy NER (non-LLM) for entity extraction
- 1.4s p95 latency
- Simple API: add, search, get, update, delete

### Letta — Agent autonomy + multi-agent
- Agent decides what to remember (no extraction pipeline)
- 74% on LoCoMo with just file tools
- Groups with multiple coordination patterns (round_robin, supervisor, sleeptime, swarm)
- Shared memory blocks between agents
- Self-modifying system prompts

### Cognee — Content diversity + feedback learning
- Handles text, audio, images, video, code, 3D models
- Feedback-weighted learning (user feedback adjusts memory importance)
- Session memory → graph sync (two-tier)
- Pipeline-based, composable

### LangMem — Prompt optimization
- Unique: optimizes agent prompts based on interaction history
- Clean functional API
- Hot path + background separation


## What to Learn From Competitors

### 1. Graphiti's temporal model
Our schema-v3 has `valid_from`/`valid_until` on Facts — similar to Graphiti.
But Graphiti also has `expired_at` (when the edge was superseded in the system,
separate from when the fact stopped being true in reality). We should consider
this three-timestamp model.

### 2. Mem0's non-LLM NER
Mem0 uses spaCy for entity extraction as a fast, reliable fallback.
Our enrichment pipeline depends entirely on LLM — when the LLM provider
fails, NER stops. We should add a local NER model or rule-based extractor.

### 3. Graphiti's prescribed ontology
Graphiti lets you define entity and edge types via Pydantic models.
The system knows what types exist before extraction, making the graph
more structured. Our NL-to-Cypher has a hardcoded schema — we should
make it dynamic and ontology-aware.

### 4. Letta's agentic memory management
Letta doesn't extract knowledge automatically — the agent decides
what to store. This is surprisingly effective (74% LoCoMo). It suggests
that giving agents tools to manage their own memory may outperform
automated extraction pipelines.

### 5. Mem0's hybrid scoring
Semantic + BM25 + entity boost with adaptive normalization.
Our Phase 3 (Broaden) does fulltext search, but we don't combine
vector and keyword scores. RRF or adaptive fusion would help.

### 6. Cognee's feedback loop
User feedback adjusts memory importance. None of the other systems
do this. It's a mechanism for the system to learn what matters
from the human's perspective.

### 7. Graphiti's community detection
Entity clusters with generated summaries. Useful for "tell me
everything about topic X" queries where you need to aggregate
across many entities.


## Where uniko Can Differentiate

### 1. Embedded, zero-infrastructure
All competitors require external services:
- Graphiti needs Neo4j/FalkorDB
- Mem0 needs a vector store (Qdrant, Pinecone, etc.)
- Zep is a managed SaaS
- Cognee needs Graph + Vector + Relational DBs

uniko runs entirely in-process on uni-db. No external databases,
no network calls for storage. This is a genuine differentiator
for edge deployment, privacy-sensitive use cases, and embedded agents.

### 2. Formal reasoning via Locy
No competitor has database-internal logic programming. Graphiti
uses LLM for all inference. Mem0 uses LLM during add().
uniko can execute Locy rules inside the database — inference
cost paid once, amortized across all future queries.

### 3. Hypothetical reasoning (ASSUME/ABDUCE)
No competitor can do "what if" simulation. ASSUME creates
temporary graph mutations, evaluates, and rolls back.
This is unique to uniko via uni-db's Locy framework.

### 4. Single database for everything
uni-db provides graph + vector + fulltext + columnar + Locy
in one embedded database. Competitors stitch together 2-4
separate systems (Neo4j + Qdrant + PostgreSQL + Redis).

### 5. Bitemporal model
uni-db has native BTIC temporal intervals. Graphiti implements
bi-temporal manually with DateTime fields. uniko can do it
at the database level with native operators.


## What's Blocking uniko From Competing

Based on our benchmark runs and competitor analysis:

1. **Schema is incomplete** — missing properties, indexes, node types
   (Episode, Action). Competitors have complete, working schemas.

2. **NER depends on LLM** — when provider fails, no entities extracted.
   Mem0 uses spaCy as fallback. We have nothing.

3. **Episode embedding is wrong** — embeds "action outcome" instead of
   content. Every episode is the same vector point.

4. **No conversation model** — no Message, Session, Thread nodes.
   Competitors (Zep, Letta, Cognee) all have session management.

5. **NL-to-Cypher is fragile** — hardcoded schema, bad few-shot examples,
   fails on newer API models. Competitors use simpler retrieval methods
   that work more reliably.

6. **No hybrid scoring** — no RRF, no BM25+vector fusion, no reranking.
   Mem0 and Graphiti both use sophisticated scoring.

7. **Single agent only** — no multi-agent support at all. Letta has
   groups, shared blocks, multiple coordination patterns.
