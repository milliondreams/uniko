# Phase 1: Schema + Pipelines P1-P3 + Basic Recall + Stdlib Rules + Offline Mode

## Objective

Phase 1 establishes the entire project foundation -- workspace structure, schema
definitions, the KnowledgeBase storage layer, pipeline infrastructure, and the
first three pipelines (Ingest, Entity Extraction, Observation Extraction). By
the end of this phase, messages flow through P1-P3 to produce searchable
Entities and Observations in the graph, with basic recall, stdlib Locy rules,
minimal embedding support, and offline operation all functional.

## Crate Structure

Phase 1 creates all crates with strict linear dependency enforcement:

```
uniko-store    → graph CRUD, search, Locy runtime (wraps uni-db)
    ↑
uniko-pipes    → Step trait, circuit breaker, retry, DLQ, health, metrics
    ↑
uniko-extract  → NER, observations, chunking, ingest steps, embedding
    ↑
uniko-memory   → PipelineSystem, workers, recall, rules management
    ↑
uniko-cortex   → procedures, topics, MCTS, rule induction (Phase 3+)
    ↑
uniko-api      → agent-facing facade, re-exports
```

## Sub-Phases

| File | Description |
|------|-------------|
| `sub-01-project-foundation.md` | Cargo workspace with 6 core crates, dependency wiring, CI/CD, shared infrastructure (errors, IDs, config) |
| `sub-02-schema-and-types.md` | All 16+ node types, 35+ edge types, BTIC temporal integration, indexes (Hash, BTree, Fulltext, Vector), and schema test suite in `uniko-store` |
| `sub-03-knowledgebase-layer.md` | KnowledgeBase API in `uniko-store`: typed CRUD, vector/fulltext/hybrid search, graph traversal, Locy runtime |
| `sub-04-pipeline-infrastructure.md` | Generic pipeline machinery in `uniko-pipes` (Step trait, circuit breaker, retry, DLQ, metrics) + orchestration in `uniko-memory` (PipelineSystem, workers) |
| `sub-05-ingest-pipeline.md` | Pipeline 1 (P1) in `uniko-extract`: synchronous ingest for Messages and Artifacts, session management, message ordering, chunking |
| `sub-06-ner-pipeline.md` | Pipeline 2 (P2) in `uniko-extract`: entity extraction (people, orgs, places, concepts, code symbols), deduplication, MENTIONS edges |
| `sub-07-observation-pipeline.md` | Pipeline 3 (P3) in `uniko-extract`: observation extraction, Observation nodes, contradiction flagging for P4, consolidation notifications |
| `sub-08-recall-rules-embedding.md` | Basic recall API in `uniko-memory`, stdlib Locy rule registration, minimal embedding support (auto-embed + entity computed embedding), offline e2e test |

## Key Milestone

> Messages -> Entities -> Observations searchable

## Prerequisites

None. This is the first phase.

## Definition of Done

- Workspace compiles cleanly with correct layering enforced at the Cargo dependency level (store → pipes → extract → memory).
- Schema is fully defined with all node types, edge types, indexes, and BTIC intervals in `uniko-store`.
- KnowledgeBase provides typed CRUD, vector search, fulltext search, hybrid search (RRF), and graph traversal.
- Pipeline infrastructure runs tasks, routes to workers, handles errors, and shuts down gracefully.
- P1 ingests messages and artifacts into the graph with correct session and ordering semantics.
- P2 extracts entities from ingested content and deduplicates against the existing graph.
- P3 extracts observations, wires edges, and flags contradictions for downstream consolidation.
- Minimal embedding support: auto-embed for Message/Chunk/Observation, computed Entity embedding for dedup.
- Basic recall returns relevant results (Messages, Chunks, Observations, Entities) via hybrid search with RRF.
- 4 stdlib Locy rules (relevance_decay, episode_pattern_detector, sequence_detector, contradiction_detector) registered and executable.
- Offline mode operates without ONNX model or LLM -- rule-based NER, rule-based observations, local embeddings all functional.
- Offline end-to-end integration test passes (ingest → NER → observations → recall, no external dependencies).
- All sub-phase test suites pass.

## Notes

- **Episode and Action schema types** are defined in sub-02 but tools to create them (`record_episode`, `record_action`) ship in Phase 2.
- **Session re-embedding** on end (topic + summary) defers to Phase 2 when P7d (summaries) ships. Phase 1 embeds from topic only.
- **Full P7 embedding pipeline** (all computed types, artifact pooling, multimodal) ships in Phase 2. Phase 1 has the minimum needed for entity dedup and basic recall.
- **Consolidation (P4)**, which derives Facts from Observations, ships in Phase 2. Phase 1 flags contradictions but does not resolve them.
