# Phase 2: P4 (Consolidation) + P7 (Embedding/Summary) + Facts with BTIC + Memory Decay + Contradiction/Drift + LoCoMo Validation

## Objective

Phase 2 builds the consolidation and embedding pipelines that transform raw
Observations into derived Facts, adds BTIC-based temporal reasoning, memory
decay, contradiction detection, entity drift handling, and the recall cascade.
It culminates with agent tools, a public API facade, and LoCoMo benchmark
validation to prove the MVP works before investing in Phase 3.

## Sub-Phases

| File | Description |
|------|-------------|
| `sub-01-embedding-pipeline.md` | Pipeline 7 (P7): auto-embed (P7a), computed embedding (P7b), artifact pooling (P7c), summarization (P7d) for all node types |
| `sub-02-consolidation-pipeline.md` | Pipeline 4 (P4): derive Facts from Observations, reinforce with evidence, BTIC invalidation, contradiction resolution, drift detection, memory decay, Locy rules |
| `sub-03-recall-cascade.md` | 3-phase recall cascade (Meta-Memory): coverage-gated early exit, MMR deduplication, drift override, RRF hybrid scoring, token budget enforcement |
| `sub-04-agent-tools.md` | Agent-facing tools (lifecycle, knowledge, query), working memory traversal, Locy stdlib rules |
| `sub-05-public-api-mvp.md` | uniko-api facade crate with ergonomic builder APIs, re-exports, and end-to-end MVP integration test suite |
| `sub-06-locomo-validation.md` | LoCoMo benchmark validation proving the full pipeline (P1-P4, P7, recall cascade) delivers measurable uplift |

## Key Milestone

> Facts derived; phase1_only_pct > 0; LoCoMo uplift proven

## Prerequisites

Phase 1 complete -- all pipelines P1-P3 operational, schema defined, KnowledgeBase layer functional.

## Definition of Done

- P7 produces correct embeddings for all node types, enabling vector search across the graph.
- P4 consolidates Observations into Facts with full provenance and BTIC temporal intervals.
- Contradiction detection and resolution work correctly via BTIC invalidation.
- Entity drift is detected and handled.
- Memory decay applies to nodes over time.
- Recall cascade returns relevant, deduplicated results within token budgets.
- Agent tools provide a complete interface for lifecycle, knowledge, and query operations.
- Public API facade is ergonomic and shippable.
- LoCoMo benchmark score validates the system works -- this gate must pass before Phase 3 investment.
- All sub-phase test suites pass.
