# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-16

First public release.

uniko is an embedded, Rust-native cognitive memory system for AI agents. It turns
conversations into structured, searchable, reasoned-over knowledge — with no LLM in the
recall hot path. It links into the host process like SQLite and requires no external
infrastructure (no Neo4j, no Qdrant, no PostgreSQL).

### Added

- **Embedded cognitive memory store on uni-db.** A single in-process graph database
  (graph + vector + full-text + Locy logic programming) backs the entire system. The
  `uniko-store` crate is the sole boundary to uni-db; all higher layers go through it.
- **Typed knowledge-graph schema** with 22 node types and 50 edge types, organized around
  communication: messages between participants are the atomic unit, and entities,
  observations, facts, procedures, and topics derive from them with full provenance.
- **Atomic ingest pipeline** that compiles each message into the graph in one
  all-or-nothing transaction, idempotent on `message_id`. CPU-side extraction runs first,
  with no LLM in the hot path:
  - local ONNX NLP cascade (kniv-deberta, INT8) producing POS / NER / SRL / DEP / CLS from
    a single encoder pass;
  - rule-based, code-AST, and ONNX named-entity recognition;
  - observation extraction (rules, a YAML rules engine, and SRL frames);
  - recursive text chunking;
  - embedding (BGE-small-en-v1.5, 384d by default; pluggable) via uni-db auto-embed.
- **Asynchronous consolidation** that runs off the ingest hot path:
  - fact derivation from observation clusters (paraphrase collapsing);
  - contradiction detection and entity-drift flagging with bitemporal (BTIC) intervals;
  - procedure promotion from repeated episode/action sequences;
  - topic detection via label-propagation communities (optional LLM naming).
- **Three-phase recall cascade with coverage gating:** Phase 1 over compiled knowledge
  (Facts / Topics / Procedures), Phase 2 hybrid vector + BM25 over
  Episodes/Observations/Messages fused with Reciprocal-Rank Fusion (with optional temporal
  and graph/PPR channels), and Phase 3 full Chunk/Artifact fallback. Query understanding is
  local and rule-based; an optional cross-encoder reranker rescores the fused results, which
  assemble into a token-budgeted context bundle.
- **Locy formal reasoning** — database-native rule execution for derived knowledge
  (the sequence detector is the live engine).
- **Agent tools and a query feedback loop:** answered queries can be recorded as Episodes
  (question, answer, recall node IDs, coverage, usage), feeding procedure learning.
- **Visibility and access control** by participant / team / org (`public`, `private:{id}`,
  `team:{id}`, `org:{id}`), applied to recall results through a cached viewer policy.
- **Benchmark harness** (`uniko-bench`) with LoCoMo and LongMemEval runners, microbenchmarks,
  and performance telemetry.

#### Benchmarks (LoCoMo10, full 10 conversations / 1,986 questions)

- LLM-judge score **0.8117** (81.2%, Gemini-3.1), retrieval hit **0.8555** (85.6%),
  F1 **0.321**, total LLM cost **$3.55**.
- Ingest of 5,882 turns in **7.5 minutes** at **$0** (local ONNX only), ~76 ms/turn
  wall-clock.
- Mean Q&A latency **4.04 s** — the fastest of six systems compared (vs Mem0, Graphiti,
  Cognee, RAG, and Full-Context).

### Known limitations

- **No HTTP or MCP API.** uniko ships as a Rust library only; there is no network surface
  or agent-facing tool server yet.
- **No CLI.** There is no command-line interface.
- **Python bindings are a non-functional skeleton.** The `uniko-py` crate exists as a PyO3
  scaffold and is not yet usable.
- **Some recall paths are still improving.** Date-anchored / temporal questions are the
  largest known failure category, and retrieval tuning is ongoing.

[0.1.0]: https://github.com/rustic-ai/uniko/releases/tag/v0.1.0
