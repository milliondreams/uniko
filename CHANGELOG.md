# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-07-08

**First public release.**

uniko is an embedded, Rust-native cognitive memory system for AI agents. It turns
conversations into structured, searchable, reasoned-over knowledge — with **no LLM in
the recall hot path**. It links into the host process like SQLite and requires no
external infrastructure (no Neo4j, no Qdrant, no PostgreSQL): a single in-process
graph + vector + full-text store with a Locy logic engine. Compile knowledge in once
at write time; query it forever.

Built on [`uni-db`](https://crates.io/crates/uni-db) (embedded multi-model graph
database) and `uni-xervo` (model runtime for embeddings, NLP, and reranking).

### Distribution & packaging

- **Rust crates on crates.io.** Six layered, independently-versioned crates, all
  sharing one workspace version:
  - `uniko-store` — Layer 1: graph storage, search, and the Locy runtime (the sole
    boundary to `uni-db`).
  - `uniko-pipes` — Layer 2: pipeline infrastructure (Step trait, circuit breaker,
    retry, dead-letter queue, metrics).
  - `uniko-extract` — Layer 3: content processing (NER, observations, chunking,
    ingest, embedding).
  - `uniko-memory` — Layer 4: memory management (pipelines, recall cascade, rules,
    consolidation).
  - `uniko-cortex` — Layer 5: higher reasoning (procedure promotion, topic detection).
  - `uniko-api` — the public facade: builders and re-exports, no logic.
- **Three prebuilt Python wheel variants**, all importing the same `uniko` package
  (one installable at a time) — this lifts 0.1.0's "build from source" requirement:
  - **`uniko`** — CPU, statically-bundled ONNX Runtime. Self-contained, zero runtime
    configuration.
  - **`uniko-cuda`** — NVIDIA CUDA: ONNX Runtime CUDA execution provider **plus**
    on-GPU local LLM inference via mistralrs + candle CUDA kernels (Ampere baseline,
    forward-compatible to Ada/Hopper/Blackwell). Linux x86_64.
  - **`uniko-metal`** — Apple Silicon: ONNX Runtime CoreML **plus** mistralrs + candle
    Metal kernels for on-GPU local LLM. macOS arm64.
  - Wheels are `abi3` for CPython ≥ 3.10 (one wheel per platform); GPU wheels resolve
    the host CUDA/cuDNN libraries at runtime rather than bundling them.
- **Single-sourced versioning.** Every crate, the Python distribution version, and the
  runtime `uniko.__version__` derive from one `[workspace.package].version`; CI guards
  the internal dependency pins and the wheel `pyproject.toml` files against drift.

### Added — cognitive memory engine

- **Embedded store on `uni-db`.** A single in-process graph database (graph + vector +
  full-text + Locy logic programming) backs the whole system; `uniko-store` is the only
  crate that touches it.
- **Typed knowledge-graph schema** — 24 node types and 53 edge types, organized around
  communication: messages between participants are the atomic unit, and entities,
  observations, facts, procedures, and topics derive from them with full provenance.
- **Atomic, LLM-free ingest.** Each message compiles into the graph in one
  all-or-nothing transaction, idempotent on `message_id`. CPU-side extraction runs
  first, with no LLM in the hot path:
  - local ONNX NLP cascade (`kniv-deberta`, INT8) producing POS / NER / SRL / DEP / CLS
    from a single encoder pass;
  - rule-based, code-AST, and ONNX named-entity recognition;
  - observation extraction (rules, a YAML rules engine, and SRL frames);
  - recursive text chunking;
  - embedding (BGE-small-en-v1.5, 384-d by default; pluggable) via `uni-db` auto-embed.
- **Entity-quality admission and canonicalization.** Strict admission (on by default)
  drops NER noise — temporal, measurement, and quoted-string spans already captured as
  observations, plus greeting/discourse fragments mis-tagged as people — and gates the
  low-confidence catch-all. Canonical text normalization collapses case and punctuation
  variants (`"Melanie."` / `"melanie"` → `melanie`) so a name keys identically across
  rule-based and ONNX NER, `ABOUT` edges, and consolidation, with a cross-source dedup
  cascade over overlapping mentions.
- **Asynchronous consolidation**, off the ingest hot path:
  - fact derivation from observation clusters (paraphrase collapsing), with
    `SUPPORTED_BY` edges weighted by Fact↔Observation cosine;
  - contradiction detection and entity-drift flagging over bitemporal (BTIC) intervals;
  - procedure promotion from repeated episode / action sequences;
  - topic detection via label-propagation communities (optional LLM naming, offline).
- **Three-phase recall cascade with coverage gating:** Phase 1 over compiled knowledge
  (Facts / Topics / Procedures) → Phase 2 hybrid **dense vector + BM25** over
  Episodes / Observations / Messages fused with Reciprocal-Rank Fusion (with optional
  temporal and graph / Personalized-PageRank channels) → Phase 3 full Chunk / Artifact
  fallback. Query understanding is local and rule-based; an optional **cross-encoder
  reranker** (MiniLM) rescores the fused results, and everything assembles into a
  token-budgeted context bundle.
- **Locy formal reasoning** — database-native rule execution for derived knowledge; the
  stdlib sequence detector is the live engine.
- **Agent tools and a query feedback loop.** Answered queries can be recorded as
  Episodes (question, answer, recall node IDs, coverage, usage), feeding procedure
  learning.
- **Visibility and access control** by participant / team / org (`public`,
  `private:{id}`, `team:{id}`, `org:{id}`), applied to recall through a cached viewer
  policy.
- **Python SDK (alpha).** `import uniko` exposes an async-first surface —
  `Uniko → agent(id) → Agent → session(id) → Session`, with verbs observe / recall /
  answer / query / ingest / forget, goals / tasks, and Locy logic — plus a blocking
  `*_sync` twin for every I/O call, a Pydantic IO layer, and a complete `py.typed`
  stub. Requires Python ≥ 3.10; runtime dependency `pydantic ≥ 2`.
- **Benchmark harness** (`uniko-bench`) with LoCoMo and LongMemEval runners,
  microbenchmarks, and performance telemetry.

#### Benchmarks

Self-measured on the full **LoCoMo10** set (10 conversations, 1,986 questions; 22-core
CPU + 8 GB consumer GPU). These are internal figures from the repo's own harness, not a
third-party leaderboard.

- LLM-judge score **0.8117** (81.2%, Gemini-3.1); retrieval hit **0.8555** (85.6%);
  F1 **0.321**; total LLM cost (answer + judge) **$3.55**.
- Ingest of 5,882 turns in **7.5 minutes at $0** (local ONNX only), ~76 ms/turn.
- Mean Q&A latency **4.04 s** (2.84 s recall + 1.20 s generation) — the fastest of six
  systems compared (vs Mem0, Graphiti, Cognee, RAG, and Full-Context), and 33–76×
  faster per-turn ingest than Graphiti / Cognee at $0 ingest cost.

### Known limitations

- **No HTTP or MCP API.** uniko ships as a Rust library (plus the Python SDK); there is
  no network surface or agent-facing tool server yet.
- **No CLI.**
- **Python SDK is alpha.** The full async surface ships with `*_sync` twins and typed
  stubs, but the API may still change before 1.0.
- **GPU wheels build the local-LLM stack from a pinned mistralrs/candle** (crates.io is
  frozen at mistralrs 0.8.1); building them requires a CUDA (13.x) or Metal toolchain.
  The CPU `uniko` wheel needs neither.
- **Some recall paths are still improving.** Date-anchored / temporal questions are the
  largest known failure category; retrieval tuning is ongoing.
- **Not yet shipped** (design / roadmap only): HTTP/MCP server, CLI, sparse / ColBERT
  late-interaction retrieval, multimodal ingest, rule induction, and cross-agent
  sharing.

### Installation

Rust:

```toml
[dependencies]
uniko-api = "0.1.1"
```

Python (CPU):

```sh
pip install uniko
```

GPU variants (install exactly one; all import as `uniko`):

```sh
pip install uniko-cuda    # NVIDIA CUDA, Linux x86_64
pip install uniko-metal   # Apple Silicon, macOS arm64
```

[0.1.1]: https://github.com/rustic-ai/uniko/releases/tag/v0.1.1
