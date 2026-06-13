# uniko — Product, Technology & Status Overview

**Cognitive memory for AI agents. Embedded, Rust-native, zero infrastructure.**

*Status snapshot: June 2026. Sources: `initial-docs/uniko-spec-v6.md`, `bugs/UNI_DB_WORKAROUNDS.md`, `PHASE1_GAPS.md`, benchmark artifacts in `data/` and `initial-docs/bench/`, and a workspace-wide architecture review (2026-06-10).*

---

## 1. What uniko is

uniko is a cognitive memory system for AI agents, built in Rust on **uni-db** — an embedded
multi-model graph database with OpenCypher queries, vector search, full-text indexing, and
Locy logic programming. It gives agents the ability to remember conversations, learn from
experience, reason over accumulated knowledge, and improve over time.

**The problem.** AI agents are stateless. They can retrieve text snippets from vector
stores, but they cannot track who said what across sessions, detect when facts change,
learn reusable procedures from repeated experience, or explain why they believe something.
Existing memory systems (Mem0, Zep/Graphiti, Letta, Cognee, LangMem) each solve a piece;
none provide the complete cognitive stack, and all require external infrastructure
(Neo4j, Qdrant, PostgreSQL).

**The approach.** Memory is a typed knowledge graph organized around communication.
Messages between participants are the atomic unit; everything else derives from them with
full provenance: *who said what → what was observed → what was learned → what works*.
Knowledge is "compiled" at ingest time (entities, observations, facts, procedures), so
queries hit compiled knowledge instead of re-deriving it with an LLM each time — the
"compile once, query forever" principle.

**The cognitive model** (five memory types from cognitive science):

| Memory type | What it stores | Graph nodes |
|---|---|---|
| Working | Active goal context (live traversal, not stored) | Goal, Task, Session → Messages, Facts, Entities |
| Episodic | What happened | Message, Action, Episode |
| Semantic | What we know | Entity, Observation, Fact, Topic |
| Procedural | What works | Procedure, Rule |
| Meta | How knowledge is managed | ConsolidationCycle, recall cascade |

**Strategic context.** uniko is the memory substrate for **GoalOS**, a platform for
goal-native multi-agent systems: working memory per goal, episodic records per agent,
shared semantic knowledge, procedural playbooks, and meta-memory governing retrieval and
consolidation.

---

## 2. Product differentiation

| Capability | uniko | Nearest competitor |
|---|---|---|
| Embedded, zero infrastructure | Single in-process database | Graphiti needs Neo4j; Mem0 needs Qdrant |
| Local NLP extraction in the hot path | ONNX cascade (POS/NER/SRL/DEP/CLS), no LLM per message | Most competitors call an LLM per ingest |
| Formal reasoning (Locy) | Database-native rule execution | All competitors use LLM at query time |
| Hypothetical reasoning | ASSUME/ABDUCE inside the DB (research track) | None |
| Bitemporal knowledge (BTIC) | Temporal intervals with per-bound certainty | Graphiti has valid_at/invalid_at, no certainty |
| Conversation-native schema | Message/Session/Participant first-class | Mem0/Letta bolt conversation onto flat memory |
| Goal-oriented working memory | Graph traversal from Goal → Task → Session | None |

Two differentiators deserve emphasis because they shape the entire architecture:

1. **No LLM in the ingest hot path.** Entity/observation extraction runs a local ONNX
   model cascade (kniv-deberta, INT8-quantized, "xsmall" tier) on consumer hardware. The
   LLM cost is paid optionally and asynchronously (triple refinement, topic naming),
   never per-message. This makes ingest cost predictable and offline-capable.
2. **Embedded deployment.** uniko links into the host process like SQLite does. There is
   no service to operate, no network hop, no separate vector store to keep consistent.

---

## 3. Technology stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Rust (edition 2024, stable toolchain) | mold linker, clippy/deny/rustfmt enforced |
| Database | uni-db (path dep `../uni`), v2.0.0 | Graph + vector + FTS + Locy in one engine; Lance storage backend |
| Model runtime | uni-xervo 0.13 (ONNX Runtime) | Shared `ModelRuntime` across KBs for VRAM efficiency |
| Embeddings | BGE-small-en-v1.5, 384d (default) | Pluggable: nomic, BGE-large, MiniLM, EmbeddingGemma, remote providers |
| NLP cascade | kniv-deberta-nlp xsmall, INT8 | Single encoder pass → POS, NER, SRL (per-verb fan-out), DEP, CLS |
| Reranker | cross-encoder ms-marco-MiniLM (optional) | BGE rerankers and Qwen3 also supported |
| LLM access | via uni-db's xervo provider layer | OpenAI/Gemini/local (mistralrs+ISQ); never hardcoded in uniko crates |
| Async runtime | tokio | Bounded channels, circuit breaker, DLQ in the pipeline layer |
| GPU | Optional (`gpu-cuda`/`gpu-metal` features) | Full benchmark suite runs on an 8 GB consumer GPU |
| Python | PyO3 + maturin (skeleton, Phase 4) | |

---

## 4. Architecture

### 4.1 Workspace shape

```
uniko-py (PyO3 skeleton, Phase 4)
    └── uniko-api          facade: re-exports the public surface (14 lines, no HTTP yet)
            └── uniko-memory     recall cascade, consolidation (P4), Episode recording, PipelineSystem
                  ├── uniko-cortex     P5 procedure promotion, P6 topics, (MCTS planned)
                  ├── uniko-extract    NLP cascade, NER, chunking, atomic ingest
                  └── uniko-pipes      generic Step trait, circuit breaker, DLQ, metrics
                        └── uniko-store     KnowledgeBase over uni-db: Cypher/bulk/vector, StripedLocks
                              └── uni-db + uni-xervo
uniko-bench (separate): LoCoMo/LongMemEval harnesses, microbenches, perf telemetry
```

The dependency graph is non-linear by design and pinned by `tests/layering_test.rs`.
"Layer N" labels are **cognitive altitude, not build order**: cortex is "Layer 5" yet sits
below memory in the build graph, because consolidation (P4) is the heartbeat that drives
cortex sweeps (P5/P6) — memory depends on cortex, not vice versa.

### 4.2 The pipeline model (P1–P8)

| P# | Pipeline | Status | Where |
|---|---|---|---|
| P1 | Ingest (messages → graph) | **Shipped** — single atomic transaction per message | `uniko-extract/src/ingest/atomic.rs` |
| P2 | NER (rule-based + code AST + ONNX) | **Shipped** | `uniko-extract/src/ner/` |
| P3 | Observation extraction (rules + YAML rules-engine + SRL frames) | **Shipped** | `uniko-extract/src/observations/` |
| P4 | Consolidation (observations → Facts, contradiction F38 + drift F39) | **Shipped** | `uniko-memory/src/consolidation.rs` |
| P5 | Procedure promotion (repeated episodes → Procedures) | **Shipped** (Locy path has a known bug, Cypher fallback carries it — RC12) | `uniko-cortex/src/procedures.rs` |
| P6 | Topic detection (label-propagation communities, optional LLM naming) | **Shipped** | `uniko-cortex/src/topics.rs` |
| P7 | Embedding / summaries | **Shipped** (auto-embed via uni-db) | store + extract |
| P8 | Rule induction | **Not started** (Phase 6, research) | — |

Ingest is **atomic**: CPU-side extraction (NER + NLP cascade + observation prep) happens
first, then one transaction writes Message, Entities, Observations, edges, and chunks —
all-or-nothing, idempotent on `message_id`.

### 4.3 Recall cascade

Retrieval is a three-phase cascade with coverage gating (spec Part IX):

- **Phase 1 (compact):** Fact / Topic / Procedure hits — the compiled knowledge.
  Strategies: Merge / Boost / Off.
- **Phase 2 (expand):** hybrid vector + BM25 over Episodes/Observations/Messages, fused
  with Reciprocal-Rank Fusion; temporal-interval and graph (PPR) channels activate when
  the query's intent profile warrants. Coverage gate (default 0.65) decides whether to go deeper.
- **Phase 3 (broaden):** full Chunk/Artifact fallback.

Query understanding is local (rule-based intent profile: entities, temporal window,
answer-type prediction). Optional cross-encoder reranking rescores the fused top-N.
Results assemble into a `ContextBundle` under a token budget, filtered by the visibility
policy (`public` / `private:{id}` / `team:{id}` / `org:{id}` via a cached `Viewer`).

### 4.4 The feedback loop

`record_query_episode` (in `uniko-memory::query`) captures every answered query as an
Episode — question, answer, recall node IDs, coverage, token/LLM usage. These Episodes
feed the P5 sequence detector, closing the loop from *answering* to *learning what works*.
Opt-in, so any recall consumer can feed procedural learning.

### 4.5 Storage layer guarantees

`uniko-store::KnowledgeBase` is the only crate that touches uni-db. Key mechanics:

- **Write paths:** Cypher for reads and ID-returning writes; uni-db bulk APIs for hot
  paths (`bulk_insert_vertices/edges`). Measured on real LoCoMo batches: bulk edges are
  ~524× faster than Cypher UNWIND; nodes ~50× when not embedding-bound (`initial-docs/bench/bulk-vs-unwind.md`).
- **Concurrency:** read-modify-write upserts are serialized by `StripedLocks` (256-stripe
  async mutex) because uni-db's SSI cannot detect logical-key insert races (RC2).
- **Crash safety:** uni-db WAL; single-transaction ingest means no half-written messages.
- **Multi-KB:** `build_shared_runtime` + `open_with_runtime` share one ONNX session across
  many KnowledgeBases (VRAM-bound deployments).

---

## 5. Capabilities today (measured)

### 5.1 Benchmark results

| Benchmark | Result | Date / artifact |
|---|---|---|
| **LoCoMo (full 10 conversations, 1,986 questions)** | LLM-judge **81.2%**, retrieval hit 85.6%, F1 0.321, total LLM cost **$3.55** (gemini-3.1) | 2026-05-26, `data/` artifacts |
| LoCoMo conv-26 (post-SRL fix) | judge 86.8% (gpt-4o-mini) / **93.4%** (Gemini); retrieval hit 75.5% | `data/locomo_conv26_post_srl_fix*.json` |
| LongMemEval (11-question slice: SSU/SSA/MS) | per-question results recorded; full run pending | `initial-docs/bench/lme-2026-05-17-11q.md` |

Context: the pre-v6 prototype scored **22.2%** on LoCoMo; published competitor numbers are
Mem0 91.6%, Zep/Graphiti 75–84%, Letta 74.0%, LangMem 58.1%. uniko's judge uses Mem0's
verbatim judge prompt for comparability. The trajectory 22% → 81% validates the v6
interaction-first redesign; the remaining gap to Mem0 is an active workstream (notably
date-anchored questions — see §6).

### 5.2 Performance characteristics

- **Ingest:** ~205 ms/turn on LoCoMo single-process CPU+GPU (config-dependent; LME/GPU/
  parallel configurations differ substantially — re-measure per config).
- **Hardware floor:** the full pipeline (NLP cascade FP32, BGE-small, optional reranker)
  runs benchmarks on a 22-core CPU + 8 GB consumer NVIDIA GPU.
- **Ingest cost:** zero LLM tokens per message by default (local extraction only).
- **Query cost:** answer-LLM cost separated from judge cost in bench reports; full
  LoCoMo10 cost $3.55 including judging.

### 5.3 Functional surface (verified in code)

- Multi-session conversation memory with speaker attribution and adversarial detection
  (graph-structural: SENT_BY edges).
- Fact lifecycle: derivation from observation clusters (cosine 0.88 paraphrase
  collapsing), contradiction invalidation (40% disagreement threshold) with BTIC
  intervals, entity-drift flagging.
- Episode/Action recording with artifact overflow (large outputs spill to Artifact nodes).
- Working-memory assembly per goal; NL-to-Cypher translation (LRU-cached).
- Procedure promotion from repeated action sequences; topic communities over entity
  co-occurrence.
- Access control by participant/team/org visibility.
- Python bindings and HTTP/MCP surface: **not yet** (Phase 4).

---

## 6. Engineering status & known gaps

### 6.1 Phase position

Phases 1–3 of the six-phase plan (`initial-docs/execution/`) are substantially **built and
benchmarked**: schema, P1–P7, recall cascade, episodes/actions, procedures, topics,
access control, NL-to-Cypher. Phase 4 is **in progress**: the benchmark harness shipped
first (LoCoMo/LongMemEval published runs exist); MCP server and Python bindings remain.
Phases 5–6 (FS/Git integration, cross-agent sharing, rule induction, MCTS, multimodal)
are future.

### 6.2 Known issues (prioritized, from the 2026-06-10 architecture review)

| Issue | Severity | Detail |
|---|---|---|
| RC2: lock discipline is convention-only | High | uni-db has no atomic SET/CAS and SSI can't catch logical-key insert races; StripedLocks compensates, but nothing forces a new upsert site to take the lock. Probed and confirmed won't-fix upstream. |
| RC12: Locy rule invocation broken | High | `execute_rule(name)` passes a bare rule name where the grammar needs a goal query; a Cypher fallback has silently carried P5 since launch and 3 of 4 stdlib rules have no fallback. Fixable in-repo. |
| Spec/doc drift | High | `PHASE1_GAPS.md` lists items resolved by the atomic-ingest refactor (e.g. B5 references a retired Step); `NLP_REQUIREMENTS.md` prescribes approaches the code diverged from. Stale north-star docs have already caused wrong review conclusions. |
| Date-anchored recall | Med | ~34% of LoCoMo single-hop failures are date-anchored; temporal filtering on `Session.started_at` planned once the new NLP temporal output lands. |
| Sentence splitting duplicated ~3× | Med | `nlp/mod.rs`, `observations/rules.rs`, `chunking/text.rs` — silent divergence corrupts span alignment. |
| NER span alignment by text search | Med | Fragile under repeated substrings/Unicode normalization; suspected factor in observation-chunks-missing-entity-edges issue. |
| Python async boundary undecided | Med | Core APIs are `async fn`; pyo3-asyncio vs blocking-wrapper choice must precede Phase 4 bindings. |
| Hardcoded tuning constants | Med | RRF k, tier weights, consolidation thresholds, lock stripes — retrieval experiments currently require recompiles. |

### 6.3 uni-db dependency posture

uni-db is co-developed (`../uni`) and is both the biggest leverage and the biggest risk.
The posture is disciplined: 14 root-cause workarounds are cataloged in
`bugs/UNI_DB_WORKAROUNDS.md` with repro tests, filed-issue status, and re-verification
dates; 5 were retired when uni-db 2.0.0 landed; all live workarounds are confined to
`uniko-store`, so higher layers never see them. Bugs are filed upstream with isolated
repros rather than worked around silently.

---

## 7. Future directions

### 7.1 Committed roadmap (spec Phases 4–6)

1. **Phase 4 — external surfaces & benchmark breadth.** MCP server (agent-facing tool
   surface), Python bindings (PyO3), contrastive retrieval, ASSUME/ABDUCE builders, and
   broader benchmark coverage (MemoryAgentBench, BEAM, Evo-Memory). This is the
   make-it-usable-by-others phase: today uniko is a Rust library; after Phase 4 it is a
   drop-in memory for any MCP-speaking agent and any Python agent framework.
2. **Phase 5 — agent-environment integration.** FS/Shell/Git integration surfaces
   (codebase-knowledge use case), cross-agent sharing, organization/team support — the
   GoalOS multi-agent scenario made real.
3. **Phase 6 — research extensions.** P8 rule induction (learned Locy rules), MCTS
   planning over procedural memory, multimodal embedding, audio/video chunking.

### 7.2 Near-term technical levers (already scoped)

- **Temporal recall filtering** — date-anchored WHERE clauses once new NLP temporal
  extraction lands; directly targets the largest known failure category on LoCoMo.
- **RC12 fix** — building proper Locy goal-queries un-breaks three dormant stdlib rules;
  highest-value single fix identified by the architecture review.
- **uni-db `bulk_update_vertices`** — entity UPDATE is the ingest hot spot (Phase-3
  apply_entity ≈ 99.8% of exec time); a bulk-update upstream PR is the next step.
- **Config-driven retrieval tuning** — externalize RRF/tier/threshold constants to enable
  cheap retrieval-quality sweeps.

### 7.3 Possibility space (not yet committed)

- **LoCoMo parity and beyond.** Closing the ~10-point judge gap to Mem0 via temporal
  filtering + reranker tuning is plausible without architectural change; uniko then
  competes with zero-infrastructure deployment and a fraction of the per-query LLM cost
  as differentiators rather than chasing leaderboard parity alone.
- **Self-improving agents as the product story.** The Episode → P5 → Procedure loop plus
  P8 rule induction is the path to demonstrable improvement-over-time (the Evo-Memory
  delta metric), which no shipping competitor demonstrates today.
- **Edge/local-first deployments.** Everything (DB, NLP, embeddings, reranking) already
  runs in-process on consumer hardware; an offline-capable personal-agent memory is a
  packaging exercise, not a research project.
- **Hypothetical reasoning as a wedge.** ASSUME/ABDUCE (temporary graph mutations,
  evaluate, roll back) is unique in the market and maps directly to agent planning
  ("what would I believe if X?"). Promoting it from research track to a flagship Phase-4+
  capability is a credible differentiation bet.
- **GoalOS as the distribution channel.** uniko's goal-oriented working memory is built
  for it; the multi-agent collaboration scenario (shared semantic memory, per-agent
  episodic memory) is the spec's centerpiece use case.

---

## 8. One-paragraph summary

uniko is an embedded, Rust-native cognitive memory for AI agents: messages in, compiled
knowledge out, with full provenance and no infrastructure to operate. The v6
interaction-first architecture is built and validated — 81.2% LLM-judge on full LoCoMo at
$3.55 total cost, local-only extraction on consumer hardware — placing it within reach of
the best published systems while being the only embedded, formally-reasoning entrant. The
core engine (Phases 1–3) is done; the current frontier is exposure (MCP + Python, Phase 4)
and the known recall gaps (temporal filtering, RC12). The long bet is the learning loop:
episodes that become procedures that become rules, on a database that can reason about
them without an LLM in the loop.
