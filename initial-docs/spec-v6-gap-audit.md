# uniko Spec v6 — Implementation Gap Audit

**Date:** 2026-06-16
**Scope:** Full audit of the implementation against `uniko-spec-v6.md` and its companion docs
(`schema-v3.md`, `pipelines-design.md`, `pipeline-management.md`, `chunking-analysis.md`,
`embedding-analysis.md`, `implementation-addendum-v6.md`, and the per-phase execution specs).
**Method:** 8 parallel exploration agents, one per dimension, each tasked to find gaps with
file:line evidence. Contested / load-bearing claims were re-verified directly against source
(grep) before inclusion. See "Verification notes" below.

---

## Verification notes (read first)

- One subagent (ingest/chunk/embed) produced **systematic false negatives** — it concluded
  "5 chunkers missing," "message chunking broken," and "5 computed-embedding functions missing"
  by searching for files/functions under guessed names. **All of those exist and are wired.**
  Those claims were discarded. Confirmed present:
  - Chunkers: `crates/uniko-extract/src/ingest/chunking/{text,code,html,structured}.rs`
  - `embed_entity` (`crates/uniko-extract/src/embedding/mod.rs:123`), `embed_episode` (`mod.rs:205`),
    used at `crates/uniko-memory/src/episode.rs:161`
  - Message >1024-token chunking: `crates/uniko-extract/src/ingest/message.rs:98-99`
    (threshold `message_chunk_threshold = 1024`, `config.rs:699`)
- **Status legend:** `V` = verified against source this audit · `C` = corroborated by ≥2 agents ·
  `?` = needs a confirming look (agents disagreed or evidence thin).
- **Tier** column uses the spec's own classification (MVP / DIF / RES), not a subjective priority.

---

## What is solid (verified — NOT gaps)

- **Provenance chain intact:** `Fact ─SUPPORTED_BY→ Observation ─OBSERVED_IN→ Message ─SENT_BY→ Participant`
  all created. The "interaction-first" claim holds.
- **Consolidation auto-runs** (worker fires at 20 observations or on timer) — not manual-only.
- **ADR-1** (UUID v7, deterministic `chunk_id = {parent}:{index}`), **ADR-2** (pure-Rust, no PyO3,
  ONNX via `ort` + tree-sitter + rule fallback), **embedding prefix consistency**, and
  **schema layer ownership** are all correct.
- Chunkers (text/code/html/structured), message chunking, computed embeddings, recall cascade
  (phases, RRF k=60, MMR λ=0.7/0.85, coverage 0.75/0.65, tier weights), BTIC, F38 contradiction
  (0.40 threshold), procedure & topic lifecycle, rule lifecycle, ASSUME, NL-to-Cypher — all present
  and matching spec constants.

---

## A. Schema gaps

| Gap | Spec | Tier | Status |
|---|---|---|---|
| ~16 **orphan edges** declared but never instantiated. Provenance-critical: `DERIVED_FROM` (Fact→Episode), `DERIVED_BY` (Fact→Rule), `INVALIDATED`/`PROMOTED`/`APPLIED_RULE` (ConsolidationCycle→*), `INVOLVED` (Episode→Action), `SUMMARIZES` (Summary→*) | F22, F53 | MVP | C |
| Deferred orphan edges: `CREATED_BY`/`MODIFIED_BY` (F30), `SHARED_FROM` (F51), `OWNED_BY`/`DEPENDS_ON`/`SUBTASK_OF` (F11/F12), `OPERATES_ON`/`USED_IN`/`COVERS` | F11, F12, F30, F51 | DIF | C |
| `Artifact.hash` dedup index possibly absent; `Summary.text` fulltext index absent | F29, F27 | MVP | ? |

---

## B. Ingest / extract gaps

| Gap | Spec | Tier | Status |
|---|---|---|---|
| **`NEXT.gap_ms` hardcoded to 0** — comment "Phase 3 will populate" (`crates/uniko-store/src/storage/edges.rs:247,250`). Temporal gap analysis non-functional | F6 | MVP | V |
| **`SENT_BY.role` hardcoded "user"** — `participant_type` (human/agent/service) not propagated to edge role; breaks "messages from agents" filtering | F8 | MVP | C |
| **URL ingestion not implemented** — code comment defers fetching; F23 lists URLs as MVP | F23 | MVP | C |
| **LLM NER enhancement is a stub** (no LLM call). F31 marks it "optional," so degradation is acceptable | F31 | MVP | C |
| `OBSERVED_DURING` (Observation→Episode, <5min window) not created | F35 | MVP | ? |
| HTML/CSV/PDF chunkers exist but addendum §19 flags HTML/CSV as **basic quality**; image/audio/video correctly Phase-6 deferred | F24 | MVP/RES | C |

---

## C. Consolidation / recall gaps

| Gap | Spec | Tier | Status |
|---|---|---|---|
| **Memory decay (F50) never applied on a cadence** — `relevance_decay` rule registered as a node but no pipeline executes it; episodes never decayed/pruned | F50 | MVP | V |
| **Drift override (F58) absent in recall** — grep of `crates/uniko-memory/src/recall/` finds no `unstable`/`drift` check; queries about unstable entities still early-exit on Phase 1 | F58 | MVP | V |
| **Summaries (F59 / P7d) not generated** — no session/task/goal/entity/topic summary pipeline | F59 | MVP | C |
| **Contrastive mode (F56)** — failure-episode retrieval not implemented | F56 | DIF | C |
| Cold-start: coverage **facet term disabled** until Facts exist (addendum §13) → Phase-1 early-exit effectively off pre-consolidation (by design) | §IX | MVP | C |
| Minor deviations: MMR may use Jaccard-only (no cosine fallback path); PPR `max_iter=30` not 20, dynamic top-k; observation polarity/negation not extracted | §IX | MVP | ? |
| Drift **30-day window** enforcement deferred to store helper — counter exists, window unconfirmed | F39 | MVP | ? |

---

## D. Cortex / reasoning gaps

| Gap | Spec | Tier | Status |
|---|---|---|---|
| **Stdlib Locy rules ship as nodes but aren't the execution path** — `episode_pattern_detector` never invoked; `contradiction_detector` logic is inline Rust, not the rule; only `sequence_detector` runs (in cortex, post-consolidation). F45 met in letter, not as the engine | F45 | MVP | C |
| **`create_goal`/`create_task` don't exist** → Goal/Task nodes never populated outside tests; working-memory traversal runs over an empty Goal/Task tier in practice | F9–F12 | MVP | C |
| **Procedure preconditions (F43)** use a comma-split `key=value` parser, not real Locy WHERE fragments | F43 | DIF | C |
| **ABDUCE confidence hardcoded 1.0** (`crates/uniko-store/src/locy/abduce.rs:77`, MVP placeholder) | F47 | DIF | V |
| **Cross-agent sharing (F51)** — `share_fact`/`shared_facts` absent | F51 | DIF | C |
| Locy `EXPLAIN RULE`, `ALONG`/`BEST BY`, `similar_to()` in rules — not available/used | §IX-B | DIF/RES | C |
| MCTS (F48), Rule induction P8 (F49) — fully absent (correctly RES/Phase-6) | F48, F49 | RES | C |

---

## E. Agent tools / workflow / provenance gaps

| Gap | Spec | Tier | Status |
|---|---|---|---|
| Missing tools: **`assert_fact`, `invalidate_fact`, `add_observation`, `create_goal`, `create_task`** (and `share_fact`) | Part VI | MVP/DIF | C |
| **Episode not linked**: no `TRIGGERED_BY` (→Message, F19) nor `INVOLVES`/`INVOLVED` (→Action, F15) wired in `record_episode` | F15, F19 | MVP | C |
| **`ingest_artifact` produces orphan nodes** — no session/action/message edge; only `HAS_CONTENT` + `HAS_CHUNK` | F22, F30 | MVP/DIF | V |
| **Session inactivity auto-close (F14)** not implemented; sessions grow unbounded | F14 | MVP | C |
| No agent **facade object** — free functions only (design choice; spec implies `agent.recall()`) | — | — | C |

---

## F. Integration / org / benchmark gaps

| Gap | Spec | Tier | Status |
|---|---|---|---|
| **MCP server (F67) entirely absent** — no `uniko-mcp` crate, zero MCP refs | F67 | DIF | V |
| **PyO3 binding (F71) is a 9-line stub** (`bindings/uniko-py/src/lib.rs`) | F71 | DIF | C |
| **Org/Team is schema-only** — no `MEMBER_OF` ever created; access-control policy reads an unpopulated graph, so F66 is non-functional | F63–F66 | DIF | C |
| **Benchmarks**: only LoCoMo + LongMemEval; **MemoryAgentBench, BEAM, Evo-Memory absent**; no `Benchmark` trait abstraction | Part XI | DIF | C |
| **Metrics not computed**: `phase1_only_pct`, `compression_ratio`, `improvement_delta`, `causal_chain_score` registered/named but never emitted | Part XI | MVP | C |
| `uniko-fs`/`uniko-shell`/git (F68–F70) absent — correctly Phase-5 deferred | F68–F70 | RES | C |

---

## G. NFR / testing / reliability gaps

| Gap | Spec | Tier | Status |
|---|---|---|---|
| **No latency assertions** for NF1–NF19 — only aspirational code comments; nothing measured/gated | NF1–NF19 | MVP | C |
| **No offline-mode test**; offline LoCoMo >50% target never measured | Offline Mode | MVP | C |
| **Scenario-keyed test gaps**: single-hop, adversarial attribution, abstention, consolidation-improvement, procedure-promotion, multimodal, scale-10K, offline lack dedicated tests (some underlying behavior is tested elsewhere) | Part X | MVP | C |
| Error policies Skip/DeadLetter/Abort, circuit-breaker transitions, consolidation triggers — logic exists, **not end-to-end tested** | pipeline-management.md | MVP | C |

---

## Open questions (status `?` — confirm before acting)

Each is a short targeted grep:

1. `Artifact.hash` / `Summary.text` index presence (A, B).
2. `OBSERVED_DURING` edge creation (B).
3. Drift 30-day window enforcement in the store helper (C).
4. MMR cosine-vs-Jaccard fallback path (C).
5. **DeadLetter 5-min retry loop** — genuine conflict: one agent reported it missing, another reported it complete (F62).

---

## Summary — the four honest buckets

The **cognitive core (Phases 1–3) is real and coherent**: provenance, consolidation, recall,
procedures, topics, and reasoning primitives all work. The gaps cluster into:

1. **MVP last-mile wiring** quietly missing: memory decay never runs, drift override absent,
   summaries absent, `gap_ms`/`SENT_BY.role` hardcoded, Goal/Task creation tools missing, several
   provenance edges declared-but-unwritten, stdlib rules registered-but-not-executed.
2. **MVP agent tools** not exposed: assert/invalidate/add_observation/create_goal/create_task.
3. **DIF surfaces** absent: MCP, PyO3, org/team operations, cross-agent sharing, contrastive recall.
4. **Validation debt**: no latency gates, no offline test, 3 of 5 benchmarks absent, key metrics
   uncomputed.

Tier reflects the spec's classification; sequencing/prioritization across buckets is a product call,
not made here.
