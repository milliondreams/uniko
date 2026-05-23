# uniko Phase 1 Benchmarking Proposal

## 1. Purpose

Phase 1 benchmarking is a gate, not a publication. The goal is to answer one question before Phase 2 begins: does uniko's basic retrieval match what's possible today with verbatim storage plus off-the-shelf embeddings?

If yes, Phase 2 builds on a validated foundation. If no, we fix Phase 1 before adding consolidation, Facts, and BTIC. This is the countermeasure to the v5 failure mode — building cognitive machinery on top of broken retrieval.

No statistical rigor. No competitor sweep. No published numbers. Signal only.


## 2. What Phase 1 Can Test

Phase 1 ships: Schema, P1 (Ingest), P2 (NER), P3 (Observations), basic recall, stdlib rules, offline mode. The recall cascade runs Phase 3 (Broaden) — fulltext + vector + Entity MENTIONS traversal. Phase 1 (Compact) has no Facts to search yet and returns empty.

| Capability | Available at Phase 1 | Benchmarkable |
|---|---|---|
| Message ingest + chunking | Yes | Yes |
| Entity extraction (local NER + LLM) | Yes | Yes |
| Observation extraction | Yes | Yes |
| Vector + fulltext search | Yes | Yes |
| Entity MENTIONS traversal | Yes | Yes |
| Speaker attribution (SENT_BY) | Yes | Yes |
| Fact derivation | No (Phase 2) | Skip |
| BTIC invalidation | No (Phase 2) | Skip |
| Episode/Procedure promotion | No (Phase 3) | Skip |


## 3. Benchmark Suite

Four benchmarks. Each has a specific diagnostic purpose.

### 3.1 LongMemEval Retrieval-Only

**What it tests:** Baseline retrieval quality. Does uniko's hybrid search + entity traversal match or beat verbatim storage with modern embeddings?

**Why it matters:** The MemPalace comparison. MemPalace hits 96.6% R@5 on LongMemEval with only ChromaDB + verbatim text. Phase 1 uniko has strictly more capability (entity graph, hybrid scoring, multi-field indexes). If uniko doesn't at least match it, something in P1–P3 is broken.

**Dataset:** LongMemEval `s_cleaned.json`, filtered to `single-session-user`, `single-session-assistant`, and `multi-session-reasoning` categories (~250 questions). Skip `temporal-reasoning` and `knowledge-update` — they require BTIC.

**Metric:** `context_contains_answer` at R@5. The ground-truth answer string appears in the top-5 retrieved items via the recall cascade.

**Pass threshold:** ≥ 90% R@5 on the eligible subset.

### 3.2 LoCoMo Adversarial Attribution

**What it tests:** Speaker disambiguation — Phase 1's unique capability over verbatim systems.

**Why it matters:** LoCoMo includes questions that misattribute statements (asking about Caroline when Melanie said it). Correct answers require traversing SENT_BY edges, not just matching text. If uniko doesn't beat verbatim on these, the speaker graph isn't adding value.

**Dataset:** LoCoMo conversations 1–3 (~1,200 turns). Filter to `adversarial` or `attribution` question types. ~50 questions.

**Metric:** Attribution accuracy. Does the system correctly identify the actual speaker?

**Pass threshold:** ≥ 85% attribution accuracy.

### 3.3 Structure Ablation

**What it tests:** Whether NER and Entity MENTIONS traversal earn their keep.

**Why it matters:** Internal diagnostic. The analog of MemPalace's "34% lift from structure" claim. If the delta between flat and structured is near zero, either P2 NER is failing or traversal is buggy.

**Dataset:** LongMemEval multi-hop questions (answer requires joining info from multiple messages). ~80 questions.

**Configuration:** Run twice.
- `uniko-flat`: Phase 3 Broaden with vector + fulltext only. Entity traversal disabled.
- `uniko-structured`: Phase 3 Broaden with full Entity MENTIONS traversal.

**Metric:** R@5 delta between configurations.

**Pass threshold:** `uniko-structured` ≥ `uniko-flat` + 10 percentage points.

### 3.4 Offline Mode Smoke Test

**What it tests:** Whether Phase 1 works without an LLM provider.

**Why it matters:** The spec commits to offline operation. This is the validation. Also catches circuit breaker fallback bugs.

**Dataset:** LoCoMo conversation 1 (~400 turns), single pass.

**Configuration:** LLM provider disabled. Local NER only. Rule-based observation extraction only. Circuit breaker forced open.

**Metric:** `context_contains_answer` at R@5 on LoCoMo retrieval questions.

**Pass threshold:** ≥ 50% R@5 (matches the spec's stated offline target).


## 4. Micro-Benchmarks (NFR Validation)

Sanity checks on the implementation. Not competitive metrics.

| NFR | Operation | Target | Measurement |
|---|---|---|---|
| NF1 | Message ingest (P1) | < 10ms p95 | Time 1,000 messages |
| NF2 | Vector search top-10 | < 20ms p95 | Time 100 queries |
| NF3 | Hybrid search | < 50ms p95 | Time 100 queries |
| NF4 | 3-hop graph traversal | < 5ms p95 | Time 100 traversals |
| NF5 | Local NER | < 100ms p95 | Time 100 chunks |
| NF7 | Bundle assembly (compact-only) | < 30ms p95 | Time 100 recalls |

Misses by > 2× block Phase 2. Misses by < 2× can be deferred unless they compound with retrieval failures.


## 5. Harness Design

Minimal. One directory, ~300 lines of Python total.

```
benchmarks/phase1/
├── run.py                   # orchestrator: ./run.py --bench longmemeval
├── loaders/
│   ├── longmemeval.py       # load + ingest LongMemEval
│   └── locomo.py            # load + ingest LoCoMo
├── scorers/
│   ├── context_contains.py  # answer string in retrieved context
│   └── attribution.py       # correct speaker identified
├── configs/
│   ├── default.yaml         # full Phase 1
│   ├── flat.yaml            # Entity traversal disabled
│   └── offline.yaml         # LLM disabled
└── results/                 # per-run JSON, per-question scores
```

The harness uses the Python binding (PyO3) and calls:
- `memory.ingest_session(session_data)` — populate the graph
- `memory.recall(query, mode="retrieval_only", limit=5)` — retrieve
- Scorers operate on the returned `ContextBundle`

One command runs everything: `python benchmarks/phase1/run.py --all`. Results are JSON per question: score, retrieved items, latency.


## 6. Gates

Phase 2 does not begin until all four retrieval benchmarks pass.

| Benchmark | Threshold | If Fails |
|---|---|---|
| LongMemEval Retrieval-Only | ≥ 90% R@5 | Fix P1–P3 or recall cascade. Do not proceed. |
| LoCoMo Adversarial | ≥ 85% attribution | Fix P2 NER or SENT_BY edge creation. Do not proceed. |
| Structure Ablation | ≥ +10pp structured vs flat | Investigate NER quality or traversal. Do not proceed. |
| Offline Mode | ≥ 50% R@5 | Fix local NER path or circuit breaker fallback. Do not proceed. |
| NFR Micro-benchmarks | Within 2× target | Investigate, but can proceed if retrieval passes. |

"Do not proceed" means don't start Phase 2 implementation. Phase 2 design work continues in parallel.


## 7. Cadence

**First full run:** End of Phase 1 implementation, before Phase 2 kickoff.

**Smoke test:** Subset of each benchmark (50 LongMemEval questions, 10 LoCoMo adversarial, 20 ablation) on every PR touching P1–P3 or recall. Runs in < 5 minutes. Regression guard.

**Full run:** Manually or nightly on `main`. Produces the tracked metrics.

**Tracked metrics:** R@5 per benchmark, structure delta, offline R@5, NFR p95s. Stored in `results/trends.json`. Regressions > 5pp block merges.


## 8. Out of Scope

- Answer synthesis (end-to-end with LLM reader) — Phase 4
- Statistical rigor, confidence intervals, multiple seeds — Phase 4
- Competitor head-to-head (Mem0, Graphiti, Letta) — Phase 4
- BEAM, Evo-Memory, MemoryAgentBench — Phase 2+
- Published numbers — Phase 4

Phase 1 benchmarking is internal validation. Publication is separate.
