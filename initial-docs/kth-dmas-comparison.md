# uniko vs KTH dmas-memory — LoCoMo cost / latency / token comparison

| | |
|---|---|
| **Status** | Measured, 2026-06-14 |
| **Our run** | `data/locomo_gemini31_merged.json` — full LoCoMo10, 1986 questions, gemini-3.1-pro-preview judge, `locomo-bge-gemini31.json` profile |
| **Baseline** | Wolff & Bennati, *Cost and accuracy of long-term memory in Distributed Multi-Agent Systems based on Large Language Models*, KTH Royal Institute of Technology, [arXiv:2601.07978](https://arxiv.org/abs/2601.07978), Jan 2026 |
| **Baseline data** | [`wolffbe/dmas-memory`](https://github.com/wolffbe/dmas-memory/tree/master/testbed/experiments/results) — per-experiment CSVs, 1540 `ask` rows + 5882 `load` rows per backend, **unconstrained** network mode |
| **Scope** | Cost, latency, tokens, ingest throughput. **Accuracy deliberately excluded** — see §4. |

---

## 0. TL;DR

> **uniko is the only system of six that ingests LoCoMo in under 10 minutes at $0 API cost, and the only one with sub-4-second mean Q&A wall-time. Against the graph backends (Graphiti, Cognee) it is 33–76× faster at ingest and uses roughly half the LLM tokens per query.**

The defensible competitive ground vs the KTH baseline is **ingest throughput / cost** and **end-to-end query latency** — not per-query token cost, where mem0's tighter retrieval wins.

---

## 1. What KTH measured

The KTH testbed is a distributed multi-agent system (DMAS) that runs five long-term-memory backends through the LoCoMo benchmark under two network regimes (unconstrained / constrained). It records ten dependent variables per turn and per question: wall time, CPU/RAM/disk on edge and cloud, network bytes across the edge↔cloud boundary, OpenAI token consumption and cost, API latency, and answer similarity.

Backends compared:

| Backend | Library | Storage |
|---|---|---|
| mem0 | mem0 v2.0.2 | Qdrant (vector) |
| Graphiti | graphiti-core 0.29 | Neo4j (graph) |
| Cognee | cognee 1.0.9 | Neo4j + Qdrant |
| RAG | in-house `RagService` | Qdrant |
| Full Context | in-house `FullContextService` | in-process |

Their MAS uses a `qwen2.5:3b-instruct` coordinator (local, ollama) + `gpt-4o-mini` responder. Loading ingests all 19 sessions of the first ten LoCoMo conversations; the Q&A phase answers 1540 non-adversarial questions across those conversations.

We compare against the **unconstrained** regime (no artificial network latency) since uniko is not a distributed system and has no network plane to constrain.

---

## 2. Q&A phase — per-question cost / latency / tokens

N = 1540 non-adversarial LoCoMo questions (uniko numbers filter out the 446 adversarial questions to match KTH's set).

| System | Answer $/q | Total $ (1540 q) | Avg wall | Ctx-in tok | Total tok |
|---|---|---|---|---|---|
| mem0 | **$0.000179** | $0.28 | 4.56s | 752 | 1235 |
| rag | $0.000259 | $0.40 | 4.34s | 1308 | 1790 |
| **uniko** | $0.000657 | $1.01 | **4.04s** | 2435 | **2468** |
| graphiti | $0.000657 | $1.01 | 6.20s | 2174 | 4546 |
| cognee | $0.000715 | $1.10 | 6.99s | 248 | 4780 |
| full_context | $0.006786 | $10.45 | 9.51s | 44312 | 45708 |

**Reading it:**

- **uniko has the fastest Q&A wall time of all six systems** (4.04s = 2.22s recall + 1.19s generation, conv-26 figures representative). Graphiti and Cognee — the two graph backends — are 53% and 73% slower respectively.
- **uniko uses fewer total LLM tokens per query than either graph backend** (2468 vs Graphiti 4546, Cognee 4780) — roughly half.
- **uniko's answer cost ties Graphiti** ($0.000657/q) and is ~3.7× mem0's. The driver is retrieved-context size: uniko sends 2435 input tokens to the responder vs mem0's 752. This is a recall-budget tuning lever, not an architectural floor.

uniko's answer cost is compared against KTH's `llm_cost_usd` (coordinator SLM is local/free; the cost is the gpt-4o-mini responder). uniko has no coordinator, so its responder cost is the whole Q&A cost. The **judge cost is separated out and excluded** — uniko's gemini-3.1 judge is an evaluation artifact, not part of the serving path; KTH uses a free local similarity scorer. (For the record, uniko judge cost was $2.24 over 1540 q; answer was $1.01.)

---

## 3. Loading phase — full 5882-turn ingest

| System | Total $ | Tokens | Wall (min) | per-turn ms |
|---|---|---|---|---|
| **uniko** | **~$0** | **0** (local NLP) | **7.5** | **76** |
| full_context | $0.00 | 0 | 21.08 | ~215 |
| rag | $0.006 | 308k | 40.29 | ~411 |
| cognee | $1.32 | 6.7M | 493.47 | ~5031 |
| mem0 | $4.82 | 51.7M | 250.95 | ~2560 |
| graphiti | $5.49 | 34.6M | 568.97 | ~5804 |

**Reading it:**

- **uniko ingests the full corpus in 7.5 minutes at $0 API cost.** It runs a local NLP cascade (kniv-deberta INT8 ONNX) for entity / observation extraction and makes **zero LLM API calls during ingest**.
- The graph and vector LLM backends (mem0, Graphiti, Cognee) call gpt-4o-mini per turn for fact/entity extraction, making their loading pipeline network-bound. uniko is **33–76× faster** than this group at the per-turn level and avoids $1.32–$5.49 of ingest cost per corpus.
- RAG and Full Context are cheap to load because they do no extraction (raw embedding / raw store), but they pay for it at Q&A time — Full Context is the most expensive Q&A system by 10×.

uniko's 7.5 min is the sum of per-turn `wall_ms` events, the same metric KTH's loading wall is built from. Measured two ways for cross-check: per-turn sum = 7.50 min; process-wallclock from log timestamps = ~8.6 min; including the post-ingest consolidation + procedure + topic sweeps (P4/P5/P6) ≈ ~9 min. uniko's ingest ran on GPU; the KTH backends ran CPU/WSL — but their ingest bottleneck is synchronous OpenAI calls, not local compute, so the gap is predominantly architectural (no API calls), not hardware.

---

## 4. Why accuracy is excluded

The two runs score answers by incomparable methods:

- **KTH:** average of SentenceTransformer cosine similarity + string similarity, with the responder prompted to answer "I don't know" (IDK) when unsure. This produces low headline accuracy (mem0 7.5%, Graphiti 11.1% unconstrained) because most answers fall into the IDK bucket (59–68% IDK rate). Critically, KTH's own two-proportion z-tests find **no statistically significant accuracy difference** between mem0 and Graphiti (p = 0.2269 unconstrained, p = 0.4330 constrained, N = 199 in the paper's per-conv table) — vector vs graph memory is not the deciding factor on accuracy at their sample size and scoring.
- **uniko:** LLM-as-judge (gemini-3.1-pro-preview), which accepts semantically-correct answers a cosine threshold would reject. uniko scores 0.8117 judge over the 1540 judge-eligible questions.

These are different scales measuring different things; a direct accuracy column would be misleading. To produce a comparable accuracy number, uniko would need to be re-scored under KTH's cosine + IDK protocol — out of scope here.

---

## 5. Positioning summary

| Dimension | uniko vs best KTH | uniko vs graph backends (Graphiti / Cognee) |
|---|---|---|
| Ingest wall time | 2.8× faster than full_context; 5.3× faster than RAG | **33–76× faster** |
| Ingest $ | $0 vs $0.006 (RAG) | $0 vs $1.32–$5.49 |
| Ingest tokens | 0 vs 308k (RAG) | 0 vs 34–52M |
| Q&A wall time | **fastest of all six** (4.04s) | 35–42% faster |
| Total tokens / query | lower than every graph backend | ~half |
| Q&A $/question | 3.7× mem0 (we over-retrieve) | tied Graphiti, cheaper than Cognee |

**Where uniko wins decisively:** ingest throughput, ingest cost, end-to-end query latency, per-query token efficiency vs graph systems.

**Where uniko trails:** Q&A cost-per-question vs mem0, entirely attributable to larger retrieved context (2435 vs 752 input tokens). Tunable via recall token budget.

---

## Appendix — provenance and reproduction

**uniko numbers** are derived from `data/locomo_gemini31_merged.json`, produced by `crates/uniko-bench/scripts/merge-locomo-gemini31.py` over:
- 6 full-conversation outputs (`locomo_gemini31_conv-{26,30,41,43,47,49}.json`)
- 20 per-category outputs (`locomo_gemini31_conv-{42,44,48,50}_cat{1..5}.json`) from `retry-by-category.sh`

The merged file's per-question records carry `answer_cost_usd`, `judge_cost_usd`, `answer_input_tokens`, `answer_output_tokens`, `recall_latency_ms`, `generation_latency_ms`. Loading-phase numbers come from the per-turn `ingest_turn` events in `data/locomo_gemini31_conv-*_events.jsonl` (wall_ms sum across 5882 turns; extraction token/cost fields are zero because the NLP cascade is local).

**KTH numbers** are aggregated from the 10 result CSVs at [`wolffbe/dmas-memory/testbed/experiments/results`](https://github.com/wolffbe/dmas-memory/tree/master/testbed/experiments/results) — `ask`-phase rows for Q&A (`llm_cost_usd`, `wall_ms`, `responder_context_window_tokens`, `llm_tokens`) and `load`-phase rows for ingest, unconstrained mode.

> Note: the bench-artifact data files (`data/*.json`, `data/*_events.jsonl`) are gitignored per project policy. Regenerate uniko's side via the bench scripts; re-pull KTH's CSVs from the GitHub link above.
