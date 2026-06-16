# uniko Performance Journey — LoCoMo & LongMemEval, ingest + query

| | |
|---|---|
| **Status** | Reconstructed from full git history, 2026-06-15 |
| **Span** | 2026-04-15 (first commit `6822710`) → 2026-06-14 (HEAD `dddb239`), 137 commits |
| **Method** | Full-history sweep of commit messages, committed analysis docs (incl. deleted), and repro-test headers. Machine result files (`data/*.json`, `*_events.jsonl`) were gitignored from the start — their numbers survive only where hand-transcribed into commit messages or markdown. |

## How to read this

Two honest caveats apply throughout:

1. **Early LoCoMo numbers are single-conversation slices** (conv-30, then conv-26), NOT the full 1986-question benchmark. The full-corpus numbers don't begin until `60c7593` (May 3). Each row says which.
2. **LME ingest numbers are per-question and concurrency-dependent** (keyed to question hashes `118b2229` / `5d3d2817`, at stated q/sess). They're directional, not a single clean wall-time. Where a row is q=2 vs q=1 it is noted — GPU contention roughly doubles per-question ingest at q=2 on the single-GPU box.

Numbers in **bold** are committed and citable. Where a number lives only in a memory note or gitignored file, it's marked *(uncommitted)*.

---

## 1. The headline: ingest, hours → seconds

| Date | Benchmark | Ingest metric | Source |
|---|---|---|---|
| 2026-04-21 | LoCoMo | **2h 7m for 369 turns**; **3s → 26s per turn** as DB grows; **~4.7s per observation node** | `b2704f4` |
| 2026-04-23 | uni-db repro | **create_edge 6ms → 2,700ms+** over 300 msgs; create_node flat **~20ms**; panic at ~97 edges | `1eb5db0` |
| 2026-06-14 | LoCoMo | **~62 ms/turn** steady-state; **7.5 min for 5882 turns** (full LoCoMo10) | this-session events |

**Same-size conversation: 2h 7m (Apr 21) → ~22-28s (Jun, 369-turn conv at ~62ms/turn) ≈ ~300× faster.** The April pathology was two uni-db bugs uniko isolated and filed (see §5), not uniko's own algorithm.

---

## 2. LoCoMo — query / retrieval quality progression

Single-conversation era (conv-30 retrieval-only, from `data/retrieval_analysis.md` + commit messages):

| Date | Overall hit | Single | Multi | Temporal | Adversarial | What changed | Source |
|---|---|---|---|---|---|---|---|
| 2026-04-21 | **13.0%** | 14.3 | 3.2 | 23.1 | 12.0 | per-obs embed (broken) — 964 obs flooding top-15 | `b2704f4` / retrieval_analysis.md |
| 2026-04-21 | **59.5%** | 65.9 | 25.8 | 76.9 | 75.0 | session-level obs chunks | retrieval_analysis.md |
| 2026-04-22 | **61.1%** | 65.3 | 25.8 | 76.9 | 80.0 | + sentence splitting | retrieval_analysis.md |
| 2026-04-22 | **67.2%** | 71.4 | 29.0 | 88.5 | 84.0 | + session DateTime fix (best conv-30) | retrieval_analysis.md / `0fd604f` |
| 2026-04-23 | **85.5%** | — | — | 96.2 | — | + image-caption ingest (conv-30, +2.3pp) | `36329bc` |

Full-corpus era (1986 questions, LLM-judge):

| Date | Scope | Judge / hit | Detail | Source |
|---|---|---|---|---|
| 2026-05-13 | conv-26 (199q) | judge **0.763** | flush-failure warnings 48 → 0; matches historical best 0.76 | `ffd0f24` |
| 2026-05-14 | conv-26 | Single **0.943**, Temporal 0.703, Overall **0.750** | Mem0 verbatim judge prompt | `f716b06` |
| 2026-05-14 | full 10-conv | Overall **0.742** | Single 0.895 / Multi 0.550 / Temporal 0.611 / Open 0.396 (vs Mem0 0.684, full-ctx ceiling 0.729) | `d20f604` |
| 2026-05-14 | conv-26+30 (304q) | hit **43.2% → 78.7%** | bench-validated recall defaults; temporal channel −85ms recall | `3d3afbd` |
| 2026-05-26 | full 10-conv | **judge 0.8117 / hit 0.8555 / F1 0.321 / $3.55** | gemini-3.1 judge — current baseline | `project_locomo10_gemini31_baseline` *(merged JSON uncommitted)* |

Reranker / embedder sweeps (conv-26, `60e0f6f` May 13): nomic 0.763/2394ms, minilm 0.757/1686ms, **bge-small 0.757/1607ms** (chosen default), bge-large 0.750/2398ms.

---

## 3. LongMemEval — the ingest perf campaign (committed in commit messages)

This is the richest *ingest* progression in the whole history. All keyed to single LME questions at stated concurrency, GPU FP32.

| Date | Stage metric | Source |
|---|---|---|
| 2026-05-17 | **ner_nlp_ms 74s → 16.6s (−78%)** (q=2, batched SRL forward — blog predicted 3.3×) | `4ef0f4a` |
| 2026-05-17 | **ingest wall 7.33 min → 3.65 min (−50%)**; ner_upsert 229s → 71s (−69%); consolidation 162s → 73s (−55%); **total wall ~13.0 min → 5.02 min (−62%)**; R@5 1.000 preserved | `188771b` |
| 2026-05-19 | **208s → 79s per question (−62%)** (q=1, sess=24); staged 208→151(mimalloc)→86(bulk edges)→**79**(bulk nodes); **edges_fast 962ms → 21ms (~46×)**; nodes 44→24ms; message-edges 370→78ms; **total ingest CPU 3083s → 1612s (−48%)**; mimalloc ~3× | `ee279e1` |
| 2026-05-20 | Phase-3 UPDATE **~18 ms/row = 99.8% of apply_entity**; :Entity label hint **−38%**; sess=1 ~85s → ~70s, sess=24 73s → **65.8s** | `f46a86b` |

The committed LME analysis doc `initial-docs/bench/lme-2026-05-17-11q.md` adds the baseline context: **single-process GPU FP32 = ~6 min/q ingest**; entity upsert (`dedup::upsert_entities`) = **53% of ingest**, NLP only ~10%; consolidation ~5-6 min/q and parallelism-independent.

---

## 4. LongMemEval — query / retrieval (the one committed run)

`initial-docs/bench/lme-2026-05-17-11q.md` — 11-question slice (5 SSU + 3 SSA + 3 MS), GPU, BGE-small 384d, retrieval-only (no LLM judge):

| Category | n | contains | R@5 | NDCG@5 |
|---|---|---|---|---|
| SSU | 5 | 1.000 | 1.000 | 1.000 |
| SSA | 3 | 0.333 | 1.000 | 1.000 |
| MS | 3 | 0.667 | 0.667 | 0.650 |
| **ALL** | 11 | **0.727** | **0.909** | **0.905** |

Phase-1 gate (≥90% contains) **fails at 72.7%**, driven by SSA chunking. Reranker (MS-MARCO-MiniLM, 6 non-SSU q): R@5 0.833 → **0.875**, contains unchanged, latency ~2×. Per-question recall_ms 1,762–13,490.

**Everything else LME — the FP32 single-process baseline, the `lme_sweep_s1..s32` concurrency sweep, full-set runs, ablations — exists only as gitignored result JSONs** (`data/lme_*.json`). No committed full-set accuracy, no committed sweep conclusion. The sweep result files (this session's analysis) showed recall_ms hovering ~1535–1823 across s1–s32, mostly N=1 probes within ±100ms noise — not citable as hard speedups.

---

## 5. Write-path microbenchmarks (perf numbers on LoCoMo data)

`crates/uniko-bench/docs/bulk-vs-unwind.md` — bulk API vs Cypher UNWIND on real conv-26 batches (419 turns, uni-db 2.0.2, BGE-small):

| Operation | Speedup | Detail |
|---|---|---|
| Edges | **524×** | ~2.6µs vs ~1367µs/op; 11.4ms vs 5954ms whole-conv |
| Nodes (no embed) | **49.6×** | 3.4ms vs 167ms |
| Nodes (with embed) | 1.4× | embedder dominates |

`bugs/UNI_DB_WORKAROUNDS.md` records the uni-db-level ratios behind these: MERGE pass ~63×, UNWIND-edge ~100× slow, multi-label scan ~18 ms/row, mimalloc ~3×.

---

## 6. The uni-db / uni-xervo adoption thread (why the speedups happened)

The perf wins are co-authored with the substrate. Every major ingest landing is also a dependency adoption:

| Date | Adoption | Perf unlock |
|---|---|---|
| 2026-04-22/23 | Filed uni-db #43 (600ms retry-storm per flush), #46 (edge compaction), #49 (O(total_rows) index rebuild) as minimal repros | uni-db fixed them → killed the 2h7m pathology |
| 2026-05-19 (`ee279e1`) | uni-db `bulk_insert` APIs + `mimalloc` feature; uni-xervo promoted to direct dep **0.12.0** (shared `ModelRuntime`) | the keystone: bulk writes (46×/524× edges), 3× allocator, shared NLP runtime |
| 2026-06-03 (`9c6e451`) | uni-db **2.0.0**, xervo **0.13.0**, mimalloc on all bench bins | version sync |
| 2026-06-13 (`302480e`, `7041365`) | uni-db **2.1.0** (async rules, removed RC8/RC1b workarounds); xervo `NlpModel` migration (dropped in-crate ONNX decode + 8MB tokenizer) | correctness cleanup + NLP consolidation |

**The 300× ingest win is predominantly a substrate win that uniko's repro discipline unlocked** — the 2h7m wasn't a slow algorithm, it was two uni-db bugs (retry-storm + index-thrash) that uniko isolated precisely enough to make fixing tractable.

---

## 7. What is NOT recoverable from git

- **No committed full-set LME accuracy / hit-rate / cost** — only the 11q retrieval slice.
- **No committed LME concurrency-sweep conclusion** — the `lme_sweep_*` files were gitignored; commit `d2e32eb` ("LME sweep merge") is a code refactor, not results.
- **No machine result files ever committed** — `data/*.json`, `*_events.jsonl`, `*.log` gitignored from the start. The LoCoMo10 merged baseline (0.8117) lives only in `data/locomo_gemini31_merged.json` (uncommitted) + `UNIKO_OVERVIEW.md` §5.1.
- **The April "hours" anchor (2h7m) is committed** (`b2704f4`) — but no committed *same-hardware* April-vs-now LoCoMo run exists. A clean apples-to-apples number would require re-running April code (worktree at `ebfe174` + `../uni@v1.1.0`); feasible, moderate effort, not necessary for the story.

---

## Appendix — canonical current numbers

| Benchmark | Metric | Value | Date |
|---|---|---|---|
| LoCoMo10 (1986q) | judge / hit / F1 / cost | **0.8117 / 0.8555 / 0.321 / $3.55** | 2026-05-26 |
| LoCoMo10 | ingest | **7.5 min / 5882 turns / $0** (~62 ms/turn) | 2026-06-14 |
| LoCoMo conv-26 | judge (post-SRL) | 0.868 gpt-4o-mini / 0.934 gemini | — |
| LME 11q slice | contains / R@5 | 0.727 / 0.909 (retrieval-only) | 2026-05-17 |
| Bulk vs UNWIND | edges / nodes | 524× / 49.6× | — |
