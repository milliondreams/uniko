# Sparse + Multi-Vector Integration & Measurement Plan

**Status:** Proposed
**Date:** 2026-06-28
**Scope:** Wire uni-db 2.4.1 sparse + multi-vector retrieval into uniko's recall path, and measure the impact on LoCoMo + LongMemEval.
**Audience:** uniko-store / uniko-memory / uniko-bench maintainers
**Companion to:** `embedding-alternatives-similarity-plan.md` (this is the now-unblocked execution of its Phases 1–3)

---

## 0. Why now — the upstream blocker is gone

The parent plan flagged uni-db as "the real blocker." Both halves of every producer/consumer pair
have since shipped, so this is now **pure uniko-side wiring** — no upstream waiting.

| Capability | Producer — uni-xervo 0.17.0 (pinned, `Cargo.toml:48`) | Consumer — uni-db 2.4.1 (current dep) |
|---|---|---|
| Dense | `Embed` | `Vector{dim}`, HNSW-SQ, cosine (in use today) |
| Learned-sparse | `EmbedSparse` (SPLADE / BGE-M3 lexical) | `SPARSE_VECTOR(N)` + `uni.sparse.query` (dot, 8-bit quant) |
| Multi-vector | `EmbedMultiVector` (ColBERT) | `List<Vector<D>>` + MaxSim + MUVERA FDE index |
| **Single-pass hybrid** | `EmbedHybrid` — BGE-M3 dense+sparse+colbert in **one forward pass** (`aapot/bge-m3-onnx`) | one auto-embed alias fills all heads together |
| Fusion | — | `uni.search(...)` RRF/weighted + `similar_to(...)` method=`rrf`/`weighted` |

The `EmbedHybrid` path is the crux: adding sparse (and colbert) heads on top of a BGE-M3 dense
upgrade costs ~zero extra inference — it's the same forward pass.

## 1. uniko baseline (starting line)

- Dense BGE-Small-384, HNSW-SQ cosine, on `Fact/Entity/Episode/Message/Observation/Chunk`
  (`uniko-store/src/config.rs:756`).
- **Hybrid dense+BM25 already wired:** `similar_to([m.embedding, m.text], [$qvec, $qtxt],
  {method:'weighted', weights:[0.5,0.5]})` on Chunk/Entity (`recall.rs:585`); weights
  `recall_vector_weight`/`recall_bm25_weight` = 0.5/0.5 (`config.rs:783-784`).
- Phase-2 recall fans out vector+FTS per node type and fuses with RRF (`memory/src/recall/mod.rs:1358`).

So uniko is **one fusion term + one rerank stage** away from full late-interaction hybrid.

---

## PART A — Implementation

### 2. Target nodes

Apply sparse + colbert to the **raw-evidence recall targets only**: `Chunk` and `Observation`.
These carry the verbatim text the benches score against; Entity/Fact are derived/short and gain
little from lexical or token-level matching. Keep them dense-only for now.

### 3. Move 1 — BGE-M3 single-pass dense + learned-sparse

**3.1 xervo preset (uniko side).** Add `bge-m3` to `EmbeddingConfig::preset()`
(`config.rs:229-242`) and the model-id alias (`config.rs:119` neighbourhood), mapping to the
hybrid `aapot/bge-m3-onnx` graph. Dimension 384 → 1024.

**3.2 Schema columns.** In `schema/chunks.rs:50-65` and `schema/observations.rs:45-62`, after the
existing dense block add:
```rust
.property_nullable("sparse_embedding", DataType::SparseVector { dimensions: SPARSE_DIM })
.index("sparse_embedding", IndexType::sparse(SPARSE_DIM) /* share auto-embed alias w/ dense */)
```
Group the dense `embedding` and `sparse_embedding` indexes under **one** `EmbeddingCfg` alias +
`source_properties=[text|content]` so uni-db fills both from a single `EmbedHybrid` pass. Extend
`auto_embed_vector_index` (`schema/mod.rs:73-88`) — or add a sibling helper — to declare the group.

**3.3 Recall fusion.** Extend the existing weighted `similar_to` (`recall.rs:585`) to a third term:
```cypher
similar_to([m.embedding, m.sparse_embedding, m.text],
           [$qvec, $qsparse, $qtxt],
           {method:'weighted', weights:[w_dense, w_sparse, w_bm25]})
```
Add `recall_sparse_weight` to config (default e.g. 0.34/0.33/0.33; tune on bench). The query-side
sparse vector comes from the same BGE-M3 `EmbedHybrid` call on the question.

### 4. Move 2 — ColBERT late-interaction rerank (MaxSim)

**4.1 Column.** Add `colbert_embedding: List<Vector<COLBERT_DIM>>` to Chunk/Observation, same
auto-embed alias (BGE-M3 colbert head — free, same forward pass). **Rerank-only first: no MUVERA
index.** Only add `VectorAlgo::Muvera{...}` if exact-MaxSim re-rank latency at our k becomes a
problem (measure §7 before deciding).

**4.2 Rerank stage.** After dense+sparse retrieval returns top-k (k≈50), re-score with MaxSim via
`similar_to(m.colbert_embedding, $q_colbert)` and re-order top-n. Slot it where the cross-encoder
reranker hooks in today so the two are A/B-swappable, not stacked.

### 5. Move 3 — (optional, later) consolidate onto `uni.search`

Collapse the hand-rolled per-node vector+FTS+RRF fan-out (`recall/mod.rs:1358`) onto native
`uni.search('Label', {vector,fts,sparse}, qtext, qvec, k, filter, {method,reranker})`. A
maintenance/simplification win, **not** a quality win — defer until Moves 1–2 are validated.

---

## PART B — Measurement

### 6. Bench-config arms (one JSON each, vary ONE thing)

The 384→1024 dense-model jump will confound sparse/colbert gains if we leap straight to full hybrid.
So four arms, read as deltas. Add `sparse`/`colbert` knobs to `bench_config.rs` (`EmbedderChoice`
at `:151-156`, reranker block) so an arm is a config file, not a code branch.

| Arm | Config | Isolates |
|---|---|---|
| **0** baseline | `{"embedder":{"preset":"bge-small"}}` + BM25 weighted | current `main` |
| **A** model-only | `{"embedder":{"preset":"bge-m3"}}` dense + BM25 | the dense-model upgrade |
| **B** + sparse | Arm A + `sparse` head, 3-term fusion | learned-sparse (Move 1) → **B−A** |
| **C** + colbert | Arm B + ColBERT MaxSim rerank | late-interaction (Move 2) → **C−B** |

`retrieval_only:true` (`locomo-bge-openai.json:44`) for the tuning loop; flip to `false` for the
judged payoff runs.

### 7. Metrics & what each arm should move

Both harnesses (`crates/uniko-bench`) already isolate retrieval from judging — exactly the signal
sparse/multivector affect.

| Bench | Retrieval-only (tuning signal) | Judged payoff | Always report |
|---|---|---|---|
| **LongMemEval** `--phase1` | `recall@5`, **`ndcg@5`**, `context_contains_rate` | `avg_judge` by category | `avg_recall_latency_ms` |
| **LoCoMo** retrieval-only | `evidence_hit_rate` | `avg_f1`, `avg_judge`, `cost_per_question_usd` | index storage size |

Expected directional reads:
- **B−A** (sparse): ↑ recall@5 / evidence_hit on rare-entity, number, and date queries (the known
  LoCoMo/LongMemEval weak spots). If B−A ≈ 0, sparse isn't earning its column — stop at A.
- **C−B** (colbert): ↑ `ndcg@5` (ranking) more than raw recall. Pair every NDCG gain with its
  latency cost — a +1% NDCG that doubles recall latency is a trade we surface, not hide.

### 8. Run protocol (both benches drive the loop)

1. **Tune cheap:** LongMemEval `--phase1` (retrieval-only, no LLM cost) to sweep fusion weights
   `[w_dense,w_sparse,w_bm25]` and rerank top-n. Fast inner loop.
2. **Cross-check:** LoCoMo retrieval-only `evidence_hit_rate` on the conv set used for the
   conv-26/conv-30 baselines — guards against LongMemEval-specific overfit.
3. **Payoff:** judged full runs (LoCoMo + LongMemEval) on the winning arm only — F1/judge/cost.
4. **Re-ingest discipline:** Moves 1–2 change the dense dimension AND add columns → **full
   re-ingest required**. KBs are reusable *within* an arm, **not** across arms — do not `--reuse`
   a bge-small KB for a bge-m3 arm.

---

## 9. Risks & decisions

- **Storage/latency:** colbert `List<Vector>` and (if added) MUVERA inflate index size and re-rank
  cost. Measure before MUVERA; rerank-only first.
- **Re-ingest cost:** four arms × full LoCoMo/LongMemEval ingest. Schedule; checkpoint per parent
  memory policy.
- **Fusion-weight overfit:** tune on phase1, validate on LoCoMo, never tune on the judged set.
- **Sparse dim (`SPARSE_DIM`):** must match the BGE-M3 sparse vocab head xervo emits — confirm from
  the `EmbedHybrid` output spec before fixing the schema constant.

## 10. One-paragraph recommendation

Do **Move 1 first** and gate on **B−A**. Because `EmbedHybrid` makes BGE-M3 dense+sparse a single
forward pass, Move 1 is a small, cheap, high-leverage diff that directly targets the lexical/temporal
recall gaps both benches punish. Only commit to Move 2 (ColBERT) if Move 1's NDCG/recall headroom
justifies the multi-vector storage and rerank latency — and measure that headroom on LongMemEval
`--phase1` before writing the rerank stage.

---

## Appendix — Key sites

- Presets/config: `uniko-store/src/config.rs:119,229-242,756,783-784`
- Schema: `uniko-store/src/schema/{chunks.rs:50-65, observations.rs:45-62, mod.rs:73-88}`
- Recall fusion: `uniko-store/src/repository/recall.rs:585`; `uniko-memory/src/recall/mod.rs:1358`
- Bench config: `uniko-bench/src/bench_config.rs:14-62,151-156`; `bench-configs/locomo-bge-openai.json`; `configs/lme_default.json`
- uni-db kernels: `uni-sparse-vector/src/{sparse,ops}.rs`; `uni-query-functions/src/{similar_to,fusion}.rs`; `uni-query/src/procedures_plugin/{sparse,search}.rs`
- xervo heads: `uni-xervo/src/provider/local_onnx/{embed_sparse,embed_multi_vector,embed_hybrid}.rs`; `api.rs:16-43`
