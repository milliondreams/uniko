# Beyond Cosine: Alternative Similarity Geometries for uniko Retrieval & Graph Curation

**Status:** Research synthesis + design/experiment plan (draft)
**Date:** 2026-06-18
**Scope:** Full-stack — (A) first-stage recall quality, (B) knowledge-graph geometry (edge-warrant / dedup / subsumption)
**Constraint:** Local-first inference (fastembed-rs / ONNX / mistralrs). Cloud OK for *training* only.
**Audience:** internal eng. Not website content.

---

## 0. Thesis

A "standard embedding" bundles four assumptions, and every alternative breaks exactly one:

> one **point** · in **flat (cosine) space** · compared by a **symmetric** dot product · at **document granularity**

| Assumption broken | Family | Best fit |
|---|---|---|
| document → token granularity | late interaction (ColBERT), learned sparse (SPLADE) | precise lexical/long-doc recall |
| flat → curved space | hyperbolic, mixed-curvature | hierarchy / taxonomy |
| point → region/distribution | box, order, Gaussian embeddings | **entailment, subsumption, dedup (asymmetric)** |
| symmetric → asymmetric/relational | KL-Gaussian, KG embeddings (RotatE/ComplEx) | direction, relations |

The honest conclusion up front (derived in §4–5): **the highest payoff-to-effort work for uniko is not exotic geometry.** It is, in order:

1. **Use the embeddings we already store** for entity-dedup and edge-creation — currently we don't (§3.2). Near-zero new infra.
2. **Swap dense embedder + reranker to the BGE-M3 / bge-reranker-v2-m3 pair** — BGE-M3 is *already a xervo preset* (dense-only today). This is a near config-only Phase 1; the sparse/ColBERT heads are **not** free (see §0-note).

> **Inference-runtime note (corrected).** uniko does **not** use `fastembed`. All model inference goes through **`uni-xervo`** (uni-db's runtime) via the `local/onnx` provider, registered as `ModelAliasSpec` aliases (`crates/uniko-store/src/config.rs:24,50`). xervo's `Embed` task returns **one dense vector**; its ONNX provider reads **one output tensor** and pools it (`uni-xervo/.../local_onnx/embedding.rs:399`). BGE-M3's sparse + ColBERT heads are **read and discarded** at the provider (`presets.rs:251`). So "three heads from one pass" is true of the *model* but not realisable in uniko without new xervo capability — see §3/§4 and §8.
3. **Add a learned-sparse leg** to the recall fusion (blocked on a uni-db sparse-vector type — §6.1).
4. **An asymmetric Gaussian re-ranking score for graph curation** (research track; box/hyperbolic/KG explicitly deprioritized with rationale in §5).

---

## 1. Current uniko baseline (as-built, grounded in code)

### Retrieval
- **Embedder:** default **BGE-Small-EN-v1.5**, 384-dim, `local/onnx` provider (`crates/uniko-store/src/config.rs:741`). Presets exist for Nomic-v1.5 (768d), MiniLM (384d), BGE-Large (1024d), EmbeddingGemma (768d).
- **Vector index:** uni-db **HNSW + scalar quantization** (`HnswSq`, m=16, ef_c=100), **cosine** metric (`config.rs:745,749`).
- **Recall cascade** (`crates/uniko-memory/src/recall/mod.rs`):
  - **Phase 1 (Compact):** vector search over `Fact`/`Procedure`/`Topic` (`:1253`). Coverage gate 0.75.
  - **Phase 2 (Expand):** RRF fusion of 5 episodic sources (Episode/Observation/Message × vector+fulltext) + optional temporal + graph PPR channels (`:1326`). MMR dedup via **Jaccard** > 0.85. Coverage gate 0.65.
  - **Phase 3 (Broaden):** 4 query-variant fan-out, RRF (`k=60`) across variants (`:1034`), then the **cross-encoder reranker** on top-`n` (`:1114`).
- **Hybrid (vector+BM25) within a source:** uni-db `similar_to([emb, content],[qvec, qtext])`, weights `vector_weight`/`bm25_weight` default 0.5/0.5 (`crates/uniko-store/src/repository/recall.rs:586`).
- **Reranker:** default `ms-marco-MiniLM-L-6-v2` (22M); `bge-reranker-base` preset exists but **disabled** (`config.rs:334`). Alias `rerank/default`.

### Graph / consolidation
- **Entity dedup:** **exact `entity_id` hash match only** (`crates/uniko-extract/src/ner/dedup.rs:168`). Entity `embedding` vectors are **stored but never queried** (`schema/entities.rs:31`). The dedup module comment claims embedding-similarity — **not implemented**.
- **ABOUT edges (Observation→Entity):** created by **substring string match** (`crates/uniko-extract/src/observations/mod.rs:461`). No warrant/validation. **Suspected root cause of `obs_chunk_missing_edges` (sessions 1–4 → 0 edges).**
- **MENTIONS edges:** created unconditionally for every extracted entity (`ner/dedup.rs:265`).
- **Consolidation** (`crates/uniko-memory/src/consolidation.rs`): groups Observations by (subject, predicate), **cosine-clusters object surface forms @ 0.88** (`:281,91`), mode-vote canonical, F38 contradiction @ 0.40, F39 entity-drift @ 4 invalidations. `SUPPORTED_BY` edges get **uniform weight 1.0** (`:444`).

**Key architectural facts that constrain the plan:**
- uni-db stores **one dense vector per node**; **no native sparse-vector type and no named-multi-vector**. New legs ⇒ new properties + indexes per node type, or a new uni-db type.
- The recall pipeline is **hardcoded to dense + BM25**; a new leg touches schema, config, ingest, recall Cypher, and fusion (5 layers — §6.2).
- Inference is **`uni-xervo` only** (no fastembed). xervo's `Embed` task = single dense vector; no `EmbedSparse`/`EmbedMultiVector` task variants exist (`uni-xervo/.../api.rs`). Sparse/late-interaction legs need **upstream xervo + uni-db work**, not just local plumbing.

---

## 2. Strong-baseline upgrade (do this first — it's not "alternative", it's overdue)

Before any new geometry, close the gap to 2025–26 SOTA *within the existing dense+BM25+rerank shape*. Our current `nomic/BGE-small + MiniLM` stack is a dated CPU baseline with large headroom.

| Slot | Current | Recommended (local, via xervo `local/onnx`) | Why |
|---|---|---|---|
| Embedder | BGE-small-384 / nomic-v1.5 (62.3 MTEB) | **BGE-M3 dense** (MIT, ONNX, *already a xervo preset*) or **Qwen3-Embedding-0.6B** (70.7 MTEB-v2, Apache) if xervo has an ONNX path | BGE-M3 dense is a config flip; Qwen3-0.6B is the raw-quality leader at small size (verify xervo ONNX support) |
| Reranker | ms-marco-MiniLM-22M (disabled BGE-base) | **bge-reranker-v2-m3** (0.6B, Apache, ONNX; xervo `Rerank` task) or **Qwen3-Reranker-0.6B** (MTEB-R 65.8 vs 57.0) | clean multilingual upgrade via the existing Rerank task; Qwen3-Reranker wins at equal params (confirm xervo ONNX path) |
| Dimension cost | fixed 384/768 | **Matryoshka** truncation (Qwen3/Nomic) — index at 256–512d | ~95–99% quality retained, ~75% storage cut |
| Query encoding | bare query | **instruction-conditioned** query side only (`Instruct: …\nQuery: …`) | +1–5%, more under domain shift |

**Highest-leverage single move:** swap to **BGE-M3 dense + bge-reranker-v2-m3** — both are plain `Embed`/`Rerank` tasks xervo already supports (BGE-M3 dense is a preset), both permissive, both CPU-viable. ⚠️ Unlike my earlier framing, this **does not** automatically unlock the sparse/late-interaction legs — those heads are discarded by the provider and require new xervo capability (§3/§4).

> ⚠️ Embedder swap changes the vector dimension (384→1024) ⇒ **full re-ingest / re-embed** of stored vectors. Budget for it; gate on a measured bench win (§7).

---

## PART A — First-stage recall quality

### 3. Learned sparse (SPLADE family)

**What & why.** A BERT MLM head predicts term weights over the whole 30k vocabulary with `log(1+ReLU(·))` saturation + FLOPS regularization → a sparse vector that lives in an **inverted index** but carries learned weights + term expansion. Beats BM25 because weights are contextual and it expands vocabulary (solves lexical mismatch) while staying index-compatible.

**Evidence (in-domain MS MARCO MRR@10 / zero-shot BEIR nDCG@10):**
- BM25 0.187 / ~0.43 → SPLADE-v3 **0.402 / 0.517** → OpenSearch doc-v3-gte **— / 0.546**.
- ~2× BM25 in-domain; zero-shot SPLADE-v3 (0.517) > ColBERTv2 (~0.499) > dense single-vector (~0.37–0.43). **Learned sparse + late interaction are the families that *reliably* beat BM25 out-of-domain** (many dense models don't).

**Local serving (xervo reality).** ONNX export of SPLADE models is solved upstream (`prithivida/Splade_PP_en_v1` ships ONNX; Apache), and BGE-M3's sparse head exists. **But xervo cannot surface it today:** the `local/onnx` provider reads one output tensor → one dense vector; there is **no `SparseEmbeddingModel` trait and no `ModelTask::EmbedSparse`**. xervo's own roadmap (`docs/proposals/provider-task-roadmap.md`, Phase 1.3) plans sparse via a `local/fastembed` provider but flags the open question "no `SparseEmbeddingModel` trait yet." **Doc-only / Efficient-SPLADE** (removes the online query encoder → BM25-class latency, P50 ~10ms vs ~176ms bi-encoder) is the right config once the trait lands.

**The blockers (two layers of `../uni/`).**
- **xervo (inference):** add a `SparseEmbeddingModel` trait + `ModelTask::EmbedSparse` + provider path to read a second ONNX output. ~2–3 days upstream; PR to uni-xervo. Reading BGE-M3's sparse head here means **no new model** (it's the same forward pass already running for dense).
- **uni-db (storage/index):** **no native sparse-vector type**. Options: (a) file a `SparseVector` + impact/WAND request (preferred long-term, follows our uni-db-bug workflow); (b) serialized `{term_id: weight}` + host-side dot product for the bench (works, doesn't scale to 10M+).

**Honest limits:** query-encoder tax (mitigated by doc-only), 512-token doc ceiling, efficiency tuning (λ_q/λ_d) trades robustness, occasional wrong expansions.

### 4. Late interaction (ColBERT) + MUVERA

**What & why.** One vector per token; score = **MaxSim** = Σ_i max_j (q_i·d_j). Recovers token-level matching that pooling blurs.

**The stale-claim correction (important).** ColBERT's BEIR edge over dense was real in 2021–22 but has **largely evaporated vs strong 2025 dense models** — April 2026 BEIR shows modern dense at 62–68 nDCG@10 vs ColBERTv2 ~55; Qdrant measured bge-small (0.737) beating ColBERT (0.696) OOD on SciFact. **Do not adopt late interaction expecting a free plain-text quality win.**

**Where it still wins:**
- **Visual / layout-rich documents (ColPali / ColQwen2)** — ViDoRe nDCG@5 81–89 vs ~65 for OCR+BM25; *eliminates the OCR/parse pipeline*. The text-cross-encoder alternative structurally doesn't exist. **This is the compelling case if/when uniko ingests PDFs** (ties to the PDF-OCR landscape work). ColQwen2 is Apache/MIT.
- **Long documents & rare-term/entity queries** — MaxSim preserves tokens pooling washes out (~5% nDCG as a reranker at negligible latency).

**MUVERA** compresses a multi-vector set into one fixed-dim vector (FDE) s.t. ⟨FDE(q),FDE(d)⟩ ≈ MaxSim → **reuse ordinary single-vector ANN** (HNSW), then rerank with true MaxSim. ~10% higher recall at ~90% lower latency vs PLAID; Qdrant reproduced ~7–8× speedup at NDCG parity. This is the bridge that makes late interaction deployable on our existing index — **as a reranker pattern, not a first-stage multi-vector index.**

**Local serving reality (layering matters).** The dependency arrow is **uni-db → uni-xervo** (uni-db depends on xervo; `uni/crates/uni-store/Cargo.toml:58`, "all ONNX access flows through uni-xervo's provider facades"). xervo has **zero** uni-db references — it is the lower inference layer and cannot depend on uni-db. xervo's multimodal proposal (`xervo-multimodal-api-proposal.md:110`) therefore *scopes itself out* of indexing: "ColBERT… requires a multi-vector index **in uni-db, a separate uni-db enhancement**; this proposal targets single-vector outputs only." The work splits across the two layers, sequenced producer→consumer (arrow never reverses):
  - **xervo (producer):** emit per-token vectors instead of one pooled vector — small, self-contained, no uni-db needed.
  - **uni-db (consumer — the real blocker):** add a multi-vector index/column + wire auto-embed to fetch multi-vectors from xervo and store them. This is the substantive lift.

Plan once unblocked = BGE-M3 ColBERT vectors + roll-our-own MaxSim over top-k candidates (rerank pattern, no first-stage multi-vector index needed). Storage blowup (per-token ~10–100× dense) is mitigated by token pooling (−50% vectors at ~100% quality) + 2-bit residual quant.

### 5. PART A verdict

Late interaction as a **first-stage** index is not worth the uni-db storage work for plain text today. The pragmatic Part-A program is:
1. **BGE-M3 dense** as the retrieval backbone now (config flip; preset exists), with the **sparse head added once xervo grows a `SparseEmbeddingModel` trait** (~2–3d upstream) + uni-db sparse storage.
2. **ColBERT-MaxSim as an optional reranker** over the top-k (using BGE-M3's multi-vector head), competing against the cheaper cross-encoder reranker — keep whichever wins the bench.
3. **ColQwen2 reserved for the PDF/visual-document track** (separate initiative).

---

## PART B — Knowledge-graph geometry (asymmetric similarity)

### 6. The core argument

Cosine is **symmetric**: `sim(dog, animal) = sim(animal, dog)`. It *structurally cannot* represent "dog is-a animal" or "A is redundant-with/subsumed-by B." Region/asymmetric embeddings replace the symmetric metric with a directional containment relation. uniko has three live decision points that are *asymmetric-similarity problems currently solved by string match or symmetric cosine*:

| Decision point | File | Today | Asymmetric upgrade |
|---|---|---|---|
| Entity dedup | `ner/dedup.rs:168` | exact `entity_id` hash | cosine fallback → directional "is new entity a variant of existing" |
| ABOUT edge | `observations/mod.rs:461` | substring match (0 edges bug) | embedding-sim fallback when string fails |
| Object clustering | `consolidation.rs:281` | symmetric cosine @ 0.88 | subsumption pre-pass: "is object X dominated by Y?" |
| SUPPORTED_BY weight | `consolidation.rs:444` | uniform 1.0 | obs↔fact cosine as weight |

### 6.1 What the research says (candid)

- **Box / order / cone embeddings:** theory sound (containment = subsumption, antisymmetric + transitive by construction). Evidence real on graded entailment (HyperLex ρ: cosine 0.205 → DOE 0.590 → LEAR 0.686 — ~3×) and **direction detection (which cosine simply cannot do)**. **But tooling is abandoned** (IESL `box-embeddings` last real commit 2022; only `geoopt` is alive, and it's a trainer). **And asymmetric/non-metric scores break the ANN stack** — HNSW/IVF assume symmetry + triangle inequality → brute-force O(N·d) scan. Research-grade.
- **Gaussian / density embeddings:** each item = N(μ,Σ); **KL divergence is asymmetric** (captures specificity/direction) and **O(d)** like cosine. **GaussCSE (EACL 2024)** fine-tunes BERT to emit μ+σ *per sentence*, hits 92–98% on entailment-*direction* vs ~62–69% baseline, weights released. **This is the single most practical "region from text" entry point** and the lowest-risk asymmetric option.
- **Hyperbolic embeddings:** 5-D Poincaré beats 200-D Euclidean on WordNet hierarchy reconstruction. But there is **no production hyperbolic text encoder**, numerical instability (NaNs near boundary), and the edge largely vanishes once Euclidean has ≥50 dims. **Only defensible for a small explicit hierarchy layer** (entity is-a / goal-task trees, 5–10 dims).
- **KG embeddings (RotatE/ComplEx/ULTRA):** strong on *static, curated* link-prediction (FB15k-237 MRR ~0.34–0.41). **But the research is unanimous they are the wrong tool for a noisy, continuously-growing extracted graph:** transductive (new entity = OOV until retrain), embedding drift on every retrain, steep degradation under extraction noise (Pujara EMNLP'17). **Production GraphRAG / HybridRAG / Zep all deliberately skip learned KGE** in favor of "embed node/edge text + traverse." If structural signal is ever wanted, use an **inductive foundation model (ULTRA)**, not classic KGE — and validate against our noise level first.

### 6.2 PART B verdict

- **Don't** adopt box/hyperbolic/KG embeddings as a retrieval leg now. They break the ANN stack, depend on abandoned tooling, and (KGE) fail in exactly our noisy-dynamic regime.
- **Do** pursue **asymmetric Gaussian (GaussCSE-style μ+σ head on our frozen local encoder)** as a **re-ranking/decision score** over cosine-ANN candidate sets at the four graph decision points. O(d), brute-force over top-k (not the whole graph), trainable in cloud on NLI + our own pairs.
- **Keep in mind** the durable query2box *idea* — multi-hop conjunction as region intersection, not a materialized join — but our uni-db Cypher `ALONG`/`FOLD` traversal already owns precise multi-hop; the soft layer is a later experiment.

---

## 7. Phased plan (sequenced by payoff : effort)

Each phase: **hypothesis → change → metric → decision gate → kill criteria.** Benches = LongMemEval (longmemeval-bench, has `--reranker/--embedding/--catalog` flags) and LoCoMo (conv-26 + LoCoMo10 baselines). Local inference throughout; cloud only for training heads.

### Phase 0 — Use the embeddings we already store (days, ~0 new infra) ★ do first
- **Hypothesis:** entity-dedup and ABOUT-edge coverage improve (and `obs_chunk_missing_edges` is fixed) by adding an embedding-similarity fallback where string match fails.
- **Change:** (1) `ner/dedup.rs:168` — on `entity_id` miss, cosine-NN over existing `Entity.embedding`, merge if > τ. (2) `observations/mod.rs:461` — on substring miss, embedding-sim fallback for ABOUT linking. (3) `consolidation.rs:444` — set `SUPPORTED_BY.weight` = obs↔fact cosine.
- **Metric:** entity-edge count on sessions 1–4 (currently 0); dedup precision/recall on a hand-labeled set; LoCoMo retrieval hit-rate.
- **Gate:** sessions 1–4 produce non-zero, precision-checked edges; no regression in LoCoMo hit-rate.
- **Kill:** if embedding fallback floods false-positive edges (precision < ~0.8 on the labeled set), revert to string-only and escalate to a learned score (Phase B-research).

### Phase 1 — BGE-M3 + bge-reranker-v2-m3 baseline (1–2 wk)
- **Hypothesis:** modern embedder+reranker lifts LME judge score / LoCoMo hit-rate over nomic+MiniLM.
- **Change:** add BGE-M3 (dense) + bge-reranker-v2-m3 presets; flip catalog; **re-ingest** one bench KB; enable reranker.
- **Metric:** LME judge score & F1, LoCoMo hit-rate, ingest+query latency (re-measure per `project_ingest_slowdown`).
- **Gate:** ≥ baseline judge score at acceptable latency. **Verify the bench exercises the swapped path** before claiming the win (per `feedback_no_overclaim_bench_impact`).
- **Kill:** latency blows the local budget with no quality gain → keep nomic, try Qwen3-0.6B via candle instead.

### Phase 2 — Learned-sparse leg (2–4 wk; gated on storage)
- **Hypothesis:** fusing BGE-M3 **sparse** as a third RRF leg beats dense+BM25, especially long-chunk / rare-term queries (MLDR-style +12 swing).
- **Change:** **two upstream prerequisites in `../uni/`** — (i) xervo: add `SparseEmbeddingModel` trait + `ModelTask::EmbedSparse` to surface BGE-M3's sparse head (~2–3d, PR); (ii) uni-db storage: file a `SparseVector` request, or use serialized-map + host-side scoring for the bench. Then locally: add `sparse_embedding` property + ingest + a third term in `similar_to(...)`/RRF with `sparse_weight`.
- **Metric:** LME/LoCoMo with sparse leg on/off; latency delta.
- **Gate:** sparse leg adds measurable hit-rate at < X ms/query overhead.
- **Kill:** if uni-db storage path is too costly and BM25 already captures most of it → defer until uni-db ships native sparse.

### Phase 3 — ColBERT-MaxSim reranker (3–6 wk; higher risk)
- **Hypothesis:** MaxSim rerank (BGE-M3 multi-vector) over top-k beats the cross-encoder reranker on entity/long queries.
- **Change:** store pooled+quantized ColBERT vectors for candidate nodes; implement MaxSim rerank in Rust over top-k (no native multi-vector index needed). MUVERA-FDE only if scale forces a first-stage.
- **Metric:** head-to-head vs cross-encoder on LME/LoCoMo; storage cost.
- **Gate:** beats cross-encoder reranker net of storage/latency.
- **Kill:** cross-encoder already saturates → shelve; **reserve late interaction for the PDF/ColQwen2 visual-doc track.**

### Phase 4 — Asymmetric Gaussian edge-judge (research track, parallel; train in cloud)
- **Hypothesis:** a μ+σ head (GaussCSE-style) on our frozen encoder, scored by KL, makes better dedup/subsumption decisions than symmetric cosine at the four graph decision points.
- **Change:** train a 2-layer μ+σ head on NLI + WordNet + **our mined edge-warrant pairs** (positive/negative dedup & subsumption pairs from LoCoMo/observation chunks); KL score as a re-ranker over cosine-ANN top-k. Encoder frozen → ~hours on one cloud GPU; inference O(d) local.
- **Metric:** dedup F1 & subsumption-direction accuracy on a labeled set; effect on consolidation cluster purity and downstream LoCoMo answer quality.
- **Gate:** beats cosine on direction accuracy *and* doesn't hurt end-to-end answers.
- **Kill:** no lift over Phase-0 cosine fallback → asymmetric geometry isn't paying for itself here; document and stop. **Explicitly do not** proceed to box/hyperbolic/KGE.

---

## 8. Risks & cross-cutting decisions

- **The upstream stack is the critical path for A — two repos, layered uni-db → uni-xervo** (uni-db depends on xervo, not vice versa; xervo is the inference *producer*, uni-db the storage/index *consumer*). Work sequences producer→consumer; the arrow never reverses, so no circular dependency.
  - **uni-xervo (`../uni-xervo/`, producer — small):** `Embed` returns one dense vector; add a `SparseEmbeddingModel` trait + `ModelTask::EmbedSparse` to emit BGE-M3's sparse head (~2–3d). Multi-vector emission is analogous, larger output. Self-contained; no uni-db needed.
  - **uni-db (`../uni/`, consumer — the real blocker):** no sparse type, no named-multi-vector storage/index, plus auto-embed wiring to request the new outputs from xervo. This is where most of the Phase 2/3 effort lives.
  - Both are separate-project PRs; file them early, follow the uni-db-bug isolated-repro workflow, don't hack around silently. **Phase 1 (dense + reranker) needs neither.**
- **Verify xervo's ONNX path per model before committing.** BGE-M3 dense and bge-reranker-v2-m3 are confirmed-shaped for xervo (`Embed`/`Rerank`); Qwen3-Embedding/Reranker need an ONNX export xervo's `local/onnx` can load — confirm before assuming.
- **Embedder swaps force re-ingest** (dimension change). Always gate on a measured bench win first.
- **Asymmetric scores break ANN** — keep them strictly as top-k re-rankers, never as the first-stage index.
- **Measure before optimizing / don't overclaim** — every phase gate requires the bench to actually exercise the changed path (`feedback_measure_before_optimize`, `feedback_no_overclaim_bench_impact`).
- **Licenses:** prefer Apache/MIT (BGE-M3 MIT, bge-reranker-v2-m3 Apache, Qwen3 Apache, ColQwen2 Apache/MIT). Avoid NV-Embed-v2 (CC-BY-NC), jina-colbert-v2 (CC-BY-NC), ColPali (Gemma license) for shipped paths.

---

## 9. One-paragraph recommendation

Do **Phase 0 now** — it fixes a real bug (`obs_chunk_missing_edges`) and an inconsistency (stored-but-unused entity embeddings) with essentially no new infrastructure, and it's the cheapest test of whether asymmetric/embedding signals help the graph at all. In parallel, run **Phase 1** (BGE-M3 + bge-reranker-v2-m3) as the overdue baseline upgrade that also unlocks the sparse and late-interaction heads for later phases. Treat Parts A-sparse/late-interaction (Phases 2–3) and B-Gaussian (Phase 4) as **gated experiments**, each with a kill criterion, not committed features. Explicitly **deprioritize box, hyperbolic, and KG embeddings** for retrieval: the research is clear that they break our ANN stack, lean on abandoned tooling, and (KGE) fail in our noisy-dynamic-graph regime — every production graph-memory system reaches the same conclusion.

---

## Appendix — Sources

**Learned sparse:** SPLADE-v3 (arxiv 2403.06789); SPLADE v2 (2109.10086); SPLADE++ (2205.04733); Efficient SPLADE (SIGIR'22, 2207.03834); BEIR (2104.08663); naver/splade-v3, prithivida/Splade_PP_en_v1, opensearch-project neural-sparse-doc-v3; OpenSearch sparse blog (opensearch.org/blog/improving-document-retrieval-with-sparse-semantic-encoders).

**Late interaction / MUVERA:** ColBERTv2 (2112.01488); PLAID (2205.09707); MUVERA (2405.19504) + Google blog; Qdrant MUVERA (qdrant.tech/articles/muvera-embeddings); answerai-colbert-small-v1; lightonai/GTE-ModernColBERT-v1; ColPali (2407.01449); vidore/colqwen2-v1.0; token pooling (2409.14683); BEIR update Apr'26 (app.ailog.fr/en/blog/news/beir-benchmark-update).

**Region / asymmetric:** box-lattice (1805.06627); SmoothBox (ICLR'19 H1xSNiRcF7); GumbelBox (2010.04831); iesl/box-embeddings; Order Embeddings (1511.06361); Hyperbolic Cones (1804.01882); word2gauss (1412.6623); Density Order Embeddings (1804.09843); **GaussCSE (2305.12990)** + github yoda122/GaussCSE; query2box (2002.05969); BetaE (2010.11465); HyperLex (J17-4004); LEAR (1710.06371); Lexical Memorization (N15-1098); geoopt (github geoopt/geoopt).

**Hyperbolic / KG:** Poincaré (1705.08039); Lorentz (1806.03417); Mixed-Curvature (ICLR'19 HJxeWnCcF7); GIE (2206.12418); TransE (NIPS'13); DistMult (1412.6575); ComplEx (1606.06357); RotatE (1902.10197); HAKE (1911.09419); ULTRA (2310.04562); Sparsity & Noise (D17-1184); PyKEEN (github pykeen/pykeen); Microsoft GraphRAG; HybridRAG (2408.04948).

**Local SOTA embedders/rerankers:** Qwen3-Embedding (2506.05176) + Qwen/Qwen3-Embedding-{0.6B,4B,8B}; BGE-M3 (2402.03216) + BAAI/bge-m3; NV-Embed-v2 (CC-BY-NC); stella_en_*_v5; nomic-embed-text-v1.5/v2-moe (2502.07972); Matryoshka (2205.13147); INSTRUCTOR (2212.09741); bge-reranker-v2-m3; Qwen3-Reranker; mxbai-rerank-v2; jina-reranker-v2 (CC-BY-NC). (NB: `fastembed-rs` appears in the public RAG literature but is **not** used by uniko — see runtime note.)

**Internal code anchors:** `crates/uniko-store/src/config.rs`; `crates/uniko-memory/src/recall/mod.rs`; `crates/uniko-store/src/repository/recall.rs`; `crates/uniko-extract/src/ner/dedup.rs`; `crates/uniko-extract/src/observations/mod.rs`; `crates/uniko-memory/src/consolidation.rs`; `crates/uniko-store/src/schema/{entities,facts,observations}.rs`; `crates/uniko-store/src/storage/mod.rs` (`embed_catalog`). **uni-xervo:** `crates/uni-xervo/src/{api.rs,traits.rs,provider/local_onnx/{embedding.rs,presets.rs}}`; `docs/proposals/provider-task-roadmap.md` (Phase 1.3 sparse, deferred ColBERT).

*Confidence flags: BM25/dense BEIR averages (~0.43 / ~0.37–0.43) and ColBERTv2 ~0.499 are well-established literature values not primary-verified this pass. Region-embedding tooling-maturity and "production GraphRAG skips KGE" are synthesized from multiple secondary sources. Treat all benchmark decimals as directional, not exact.*
