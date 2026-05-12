# RFE: P4 Recall Evolution — Structured Facts, Session-Aware Ranking, NLP Triple Quality

**Status:** Draft
**Author:** Rohit Rai (with Claude pairing)
**Date:** 2026-05-12
**Scope:** uniko-memory recall cascade, P3 observation extraction, P4 consolidation
**Target benchmarks:** LoCoMo (10-conversation full sweep)

---

## 1. Executive summary

P4 Consolidation shipped (commits `3460b65`, `e2d7e62`) and lifted LoCoMo conv-30 from
**0.494 → 0.815** judge (gpt-4o-mini gen + judge). The current best stack is:

- LLM-extracted Fact triples at ingest (`--extract-triples-llm-alias`)
- MiniLM-L6 cross-encoder reranker (`--reranker`)
- Recall Phase 1 (Compact) merging the top-3 Facts into the Phase-3 bundle by score
- `recall_limit=15`

Conv-26 at this stack scores **0.717**. The bench is now sweeping all 10 conversations
for a Mem0-comparable headline.

Ablations during the conv-26 deep-dive identified four classes of remaining work,
ordered by ROI:

1. **Temporal-slot plumbing on Observation** (~150 LOC, no ML) — directly unblocks
   Q29/Q53-style temporal failures by promoting SRL's already-extracted ARGM-TMP
   from rendered content into a structured slot.
2. **NLP matcher quality** (~250 LOC, no ML) — lemmatization, object trimming,
   stop-word/light-verb filtering. Closes ~80% of the SRL-vs-LLM fact-quality gap
   without training.
3. **Session-aware reranking** (~200 LOC) — Facts and Observations boost the
   ranking of their containing session's Chunks rather than competing for bundle
   slots. Addresses Adversarial collapse, position-bias artifacts, and
   crowding-out effects.
4. **Open-domain prompt tweak** (~5 LOC) — license speculative inference for
   "would X likely…?" questions where context contains preferences/behaviors.

F38 (contradiction) and F39 (drift) detection unlock as side effects of #1 and #2.

---

## 2. Motivation and current state

### 2.1 What we proved with P4

| Configuration | conv-30 judge | conv-26 judge |
|---|---|---|
| Pre-P4 baseline (Gemma-4 + gpt-5-mini judge) | 0.790 | 0.526 |
| P4 + LLM facts + MiniLM rerank + cap=3 | 0.802 | 0.750 |
| P4 + LLM facts + MiniLM rerank + cap=0 | 0.815 | 0.717 |

cap=3 (interleave) wins on conv-26; cap=0 wins on conv-30. Conversation-dependent,
nets positive at +0.017 weighted by question count. Currently `PHASE1_FALLBACK_CAP=3`
in `crates/uniko-memory/src/recall/mod.rs`.

### 2.2 What failure analysis revealed

**Conv-26 open-domain (13q, judge 0.308) is the weakest tier.** Failure pattern
across 9 of 13 losses:

- 7 of 9 say "not mentioned" (LLM abstained)
- 6 of those 7 had retrieved evidence

Speculative-inference questions ("Would Caroline have Dr. Seuss books on her
bookshelf?") trigger over-abstention because our gen prompt licenses *direct*
inferences but not *speculative* inferences from preferences.

**Cap=3 vs prepend deep-dive (Q29 — Temporal flip-win):**

- Same 15 reranker items in both bundles (byte-identical content)
- Same 3 Facts in both (also byte-identical: `"in pottery class"`, `"pottery"`,
  `"pottery class"`)
- cap=3 places Facts at positions 9-12 → judge=1.0 ("Melanie went... last
  Friday from the context")
- prepend forces Facts to positions 1-3 → judge=0.0 ("Melanie went on July 15"
  — wrong by one day)

Position effect dominates. Forced Fact prominence at positions 0-2 distorts the
LLM's framing. Adversarial F1 collapses from 0.553 to 0.383 under prepend
because head-positioned Facts encourage confabulation on unanswerable questions.

**The LLM-extraction win mechanism:** LLM triples are cleaner because of surface
cleanup (lemmatization, object trimming, stop-word filtering), not because of
deeper semantic understanding. 4 of 5 LLM advantages are rule-fixable in
matcher.rs.

### 2.3 What we discovered about the existing NLP model

The DeBERTa-v3-xsmall NLP model (`kniv-deberta-nlp-base-en-xsmall`) already ships
heads for NER, POS, DEP, CLS, and SRL. Per-verb SRL re-forward is wired
(`compute_srl_frames()`), enabled by default
(`UnikoConfig.nlp_srl_enabled=true`), and emits ARGM-TMP/ARGM-LOC role labels
via the bundled `english.yml` rule patterns (`srl_action`,
`srl_action_temporal_only`, `srl_action_locative_only`).

**The temporal information is extracted but discarded after rendering.** The
matcher captures `ARGM-TMP → time` variable, interpolates it into the rendered
content string ("Melanie took kids Last Fri"), and never preserves it as
structured data on the Observation node.

The LLM's apparent temporal advantage on Q29 isn't extraction skill — it's that
we throw away what our model already finds.

---

## 3. Proposed enhancements

### Phase A — Temporal slot plumbing (highest leverage, no ML)

**Goal:** Preserve ARGM-TMP captures as structured fields on Observation, resolved
to absolute dates via `resolve_temporal()` at ingest time.

**Changes:**

| File | Edit |
|---|---|
| `crates/uniko-extract/src/nlp/decode.rs` | Add `temporal: Option<String>` to `DepObservation` |
| `crates/uniko-extract/src/observations/rules_engine/matcher.rs` | In `try_match_srl`, return captured `time` variable as `temporal` field on `DepObservation` (keep template render unchanged for back-compat) |
| `crates/uniko-extract/src/observations/types.rs` | Add `temporal_phrase: Option<String>` and `temporal_anchor: Option<DateTime<Utc>>` to `RawObservation` |
| `crates/uniko-extract/src/observations/mod.rs` | When converting `DepObservation → RawObservation`, call `temporal::resolve_temporal(temporal_phrase, msg_timestamp)` and populate both fields |
| `crates/uniko-store/src/schema/observations.rs` | Add `temporal_phrase: String` (nullable) and `temporal_anchor: DateTime` (nullable, BTree-indexed) |
| `crates/uniko-extract/src/observations/temporal.rs` | Extend regex coverage: `"Last Fri"`/`"Last Friday"` (specific weekday), `"two weekends ago"`, `"this Tuesday"`, weekday abbreviations, forward variants (`"next month"`, `"in N days"`) |
| `crates/uniko-memory/src/consolidation.rs` | Compute `valid_at_lo = min(contributors.temporal_anchor ?? observed_at)` for derived Facts |
| `crates/uniko-bench/src/lib.rs` | `format_context()` emits `temporal=YYYY-MM-DD` field when populated |

**Estimated effort:** ~150 LOC + ~80 LOC of temporal regex + ~70 LOC tests = ~300 LOC.
Half a day.

**Expected impact:**
- Q29/Q53 style temporal failures: ~+0.10 on Temporal judge (5/26 conv-26 temporal
  failures had retrieved evidence but no resolved date)
- Multi-hop temporal sub-questions also benefit
- No ML training, no model changes

**Backwards-compatibility:** Observation schema gains nullable columns. Existing
ingested KBs work unchanged; new fields are `None` for pre-Phase-A nodes.

### Phase B — NLP matcher quality (rule-based cleanup)

**Goal:** Bring SRL/DEP-derived triples to ~80% of LLM-extracted-triple quality
without training.

**Changes in `crates/uniko-extract/src/observations/rules_engine/matcher.rs`:**

1. **Object-phrase cleanup** (~40 LOC) applied to `collect_children()` output
   before template render:
   - Strip leading prepositions/articles: `"in pottery class"` → `"pottery class"`
   - Strip trailing punctuation inside captured phrases: `"yesterday!"` → reject;
     `"it happen."` → `"it happen"`
   - Reject if remaining text is pure pronoun, stop-word, or single-token
     deictic (`"what"`, `"it"`, `"thing"`)
   - Reject if temporal-only (already handled by Phase A — temporal goes to
     its own slot, not the object slot)

2. **Predicate lemmatization** (~60 LOC) in `normalize_predicate()`:
   - Use POS tag to identify verb form; map to lemma
   - Strip auxiliaries: `"is starting"` → `"starts"`, `"has been doing"` →
     `"does"`
   - Reuse `chrono`-style rule table for English morphology (irregular verbs
     hard-coded; regular forms via suffix rules)
   - Reduces predicate vocabulary by ~40% based on conv-26 distribution

3. **Per-frame quality filter** (~30 LOC) as a final rejection step:
   - Require ≥1 non-stop content word in object phrase
   - Reject if `subject == object` literally
   - Reject if predicate is a generic "light verb" (`do`, `make`, `get`, `have`,
     `be`) with no specifying complement
   - Raise minimum rendered-content threshold from 3 → 4 content words

4. **Negation extraction** (~20 LOC):
   - Detect `not`/`n't`/`never`/`no` as `advmod` children of the anchor verb
   - Add `polarity: bool` field on `DepObservation` (default `true`, `false`
     when negated)
   - Schema addition on `Observation.polarity: bool`

5. **Multi-frame fusion** (~30 LOC) post-step in `extract_with_rules`:
   - When two patterns produce the same `(subject, predicate)` with different
     objects, prefer the longer/more-specific object
   - When a predicate is a copular phrase (`is happy`), prefer the attributive
     form over the bare verb

**Estimated effort:** ~200 LOC + ~100 LOC tests = ~300 LOC. One day.

**Expected impact:**
- Top-15-fact quality jumps from current `"do | what"`, `"make | it happen."`
  style to LLM-comparable on the surface-cleanup axis
- F38 contradiction detection becomes possible (depends on polarity)
- Predicate dedup improves Fact `observation_count` accuracy (more contributors
  per Fact → higher confidence under Laplace smoothing)

**Note:** This does *not* close the semantic-equivalence gap ("buy" vs "purchase"
collapsing to the same Fact). That gap is addressed in Phase D if needed.

### Phase C — Session-aware reranking

**Goal:** Eliminate the "Facts vs Chunks compete for bundle slots" pattern by
making Facts and Observations *boost* the ranking of their containing session's
Chunks instead of occupying slots themselves.

**Design (from earlier conversation):**

For each query:

1. **Phase 1 (Facts)** → top-30 `(fact_node, fact_score)` via cosine on
   `Fact.embedding`
2. **Phase 2 (Observations)** → top-30 `(obs_node, obs_score)` via cosine on
   `Observation.embedding`
3. **Phase 3 (Chunks)** as today → hybrid (vector + BM25) → `(chunk_node,
   chunk_score)`
4. **Walk each Fact/Obs to its containing session chunks** (batched Cypher):
   ```cypher
   MATCH (f:Fact)<-[:SUPPORTED_BY]-(o:Observation)-[:OBSERVED_IN]->(:Message)
        -[:IN_SESSION]->(:Session)-[:HAS_CHUNK]->(c:Chunk {chunk_type:'session'})
   WHERE f.fact_id IN $hit_fact_ids
   RETURN f.fact_id AS hit, id(c) AS chunk_id
   ```
5. **Aggregate boost per chunk**:
   ```
   chunk_boost[c] = α · Σ fact_scores[f]  for f hitting c's session
                  + β · Σ obs_scores[o]   for o hitting c's session
   final_score[c] = chunk_score[c] + chunk_boost[c]
   ```
6. **Return top-k session chunks** by `final_score`. Bundle is 100% Chunks
   (preserves evidence_hit semantics, gold-bearing text always surfaces).

**Tunables:** `α` (fact boost weight, default 0.3), `β` (observation boost
weight, default 0.2). Both small enough that boost moves chunks by ~one rank
position without overwhelming the primary cosine + reranker ranking.

**Open question:** session granularity vs per-chunk-window granularity. v1 uses
whole-session for simplicity; revisit if it over-promotes long sessions.

**Changes:**

| File | Edit |
|---|---|
| `crates/uniko-memory/src/recall/mod.rs` | New `session_boost_signals()` function; rewrite `recall()` to use boost-then-rerank pipeline instead of merge-after-rerank |
| `crates/uniko-memory/src/recall/intent.rs` | No change |
| `crates/uniko-bench/src/main.rs` | Add `--phase1-strategy {merge,boost,off}` flag to toggle between v1 (current cap=3 merge) and v2 (session boost) for A/B comparison |

**Estimated effort:** ~250 LOC + ~80 LOC tests = ~330 LOC. One day.

**Expected impact:**
- Adversarial F1 recovers (no head-positioned Facts forcing confabulation)
- Multi-hop benefits because sessions where *both* hops appear get a double
  boost — first time a recall mechanism rewards entity co-occurrence rather
  than just entity-keyword match
- Open-domain benefits if Phase B already made Facts useful enough to drive
  session boost on speculative questions

### Phase D — ML extensions (only if Phases A-C don't close the gap)

**D.1 Coreference resolution beyond first-person:**

Current matcher handles `I/we → speaker` via `SentenceContext`. Second-person
(`you`) and third-person (`he/she/they`) require coreference. Two options:

- **Rule-based**: extend `SentenceContext` to track recent named-entity mentions;
  `she` resolves to the most recent female-named entity in the conversation.
  ~80 LOC; brittle on long contexts.
- **Model-based**: add a coref task head on the existing DeBERTa encoder. Fine-tune
  on OntoNotes coref data. ~10M new parameters; shares the 35M encoder.

**D.2 Triple-extraction head (knowledge distillation):**

If Phase B's surface cleanup isn't enough, add a sequence-to-sequence head on the
existing DeBERTa encoder that maps encoded sentence → `(subject, predicate,
object, polarity, temporal)` JSON. Train ONLY the new head on `(sentence, LLM-extracted
triple)` pairs gathered from our `--extract-triples-llm-alias` runs.

- Encoder stays frozen (35M params, already trained)
- New head: ~10M trainable parameters
- Inference: ~5ms per sentence vs ~500ms LLM call (100× speedup)
- Quality: typically reaches 80-90% of teacher LLM quality on the trained
  distribution

This is **not** "train a new model" — it's "add a head and fine-tune." Avoids
the temptation to train a 400M-param REBEL-style standalone extractor.

**D.3 Semantic-equivalence dedup:**

For Facts where surface predicate forms differ but semantics match
(`(Caroline, purchased, books)` vs `(Caroline, bought, books)`), use the
embedding similarity already computed for Phase 1 retrieval. At consolidation
time, after grouping by `(subject, predicate)`, optionally collapse near-duplicate
groups whose centroid cosine > 0.85.

~50 LOC in `consolidation.rs`.

---

## 4. Prompt-level enhancements

### 4.1 Open-domain speculative inference

Current system prompt (`crates/uniko-bench/src/query.rs:122-126`):

> "You are a helpful assistant answering questions about conversations. Answer
> using the provided context. You may paraphrase or make direct inferences from
> what the context says, including using the `session_date` field to resolve
> relative dates like 'yesterday' or 'next month'. If the answer is genuinely
> not present in the context, say 'The information is not mentioned in the
> conversation.' Answer concisely in one or two sentences."

Proposed addition:

> "...For speculative questions phrased as 'Would X likely…?' or 'What would X
> think about…?', reason from the speaker's stated preferences, beliefs, and
> behaviors to give a reasoned inference rather than abstaining. Only say
> 'The information is not mentioned' when no relevant preferences or behaviors
> appear in the context."

5-line change. Expected to lift conv-26 open-domain from 0.308 toward
0.5-0.6 (matching the 4/13 wins where the model already infers correctly).

### 4.2 Mem0-comparable judge prompt (optional)

For fair Mem0 comparisons, our LLM judge prompt could match Mem0's explicit
leniency:

> "...you should be generous with your grading — as long as it touches on the
> same topic as the gold answer, it should be counted as CORRECT. For temporal
> questions, even if the format differs ('May 7th' vs '7 May'), consider it
> CORRECT if it's the same date or time period."

Mem0 publishes 0.669 across 10 conv with this prompt. Our judge uses a default
gpt-4o-mini call without leniency hints. Worth a side-by-side comparison
before claiming SOTA.

---

## 5. Implementation order

```
Step 1 — wait for 10-conv sweep at current best stack (cap=3) to finish
Step 2 — Phase A (temporal slot plumbing) + Phase B (matcher rule cleanup)
         can ship in one PR — they touch overlapping files in matcher.rs
Step 3 — re-bench conv-26 + conv-30 against Phase A+B
Step 4 — Phase C (session-aware reranking) as a separate PR with A/B flag
Step 5 — full 10-conv re-sweep against Phase A+B+C
Step 6 — open-domain prompt tweak as a one-line follow-up
Step 7 — Phase D items, only if needed
```

Each step is independently shippable. Phase A unlocks F38/F39 spec items.
Phase C is the only architectural change; A, B, D.1-D.3 are local refinements.

---

## 6. Success metrics

**Per-phase targets** (single-conversation conv-26 retest after each phase):

| Phase | Single | Multi | Temporal | Open-d | Adv F1 | Overall |
|---|---|---|---|---|---|---|
| Baseline (current cap=3) | 0.971 | 0.469 | 0.730 | 0.308 | 0.553 | 0.750 |
| +Phase A (temporal) | 0.97 | 0.50 | **0.85** | 0.31 | 0.55 | **0.79** |
| +Phase B (matcher rules) | 0.98 | **0.55** | 0.85 | 0.35 | **0.60** | **0.82** |
| +Phase C (session boost) | 0.98 | **0.60** | 0.86 | **0.50** | **0.65** | **0.86** |
| +open-domain prompt | 0.98 | 0.60 | 0.86 | **0.65** | 0.65 | **0.88** |

**Full 10-conv target after all phases:** judge ≥ 0.75 (beating Mem0's published
0.669 cross-conv with matching gpt-4o-mini both-sides setup).

**Hard floors per phase:**
- No regression > 0.02 on any category
- Adversarial F1 ≥ 0.55 (current floor)
- Recall latency ≤ 2 sec at p95 (current MiniLM stack: ~1.6 sec)

---

## 7. Risks and mitigations

**R1: SRL re-forward is expensive (N forwards per sentence).** Currently
N = #verbs in sentence, typically 1-3. Already in production. Phase A
doesn't add cost; Phase B is rule-only.

**R2: Phase C over-promotes long sessions.** A session with many Facts and
Observations might dominate even when none of them are individually
high-relevance. Mitigation: divide `chunk_boost[c]` by `log(session_size)` to
normalize.

**R3: Phase A's resolved dates conflict with chunk-text dates.** If LLM sees
both `temporal=2023-07-14` and "Last Fri" in the chunk text, it might prefer
one over the other inconsistently. Mitigation: temporal field is presented as
authoritative in the prompt format; chunk text retains the original phrasing
for context.

**R4: Phase B's lemmatization breaks domain-specific verbs.** "Adopting" might
not lemmatize correctly. Mitigation: lemmatization is a fall-back; if no lemma
found, keep the snake_cased surface form. Conservative.

**R5: Phase D.2's distilled head underperforms on out-of-distribution sentences.**
Mitigation: keep LLM extraction as opt-in `--extract-triples-llm-alias` flag;
distilled head is the fast default, LLM is the quality fallback.

**R6: Configuration explosion.** Six tunables (PHASE1_FALLBACK_CAP, recall_limit,
α, β, reranker_top_n, lemmatize-on/off). Mitigation: ship strong defaults from
the 10-conv sweep; document the meaningful axes in `UnikoConfig`.

---

## 8. Open questions

1. **PHASE1_FALLBACK_CAP after Phase C:** if session-aware reranking eliminates
   the need to merge Facts into the bundle, PHASE1_FALLBACK_CAP becomes 0 and
   the merge code can be deleted. Keep both code paths during the transition?

2. **Should `temporal_anchor` go on Chunks too?** Currently chunks include
   session text with embedded temporal references. Resolving them at chunk-creation
   time (rather than in the recall format step) would let chunks carry pre-resolved
   dates without ambiguity. Trade-off: bigger schema change, more rebuild needed.

3. **REBEL/T-REx still worth a shot?** They're trained on Wikipedia abstracts
   (formal, encyclopedic) and our use case is conversational (informal,
   first-person). Pretrained quality on our distribution is likely poor; we'd
   need fine-tuning anyway. Phase D.2 (head on existing encoder) is the cleaner
   path; this open question can probably be closed "no."

4. **Open-domain failure mode is partly LLM-prompt:** how much of the open-domain
   gap is recall-side (missing the right context) vs gen-side (refusing to
   infer when context is sufficient)? Failure analysis suggests 6/7 abstentions
   had evidence → mostly gen-side → prompt tweak should fix without touching
   recall.

5. **F39 drift detection threshold:** spec says "> 4 invalidations in 30 days."
   With per-Observation `temporal_anchor` and per-Fact BTIC, we can compute
   this directly. Default 4 / 30d or tunable?

---

## 9. References

**Codebase:**
- `crates/uniko-memory/src/recall/mod.rs` — Phase 1/3 recall pipeline
- `crates/uniko-memory/src/consolidation.rs` — P4 cycle
- `crates/uniko-extract/src/nlp/mod.rs` — SRL re-forward path
- `crates/uniko-extract/src/observations/rules_engine/matcher.rs` — DEP/SRL → triple
- `crates/uniko-extract/src/observations/temporal.rs` — relative→absolute resolver
- `crates/uniko-extract/src/observations/assets/english.yml` — rule definitions

**Committed work:**
- `3460b65` P4 Consolidation: derive Facts from Observations + activate recall Phase 1
- `e2d7e62` P4 Consolidation: optional LLM triple refinement
- `420524b` SRL plumbing (Phase A): per-verb re-forward + frame decoder
- `5f99f75` SRL-driven observation extraction (Phase B)

**Bench artifacts (conv-26, 199 questions, gpt-4o-mini gen + judge, MiniLM
rerank):**
- `data/locomo_conv26_p4_llmfacts_minilm_gpt4omini.json` — cap=0 → 0.717
- `data/locomo_conv26_p4_llmfacts_minilm_cap3_gpt4omini.json` — cap=3 → 0.750
- `data/locomo_conv26_p4_llmfacts_minilm_appendtail3_gpt4omini.json` — append → 0.724
- `data/locomo_conv26_p4_llmfacts_minilm_prepend3_gpt4omini.json` — prepend → 0.711
- `data/locomo_conv26_p4_llmfacts_minilm_limit30_gpt4omini.json` — limit=30 → 0.753

**Bench artifacts (conv-30):**
- `data/locomo_conv30_p4_llmfacts_minilm_gpt4omini.json` — cap=3 → 0.802
- `data/locomo_conv30_p4_llmfacts_minilm_nophase1_gpt4omini.json` — cap=0 → 0.815

**Spec items unlocked:**
- F38 (contradiction at 40% threshold) — needs Phase B negation extraction
- F39 (drift at 4 invalidations / 30 days) — needs Phase A temporal anchor

**Comparisons:**
- Mem0 published cross-10-conv overall: 0.669 (gpt-4o-mini gen + lenient
  gpt-4o-mini judge)
- Our prior conv-30 baseline (Gemma-4 gen + gpt-5-mini judge): 0.790
- Our best conv-30 (P4 + reranker, gpt-4o-mini gen + judge): 0.815

---

## 10. Glossary

- **Phase 1 (Compact)** — recall cascade phase that searches consolidated
  Semantic/Procedural tier (Facts, Procedures, Topics) via vector similarity.
  Returns early if coverage clears the 0.75 gate.
- **Phase 3 (Broaden)** — recall cascade phase that searches raw Chunks +
  Observations + Messages via hybrid (vector + BM25) + cross-encoder rerank.
- **PHASE1_FALLBACK_CAP** — max number of Phase 1 Facts merged into the
  Phase 3 bundle when the Phase 1 gate misses. Default 3.
- **cap=N** — shorthand for `PHASE1_FALLBACK_CAP=N`.
- **prepend / append** — alternative Phase 1 placement strategies tested in
  ablations; both underperform `cap=3 interleave by score`.
- **ARGM-TMP / ARGM-LOC** — PropBank semantic role labels for temporal and
  location arguments of a predicate. Emitted by the SRL head.
- **Session-rerank** — proposed v2 architecture where Facts/Observations boost
  the ranking of their containing session's Chunks rather than competing for
  bundle slots.
