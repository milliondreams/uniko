# Retrieval Analysis — LoCoMo conv-30

## Benchmark Progression

| Run | Overall | Single-hop | Multi-hop | Temporal | Adversarial | Notes |
|-----|---------|------------|-----------|----------|-------------|-------|
| Per-obs embed (broken) | 13.0% | 14.3% | 3.2% | 23.1% | 12.0% | 964 obs flooding top-15 |
| Session obs chunks | 59.5% | 65.9% | 25.8% | 76.9% | 75.0% | Obs trace-only, chunks searchable |
| + Sentence splitting | 61.1% | 65.3% | 25.8% | 76.9% | 80.0% | Per-sentence CLS gate |
| + Session DateTime fix | 67.2% | 71.4% | 29.0% | 88.5% | 84.0% | Ghost Session nodes fixed |
| + Participant.name fix | 59.5% | 57.1% | 29.0% | 84.6% | 76.0% | REGRESSION — see below |

**Best run: 67.2% overall, 71.4% single-hop** (Session DateTime fix, before Participant.name).

## Root Causes of Remaining Single-Hop Misses (11/44)

### Investigation Method

Used `recall_debug.rs` to dump ranked retrieval results for each missed
question, showing node type, score, and content preview.

### Finding 1: Topic Saturation

This is a conversation about two people (Jon and Gina) starting businesses
(dance studio and clothing store). Nearly every message and chunk contains
the words "dance", "studio", "business", "store", "Jon", "Gina".

When a query like "What is Jon's favorite style of dance?" is issued:
- BM25 matches on "dance" in 50+ nodes → IDF near zero, no discrimination
- Vector embeddings of "dance studio" messages are all close together
- The specific evidence message (D1:8: "contemporary is my top pick")
  scores 0.303, while irrelevant messages score 0.275-0.295
- Score spread across 15 results is only ~10%, too tight to rank correctly

### Finding 2: Paraphrase Gap

The query uses different words than the evidence:

| Query term | Evidence term | Gap type |
|-----------|--------------|----------|
| "destress" | "stress relief", "passion and escape" | Synonym |
| "favorite style" | "top pick" | Paraphrase |
| "won't do" | "not giving up" | Negation paraphrase |
| "make him" → "happy" | "makes me so happy" | Pronoun + structure |
| "design for her store" | "space I designed" | Noun vs verb form |
| "compare...journeys to" | "like having a partner to dance with" | Metaphor |

BM25 cannot bridge these gaps. Vector similarity handles some (synonym)
but not others (negation, metaphor).

### Finding 3: Entity-Scoped Search Dominance

Entity-scoped search (Messages matching SENT_BY/ADDRESSED_TO for
Jon or Gina) produces higher scores than broad search because the
candidate set is smaller → higher relative BM25/vector scores.

In a 2-person conversation, every message matches one entity, so
entity scoping doesn't narrow the search — it just inflates scores
uniformly. This pushes Messages above Chunks in ranking.

**Evidence**: All 15 results for most queries are Messages. Chunks
appear only when the broad search score exceeds entity-scoped scores.

### Finding 4: Session 1 Concentration

7 of 11 misses need evidence from session 1 (D1:2 through D1:26).
Session 1 has 28 turns spanning many topics. The session chunk for
session 1 is 2137 chars covering everything from job loss to dance
styles to festivals. The evidence for any single question is a small
fraction of the chunk, diluted by unrelated content.

### Finding 5: Observation Chunk Quality

Observation chunks contain DEP-reconstructed fragments:
```
Jon Lost my job as a banker
Jon gonna so take a shot at starting my own business
Jon starting my own business
```

These are degraded versions of the original text. They lose articles,
sentence structure, and natural phrasing. They compete with session
chunks (which have the full text) for top-15 slots without adding
distinct signal.

### Finding 6: Participant.name Regression

Setting `Participant.name = participant_id` caused a 7.7% overall
regression because:

1. Entity-scoped search matches `Participant {name: $ename}` on the
   name property. Previously name was NULL → no matches → entity scoping
   was effectively disabled.
2. With name set, entity scoping activates for every query mentioning
   "Jon" or "Gina". In a 2-person conversation, this matches ALL
   messages, inflating all Message scores and pushing Chunks out.
3. The entity-scoped boost is non-discriminating — it adds ~0.05 to
   every Message score equally.

**Fix options**:
- Don't use entity-scoped search when the matched entity is a Participant
  in the conversation (everyone matches)
- Only entity-scope on rare entities (not the main speakers)
- Weight entity-scoped results lower for high-frequency entities

## Recommendations (Priority Order)

### 1. Revert Participant.name regression
Revert commit 583c0bc or adjust entity-scoped search to skip
conversation participants. Keep the name property but don't use it
for entity scoping on Participants — only on Entities.

### 2. Increase top-k from 15 to 25-30
The evidence messages score 0.27-0.30, just below the top-15 cutoff
in many cases. Increasing to 25 would capture them. The LLM context
window can handle 25 items.

### 3. Reduce session chunk size from 400-512 to 200-256 tokens
Session 1 has one chunk covering 15 turns. Smaller chunks would give
more precise matching — the "contemporary is my top pick" turn would
be in a chunk with 3-4 related turns instead of 15.

### 4. Query expansion / rewriting
Transform queries before search:
- "destress" → "destress stress relief relax"
- "favorite style" → "favorite style top pick preferred"
- "won't do" → "won't do not giving up refuse quit"

Can be done with the NLP model (extract keywords) or a simple
synonym table.

### 5. Disable observation chunks (evaluate impact)
Observation chunks may be hurting more than helping — they add
garbled fragments that compete with session chunks. Run a benchmark
with observation chunks disabled to measure impact.

### 6. Cross-encoder re-ranking
Retrieve top-50, re-rank with a cross-encoder model (e.g.,
cross-encoder/ms-marco-MiniLM-L-6-v2). This catches paraphrase
gaps that BM25 and bi-encoder miss. Adds ~50 ONNX calls per query
at ~5ms each = ~250ms.

## Graph Structure Issues Found

### Session.started_at type mismatch (FIXED)
`get_or_create_session()` stored `started_at` as `Value::String`
but the schema declared it as `DataType::DateTime`. This caused
Session nodes to become "ghost nodes" — edges pointed to them but
`MATCH (n:Session)` couldn't find them. All chunk and observation
aggregation silently failed.

**Fix**: Use `Value::Temporal(DateTime)` in session creation.

### Participant.name not set (FIXED but regressed)
`ensure_participant()` never set the `name` property. The session
chunk query `p.name AS speaker` returned NULL → "unknown" prefix
in all chunks. Observation `load_sender_ref()` also returned None.

**Fix**: Set `name = participant_id` in merge_node. But this caused
entity-scoped search regression (see Finding 6).

### HAS_CHUNK edges missing (FIXED with Session fix)
Before the Session DateTime fix, `chunk_session()` couldn't find
Session nodes by `session_id` → no HAS_CHUNK edges created → Chunks
were orphaned and unreachable by graph traversal.

### Auto-embed insert latency (uni-db#43)
Insert latency jumps 25x after ~150 nodes with auto-embed. Step
function pattern, not gradual. Reproduced in isolated test. Filed
as rustic-ai/uni-db#43.

## Test Infrastructure Created

| Test file | Purpose |
|-----------|---------|
| `nlp_observation_integration_test.rs` | End-to-end NLP → observation quality (5 tests, 13 cases) |
| `ner_quality_test.rs` | NER accuracy on LoCoMo messages |
| `obs_for_misses_test.rs` | Observations produced for missed evidence messages |
| `unidb_insert_latency.rs` | Auto-embed latency regression repro |
| `graph_debug.rs` | Graph structure diagnostics (node counts, edge existence) |
| `recall_debug.rs` | Retrieval ranking diagnostics (BM25, vector, hybrid scores) |
| `single_hop_debug.rs` | Ranked results for specific missed questions |
