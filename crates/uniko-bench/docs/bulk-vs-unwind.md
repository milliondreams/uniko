# Bulk API vs Cypher `UNWIND` — write-path benchmark & profile

**What:** how much the uni-db bulk write API (`Transaction::bulk_insert_vertices` /
`bulk_insert_edges`) saves over the equivalent Cypher `UNWIND … CREATE`, measured on the
*real* batch-size distribution that LoCoMo ingestion produces, plus a `PROFILE`-level
dissection of where the Cypher time goes.

**Environment:** conv-26 (419 turns, 19 sessions), single process, CPU, `uni-db 2.0.2`,
mimalloc, BGE-small (384-d) embedder, `reps=5`, median per batch.

**Tooling:** `bulk-vs-unwind` binary (`crates/uniko-bench/src/bulk_vs_unwind_main.rs`) +
the `batch-record` feature in `uniko-store` (`storage/batch_record.rs`).

---

## TL;DR

| condition | bulk (per op) | UNWIND (per op) | speedup | weighted total (whole conv) |
|---|---|---|---|---|
| **edges** | ~2.6 µs | ~1367 µs | **524×** | 11.4 ms vs 5954 ms |
| **nodes — no embed** | ~2.5 µs | ~123 µs | **49.6×** | 3.4 ms vs 167 ms |
| **nodes — with embed** | ~5074 µs | ~6930 µs | **1.4×** | 6860 ms vs 9370 ms |

- The bulk API removes **three fixed per-statement costs** that `UNWIND` pays: Cypher
  **parse** (~82 µs/statement), **plan**, and the generic **per-row executor** machinery.
- **Edges** show the biggest gap (500×) because the Cypher path must additionally
  **re-resolve both endpoints** from integer ids (two `GraphScan → Filter → HashJoin`
  legs per edge) — work the bulk path skips entirely since it already holds the VIDs.
- **With auto-embedding on, the gap collapses to ~1.4×** for nodes: embedding runs on
  *both* arms (it is an index-level on-write hook) and dominates wall time, so the
  executor delta is in the noise. The tell that this is purely an embedding artifact:
  `Entity` (which carries a *precomputed* vector, no on-write embed) still shows **57×**
  in the with-embed run, while auto-embed `Chunk`/`Observation` sit at 1.1–1.7×.

**Bottom line:** the bulk path is strongly justified for **edges always**, and for
**nodes whenever embedding is not the bottleneck**. When a node label auto-embeds, the
write path is irrelevant next to the embed cost.

---

## 1. Background: two write paths

Uniko's ingest hot paths write through uni-db's **bulk API** rather than Cypher. The
single chokepoints live in `crates/uniko-store/src/storage/batch.rs` and
`edges.rs`. The bulk signatures (uni-db public API):

```rust
// Vertices: one label, N property maps → N new VIDs.
async fn bulk_insert_vertices(
    &self, label: &str, properties_list: Vec<HashMap<String, Value>>,
) -> Result<Vec<Vid>>;

// Edges: one type, N (src, dst, props) tuples. Endpoints addressed by VID.
async fn bulk_insert_edges(
    &self, edge_type: &str, edges: Vec<(Vid, Vid, HashMap<String, Value>)>,
) -> Result<()>;
```

The bulk path skips the Cypher pipeline (parse → plan → per-row executor) and writes
straight into the transaction's L0 overlay. The `batch.rs` comment claims ~150 µs/edge
for bulk vs ~147 ms/edge for the Cypher path in one synthetic config (~980×). This
benchmark checks that against real batches.

### 1.1 Node create — bulk vs UNWIND

**Bulk** (`batch_create_nodes_in_tx`, `batch.rs`):

```rust
let properties_list: Vec<HashMap<String, Value>> = items.to_vec();
let vids = tx.bulk_insert_vertices(label, properties_list).await?;
```

**Cypher equivalent** (the baseline this bench reconstructs — inline-create so
non-nullable columns are satisfied at creation):

```cypher
UNWIND $rows AS r
CREATE (n:Chunk {chunk_id: r.chunk_id, text: r.text, index: r.index, …})
```

### 1.2 Edge create — bulk vs UNWIND

**Bulk** (`batch_create_edges_inner_in_tx` fast arm, `batch.rs`):

```rust
let bulk_edges: Vec<(Vid, Vid, HashMap<String, Value>)> = edges
    .iter()
    .map(|(s, d, props)| (Vid::new(*s as u64), Vid::new(*d as u64), props.clone()))
    .collect();
tx.bulk_insert_edges(edge_type, bulk_edges).await?;
```

**Cypher equivalent** (the production path still used when edge IDs are needed — the
`return_ids = true` arm in `batch.rs:303`):

```cypher
UNWIND $edges AS e
MATCH (a) WHERE id(a) = e.src
MATCH (b) WHERE id(b) = e.dst
CREATE (a)-[r:SENT_BY]->(b) SET r.role = e.role
RETURN id(r) AS eid
```

The two `MATCH … WHERE id(x) = …` clauses are the crux: Cypher only has the integer ids
as parameters, so it must re-find each endpoint row in the store. Bulk already has the
`Vid`s in hand.

---

## 2. Schema under test

Two schema conditions are measured for nodes. They differ only in vector indexing, which
is what makes auto-embed fire.

### 2.1 No-embed (isolates executor overhead)

The bench infers a plain schema from the recorded rows — every property nullable, types
mapped from the captured `Value` variants, **no vector index**:

```rust
// per recorded label, e.g. inferred for Chunk:
db.schema()
    .label("Chunk")
    .property_nullable("chunk_id", DataType::String)
    .property_nullable("text", DataType::String)
    .property_nullable("index", DataType::Int64)
    .property_nullable("embedding", DataType::Vector { dimensions: 384 }) // stored, NOT indexed
    // …
    .apply().await?;
```

A stored-but-unindexed vector costs nothing on write (no HNSW maintenance, no embed
call), so this condition measures the pure parse/plan/executor delta.

### 2.2 With-embed (the real uniko schema)

The real schema (`crates/uniko-store/src/schema/`) puts a vector index on several node
labels. Two flavours matter:

**Auto-embed** — `Chunk` embeds its `text` column on every write (`schema/chunks.rs`):

```rust
builder
    .label(labels::CHUNK)
    .property("text", DataType::String)
    .property_nullable("embedding", DataType::Vector { dimensions: config.embedding.dimensions })
    .index("text", IndexType::FullText)
    .index("embedding", IndexType::Vector(super::auto_embed_vector_index("text", config)))
    //                                     ^^^^^^^^^^^^^^^^^^^ source = "text" → embed on write
    .done()
```

`Observation` is the same shape (auto-embeds `content`). On any write — bulk **or**
Cypher `CREATE` — uni-db concatenates the source column(s) and runs the embedder. It is
an index-level on-write hook, so **both arms pay it equally**.

**Manual-embed** — `Entity` has a vector index but **no source** (`schema/entities.rs`);
the application computes the embedding and passes it in the row:

```rust
builder
    .label(labels::ENTITY)
    .property("entity_id", DataType::String)
    .property("name", DataType::String)
    .property_nullable("embedding", DataType::Vector { dimensions: config.embedding.dimensions })
    .index("entity_id", IndexType::Scalar(ScalarType::Hash))
    .index("embedding", IndexType::Vector(super::vector_index(config))) // no `source` → no on-write embed
    .done()
```

This distinction is why `Entity` behaves like a no-embed node even in the with-embed run:
the vector is already in the row, so neither arm calls the embedder.

### 2.3 Edge schema (single condition)

Edges carry no vector index, so the no-/with-embed split is irrelevant to them — they are
measured once. The bench declares a generic `Ep → Ep` schema and synthesizes endpoints
(per-edge write cost is independent of *which* nodes are joined):

```rust
db.schema().label("Ep").property("id", DataType::Int64).apply().await?;
db.schema().edge_type("SENT_BY", &["Ep"], &["Ep"]).property_nullable("role", DataType::String).apply().await?;
```

Real edge types created during ingest: `SENT_BY`, `IN_SESSION`, `ADDRESSED_TO`, `NEXT`
(per message), `HAS_CHUNK`, `MENTIONS`, `OBSERVED_IN`, `ABOUT`.

---

## 3. Methodology: record → replay

The benchmark cannot fairly invent batch sizes, so it captures the real ones.

1. **Record.** A feature-gated, in-process recorder (`storage::batch_record`, compiled to
   no-op without the `batch-record` feature) captures every batch handed to the bulk API
   during one live ingest:

   ```rust
   // hook at each bulk-write site, e.g. batch_create_nodes_in_tx:
   super::batch_record::record_node_batch(label, || items.to_vec());
   ```

   Batches stay as native `uni_db::Value` (so `Vector`/`Temporal` types round-trip
   exactly — a JSON dump would corrupt them, because `Value` is `#[serde(untagged)]`).

2. **Replay.** For each captured batch, run both arms — `bulk_insert_*` and the
   hand-built `UNWIND` — **timing only the write call** and **rolling back** (no commit →
   no flush/compaction skew). Median of `reps`, arm order alternated per rep to cancel
   warmup bias. The clone of the payload happens *outside* the timed region for both arms:

   ```rust
   let payload = rows.to_vec();                 // outside the timer
   let t = Instant::now();
   tx.bulk_insert_vertices(label, payload).await?;
   let bulk_us = t.elapsed().as_micros();
   tx.rollback();
   ```

3. **Report.** Per (condition, label/type), bucketed by batch size, plus a
   frequency-weighted total over the whole conversation.

### Fairness notes
- Time the write only, not commit; rollback avoids flush/compaction skew.
- Auto-embed fires on both arms (verified: it is an index on-write hook) → the with-embed
  comparison is apples-to-apples; embedding is shared overhead, not a bulk-only cost.
- Edge endpoints are synthesized in a tiny `Ep` pool. Real endpoint labels would make the
  Cypher `GraphScan` legs *more* expensive, not less — so the edge gap is conservative.
- Consolidation **Fact** batches are excluded (see §7).

---

## 4. The real batch distribution (conv-26)

2908 batches captured. Batches are small — mostly size 1, rarely above 16.

```
── node batches ──            ── edge batches ──
label         batches  rows   edge_type     batches  edges  avg
Chunk             115   197    ABOUT             270   1205  4.5
Entity             95   133    ADDRESSED_TO      419    419  1.0
Observation       237  1022    HAS_CHUNK         115    197  1.7
                                IN_SESSION        419    419  1.0
                                MENTIONS          183    276  1.5
                                NEXT              400    400  1.0
                                OBSERVED_IN       237   1022  4.3
                                SENT_BY           419    419  1.0
```

This matters: the bulk advantage is largest at small batches, and small batches are what
actually run. Of 2461 edge batches, **1940 are size 1**.

---

## 5. Results — timing (median over 5 reps)

### 5.1 Nodes — no embed (isolates executor overhead)

```
key            batches  rows   bulk_us/op  unwd_us/op  speedup
Chunk              115   197         3.48      208.48    59.9x
Entity              95   133         3.96      238.27    60.1x
Observation        237  1022         2.10       91.95    43.7x
TOTAL              447  1352         2.49      123.33    49.6x
```

### 5.2 Nodes — with embed (real schema; auto-embed on both arms)

```
key            batches  rows   bulk_us/op  unwd_us/op  speedup
Chunk              115   197     20235.07    22346.84     1.1x   ← auto-embed dominates
Entity              95   133         3.71      210.99    56.9x   ← precomputed vector, no on-write embed
Observation        237  1022      2811.64     4832.97     1.7x   ← auto-embed dominates
TOTAL              447  1352      5074.19     6930.24     1.4x
```

The `Chunk`/`Observation` per-op cost jumps from ~2–3 µs to ~3000–20000 µs — that is the
BGE-small embedder running on CPU, identically on both arms. `Entity` is unchanged from
the no-embed run, confirming the effect is purely embedding.

### 5.3 Edges (generic `Ep → Ep`, no vector index)

```
key            batches  rows   bulk_us/op  unwd_us/op  speedup
ABOUT              270  1205         1.39      561.94   405.5x
ADDRESSED_TO       419   419         4.02     2325.25   578.9x
HAS_CHUNK          115   197         2.81     1468.73   522.3x
IN_SESSION         419   419         3.97     2319.19   583.6x
MENTIONS           183   276         2.96     1621.87   547.2x
NEXT               400   400         4.40     2462.55   560.0x
OBSERVED_IN        237  1022         1.41      578.34   410.7x
SENT_BY            419   419         4.23     2430.12   574.9x
TOTAL             2462  4357         2.61     1366.64   524.2x
```

### 5.4 Speedup vs batch size

Speedup shrinks as batches grow, because `UNWIND`'s fixed per-statement parse/plan cost
amortizes over more rows. It never gets small in the regime that runs (size 1–16):

```
size    nodes(no-embed)   edges
1            64.2x        574.2x
2-4          51.6x        481.6x
5-8          40.0x        352.8x
9-16         31.0x        264.9x
```

---

## 6. Profile — where the `UNWIND` time goes

Run with `--profile`: each statement run once and rolled back; parse/plan/exec from
`.metrics()`, per-operator times from `PROFILE`.

### 6.1 Parse / plan / exec split (per batch, µs)

**Nodes — `UNWIND $rows AS r CREATE (n:Label {…})`**

```
size   batches  parse_us  plan_us  exec_us  total_us  exec%
1          194      83.8      5.9    165.9     257.6    64%
2-4        157      83.3      6.6    210.4     302.2    70%
5-8         69      82.4      7.3    263.8     355.4    74%
9-16        27      89.9      9.1    400.4     501.3    80%
ALL        447      83.8      6.6    210.8     303.1    70%
```

**Edges — `UNWIND $edges AS e MATCH … MATCH … CREATE … SET … RETURN`**

```
size   batches  parse_us  plan_us  exec_us  total_us  exec%
1         1940      82.8     18.4    488.8     592.8    82%
2-4        318      80.3     18.3    526.5     628.0    84%
5-8        142      80.1     18.7    581.8     683.8    85%
9-16        61      77.8     18.8    693.0     793.0    87%
ALL       2461      82.2     18.4    504.1     607.5    83%
```

Observations:
- **Parse is a flat ~82 µs/statement** for both, independent of batch size (string → AST).
- **Plan is small** (~7 µs nodes, ~18 µs edges) and never cached (`plan_cache_hit = 0%` —
  each batch builds a fresh query string).
- **Exec is the bulk of it and scales with rows** — exactly the per-row cost the bulk path
  collapses. `exec%` rising with batch size confirms parse/plan are fixed overheads.

### 6.2 Per-operator profile (batch = 1)

**Node** `CREATE` — two operators after the unwind, no endpoint work:

```
operator              rows   time_ms
GraphUnwindExec          1    0.0067
MutationCreateExec       1    0.0445   ← the whole node cost
```

**Edge** `MATCH … MATCH … CREATE … SET` — the same create machinery **plus** full
endpoint re-resolution:

```
operator              rows   time_ms
GraphUnwindExec          1    0.0047
GraphScanExec            1    0.0178   ┐
FilterExec               1    0.0045   │ MATCH (a) WHERE id(a)=e.src
ProjectionExec           1    0.0030   │
HashJoinExec             1    0.0341   ┘
GraphScanExec            1    0.0114   ┐
FilterExec               1    0.0029   │ MATCH (b) WHERE id(b)=e.dst
ProjectionExec           1    0.0023   │
HashJoinExec             1    0.0979   ┘
MutationCreateExec       1    0.1807
MutationSetExec          1    0.2276   ← SET r.role
ProjectionExec           1    0.0005   ← RETURN id(r)
```

(The id-equality `MATCH` lowers to `GraphScan → Filter → HashJoin` — the HashJoin rewrite,
uni-db #53/#54 — so it is a join rather than a cross-product, but it is still two scans +
two joins per edge.)

---

## 7. Why nodes are 50× and edges are 500×

Both arms ultimately write to L0. The bulk path does *only* that. The `UNWIND` path adds:

| cost | nodes | edges | does bulk pay it? |
|---|---|---|---|
| Cypher parse (~82 µs) | ✔ | ✔ | ✗ |
| Cypher plan (~7 / ~18 µs) | ✔ | ✔ | ✗ |
| generic `MutationCreateExec` | ✔ | ✔ | ✗ (writes L0 directly) |
| `MutationSetExec` (edge props) | — | ✔ | ✗ |
| endpoint re-resolution: 2× `GraphScan+Filter+HashJoin` | — | ✔ | ✗ (VIDs already known) |

- **Nodes (~250 µs → ~2.5 µs ≈ 50×):** bulk drops parse + plan + the create-executor
  wrapper. That's the whole difference.
- **Edges (~500 µs → ~2.6 µs ≈ 500×):** edges pay everything nodes pay **plus** ~30% of
  the cost just re-finding both endpoints (two scans + two hash joins) and a SET operator
  — none of which exists in the bulk path, because the VIDs are already in hand. More
  removed work ⇒ bigger gap.
- **With-embed (~1.4×):** when the label auto-embeds, the embedder call (thousands of µs
  on CPU) runs on both arms and swamps the ~250 µs executor delta. The write-path choice
  stops mattering; the embedder is the bottleneck.

---

## 8. Caveats & limitations

- **Consolidation Fact batches excluded.** `run_post_ingest_sweep` (which creates the
  large `Fact` batches via consolidation) deadlocked in uni-db on CPU
  (`futex_do_wait`, ~4% CPU, 20+ min silent) — ingest itself is fine (~53 s). Worked
  around with `--no-sweep`, so the **large-batch tail is not measured**. Since bulk's
  *relative* win is smaller at large batches but the *absolute* time saved grows with
  volume, the real-world benefit is at least what's shown here. Root-causing that
  deadlock is unfinished; by inspection the recorder mutex is never held across an
  `.await`, so it is unlikely the cause, but this was not empirically isolated.
- **CPU only.** Embedding (the with-embed condition) and NLP run on CPU; GPU would shrink
  the with-embed wall time but not the no-embed / edge executor gaps.
- **Synthesized edge endpoints** in a tiny `Ep` pool — real endpoint labels would make the
  Cypher scan legs costlier, so the edge gap is conservative.
- **`PROFILE` metrics quirk:** under `.profile()` the `ExecuteResult.metrics()` returned
  all-zero, so the parse/plan/exec split (§6.1) comes from the `.run()` pass and the
  operator times (§6.2) from the `.profile()` pass — two passes, both valid.
- Median of 5 reps; first-rep warmup folded in via alternation.

---

## 9. Reproduce

```bash
# build (CPU)
cargo build -p uniko-bench --bin bulk-vs-unwind --no-default-features --release

# timing tables (nodes no-embed / with-embed, edges, size curves, weighted totals)
./target/release/bulk-vs-unwind \
  --data data/locomo10.json --conversations conv-26 \
  --bench-config crates/uniko-bench/bench-configs/locomo-bge-openai.json \
  --reps 5 --no-sweep --ingest-dir /tmp/bvu_kb

# profile dissection (parse/plan/exec + per-operator), edges + nodes-no-embed
./target/release/bulk-vs-unwind \
  --data data/locomo10.json --conversations conv-26 \
  --bench-config crates/uniko-bench/bench-configs/locomo-bge-openai.json \
  --no-sweep --profile --ingest-dir /tmp/bvu_kb
```

The recorder is gated behind the `batch-record` feature, which `uniko-bench` enables on
its `uniko-store` dependency. It compiles to a no-op (zero cost, no global state) in any
build that does not enable the feature.

---

## 10. Recommendations

1. **Keep the bulk API for all edge writes.** The 500× gap is structural (parse + plan +
   endpoint re-resolution + per-row executor), not a tuning artifact, and it holds across
   every edge type and batch size that ingest produces.
2. **Keep the bulk API for node writes**, but recognize the win is ~50× only when the
   label is *not* auto-embedding. For auto-embed labels (`Chunk`, `Observation`,
   `Message`, `Summary`), the embedder dominates and the write path is a rounding error —
   so optimization effort there belongs on embedding (batching, GPU, caching), not the
   write call.
3. If a Cypher `UNWIND` write path is ever reintroduced (e.g. when edge IDs are needed),
   note that its plan is never cached (`plan_cache_hit = 0%`); parameterizing statement
   shapes could reclaim the ~2–3% plan cost, but it would not touch the dominant
   parse + executor costs.
```
