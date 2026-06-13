//! Benchmark: uni-db bulk write API vs Cypher `UNWIND`, on real batches.
//!
//! Several ingest hot paths write through uni-db's bulk API
//! (`bulk_insert_vertices` / `bulk_insert_edges`) instead of Cypher.
//! This binary measures how much that actually saves versus an
//! equivalent Cypher `UNWIND … CREATE`, on the *real* batch-size
//! distribution that LoCoMo ingestion produces (rather than a synthetic
//! microbench at a fixed batch size).
//!
//! ## Method
//!
//! 1. **Record.** Enable [`uniko_store::enable_batch_recording`], ingest
//!    one LoCoMo conversation (plus the post-ingest consolidation sweep,
//!    which produces the large Fact batches), and drain every node/edge
//!    batch handed to the bulk API.
//! 2. **Replay.** For each captured batch, run both the bulk call and a
//!    hand-built `UNWIND` query, timing only the write call and rolling
//!    back (no mutation, no flush). Median of `--reps`, arm order
//!    alternated per rep to cancel warmup bias.
//! 3. **Report.** Per (condition, label/type), bucketed by batch size,
//!    plus a frequency-weighted total across the whole conversation.
//!
//! ## Conditions
//!
//! * **Nodes / no-embed** — a plain schema (types inferred from the
//!   recorded rows, no vector index). Isolates the Cypher parse + plan +
//!   per-row executor overhead that the bulk path removes.
//! * **Nodes / with-embed** — the real uniko schema, replayed against
//!   the populated ingest KB. Auto-embed fires on *both* arms (it is an
//!   index-level on-write hook), so this measures the realistic
//!   end-to-end per-batch cost where embedding is shared overhead.
//! * **Edges** — measured once under a generic `Ep→Ep` schema: edges
//!   carry no vector index, so the no-/with-embed split is irrelevant
//!   to them.
//!
//! Edge endpoints are synthesized (per-edge write cost is independent of
//! which nodes are joined), keeping both arms symmetric.
//!
//! Build & run:
//!   cargo build -p uniko-bench --bin bulk-vs-unwind --release
//!   ./target/release/bulk-vs-unwind \
//!       --data data/locomo10.json --conversations conv-26 \
//!       --bench-config crates/uniko-bench/bench-configs/<cfg>.json --reps 5

// The async `main` chains many awaits (ingest + replay); the nested
// future type exceeds the default type-layout recursion depth.
#![recursion_limit = "512"]

// mimalloc as global allocator — the bulk/Cypher write paths are
// mutation-heavy, where glibc's allocator is the dominant cost
// (M-MIMALLOC-APPS; measured ~3x on uni-db's concurrent_mutations).
#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use uni_db::common::TemporalValue;
use uni_db::{DataType, Uni, Value, Vid};
use uniko_bench::bench_config::BenchConfig;
use uniko_bench::{data, ingest};
use uniko_memory::consolidation::TripleSource;
use uniko_store::RecordedBatch;
use uniko_store::config::UnikoConfig;

#[derive(Parser, Debug)]
#[command(about = "Benchmark uni-db bulk write API vs Cypher UNWIND on real LoCoMo batches.")]
struct Cli {
    /// LoCoMo dataset JSON (e.g. data/locomo10.json).
    #[arg(long)]
    data: PathBuf,

    /// Bench config JSON — supplies the embedder/model catalog used for
    /// ingest and the with-embed substrate.
    #[arg(long)]
    bench_config: PathBuf,

    /// Conversation id to ingest and replay (first match wins).
    #[arg(long, default_value = "conv-26")]
    conversations: String,

    /// KB storage directory for the recording ingest.
    #[arg(long, default_value = "data/bulk_vs_unwind_kb")]
    ingest_dir: PathBuf,

    /// Timed repetitions per batch per arm (median reported).
    #[arg(long, default_value = "5")]
    reps: usize,

    /// Skip the post-ingest consolidation sweep (which otherwise
    /// captures the large Fact node batches).
    #[arg(long)]
    no_sweep: bool,

    /// Profile the UNWIND path instead of timing bulk-vs-UNWIND: print
    /// the parse/plan/exec split and the per-operator profile for edges
    /// and nodes (no-embed). Skips the with-embed condition.
    #[arg(long)]
    profile: bool,

    /// Optional catalog path override (forwarded to UnikoConfig).
    #[arg(long)]
    catalog: Option<PathBuf>,

    /// Optional schema path override (forwarded to UnikoConfig).
    #[arg(long)]
    schema: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    let cli = Cli::parse();

    // ── Phase A: record real batches during a live ingest ──
    let bench_cfg = BenchConfig::load(&cli.bench_config)
        .with_context(|| format!("loading bench config {}", cli.bench_config.display()))?;
    let mut config = UnikoConfig {
        catalog_path: cli.catalog.clone(),
        schema_path: cli.schema.clone(),
        ..Default::default()
    };
    bench_cfg.apply_to_uniko_config(&mut config)?;
    let extra_catalog = bench_cfg.build_catalog_specs();

    let samples = data::load_locomo(&cli.data)?;
    let sample = samples
        .into_iter()
        .find(|s| s.sample_id == cli.conversations)
        .ok_or_else(|| {
            anyhow!(
                "conversation {} not found in {}",
                cli.conversations,
                cli.data.display()
            )
        })?;
    let sessions = data::parse_sessions(&sample.sample_id, &sample.conversation)
        .with_context(|| format!("parsing sessions for {}", sample.sample_id))?;
    let total_turns: usize = sessions.iter().map(|s| s.turns.len()).sum();
    println!(
        "ingesting {} ({} sessions, {} turns) with batch recording on …",
        sample.sample_id,
        sessions.len(),
        total_turns
    );

    let kb_dir = cli.ingest_dir.join(&sample.sample_id);
    // Fresh KB each run — recording needs the live ingest, so we cannot
    // reuse a pre-built KB (the batches live in-process, not on disk).
    let _ = std::fs::remove_dir_all(&kb_dir);

    uniko_store::enable_batch_recording();
    let ingest_start = Instant::now();
    let kb =
        ingest::ingest_conversation(&sample, &sessions, &kb_dir, config.clone(), &extra_catalog)
            .await
            .context("ingest (recording)")?;
    if !cli.no_sweep {
        uniko_bench::run_post_ingest_sweep(&kb, &sample.sample_id, &TripleSource::SrlDep).await;
    }
    let batches = uniko_store::take_recorded_batches();
    println!(
        "ingest done in {:.1}s — captured {} batches",
        ingest_start.elapsed().as_secs_f64(),
        batches.len()
    );
    print_recorded_summary(&batches);

    // ── Phase B: build replay substrates ──
    // No-embed: plain schema inferred from the recorded node rows, plus
    // a generic Ep→Ep schema for every recorded edge type.
    let node_batches: Vec<&RecordedBatch> = batches
        .iter()
        .filter(|b| matches!(b, RecordedBatch::Node { .. }) && !b.is_empty())
        .collect();
    let edge_batches: Vec<&RecordedBatch> = batches
        .iter()
        .filter(|b| matches!(b, RecordedBatch::Edge { .. }) && !b.is_empty())
        .collect();

    let noembed = Uni::in_memory()
        .build()
        .await
        .map_err(|e| anyhow!("noembed Uni::in_memory: {e}"))?;
    build_noembed_node_schema(&noembed, &node_batches).await?;
    let pool = build_edge_substrate(&noembed, &edge_batches).await?;

    // ── Profile mode: dissect the UNWIND path, then exit ──
    if cli.profile {
        run_profile(&noembed, &node_batches, &edge_batches, &pool).await?;
        let _ = noembed.shutdown().await;
        if let Some(kb_owned) = std::sync::Arc::into_inner(kb) {
            let _ = kb_owned.shutdown().await;
        }
        return Ok(());
    }

    println!(
        "\nreplaying {} reps/arm, rolling back each write …\n",
        cli.reps
    );

    // ── Replay: nodes (no-embed + with-embed) and edges ──
    let mut node_noembed: Vec<Measured> = Vec::new();
    let mut node_embed: Vec<Measured> = Vec::new();
    for b in &node_batches {
        let RecordedBatch::Node { label, rows, .. } = b else {
            continue;
        };
        match replay_node(&noembed, label, rows, cli.reps).await {
            Ok((bulk, unwind)) => node_noembed.push(Measured::new(label, rows.len(), bulk, unwind)),
            Err(e) => tracing::warn!(label, error = %e, "no-embed node replay failed; skipping"),
        }
        match replay_node(kb.db(), label, rows, cli.reps).await {
            Ok((bulk, unwind)) => node_embed.push(Measured::new(label, rows.len(), bulk, unwind)),
            Err(e) => tracing::warn!(label, error = %e, "with-embed node replay failed; skipping"),
        }
    }

    let mut edge_rows: Vec<Measured> = Vec::new();
    for b in &edge_batches {
        let RecordedBatch::Edge {
            edge_type, props, ..
        } = b
        else {
            continue;
        };
        match replay_edge(&noembed, edge_type, props, &pool, cli.reps).await {
            Ok((bulk, unwind)) => {
                edge_rows.push(Measured::new(edge_type, props.len(), bulk, unwind))
            }
            Err(e) => tracing::warn!(edge_type, error = %e, "edge replay failed; skipping"),
        }
    }

    // ── Report ──
    print_table(
        "NODES — no-embed (plain schema; isolates executor overhead)",
        &node_noembed,
    );
    print_table(
        "NODES — with-embed (real schema; auto-embed on both arms)",
        &node_embed,
    );
    print_table("EDGES — generic Ep→Ep schema (no vector index)", &edge_rows);
    print_size_curve("BATCH-SIZE CURVE — nodes (no-embed)", &node_noembed);
    print_size_curve("BATCH-SIZE CURVE — edges", &edge_rows);
    print_weighted_totals(&node_noembed, &node_embed, &edge_rows);

    if let Some(kb_owned) = std::sync::Arc::into_inner(kb) {
        let _ = kb_owned.shutdown().await;
    }
    let _ = noembed.shutdown().await;
    Ok(())
}

/// One replayed batch's median timings for both arms.
struct Measured {
    key: String,
    size: usize,
    bulk_us: f64,
    unwind_us: f64,
}

impl Measured {
    fn new(key: &str, size: usize, bulk_us: f64, unwind_us: f64) -> Self {
        Self {
            key: key.to_string(),
            size,
            bulk_us,
            unwind_us,
        }
    }
}

// ── Replay ──────────────────────────────────────────────────────────

/// Replay one node batch through bulk vs `UNWIND … CREATE`, returning
/// `(median_bulk_us, median_unwind_us)` over `reps`.
async fn replay_node(
    db: &Uni,
    label: &str,
    rows: &[HashMap<String, Value>],
    reps: usize,
) -> Result<(f64, f64)> {
    let keys = union_keys(rows);
    let create_q = build_node_create(label, &keys);
    let row_values = node_unwind_param(rows);
    let session = db.session();
    let mut bulk = Vec::with_capacity(reps);
    let mut unwind = Vec::with_capacity(reps);
    for rep in 0..reps {
        for is_bulk in arm_order(rep) {
            let tx = session.tx().await.map_err(|e| anyhow!("session.tx: {e}"))?;
            let us = if is_bulk {
                let payload = rows.to_vec(); // clone outside the timed region
                let t = Instant::now();
                tx.bulk_insert_vertices(label, payload)
                    .await
                    .map_err(|e| anyhow!("bulk_insert_vertices: {e}"))?;
                t.elapsed().as_micros()
            } else {
                let param = Value::List(row_values.clone()); // clone outside timer
                let t = Instant::now();
                tx.query_with(&create_q)
                    .param("rows", param)
                    .fetch_all()
                    .await
                    .map_err(|e| anyhow!("UNWIND create: {e}"))?;
                t.elapsed().as_micros()
            };
            tx.rollback();
            if is_bulk {
                bulk.push(us);
            } else {
                unwind.push(us);
            }
        }
    }
    Ok((median(&mut bulk), median(&mut unwind)))
}

/// Replay one edge batch through bulk vs the production `UNWIND … MATCH …
/// CREATE` query, against synthesized `Ep` endpoints.
async fn replay_edge(
    db: &Uni,
    edge_type: &str,
    props: &[HashMap<String, Value>],
    pool: &EpPool,
    reps: usize,
) -> Result<(f64, f64)> {
    let count = props.len();
    if pool.src.len() < count || pool.dst.len() < count {
        return Err(anyhow!("endpoint pool too small for batch of {count}"));
    }
    let keys = union_keys(props);
    let unwind_q = build_edge_unwind(edge_type, &keys);
    let unwind_param = edge_unwind_param(props, pool);

    let session = db.session();
    let mut bulk = Vec::with_capacity(reps);
    let mut unwind = Vec::with_capacity(reps);
    for rep in 0..reps {
        for is_bulk in arm_order(rep) {
            let tx = session.tx().await.map_err(|e| anyhow!("session.tx: {e}"))?;
            let us = if is_bulk {
                let payload: Vec<(Vid, Vid, HashMap<String, Value>)> = (0..count)
                    .map(|i| (pool.src[i], pool.dst[i], props[i].clone()))
                    .collect();
                let t = Instant::now();
                tx.bulk_insert_edges(edge_type, payload)
                    .await
                    .map_err(|e| anyhow!("bulk_insert_edges: {e}"))?;
                t.elapsed().as_micros()
            } else {
                let param = Value::List(unwind_param.clone());
                let t = Instant::now();
                tx.query_with(&unwind_q)
                    .param("edges", param)
                    .fetch_all()
                    .await
                    .map_err(|e| anyhow!("UNWIND edge create: {e}"))?;
                t.elapsed().as_micros()
            };
            tx.rollback();
            if is_bulk {
                bulk.push(us);
            } else {
                unwind.push(us);
            }
        }
    }
    Ok((median(&mut bulk), median(&mut unwind)))
}

/// Alternate which arm runs first per rep to cancel warmup bias.
fn arm_order(rep: usize) -> [bool; 2] {
    if rep.is_multiple_of(2) {
        [true, false]
    } else {
        [false, true]
    }
}

// ── Substrate construction ──────────────────────────────────────────

/// Committed pool of generic `Ep` endpoint nodes for edge replay.
struct EpPool {
    src: Vec<Vid>,
    dst: Vec<Vid>,
}

/// Declare each recorded node label on `db` with nullable properties
/// whose types are inferred from the captured rows (no vector index).
async fn build_noembed_node_schema(db: &Uni, node_batches: &[&RecordedBatch]) -> Result<()> {
    // label -> (prop -> inferred DataType)
    let mut schema: BTreeMap<String, BTreeMap<String, DataType>> = BTreeMap::new();
    for b in node_batches {
        let RecordedBatch::Node { label, rows, .. } = b else {
            continue;
        };
        let entry = schema.entry(label.clone()).or_default();
        for row in rows {
            for (k, v) in row {
                // Prefer a concrete type over a previously-inferred
                // CypherValue/Null fallback.
                let dt = infer_dtype(v);
                entry
                    .entry(k.clone())
                    .and_modify(|cur| {
                        if matches!(cur, DataType::CypherValue)
                            && !matches!(dt, DataType::CypherValue)
                        {
                            *cur = dt.clone();
                        }
                    })
                    .or_insert(dt);
            }
        }
    }
    for (label, props) in schema {
        let mut lb = db.schema().label(&label);
        for (name, dt) in props {
            lb = lb.property_nullable(&name, dt);
        }
        lb.apply()
            .await
            .map_err(|e| anyhow!("schema apply ({label}): {e}"))?;
    }
    Ok(())
}

/// Declare the generic `Ep` label and every recorded edge type as
/// `Ep→Ep`, then commit a pool of endpoint nodes sized to the largest
/// edge batch.
async fn build_edge_substrate(db: &Uni, edge_batches: &[&RecordedBatch]) -> Result<EpPool> {
    db.schema()
        .label("Ep")
        .property("id", DataType::Int64)
        .apply()
        .await
        .map_err(|e| anyhow!("schema apply (Ep): {e}"))?;

    // edge_type -> prop -> DataType
    let mut etypes: BTreeMap<String, BTreeMap<String, DataType>> = BTreeMap::new();
    let mut max_count = 1usize;
    for b in edge_batches {
        let RecordedBatch::Edge {
            edge_type, props, ..
        } = b
        else {
            continue;
        };
        max_count = max_count.max(props.len());
        let entry = etypes.entry(edge_type.clone()).or_default();
        for p in props {
            for (k, v) in p {
                entry.entry(k.clone()).or_insert_with(|| infer_dtype(v));
            }
        }
    }
    for (etype, props) in etypes {
        let mut eb = db.schema().edge_type(&etype, &["Ep"], &["Ep"]);
        for (name, dt) in props {
            eb = eb.property_nullable(&name, dt);
        }
        eb.apply()
            .await
            .map_err(|e| anyhow!("schema apply (edge {etype}): {e}"))?;
    }

    // One committed pool: 2 * max_count Ep nodes (src half + dst half).
    let session = db.session();
    let tx = session.tx().await.map_err(|e| anyhow!("Ep pool tx: {e}"))?;
    let rows: Vec<HashMap<String, Value>> = (0..2 * max_count)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert("id".to_string(), Value::Int(i as i64));
            m
        })
        .collect();
    let vids = tx
        .bulk_insert_vertices("Ep", rows)
        .await
        .map_err(|e| anyhow!("Ep pool bulk_insert: {e}"))?;
    tx.commit()
        .await
        .map_err(|e| anyhow!("Ep pool commit: {e}"))?;
    let (src, dst) = vids.split_at(max_count);
    Ok(EpPool {
        src: src.to_vec(),
        dst: dst.to_vec(),
    })
}

// ── Cypher / type helpers ───────────────────────────────────────────

/// Sorted union of property keys across a batch's rows.
fn union_keys(rows: &[HashMap<String, Value>]) -> Vec<String> {
    let mut keys = BTreeSet::new();
    for r in rows {
        for k in r.keys() {
            keys.insert(k.clone());
        }
    }
    keys.into_iter().collect()
}

/// `UNWIND $rows AS r CREATE (n:Label {k: r.k, …})` — inline-create form
/// so non-nullable columns are satisfied at creation time.
fn build_node_create(label: &str, keys: &[String]) -> String {
    if keys.is_empty() {
        format!("UNWIND $rows AS r CREATE (n:{label})")
    } else {
        let assigns: Vec<String> = keys.iter().map(|k| format!("{k}: r.{k}")).collect();
        format!(
            "UNWIND $rows AS r CREATE (n:{label} {{{}}})",
            assigns.join(", ")
        )
    }
}

/// ` SET r.k = e.k, …` for the edge UNWIND path (empty when no props).
fn build_edge_set_clause(keys: &[String]) -> String {
    if keys.is_empty() {
        String::new()
    } else {
        let assigns: Vec<String> = keys.iter().map(|k| format!("r.{k} = e.{k}")).collect();
        format!(" SET {}", assigns.join(", "))
    }
}

/// The production edge UNWIND statement: two `id()`-keyed MATCHes + CREATE.
fn build_edge_unwind(edge_type: &str, keys: &[String]) -> String {
    let set_clause = build_edge_set_clause(keys);
    format!(
        "UNWIND $edges AS e \
         MATCH (a) WHERE id(a) = e.src \
         MATCH (b) WHERE id(b) = e.dst \
         CREATE (a)-[r:{edge_type}]->(b){set_clause} \
         RETURN id(r) AS eid"
    )
}

/// `$rows` param for the node UNWIND: one map per vertex.
fn node_unwind_param(rows: &[HashMap<String, Value>]) -> Vec<Value> {
    rows.iter().map(|r| Value::Map(r.clone())).collect()
}

/// `$edges` param for the edge UNWIND: one map per edge with `src`/`dst`
/// VIDs (from the synthesized pool) plus the edge's properties.
fn edge_unwind_param(props: &[HashMap<String, Value>], pool: &EpPool) -> Vec<Value> {
    (0..props.len())
        .map(|i| {
            let mut m = props[i].clone();
            m.insert("src".into(), Value::Int(pool.src[i].as_u64() as i64));
            m.insert("dst".into(), Value::Int(pool.dst[i].as_u64() as i64));
            Value::Map(m)
        })
        .collect()
}

/// Infer a uni-db column type from a property value.
///
/// `Null`, lists, maps, and graph values fall back to `CypherValue`
/// (JSON-backed); the schema builder declares the column nullable so a
/// later concrete value of the same key can refine it.
fn infer_dtype(v: &Value) -> DataType {
    match v {
        Value::Bool(_) => DataType::Bool,
        Value::Int(_) => DataType::Int64,
        Value::Float(_) => DataType::Float64,
        Value::String(_) => DataType::String,
        Value::Bytes(_) => DataType::Bytes,
        Value::Vector(vec) => DataType::Vector {
            dimensions: vec.len(),
        },
        Value::Temporal(t) => match t {
            TemporalValue::Date { .. } => DataType::Date,
            TemporalValue::LocalTime { .. } | TemporalValue::Time { .. } => DataType::Time,
            TemporalValue::LocalDateTime { .. } | TemporalValue::DateTime { .. } => {
                DataType::DateTime
            }
            TemporalValue::Duration { .. } => DataType::Duration,
            TemporalValue::Btic { .. } => DataType::Btic,
        },
        _ => DataType::CypherValue,
    }
}

/// Median of microsecond samples (0.0 when empty).
fn median(xs: &mut [u128]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_unstable();
    let n = xs.len();
    if n.is_multiple_of(2) {
        (xs[n / 2 - 1] + xs[n / 2]) as f64 / 2.0
    } else {
        xs[n / 2] as f64
    }
}

// ── Profiling: UNWIND path dissection ───────────────────────────────

/// Batch-size buckets for the profile breakdown (inclusive ranges).
const SIZE_BUCKETS: &[(&str, usize, usize)] = &[
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-8", 5, 8),
    ("9-16", 9, 16),
    ("17-32", 17, 32),
    ("33-64", 33, 64),
    ("65+", 65, usize::MAX),
];

fn bucket_index(n: usize) -> usize {
    SIZE_BUCKETS
        .iter()
        .position(|(_, lo, hi)| n >= *lo && n <= *hi)
        .unwrap_or(SIZE_BUCKETS.len() - 1)
}

/// Parse/plan/exec metrics accumulated over one or more UNWIND runs.
#[derive(Default, Clone)]
struct UnwindMetrics {
    n: usize,
    parse_us: u128,
    plan_us: u128,
    exec_us: u128,
    total_us: u128,
    cache_hits: usize,
    rows_scanned: usize,
    l0_reads: usize,
}

impl UnwindMetrics {
    fn add(&mut self, o: &UnwindMetrics) {
        self.n += o.n;
        self.parse_us += o.parse_us;
        self.plan_us += o.plan_us;
        self.exec_us += o.exec_us;
        self.total_us += o.total_us;
        self.cache_hits += o.cache_hits;
        self.rows_scanned += o.rows_scanned;
        self.l0_reads += o.l0_reads;
    }
}

/// Run one UNWIND statement and capture its parse/plan/exec metrics,
/// rolling back (no mutation).
async fn unwind_metrics_once(db: &Uni, q: &str, key: &str, param: Value) -> Result<UnwindMetrics> {
    let session = db.session();
    let tx = session.tx().await.map_err(|e| anyhow!("session.tx: {e}"))?;
    let res = tx
        .execute_with(q)
        .param(key, param)
        .run()
        .await
        .map_err(|e| anyhow!("UNWIND run: {e}"))?;
    let m = res.metrics();
    let out = UnwindMetrics {
        n: 1,
        parse_us: m.parse_time.as_micros(),
        plan_us: m.plan_time.as_micros(),
        exec_us: m.exec_time.as_micros(),
        total_us: m.total_time.as_micros(),
        cache_hits: usize::from(m.plan_cache_hit),
        rows_scanned: m.rows_scanned,
        l0_reads: m.l0_reads,
    };
    tx.rollback();
    Ok(out)
}

/// `PROFILE` one UNWIND statement and print the per-operator breakdown
/// plus the logical plan, rolling back.
async fn profile_one(db: &Uni, tag: &str, q: &str, key: &str, param: Value) -> Result<()> {
    let session = db.session();
    let tx = session.tx().await.map_err(|e| anyhow!("session.tx: {e}"))?;
    let (res, prof) = tx
        .execute_with(q)
        .param(key, param)
        .profile()
        .await
        .map_err(|e| anyhow!("profile {tag}: {e}"))?;
    let m = res.metrics();
    println!("\n── PROFILE: {tag} ──");
    println!("  query: {q}");
    println!(
        "  metrics: parse={}us plan={}us exec={}us total={}us | plan_cache_hit={} rows_scanned={} l0_reads={}",
        m.parse_time.as_micros(),
        m.plan_time.as_micros(),
        m.exec_time.as_micros(),
        m.total_time.as_micros(),
        m.plan_cache_hit,
        m.rows_scanned,
        m.l0_reads,
    );
    println!(
        "  profile: total_time_ms={} peak_memory_bytes={}",
        prof.total_time_ms, prof.peak_memory_bytes
    );
    println!("  operators (leaf→root):");
    println!("    {:<28} {:>8} {:>12}", "operator", "rows", "time_ms");
    for op in &prof.runtime_stats {
        println!(
            "    {:<28} {:>8} {:>12.4}",
            op.operator, op.actual_rows, op.time_ms
        );
    }
    if !prof.explain.warnings.is_empty() {
        println!("  warnings: {:?}", prof.explain.warnings);
    }
    println!("  logical plan:");
    for line in prof.explain.plan_text.lines() {
        println!("    {line}");
    }
    tx.rollback();
    Ok(())
}

/// Up to three representative batches by size: smallest, median, largest.
fn representatives<'a>(batches: &[&'a RecordedBatch]) -> Vec<&'a RecordedBatch> {
    let mut by_size: Vec<&RecordedBatch> = batches.to_vec();
    by_size.sort_by_key(|b| b.len());
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for idx in [0, by_size.len() / 2, by_size.len().saturating_sub(1)] {
        if let Some(b) = by_size.get(idx)
            && seen.insert(b.len())
        {
            out.push(*b);
        }
    }
    out
}

/// Profile the UNWIND path for nodes (no-embed) and edges: an aggregate
/// parse/plan/exec breakdown by batch size, plus a per-operator
/// `PROFILE` of representative batches.
async fn run_profile(
    db: &Uni,
    node_batches: &[&RecordedBatch],
    edge_batches: &[&RecordedBatch],
    pool: &EpPool,
) -> Result<()> {
    println!("\n══ PROFILE: UNWIND path dissection (single run/batch, rolled back) ══");
    println!(
        "exec dominating total → executor cost; plan large → per-statement planning;\n\
         plan_cache_hit shows plan-cache reuse across same-shape statements.\n"
    );

    // NODES (no-embed): UNWIND $rows AS r CREATE (n:Label {{…}})
    let mut nbk = vec![UnwindMetrics::default(); SIZE_BUCKETS.len()];
    let mut ntot = UnwindMetrics::default();
    for b in node_batches {
        let RecordedBatch::Node { label, rows, .. } = b else {
            continue;
        };
        let keys = union_keys(rows);
        let q = build_node_create(label, &keys);
        let param = Value::List(node_unwind_param(rows));
        let m = unwind_metrics_once(db, &q, "rows", param).await?;
        nbk[bucket_index(rows.len())].add(&m);
        ntot.add(&m);
    }
    print_metrics(
        "NODES (no-embed) UNWIND — parse/plan/exec by batch size",
        &nbk,
        &ntot,
    );
    for b in representatives(node_batches) {
        let RecordedBatch::Node { label, rows, .. } = b else {
            continue;
        };
        let keys = union_keys(rows);
        let q = build_node_create(label, &keys);
        let param = Value::List(node_unwind_param(rows));
        profile_one(
            db,
            &format!("node {label}, batch={}", rows.len()),
            &q,
            "rows",
            param,
        )
        .await?;
    }

    // EDGES: UNWIND $edges AS e MATCH … MATCH … CREATE … RETURN
    let mut ebk = vec![UnwindMetrics::default(); SIZE_BUCKETS.len()];
    let mut etot = UnwindMetrics::default();
    for b in edge_batches {
        let RecordedBatch::Edge {
            edge_type, props, ..
        } = b
        else {
            continue;
        };
        if pool.src.len() < props.len() {
            continue;
        }
        let keys = union_keys(props);
        let q = build_edge_unwind(edge_type, &keys);
        let param = Value::List(edge_unwind_param(props, pool));
        let m = unwind_metrics_once(db, &q, "edges", param).await?;
        ebk[bucket_index(props.len())].add(&m);
        etot.add(&m);
    }
    print_metrics("EDGES UNWIND — parse/plan/exec by batch size", &ebk, &etot);
    for b in representatives(edge_batches) {
        let RecordedBatch::Edge {
            edge_type, props, ..
        } = b
        else {
            continue;
        };
        if pool.src.len() < props.len() {
            continue;
        }
        let keys = union_keys(props);
        let q = build_edge_unwind(edge_type, &keys);
        let param = Value::List(edge_unwind_param(props, pool));
        profile_one(
            db,
            &format!("edge {edge_type}, batch={}", props.len()),
            &q,
            "edges",
            param,
        )
        .await?;
    }
    Ok(())
}

/// Print the aggregate parse/plan/exec breakdown, per batch-size bucket.
fn print_metrics(title: &str, buckets: &[UnwindMetrics], total: &UnwindMetrics) {
    println!("\n── {title} ──");
    println!(
        "{:<8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>7} {:>7} {:>9}",
        "size",
        "batches",
        "parse_us",
        "plan_us",
        "exec_us",
        "total_us",
        "exec%",
        "cache%",
        "scan/b"
    );
    println!("{:-<82}", "");
    let row = |label: &str, m: &UnwindMetrics| {
        if m.n == 0 {
            return;
        }
        let n = m.n as f64;
        let exec_pct = 100.0 * m.exec_us as f64 / (m.total_us.max(1) as f64);
        let cache_pct = 100.0 * m.cache_hits as f64 / n;
        println!(
            "{:<8} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>6.0}% {:>6.0}% {:>9.1}",
            label,
            m.n,
            m.parse_us as f64 / n,
            m.plan_us as f64 / n,
            m.exec_us as f64 / n,
            m.total_us as f64 / n,
            exec_pct,
            cache_pct,
            m.rows_scanned as f64 / n,
        );
    };
    for (i, (label, _, _)) in SIZE_BUCKETS.iter().enumerate() {
        row(label, &buckets[i]);
    }
    println!("{:-<82}", "");
    row("ALL", total);
}

// ── Reporting ───────────────────────────────────────────────────────

fn print_recorded_summary(batches: &[RecordedBatch]) {
    let mut nodes: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // key -> (n_batches, total_rows)
    let mut edges: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for b in batches {
        match b {
            RecordedBatch::Node { label, rows, .. } => {
                let e = nodes.entry(label.clone()).or_default();
                e.0 += 1;
                e.1 += rows.len();
            }
            RecordedBatch::Edge {
                edge_type, props, ..
            } => {
                let e = edges.entry(edge_type.clone()).or_default();
                e.0 += 1;
                e.1 += props.len();
            }
        }
    }
    println!("\n── captured node batches ──");
    println!(
        "{:<16} {:>9} {:>10} {:>9}",
        "label", "batches", "rows", "avg_bs"
    );
    for (k, (nb, rows)) in &nodes {
        println!(
            "{:<16} {:>9} {:>10} {:>9.1}",
            k,
            nb,
            rows,
            *rows as f64 / *nb as f64
        );
    }
    println!("── captured edge batches ──");
    println!(
        "{:<16} {:>9} {:>10} {:>9}",
        "edge_type", "batches", "edges", "avg_bs"
    );
    for (k, (nb, rows)) in &edges {
        println!(
            "{:<16} {:>9} {:>10} {:>9.1}",
            k,
            nb,
            rows,
            *rows as f64 / *nb as f64
        );
    }
}

fn print_table(title: &str, rows: &[Measured]) {
    println!("\n── {title} ──");
    if rows.is_empty() {
        println!("  (no batches)");
        return;
    }
    println!(
        "{:<16} {:>8} {:>8} {:>12} {:>12} {:>11} {:>11} {:>8}",
        "key", "batches", "rows", "bulk_us/b", "unwd_us/b", "bulk_us/op", "unwd_us/op", "speedup"
    );
    println!("{:-<96}", "");
    // Aggregate per key.
    let mut agg: BTreeMap<String, (usize, usize, f64, f64)> = BTreeMap::new();
    for m in rows {
        let e = agg.entry(m.key.clone()).or_insert((0, 0, 0.0, 0.0));
        e.0 += 1;
        e.1 += m.size;
        e.2 += m.bulk_us;
        e.3 += m.unwind_us;
    }
    let (mut tb, mut trows, mut tbulk, mut tunwd) = (0usize, 0usize, 0.0f64, 0.0f64);
    for (key, (nb, nrows, bulk, unwd)) in &agg {
        print_row(key, *nb, *nrows, *bulk, *unwd);
        tb += nb;
        trows += nrows;
        tbulk += bulk;
        tunwd += unwd;
    }
    println!("{:-<96}", "");
    print_row("TOTAL", tb, trows, tbulk, tunwd);
}

fn print_row(key: &str, nb: usize, nrows: usize, bulk: f64, unwd: f64) {
    let per_op_bulk = if nrows > 0 { bulk / nrows as f64 } else { 0.0 };
    let per_op_unwd = if nrows > 0 { unwd / nrows as f64 } else { 0.0 };
    let speedup = if bulk > 0.0 { unwd / bulk } else { 0.0 };
    println!(
        "{:<16} {:>8} {:>8} {:>12.1} {:>12.1} {:>11.2} {:>11.2} {:>7.1}x",
        key, nb, nrows, bulk, unwd, per_op_bulk, per_op_unwd, speedup
    );
}

/// Bucket batches by size and show the per-op cost curve for both arms.
fn print_size_curve(title: &str, rows: &[Measured]) {
    println!("\n── {title} ──");
    if rows.is_empty() {
        println!("  (no batches)");
        return;
    }
    println!(
        "{:<10} {:>8} {:>8} {:>13} {:>13} {:>8}",
        "size", "batches", "rows", "bulk_us/op", "unwd_us/op", "speedup"
    );
    println!("{:-<64}", "");
    // (bucket label, lo, hi) — inclusive ranges by batch size.
    let buckets: &[(&str, usize, usize)] = &[
        ("1", 1, 1),
        ("2-4", 2, 4),
        ("5-8", 5, 8),
        ("9-16", 9, 16),
        ("17-32", 17, 32),
        ("33-64", 33, 64),
        ("65-128", 65, 128),
        ("129-256", 129, 256),
        ("257+", 257, usize::MAX),
    ];
    for (label, lo, hi) in buckets {
        let mut nb = 0usize;
        let mut nrows = 0usize;
        let mut bulk = 0.0f64;
        let mut unwd = 0.0f64;
        for m in rows {
            if m.size >= *lo && m.size <= *hi {
                nb += 1;
                nrows += m.size;
                bulk += m.bulk_us;
                unwd += m.unwind_us;
            }
        }
        if nb == 0 {
            continue;
        }
        let per_op_bulk = bulk / nrows as f64;
        let per_op_unwd = unwd / nrows as f64;
        let speedup = if bulk > 0.0 { unwd / bulk } else { 0.0 };
        println!(
            "{:<10} {:>8} {:>8} {:>13.2} {:>13.2} {:>7.1}x",
            label, nb, nrows, per_op_bulk, per_op_unwd, speedup
        );
    }
}

/// Sum of per-batch medians across the whole conversation, per condition.
fn print_weighted_totals(node_noembed: &[Measured], node_embed: &[Measured], edges: &[Measured]) {
    let sum = |rows: &[Measured]| -> (f64, f64) {
        rows.iter()
            .fold((0.0, 0.0), |(b, u), m| (b + m.bulk_us, u + m.unwind_us))
    };
    println!("\n── WEIGHTED TOTALS (sum of per-batch medians across the conversation) ──");
    println!(
        "{:<28} {:>14} {:>14} {:>9}",
        "condition", "bulk_ms", "unwind_ms", "speedup"
    );
    println!("{:-<68}", "");
    for (label, rows) in [
        ("nodes (no-embed)", node_noembed),
        ("nodes (with-embed)", node_embed),
        ("edges", edges),
    ] {
        let (b, u) = sum(rows);
        let speedup = if b > 0.0 { u / b } else { 0.0 };
        println!(
            "{:<28} {:>14.2} {:>14.2} {:>8.1}x",
            label,
            b / 1000.0,
            u / 1000.0,
            speedup
        );
    }
}

// Rust guideline compliant
