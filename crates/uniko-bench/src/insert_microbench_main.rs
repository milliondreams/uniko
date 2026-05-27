//! Isolated single-vs-concurrent insert microbench.
//!
//! Opens a fresh uni-db, defines a simple schema (no auto-embed,
//! no vector indexes, no NLP, no edges), and times node inserts
//! across a few configurations:
//!
//! * api: cypher single-row `CREATE` via `tx.execute_with` OR
//!   `tx.bulk_insert_vertices` with batch=1
//! * label-mode: every worker writes to the SAME label, or each
//!   worker writes to a DIFFERENT label (Item0..Item23). The latter
//!   tests whether contention is per-label (per-table) or more global.
//! * sess: number of concurrent worker tasks
//!
//! Each worker opens its own transaction, inserts one node, commits.
//! Per-node wall is averaged across all workers across reps.
//!
//! Build & run:
//!   cargo build -p uniko-bench --bin insert-microbench --release
//!   ./target/release/insert-microbench --nodes 510 --sess 1,8,24 --reps 3

use std::sync::Arc;
use std::time::Instant;

use std::collections::HashMap;

use anyhow::{Result, anyhow};
use clap::Parser;
use uni_db::{DataType, Uni, Value, Vid};

#[derive(Parser, Debug)]
#[command(about = "Isolated single-vs-concurrent insert microbench (no NLP, no embed).")]
struct Cli {
    /// KB directory; defaults to a temp dir per scenario.
    #[arg(long)]
    kb_dir: Option<std::path::PathBuf>,

    /// Total operations per scenario (nodes or edges).
    #[arg(long, default_value = "510")]
    nodes: usize,

    /// Concurrency levels (comma-separated).
    #[arg(long, default_value = "1,8,24")]
    sess: String,

    /// Repetitions per scenario (results reported as mean).
    #[arg(long, default_value = "3")]
    reps: usize,

    /// Max distinct labels for the multi-label scenarios.
    #[arg(long, default_value = "24")]
    label_pool: usize,

    /// What to insert: `node`, `edge`, or `both`.
    #[arg(long, default_value = "both")]
    op: String,
}

/// Insert pattern under test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Api {
    Cypher,
    Bulk,
}

/// Whether all workers share one label or fan out across many.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabelMode {
    Same,
    PerWorker,
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
    let sess_levels: Vec<usize> = cli
        .sess
        .split(',')
        .map(|s| s.trim().parse::<usize>().expect("--sess parses as u32"))
        .collect();

    println!(
        "microbench: ops={} reps={} sess={:?} label_pool={} op_kind={}",
        cli.nodes, cli.reps, sess_levels, cli.label_pool, cli.op
    );
    println!();

    let ops: Vec<OpKind> = match cli.op.as_str() {
        "node" => vec![OpKind::Node],
        "edge" => vec![OpKind::Edge],
        "both" => vec![OpKind::Node, OpKind::Edge],
        other => return Err(anyhow!("invalid --op '{other}', expected node|edge|both")),
    };

    for &op_kind in &ops {
        println!("── {op_kind:?} insert ──");
        println!(
            "{:<8} {:<10} {:<5} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "api",
            "lblmode",
            "sess",
            "wall_ms",
            "per_op_us",
            "parse_us",
            "plan_us",
            "exec_us",
            "ops/sec"
        );
        println!("{:-<92}", "");
        for sess in &sess_levels {
            for &api in &[Api::Cypher, Api::Bulk] {
                for &lblmode in &[LabelMode::Same, LabelMode::PerWorker] {
                    if lblmode == LabelMode::PerWorker && *sess == 1 {
                        continue;
                    }
                    let mut walls = Vec::with_capacity(cli.reps);
                    let mut metrics_sum = CypherMetrics::default();
                    for _ in 0..cli.reps {
                        let (wall, metrics) =
                            run_scenario(op_kind, api, lblmode, *sess, cli.nodes, cli.label_pool)
                                .await?;
                        walls.push(wall);
                        metrics_sum.add(&metrics);
                    }
                    let wall_avg_ms = walls.iter().map(|w| w.as_secs_f64() * 1000.0).sum::<f64>()
                        / walls.len() as f64;
                    let per_op_us = (wall_avg_ms * 1000.0) / cli.nodes as f64;
                    let ops_per_sec = (cli.nodes as f64 / wall_avg_ms) * 1000.0;
                    // Per-call cypher phase breakdowns (only for cypher path).
                    let (parse_avg, plan_avg, exec_avg) = if metrics_sum.calls > 0 {
                        let c = metrics_sum.calls as f64;
                        (
                            metrics_sum.parse_us as f64 / c,
                            metrics_sum.plan_us as f64 / c,
                            metrics_sum.exec_us as f64 / c,
                        )
                    } else {
                        (0.0, 0.0, 0.0)
                    };
                    println!(
                        "{:<8} {:<10} {:<5} {:>9.2} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.0}",
                        format!("{:?}", api).to_lowercase(),
                        format!("{:?}", lblmode).to_lowercase(),
                        sess,
                        wall_avg_ms,
                        per_op_us,
                        parse_avg,
                        plan_avg,
                        exec_avg,
                        ops_per_sec,
                    );
                }
            }
        }
        println!();
    }

    Ok(())
}

/// Which insert pattern to time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpKind {
    Node,
    Edge,
}

/// Per-scenario aggregate metrics, populated for Cypher paths.
#[derive(Debug, Clone, Default)]
struct CypherMetrics {
    calls: u64,
    parse_us: u64,
    plan_us: u64,
    exec_us: u64,
    total_us: u64,
}

impl CypherMetrics {
    fn add(&mut self, other: &CypherMetrics) {
        self.calls += other.calls;
        self.parse_us += other.parse_us;
        self.plan_us += other.plan_us;
        self.exec_us += other.exec_us;
        self.total_us += other.total_us;
    }
}

/// Run one scenario: open a fresh Uni, define schema, fan out
/// `nodes` inserts across `sess` workers, return wall.
async fn run_scenario(
    op: OpKind,
    api: Api,
    lblmode: LabelMode,
    sess: usize,
    nodes: usize,
    label_pool: usize,
) -> Result<(std::time::Duration, CypherMetrics)> {
    let tmp = tempfile::tempdir()?;
    let db = Uni::open(tmp.path().to_string_lossy().as_ref())
        .build()
        .await
        .map_err(|e| anyhow!("Uni::open: {e}"))?;

    // Schema: `label_pool` labels, each with the same three properties.
    // No vector indexes, no auto-embed.
    let n_labels = match lblmode {
        LabelMode::Same => 1,
        LabelMode::PerWorker => label_pool,
    };
    // Apply schema one label at a time to sidestep SchemaBuilder ↔
    // LabelBuilder type-juggling in the chained API.
    for i in 0..n_labels {
        let label_name = label_for(i);
        db.schema()
            .label(&label_name)
            .property("id", DataType::String)
            .property("val", DataType::Int64)
            .property("text", DataType::String)
            .apply()
            .await
            .map_err(|e| anyhow!("schema apply ({label_name}): {e}"))?;
    }
    // For edge scenarios, declare one edge type per label pair so that
    // a worker writing to Item{i} also has an edge type Edge{i} that
    // both endpoints are valid for.
    if op == OpKind::Edge {
        for i in 0..n_labels {
            let label = label_for(i);
            let etype = edge_for(i);
            db.schema()
                .edge_type(&etype, &[label.as_str()], &[label.as_str()])
                .apply()
                .await
                .map_err(|e| anyhow!("schema apply ({etype}): {e}"))?;
        }
    }

    // For edge scenarios, pre-populate 2 × nodes vertices so edges have
    // valid endpoints. Bulk path so this setup doesn't dominate wall.
    let mut endpoint_vids: Vec<(String, Vec<Vid>)> = Vec::new();
    if op == OpKind::Edge {
        let session = db.session();
        for i in 0..n_labels {
            let label = label_for(i);
            // Each label gets enough vertices to cover the scenario's
            // workers writing to it.
            let workers_for_label = match lblmode {
                LabelMode::Same => sess,
                LabelMode::PerWorker => sess.div_ceil(n_labels),
            };
            let per_worker_edges = nodes.div_ceil(sess);
            let needed = workers_for_label * per_worker_edges * 2;
            let tx = session.tx().await.map_err(|e| anyhow!("setup tx: {e}"))?;
            let props: Vec<HashMap<String, Value>> = (0..needed)
                .map(|n| {
                    let mut p: HashMap<String, Value> = HashMap::new();
                    p.insert("id".into(), Value::String(format!("setup-{i}-{n}")));
                    p.insert("val".into(), Value::Int(n as i64));
                    p.insert("text".into(), Value::String(String::new()));
                    p
                })
                .collect();
            let vids = tx
                .bulk_insert_vertices(&label, props)
                .await
                .map_err(|e| anyhow!("setup bulk_insert_vertices: {e}"))?;
            tx.commit()
                .await
                .map_err(|e| anyhow!("setup commit: {e}"))?;
            endpoint_vids.push((label.clone(), vids));
        }
    }

    let db = Arc::new(db);
    let per_worker = nodes.div_ceil(sess);
    let endpoint_vids = Arc::new(endpoint_vids);
    let wall_start = Instant::now();
    let mut handles = Vec::with_capacity(sess);
    for w in 0..sess {
        let db = db.clone();
        let label_idx = match lblmode {
            LabelMode::Same => 0,
            LabelMode::PerWorker => w % label_pool,
        };
        let label = label_for(label_idx);
        let start_id = w * per_worker;
        let end_id = std::cmp::min(start_id + per_worker, nodes);
        let endpoint_vids = endpoint_vids.clone();
        handles.push(tokio::spawn(async move {
            match op {
                OpKind::Node => run_worker_nodes(db, &label, start_id, end_id, w, api).await,
                OpKind::Edge => {
                    let etype = edge_for(label_idx);
                    let vids = endpoint_vids
                        .iter()
                        .find(|(l, _)| l == &label)
                        .map(|(_, v)| v.clone())
                        .ok_or_else(|| anyhow!("no setup vids for label {label}"))?;
                    run_worker_edges(db, &label, &etype, &vids, start_id, end_id, w, api).await
                }
            }
        }));
    }
    let mut metrics = CypherMetrics::default();
    for h in handles {
        let m = h
            .await
            .map_err(|e| anyhow!("worker join: {e}"))?
            .map_err(|e| anyhow!("worker err: {e}"))?;
        metrics.add(&m);
    }
    let wall = wall_start.elapsed();

    if let Some(db_inner) = Arc::into_inner(db) {
        let _ = db_inner.shutdown().await;
    }
    Ok((wall, metrics))
}

/// One worker for node-insert scenarios.
async fn run_worker_nodes(
    db: Arc<Uni>,
    label: &str,
    start_id: usize,
    end_id: usize,
    worker_id: usize,
    api: Api,
) -> Result<CypherMetrics> {
    let session = db.session();
    let mut metrics = CypherMetrics::default();
    for i in start_id..end_id {
        let tx = session.tx().await.map_err(|e| anyhow!("session.tx: {e}"))?;
        let id_str = format!("w{worker_id}-{i}");
        let val = i as i64;
        let text = format!("text-{i}");
        match api {
            Api::Cypher => {
                let cypher = format!(
                    "CREATE (n:{label} {{id: $id, val: $val, text: $text}}) RETURN id(n) AS vid"
                );
                let result = tx
                    .query_with(&cypher)
                    .param("id", Value::String(id_str))
                    .param("val", Value::Int(val))
                    .param("text", Value::String(text))
                    .fetch_all()
                    .await
                    .map_err(|e| anyhow!("cypher: {e}"))?;
                let m = result.metrics();
                metrics.calls += 1;
                metrics.parse_us += m.parse_time.as_micros() as u64;
                metrics.plan_us += m.plan_time.as_micros() as u64;
                metrics.exec_us += m.exec_time.as_micros() as u64;
                metrics.total_us += m.total_time.as_micros() as u64;
            }
            Api::Bulk => {
                let mut props: HashMap<String, Value> = HashMap::new();
                props.insert("id".into(), Value::String(id_str));
                props.insert("val".into(), Value::Int(val));
                props.insert("text".into(), Value::String(text));
                let _vids: Vec<Vid> = tx
                    .bulk_insert_vertices(label, vec![props])
                    .await
                    .map_err(|e| anyhow!("bulk: {e}"))?;
            }
        }
        tx.commit().await.map_err(|e| anyhow!("commit: {e}"))?;
    }
    Ok(metrics)
}

/// One worker for edge-insert scenarios. Picks endpoints from the
/// pre-populated `vids` for this label. Each per-tx insert is one edge.
#[allow(clippy::too_many_arguments)]
async fn run_worker_edges(
    db: Arc<Uni>,
    _label: &str,
    etype: &str,
    vids: &[Vid],
    start_id: usize,
    end_id: usize,
    _worker_id: usize,
    api: Api,
) -> Result<CypherMetrics> {
    if vids.len() < 2 {
        return Err(anyhow!("need ≥2 endpoint vids"));
    }
    let session = db.session();
    let mut metrics = CypherMetrics::default();
    for i in start_id..end_id {
        // Pair (2i, 2i+1) endpoints — wrapped to fit available vids.
        let src = vids[(2 * i) % vids.len()];
        let dst = vids[(2 * i + 1) % vids.len()];
        let tx = session.tx().await.map_err(|e| anyhow!("session.tx: {e}"))?;
        match api {
            Api::Cypher => {
                // Two-MATCH-then-CREATE: the canonical Cypher edge-create
                // pattern when both endpoint VIDs are known.
                let cypher = format!(
                    "MATCH (a) WHERE id(a) = $src \
                     MATCH (b) WHERE id(b) = $dst \
                     CREATE (a)-[r:{etype}]->(b) RETURN id(r) AS eid"
                );
                let result = tx
                    .query_with(&cypher)
                    .param("src", Value::Int(src.as_u64() as i64))
                    .param("dst", Value::Int(dst.as_u64() as i64))
                    .fetch_all()
                    .await
                    .map_err(|e| anyhow!("cypher edge: {e}"))?;
                let m = result.metrics();
                metrics.calls += 1;
                metrics.parse_us += m.parse_time.as_micros() as u64;
                metrics.plan_us += m.plan_time.as_micros() as u64;
                metrics.exec_us += m.exec_time.as_micros() as u64;
                metrics.total_us += m.total_time.as_micros() as u64;
            }
            Api::Bulk => {
                tx.bulk_insert_edges(etype, vec![(src, dst, HashMap::new())])
                    .await
                    .map_err(|e| anyhow!("bulk edge: {e}"))?;
            }
        }
        tx.commit().await.map_err(|e| anyhow!("commit: {e}"))?;
    }
    Ok(metrics)
}

fn label_for(i: usize) -> String {
    format!("Item{i}")
}
fn edge_for(i: usize) -> String {
    format!("Edge{i}")
}
