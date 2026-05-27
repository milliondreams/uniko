//! Profile the per-edge-type UNWIND-MATCH-CREATE patterns used by the
//! per-message ingest path, against an already-populated KB.
//!
//! Background: at `--session-concurrency 24` we measured ~600 ms per
//! `batch_create_edges_fast_in_tx` call, regardless of how many edges
//! were in the batch. The hypothesis was that the per-call fixed cost
//! (writer-lock acquisition + Lance WAL append) dominates, not the
//! query plan. This binary uses uni-db's `ExecuteBuilder::profile()`
//! (which surfaces per-operator runtime stats for *write* queries) to
//! confirm where time actually goes inside a single isolated query —
//! the no-contention floor against which the bench's contended
//! numbers can be compared.
//!
//! Writes are rolled back by dropping the transaction without commit.
//!
//! Run:
//!     cargo run -p uniko-bench --release --features gpu-cuda \
//!         --bin profile-writes -- --kb-dir data/lme_kb_dbg/5d3d2817

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use uni_db::Value;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::KnowledgeBase;

/// CLI options.
#[derive(Parser, Debug)]
#[command(about = "Profile per-edge-type write patterns against an existing KB.")]
struct Cli {
    /// Path to an existing persistent KB directory.
    #[arg(long)]
    kb_dir: PathBuf,
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
    if !cli.kb_dir.exists() {
        return Err(anyhow!("--kb-dir does not exist: {}", cli.kb_dir.display()));
    }

    let kb =
        KnowledgeBase::open_with_xervo_no_prefetch(&cli.kb_dir, UnikoConfig::default(), Vec::new())
            .await
            .context("opening KB")?;
    let db = kb.db();
    let session = db.session();

    // Sample one node id per label by scanning the graph. If a label
    // has no rows we skip the patterns that need it.
    let msg_id = fetch_one_id(&session, "Message").await?;
    let session_id = fetch_one_id(&session, "Session").await?;
    let participant_id = fetch_one_id(&session, "Participant").await?;
    let entity_id = fetch_one_id(&session, "Entity").await?;
    let obs_id = fetch_one_id(&session, "Observation").await?;

    println!(
        "sampled ids: Message={:?} Session={:?} Participant={:?} Entity={:?} Observation={:?}\n",
        msg_id, session_id, participant_id, entity_id, obs_id
    );

    // For each pattern, build the exact Cypher
    // `batch_create_edges_fast_in_tx` generates (UNWIND $edges AS e
    // MATCH ... id(a)=e.src MATCH ... id(b)=e.dst CREATE
    // (a)-[r:TYPE]->(b)). The writes are intentionally rolled back —
    // we open a fresh tx per pattern and drop it without committing.
    if let (Some(msg), Some(p)) = (msg_id, participant_id) {
        profile_edge(
            db,
            "SENT_BY",
            "Message",
            "Participant",
            &[(msg, p)],
            &[("role", Value::String("user".into()))],
        )
        .await?;
    }
    if let (Some(msg), Some(s)) = (msg_id, session_id) {
        profile_edge(db, "IN_SESSION", "Message", "Session", &[(msg, s)], &[]).await?;
    }
    if let (Some(msg), Some(p)) = (msg_id, participant_id) {
        profile_edge(
            db,
            "ADDRESSED_TO",
            "Message",
            "Participant",
            &[(msg, p)],
            &[],
        )
        .await?;
    }
    if let Some(msg) = msg_id {
        profile_edge(
            db,
            "NEXT",
            "Message",
            "Message",
            &[(msg, msg)],
            &[("gap_ms", Value::Int(0))],
        )
        .await?;
    }
    if let (Some(msg), Some(e)) = (msg_id, entity_id) {
        // MENTIONS gets ~10 edges per message in the bench; replay 10
        // pointing at the same node so the batch shape matches.
        let edges: Vec<(i64, i64)> = (0..10).map(|_| (msg, e)).collect();
        profile_edge(
            db,
            "MENTIONS",
            "Message",
            "Entity",
            &edges,
            &[("count", Value::Int(1))],
        )
        .await?;
    }
    if let (Some(obs), Some(msg)) = (obs_id, msg_id) {
        let edges: Vec<(i64, i64)> = (0..15).map(|_| (obs, msg)).collect();
        profile_edge(db, "OBSERVED_IN", "Observation", "Message", &edges, &[]).await?;
    }
    if let (Some(obs), Some(e)) = (obs_id, entity_id) {
        // ABOUT averages ~30 edges per message in the bench data.
        let edges: Vec<(i64, i64)> = (0..30).map(|_| (obs, e)).collect();
        profile_edge(db, "ABOUT", "Observation", "Entity", &edges, &[]).await?;
    }

    Ok(())
}

/// Profile one [`KnowledgeBase::batch_create_edges_fast_in_tx`]-style
/// query against the supplied src/dst node ids.
///
/// # Errors
///
/// Returns an error if the transaction or query execution fails. The
/// transaction is intentionally dropped without commit — writes never
/// land in the KB. This means consecutive runs are idempotent on the
/// underlying storage.
async fn profile_edge(
    db: &uni_db::Uni,
    edge_type: &str,
    src_label: &str,
    dst_label: &str,
    edges: &[(i64, i64)],
    prop_pairs: &[(&str, Value)],
) -> Result<()> {
    let prop_keys: Vec<String> = prop_pairs.iter().map(|(k, _)| (*k).into()).collect();
    let list: Vec<Value> = edges
        .iter()
        .map(|(src, dst)| {
            let mut m = HashMap::with_capacity(2 + prop_keys.len());
            m.insert("src".to_string(), Value::Int(*src));
            m.insert("dst".to_string(), Value::Int(*dst));
            for (k, v) in prop_pairs {
                m.insert((*k).to_string(), v.clone());
            }
            Value::Map(m)
        })
        .collect();

    let set_clause = if prop_keys.is_empty() {
        String::new()
    } else {
        let assigns: Vec<String> = prop_keys.iter().map(|k| format!("r.{k} = e.{k}")).collect();
        format!(" SET {}", assigns.join(", "))
    };

    let cypher = format!(
        "UNWIND $edges AS e \
         MATCH (a:{src_label}) WHERE id(a) = e.src \
         MATCH (b:{dst_label}) WHERE id(b) = e.dst \
         CREATE (a)-[r:{edge_type}]->(b){set_clause}"
    );

    let session = db.session();
    let tx = session.tx().await.map_err(|e| anyhow!("tx open: {e}"))?;
    let (exec_result, profile) = tx
        .execute_with(&cypher)
        .param("edges", Value::List(list))
        .profile()
        .await
        .map_err(|e| anyhow!("profile: {e}"))?;
    drop(tx); // rollback — never commit

    println!(
        "─── {edge_type} ({edges_count} edge(s), {src_label} → {dst_label}) ───",
        edges_count = edges.len()
    );
    println!(
        "  total_time_ms = {} ms   peak_mem = {} bytes   nodes_created = {}   relationships_created = {}",
        profile.total_time_ms,
        profile.peak_memory_bytes,
        exec_result.nodes_created(),
        exec_result.relationships_created(),
    );
    for op in &profile.runtime_stats {
        println!(
            "    {:30}  rows={:6}  time={:>7.2} ms  mem={} bytes  idx_hits={:?}  idx_misses={:?}",
            op.operator,
            op.actual_rows,
            op.time_ms,
            op.memory_bytes,
            op.index_hits,
            op.index_misses,
        );
    }
    println!();
    Ok(())
}

/// Fetch one node id for a given label, or `None` if the label has no rows.
async fn fetch_one_id(session: &uni_db::Session, label: &str) -> Result<Option<i64>> {
    let cypher = format!("MATCH (n:{label}) RETURN id(n) AS nid LIMIT 1");
    let result = session
        .query(&cypher)
        .await
        .map_err(|e| anyhow!("MATCH {label}: {e}"))?;
    let nid = result.rows().first().and_then(|r| r.get::<i64>("nid").ok());
    Ok(nid)
}
