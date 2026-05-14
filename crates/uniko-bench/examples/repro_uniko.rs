//! Repro for the conv-26 post-shutdown bugs, this time via the
//! `uniko-store` API (one layer above raw uni-db, same as the bench).
//!
//! Splits into three variants so we can binary-search where the trigger
//! lives between `uniko-store` and `uniko-bench`:
//!
//! - `--prefetch`      : `KnowledgeBase::open_with_xervo` (default; with prefetch)
//! - `--no-prefetch`   : `KnowledgeBase::open_with_xervo_no_prefetch`
//! - `--in-memory`     : `KnowledgeBase::in_memory` (no shutdown roundtrip)
//!
//! Workload mirrors the bench: bulk Facts + consolidation cycle + late
//! Participant + 200 Episodes with interleaved read/write.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use uni_db::Value;
use uni_db::common::TemporalValue;
use uniko_store::config::UnikoConfig;
use uniko_store::schema::labels;
use uniko_store::KnowledgeBase;

#[derive(Parser, Debug)]
struct Cli {
    /// Number of Bulk-equivalent nodes to write before the target writes.
    #[arg(long, default_value_t = 900)]
    n_bulk: usize,
    /// Number of Fact nodes to write.
    #[arg(long, default_value_t = 400)]
    n_facts: usize,
    /// Number of Episodes to write in the question-loop phase.
    #[arg(long, default_value_t = 200)]
    n_episodes: usize,
    /// Skip xervo prefetch on open.
    #[arg(long)]
    no_prefetch: bool,
    /// Use in-memory KB (no shutdown roundtrip — sanity check).
    #[arg(long)]
    in_memory: bool,
    /// Reuse existing KB directory instead of creating a tempdir.
    #[arg(long)]
    kb_dir: Option<PathBuf>,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,uniko_bench=info".into()),
        )
        .init();

    let kb_path = cli.kb_dir.clone().unwrap_or_else(|| {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        std::env::temp_dir().join(format!("repro_uniko_{ts}"))
    });
    // Clean any previous KB at this path.
    if kb_path.exists() {
        std::fs::remove_dir_all(&kb_path).ok();
    }
    println!("kb path: {}", kb_path.display());

    // ── Phase 1+2+3+4+5: write workload ───────────────────────────
    {
        let kb = open_kb(&kb_path, &cli).await?;
        run_workload(&kb, &cli).await?;
        shutdown_kb(kb).await?;
    }

    if cli.in_memory {
        println!("in-memory mode: skipping reopen probe");
        return Ok(());
    }

    // ── Phase 6: reopen + diagnostics ─────────────────────────────
    let kb = open_kb(&kb_path, &cli).await?;
    let s = kb.db().session();

    let cycle: i64 = scalar(&s, "MATCH (c:ConsolidationCycle) RETURN count(c) AS c").await?;
    let facts: i64 = scalar(&s, "MATCH (f:Fact) RETURN count(f) AS c").await?;
    let bench_agent: i64 = scalar(
        &s,
        "MATCH (p:Participant) WHERE p.participant_id = 'bench-agent-repro' RETURN count(p) AS c",
    )
    .await?;
    let eps_lbl: i64 = scalar(&s, "MATCH (e:Episode) RETURN count(e) AS c").await?;
    let eps_edge: i64 = scalar(
        &s,
        "MATCH (e)-[:RECORDED_BY]->(p) RETURN count(DISTINCT id(e)) AS c",
    )
    .await?;

    println!("\n=== post-reopen counts ===");
    println!("  Facts (label):              {facts}   (expected {})", cli.n_facts);
    println!("  ConsolidationCycle (label): {cycle}   (expected 1)");
    println!("  bench-agent (label):        {bench_agent}   (expected 1)");
    println!("  Episode (label):            {eps_lbl}   (expected {})", cli.n_episodes);
    println!("  Episode via edge-traversal: {eps_edge}");

    let bug_a = eps_lbl < eps_edge;
    let bug_b_candidate = bench_agent == 0 && eps_edge > 0;

    if bug_a {
        println!("\n--- Bug A: Episode label-scan invisibility ---");
        let r = s
            .query_with(
                "MATCH (e)-[:RECORDED_BY]->(p) RETURN id(e) AS vid, labels(e) AS lbls LIMIT 3",
            )
            .fetch_all()
            .await?;
        for row in r.rows() {
            let vid: i64 = row.get("vid").unwrap_or(-1);
            let lbls: Vec<String> = row.get("lbls").unwrap_or_default();
            let n: i64 = scalar_with(
                &s,
                "MATCH (e:Episode) WHERE id(e) = $v RETURN count(e) AS c",
                "v",
                vid,
            )
            .await?;
            println!("  vid={vid} labels(e)={lbls:?}  MATCH (e:Episode) WHERE id(e)={vid} -> {n}");
        }
    }

    if bug_b_candidate {
        println!("\n--- Bug B: bench-agent property probe ---");
        let r = s
            .query_with("MATCH (e)-[:RECORDED_BY]->(p) RETURN DISTINCT p AS p, id(p) AS vid LIMIT 1")
            .fetch_all()
            .await?;
        for row in r.rows() {
            let vid: i64 = row.get("vid").unwrap_or(-1);
            let n: uni_db::Node = row.get("p")?;
            println!("  vid={vid} labels={:?} prop_count={}", n.labels, n.properties.len());
            if n.properties.is_empty() {
                println!("    ❌ ZERO properties (Bug B reproduced)");
            } else {
                for (k, v) in &n.properties {
                    let s = format!("{v:?}");
                    let s = if s.len() > 80 { format!("{}…", &s[..80]) } else { s };
                    println!("    {k} = {s}");
                }
            }
        }
    }

    println!("\n=== verdict ===");
    println!("  Bug A reproduced: {bug_a}");
    println!("  Bug B reproduced: {bug_b_candidate}");

    drop(s);
    shutdown_kb(kb).await?;
    if bug_a || bug_b_candidate {
        std::process::exit(1);
    }
    Ok(())
}

async fn open_kb(path: &std::path::Path, cli: &Cli) -> Result<Arc<KnowledgeBase>> {
    let config = UnikoConfig::default();
    let kb = if cli.in_memory {
        KnowledgeBase::in_memory(config).await?
    } else if cli.no_prefetch {
        KnowledgeBase::open_with_xervo_no_prefetch(path, config, vec![]).await?
    } else {
        KnowledgeBase::open_with_xervo(path, config, vec![]).await?
    };
    Ok(Arc::new(kb))
}

async fn shutdown_kb(kb: Arc<KnowledgeBase>) -> Result<()> {
    match Arc::try_unwrap(kb) {
        Ok(k) => k.shutdown().await?,
        Err(_) => {
            anyhow::bail!("outstanding Arc references prevented shutdown");
        }
    }
    Ok(())
}

async fn run_workload(kb: &Arc<KnowledgeBase>, cli: &Cli) -> Result<()> {
    // Phase 1: bulk Observation-equivalent nodes (via uniko-store create_node).
    println!("phase 1: {} Observations", cli.n_bulk);
    for i in 0..cli.n_bulk {
        let mut props: HashMap<String, Value> = HashMap::new();
        props.insert("observation_id".into(), Value::String(format!("obs-{i}")));
        props.insert("content".into(), Value::String(format!("obs-{i} content")));
        props.insert(
            "subject".into(),
            Value::String(format!("s{}", i % 50)),
        );
        props.insert(
            "predicate".into(),
            Value::String(format!("p{}", i % 30)),
        );
        props.insert(
            "object".into(),
            Value::String(format!("o{}", i % 70)),
        );
        kb.create_node(labels::OBSERVATION, &props).await?;
    }

    // Phase 2: bulk Facts.
    println!("phase 2: {} Facts", cli.n_facts);
    for i in 0..cli.n_facts {
        let mut props: HashMap<String, Value> = HashMap::new();
        props.insert("fact_id".into(), Value::String(format!("fact-{i}")));
        props.insert("subject".into(), Value::String(format!("s{}", i % 50)));
        props.insert("predicate".into(), Value::String(format!("p{}", i % 30)));
        props.insert("object".into(), Value::String(format!("o{}", i % 70)));
        props.insert("confidence".into(), Value::Float(0.8));
        props.insert("observation_count".into(), Value::Int(1));
        kb.create_node(labels::FACT, &props).await?;
    }

    // Phase 3: ConsolidationCycle audit via the real uniko-store API.
    println!("phase 3: write_consolidation_cycle");
    let started = Utc::now();
    let completed = started + chrono::Duration::seconds(1);
    kb.write_consolidation_cycle(
        "repro-agent",
        started,
        completed,
        &[],
        &[],
        &[],
        &[],
        0,
        &[],
    )
    .await?;

    // Phase 4: bench-agent Participant via merge_node (the Bug B candidate).
    println!("phase 4: bench-agent Participant via merge_node");
    let agent_id = "bench-agent-repro";
    let mut agent_props: HashMap<String, Value> = HashMap::new();
    agent_props.insert("kind".into(), Value::String("agent".into()));
    agent_props.insert("name".into(), Value::String("bench-agent".into()));
    kb.merge_node(labels::PARTICIPANT, "participant_id", agent_id, &agent_props)
        .await?;

    // In-session check.
    let n: i64 = scalar(
        &kb.db().session(),
        &format!(
            "MATCH (p:Participant) WHERE p.participant_id = '{agent_id}' RETURN count(p) AS c"
        ),
    )
    .await?;
    println!("  in-session bench-agent label-match: {n}");

    // Phase 5: Episodes via interleaved read/write.  Uses raw db() to
    // exercise the same paths as `record_episode` (merge_node + edges
    // + labelless SET embedding).
    println!("phase 5: {} Episodes (interleaved)", cli.n_episodes);
    let dummy_vec: Vec<f32> =
        (0..768).map(|i| (i as f32) * 0.0001 - 0.5).collect();

    let mut prev_vid: Option<i64> = None;
    let (participant_nid, _) = kb
        .get_node_by_ext_id(labels::PARTICIPANT, "participant_id", agent_id)
        .await?
        .expect("participant should exist");

    for i in 0..cli.n_episodes {
        let eid = format!("ep-{i:04}");

        // Read prev episode (mimics find_previous_episode).
        if prev_vid.is_some() {
            let _ = kb.db().session()
                .query_with("MATCH (e:Episode) RETURN e ORDER BY e.timestamp DESC LIMIT 1")
                .fetch_all()
                .await?;
        }

        // MERGE Episode (single-vertex creation via merge_node).
        let mut ep_props: HashMap<String, Value> = HashMap::new();
        ep_props.insert("action_type".into(), Value::String("retrieve".into()));
        ep_props.insert("outcome".into(), Value::String("failure".into()));
        let now = Utc::now();
        ep_props.insert(
            "timestamp".into(),
            Value::Temporal(TemporalValue::DateTime {
                nanos_since_epoch: now.timestamp_nanos_opt().unwrap_or(0),
                offset_seconds: 0,
                timezone_name: Some("UTC".into()),
            }),
        );
        ep_props.insert("importance".into(), Value::Float(0.5));
        let episode_nid = kb
            .merge_node(labels::EPISODE, "episode_id", &eid, &ep_props)
            .await?;

        // RECORDED_BY edge.
        kb.create_edge(
            uniko_store::schema::edges::RECORDED_BY,
            episode_nid,
            participant_nid,
            &HashMap::new(),
        )
        .await?;

        // FOLLOWED_BY edge from prev.
        if let Some(pv) = prev_vid {
            let mut edge_props = HashMap::new();
            edge_props.insert("gap_ms".into(), Value::Int(1000));
            kb.create_edge(
                uniko_store::schema::edges::FOLLOWED_BY,
                pv,
                episode_nid,
                &edge_props,
            )
            .await?;
        }

        // update_node for embedding (labelless SET — same as embed_episode).
        let mut emb_props = HashMap::new();
        emb_props.insert("embedding".into(), Value::Vector(dummy_vec.clone()));
        kb.update_node(episode_nid, &emb_props).await?;

        prev_vid = Some(episode_nid);

        if (i + 1) % 50 == 0 {
            println!("    {}/{}", i + 1, cli.n_episodes);
        }
    }

    let pre: i64 = scalar(&kb.db().session(), "MATCH (e:Episode) RETURN count(e) AS c").await?;
    println!("  in-session post-loop Episode count: {pre}");
    Ok(())
}

async fn scalar(s: &uni_db::Session, cypher: &str) -> Result<i64> {
    let r = s.query_with(cypher).fetch_all().await?;
    Ok(r.rows().first().and_then(|row| row.get::<i64>("c").ok()).unwrap_or(-1))
}

async fn scalar_with(s: &uni_db::Session, cypher: &str, key: &str, val: i64) -> Result<i64> {
    let r = s.query_with(cypher).param(key, val).fetch_all().await?;
    Ok(r.rows().first().and_then(|row| row.get::<i64>("c").ok()).unwrap_or(-1))
}
