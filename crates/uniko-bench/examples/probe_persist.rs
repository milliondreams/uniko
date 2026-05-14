//! Inspect what's actually in the conv-26 KB after the puzzle run:
//! list participants, dump cycle edge counts, then try writing a fresh
//! ConsolidationCycle + Episode, shutdown, reopen, verify whether they
//! survive. Tests whether the bug is at write time or at reopen.

use std::path::PathBuf;
use std::sync::Arc;

use uni_db::Value;
use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,uni_db=debug".into()),
        )
        .init();
    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: probe_persist <kb_dir>")
        .into();

    // ── Phase 1: open & report state ───────────────────────────────
    println!("\n== Phase 1: initial open ==");
    let kb = open_kb(&path, UnikoConfig::default(), &[]).await?;
    let session = kb.db().session();

    let r = session
        .query_with("MATCH (p:Participant) RETURN p.participant_id AS pid, p.kind AS k, p.name AS n")
        .fetch_all()
        .await?;
    println!("Participants ({}):", r.rows().len());
    for row in r.rows() {
        let pid: String = row.get("pid").unwrap_or_default();
        let k: String = row.get("k").unwrap_or_default();
        let n: String = row.get("n").unwrap_or_default();
        println!("  pid={pid} kind={k} name={n}");
    }

    let r = session
        .query_with("MATCH (c:ConsolidationCycle) RETURN count(c) AS n")
        .fetch_all()
        .await?;
    let n: i64 = r.rows().first().and_then(|row| row.get("n").ok()).unwrap_or(-1);
    println!("ConsolidationCycle nodes: {n}");

    let r = session
        .query_with("MATCH (c)-[r:PROCESSED]->(o) RETURN count(r) AS n")
        .fetch_all()
        .await?;
    let n: i64 = r.rows().first().and_then(|row| row.get("n").ok()).unwrap_or(-1);
    println!("PROCESSED edges (via match): {n}");

    let r = session
        .query_with("MATCH (e:Episode) RETURN count(e) AS n")
        .fetch_all()
        .await?;
    let n: i64 = r.rows().first().and_then(|row| row.get("n").ok()).unwrap_or(-1);
    println!("Episode nodes: {n}");

    let r = session
        .query_with("MATCH (e)-[r:RECORDED_BY]->(p) RETURN count(r) AS n")
        .fetch_all()
        .await?;
    let n: i64 = r.rows().first().and_then(|row| row.get("n").ok()).unwrap_or(-1);
    println!("RECORDED_BY edges (via match): {n}");

    drop(session);

    // ── Phase 2: try writing one fresh cycle + participant ─────────
    println!("\n== Phase 2: writing fresh Participant + Cycle ==");
    let mut props: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String("probe-agent".into()));
    let pid = format!("probe-agent-{}", chrono::Utc::now().timestamp());
    match kb
        .merge_node("Participant", "participant_id", &pid, &props)
        .await
    {
        Ok(nid) => println!("  merge_node Participant → vid={nid}"),
        Err(e) => println!("  merge_node Participant FAILED: {e}"),
    }

    let started = chrono::Utc::now();
    let completed = started + chrono::Duration::seconds(1);
    match kb
        .write_consolidation_cycle(&pid, started, completed, &[], &[], &[], &[], 0, &[])
        .await
    {
        Ok(nid) => println!("  write_consolidation_cycle → vid={nid}"),
        Err(e) => println!("  write_consolidation_cycle FAILED: {e}"),
    }

    // ── Phase 3: shutdown ──────────────────────────────────────────
    println!("\n== Phase 3: shutdown ==");
    match Arc::try_unwrap(kb) {
        Ok(kb_owned) => match kb_owned.shutdown().await {
            Ok(()) => println!("  shutdown ok"),
            Err(e) => println!("  shutdown ERR: {e}"),
        },
        Err(_) => println!("  shutdown SKIPPED — outstanding Arc refs"),
    }

    // ── Phase 4: reopen & verify ───────────────────────────────────
    println!("\n== Phase 4: reopen ==");
    let kb = open_kb(&path, UnikoConfig::default(), &[]).await?;
    let session = kb.db().session();

    let r = session
        .query_with(
            "MATCH (p:Participant) WHERE p.participant_id STARTS WITH 'probe-agent-' \
             RETURN p.participant_id AS pid",
        )
        .fetch_all()
        .await?;
    println!("Probe Participants found after reopen: {}", r.rows().len());
    for row in r.rows() {
        let pid: String = row.get("pid").unwrap_or_default();
        println!("  pid={pid}");
    }

    let r = session
        .query_with("MATCH (c:ConsolidationCycle) RETURN count(c) AS n")
        .fetch_all()
        .await?;
    let n: i64 = r.rows().first().and_then(|row| row.get("n").ok()).unwrap_or(-1);
    println!("ConsolidationCycle nodes after reopen: {n}");

    drop(session);
    if let Ok(kb_owned) = Arc::try_unwrap(kb) {
        kb_owned.shutdown().await.ok();
    }
    Ok(())
}
