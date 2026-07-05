//! Reproduce the consolidation hang from scratch in a BRAND-NEW KB.
//!
//! Rules out any shared state/corruption in the existing KBs: creates a fresh
//! KB with real models, inserts N synthetic Observations (real auto-embedded
//! `content`, plus the `subject`/`predicate`/`object` triple consolidation
//! groups by), then runs one P4 cycle under a wall-clock timeout. If the cycle
//! times out, the hang is reproduced on a controlled, freshly-built KB.
//!
//! Usage:
//!   cargo run --release --example repro_fresh_consolidate -p uniko-bench -- \
//!       <fresh_kb_dir> <bge-small bench_config.json> [n_obs]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use uniko_bench::bench_config::BenchConfig;
use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let kb_dir: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: <kb_dir> <cfg> [n]")
        .into();
    let bc: PathBuf = std::env::args()
        .nth(2)
        .expect("usage: <kb_dir> <cfg> [n]")
        .into();
    let n: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    let mut config = UnikoConfig::default();
    BenchConfig::load(&bc)
        .and_then(|b| b.apply_to_uniko_config(&mut config))
        .map_err(|e| anyhow::anyhow!("loading bench config {}: {e}", bc.display()))?;

    let _ = std::fs::remove_dir_all(&kb_dir);
    let kb = open_kb(&kb_dir, config, &[]).await?;
    eprintln!(
        "fresh KB opened at {}; inserting {n} observations...",
        kb_dir.display()
    );

    // Insert N observations, each its own commit (like ingest). The `content`
    // column is auto-embedded by the real model on write.
    let t_ins = Instant::now();
    let sess = kb.db().session();
    for i in 0..n {
        let subj = format!("entity {}", i % 50);
        let pred = format!("predicate {}", i % 10);
        let obj = format!("object {}", i % 80);
        let content = format!("{subj} {pred} {obj} observation number {i} sample content");
        let tx = sess.tx().await?;
        tx.execute(&format!(
            "CREATE (o:Observation {{observation_id: 'obs-{i:05}', content: '{content}', \
             subject: '{subj}', predicate: '{pred}', object: '{obj}'}})"
        ))
        .await?;
        tx.commit().await?;
        if i % 100 == 0 {
            eprintln!("  inserted {i}/{n} ({}ms)", t_ins.elapsed().as_millis());
        }
    }
    eprintln!(
        "inserted {n} obs in {}ms; running consolidation (180s timeout)...",
        t_ins.elapsed().as_millis()
    );

    // Isolate the FIRST run_cycle await — the read query — from the rest.
    eprintln!("calling fetch_unprocessed_observations directly (60s timeout)...");
    let t_f = Instant::now();
    match tokio::time::timeout(
        Duration::from_secs(60),
        kb.fetch_unprocessed_observations(10_000),
    )
    .await
    {
        Ok(Ok(obs)) => eprintln!(
            "FETCH OK: {} unprocessed obs in {}ms",
            obs.len(),
            t_f.elapsed().as_millis()
        ),
        Ok(Err(e)) => eprintln!("FETCH ERRORED in {}ms: {e}", t_f.elapsed().as_millis()),
        Err(_) => {
            println!(
                "FETCH HUNG: fetch_unprocessed_observations timed out after 60s (the hang is the READ query, not the writes)"
            );
            return Ok(());
        }
    }

    let t_con = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_secs(180),
        uniko_memory::consolidation::run_cycle(&kb, "fresh-repro", None),
    )
    .await;
    match res {
        Ok(Ok(stats)) => println!(
            "CONSOLIDATION COMPLETED in {}ms: obs_processed={} facts_created={}",
            t_con.elapsed().as_millis(),
            stats.observations_processed,
            stats.facts_created
        ),
        Ok(Err(e)) => println!(
            "consolidation ERRORED after {}ms: {e}",
            t_con.elapsed().as_millis()
        ),
        Err(_) => println!(
            "CONSOLIDATION HUNG: timed out after 180s with n={n} (REPRODUCED on a fresh KB)"
        ),
    }

    if let Ok(owned) = std::sync::Arc::try_unwrap(kb) {
        owned.shutdown().await.ok();
    }
    Ok(())
}
