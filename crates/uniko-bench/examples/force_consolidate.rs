//! Force one P4 consolidation cycle on an already-ingested KB, then
//! report stats. Used to validate that consolidation produces Facts +
//! weighted SUPPORTED_BY edges (1c) on a model-backed KB when the bench
//! flow didn't run consolidation.
//!
//! Usage: `cargo run --example force_consolidate -p uniko-bench -- <kb_dir> <bench_config.json> [agent_id]`

use std::path::PathBuf;

use uniko_bench::bench_config::BenchConfig;
use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let kb_dir: PathBuf = std::env::args().nth(1).expect("usage: <kb_dir> <bench_config> [agent]").into();
    let bc: PathBuf = std::env::args().nth(2).expect("usage: <kb_dir> <bench_config> [agent]").into();
    let agent = std::env::args().nth(3).unwrap_or_else(|| "force".to_string());

    let mut config = UnikoConfig::default();
    BenchConfig::load(&bc)
        .and_then(|b| b.apply_to_uniko_config(&mut config))
        .map_err(|e| anyhow::anyhow!("loading bench config {}: {e}", bc.display()))?;

    let kb = open_kb(&kb_dir, config, &[]).await?;
    let stats = uniko_memory::consolidation::run_cycle(&kb, &agent, None).await?;
    println!(
        "cycle: obs_processed={} facts_created={} reinforced={} invalidated={} drift={}",
        stats.observations_processed,
        stats.facts_created,
        stats.facts_reinforced,
        stats.facts_invalidated,
        stats.drift_alerts
    );

    if let Ok(owned) = std::sync::Arc::try_unwrap(kb) {
        owned.shutdown().await.ok();
    }
    Ok(())
}
