//! Full per-KB count probe: counts every node label and a few edges
//! useful for ingest-performance analysis.

use std::path::PathBuf;

use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: probe_full <kb_dir>")
        .into();
    let kb = open_kb(&path, UnikoConfig::default(), &[]).await?;
    let session = kb.db().session();

    let queries = [
        ("Sessions", "MATCH (n:Session) RETURN count(n) AS n"),
        ("Messages", "MATCH (n:Message) RETURN count(n) AS n"),
        ("Chunks", "MATCH (n:Chunk) RETURN count(n) AS n"),
        ("Entities", "MATCH (n:Entity) RETURN count(n) AS n"),
        ("Observations", "MATCH (n:Observation) RETURN count(n) AS n"),
        ("Obs w/ triple", "MATCH (n:Observation) WHERE n.predicate IS NOT NULL RETURN count(n) AS n"),
        ("Obs w/ temp", "MATCH (n:Observation) WHERE n.temporal_anchor IS NOT NULL RETURN count(n) AS n"),
        ("Facts", "MATCH (n:Fact) RETURN count(n) AS n"),
        ("Facts w/ embed", "MATCH (n:Fact) WHERE n.embedding IS NOT NULL RETURN count(n) AS n"),
        ("Cycles", "MATCH (n:ConsolidationCycle) RETURN count(n) AS n"),
        ("HAS_CHUNK", "MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS n"),
        ("OBSERVED_IN", "MATCH ()-[r:OBSERVED_IN]->() RETURN count(r) AS n"),
        ("ABOUT", "MATCH ()-[r:ABOUT]->() RETURN count(r) AS n"),
        ("SUPPORTED_BY", "MATCH ()-[r:SUPPORTED_BY]->() RETURN count(r) AS n"),
        ("MENTIONS", "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n"),
    ];
    for (label, q) in queries {
        let r = session.query_with(q).fetch_all().await?;
        let n: i64 = r
            .rows()
            .first()
            .and_then(|row| row.get("n").ok())
            .unwrap_or(-1);
        println!("{label}\t{n}");
    }

    drop(session);
    if let Ok(kb_owned) = std::sync::Arc::try_unwrap(kb) {
        kb_owned.shutdown().await.ok();
    }
    Ok(())
}
