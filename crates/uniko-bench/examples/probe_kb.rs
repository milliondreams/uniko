//! Quick read-only probe of an ingested KB — counts nodes by label
//! and prints a few sample Facts.  Used to validate P4 Consolidation
//! produced the expected schema-side artifacts after a bench run.

use std::path::PathBuf;

use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: probe_kb <kb_dir>")
        .into();
    let kb = open_kb(&path, UnikoConfig::default(), &[]).await?;
    let session = kb.db().session();

    let counts = [
        ("Facts                  ", "MATCH (f:Fact) RETURN count(f) AS n"),
        ("Observations (total)   ", "MATCH (o:Observation) RETURN count(o) AS n"),
        ("Observations (w/triple)", "MATCH (o:Observation) WHERE o.predicate IS NOT NULL RETURN count(o) AS n"),
        ("Processed observations ", "MATCH (:ConsolidationCycle)-[:PROCESSED]->(o) RETURN count(o) AS n"),
        ("ConsolidationCycles    ", "MATCH (c:ConsolidationCycle) RETURN count(c) AS n"),
        ("SUPPORTED_BY edges     ", "MATCH (:Fact)-[r:SUPPORTED_BY]->(:Observation) RETURN count(r) AS n"),
        ("Facts with embedding   ", "MATCH (f:Fact) WHERE f.embedding IS NOT NULL RETURN count(f) AS n"),
        ("Bench Participant     ", "MATCH (p:Participant {participant_id: 'bench-agent-conv-26'}) RETURN count(p) AS n"),
        ("Participants total    ", "MATCH (p:Participant) RETURN count(p) AS n"),
        ("Episodes               ", "MATCH (e:Episode) RETURN count(e) AS n"),
        ("Episodes w/ embedding  ", "MATCH (e:Episode) WHERE e.embedding IS NOT NULL RETURN count(e) AS n"),
        ("Entities (unstable=true)", "MATCH (e:Entity) WHERE e.unstable = true RETURN count(e) AS n"),
        ("INVALIDATES edges      ", "MATCH (:Fact)-[r:INVALIDATES]->(:Fact) RETURN count(r) AS n"),
        ("RECORDED_BY edges      ", "MATCH (:Episode)-[r:RECORDED_BY]->() RETURN count(r) AS n"),
    ];
    for (label, q) in counts {
        let r = session.query_with(q).fetch_all().await?;
        let n: i64 = r
            .rows()
            .first()
            .and_then(|row| row.get("n").ok())
            .unwrap_or(-1);
        println!("{label}: {n}");
    }

    println!("\nConsolidationCycle audit rows:");
    let r = session
        .query_with(
            "MATCH (c:ConsolidationCycle) RETURN c.agent_id AS a, c.facts_created AS fc, \
             c.facts_invalidated AS fi, c.drift_alerts AS da, c.started_at AS t ORDER BY c.started_at",
        )
        .fetch_all()
        .await?;
    for row in r.rows() {
        let a: String = row.get("a").unwrap_or_default();
        let fc: i64 = row.get("fc").unwrap_or(-1);
        let fi: i64 = row.get("fi").unwrap_or(-1);
        let da: i64 = row.get("da").unwrap_or(-1);
        let t: String = row.get("t").unwrap_or_default();
        println!("  agent={a} fc={fc} fi={fi} da={da} t={t}");
    }

    println!("\nTop-10 most-supported Facts:");
    let r = session
        .query_with(
            "MATCH (f:Fact) RETURN f.subject AS s, f.predicate AS p, f.object AS o, \
             f.observation_count AS n, f.confidence AS c \
             ORDER BY f.observation_count DESC LIMIT 10",
        )
        .fetch_all()
        .await?;
    for row in r.rows() {
        let s: String = row.get("s").unwrap_or_default();
        let p: String = row.get("p").unwrap_or_default();
        let o: String = row.get("o").unwrap_or_default();
        let n: i64 = row.get("n").unwrap_or(0);
        let c: f64 = row.get("c").unwrap_or(0.0);
        println!("  ({s} | {p} | {o})  n={n}  conf={c:.3}");
    }

    drop(session);
    if let Ok(kb_owned) = std::sync::Arc::try_unwrap(kb) {
        kb_owned.shutdown().await.ok();
    }
    Ok(())
}
