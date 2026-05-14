//! Dump raw properties for the bench-agent Participant and a few Episodes
//! reached via their edges (since label-scan doesn't find them).

use std::path::PathBuf;
use std::sync::Arc;
use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path: PathBuf = std::env::args().nth(1).expect("usage: probe_props <kb>").into();
    let kb = open_kb(&path, UnikoConfig::default(), &[]).await?;
    let session = kb.db().session();

    println!("== LoCoMo speaker Participants via label-scan ==");
    let r = session
        .query_with("MATCH (p:Participant) RETURN p AS p, id(p) AS vid")
        .fetch_all()
        .await?;
    for row in r.rows() {
        let vid: i64 = row.get("vid").unwrap_or(-1);
        let n: uni_db::Node = row.get("p")?;
        println!("  vid={vid} labels={:?} props={:?}", n.labels, n.properties.keys().collect::<Vec<_>>());
    }

    println!("\n== bench-agent Participant via RECORDED_BY edge ==");
    let r = session
        .query_with(
            "MATCH (e)-[:RECORDED_BY]->(p) RETURN DISTINCT p AS p, id(p) AS vid LIMIT 3",
        )
        .fetch_all()
        .await?;
    for row in r.rows() {
        let vid: i64 = row.get("vid").unwrap_or(-1);
        let n: uni_db::Node = row.get("p")?;
        println!("  vid={vid} labels={:?}", n.labels);
        for (k, v) in &n.properties {
            println!("    {k} = {v:?}");
        }
    }

    println!("\n== first Episode via RECORDED_BY edge ==");
    let r = session
        .query_with(
            "MATCH (e)-[:RECORDED_BY]->(p) RETURN e AS e, id(e) AS vid LIMIT 3",
        )
        .fetch_all()
        .await?;
    for row in r.rows() {
        let vid: i64 = row.get("vid").unwrap_or(-1);
        let n: uni_db::Node = row.get("e")?;
        println!("  vid={vid} labels={:?}", n.labels);
        for (k, v) in &n.properties {
            let s = format!("{v:?}");
            let s = if s.len() > 80 { format!("{}…", &s[..80]) } else { s };
            println!("    {k} = {s}");
        }
    }

    drop(session);
    if let Ok(o) = Arc::try_unwrap(kb) { o.shutdown().await.ok(); }
    Ok(())
}
