//! Check what uni-db actually returns for the "missing" vertices.
//! Reach the source nodes via their edges instead of by label.

use std::path::PathBuf;
use std::sync::Arc;

use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: probe_unconstrained <kb_dir>")
        .into();
    let kb = open_kb(&path, UnikoConfig::default(), &[]).await?;
    let session = kb.db().session();

    println!("\n== via edge: PROCESSED source nodes ==");
    let r = session
        .query_with(
            "MATCH (c)-[:PROCESSED]->(o) \
             RETURN labels(c) AS lbls, c.cycle_id AS cid, c.agent_id AS aid, id(c) AS vid \
             LIMIT 5",
        )
        .fetch_all()
        .await?;
    println!("rows: {}", r.rows().len());
    for row in r.rows() {
        let lbls: Vec<String> = row.get("lbls").unwrap_or_default();
        let cid: String = row.get("cid").unwrap_or_default();
        let aid: String = row.get("aid").unwrap_or_default();
        let vid: i64 = row.get("vid").unwrap_or(-1);
        println!("  vid={vid} labels={lbls:?} cycle_id={cid:?} agent_id={aid:?}");
    }

    println!("\n== via edge: RECORDED_BY source nodes (Episodes) ==");
    let r = session
        .query_with(
            "MATCH (e)-[:RECORDED_BY]->(p) \
             RETURN labels(e) AS lbls, e.episode_id AS eid, e.action_type AS at, id(e) AS vid \
             LIMIT 5",
        )
        .fetch_all()
        .await?;
    println!("rows: {}", r.rows().len());
    for row in r.rows() {
        let lbls: Vec<String> = row.get("lbls").unwrap_or_default();
        let eid: String = row.get("eid").unwrap_or_default();
        let at: String = row.get("at").unwrap_or_default();
        let vid: i64 = row.get("vid").unwrap_or(-1);
        println!("  vid={vid} labels={lbls:?} episode_id={eid:?} action_type={at:?}");
    }

    println!("\n== via edge: RECORDED_BY destination nodes (Participants) ==");
    let r = session
        .query_with(
            "MATCH (e)-[:RECORDED_BY]->(p) \
             RETURN DISTINCT labels(p) AS lbls, p.participant_id AS pid, id(p) AS vid \
             LIMIT 10",
        )
        .fetch_all()
        .await?;
    println!("rows: {}", r.rows().len());
    for row in r.rows() {
        let lbls: Vec<String> = row.get("lbls").unwrap_or_default();
        let pid: String = row.get("pid").unwrap_or_default();
        let vid: i64 = row.get("vid").unwrap_or(-1);
        println!("  vid={vid} labels={lbls:?} participant_id={pid:?}");
    }

    println!("\n== MATCH (c:ConsolidationCycle) RETURN id(c), c.cycle_id ==");
    let r = session
        .query_with(
            "MATCH (c:ConsolidationCycle) RETURN id(c) AS vid, c.cycle_id AS cid LIMIT 5",
        )
        .fetch_all()
        .await?;
    println!("rows: {}", r.rows().len());

    println!("\n== get_node by edge-source VID, with and without label ==");
    // Get a vid from PROCESSED first
    let r = session
        .query_with("MATCH (c)-[:PROCESSED]->(o) RETURN id(c) AS vid LIMIT 1")
        .fetch_all()
        .await?;
    if let Some(row) = r.rows().first() {
        let vid: i64 = row.get("vid").unwrap_or(-1);
        println!("source vid from PROCESSED: {vid}");

        let r = session
            .query_with("MATCH (n) WHERE id(n) = $v RETURN labels(n) AS lbls, n")
            .param("v", vid)
            .fetch_all()
            .await?;
        for row in r.rows() {
            let lbls: Vec<String> = row.get("lbls").unwrap_or_default();
            let node: uni_db::Node = row.get("n")?;
            println!(
                "  unlabeled MATCH id={vid}: labels={lbls:?} props={:?}",
                node.properties.keys().collect::<Vec<_>>()
            );
        }

        let r = session
            .query_with("MATCH (n:ConsolidationCycle) WHERE id(n) = $v RETURN n")
            .param("v", vid)
            .fetch_all()
            .await?;
        println!("  labeled MATCH (:ConsolidationCycle) id={vid}: rows={}", r.rows().len());
    }

    drop(session);
    if let Ok(kb_owned) = Arc::try_unwrap(kb) {
        kb_owned.shutdown().await.ok();
    }
    Ok(())
}
