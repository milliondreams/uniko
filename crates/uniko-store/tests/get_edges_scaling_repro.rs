//! Repro: `get_edges` latency scales with total graph size, not out-degree.
//!
//! A node with ~20 outgoing edges should return in sub-millisecond time
//! via CSR adjacency lookup (O(out-degree)).  This test shows that
//! `get_edges` takes progressively longer as the graph grows, even
//! though the target node's out-degree stays constant.
//!
//! Filed as <https://github.com/rustic-ai/uni-db/issues/55>.

// Rust guideline compliant

use std::time::Instant;

use uni_db::{Uni, Value};

/// Number of "sessions" to link to the participant.
/// After setup, participant has exactly this many outgoing LINK edges.
const PARTICIPANT_EDGES: usize = 20;

/// Number of filler nodes + edges to add per round.
/// Each round adds this many nodes and edges to grow the graph
/// without touching the participant node.
const FILLER_PER_ROUND: usize = 200;

/// Number of growth rounds.
const ROUNDS: usize = 15;

/// Build a minimal schema: Participant, Session, Message nodes + edges.
async fn setup_db() -> Uni {
    let db = Uni::in_memory().build().await.unwrap();

    db.schema()
        .label("Participant")
        .property("name", uni_db::DataType::String)
        .done()
        .label("Session")
        .property("session_id", uni_db::DataType::String)
        .done()
        .label("Message")
        .property("content", uni_db::DataType::String)
        .done()
        .edge_type("LINK", &["Participant"], &["Session"])
        .done()
        .edge_type("IN_SESSION", &["Message"], &["Session"])
        .done()
        .apply()
        .await
        .unwrap();

    db
}

/// Create a node and return its internal ID.
async fn create_node(db: &Uni, label: &str, props: &[(&str, Value)]) -> i64 {
    let session = db.session();
    let tx = session.tx().await.unwrap();

    let prop_str: Vec<String> = props
        .iter()
        .enumerate()
        .map(|(i, (k, _))| format!("{k}: $p{i}"))
        .collect();
    let cypher = format!("CREATE (n:{label} {{{}}}) RETURN id(n) AS vid", prop_str.join(", "));

    let mut qb = tx.query_with(&cypher);
    for (i, (_, v)) in props.iter().enumerate() {
        qb = qb.param(&format!("p{i}"), v.clone());
    }
    let result = qb.fetch_all().await.unwrap();
    tx.commit().await.unwrap();

    result.rows().first().unwrap().get::<i64>("vid").unwrap()
}

/// Create an edge and return its ID.
async fn create_edge(db: &Uni, edge_type: &str, from: i64, to: i64) -> i64 {
    let session = db.session();
    let tx = session.tx().await.unwrap();
    let cypher = format!(
        "MATCH (a), (b) WHERE id(a) = $src AND id(b) = $dst \
         CREATE (a)-[r:{edge_type}]->(b) RETURN id(r) AS eid"
    );
    let result = tx
        .query_with(&cypher)
        .param("src", from)
        .param("dst", to)
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    result.rows().first().unwrap().get::<i64>("eid").unwrap()
}

/// Measure `get_edges` latency by reading all outgoing LINK edges
/// from a node via Cypher MATCH.
async fn measure_get_edges(db: &Uni, node_id: i64) -> (usize, f64) {
    let session = db.session();

    let start = Instant::now();
    let result = session
        .query_with(
            "MATCH (a)-[r:LINK]->(b) WHERE id(a) = $nid RETURN id(r) AS eid",
        )
        .param("nid", node_id)
        .fetch_all()
        .await
        .unwrap();
    let ms = start.elapsed().as_secs_f64() * 1000.0;

    (result.rows().len(), ms)
}

#[tokio::test]
async fn repro_get_edges_scales_with_graph_size() {
    let db = setup_db().await;

    // 1. Create one Participant node — the target for get_edges.
    let participant_id = create_node(
        &db,
        "Participant",
        &[("name", Value::String("alice".into()))],
    )
    .await;

    // 2. Create PARTICIPANT_EDGES sessions and link them.
    //    After this, participant has exactly PARTICIPANT_EDGES outgoing LINK edges.
    let mut session_ids = Vec::with_capacity(PARTICIPANT_EDGES);
    for i in 0..PARTICIPANT_EDGES {
        let sid = create_node(
            &db,
            "Session",
            &[("session_id", Value::String(format!("s-{i:03}")))],
        )
        .await;
        create_edge(&db, "LINK", participant_id, sid).await;
        session_ids.push(sid);
    }

    // 3. Baseline measurement.
    let (count, baseline_ms) = measure_get_edges(&db, participant_id).await;
    assert_eq!(count, PARTICIPANT_EDGES);
    eprintln!(
        "baseline: {count} edges in {baseline_ms:.2}ms (graph has ~{} nodes)",
        PARTICIPANT_EDGES + 1
    );

    // 4. Grow the graph without touching participant's edges.
    //    Add filler Message nodes + IN_SESSION edges each round.
    let mut total_filler = 0usize;
    let mut latencies = Vec::with_capacity(ROUNDS);
    latencies.push(("baseline".to_string(), baseline_ms));

    for round in 0..ROUNDS {
        // Add filler nodes + edges.
        for j in 0..FILLER_PER_ROUND {
            let msg_id = create_node(
                &db,
                "Message",
                &[(
                    "content",
                    Value::String(format!("filler message r{round}-{j}")),
                )],
            )
            .await;
            let target_session = session_ids[j % session_ids.len()];
            create_edge(&db, "IN_SESSION", msg_id, target_session).await;
        }
        total_filler += FILLER_PER_ROUND;

        // Measure get_edges on the participant — out-degree is unchanged.
        let (count, ms) = measure_get_edges(&db, participant_id).await;
        assert_eq!(count, PARTICIPANT_EDGES, "out-degree should not change");
        let label = format!("round {round}: +{total_filler} filler");
        eprintln!("{label}: {count} edges in {ms:.2}ms");
        latencies.push((label, ms));
    }

    // 5. Report.
    let final_ms = latencies.last().unwrap().1;
    eprintln!("\n--- Summary ---");
    eprintln!(
        "out-degree: {PARTICIPANT_EDGES} (constant), graph grew from {} to {} nodes",
        PARTICIPANT_EDGES + 1,
        PARTICIPANT_EDGES + 1 + ROUNDS * FILLER_PER_ROUND
    );
    eprintln!("baseline: {baseline_ms:.2}ms → final: {final_ms:.2}ms");
    eprintln!(
        "slowdown: {:.1}x",
        if baseline_ms > 0.0 {
            final_ms / baseline_ms
        } else {
            f64::NAN
        }
    );

    // Soft assertion: with O(out-degree) we expect <5x slowdown.
    // If it's >10x, the scan is likely O(graph_size).
    if final_ms > baseline_ms * 10.0 && final_ms > 5.0 {
        eprintln!(
            "\n⚠ REGRESSION: get_edges for {PARTICIPANT_EDGES} edges took {final_ms:.1}ms \
             after adding {total_filler} unrelated nodes. Expected O(out-degree) = ~{baseline_ms:.1}ms."
        );
    }

    db.shutdown().await.unwrap();
}
