//! Focused repro for the conv-26 bench bugs.
//!
//! Tries to provoke two distinct post-shutdown anomalies seen in uniko2:
//!
//! ## Bug A — label-scan invisibility
//!
//! Vertices written during a long workload are fully persisted (visible via
//! `MATCH (n) WHERE id(n) = $vid` with correct `labels(n)`) but invisible to
//! `MATCH (n:Label)` after shutdown + reopen.
//!
//! ## Bug B — properties dropped for one specific late merge
//!
//! A `merge_node`-style insert immediately after a heavy write phase
//! results in a node whose label is preserved on disk but whose property
//! payload is empty after shutdown + reopen.
//!
//! ## Workload shape (mirrors bench)
//!
//! 1. Apply schema with ~5 labels and a vector index on Episode.embedding
//!    (similar fan-out to uniko-store schema).
//! 2. Bulk-write 1000 Bulk vertices (ingest analogue).
//! 3. Bulk-write 400 Fact vertices (consolidation Facts).
//! 4. Write ONE ConsolidationCycle vertex + 400 PROCESSED edges + 400
//!    CREATED edges (audit record).
//! 5. `merge_node`-style insert of ONE bench-agent Participant (the
//!    candidate Bug B victim).
//! 6. Loop 200×: read-previous-Episode + CREATE Episode + RECORDED_BY edge
//!    + FOLLOWED_BY edge + labelless `MATCH ... SET embedding=...`.  This
//!    is the interleaved read/write pattern that scenarios 8 and 9 omit.
//! 7. Shutdown.
//! 8. Reopen.  Compare label-scan vs edge-traversal counts.

use std::error::Error;
use std::time::Duration;

use tempfile::TempDir;
use uni_db::{
    DataType, IndexType, ScalarType, Uni, Value, VectorAlgo, VectorIndexCfg, VectorMetric,
};

const N_BULK: usize = 900;
const N_FACTS: usize = 400;
const N_EPISODES: usize = 200;
const DIM: usize = 768;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();
    println!("kb path: {path}");

    // ── Phase 0: schema ───────────────────────────────────────────
    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        db.shutdown().await?;
    }

    // ── Phases 1-6: workload ──────────────────────────────────────
    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        let session = db.session();

        let make_vec = |seed: usize| -> Vec<f32> {
            (0..DIM)
                .map(|i| ((seed.wrapping_mul(31).wrapping_add(i)) as f32) * 0.0001 - 0.5)
                .collect()
        };

        // Phase 1: N_BULK Bulks with embeddings (ingest analogue:
        // Observations get vectored content).
        println!("phase 1: writing {N_BULK} Bulks with embeddings");
        for i in 0..N_BULK {
            let tx = session.tx().await?;
            tx.execute_with(
                "CREATE (n:Bulk {bulk_id: $b, payload: 'p', embedding: $vec})",
            )
            .param("b", Value::String(format!("bulk-{i}")))
            .param("vec", Value::Vector(make_vec(i)))
            .run()
            .await?;
            tx.commit().await?;
        }

        // Phase 2: N_FACTS Facts with embeddings.
        println!("phase 2: writing {N_FACTS} Facts with embeddings");
        for i in 0..N_FACTS {
            let tx = session.tx().await?;
            tx.execute_with(
                "CREATE (f:Fact {fact_id: $f, subject: 's', predicate: 'p', object: 'o', embedding: $vec})",
            )
            .param("f", Value::String(format!("fact-{i}")))
            .param("vec", Value::Vector(make_vec(N_BULK + i)))
            .run()
            .await?;
            tx.commit().await?;
        }

        // Phase 3: ConsolidationCycle + edges.
        println!("phase 3: one ConsolidationCycle + {N_FACTS} CREATED edges + {N_FACTS} PROCESSED");
        let tx = session.tx().await?;
        let r = tx
            .query_with(
                "CREATE (c:ConsolidationCycle {cycle_id: 'cycle-1', agent_id: 'agent', facts_created: $fc}) RETURN id(c) AS vid",
            )
            .param("fc", Value::Int(N_FACTS as i64))
            .fetch_all()
            .await?;
        let cycle_vid: i64 = r.rows().first().unwrap().get("vid")?;
        tx.commit().await?;

        for i in 0..N_FACTS {
            let tx = session.tx().await?;
            tx.execute_with(
                "MATCH (c:ConsolidationCycle), (f:Fact {fact_id: $f}) \
                 WHERE id(c) = $cv \
                 CREATE (c)-[:CREATED]->(f), (c)-[:PROCESSED]->(f)",
            )
            .param("cv", cycle_vid)
            .param("f", Value::String(format!("fact-{i}")))
            .run()
            .await?;
            tx.commit().await?;
        }

        // Phase 4: ONE bench-agent Participant via merge_node-style
        // two-step (get-then-create).  This is the Bug B candidate.
        println!("phase 4: bench-agent Participant via two-step merge");
        let agent_id = "bench-agent-conv-26";
        // tx A: read
        let tx = session.tx().await?;
        let r = tx
            .query_with(
                "MATCH (p:Participant {participant_id: $eid}) RETURN id(p) AS vid",
            )
            .param("eid", Value::String(agent_id.to_string()))
            .fetch_all()
            .await?;
        let existing = r.rows().first().is_some();
        tx.commit().await?;
        // tx B: create
        let bench_agent_vid: i64 = if !existing {
            let tx = session.tx().await?;
            let r = tx
                .query_with(
                    "CREATE (p:Participant {participant_id: $eid, kind: 'agent', name: 'bench-agent'}) RETURN id(p) AS vid",
                )
                .param("eid", Value::String(agent_id.to_string()))
                .fetch_all()
                .await?;
            let vid: i64 = r.rows().first().unwrap().get("vid")?;
            tx.commit().await?;
            vid
        } else {
            panic!("unexpected: bench-agent already exists");
        };

        // Verify in-session: label-anchored MATCH on this exact id.
        let r = session
            .query_with(
                "MATCH (p:Participant) WHERE p.participant_id = $eid RETURN count(p) AS c",
            )
            .param("eid", Value::String(agent_id.to_string()))
            .fetch_all()
            .await?;
        let n: i64 = r.rows().first().unwrap().get("c")?;
        println!("  in-session bench-agent label-match count: {n} (expected 1)");

        // Phase 5: 200 Episodes — interleaved read/write per iteration.
        println!("phase 5: writing {N_EPISODES} Episodes (interleaved read/write)");
        let dummy_vec: Vec<f32> = (0..DIM).map(|i| (i as f32) * 0.001).collect();
        let mut prev_vid: Option<i64> = None;
        for i in 0..N_EPISODES {
            let eid = format!("ep-{i:04}");

            // tx 1: read participant by ext_id (mimics record_episode's
            // get_node_by_ext_id resolution).
            let tx = session.tx().await?;
            let r = tx
                .query_with(
                    "MATCH (p:Participant) WHERE p.participant_id = $a RETURN id(p) AS vid",
                )
                .param("a", Value::String(agent_id.to_string()))
                .fetch_all()
                .await?;
            let _participant_vid: i64 = r.rows().first().unwrap().get("vid")?;
            tx.commit().await?;

            // tx 2: read previous episode by walking back via FOLLOWED_BY
            // (mimics find_previous_episode).
            if prev_vid.is_some() {
                let tx = session.tx().await?;
                let _r = tx
                    .query_with(
                        "MATCH (e:Episode) RETURN e ORDER BY e.timestamp DESC LIMIT 1",
                    )
                    .fetch_all()
                    .await?;
                tx.commit().await?;
            }

            // tx 3: CREATE Episode + RECORDED_BY edge (mimics merge_node
            // + create_edge for the new vertex).
            let tx = session.tx().await?;
            let r = tx
                .query_with(
                    "MATCH (p:Participant) WHERE id(p) = $pv \
                     CREATE (e:Episode {episode_id: $eid, action_type: 'retrieve', \
                                        outcome: 'failure', timestamp: $ts}) \
                     CREATE (e)-[:RECORDED_BY]->(p) \
                     RETURN id(e) AS vid",
                )
                .param("pv", bench_agent_vid)
                .param("eid", Value::String(eid.clone()))
                .param("ts", Value::String(format!("2026-05-13T18:22:{:02}.000Z", i % 60)))
                .fetch_all()
                .await?;
            let vid: i64 = r.rows().first().unwrap().get("vid")?;
            tx.commit().await?;

            // tx 4: FOLLOWED_BY edge from prev (if any).
            if let Some(pv) = prev_vid {
                let tx = session.tx().await?;
                tx.execute_with(
                    "MATCH (a), (b) WHERE id(a) = $pv AND id(b) = $cv \
                     CREATE (a)-[:FOLLOWED_BY]->(b)",
                )
                .param("pv", pv)
                .param("cv", vid)
                .run()
                .await?;
                tx.commit().await?;
            }

            // tx 5: labelless SET of vector (mimics embed_episode's update_node).
            let tx = session.tx().await?;
            tx.execute_with("MATCH (n) WHERE id(n) = $v SET n.embedding = $vec")
                .param("v", vid)
                .param("vec", Value::Vector(dummy_vec.clone()))
                .run()
                .await?;
            tx.commit().await?;

            prev_vid = Some(vid);

            if (i + 1) % 50 == 0 {
                println!("    {}/{} episodes", i + 1, N_EPISODES);
            }
        }

        // In-session label-scan check before shutdown.
        let r = session
            .query_with("MATCH (e:Episode) RETURN count(e) AS c")
            .fetch_all()
            .await?;
        let pre: i64 = r.rows().first().unwrap().get("c")?;
        println!("  in-session post-loop: MATCH (e:Episode) = {pre}  (expected {N_EPISODES})");

        let r = session
            .query_with(
                "MATCH (p:Participant) WHERE p.participant_id = $eid RETURN count(p) AS c",
            )
            .param("eid", Value::String(agent_id.to_string()))
            .fetch_all()
            .await?;
        let pa: i64 = r.rows().first().unwrap().get("c")?;
        println!("  in-session bench-agent label-match: {pa}  (expected 1)");

        // Give auto-flush a moment to fire, then shutdown.
        tokio::time::sleep(Duration::from_millis(500)).await;
        println!("shutdown...");
        db.shutdown().await?;
    }

    // ── Phase 7: reopen + diagnostics ─────────────────────────────
    let db = open_db(&path).await?;
    apply_schema(&db).await?;
    let session = db.session();

    let cycle = count(&db, "MATCH (c:ConsolidationCycle) RETURN count(c) AS c").await?;
    let facts = count(&db, "MATCH (f:Fact) RETURN count(f) AS c").await?;
    let bulks = count(&db, "MATCH (b:Bulk) RETURN count(b) AS c").await?;
    let parts = count(&db, "MATCH (p:Participant) RETURN count(p) AS c").await?;
    let bench_agent_lbl = count(
        &db,
        "MATCH (p:Participant) WHERE p.participant_id = 'bench-agent-conv-26' RETURN count(p) AS c",
    )
    .await?;
    let eps_lbl = count(&db, "MATCH (e:Episode) RETURN count(e) AS c").await?;
    let eps_edge_src = count(
        &db,
        "MATCH (e)-[:RECORDED_BY]->(p) RETURN count(DISTINCT id(e)) AS c",
    )
    .await?;

    println!("\n=== post-reopen counts (label-anchored) ===");
    println!("  Bulks:               {bulks}  (expected {N_BULK})");
    println!("  Facts:               {facts}  (expected {N_FACTS})");
    println!("  ConsolidationCycle:  {cycle}  (expected 1)");
    println!("  Participants:        {parts}  (expected 1)");
    println!("  bench-agent (by id): {bench_agent_lbl}  (expected 1)");
    println!("  Episodes:            {eps_lbl}  (expected {N_EPISODES})");
    println!("  RECORDED_BY src VIDs (edge-traversal): {eps_edge_src}");

    // If Episode label-scan is short, dig into one missing vid via edge traversal.
    if eps_lbl < N_EPISODES as i64 {
        println!("\n--- Bug A diagnostic: edge-traversal probe ---");
        let r = session
            .query_with(
                "MATCH (e)-[:RECORDED_BY]->(p) RETURN id(e) AS vid, labels(e) AS lbls LIMIT 3",
            )
            .fetch_all()
            .await?;
        for row in r.rows() {
            let vid: i64 = row.get("vid").unwrap_or(-1);
            let lbls: Vec<String> = row.get("lbls").unwrap_or_default();
            // Try label-anchored match on the same vid.
            let r2 = session
                .query_with("MATCH (e:Episode) WHERE id(e) = $v RETURN count(e) AS c")
                .param("v", vid)
                .fetch_all()
                .await?;
            let n: i64 = r2.rows().first().unwrap().get("c").unwrap_or(-1);
            println!(
                "  edge-traversal vid={vid} labels(e)={lbls:?}  |  MATCH (e:Episode) WHERE id(e)={vid} → {n}"
            );
        }
    }

    // If bench-agent label-scan is short, inspect via edge traversal.
    if bench_agent_lbl == 0 {
        println!("\n--- Bug B diagnostic: bench-agent edge-traversal probe ---");
        let r = session
            .query_with(
                "MATCH (e)-[:RECORDED_BY]->(p) RETURN DISTINCT p AS p, id(p) AS vid LIMIT 1",
            )
            .fetch_all()
            .await?;
        for row in r.rows() {
            let vid: i64 = row.get("vid").unwrap_or(-1);
            let n: uni_db::Node = row.get("p")?;
            println!("  edge-traversal vid={vid} labels={:?}", n.labels);
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

    let bug_a = eps_lbl < eps_edge_src;
    let bug_b = bench_agent_lbl == 0;

    drop(session);
    db.shutdown().await?;

    println!("\n=== verdict ===");
    println!("  Bug A (label-scan invisibility): {}", if bug_a { "REPRODUCED" } else { "not reproduced" });
    println!("  Bug B (property loss):          {}", if bug_b { "REPRODUCED" } else { "not reproduced" });

    if bug_a || bug_b {
        std::process::exit(1);
    }
    Ok(())
}

async fn open_db(path: &str) -> Result<Uni, Box<dyn Error>> {
    Ok(Uni::open(path).xervo_catalog(vec![]).build().await?)
}

async fn count(db: &Uni, cypher: &str) -> Result<i64, Box<dyn Error>> {
    let r = db.session().query_with(cypher).fetch_all().await?;
    Ok(r.rows().first().and_then(|row| row.get::<i64>("c").ok()).unwrap_or(-1))
}

async fn apply_schema(db: &Uni) -> Result<(), Box<dyn Error>> {
    db.schema()
        .label("Bulk")
        .property("bulk_id", DataType::String)
        .property_nullable("payload", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("bulk_id", IndexType::Scalar(ScalarType::Hash))
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("Fact")
        .property("fact_id", DataType::String)
        .property_nullable("subject", DataType::String)
        .property_nullable("predicate", DataType::String)
        .property_nullable("object", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("fact_id", IndexType::Scalar(ScalarType::Hash))
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("ConsolidationCycle")
        .property("cycle_id", DataType::String)
        .property_nullable("agent_id", DataType::String)
        .property_nullable("facts_created", DataType::Int64)
        .index("cycle_id", IndexType::Scalar(ScalarType::Hash))
        .index("agent_id", IndexType::Scalar(ScalarType::Hash))
        .done()
        .label("Participant")
        .property("participant_id", DataType::String)
        .property_nullable("kind", DataType::String)
        .property_nullable("name", DataType::String)
        .property_nullable("first_seen", DataType::DateTime)
        .property_nullable("last_seen", DataType::DateTime)
        .index("participant_id", IndexType::Scalar(ScalarType::Hash))
        .index("kind", IndexType::Scalar(ScalarType::Hash))
        .index("name", IndexType::FullText)
        .done()
        .label("Episode")
        .property("episode_id", DataType::String)
        .property("action_type", DataType::String)
        .property_nullable("outcome", DataType::String)
        .property_nullable("timestamp", DataType::String)
        .property_nullable(
            "embedding",
            DataType::Vector { dimensions: DIM },
        )
        .index("episode_id", IndexType::Scalar(ScalarType::Hash))
        .index("action_type", IndexType::Scalar(ScalarType::Hash))
        .index(
            "embedding",
            IndexType::Vector(VectorIndexCfg {
                algorithm: VectorAlgo::Flat,
                metric: VectorMetric::Cosine,
                embedding: None,
            }),
        )
        .done()
        .edge_type("CREATED", &["ConsolidationCycle"], &["Fact"])
        .done()
        .edge_type("PROCESSED", &["ConsolidationCycle"], &["Fact"])
        .done()
        .edge_type("RECORDED_BY", &["Episode"], &["Participant"])
        .done()
        .edge_type("FOLLOWED_BY", &["Episode"], &["Episode"])
        .done()
        // Dummy labels to mirror the schema burden of the real bench
        // (20+ labels with vector + FullText indexes each).  These never
        // get vertices written to them — exactly the "deferred index"
        // state uni-db logs at open time in the bench.
        .label("Action")
        .property("action_id", DataType::String)
        .property_nullable("text", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("action_id", IndexType::Scalar(ScalarType::Hash))
        .index("text", IndexType::FullText)
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("Artifact")
        .property("artifact_id", DataType::String)
        .property_nullable("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("artifact_id", IndexType::Scalar(ScalarType::Hash))
        .index("content", IndexType::FullText)
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("Chunk")
        .property("chunk_id", DataType::String)
        .property_nullable("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("chunk_id", IndexType::Scalar(ScalarType::Hash))
        .index("content", IndexType::FullText)
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("Goal")
        .property("goal_id", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("goal_id", IndexType::Scalar(ScalarType::Hash))
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("Message")
        .property("message_id", DataType::String)
        .property_nullable("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("message_id", IndexType::Scalar(ScalarType::Hash))
        .index("content", IndexType::FullText)
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("Observation")
        .property("obs_id", DataType::String)
        .property_nullable("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("obs_id", IndexType::Scalar(ScalarType::Hash))
        .index("content", IndexType::FullText)
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("Procedure")
        .property("proc_id", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: DIM })
        .index("proc_id", IndexType::Scalar(ScalarType::Hash))
        .index("embedding", IndexType::Vector(VectorIndexCfg {
            algorithm: VectorAlgo::Flat,
            metric: VectorMetric::Cosine,
            embedding: None,
        }))
        .done()
        .label("Rule")
        .property("rule_id", DataType::String)
        .index("rule_id", IndexType::Scalar(ScalarType::Hash))
        .done()
        .label("Session")
        .property("session_id", DataType::String)
        .property_nullable("started_at", DataType::DateTime)
        .index("session_id", IndexType::Scalar(ScalarType::Hash))
        .index("started_at", IndexType::Scalar(ScalarType::BTree))
        .done()
        .label("Topic")
        .property("topic_id", DataType::String)
        .index("topic_id", IndexType::Scalar(ScalarType::Hash))
        .done()
        .apply()
        .await?;
    Ok(())
}
