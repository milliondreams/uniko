//! Reproduction test for slow pattern matching in WHERE clause.
//!
//! When using `(m)-[:EDGE]->(n)` pattern matching inside a WHERE clause
//! combined with similar_to scoring, query execution is ~10x slower
//! than expected. The pattern matching appears to do full scans
//! instead of using indexes.
//!
//! ## Expected behavior
//! Scoping a similar_to search to messages connected to a specific entity
//! via edges should be FASTER than searching all messages, because the
//! candidate set is smaller.
//!
//! ## Actual behavior
//! The entity-scoped query takes ~2.4s per query vs ~0.1s for the
//! unscoped query on the same dataset (369 messages).

use std::time::Instant;
use uni_db::api::schema::EmbeddingCfg;
use uni_db::{
    DataType, IndexType, ModelAliasSpec, ModelTask, Uni, VectorAlgo, VectorIndexCfg,
    VectorMetric, WarmupPolicy,
};

async fn setup_db() -> Uni {
    let db = Uni::in_memory()
        .xervo_catalog(vec![ModelAliasSpec {
            alias: "embed/default".to_string(),
            task: ModelTask::Embed,
            provider_id: "local/fastembed".to_string(),
            model_id: "AllMiniLML6V2".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({}),
        }])
        .build()
        .await
        .unwrap();

    db.schema()
        .label("Message")
        .property("content", DataType::String)
        .property_nullable("embedding", DataType::Vector { dimensions: 384 })
        .index("content", IndexType::FullText)
        .index(
            "embedding",
            IndexType::Vector(VectorIndexCfg {
                algorithm: VectorAlgo::HnswSq {
                    m: 16,
                    ef_construction: 100,
                    partitions: None,
                },
                metric: VectorMetric::Cosine,
                embedding: Some(EmbeddingCfg {
                    alias: "embed/default".to_string(),
                    source_properties: vec!["content".to_string()],
                    batch_size: 32,
                }),
            }),
        )
        .done()
        .label("Entity")
        .property("name", DataType::String)
        .index("name", IndexType::Scalar(uni_db::ScalarType::Hash))
        .done()
        .label("Participant")
        .property("name", DataType::String)
        .index("name", IndexType::Scalar(uni_db::ScalarType::Hash))
        .done()
        .edge_type("MENTIONS", &["Message"], &["Entity"])
        .done()
        .edge_type("SENT_BY", &["Message"], &["Participant"])
        .done()
        .apply()
        .await
        .unwrap();

    // Create entities and participants
    let session = db.session();
    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Entity {name: 'Jon'})").await.unwrap();
    tx.execute("CREATE (:Entity {name: 'Gina'})").await.unwrap();
    tx.execute("CREATE (:Participant {name: 'Jon'})").await.unwrap();
    tx.execute("CREATE (:Participant {name: 'Gina'})").await.unwrap();
    tx.commit().await.unwrap();

    // Insert 300 messages, alternating speakers, each mentioning an entity
    let session = db.session();
    for i in 0..300 {
        let speaker = if i % 2 == 0 { "Jon" } else { "Gina" };
        let other = if i % 2 == 0 { "Gina" } else { "Jon" };
        let content = format!("Message {i} from {speaker}: talking about business and dance and life");

        let tx = session.tx().await.unwrap();
        tx.execute(&format!(
            "CREATE (m:Message {{content: '{content}'}}) \
             WITH m \
             MATCH (p:Participant {{name: '{speaker}'}}) \
             CREATE (m)-[:SENT_BY]->(p) \
             WITH m \
             MATCH (e:Entity {{name: '{other}'}}) \
             CREATE (m)-[:MENTIONS]->(e)"
        ))
        .await
        .unwrap();
        tx.commit().await.unwrap();
    }

    db
}

#[tokio::test]
async fn pattern_in_where_should_not_be_slow() {
    let db = setup_db().await;
    let session = db.session();

    // Warm up embedding model
    let _ = session
        .query("MATCH (m:Message) RETURN m.content LIMIT 1")
        .await;

    // Query 1: Unscoped similar_to (search all 300 messages)
    let start = Instant::now();
    for _ in 0..10 {
        let _ = session
            .query_with(
                "MATCH (m:Message) \
                 RETURN m.content AS content, \
                        similar_to(m.content, $q) AS score \
                 ORDER BY score DESC LIMIT 15",
            )
            .param("q", "business dance")
            .fetch_all()
            .await
            .unwrap();
    }
    let unscoped_ms = start.elapsed().as_millis() / 10;

    // Query 2: Entity-scoped similar_to (search messages connected to Jon)
    let start = Instant::now();
    for _ in 0..10 {
        let _ = session
            .query_with(
                "MATCH (m:Message) \
                 WHERE (m)-[:SENT_BY]->(:Participant {name: $ename}) \
                    OR (m)-[:MENTIONS]->(:Entity {name: $ename}) \
                 RETURN m.content AS content, \
                        similar_to(m.content, $q) AS score \
                 ORDER BY score DESC LIMIT 15",
            )
            .param("ename", "Jon")
            .param("q", "business dance")
            .fetch_all()
            .await
            .unwrap();
    }
    let scoped_ms = start.elapsed().as_millis() / 10;

    eprintln!("Unscoped: {unscoped_ms}ms per query");
    eprintln!("Entity-scoped: {scoped_ms}ms per query");
    eprintln!("Ratio: {:.1}x", scoped_ms as f64 / unscoped_ms.max(1) as f64);

    // The scoped query searches ~150/300 messages (half mention Jon).
    // It should be at most 2x slower (pattern matching overhead),
    // not 10x+ slower.
    assert!(
        scoped_ms < unscoped_ms * 5,
        "Entity-scoped query ({scoped_ms}ms) should not be >5x slower \
         than unscoped ({unscoped_ms}ms)"
    );

    db.shutdown().await.unwrap();
}
