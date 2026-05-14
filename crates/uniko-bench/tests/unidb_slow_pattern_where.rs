//! Reproduction test for rustic-ai/uni-db: slow pattern matching in WHERE
//! with similar_to scoring.

use std::time::Instant;
use uni_db::api::schema::EmbeddingCfg;
use uni_db::{
    DataType, IndexType, ModelAliasSpec, ModelTask, ScalarType, Uni, VectorAlgo, VectorIndexCfg,
    VectorMetric, WarmupPolicy,
};

async fn setup_db() -> Uni {
    let db = Uni::in_memory()
        .xervo_catalog(vec![ModelAliasSpec {
            alias: "embed/default".to_string(),
            task: ModelTask::Embed,
            provider_id: "local/onnx".to_string(),
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
                    document_prefix: None,
                    query_prefix: None,
                }),
            }),
        )
        .done()
        .label("Entity")
        .property("name", DataType::String)
        .index("name", IndexType::Scalar(ScalarType::Hash))
        .done()
        .label("Participant")
        .property("name", DataType::String)
        .index("name", IndexType::Scalar(ScalarType::Hash))
        .done()
        .edge_type("MENTIONS", &["Message"], &["Entity"])
        .done()
        .edge_type("SENT_BY", &["Message"], &["Participant"])
        .done()
        .apply()
        .await
        .unwrap();

    let session = db.session();
    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Entity {name: 'Jon'})").await.unwrap();
    tx.execute("CREATE (:Entity {name: 'Gina'})").await.unwrap();
    tx.execute("CREATE (:Participant {name: 'Jon'})")
        .await
        .unwrap();
    tx.execute("CREATE (:Participant {name: 'Gina'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    for i in 0..300 {
        let speaker = if i % 2 == 0 { "Jon" } else { "Gina" };
        let other = if i % 2 == 0 { "Gina" } else { "Jon" };
        let content = format!("Message {i} from {speaker} about business and dance");

        let tx = session.tx().await.unwrap();
        tx.execute(&format!(
            "CREATE (m:Message {{content: '{content}'}}) \
             WITH m \
             MATCH (p:Participant {{name: '{speaker}'}}) CREATE (m)-[:SENT_BY]->(p) \
             WITH m \
             MATCH (e:Entity {{name: '{other}'}}) CREATE (m)-[:MENTIONS]->(e)"
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

    // Warm up
    let _ = session
        .query("MATCH (m:Message) RETURN m.content LIMIT 1")
        .await;

    // Unscoped: search all 300 messages
    let start = Instant::now();
    for _ in 0..5 {
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
    let unscoped_ms = start.elapsed().as_millis() / 5;

    // Entity-scoped: search messages connected to Jon
    let start = Instant::now();
    for _ in 0..5 {
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
    let scoped_ms = start.elapsed().as_millis() / 5;

    eprintln!("Unscoped (all 300): {unscoped_ms}ms");
    eprintln!("Entity-scoped (~150): {scoped_ms}ms");
    eprintln!(
        "Ratio: {:.1}x",
        scoped_ms as f64 / unscoped_ms.max(1) as f64
    );

    assert!(
        scoped_ms < unscoped_ms * 5,
        "Entity-scoped ({scoped_ms}ms) should not be >5x slower than unscoped ({unscoped_ms}ms). \
         The scoped query searches ~150/300 messages and should be faster, not slower."
    );

    db.shutdown().await.unwrap();
}
