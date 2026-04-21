//! Self-contained repro for rustic-ai/uni-db#39
//! Exactly the test posted in the issue comment.

use uni_db::api::schema::EmbeddingCfg;
use uni_db::{DataType, IndexType, Uni, VectorAlgo, VectorIndexCfg, VectorMetric};

#[tokio::test]
async fn similar_to_fts_zero_with_auto_embed() {
    // Need fastembed provider for the EmbeddingCfg to be accepted
    let db = Uni::in_memory()
        .xervo_catalog(vec![uni_db::ModelAliasSpec {
            alias: "embed/default".to_string(),
            task: uni_db::ModelTask::Embed,
            provider_id: "local/fastembed".to_string(),
            model_id: "AllMiniLML6V2".to_string(),
            revision: None,
            warmup: uni_db::WarmupPolicy::Lazy,
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
        .property_nullable("embedding", DataType::Vector { dimensions: 768 })
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
        .apply()
        .await
        .unwrap();

    let session = db.session();
    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Message {content: 'Lost my job as a banker yesterday'})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    // fts.query works
    let fts = session
        .query_with(
            "CALL uni.fts.query('Message', 'content', $q, 1) \
             YIELD node, score RETURN score",
        )
        .param("q", "banker")
        .fetch_all()
        .await
        .unwrap();
    let fts_score: f64 = fts.rows()[0].get("score").unwrap();
    eprintln!("fts.query score: {fts_score}");
    assert!(fts_score > 0.0);

    // similar_to returns 0
    let sim = session
        .query_with(
            "MATCH (m:Message) \
             RETURN similar_to(m.content, $q) AS score \
             ORDER BY score DESC",
        )
        .param("q", "banker")
        .fetch_all()
        .await
        .unwrap();
    let sim_score: f64 = sim.rows()[0].get("score").unwrap();
    eprintln!("similar_to score: {sim_score}");

    assert!(
        sim_score > 0.0,
        "similar_to should be positive (fts.query={fts_score}), got {sim_score}"
    );

    db.shutdown().await.unwrap();
}
