//! Reproduction test for rustic-ai/uni-db#39:
//! similar_to() on fulltext-indexed field returns 0.0

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use uni_db::ModelAliasSpec;
use uniko_extract::ingest::message::ingest_message;
use uniko_pipes::types::IngestMessage;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

#[tokio::test]
async fn similar_to_fts_with_real_uniko_schema() {
    // Use the real uniko KnowledgeBase (full schema with auto-embed)
    let config = UnikoConfig::default();
    let kb = Arc::new(
        KnowledgeBase::in_memory_with_xervo(config, Vec::<ModelAliasSpec>::new())
            .await
            .unwrap(),
    );

    // Ingest a message through the real pipeline
    let msg = IngestMessage {
        message_id: "m1".into(),
        content: "Lost my job as a banker yesterday".into(),
        content_type: "text".into(),
        sender_id: "Jon".into(),
        session_id: "s1".into(),
        addressed_to: None,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    };
    ingest_message(&kb, &msg).await.unwrap();

    // Test flush
    match kb.db().flush().await {
        Ok(_) => eprintln!("flush: OK"),
        Err(e) => eprintln!("flush: FAILED — {e}"),
    }

    let session = kb.db().session();

    // CALL uni.fts.query
    let fts_result = session
        .query_with(
            "CALL uni.fts.query('Message', 'content', $q, 1) \
             YIELD node, score RETURN score",
        )
        .param("q", "banker")
        .fetch_all()
        .await
        .unwrap();

    let fts_score: f64 = fts_result.rows()[0].get("score").unwrap();
    eprintln!("fts.query score: {fts_score}");

    // similar_to fulltext only
    let sim_result = session
        .query_with(
            "MATCH (m:Message) \
             RETURN similar_to(m.content, $qtxt) AS fts_score \
             ORDER BY fts_score DESC",
        )
        .param("qtxt", "banker")
        .fetch_all()
        .await
        .unwrap();

    let sim_fts: f64 = sim_result.rows()[0].get("fts_score").unwrap();
    eprintln!("similar_to fts score: {sim_fts}");

    assert!(
        sim_fts > 0.0,
        "similar_to on fulltext should be positive (fts.query={fts_score}), got {sim_fts}"
    );
}
