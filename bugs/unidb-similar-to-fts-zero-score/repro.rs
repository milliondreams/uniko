//! Reproduction test: similar_to() on fulltext-indexed field returns 0.0
//!
//! ## Summary
//! `similar_to(m.content, 'keyword')` returns 0.0 for a fulltext-indexed
//! String property, even when `CALL uni.fts.query()` on the same field
//! and query returns correct positive BM25 scores.
//!
//! ## Expected Behavior
//! `similar_to(m.content, 'banker')` should return a positive score
//! (consistent with `CALL uni.fts.query`) when the node's content
//! property contains the word "banker".
//!
//! ## Actual Behavior
//! `similar_to()` returns 0.0 for all nodes, while `CALL uni.fts.query()`
//! returns correct scores (e.g., 0.5).
//!
//! ## Environment
//! - uni-db: local build from source (version 1.1.0)
//! - Schema: label "Message" with property "content" (String) + FullText index
//! - Tested both with and without `db.flush().await` after insert
//!
//! ## Note
//! The existing `test_similar_to_fts` test in uni-db's test suite passes.
//! The difference may be related to:
//! - Schema registration method (builder.label().property().index() vs other)
//! - The presence of additional indexes (vector + fulltext on same label)
//! - Auto-embed configuration on the vector index
//!
//! ## To Run
//! Add this as a test file in `uni/crates/uni/tests/` and run:
//!   cargo test --test repro -- --nocapture

use uni_db::{DataType, IndexType, Uni};

#[tokio::test]
async fn similar_to_fts_returns_zero_on_message_schema() {
    // 1. Create in-memory DB
    let db = Uni::in_memory().build().await.unwrap();

    // 2. Register schema: String property with FullText index
    //    (mimics the Message node type in uniko)
    db.schema()
        .label("Message")
        .property("message_id", DataType::String)
        .property("content", DataType::String)
        .property("timestamp", DataType::DateTime)
        .index("content", IndexType::FullText)
        .done()
        .apply()
        .await
        .unwrap();

    // 3. Insert test data
    let session = db.session();
    let tx = session.tx().await.unwrap();
    tx.execute(
        "CREATE (:Message {message_id: 'm1', content: 'Lost my job as a banker yesterday', timestamp: datetime()})",
    )
    .await
    .unwrap();
    tx.execute(
        "CREATE (:Message {message_id: 'm2', content: 'Starting my own dance studio next week', timestamp: datetime()})",
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    // 4. Flush to ensure indexes are up to date
    db.flush().await.unwrap();

    // 5. CALL uni.fts.query — works correctly (positive scores)
    let fts_result = session
        .query_with(
            "CALL uni.fts.query('Message', 'content', $q, 3) \
             YIELD node, score \
             RETURN node.content AS content, score",
        )
        .param("q", "banker")
        .fetch_all()
        .await
        .unwrap();

    assert!(
        !fts_result.is_empty(),
        "fts.query should find at least 1 result for 'banker'"
    );
    let fts_score: f64 = fts_result.rows()[0].get("score").unwrap();
    assert!(
        fts_score > 0.0,
        "fts.query should return positive score, got {fts_score}"
    );
    eprintln!("PASS: CALL uni.fts.query score = {fts_score}");

    // 6. similar_to on same field — returns 0.0 (BUG)
    let sim_result = session
        .query(
            "MATCH (m:Message) \
             RETURN m.content AS content, \
                    similar_to(m.content, 'banker') AS score \
             ORDER BY score DESC",
        )
        .await
        .unwrap();

    assert!(!sim_result.is_empty());
    let sim_score: f64 = sim_result.rows()[0].get("score").unwrap();
    eprintln!("similar_to score = {sim_score}");

    // This assertion FAILS — similar_to returns 0.0
    assert!(
        sim_score > 0.0,
        "similar_to(m.content, 'banker') should return positive score \
         (same as fts.query which returned {fts_score}), but got {sim_score}"
    );

    db.shutdown().await.unwrap();
}
