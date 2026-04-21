//! Reproduction test for rustic-ai/uni-db#40:
//! flush() fails with non-nullable column on empty labels.

use uni_db::{DataType, Uni};

/// Test with minimal schema — exactly as filed in #40.
#[tokio::test]
async fn flush_minimal_schema() {
    let db = Uni::in_memory().build().await.unwrap();

    db.schema()
        .label("Session")
        .property("session_id", DataType::String)
        .property("started_at", DataType::DateTime)
        .done()
        .apply()
        .await
        .unwrap();

    let session = db.session();
    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Message {content: 'hello', timestamp: datetime()})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    match db.flush().await {
        Ok(_) => eprintln!("flush: OK"),
        Err(e) => {
            eprintln!("flush: FAILED — {e}");
            panic!("flush should not fail: {e}");
        }
    }

    db.shutdown().await.unwrap();
}

/// Test with full uniko schema — the actual context where we hit this.
#[tokio::test]
async fn flush_full_uniko_schema() {
    use uni_db::ModelAliasSpec;
    use uniko_store::KnowledgeBase;
    use uniko_store::config::UnikoConfig;

    let config = UnikoConfig::default();
    let kb = KnowledgeBase::in_memory_with_xervo(config, Vec::<ModelAliasSpec>::new())
        .await
        .unwrap();

    // Insert one Message — other labels (Session, Entity, etc.) have 0 rows.
    let session = kb.db().session();
    let tx = session.tx().await.unwrap();
    tx.execute("CREATE (:Message {message_id: 'm1', content: 'hello', timestamp: datetime()})")
        .await
        .unwrap();
    tx.commit().await.unwrap();

    match kb.db().flush().await {
        Ok(_) => eprintln!("flush full schema: OK"),
        Err(e) => {
            eprintln!("flush full schema: FAILED — {e}");
            panic!("flush should not fail: {e}");
        }
    }
}
