//! Repro: label disjunction `(:Foo|Bar)` in MATCH is silently dropped.
//!
//! In OpenCypher, `(n:Foo|Bar)` matches nodes labelled with EITHER
//! `Foo` OR `Bar`. uni-db's parser appears to silently no-op this
//! pattern: the query parses without error, but the MATCH returns no
//! rows even when matching nodes exist — leading to subsequent CREATE
//! edges being silently skipped (no edges created, no error raised).
//!
//! Real-world impact: in our benchmark we have ABOUT edges going from
//! `:Observation` to either `:Participant` or `:Entity`. Adding the
//! source-only label hint gives ~1.8x speedup; adding both endpoints
//! (when possible, e.g. OBSERVED_IN with single target label) gives
//! ~3.9x. The missing label disjunction support costs us ~2x on this
//! workload.

// Rust guideline compliant

use uni_db::{Uni, Value};

async fn setup_db() -> Uni {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Source")
        .property("name", uni_db::DataType::String)
        .done()
        .label("DestA")
        .property("name", uni_db::DataType::String)
        .done()
        .label("DestB")
        .property("name", uni_db::DataType::String)
        .done()
        .edge_type("LINK", &["Source"], &["DestA", "DestB"])
        .done()
        .apply()
        .await
        .unwrap();
    db
}

async fn create_node(db: &Uni, label: &str, name: &str) -> i64 {
    let session = db.session();
    let tx = session.tx().await.unwrap();
    let cypher = format!("CREATE (n:{label} {{name: $n}}) RETURN id(n) AS vid");
    let result = tx
        .query_with(&cypher)
        .param("n", Value::String(name.to_string()))
        .fetch_all()
        .await
        .unwrap();
    tx.commit().await.unwrap();
    result.rows().first().unwrap().get::<i64>("vid").unwrap()
}

#[tokio::test]
async fn repro_label_disjunction_in_match() {
    let db = setup_db().await;

    let src = create_node(&db, "Source", "src").await;
    let dst_a = create_node(&db, "DestA", "a").await;
    let dst_b = create_node(&db, "DestB", "b").await;

    // 1. Single-label MATCH works.
    {
        let session = db.session();
        let tx = session.tx().await.unwrap();
        tx.execute_with(
            "MATCH (a:Source) WHERE id(a) = $src \
             MATCH (b:DestA) WHERE id(b) = $dst \
             CREATE (a)-[:LINK]->(b)",
        )
        .param("src", src)
        .param("dst", dst_a)
        .run()
        .await
        .unwrap();
        tx.commit().await.unwrap();
    }

    // 2. Label disjunction MATCH should match :DestA OR :DestB.
    //    Per OpenCypher spec this should work.
    {
        let session = db.session();
        let tx = session.tx().await.unwrap();
        let result = tx
            .execute_with(
                "MATCH (a:Source) WHERE id(a) = $src \
                 MATCH (b:DestA|DestB) WHERE id(b) = $dst \
                 CREATE (a)-[:LINK]->(b)",
            )
            .param("src", src)
            .param("dst", dst_b)
            .run()
            .await;
        // Document what happens — we expect this to work and create the edge,
        // OR return a parse error. Currently it appears to silently no-op.
        eprintln!("disjunction execute result: {result:?}");
        tx.commit().await.unwrap();
    }

    // 3. Verify edge counts.
    let session = db.session();
    let counted = session
        .query_with("MATCH (a:Source)-[r:LINK]->(b) RETURN id(b) AS dst, labels(b)[0] AS lbl")
        .fetch_all()
        .await
        .unwrap();

    let mut found_a = false;
    let mut found_b = false;
    for row in counted.rows() {
        let lbl: String = row.get("lbl").unwrap_or_default();
        if lbl == "DestA" {
            found_a = true;
        }
        if lbl == "DestB" {
            found_b = true;
        }
    }

    eprintln!("found edge to :DestA = {found_a}");
    eprintln!("found edge to :DestB = {found_b}");

    assert!(
        found_a,
        "single-label match should have created Source→DestA edge"
    );
    assert!(
        found_b,
        "label disjunction (:DestA|DestB) should have matched :DestB and created the edge — but it did not"
    );

    db.shutdown().await.unwrap();
}

/// Variant: try alternate disjunction syntaxes that some Cypher dialects
/// support. Documents which (if any) work.
#[tokio::test]
async fn repro_label_disjunction_alternate_syntaxes() {
    let db = setup_db().await;
    let src = create_node(&db, "Source", "src").await;
    let dst_b = create_node(&db, "DestB", "b").await;

    let alternatives = [
        // Standard OpenCypher pipe.
        "MATCH (a:Source) WHERE id(a) = $src MATCH (b:DestA|DestB) WHERE id(b) = $dst CREATE (a)-[:LINK]->(b)",
        // GQL / Neo4j 5 colon-pipe style.
        "MATCH (a:Source) WHERE id(a) = $src MATCH (b:DestA|:DestB) WHERE id(b) = $dst CREATE (a)-[:LINK]->(b)",
        // Without source label, only target disjunction.
        "MATCH (a) WHERE id(a) = $src MATCH (b:DestA|DestB) WHERE id(b) = $dst CREATE (a)-[:LINK]->(b)",
    ];

    for (i, cypher) in alternatives.iter().enumerate() {
        let session = db.session();
        let tx = session.tx().await.unwrap();
        let result = tx
            .execute_with(cypher)
            .param("src", src)
            .param("dst", dst_b)
            .run()
            .await;
        eprintln!("syntax {i}: {result:?}");
        let _ = tx.commit().await;
    }

    db.shutdown().await.unwrap();
}
