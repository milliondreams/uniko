//! Repro: a Locy rule's `KEY` columns project as NULL on `QUERY … RETURN` when
//! the rule has no `FOLD`. Adding a trivial `FOLD` makes the values appear.
//!
//! Real-world impact: uniko's `contradiction_detector` yielded
//! `KEY f.fact_id AS stale_fact, KEY e.episode_id AS episode_id` and both came
//! back NULL; adding `FOLD cnt = COUNT(e)` fixed it.

use uni_db::{DataType, Uni, Value};

async fn setup() -> Uni {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Item").property("tag", DataType::String).done()
        .label("Owner").property("oid", DataType::String).done()
        .edge_type("OWNED_BY", &["Item"], &["Owner"]).done()
        .apply().await.unwrap();
    let s = db.session();
    let tx = s.tx().await.unwrap();
    tx.execute("CREATE (:Owner {oid: 'o1'})").await.unwrap();
    tx.execute("MATCH (o:Owner {oid:'o1'}) CREATE (i:Item {tag:'a'})-[:OWNED_BY]->(o)").await.unwrap();
    tx.execute("MATCH (o:Owner {oid:'o1'}) CREATE (i:Item {tag:'b'})-[:OWNED_BY]->(o)").await.unwrap();
    tx.commit().await.unwrap();
    db
}

async fn first_tag(db: &Uni, program: &str, ret: &str) -> Option<Value> {
    let s = db.session();
    let r = s.locy_with(program).run().await.unwrap();
    r.rows().and_then(|rows| rows.first().and_then(|row| row.get(ret).cloned()))
}

// Fixed in uni-db 2.4.1 (rustic-ai/uni-db#112); kept as a passing regression guard.
#[tokio::test]
async fn single_node_key_without_fold() {
    let db = setup().await;
    db.rules().register("CREATE RULE sn AS MATCH (i:Item) YIELD KEY i.tag AS tag").await.unwrap();
    let v = first_tag(&db, "QUERY sn RETURN tag", "tag").await;
    eprintln!("PROBE single-node no-FOLD tag = {v:?}");
    assert!(matches!(v, Some(Value::String(_))), "KEY tag should be a String, got {v:?}");
    db.shutdown().await.unwrap();
}

// Fixed in uni-db 2.4.1 (rustic-ai/uni-db#112); kept as a passing regression guard.
#[tokio::test]
async fn multi_pattern_key_without_fold() {
    let db = setup().await;
    db.rules().register(
        "CREATE RULE mp AS MATCH (i:Item)-[:OWNED_BY]->(o:Owner {oid:'o1'}) YIELD KEY i.tag AS tag, KEY o.oid AS oid",
    ).await.unwrap();
    let v = first_tag(&db, "QUERY mp RETURN tag, oid", "tag").await;
    eprintln!("PROBE multi-pattern no-FOLD tag = {v:?}");
    assert!(matches!(v, Some(Value::String(_))), "KEY tag should be a String, got {v:?}");
    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn multi_pattern_key_with_fold_works() {
    let db = setup().await;
    db.rules().register(
        "CREATE RULE mpf AS MATCH (i:Item)-[:OWNED_BY]->(o:Owner {oid:'o1'}) FOLD cnt = COUNT(i) YIELD KEY i.tag AS tag, KEY o.oid AS oid, cnt AS cnt",
    ).await.unwrap();
    let v = first_tag(&db, "QUERY mpf RETURN tag, oid, cnt", "tag").await;
    eprintln!("PROBE multi-pattern WITH-FOLD tag = {v:?}");
    assert!(matches!(v, Some(Value::String(_))), "with FOLD, KEY tag should be a String, got {v:?}");
    db.shutdown().await.unwrap();
}
