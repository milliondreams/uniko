//! Repro: label disjunction `(b:DestA|DestB)` panics in DataFusion's
//! `union_schema` when the two labels have different property counts.
//!
//! `rustic-ai/uni-db#56` enabled label disjunction parsing, but the
//! resulting plan constructs a `UnionExec` over the two label scans.
//! When the two branches' physical schemas have different column
//! counts, `datafusion_physical_plan::union::union_schema` panics
//! with `index out of bounds: the len is N but the index is N`
//! (in `arrow-schema-57.3.0/src/schema.rs:375`).
//!
//! Real-world impact: in `uniko-extract` the ABOUT edge has dst =
//! `Entity` OR `Participant`. `Entity` and `Participant` carry
//! different property sets, so any attempt to use
//! `(b:Entity|Participant)` as a MATCH target panics during plan
//! execution. We currently work around by leaving dst unlabelled,
//! losing the planner's narrow-scan benefit (~2× on the LoCoMo
//! workload per the original #56 numbers).
//!
//! Expected behavior: either succeed (UNION the two schemas to
//! their column-name union, padding missing columns with NULL) or
//! return a typed error before reaching DataFusion.

use uni_db::{DataType, Uni, Value};

async fn setup_db_asymmetric_schemas() -> Uni {
    let db = Uni::in_memory().build().await.unwrap();
    // All non-name properties are nullable so the test fixture stays
    // valid as uni-db tightens NOT NULL enforcement; the disjunction
    // bug being reproduced is about column-count asymmetry, not
    // nullability.
    db.schema()
        .label("Source")
        .property("name", DataType::String)
        .done()
        // DestA has 2 properties.
        .label("DestA")
        .property("name", DataType::String)
        .property_nullable("category", DataType::String)
        .done()
        // DestB has 4 properties (different count from DestA).
        .label("DestB")
        .property("name", DataType::String)
        .property_nullable("kind", DataType::String)
        .property_nullable("first_seen", DataType::String)
        .property_nullable("last_seen", DataType::String)
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
async fn disjunction_panics_when_branch_schemas_differ() {
    let db = setup_db_asymmetric_schemas().await;

    let src = create_node(&db, "Source", "src").await;
    let dst_b = create_node(&db, "DestB", "b").await;

    // Try to match a DestB target via label disjunction. With #56's
    // parsing fix this *should* succeed, but it panics during
    // physical plan execution because the two branches' schemas
    // have 2 vs 4 property columns.
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

    // Document what happens. Originally this panicked inside
    // DataFusion's `union_schema` — uni-db has since fixed the bug,
    // so the test now also verifies that the disjunction actually
    // matched `:DestB` and the edge was created.
    eprintln!("execute result: {result:?}");
    result.expect("disjunction execute should succeed");
    tx.commit().await.expect("tx commit");

    // If we ever reach this line, the panic was fixed and we
    // should also see the edge created.
    let counted = db
        .session()
        .query_with("MATCH (a:Source)-[r:LINK]->(b:DestB) RETURN count(r) AS c")
        .fetch_all()
        .await
        .unwrap();
    let c: i64 = counted.rows().first().unwrap().get("c").unwrap_or(0);
    assert_eq!(
        c, 1,
        "label disjunction (:DestA|DestB) should have matched :DestB and created the edge",
    );

    db.shutdown().await.unwrap();
}
