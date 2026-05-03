//! Profile `merge_node`'s underlying lookup query to see if it has the
//! same scan-then-filter shape that `batch_create_edges` had before the
//! IN-list-pushdown fix.
//!
//! `merge_node` does:
//! 1. `MATCH (n:Label {ext_id_field: $eid}) RETURN n, id(n) AS vid`
//! 2. Update if found, else create.
//!
//! Step 1 is the hot path on every entity upsert. With LongMemEval averaging
//! 9.2 entities per turn, ~430ms/upsert call, this is the new bottleneck.
//! We want to see operator-level whether the inline-property MATCH is
//! using the `entity_id` hash index or doing a full label scan.
//!
//! Run:
//! ```
//! cargo test --release -p uniko-store --test profile_merge_node -- \
//!   --ignored --nocapture
//! ```

// Rust guideline compliant

use std::path::PathBuf;

use uni_db::{ModelAliasSpec, ModelTask, Uni, Value, WarmupPolicy};

const KB_PATH: &str = "/tmp/lme-bench-v5-kb/001be529";

fn minilm_catalog() -> Vec<ModelAliasSpec> {
    vec![ModelAliasSpec {
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
    }]
}

async fn open_kb() -> Uni {
    let path = PathBuf::from(KB_PATH);
    assert!(path.exists(), "KB not found at {KB_PATH}");
    Uni::open(path.to_string_lossy())
        .xervo_catalog(minilm_catalog())
        .build()
        .await
        .expect("open existing KB")
}

async fn count_entities(db: &Uni) -> i64 {
    let session = db.session();
    let r = session
        .query_with("MATCH (n:Entity) RETURN count(n) AS c")
        .fetch_all()
        .await
        .unwrap();
    r.rows().first().unwrap().get::<i64>("c").unwrap()
}

/// Pull a few real `entity_id` strings from the KB so we can profile real lookups.
async fn sample_entity_ids(db: &Uni, n: usize) -> Vec<String> {
    let session = db.session();
    let cypher = format!(
        "MATCH (n:Entity) WHERE n.entity_id IS NOT NULL \
         RETURN n.entity_id AS eid LIMIT {n}"
    );
    let r = session.query_with(&cypher).fetch_all().await.unwrap();
    r.rows()
        .iter()
        .map(|row| row.get::<String>("eid").unwrap_or_default())
        .filter(|s| !s.is_empty())
        .collect()
}

fn print_profile(name: &str, p: &uni_db::ProfileOutput) {
    println!("\n────────────────────────────────────────────────────");
    println!("{name}");
    println!("────────────────────────────────────────────────────");
    println!("  total_time_ms       = {}", p.total_time_ms);
    println!("  cost.estimated_rows = {}", p.explain.cost_estimates.estimated_rows);
    if !p.explain.warnings.is_empty() {
        println!("  warnings:");
        for w in &p.explain.warnings {
            println!("    - {w}");
        }
    }
    if !p.explain.suggestions.is_empty() {
        println!("  suggestions:");
        for s in &p.explain.suggestions {
            println!("    - {} {} ({}): {}", s.label_or_type, s.property, s.index_type, s.reason);
        }
    }
    if !p.explain.index_usage.is_empty() {
        println!("  index_usage:");
        for ix in &p.explain.index_usage {
            println!("    - {ix:?}");
        }
    }
    println!("  operators:");
    println!("      {:<30} {:>8} {:>10}", "operator", "rows", "time_ms");
    for op in &p.runtime_stats {
        println!(
            "      {:<30} {:>8} {:>10.2}",
            op.operator, op.actual_rows, op.time_ms
        );
    }
}

#[tokio::test]
#[ignore]
async fn profile_merge_node_lookup() {
    let db = open_kb().await;
    let entity_count = count_entities(&db).await;
    println!("KB has {entity_count} :Entity nodes");

    let ids = sample_entity_ids(&db, 5).await;
    assert!(!ids.is_empty(), "no entities in KB");
    println!("Sampled {} entity_ids: {:?}\n", ids.len(), &ids[..ids.len().min(3)]);

    let session = db.session();

    // ── Query 1: existing-entity lookup (hot path on every upsert) ───
    // This is exactly what merge_node runs first.
    {
        let cypher = "MATCH (n:Entity {entity_id: $eid}) RETURN n, id(n) AS vid";
        let (_, p) = session
            .query_with(cypher)
            .param("eid", Value::String(ids[0].clone()))
            .profile()
            .await
            .expect("profile existing");
        print_profile(
            "1. merge_node lookup — EXISTING entity_id (hot path)",
            &p,
        );
    }

    // ── Query 2: non-existent entity (also hot path — first creation) ───
    {
        let cypher = "MATCH (n:Entity {entity_id: $eid}) RETURN n, id(n) AS vid";
        let (_, p) = session
            .query_with(cypher)
            .param(
                "eid",
                Value::String("__definitely_does_not_exist_12345__".to_string()),
            )
            .profile()
            .await
            .expect("profile missing");
        print_profile(
            "2. merge_node lookup — MISSING entity_id (also hot path)",
            &p,
        );
    }

    // ── Query 3: equivalent rewritten as MATCH ... WHERE n.prop = $eid
    //    (sometimes property-predicate-in-pattern parses differently
    //    than WHERE-clause predicate)
    {
        let cypher = "MATCH (n:Entity) WHERE n.entity_id = $eid RETURN n, id(n) AS vid";
        let (_, p) = session
            .query_with(cypher)
            .param("eid", Value::String(ids[0].clone()))
            .profile()
            .await
            .expect("profile where");
        print_profile(
            "3. WHERE-clause variant of (1) — does the planner pick the index?",
            &p,
        );
    }

    // ── Query 4: count nodes in label to know the scan size for context.
    {
        let (_, p) = session
            .query_with("MATCH (n:Entity) RETURN count(n) AS c")
            .profile()
            .await
            .expect("profile count");
        print_profile("4. Full Entity label scan (reference: scan size)", &p);
    }

    db.shutdown().await.unwrap();
}
