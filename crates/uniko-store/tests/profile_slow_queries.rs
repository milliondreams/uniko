//! Profile our top slow queries against an existing LoCoMo KB.
//!
//! Picks real node IDs from a previously-ingested conversation and
//! runs `session.profile()` on each query shape we know is hot in our
//! pipeline. Reports `total_time_ms`, `peak_memory_bytes`, per-operator
//! stats, and any planner warnings/suggestions.
//!
//! Write queries (e.g. `CREATE (a)-[r:TYPE]->(b)`) are converted to
//! their read-only equivalent (`RETURN id(a), id(b)`) so the profile
//! is non-destructive — the dominant cost we're measuring is in the
//! `UNWIND … MATCH … MATCH …` pipeline.
//!
//! Run:
//! ```
//! cargo test --release -p uniko-store --test profile_slow_queries -- \
//!   --ignored --nocapture
//! ```
//!
//! Requires `/tmp/locomo-minilm-v4-kb/conv-44` to exist.

use std::collections::HashMap;
use std::path::PathBuf;

use uni_db::api::schema::EmbeddingCfg;
use uni_db::{
    IndexType, ModelAliasSpec, ModelTask, Uni, Value, VectorAlgo, VectorIndexCfg, VectorMetric,
    WarmupPolicy,
};

const KB_PATH: &str = "/tmp/locomo-minilm-v4-kb/conv-44";
const SAMPLE_QUERY_TEXT: &str = "What did Caroline say about her family?";

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

#[allow(dead_code)]
fn auto_embed_index() -> IndexType {
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
    })
}

async fn open_kb() -> Uni {
    let path = PathBuf::from(KB_PATH);
    assert!(
        path.exists(),
        "KB not found at {KB_PATH}. Re-run the LoCoMo ingest first."
    );
    Uni::open(path.to_string_lossy())
        .xervo_catalog(minilm_catalog())
        .build()
        .await
        .expect("open existing KB with xervo catalog")
}

/// Pick `limit` node ids for a given label.
async fn sample_node_ids(db: &Uni, label: &str, limit: usize) -> Vec<i64> {
    let session = db.session();
    let cypher = format!("MATCH (n:{label}) RETURN id(n) AS vid LIMIT {limit}");
    let result = session
        .query_with(&cypher)
        .fetch_all()
        .await
        .expect("sample query");
    result
        .rows()
        .iter()
        .map(|r| r.get::<i64>("vid").unwrap())
        .collect()
}

/// Pick a real participant name to feed entity-scoped recall.
async fn sample_participant_name(db: &Uni) -> Option<String> {
    let session = db.session();
    let result = session
        .query_with("MATCH (p:Participant) WHERE p.name IS NOT NULL RETURN p.name AS name LIMIT 1")
        .fetch_all()
        .await
        .ok()?;
    result
        .rows()
        .first()
        .and_then(|r| r.get::<String>("name").ok())
}

/// Embed a query string via the xervo catalog so we have a real $qvec.
async fn embed_query(db: &Uni, text: &str) -> Vec<f32> {
    let xervo = db.xervo();
    if !xervo.is_available() {
        panic!("xervo runtime not available — was the catalog wired?");
    }
    let v = xervo
        .embed("embed/default", &[text])
        .await
        .expect("embed query");
    v.into_iter().next().expect("one embedding")
}

/// Render a ProfileOutput nicely.
fn print_profile(name: &str, profile: &uni_db::ProfileOutput) {
    println!("\n────────────────────────────────────────────────────");
    println!("{name}");
    println!("────────────────────────────────────────────────────");
    println!("  total_time_ms   = {}", profile.total_time_ms);
    println!(
        "  peak_memory_KB  = {:.1}",
        profile.peak_memory_bytes as f64 / 1024.0
    );
    println!(
        "  cost.estimated_rows = {}",
        profile.explain.cost_estimates.estimated_rows
    );
    if !profile.explain.warnings.is_empty() {
        println!("  warnings:");
        for w in &profile.explain.warnings {
            println!("    - {w}");
        }
    }
    if !profile.explain.suggestions.is_empty() {
        println!("  suggestions:");
        for s in &profile.explain.suggestions {
            println!(
                "    - index ({}, {}, {}): {}",
                s.label_or_type, s.property, s.index_type, s.reason
            );
        }
    }
    if !profile.explain.index_usage.is_empty() {
        println!("  index_usage:");
        for ix in &profile.explain.index_usage {
            // IndexUsage fields vary; print Debug for safety.
            println!("    - {ix:?}");
        }
    }
    println!("  operators (post-order, leaves first):");
    println!(
        "      {:<28} {:>8} {:>10} {:>12}",
        "operator", "rows", "time_ms", "memory_KB"
    );
    for op in &profile.runtime_stats {
        println!(
            "      {:<28} {:>8} {:>10.2} {:>12.1}",
            op.operator,
            op.actual_rows,
            op.time_ms,
            op.memory_bytes as f64 / 1024.0
        );
    }
}

#[tokio::test]
#[ignore]
async fn profile_top_slow_queries() {
    let db = open_kb().await;

    // Sample real node IDs we can plug into MATCH WHERE id() = …
    let obs_ids = sample_node_ids(&db, "Observation", 3).await;
    let msg_ids = sample_node_ids(&db, "Message", 3).await;
    let chunk_ids = sample_node_ids(&db, "Chunk", 3).await;
    let entity_ids = sample_node_ids(&db, "Entity", 3).await;
    let participant_name = sample_participant_name(&db)
        .await
        .unwrap_or_else(|| "unknown".to_string());

    println!(
        "Sampled IDs: {} obs, {} msg, {} chunks, {} entities; participant=\"{}\"",
        obs_ids.len(),
        msg_ids.len(),
        chunk_ids.len(),
        entity_ids.len(),
        participant_name
    );
    assert!(
        !obs_ids.is_empty() && !msg_ids.is_empty(),
        "need at least one Observation and Message in the KB"
    );

    let qvec = embed_query(&db, SAMPLE_QUERY_TEXT).await;
    println!("Embedded query (dim={})", qvec.len());

    let session = db.session();

    // ───────────────────────────────────────────────────────
    // Query A: OBSERVED_IN write-path read-only equivalent.
    // Same UNWIND … MATCH … MATCH shape as our hot batch_create_edges.
    // ───────────────────────────────────────────────────────
    {
        let edges: Vec<Value> = obs_ids
            .iter()
            .zip(msg_ids.iter().cycle())
            .map(|(&o, &m)| {
                let mut h = HashMap::new();
                h.insert("src".into(), Value::Int(o));
                h.insert("dst".into(), Value::Int(m));
                Value::Map(h)
            })
            .collect();
        let cypher = "UNWIND $edges AS e \
                      MATCH (a:Observation) WHERE id(a) = e.src \
                      MATCH (b:Message) WHERE id(b) = e.dst \
                      RETURN id(a) AS sa, id(b) AS sb";
        let (_, profile) = session
            .query_with(cypher)
            .param("edges", Value::List(edges))
            .profile()
            .await
            .expect("profile A");
        print_profile(
            "A. OBSERVED_IN match path  (UNWIND + 2x labelled MATCH by id)",
            &profile,
        );
    }

    // ───────────────────────────────────────────────────────
    // Query B: ABOUT write-path — only source labelled because
    // uni-db doesn't accept disjunction `:Participant|Entity` (#56).
    // ───────────────────────────────────────────────────────
    {
        let edges: Vec<Value> = obs_ids
            .iter()
            .zip(entity_ids.iter().chain(std::iter::repeat(&entity_ids[0])))
            .map(|(&o, &e)| {
                let mut h = HashMap::new();
                h.insert("src".into(), Value::Int(o));
                h.insert("dst".into(), Value::Int(e));
                Value::Map(h)
            })
            .take(3)
            .collect();
        let cypher = "UNWIND $edges AS e \
                      MATCH (a:Observation) WHERE id(a) = e.src \
                      MATCH (b) WHERE id(b) = e.dst \
                      RETURN id(a) AS sa, id(b) AS sb";
        let (_, profile) = session
            .query_with(cypher)
            .param("edges", Value::List(edges))
            .profile()
            .await
            .expect("profile B");
        print_profile(
            "B. ABOUT match path        (UNWIND + labelled src + UNLABELLED dst MATCH)",
            &profile,
        );
    }

    // ───────────────────────────────────────────────────────
    // Query C: HAS_CHUNK write-path — Message → Chunk.
    // ───────────────────────────────────────────────────────
    {
        let edges: Vec<Value> = msg_ids
            .iter()
            .zip(chunk_ids.iter())
            .map(|(&m, &c)| {
                let mut h = HashMap::new();
                h.insert("src".into(), Value::Int(m));
                h.insert("dst".into(), Value::Int(c));
                Value::Map(h)
            })
            .collect();
        let cypher = "UNWIND $edges AS e \
                      MATCH (a:Message) WHERE id(a) = e.src \
                      MATCH (b:Chunk) WHERE id(b) = e.dst \
                      RETURN id(a) AS sa, id(b) AS sb";
        let (_, profile) = session
            .query_with(cypher)
            .param("edges", Value::List(edges))
            .profile()
            .await
            .expect("profile C");
        print_profile(
            "C. HAS_CHUNK match path    (UNWIND + 2x labelled MATCH by id)",
            &profile,
        );
    }

    // ───────────────────────────────────────────────────────
    // Query D: hybrid similar_to recall over session Chunks.
    // This is the dominant query at retrieval time.
    // ───────────────────────────────────────────────────────
    {
        // LIMIT must be a literal integer in this Cypher dialect.
        let cypher = "MATCH (m:Chunk) WHERE m.chunk_type = 'session' \
                      RETURN id(m) AS nid, m.text AS content, \
                             similar_to([m.embedding, m.text], [$qvec, $qtxt], \
                                        {method: 'weighted', weights: [0.5, 0.5]}) AS score \
                      ORDER BY score DESC LIMIT 10";
        let (_, profile) = session
            .query_with(cypher)
            .param("qvec", Value::Vector(qvec.clone()))
            .param("qtxt", Value::String(SAMPLE_QUERY_TEXT.into()))
            .profile()
            .await
            .expect("profile D");
        print_profile(
            "D. Recall hybrid           (session Chunks, similar_to vec+fts, ORDER BY)",
            &profile,
        );
    }

    // ───────────────────────────────────────────────────────
    // Query E: entity-scoped recall — chunk via traversal pattern.
    // ───────────────────────────────────────────────────────
    {
        let cypher = "MATCH (m:Chunk) \
                      WHERE (m)<-[:HAS_CHUNK]-(:Session)<-[:PARTICIPATED_IN]-(:Participant {name: $ename}) \
                      RETURN id(m) AS nid, m.text AS content, \
                             similar_to([m.embedding, m.text], [$qvec, $qtxt]) AS score \
                      ORDER BY score DESC LIMIT 10";
        let (_, profile) = session
            .query_with(cypher)
            .param("ename", Value::String(participant_name.clone()))
            .param("qvec", Value::Vector(qvec.clone()))
            .param("qtxt", Value::String(SAMPLE_QUERY_TEXT.into()))
            .profile()
            .await
            .expect("profile E");
        print_profile(
            "E. Recall entity-scoped    (Chunk via HAS_CHUNK ← Session ← PARTICIPATED_IN)",
            &profile,
        );
    }

    db.shutdown().await.unwrap();
}
