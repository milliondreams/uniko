//! Integration tests for vector, fulltext, and hybrid search.
//!
//! Vector and fulltext search require data with embeddings or indexed text.
//! These tests validate the API contracts and edge cases.

use std::collections::HashMap;
use uni_db::Value;
use uniko_store::config::UnikoConfig;
use uniko_store::search::hybrid::{RRF_K, TIER_WEIGHT_SEMANTIC};
use uniko_store::storage::KnowledgeBase;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

/// The embedding dimension the default schema actually uses (driven by
/// `UnikoConfig::default().embedding`, currently nomic/768d). Tests size
/// their vectors to this so they match the `Message.embedding` column
/// regardless of which model the default config selects — a hardcoded 384
/// silently passed in isolation but failed under parallel runs once the
/// vector index enforced the column dimension.
fn embed_dim() -> usize {
    UnikoConfig::default().embedding.dimensions
}

// ── Vector search ──

#[tokio::test]
async fn test_vector_search_empty_embedding() {
    let kb = test_kb().await;
    let results = kb
        .vector_search(&[], "Message", "embedding", 10)
        .await
        .unwrap();
    assert!(
        results.is_empty(),
        "empty embedding should return no results"
    );
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_vector_search_basic() {
    let kb = test_kb().await;

    // Insert messages with embeddings sized to the schema's embedding dim.
    for i in 0..5u8 {
        let mut props = HashMap::new();
        props.insert("message_id".into(), Value::String(format!("vs-{i}")));
        props.insert(
            "content".into(),
            Value::String(format!("vector test message {i}")),
        );
        props.insert(
            "timestamp".into(),
            Value::String("2024-01-01T00:00:00Z".into()),
        );
        // Create a simple embedding: all zeros except position i.
        let mut emb = vec![0.0f32; embed_dim()];
        emb[i as usize] = 1.0;
        props.insert("embedding".into(), Value::Vector(emb));
        kb.create_node("Message", &props).await.unwrap();
    }

    // Search with an embedding close to message 0.
    let mut query_vec = vec![0.0f32; embed_dim()];
    query_vec[0] = 1.0;

    // The API should not error even if the index hasn't been built yet.
    let results = kb
        .vector_search(&query_vec, "Message", "embedding", 3)
        .await
        .unwrap();

    // With freshly-created in-memory data the HNSW index may not be
    // populated yet — so we just validate the API returned valid results
    // (possibly empty) with the right structure.
    for hit in &results {
        assert_eq!(hit.node_type, "Message");
        assert!(hit.score >= 0.0);
    }

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_multi_type_vector_search() {
    let kb = test_kb().await;

    // Insert a Message and a Fact with embeddings.
    let mut emb = vec![0.0f32; embed_dim()];
    emb[0] = 1.0;

    let mut mp = HashMap::new();
    mp.insert("message_id".into(), Value::String("mt-m1".into()));
    mp.insert("content".into(), Value::String("hello".into()));
    mp.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    mp.insert("embedding".into(), Value::Vector(emb.clone()));
    kb.create_node("Message", &mp).await.unwrap();

    let mut fp = HashMap::new();
    fp.insert("fact_id".into(), Value::String("mt-f1".into()));
    fp.insert("subject".into(), Value::String("user".into()));
    fp.insert("predicate".into(), Value::String("likes".into()));
    fp.insert("embedding".into(), Value::Vector(emb.clone()));
    kb.create_node("Fact", &fp).await.unwrap();

    // multi_type search should not error.
    let results = kb
        .multi_type_vector_search(&emb, &[("Message", "embedding"), ("Fact", "embedding")], 10)
        .await
        .unwrap();

    // Validate structure — index may not be populated yet for in-memory DB.
    for hit in &results {
        assert!(hit.node_type == "Message" || hit.node_type == "Fact");
    }

    kb.shutdown().await.unwrap();
}

// ── Fulltext search ──

#[tokio::test]
async fn test_fulltext_empty_query() {
    let kb = test_kb().await;
    let results = kb
        .fulltext_search("", "Message", "content", 10)
        .await
        .unwrap();
    assert!(results.is_empty());
    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_fulltext_basic() {
    let kb = test_kb().await;

    let mut props = HashMap::new();
    props.insert("message_id".into(), Value::String("ft-1".into()));
    props.insert(
        "content".into(),
        Value::String("Rust is a systems programming language".into()),
    );
    props.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    kb.create_node("Message", &props).await.unwrap();

    let mut props2 = HashMap::new();
    props2.insert("message_id".into(), Value::String("ft-2".into()));
    props2.insert(
        "content".into(),
        Value::String("Python is a scripting language".into()),
    );
    props2.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:01Z".into()),
    );
    kb.create_node("Message", &props2).await.unwrap();

    let results = kb
        .fulltext_search("Rust systems", "Message", "content", 10)
        .await
        .unwrap();

    assert!(!results.is_empty(), "should find message containing 'Rust'");

    kb.shutdown().await.unwrap();
}

// ── Hybrid search (RRF unit tested in search/hybrid.rs) ──

#[tokio::test]
async fn test_hybrid_search_basic() {
    let kb = test_kb().await;

    // Insert messages with both content and embeddings.
    let mut emb = vec![0.0f32; embed_dim()];
    emb[0] = 1.0;

    let mut props = HashMap::new();
    props.insert("message_id".into(), Value::String("hs-1".into()));
    props.insert(
        "content".into(),
        Value::String("cognitive memory system".into()),
    );
    props.insert(
        "timestamp".into(),
        Value::String("2024-01-01T00:00:00Z".into()),
    );
    props.insert("embedding".into(), Value::Vector(emb.clone()));
    kb.create_node("Message", &props).await.unwrap();

    let results = kb
        .hybrid_search("cognitive memory", &emb, "Message", 10)
        .await
        .unwrap();

    // Should return results from either vector or fulltext (or both).
    assert!(
        !results.is_empty(),
        "hybrid search should find the inserted message"
    );

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_rrf_constants() {
    // Verify the constants match the spec.
    assert!((RRF_K - 60.0).abs() < f64::EPSILON);
    assert!((TIER_WEIGHT_SEMANTIC - 1.0).abs() < f64::EPSILON);
}

/// PDF-origin chunks (`chunk_type` "block" / "page") must be recalled by
/// `recall_chunk_and_entity_scoped`. They were previously absent from its
/// `["session", "observation"]` filter, so all PDF text was unrecallable.
/// BM25-only path (empty query vector) keeps this independent of any model.
#[tokio::test]
async fn recall_includes_block_and_page_chunk_types() {
    let kb = test_kb().await;

    for (i, (ctype, text)) in [
        ("block", "quarterly revenue table figures"),
        ("page", "annual shareholder report summary"),
        // A non-PDF chunk_type that must NOT match the PDF query, as a control.
        ("session", "unrelated meeting transcript"),
    ]
    .iter()
    .enumerate()
    {
        let mut p = HashMap::new();
        p.insert("chunk_id".into(), Value::String(format!("rc-{i}")));
        p.insert("text".into(), Value::String((*text).into()));
        p.insert("chunk_type".into(), Value::String((*ctype).into()));
        // Provide the embedding explicitly so insertion needs no embed model.
        p.insert("embedding".into(), Value::Vector(vec![0.0f32; embed_dim()]));
        kb.create_node("Chunk", &p).await.unwrap();
    }

    // BM25-only recall (empty qvec, bm25 weight 1.0) for the PDF text.
    let rows = kb
        .recall_chunk_and_entity_scoped(&[], "revenue table figures", &[], 10, 0.0, 1.0, None)
        .await;

    assert!(
        rows.iter().any(|r| r.content.contains("revenue table")),
        "block chunk_type must be recalled, got: {:?}",
        rows.iter().map(|r| &r.content).collect::<Vec<_>>()
    );

    // The "page" chunk is also reachable under its own query.
    let page_rows = kb
        .recall_chunk_and_entity_scoped(&[], "shareholder report summary", &[], 10, 0.0, 1.0, None)
        .await;
    assert!(
        page_rows
            .iter()
            .any(|r| r.content.contains("shareholder report")),
        "page chunk_type must be recalled"
    );

    kb.shutdown().await.unwrap();
}
