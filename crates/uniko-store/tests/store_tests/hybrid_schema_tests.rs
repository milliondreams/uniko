//! Tests for the single-pass hybrid (dense + sparse + ColBERT) embedder
//! wiring: catalog task selection, schema registration of the extra
//! columns, and config validation.
//!
//! These cover the uniko-side wiring without downloading BGE-M3 — the
//! end-to-end auto-embed mechanics (one `EmbedHybrid` pass filling all three
//! columns, sparse/MaxSim querying) are proven by uni-db's own
//! `hybrid_autoembed` / `sparse_autoembed` / `multivector_*` tests.

use uni_db::{ModelTask, Uni};
use uniko_store::config::{EmbeddingConfig, RerankerConfig, UnikoConfig};
use uniko_store::schema::register_schema;
use uniko_store::schema::{EMBED_ALIAS, HYBRID_EMBED_ALIAS, RERANK_ALIAS};
use uniko_store::storage::embed_catalog;

/// A config whose embedder is the single-pass hybrid BGE-M3 preset.
fn hybrid_config() -> UnikoConfig {
    UnikoConfig {
        embedding: EmbeddingConfig::bge_m3(),
        ..UnikoConfig::default()
    }
}

// ── Catalog task selection ──

#[test]
fn dense_embedder_registers_only_embed_alias() {
    let config = UnikoConfig::default();
    let catalog = embed_catalog(&config);
    let embed = catalog
        .iter()
        .find(|s| s.alias == EMBED_ALIAS)
        .expect("embed alias present");
    assert_eq!(
        embed.task,
        ModelTask::Embed,
        "dense embedder stays on Embed"
    );
    assert!(
        catalog.iter().all(|s| s.alias != HYBRID_EMBED_ALIAS),
        "dense-only embedder must not register a hybrid alias"
    );
}

#[test]
fn hybrid_embedder_registers_both_aliases() {
    let config = hybrid_config();
    let catalog = embed_catalog(&config);
    // The dense alias stays `Embed` — it backs lone-dense columns (Message),
    // computed embeds, and queries. A hybrid model can't serve those, so a
    // SEPARATE `EmbedHybrid` alias is added for the Chunk/Observation group.
    let embed = catalog
        .iter()
        .find(|s| s.alias == EMBED_ALIAS)
        .expect("dense embed alias present");
    assert_eq!(
        embed.task,
        ModelTask::Embed,
        "dense alias must stay Embed even when a hybrid embedder is configured"
    );
    let hybrid = catalog
        .iter()
        .find(|s| s.alias == HYBRID_EMBED_ALIAS)
        .expect("hybrid alias present for a hybrid embedder");
    assert_eq!(hybrid.task, ModelTask::EmbedHybrid);
    assert_eq!(
        hybrid.model_id, embed.model_id,
        "both aliases back the same model"
    );
}

#[test]
fn colbert_reranker_style_registers_no_rerank_alias() {
    let config = UnikoConfig {
        embedding: EmbeddingConfig::bge_m3(),
        reranker: RerankerConfig {
            enabled: true,
            style: "colbert".to_string(),
            ..RerankerConfig::default()
        },
        ..UnikoConfig::default()
    };
    let catalog = embed_catalog(&config);
    assert!(
        catalog.iter().all(|s| s.alias != RERANK_ALIAS),
        "colbert rerank is in-process MaxSim — it must not register a rerank model"
    );
}

#[test]
fn cross_encoder_reranker_still_registers_alias() {
    let config = UnikoConfig {
        embedding: EmbeddingConfig::bge_m3(),
        reranker: RerankerConfig {
            enabled: true,
            ..RerankerConfig::default()
        },
        ..UnikoConfig::default()
    };
    let catalog = embed_catalog(&config);
    assert!(
        catalog.iter().any(|s| s.alias == RERANK_ALIAS),
        "cross-encoder style must register the rerank alias"
    );
}

// ── Schema registration ──

#[tokio::test]
async fn hybrid_schema_registers_and_is_idempotent() {
    let config = hybrid_config();
    // The embed catalog references BGE-M3 with WarmupPolicy::Lazy, so the
    // model is never loaded here; registering the sparse + multi-vector
    // indexes exercises the schema builders without any download.
    let db = Uni::in_memory()
        .xervo_catalog(embed_catalog(&config))
        .build()
        .await
        .expect("in-memory db");
    register_schema(&db, &config)
        .await
        .expect("hybrid schema registration");
    register_schema(&db, &config)
        .await
        .expect("hybrid schema registration is idempotent");
    assert!(db.label_exists("Chunk").await.unwrap());
    assert!(db.label_exists("Observation").await.unwrap());
    db.shutdown().await.unwrap();
}

// ── Config validation ──

#[test]
fn validate_rejects_sparse_channel_without_hybrid_embedder() {
    let config = UnikoConfig {
        recall_sparse_enabled: true,
        ..UnikoConfig::default() // bge-small, no sparse dims
    };
    assert!(
        config.validate().is_err(),
        "sparse channel without a hybrid embedder must fail validation"
    );
}

#[test]
fn validate_rejects_colbert_style_without_multivector() {
    let config = UnikoConfig {
        reranker: RerankerConfig {
            enabled: true,
            style: "colbert".to_string(),
            ..RerankerConfig::default()
        },
        ..UnikoConfig::default() // bge-small, no multivector dims
    };
    assert!(
        config.validate().is_err(),
        "colbert rerank without a multi-vector embedder must fail validation"
    );
}

#[test]
fn validate_accepts_hybrid_sparse_and_colbert() {
    let config = UnikoConfig {
        embedding: EmbeddingConfig::bge_m3(),
        recall_sparse_enabled: true,
        reranker: RerankerConfig {
            enabled: true,
            style: "colbert".to_string(),
            ..RerankerConfig::default()
        },
        ..UnikoConfig::default()
    };
    config
        .validate()
        .expect("hybrid embedder with sparse + colbert must validate");
}
