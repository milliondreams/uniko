/// Runtime configuration for the uniko cognitive memory system.
///
/// All fields have spec-mandated defaults. Use `UnikoConfig::default()` and override
/// individual fields as needed. Call `validate()` before use to catch constraint violations.
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{Result, UnikoError};

// ── Embedding + Vector index config types ─────────────────────────

/// Embedding model selection.
///
/// Controls which ONNX embedding model is used for auto-embedding and
/// computed embeddings.  Dimensions must match the model's output.
///
/// Use [`EmbeddingConfig::nomic_v15`] (768d, recommended) or
/// [`EmbeddingConfig::minilm_l6_v2`] (384d, legacy) for presets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding model identifier (e.g., `"NomicEmbedTextV15"`). Resolved
    /// by the `local/onnx` provider in uni-db's xervo catalog.
    pub model_id: String,
    /// Output vector dimensions (must match model).
    pub dimensions: usize,
    /// Batch size for auto-embed operations.
    pub batch_size: usize,
    /// Prefix prepended to documents before embedding.
    ///
    /// Nomic Embed Text v1.5 requires `"search_document: "` for optimal
    /// retrieval. Models without task prefixes (e.g., MiniLM) use `None`.
    pub document_prefix: Option<String>,
    /// Prefix prepended to queries before embedding.
    ///
    /// Nomic Embed Text v1.5 requires `"search_query: "` for optimal
    /// retrieval. Models without task prefixes (e.g., MiniLM) use `None`.
    pub query_prefix: Option<String>,
    /// Optional override for ONNX execution providers when registering
    /// the embedder with `local/onnx`. Values are xervo's EP string ids
    /// (e.g. `["cuda", "cpu"]`, `["coreml", "cpu"]`, `["cpu"]`). When
    /// `None`, [`embed_catalog`](crate::storage::embed_catalog) picks
    /// a feature-aware default (CUDA → CPU on `gpu-cuda`, CoreML → CPU
    /// on `gpu-metal`, CPU otherwise).
    #[serde(default)]
    pub execution_providers: Option<Vec<String>>,
}

impl EmbeddingConfig {
    /// Nomic Embed Text v1.5 — 768d, 8192 context, recommended.
    pub fn nomic_v15() -> Self {
        Self {
            model_id: "NomicEmbedTextV15".into(),
            dimensions: 768,
            batch_size: 32,
            document_prefix: Some("search_document: ".into()),
            query_prefix: Some("search_query: ".into()),
            execution_providers: None,
        }
    }

    /// Nomic Embed Text v1.5 quantized — 768d, faster, lower memory.
    pub fn nomic_v15_quantized() -> Self {
        Self {
            model_id: "NomicEmbedTextV15Q".into(),
            dimensions: 768,
            batch_size: 32,
            document_prefix: Some("search_document: ".into()),
            query_prefix: Some("search_query: ".into()),
            execution_providers: None,
        }
    }

    /// All-MiniLM-L6-v2 — 384d, legacy default for existing databases.
    pub fn minilm_l6_v2() -> Self {
        Self {
            model_id: "AllMiniLML6V2".into(),
            dimensions: 384,
            batch_size: 32,
            document_prefix: None,
            query_prefix: None,
            execution_providers: None,
        }
    }

    /// BAAI/bge-small-en-v1.5 — 384d, BERT-based, MTEB-strong general
    /// retriever. Uses a query-side prefix only (documents go in raw).
    pub fn bge_small_en_v15() -> Self {
        Self {
            model_id: "BGESmallENV15".into(),
            dimensions: 384,
            batch_size: 32,
            document_prefix: None,
            query_prefix: Some(
                "Represent this sentence for searching relevant passages: ".into(),
            ),
            execution_providers: None,
        }
    }

    /// BAAI/bge-large-en-v1.5 — 1024d, BERT-large-based, top-tier MTEB
    /// quality. Same query-prefix convention as BGE-small. ~10× the
    /// parameter count of BGE-small; expect 2-3× slower per-query
    /// embedding.
    pub fn bge_large_en_v15() -> Self {
        Self {
            model_id: "BGELargeENV15".into(),
            dimensions: 1024,
            batch_size: 16,
            document_prefix: None,
            query_prefix: Some(
                "Represent this sentence for searching relevant passages: ".into(),
            ),
            execution_providers: None,
        }
    }
}

/// Cross-encoder reranker selection.
///
/// When `enabled`, recall fuses RRF candidates and then re-scores the
/// top `top_n` items with a cross-encoder via uni-db's `local/onnx`
/// provider.  Disabled by default to keep CPU-only CI cheap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RerankerConfig {
    /// Whether to register and invoke the reranker.
    pub enabled: bool,
    /// HuggingFace model id for the cross-encoder ONNX export.
    pub model_id: String,
    /// Number of top RRF candidates to send to the reranker.
    ///
    /// Must be `>= recall_limit` when enabled; otherwise truncation
    /// would discard items the reranker hasn't seen.
    pub top_n: usize,
    /// Apply sigmoid to raw cross-encoder logits to map to `[0, 1]`.
    pub apply_sigmoid: bool,
    /// Optional override for ONNX execution providers when registering
    /// the reranker with `local/onnx`. Same semantics as the embedder
    /// equivalent on [`EmbeddingConfig`]. `None` → feature-aware default.
    #[serde(default)]
    pub execution_providers: Option<Vec<String>>,
    /// xervo `local/onnx` reranker code path. `"cross-encoder"` (default)
    /// loads BERT-family encoders that emit a relevance logit per
    /// `(query, doc)` pair (e.g. `BAAI/bge-reranker-base`,
    /// `cross-encoder/ms-marco-MiniLM-L-6-v2`). `"generative"` loads a
    /// decoder-LM reranker that scores yes/no via next-token logits
    /// (e.g. `onnx-community/Qwen3-Reranker-0.6B-ONNX`). Any other
    /// value triggers a runtime error from xervo.
    #[serde(default = "default_reranker_style")]
    pub style: String,
}

fn default_reranker_style() -> String {
    "cross-encoder".to_string()
}

fn default_rrf_k() -> f64 {
    60.0
}

fn default_nlp_srl_enabled() -> bool {
    true
}

impl RerankerConfig {
    /// BAAI/bge-reranker-base — 278M params, accurate, CPU-feasible.
    pub fn bge_base() -> Self {
        Self {
            enabled: false,
            model_id: "BAAI/bge-reranker-base".into(),
            top_n: 50,
            apply_sigmoid: true,
            execution_providers: None,
            style: default_reranker_style(),
        }
    }

    /// MS-MARCO MiniLM-L6-v2 — 22M params, ~12× faster, lower accuracy.
    pub fn minilm_l6() -> Self {
        Self {
            enabled: false,
            model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2".into(),
            top_n: 50,
            apply_sigmoid: true,
            execution_providers: None,
            style: default_reranker_style(),
        }
    }
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self::bge_base()
    }
}

/// Vector index algorithm and quantization strategy.
///
/// Controls how embedding vectors are indexed for similarity search.
/// Choose based on dataset scale and recall/memory tradeoff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorAlgorithm {
    /// HNSW with scalar quantization (default, best all-round).
    HnswSq {
        /// Max connections per node (higher = better recall, more memory).
        m: u32,
        /// Build-time search width (higher = better index quality, slower build).
        ef_construction: u32,
    },
    /// HNSW without quantization (maximum recall, ~4x more memory).
    HnswFlat {
        /// Max connections per node.
        m: u32,
        /// Build-time search width.
        ef_construction: u32,
    },
    /// HNSW with product quantization (large-scale, ~2-5% recall loss).
    HnswPq {
        /// Max connections per node.
        m: u32,
        /// Build-time search width.
        ef_construction: u32,
        /// Number of sub-vector segments for PQ (e.g., 48 for 768d → 16d each).
        sub_vectors: u32,
    },
    /// IVF with scalar quantization.
    IvfSq {
        /// Number of Voronoi partitions.
        partitions: u32,
    },
    /// IVF with product quantization.
    IvfPq {
        /// Number of Voronoi partitions.
        partitions: u32,
        /// Number of sub-vector segments.
        sub_vectors: u32,
    },
    /// IVF with residual quantization (best compression ratio).
    IvfRq {
        /// Number of Voronoi partitions.
        partitions: u32,
        /// Bits per dimension (default 8 if None).
        num_bits: Option<u8>,
    },
}

impl VectorAlgorithm {
    /// Convert to the uni-db `VectorAlgo` enum.
    pub(crate) fn to_uni_algo(&self) -> uni_db::VectorAlgo {
        match self {
            Self::HnswSq { m, ef_construction } => uni_db::VectorAlgo::HnswSq {
                m: *m,
                ef_construction: *ef_construction,
                partitions: None,
            },
            Self::HnswFlat { m, ef_construction } => uni_db::VectorAlgo::HnswFlat {
                m: *m,
                ef_construction: *ef_construction,
                partitions: None,
            },
            Self::HnswPq {
                m,
                ef_construction,
                sub_vectors,
            } => uni_db::VectorAlgo::HnswPq {
                m: *m,
                ef_construction: *ef_construction,
                sub_vectors: *sub_vectors,
                partitions: None,
            },
            Self::IvfSq { partitions } => uni_db::VectorAlgo::IvfSq {
                partitions: *partitions,
            },
            Self::IvfPq {
                partitions,
                sub_vectors,
            } => uni_db::VectorAlgo::IvfPq {
                partitions: *partitions,
                sub_vectors: *sub_vectors,
            },
            Self::IvfRq {
                partitions,
                num_bits,
            } => uni_db::VectorAlgo::IvfRq {
                partitions: *partitions,
                num_bits: *num_bits,
            },
        }
    }
}

/// Distance metric for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorMetricChoice {
    /// Cosine similarity (default, best for normalized embeddings).
    Cosine,
    /// Euclidean (L2) distance.
    L2,
    /// Dot product (for unnormalized embeddings).
    Dot,
}

impl VectorMetricChoice {
    /// Convert to the uni-db `VectorMetric` enum.
    pub(crate) fn to_uni_metric(self) -> uni_db::VectorMetric {
        match self {
            Self::Cosine => uni_db::VectorMetric::Cosine,
            Self::L2 => uni_db::VectorMetric::L2,
            Self::Dot => uni_db::VectorMetric::Dot,
        }
    }
}

// ── Main config ───────────────────────────────────────────────────

/// Configuration for all uniko runtime parameters.
///
/// Default values match the uniko specification v6.0 exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnikoConfig {
    // External configuration files
    /// Path to xervo model catalog JSON. When set, models are loaded from
    /// this file instead of the built-in catalog. Use `None` for defaults.
    #[serde(default)]
    pub catalog_path: Option<PathBuf>,
    /// Path to schema JSON. When set, schema is loaded from this file
    /// instead of the builder-based registration. Use `None` for defaults.
    #[serde(default)]
    pub schema_path: Option<PathBuf>,
    /// Path to observation extraction rules YAML. When set, the
    /// observation pipeline loads patterns from this file instead of
    /// the bundled `english.yml`. Lets us iterate on extraction
    /// patterns without recompiling.
    #[serde(default)]
    pub observation_rules_path: Option<PathBuf>,

    // Embedding + vector index
    /// Embedding model selection (controls dimensions and model ID).
    pub embedding: EmbeddingConfig,
    /// Cross-encoder reranker selection (disabled by default).
    #[serde(default)]
    pub reranker: RerankerConfig,
    /// Vector index algorithm and quantization strategy.
    pub vector_algorithm: VectorAlgorithm,
    /// Distance metric for similarity search.
    pub vector_metric: VectorMetricChoice,

    // Pipeline capacities
    /// Bounded channel capacity for the ingest worker.
    pub ingest_queue_capacity: usize,
    /// Bounded channel capacity for the consolidation worker.
    pub consolidation_queue_capacity: usize,

    // Consolidation triggers
    /// Number of observations that trigger consolidation.
    pub consolidation_threshold: u32,
    /// Seconds between periodic consolidation runs.
    pub consolidation_interval_secs: u64,

    // Retry policy
    /// Maximum retry attempts for retryable operations.
    pub retry_max_attempts: u32,
    /// Initial delay in milliseconds before first retry (exponential backoff base).
    pub retry_initial_delay_ms: u64,
    /// Maximum delay in milliseconds between retries (backoff cap).
    pub retry_max_delay_ms: u64,

    // Circuit breaker
    /// Number of consecutive failures before the circuit breaker opens.
    pub circuit_failure_threshold: u32,
    /// Milliseconds the circuit breaker stays open before probing.
    pub circuit_recovery_ms: u64,

    // Chunking thresholds
    /// Token count above which messages are chunked.
    pub message_chunk_threshold: usize,
    /// Token count above which action outputs overflow to Artifact nodes.
    pub action_output_artifact_threshold: usize,

    // Chunk sizing
    /// Maximum tokens per chunk.
    pub max_chunk_tokens: usize,
    /// Minimum tokens per chunk (fragments below this are merged).
    pub min_chunk_tokens: usize,
    /// Overlap tokens between adjacent chunks (0 = auto: 10% of max, capped at 50).
    pub chunk_overlap_tokens: usize,

    // NLP cascade
    /// Whether to compute SRL frames (one extra ONNX forward per VERB
    /// per sentence). Phase A landed inert SRL plumbing — when `false`,
    /// `NlpResult.srl_frames` stays empty and downstream extraction
    /// behaves exactly as before. Default `true` so the model's SRL
    /// head is actually used; flip to `false` if profiling shows the
    /// per-verb re-forward cost is unacceptable for a given workload.
    #[serde(default = "default_nlp_srl_enabled")]
    pub nlp_srl_enabled: bool,

    // Recall parameters
    /// Maximum items returned from recall.
    pub recall_limit: usize,
    /// Maximum total tokens in the context bundle.
    pub recall_token_budget: usize,
    /// Minimum fused score for result inclusion.
    pub recall_min_score: f64,
    /// Vector similarity weight in hybrid fusion \[0.0–1.0\].
    pub recall_vector_weight: f64,
    /// BM25 fulltext weight in hybrid fusion \[0.0–1.0\].
    pub recall_bm25_weight: f64,
    /// Variant labels for multi-query reformulation. Empty = use the
    /// default 4-variant set (`keywords`, `original`, `declarative`,
    /// `type_anchored`). Pass `vec!["keywords".into()]` to reproduce
    /// the legacy single-query behaviour. See
    /// `uniko_memory::recall::intent::QueryVariant` for the catalogue.
    #[serde(default)]
    pub query_variants: Vec<String>,
    /// `k` constant for reciprocal rank fusion across query variants.
    /// Higher values flatten the weight given to top ranks.
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f64,
    /// LIMIT applied to each per-variant Cypher query. `None` means
    /// "use `recall_limit`" (the recall layer falls back to it). Set
    /// to a smaller value when running 4 variants to keep the candidate
    /// union manageable.
    #[serde(default)]
    pub recall_per_variant_limit: Option<usize>,

    // Memory decay
    /// Half-life in days for importance decay: `importance * exp(-ln(2) / half_life * age_days)`.
    pub half_life_days: f64,
    /// Importance threshold below which nodes are pruned.
    pub prune_below: f64,

    // Recall cascade thresholds
    /// Coverage threshold for Phase 1 (Compact) early exit.
    pub phase1_coverage_threshold: f64,
    /// Coverage threshold for Phase 2 (Expand) early exit.
    pub phase2_coverage_threshold: f64,
}

impl Default for UnikoConfig {
    fn default() -> Self {
        Self {
            catalog_path: None,
            schema_path: None,
            observation_rules_path: None,
            embedding: EmbeddingConfig::nomic_v15(),
            reranker: RerankerConfig::default(),
            vector_algorithm: VectorAlgorithm::HnswSq {
                m: 16,
                ef_construction: 100,
            },
            vector_metric: VectorMetricChoice::Cosine,
            ingest_queue_capacity: 200,
            consolidation_queue_capacity: 32,
            consolidation_threshold: 20,
            consolidation_interval_secs: 900,
            retry_max_attempts: 3,
            retry_initial_delay_ms: 500,
            retry_max_delay_ms: 30_000,
            circuit_failure_threshold: 5,
            circuit_recovery_ms: 60_000,
            message_chunk_threshold: 1024,
            action_output_artifact_threshold: 256,
            max_chunk_tokens: 256,
            min_chunk_tokens: 32,
            chunk_overlap_tokens: 0, // 0 = auto: 10% of max, capped at 50
            recall_limit: 15,
            recall_token_budget: 8192,
            recall_min_score: 0.001,
            recall_vector_weight: 0.5,
            recall_bm25_weight: 0.5,
            nlp_srl_enabled: default_nlp_srl_enabled(),
            query_variants: Vec::new(),
            rrf_k: default_rrf_k(),
            recall_per_variant_limit: None,
            half_life_days: 30.0,
            prune_below: 0.05,
            phase1_coverage_threshold: 0.75,
            phase2_coverage_threshold: 0.65,
        }
    }
}

impl UnikoConfig {
    /// Validate configuration constraints.
    ///
    /// Returns `Err(UnikoError::Config)` if any constraint is violated.
    pub fn validate(&self) -> Result<()> {
        if self.embedding.dimensions == 0 {
            return Err(UnikoError::Config(
                "embedding.dimensions must be positive".into(),
            ));
        }

        if self.embedding.batch_size == 0 {
            return Err(UnikoError::Config(
                "embedding.batch_size must be positive".into(),
            ));
        }

        if self.min_chunk_tokens >= self.max_chunk_tokens {
            return Err(UnikoError::Config(format!(
                "min_chunk_tokens ({}) must be less than max_chunk_tokens ({})",
                self.min_chunk_tokens, self.max_chunk_tokens,
            )));
        }

        if self.half_life_days <= 0.0 {
            return Err(UnikoError::Config(format!(
                "half_life_days ({}) must be positive",
                self.half_life_days,
            )));
        }

        if self.prune_below < 0.0 || self.prune_below >= 1.0 {
            return Err(UnikoError::Config(format!(
                "prune_below ({}) must be in [0.0, 1.0)",
                self.prune_below,
            )));
        }

        if self.phase1_coverage_threshold <= 0.0 || self.phase1_coverage_threshold > 1.0 {
            return Err(UnikoError::Config(format!(
                "phase1_coverage_threshold ({}) must be in (0.0, 1.0]",
                self.phase1_coverage_threshold,
            )));
        }

        if self.phase2_coverage_threshold <= 0.0 || self.phase2_coverage_threshold > 1.0 {
            return Err(UnikoError::Config(format!(
                "phase2_coverage_threshold ({}) must be in (0.0, 1.0]",
                self.phase2_coverage_threshold,
            )));
        }

        if self.reranker.enabled && self.reranker.top_n < self.recall_limit {
            return Err(UnikoError::Config(format!(
                "reranker.top_n ({}) must be >= recall_limit ({}) when enabled",
                self.reranker.top_n, self.recall_limit,
            )));
        }

        if self.retry_initial_delay_ms > self.retry_max_delay_ms {
            return Err(UnikoError::Config(format!(
                "retry_initial_delay_ms ({}) must not exceed retry_max_delay_ms ({})",
                self.retry_initial_delay_ms, self.retry_max_delay_ms,
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let c = UnikoConfig::default();
        // Embedding defaults to Nomic v1.5 / 768d
        assert_eq!(c.embedding.model_id, "NomicEmbedTextV15");
        assert_eq!(c.embedding.dimensions, 768);
        assert_eq!(c.embedding.batch_size, 32);
        // Vector defaults to HnswSq / Cosine
        assert_eq!(
            c.vector_algorithm,
            VectorAlgorithm::HnswSq {
                m: 16,
                ef_construction: 100
            }
        );
        assert_eq!(c.vector_metric, VectorMetricChoice::Cosine);
        // Pipeline defaults
        assert_eq!(c.ingest_queue_capacity, 200);
        assert_eq!(c.consolidation_queue_capacity, 32);
        assert_eq!(c.consolidation_threshold, 20);
        assert_eq!(c.consolidation_interval_secs, 900);
        assert_eq!(c.retry_max_attempts, 3);
        assert_eq!(c.retry_initial_delay_ms, 500);
        assert_eq!(c.retry_max_delay_ms, 30_000);
        assert_eq!(c.circuit_failure_threshold, 5);
        assert_eq!(c.circuit_recovery_ms, 60_000);
        assert_eq!(c.message_chunk_threshold, 1024);
        assert_eq!(c.action_output_artifact_threshold, 256);
        assert_eq!(c.max_chunk_tokens, 256);
        assert_eq!(c.min_chunk_tokens, 32);
        assert_eq!(c.chunk_overlap_tokens, 0);
        assert_eq!(c.recall_limit, 15);
        assert_eq!(c.recall_token_budget, 8192);
        assert!((c.recall_min_score - 0.001).abs() < f64::EPSILON);
        assert!((c.recall_vector_weight - 0.5).abs() < f64::EPSILON);
        assert!((c.recall_bm25_weight - 0.5).abs() < f64::EPSILON);
        assert_eq!(c.half_life_days, 30.0);
        assert_eq!(c.prune_below, 0.05);
        assert_eq!(c.phase1_coverage_threshold, 0.75);
        assert_eq!(c.phase2_coverage_threshold, 0.65);
    }

    #[test]
    fn test_embedding_presets() {
        let nomic = EmbeddingConfig::nomic_v15();
        assert_eq!(nomic.dimensions, 768);

        let minilm = EmbeddingConfig::minilm_l6_v2();
        assert_eq!(minilm.dimensions, 384);

        let nomic_q = EmbeddingConfig::nomic_v15_quantized();
        assert_eq!(nomic_q.dimensions, 768);
        assert_eq!(nomic_q.model_id, "NomicEmbedTextV15Q");
    }

    #[test]
    fn test_config_validation_ok() {
        UnikoConfig::default()
            .validate()
            .expect("default config must be valid");
    }

    #[test]
    fn test_config_validation_fails() {
        // min >= max chunk tokens
        let c = UnikoConfig {
            min_chunk_tokens: 600,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // half_life_days <= 0
        let c = UnikoConfig {
            half_life_days: 0.0,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // prune_below out of range
        let c = UnikoConfig {
            prune_below: 1.0,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // phase1 threshold out of range
        let c = UnikoConfig {
            phase1_coverage_threshold: 0.0,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // phase2 threshold out of range
        let c = UnikoConfig {
            phase2_coverage_threshold: 1.5,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());

        // retry initial > max
        let c = UnikoConfig {
            retry_initial_delay_ms: 50_000,
            ..UnikoConfig::default()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let original = UnikoConfig::default();
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: UnikoConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(original, restored);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_config_roundtrip(
            ingest_cap in 1usize..1000,
            consol_cap in 1usize..100,
            max_chunk in 100usize..2000,
            // Use integer-derived f64 to avoid JSON float precision drift
            half_life_tenths in 1u32..3650,
            dims in proptest::prop_oneof![Just(384usize), Just(768usize)],
        ) {
            let half_life = f64::from(half_life_tenths) / 10.0;
            let embedding = if dims == 384 {
                EmbeddingConfig::minilm_l6_v2()
            } else {
                EmbeddingConfig::nomic_v15()
            };
            let config = UnikoConfig {
                embedding,
                ingest_queue_capacity: ingest_cap,
                consolidation_queue_capacity: consol_cap,
                max_chunk_tokens: max_chunk,
                min_chunk_tokens: max_chunk / 2,
                half_life_days: half_life,
                ..UnikoConfig::default()
            };
            let json = serde_json::to_string(&config).unwrap();
            let restored: UnikoConfig = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(config, restored);
        }
    }
}
