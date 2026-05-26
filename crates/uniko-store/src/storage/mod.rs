//! Layer 1 storage engine wrapping uni-db.
//!
//! [`KnowledgeBase`] is the single entry point for all graph operations in
//! uniko.  Higher layers (Extract, Memory, Cortex) interact with the graph
//! exclusively through this struct.

// Rust guideline compliant

pub mod batch;
pub mod blob;
pub mod edges;
pub mod filter;
pub mod kb_stats;
pub mod migrations;
pub mod nodes;

use std::path::Path;
use std::sync::Arc;

use uni_db::{ModelAliasSpec, ModelTask, Uni, UniConfig, WarmupPolicy};

/// Diagnostic perf-knob override read from env vars. Set
/// `UNIKO_WAL_DISABLED=1` to disable WAL, or
/// `UNIKO_AUTOFLUSH_THRESHOLD=N` to override the L0 auto-flush
/// threshold. Returns Some(cfg) only if at least one knob is set, so
/// the default UniConfig isn't perturbed in normal use.
fn apply_perf_knobs_from_env() -> Option<UniConfig> {
    let wal = std::env::var("UNIKO_WAL_DISABLED").ok();
    let flush = std::env::var("UNIKO_AUTOFLUSH_THRESHOLD").ok();
    let flush_interval_off = std::env::var("UNIKO_AUTOFLUSH_INTERVAL_OFF").ok();
    if wal.is_none() && flush.is_none() && flush_interval_off.is_none() {
        return None;
    }
    let mut cfg = UniConfig::default();
    if matches!(wal.as_deref(), Some("1") | Some("true")) {
        cfg.wal_enabled = false;
        tracing::warn!("UNIKO_WAL_DISABLED=1 — running with WAL OFF (diagnostic only)");
    }
    if let Some(s) = flush
        && let Ok(n) = s.parse::<usize>()
    {
        cfg.auto_flush_threshold = n;
        tracing::warn!(threshold = n, "UNIKO_AUTOFLUSH_THRESHOLD set");
    }
    if matches!(flush_interval_off.as_deref(), Some("1") | Some("true")) {
        cfg.auto_flush_interval = None;
        tracing::warn!("UNIKO_AUTOFLUSH_INTERVAL_OFF=1 — disabling time-based flush");
    }
    Some(cfg)
}

use crate::config::UnikoConfig;
use crate::error::{Result, UnikoError};
use crate::schema::constants::{edges as edge_consts, labels};
use crate::schema::{EMBED_ALIAS, NLP_ALIAS, RERANK_ALIAS, register_schema};
use crate::types::{EdgeId, NodeId};

pub use edges::{Direction, EdgeRecord};
pub use filter::Filter;

/// Layer 1 storage engine wrapping a uni-db instance.
///
/// Provides typed CRUD operations, vector/fulltext/hybrid search, graph
/// traversal, and Locy runtime access.  All operations are async because
/// uni-db transactions are async.
///
/// # Examples
///
/// ```no_run
/// # async fn example() -> uniko_store::Result<()> {
/// use uniko_store::config::UnikoConfig;
/// use uniko_store::storage::KnowledgeBase;
///
/// let kb = KnowledgeBase::in_memory(UnikoConfig::default()).await?;
/// // ... use kb for CRUD, search, Locy ...
/// kb.shutdown().await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct KnowledgeBase {
    pub(crate) db: Arc<Uni>,
    pub(crate) config: UnikoConfig,
}

impl KnowledgeBase {
    /// Create an in-memory knowledge base for testing or ephemeral use.
    ///
    /// Registers the full schema on creation and eagerly warms xervo
    /// models via [`prefetch_all`](uni_db::api::xervo::UniXervo::prefetch_all).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be created.
    pub async fn in_memory(config: UnikoConfig) -> Result<Self> {
        config.validate()?;
        let catalog = load_catalog(&config, &[])?;
        let db = Uni::in_memory()
            .xervo_catalog(catalog)
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        apply_schema(&db, &config).await?;
        prefetch_models(&db).await;
        Self {
            db: Arc::new(db),
            config,
        }
        .finalize_init()
        .await
    }

    /// Create an in-memory knowledge base with extra xervo model aliases.
    ///
    /// Merges the catalog (from file or built-in) with `extra_catalog`.
    /// Use this to add LLM generation models (e.g., for benchmarks or
    /// answer synthesis).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be created.
    pub async fn in_memory_with_xervo(
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
    ) -> Result<Self> {
        config.validate()?;
        let catalog = load_catalog(&config, &extra_catalog)?;
        let db = Uni::in_memory()
            .xervo_catalog(catalog)
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        apply_schema(&db, &config).await?;
        prefetch_models(&db).await;
        Self {
            db: Arc::new(db),
            config,
        }
        .finalize_init()
        .await
    }

    /// Open or create a persistent knowledge base at `path`.
    ///
    /// Registers the full schema on open (idempotent) and eagerly
    /// warms xervo models.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be opened.
    pub async fn open(path: impl AsRef<Path>, config: UnikoConfig) -> Result<Self> {
        config.validate()?;
        let catalog = load_catalog(&config, &[])?;
        let mut builder = Uni::open(path.as_ref().to_string_lossy()).xervo_catalog(catalog);
        if let Some(uni_cfg) = apply_perf_knobs_from_env() {
            builder = builder.config(uni_cfg);
        }
        let db = builder
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        apply_schema(&db, &config).await?;
        prefetch_models(&db).await;
        Self {
            db: Arc::new(db),
            config,
        }
        .finalize_init()
        .await
    }

    /// Open a persistent knowledge base with extra xervo model aliases.
    ///
    /// Merges the catalog (from file or built-in) with `extra_catalog`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be opened.
    pub async fn open_with_xervo(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
    ) -> Result<Self> {
        Self::open_with_xervo_inner(path, config, extra_catalog, true).await
    }

    /// Open a KB without pre-warming any xervo models.
    ///
    /// Useful for tools that only inspect the graph (e.g. read-only
    /// Cypher) and never call `similar_to`/`generate`. Skipping prefetch
    /// saves the multi-minute model-download/load cost on each launch.
    /// Models still load lazily on first use if a query needs them.
    pub async fn open_with_xervo_no_prefetch(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
    ) -> Result<Self> {
        Self::open_with_xervo_inner(path, config, extra_catalog, false).await
    }

    async fn open_with_xervo_inner(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        extra_catalog: Vec<ModelAliasSpec>,
        prefetch: bool,
    ) -> Result<Self> {
        config.validate()?;
        let catalog = load_catalog(&config, &extra_catalog)?;
        let db = Uni::open(path.as_ref().to_string_lossy())
            .xervo_catalog(catalog)
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        apply_schema(&db, &config).await?;
        if prefetch {
            prefetch_models(&db).await;
        }
        Self {
            db: Arc::new(db),
            config,
        }
        .finalize_init()
        .await
    }

    /// Build a single ONNX/Xervo runtime that multiple KBs can share.
    ///
    /// At `--question-concurrency N` the bench opens N persistent KBs;
    /// without sharing, each opens its own `ModelRuntime`, which loads
    /// its own ONNX sessions (the per-session BFC arena dominates GPU
    /// VRAM). Sharing one runtime keeps weights and the activation
    /// arena resident exactly once.
    ///
    /// Implementation note: uni-db's `UniBuilder::xervo_runtime` takes
    /// a pre-built `Arc<ModelRuntime>`, but the runtime's provider
    /// registration is gated by `#[cfg(feature = "provider-*")]`
    /// inside uni-db — reproducing that gating outside the crate is
    /// fragile. Instead, we bootstrap by opening a tiny
    /// `Uni::in_memory()` with the catalog (which goes through all
    /// the provider-registration gates correctly), extract the
    /// resulting `Arc<ModelRuntime>`, and drop the bootstrap `Uni`.
    /// The model warmup runs once here, so callers get a hot runtime
    /// they can hand to many [`KnowledgeBase::open_with_runtime`]
    /// calls.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if catalog validation fails,
    /// or [`UnikoError::Storage`] if the bootstrap `Uni` cannot be
    /// opened (the in-memory backend should not fail in practice).
    /// Returns [`UnikoError::Internal`] if the bootstrap `Uni`
    /// somehow finishes without registering an xervo runtime.
    pub async fn build_shared_runtime(
        config: &UnikoConfig,
        extra_catalog: &[ModelAliasSpec],
    ) -> Result<Arc<uni_xervo::runtime::ModelRuntime>> {
        config.validate()?;
        let catalog = load_catalog(config, extra_catalog)?;
        let bootstrap = Uni::in_memory()
            .xervo_catalog(catalog)
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        let runtime = bootstrap.xervo().raw_runtime().cloned().ok_or_else(|| {
            UnikoError::Internal(
                "bootstrap Uni did not register an xervo runtime (was the catalog empty?)".into(),
            )
        })?;
        // Warm every alias up front so the first inference per KB
        // doesn't pay the cold-start latency.
        prefetch_models(&bootstrap).await;
        // Dropping the bootstrap Uni at the end of scope is fine —
        // the runtime is held by the returned `Arc`.
        drop(bootstrap);
        Ok(runtime)
    }

    /// Open a persistent KB that **shares** the supplied `ModelRuntime`.
    ///
    /// Use with [`KnowledgeBase::build_shared_runtime`] to run many
    /// concurrent KBs against one ONNX session, instead of paying the
    /// per-KB session cost. Each KB still has its own graph storage
    /// on disk; only the inference runtime is shared.
    ///
    /// Skips the per-KB model prefetch (the shared runtime is already
    /// warmed up at construction). Still applies the schema per-KB.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be opened.
    pub async fn open_with_runtime(
        path: impl AsRef<Path>,
        config: UnikoConfig,
        runtime: Arc<uni_xervo::runtime::ModelRuntime>,
    ) -> Result<Self> {
        config.validate()?;
        let db = Uni::open(path.as_ref().to_string_lossy())
            .xervo_runtime(runtime)
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        apply_schema(&db, &config).await?;
        Self {
            db: Arc::new(db),
            config,
        }
        .finalize_init()
        .await
    }

    /// Run post-construction init steps. Currently:
    ///
    /// - [`init_kb_stats`](Self::init_kb_stats): writes the
    ///   `:KnowledgeBaseStats` singleton on first open, or verifies
    ///   `blob_storage` matches on reopen.
    ///
    /// All five public constructors funnel through this so the
    /// singleton row is always present after a successful `open` /
    /// `in_memory`.
    async fn finalize_init(self) -> Result<Self> {
        self.init_kb_stats().await?;
        Ok(self)
    }

    /// Direct access to the underlying uni-db instance.
    ///
    /// Escape hatch for advanced operations not covered by the
    /// [`KnowledgeBase`] API.
    pub fn db(&self) -> &Uni {
        &self.db
    }

    /// Runtime configuration.
    pub fn config(&self) -> &UnikoConfig {
        &self.config
    }

    /// Graceful shutdown, flushing pending writes.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Internal`] if other `Arc` references still
    /// exist, or [`UnikoError::Storage`] if shutdown fails.
    pub async fn shutdown(self) -> Result<()> {
        let db = Arc::try_unwrap(self.db).map_err(|_| {
            UnikoError::Internal("cannot shutdown: outstanding references exist".into())
        })?;
        db.shutdown()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))
    }
}

// ── Internal helpers ────────────────────────────────────────────────

/// Load the xervo model catalog from a JSON file or build the default.
///
/// When `config.catalog_path` is set, reads from that file and appends
/// `extra`. Otherwise builds the default catalog from config + `extra`.
fn load_catalog(config: &UnikoConfig, extra: &[ModelAliasSpec]) -> Result<Vec<ModelAliasSpec>> {
    let mut catalog = if let Some(path) = &config.catalog_path {
        uni_db::xervo_catalog_from_file(path)
            .map_err(|e| UnikoError::Config(format!("catalog {}: {e}", path.display())))?
    } else {
        embed_catalog(config)
    };
    catalog.extend_from_slice(extra);
    Ok(catalog)
}

/// Eagerly download and warm every xervo model in the catalog.
///
/// `prefetch_all()` materializes each artifact into the repo snapshot
/// and loads the model into memory, so the first inference call hits a
/// pre-warmed runner.  Errors are surfaced at `warn` level (we keep
/// `required: false` so the KB still opens; failures are operationally
/// important to see).
async fn prefetch_models(db: &Uni) {
    let started = std::time::Instant::now();
    match db.xervo().prefetch_all().await {
        Ok(()) => {
            tracing::info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                "xervo prefetch_all complete"
            );
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                elapsed_ms = started.elapsed().as_millis() as u64,
                "xervo prefetch_all failed — models may load lazily on first use"
            );
        }
    }
}

/// Apply the schema from a JSON file or the builder-based registration.
async fn apply_schema(db: &Uni, config: &UnikoConfig) -> Result<()> {
    if let Some(path) = &config.schema_path {
        db.load_schema(path)
            .await
            .map_err(|e| UnikoError::Schema(e.to_string()))
    } else {
        register_schema(db, config).await
    }
}

/// Build the Xervo model catalog for embedding, reranking, and NLP inference.
///
/// Registers up to three models:
/// - `"embed/default"` — ONNX embedding model (Nomic 768d default).
/// - `"rerank/default"` — ONNX cross-encoder reranker (only when `config.reranker.enabled`).
/// - `"nlp/default"` — multi-task NER/POS/Dep/CLS via ONNX from HuggingFace.
///
/// All entries use `WarmupPolicy::Lazy` (loaded on first call) and
/// `required: false` (startup succeeds even if providers are unavailable).
///
/// Exposed publicly so tests and downstream crates that create raw
/// `Uni` instances can configure the same catalog.
pub fn embed_catalog(config: &UnikoConfig) -> Vec<ModelAliasSpec> {
    let embed_eps = resolve_eps(config.embedding.execution_providers.as_deref());
    let rerank_eps = resolve_eps(config.reranker.execution_providers.as_deref());
    // NLP defaults to the embedder's device list when the operator
    // hasn't explicitly overridden it — keeps all three local/onnx
    // aliases on the same device by default while still letting a
    // bench profile pin NLP to CPU when the embedder is on GPU.
    let nlp_eps = match config.nlp.execution_providers.as_deref() {
        Some(eps) => eps.to_vec(),
        None => embed_eps.clone(),
    };

    let mut catalog = vec![
        ModelAliasSpec {
            alias: EMBED_ALIAS.to_string(),
            task: ModelTask::Embed,
            provider_id: config.embedding.provider.clone(),
            model_id: config.embedding.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: build_embed_options(&config.embedding, &embed_eps),
        },
        ModelAliasSpec {
            alias: NLP_ALIAS.to_string(),
            task: ModelTask::Raw,
            provider_id: "local/onnx".to_string(),
            model_id: config.nlp.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({
                "artifact": config.nlp.artifact,
                "max_batch_size": config.nlp.max_batch_size,
                "execution_providers": nlp_eps,
            }),
        },
    ];

    if config.reranker.enabled {
        catalog.push(ModelAliasSpec {
            alias: RERANK_ALIAS.to_string(),
            task: ModelTask::Rerank,
            provider_id: "local/onnx".to_string(),
            model_id: config.reranker.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({
                "execution_providers": rerank_eps,
                "style": config.reranker.style,
            }),
        });
    }

    catalog
}

/// Resolve the ONNX execution-provider list for an alias.
///
/// Honours an explicit override from config when provided; otherwise
/// falls back to the build-time default — CUDA → CPU on `gpu-cuda`,
/// CoreML → CPU on `gpu-metal`, CPU otherwise. The returned `Vec`
/// goes into the alias's `options.execution_providers` JSON which
/// uni-xervo's `parse_execution_providers_option` consumes (see
/// `uni-xervo/src/provider/onnx_ep.rs`).
fn resolve_eps(override_eps: Option<&[String]>) -> Vec<String> {
    if let Some(eps) = override_eps {
        return eps.to_vec();
    }
    default_eps()
}

#[cfg(feature = "gpu-cuda")]
fn default_eps() -> Vec<String> {
    vec!["cuda".to_string(), "cpu".to_string()]
}

#[cfg(all(feature = "gpu-metal", not(feature = "gpu-cuda")))]
fn default_eps() -> Vec<String> {
    vec!["coreml".to_string(), "cpu".to_string()]
}

#[cfg(not(any(feature = "gpu-cuda", feature = "gpu-metal")))]
fn default_eps() -> Vec<String> {
    vec!["cpu".to_string()]
}

/// Compose the `options` JSON for the embed alias based on the
/// selected provider.
///
/// `local/onnx` accepts `execution_providers`; remote providers
/// (`remote/openai`, `remote/voyageai`, ...) take any
/// `provider_options` the user supplied verbatim and additionally
/// receive `embedding_dimensions` so xervo's OpenAI provider can
/// thread it into the `/embeddings` request.
fn build_embed_options(
    cfg: &crate::config::EmbeddingConfig,
    embed_eps: &[String],
) -> serde_json::Value {
    let mut opts = match cfg.provider_options.clone() {
        Some(serde_json::Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    if cfg.provider == "local/onnx" {
        // Caller-supplied provider_options pass through for models that
        // resolve via uni-xervo's HF-Hub fallback path (no preset).
        // Those models need `artifact`, `pooling`, `dimensions`,
        // `token_type_ids`, etc. on the alias options.  Plus the
        // execution-provider list xervo always wants for local/onnx.
        opts.insert(
            "execution_providers".to_string(),
            serde_json::json!(embed_eps),
        );
        return serde_json::Value::Object(opts);
    }
    opts.insert(
        "embedding_dimensions".to_string(),
        serde_json::json!(cfg.dimensions),
    );
    serde_json::Value::Object(opts)
}

/// Convert a uni-db `Vid` to our `NodeId` (`i64`).
#[expect(
    dead_code,
    reason = "will be used when higher layers operate on Vid directly"
)]
pub(crate) fn vid_to_node_id(vid: uni_db::Vid) -> NodeId {
    vid.as_u64() as i64
}

/// Convert our `NodeId` to a uni-db `Vid`.
#[expect(
    dead_code,
    reason = "will be used when higher layers operate on Vid directly"
)]
pub(crate) fn node_id_to_vid(id: NodeId) -> uni_db::Vid {
    uni_db::Vid::new(id as u64)
}

/// Convert a uni-db `Eid` to our `EdgeId` (`i64`).
#[expect(
    dead_code,
    reason = "will be used when higher layers operate on Eid directly"
)]
pub(crate) fn eid_to_edge_id(eid: uni_db::Eid) -> EdgeId {
    eid.as_u64() as i64
}

/// Verify that `label` is a known node label.
pub(crate) fn validate_label(label: &str) -> Result<()> {
    if !labels::ALL.contains(&label) {
        return Err(UnikoError::Schema(format!("unknown node label: {label}")));
    }
    Ok(())
}

/// Verify that `edge_type` is a known edge type.
pub(crate) fn validate_edge_type(edge_type: &str) -> Result<()> {
    if !edge_consts::ALL.contains(&edge_type) {
        return Err(UnikoError::Schema(format!(
            "unknown edge type: {edge_type}"
        )));
    }
    Ok(())
}

/// Verify that a property name is safe for Cypher interpolation.
///
/// Accepts `[a-zA-Z_][a-zA-Z0-9_]*`.
pub(crate) fn validate_property_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(UnikoError::Schema("empty property name".into()));
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(UnikoError::Schema(format!("invalid property name: {name}")));
    }
    for ch in chars {
        if !ch.is_ascii_alphanumeric() && ch != '_' {
            return Err(UnikoError::Schema(format!("invalid property name: {name}")));
        }
    }
    Ok(())
}

/// Build inline property syntax `prop1: $s0, prop2: $s1, ...` for CREATE.
///
/// Returns `(inline_fragment, params)`.
pub(crate) fn build_inline_props(
    properties: &std::collections::HashMap<String, uni_db::Value>,
    offset: usize,
) -> Result<(String, Vec<(String, uni_db::Value)>)> {
    if properties.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    let mut fragments = Vec::with_capacity(properties.len());
    let mut params = Vec::with_capacity(properties.len());
    for (i, (key, val)) in properties.iter().enumerate() {
        validate_property_name(key)?;
        let param = format!("s{}", offset + i);
        fragments.push(format!("{key}: ${param}"));
        params.push((param, val.clone()));
    }
    Ok((fragments.join(", "), params))
}

/// Build a `SET` clause from a property map, returning `(cypher_fragment, params)`.
///
/// Generates `SET {var}.prop0 = $s{offset}, {var}.prop1 = $s{offset+1}, ...`
/// and a list of `(param_name, Value)` bindings.
pub(crate) fn build_set_clause(
    var: &str,
    properties: &std::collections::HashMap<String, uni_db::Value>,
    offset: usize,
) -> Result<(String, Vec<(String, uni_db::Value)>)> {
    if properties.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    let mut fragments = Vec::with_capacity(properties.len());
    let mut params = Vec::with_capacity(properties.len());
    for (i, (key, val)) in properties.iter().enumerate() {
        validate_property_name(key)?;
        let param = format!("s{}", offset + i);
        fragments.push(format!("{var}.{key} = ${param}"));
        params.push((param, val.clone()));
    }
    let clause = format!("SET {}", fragments.join(", "));
    Ok((clause, params))
}

impl std::fmt::Debug for KnowledgeBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KnowledgeBase")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
