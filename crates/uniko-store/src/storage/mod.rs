//! Layer 1 storage engine wrapping uni-db.
//!
//! [`KnowledgeBase`] is the single entry point for all graph operations in
//! uniko.  Higher layers (Extract, Memory, Cortex) interact with the graph
//! exclusively through this struct.

// Rust guideline compliant

pub mod batch;
pub mod edges;
pub mod filter;
pub mod nodes;

use std::path::Path;
use std::sync::Arc;

use uni_db::{ModelAliasSpec, ModelTask, Uni, WarmupPolicy};

use crate::config::UnikoConfig;
use crate::error::{Result, UnikoError};
use crate::schema::constants::{edges as edge_consts, labels};
use crate::schema::{EMBED_ALIAS, NLP_ALIAS, register_schema};
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
    /// Registers the full schema on creation.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be created.
    pub async fn in_memory(config: UnikoConfig) -> Result<Self> {
        config.validate()?;
        let db = Uni::in_memory()
            .xervo_catalog(embed_catalog(&config))
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        register_schema(&db, &config).await?;
        Ok(Self {
            db: Arc::new(db),
            config,
        })
    }

    /// Create an in-memory knowledge base with extra xervo model aliases.
    ///
    /// Merges the default embed + NLP catalog with `extra_catalog` entries.
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
        let mut catalog = embed_catalog(&config);
        catalog.extend(extra_catalog);
        let db = Uni::in_memory()
            .xervo_catalog(catalog)
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        register_schema(&db, &config).await?;
        Ok(Self {
            db: Arc::new(db),
            config,
        })
    }

    /// Open or create a persistent knowledge base at `path`.
    ///
    /// Registers the full schema on open (idempotent).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Config`] if validation fails, or
    /// [`UnikoError::Storage`] if the database cannot be opened.
    pub async fn open(path: impl AsRef<Path>, config: UnikoConfig) -> Result<Self> {
        config.validate()?;
        let db = Uni::open(path.as_ref().to_string_lossy())
            .xervo_catalog(embed_catalog(&config))
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        register_schema(&db, &config).await?;
        Ok(Self {
            db: Arc::new(db),
            config,
        })
    }

    /// Open a persistent knowledge base with extra xervo model aliases.
    ///
    /// Merges the default embed + NLP catalog with `extra_catalog` entries.
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
        config.validate()?;
        let mut catalog = embed_catalog(&config);
        catalog.extend(extra_catalog);
        let db = Uni::open(path.as_ref().to_string_lossy())
            .xervo_catalog(catalog)
            .build()
            .await
            .map_err(|e| UnikoError::Storage(e.to_string()))?;
        register_schema(&db, &config).await?;
        Ok(Self {
            db: Arc::new(db),
            config,
        })
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

/// Build the Xervo model catalog for embedding and NLP inference.
///
/// Registers two models:
/// - `"embed/default"` — fastembed model specified by config (Nomic 768d default).
/// - `"nlp/distilroberta"` — multi-task NER/POS/Dep/CLS via ONNX from HuggingFace.
///
/// Both use `WarmupPolicy::Lazy` (loaded on first call) and `required: false`
/// (startup succeeds even if providers are unavailable).
///
/// Exposed publicly so tests and downstream crates that create raw
/// `Uni` instances can configure the same catalog.
pub fn embed_catalog(config: &UnikoConfig) -> Vec<ModelAliasSpec> {
    vec![
        ModelAliasSpec {
            alias: EMBED_ALIAS.to_string(),
            task: ModelTask::Embed,
            provider_id: "local/fastembed".to_string(),
            model_id: config.embedding.model_id.clone(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({}),
        },
        ModelAliasSpec {
            alias: NLP_ALIAS.to_string(),
            task: ModelTask::Raw,
            provider_id: "local/onnx".to_string(),
            model_id: "dragonscale-ai/kniv-deberta-v3-nlp-en".to_string(),
            revision: None,
            warmup: WarmupPolicy::Lazy,
            required: false,
            timeout: None,
            load_timeout: None,
            retry: None,
            options: serde_json::json!({
                "artifact": "model-int8.onnx",
                "max_batch_size": 16
            }),
        },
    ]
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
