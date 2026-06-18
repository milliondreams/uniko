//! PDF ingest — text-only path with a pluggable extractor.
//!
//! See the multimodal design notes §5.3.4.
//!
//! Rasterization for VLM is deliberately out of scope: no production-
//! ready pure-Rust PDF rasterizer exists today, and pdfium / mupdf /
//! poppler are C/C++ FFI. The graph shape produced here is forward-
//! compatible with that future path — page chunks already carry
//! `metadata.page_number`, so adding rendered-page derivations later
//! does not require re-chunking.

pub mod chunker;
pub mod extractor;
pub mod input;
#[cfg(feature = "pdf-ocr")]
mod tiered;

pub use chunker::{PdfPageChunker, chunk_pages};
pub use extractor::{ExtractedPage, PdfExtractCrate, PdfExtractError, PdfTextExtractor};
pub use input::{PdfIngestOptions, PdfInput};

use std::collections::HashMap;
use std::sync::Arc;

use uniko_store::Value;

use uniko_store::storage::blob::MergeContent;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::ChunkConfig;
use super::message::create_chunks;

/// Result of [`ingest_pdf`].
#[derive(Debug)]
pub struct PdfIngestResult {
    /// Internal node ID of the `:Artifact{kind="pdf"}` row.
    pub artifact_node_id: NodeId,
    /// External artifact id — pass to `agent.data().artifact(..)` to fetch
    /// it back.
    pub artifact_id: String,
    /// Internal node IDs of created chunks (empty on extractor failure).
    pub chunk_node_ids: Vec<NodeId>,
    /// Number of pages reported by the extractor. Zero on failure.
    pub page_count: u32,
    /// `true` if the artifact already existed by hash or `artifact_id`.
    pub was_deduplicated: bool,
    /// `None` on success, `Some(err)` if extraction failed but the
    /// artifact + content were still persisted.
    pub extraction_failure: Option<PdfExtractError>,
    /// Internal node IDs of created `:Page` rows (tiered path only; empty
    /// on the text-only path).
    pub page_node_ids: Vec<NodeId>,
    /// Internal node IDs of created `:Block` rows (tiered path only; empty
    /// on the text-only path).
    pub block_node_ids: Vec<NodeId>,
}

/// Ingest a PDF document into the knowledge graph (text-only).
///
/// Hashes the bytes for dedup, persists them via [`KnowledgeBase::put_blob`]
/// then [`KnowledgeBase::merge_artifact_content`], creates the
/// `:Artifact{kind="pdf"}` node and its `HAS_CONTENT` edge, then runs the
/// configured [`PdfTextExtractor`] to produce per-page chunks.
///
/// On extractor failure (parse error, panic, or empty document) the
/// artifact and its content blob are still persisted, so callers can
/// re-extract later with a different backend. Zero chunks are created
/// and the failure is surfaced via
/// `PdfIngestResult::extraction_failure`.
///
/// # Errors
///
/// Returns a storage error if any graph or blob-backend operation
/// fails. Extraction failures do **not** propagate as `Err` — they
/// land in the result's `extraction_failure` field.
pub async fn ingest_pdf(
    kb: &KnowledgeBase,
    input: PdfInput,
    opts: PdfIngestOptions,
) -> uniko_store::Result<PdfIngestResult> {
    // 1. Materialize bytes.
    let bytes: Vec<u8> = match input {
        PdfInput::Bytes(b) => b,
        PdfInput::Path(p) => tokio::fs::read(&p)
            .await
            .map_err(|e| uniko_store::UnikoError::Storage(format!("read {}: {e}", p.display())))?,
    };

    if opts.artifact_id.is_empty() {
        return Err(uniko_store::UnikoError::Pipeline(
            "PdfIngestOptions.artifact_id must be non-empty".into(),
        ));
    }

    // 2. Hash + dedup (by hash, then by artifact_id).
    let hash = KnowledgeBase::sha256_hex(&bytes);
    for (key, value) in [("hash", hash.as_str()), ("artifact_id", &opts.artifact_id)] {
        if let Some((existing_id, _)) = kb.get_node_by_ext_id("Artifact", key, value).await? {
            return Ok(PdfIngestResult {
                artifact_node_id: existing_id,
                artifact_id: opts.artifact_id.clone(),
                chunk_node_ids: Vec::new(),
                page_count: 0,
                was_deduplicated: true,
                extraction_failure: None,
                page_node_ids: Vec::new(),
                block_node_ids: Vec::new(),
            });
        }
    }

    let size = bytes.len() as i64;

    // 3. Run extraction up-front so we know `page_count` for the
    //    Artifact row. We always persist the artifact + content even
    //    on extractor failure.
    //
    //    Tiered Native+OCR (doc-IR graph) is used when the `pdf-ocr`
    //    feature is built, OCR is enabled in config, and a model runtime
    //    is available; otherwise the pure-Rust text-only path runs.
    let use_tiered =
        cfg!(feature = "pdf-ocr") && kb.config().ocr.enabled && kb.model_runtime().is_some();

    let mut extraction_failure: Option<PdfExtractError> = None;

    #[cfg(feature = "pdf-ocr")]
    let mut tiered_pages: Vec<uni_xervo_pdf::TieredPageResult> = Vec::new();
    let mut legacy_pages: Vec<ExtractedPage> = Vec::new();

    #[cfg(feature = "pdf-ocr")]
    if use_tiered {
        match tiered::extract_tiered(kb, bytes.clone()).await {
            Ok(p) => tiered_pages = p,
            Err(e) => {
                tracing::warn!(
                    target: "uniko_extract::ingest::pdf",
                    artifact_id = %opts.artifact_id,
                    error = %e,
                    "tiered pdf extraction failed; persisting artifact without doc-IR",
                );
                extraction_failure = Some(e);
            }
        }
    }

    if !use_tiered {
        let extractor: Arc<dyn PdfTextExtractor> = opts
            .extractor
            .clone()
            .unwrap_or_else(|| Arc::new(PdfExtractCrate));
        match extractor.extract(&bytes) {
            Ok(p) => legacy_pages = p,
            Err(e) => {
                tracing::warn!(
                    target: "uniko_extract::ingest::pdf",
                    artifact_id = %opts.artifact_id,
                    error = %e,
                    "pdf extraction failed; persisting artifact without chunks",
                );
                extraction_failure = Some(e);
            }
        }
    }

    #[cfg(feature = "pdf-ocr")]
    let page_count = if use_tiered {
        tiered_pages.len() as u32
    } else {
        legacy_pages.len() as u32
    };
    #[cfg(not(feature = "pdf-ocr"))]
    let page_count = legacy_pages.len() as u32;

    // 4. Persist content via the blob backend + :ArtifactContent.
    let put = kb.put_blob(&hash, &bytes).await?;
    let content_nid = kb
        .merge_artifact_content(MergeContent {
            content_id: hash.clone(),
            bytes: put.bytes_inline,
            uri: put.uri,
            mime: "application/pdf".into(),
            size,
            perceptual_hash: None,
            audio_fingerprint: None,
        })
        .await?;

    // 5. Create :Artifact{kind="pdf"} metadata node.
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert(
        "artifact_id".into(),
        Value::String(opts.artifact_id.clone()),
    );
    props.insert("kind".into(), Value::String("pdf".into()));
    if let Some(ref path) = opts.source_path {
        props.insert("path".into(), Value::String(path.clone()));
    }
    props.insert("hash".into(), Value::String(hash));
    props.insert("size".into(), Value::Int(size));
    props.insert("page_count".into(), Value::Int(page_count as i64));
    let artifact_nid = kb.create_node("Artifact", &props).await?;

    // 5b. Conversational provenance: link the PDF to the Session it was
    //     shared in and/or the Message it was attached to, when resolvable.
    for (field, label, id_field) in [
        (opts.session_id.as_deref(), "Session", "session_id"),
        (
            opts.triggered_by_message_id.as_deref(),
            "Message",
            "message_id",
        ),
    ] {
        if let Some(ext_id) = field
            && let Some((target_nid, _)) = kb.get_node_by_ext_id(label, id_field, ext_id).await?
        {
            kb.create_edge("ATTACHED_TO", artifact_nid, target_nid, &HashMap::new())
                .await?;
        }
    }

    // 6. HAS_CONTENT edge.
    let mut edge_props: HashMap<String, Value> = HashMap::new();
    edge_props.insert("role".into(), Value::String("primary".into()));
    kb.create_edge("HAS_CONTENT", artifact_nid, content_nid, &edge_props)
        .await?;

    // 7. Materialize chunks. Tiered path builds the :Page/:Block doc-IR
    //    graph (each block owns a child :Chunk); legacy path emits one
    //    :Chunk per page. Both attach chunks to the Artifact so mean-pool
    //    and recall work uniformly. Skipped when extraction yielded nothing.
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    #[allow(unused_mut)]
    let mut page_node_ids: Vec<NodeId> = Vec::new();
    #[allow(unused_mut)]
    let mut block_node_ids: Vec<NodeId> = Vec::new();

    #[cfg(feature = "pdf-ocr")]
    let chunk_nids = if use_tiered {
        let mat = tiered::materialize_tiered(
            kb,
            &opts.artifact_id,
            artifact_nid,
            &tiered_pages,
            &chunk_cfg,
        )
        .await?;
        page_node_ids = mat.page_node_ids;
        block_node_ids = mat.block_node_ids;
        mat.chunk_node_ids
    } else {
        legacy_chunks(
            kb,
            &opts.artifact_id,
            artifact_nid,
            &legacy_pages,
            &chunk_cfg,
        )
        .await?
    };
    #[cfg(not(feature = "pdf-ocr"))]
    let chunk_nids = legacy_chunks(
        kb,
        &opts.artifact_id,
        artifact_nid,
        &legacy_pages,
        &chunk_cfg,
    )
    .await?;

    // 8. Mean-pool Artifact.text_embedding when chunks exist. Same
    //    best-effort warn-on-failure pattern as ingest_artifact.
    if !chunk_nids.is_empty()
        && let Err(e) = kb.mean_pool_artifact_text_embedding(artifact_nid).await
    {
        tracing::warn!(
            target: "uniko_extract::ingest::pdf",
            error = %e,
            "mean_pool_artifact_text_embedding failed; leaving NULL for backfill",
        );
    }

    Ok(PdfIngestResult {
        artifact_node_id: artifact_nid,
        artifact_id: opts.artifact_id.clone(),
        chunk_node_ids: chunk_nids,
        page_count,
        was_deduplicated: false,
        extraction_failure,
        page_node_ids,
        block_node_ids,
    })
}

/// Emit one `:Chunk` per page (text-only path) and link them to the artifact.
async fn legacy_chunks(
    kb: &KnowledgeBase,
    artifact_id: &str,
    artifact_nid: NodeId,
    pages: &[ExtractedPage],
    chunk_cfg: &ChunkConfig,
) -> uniko_store::Result<Vec<NodeId>> {
    let chunks = chunk_pages(pages, chunk_cfg);
    if chunks.is_empty() {
        return Ok(Vec::new());
    }
    create_chunks(kb, artifact_id, artifact_nid, &chunks, "Artifact").await
}
