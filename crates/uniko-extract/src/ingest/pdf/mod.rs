//! PDF ingest — text-only path with a pluggable extractor.
//!
//! See `initial-docs/multimodal-knowledge-store-design.md` §5.3.4.
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

pub use chunker::{PdfPageChunker, chunk_pages};
pub use extractor::{ExtractedPage, PdfExtractCrate, PdfExtractError, PdfTextExtractor};
pub use input::{PdfIngestOptions, PdfInput};

use std::collections::HashMap;
use std::sync::Arc;

use uni_db::Value;

use uniko_store::storage::blob::MergeContent;
use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::ChunkConfig;
use super::message::create_chunks;

/// Result of [`ingest_pdf`].
#[derive(Debug)]
pub struct PdfIngestResult {
    /// Internal node ID of the `:Artifact{kind="pdf"}` row.
    pub artifact_node_id: NodeId,
    /// Internal node IDs of created chunks (empty on extractor failure).
    pub chunk_node_ids: Vec<NodeId>,
    /// Number of pages reported by the extractor. Zero on failure.
    pub page_count: u32,
    /// `true` if the artifact already existed by hash or `artifact_id`.
    pub was_deduplicated: bool,
    /// `None` on success, `Some(err)` if extraction failed but the
    /// artifact + content were still persisted.
    pub extraction_failure: Option<PdfExtractError>,
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
                chunk_node_ids: Vec::new(),
                page_count: 0,
                was_deduplicated: true,
                extraction_failure: None,
            });
        }
    }

    let size = bytes.len() as i64;

    // 3. Run extraction up-front so we know `page_count` for the
    //    Artifact row. We always persist the artifact + content even
    //    on extractor failure.
    let extractor: Arc<dyn PdfTextExtractor> = opts
        .extractor
        .clone()
        .unwrap_or_else(|| Arc::new(PdfExtractCrate));
    let extraction = extractor.extract(&bytes);
    let (pages, extraction_failure): (Vec<ExtractedPage>, Option<PdfExtractError>) =
        match extraction {
            Ok(p) => (p, None),
            Err(e) => {
                tracing::warn!(
                    target: "uniko_extract::ingest::pdf",
                    artifact_id = %opts.artifact_id,
                    error = %e,
                    "pdf extraction failed; persisting artifact without chunks",
                );
                (Vec::new(), Some(e))
            }
        };
    let page_count = pages.len() as u32;

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

    // 6. HAS_CONTENT edge.
    let mut edge_props: HashMap<String, Value> = HashMap::new();
    edge_props.insert("role".into(), Value::String("primary".into()));
    kb.create_edge("HAS_CONTENT", artifact_nid, content_nid, &edge_props)
        .await?;

    // 7. Chunk pages (skipped on extractor failure — pages is empty).
    let chunk_cfg = ChunkConfig::from_uniko_config(kb.config());
    let chunks = chunk_pages(&pages, &chunk_cfg);
    let chunk_nids = if chunks.is_empty() {
        Vec::new()
    } else {
        create_chunks(kb, &opts.artifact_id, artifact_nid, &chunks, "Artifact").await?
    };

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
        chunk_node_ids: chunk_nids,
        page_count,
        was_deduplicated: false,
        extraction_failure,
    })
}
