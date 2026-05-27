//! PDF ingest inputs and options.
//!
//! [`PdfInput`] describes where the PDF bytes come from. [`PdfIngestOptions`]
//! carries optional knobs — caller-provided `artifact_id`, custom extractor,
//! optional source `path` for the [`Artifact.path`](uniko_store) field.

use std::path::PathBuf;
use std::sync::Arc;

use super::extractor::PdfTextExtractor;

/// Source of PDF bytes for [`super::ingest_pdf`].
///
/// `Url` is deliberately omitted in v1 — URL fetching belongs to a
/// higher-level transport layer that hands us the resulting bytes.
#[derive(Debug, Clone)]
pub enum PdfInput {
    /// In-memory bytes.
    Bytes(Vec<u8>),
    /// Filesystem path; bytes are read at ingest time.
    Path(PathBuf),
}

/// Optional knobs for [`super::ingest_pdf`].
///
/// All fields are optional. The extractor defaults to
/// [`super::PdfExtractCrate`] when unset.
#[derive(Clone, Default)]
pub struct PdfIngestOptions {
    /// Caller-supplied artifact identifier. Required: every artifact in
    /// the graph carries an `artifact_id` ext-id; v1 does not auto-mint
    /// IDs (the workspace has no `uuid` dep yet).
    pub artifact_id: String,
    /// Optional override for the text-extraction backend.
    pub extractor: Option<Arc<dyn PdfTextExtractor>>,
    /// Optional original filesystem path / URL for the
    /// [`Artifact.path`](uniko_store) metadata field. Independent of
    /// [`PdfInput::Path`] — set this when ingesting `Bytes(...)` from a
    /// known source you want recorded.
    pub source_path: Option<String>,
}

impl std::fmt::Debug for PdfIngestOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdfIngestOptions")
            .field("artifact_id", &self.artifact_id)
            .field(
                "extractor",
                &self.extractor.as_ref().map(|_| "<dyn PdfTextExtractor>"),
            )
            .field("source_path", &self.source_path)
            .finish()
    }
}
