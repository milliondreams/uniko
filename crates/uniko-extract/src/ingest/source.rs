//! Unified ingest entry: one [`IngestSource`] that resolves its MIME and
//! routes to the right modality pipeline.
//!
//! `Session::observe` stays separate (conversational turns carry speaker /
//! recipient / threading state); this is the front door for *blobs* —
//! documents, PDFs, and future image/audio.

use serde_json::Value as JsonValue;
use uniko_pipes::content::{Mime, Modality, modality_for_mime};
use uniko_pipes::types::{IngestArtifact, IngestData, IngestSource};
use uniko_store::{KnowledgeBase, UnikoError};

use super::artifact::{ArtifactIngestResult, ingest_artifact};
use super::modality::ModalityRegistry;
use super::pdf::{PdfIngestOptions, PdfIngestResult, PdfInput, ingest_pdf};

/// What a unified ingest produced.
#[derive(Debug)]
pub enum IngestOutcome {
    /// A text/code/markup/structured/document/image/audio/video artifact.
    Artifact(ArtifactIngestResult),
    /// A PDF (tiered or legacy extraction).
    Pdf(PdfIngestResult),
}

/// Resolve the effective MIME for a payload.
///
/// Order: explicit override → magic bytes ([`infer`]) → file extension
/// ([`mime_guess`]) → `text/plain` for text payloads → `octet-stream`.
#[must_use]
pub fn resolve_mime(
    explicit: Option<&Mime>,
    bytes: Option<&[u8]>,
    path: Option<&str>,
    is_text: bool,
) -> Mime {
    if let Some(m) = explicit {
        return m.clone();
    }
    if let Some(b) = bytes
        && let Some(kind) = infer::get(b)
        && let Ok(m) = Mime::parse(kind.mime_type())
    {
        return m;
    }
    if let Some(p) = path
        && let Some(guess) = mime_guess::from_path(p).first_raw()
        && let Ok(m) = Mime::parse(guess)
    {
        return m;
    }
    if is_text {
        Mime::text_plain()
    } else {
        Mime::octet_stream()
    }
}

/// Provenance for an ingested blob: the session it was loaded into and/or
/// the message it was attached to.
///
/// `Session::ingest` sets `session_id` only; `Session::observe` sets both so
/// a conversational attachment links `Artifact -ATTACHED_TO-> Message`.
/// [`Default`] (both `None`) is the context-free streaming/corpus case.
#[derive(Debug, Default, Clone)]
pub struct IngestContext {
    /// Session the blob was loaded into.
    pub session_id: Option<String>,
    /// Message the blob was attached to (conversational provenance).
    pub triggered_by_message_id: Option<String>,
}

/// Ingest `src`, routing by its resolved [`Modality`].
///
/// Text/Code/Markup/Structured/Document become an artifact; Pdf takes the
/// tiered PDF path; Image/Audio/Video defer to a registered
/// [`ModalityExtractor`](super::modality::ModalityExtractor). `context`
/// carries session/message provenance set on the created artifact.
///
/// # Errors
///
/// Returns [`UnikoError::Unsupported`] when a non-text modality has no
/// registered extractor, or propagates the underlying ingest error.
pub async fn ingest_source(
    kb: &KnowledgeBase,
    registry: &ModalityRegistry,
    src: IngestSource,
    context: IngestContext,
) -> Result<IngestOutcome, UnikoError> {
    let is_text = matches!(src.data, IngestData::Text(_));
    let sniff_bytes = match &src.data {
        IngestData::Bytes(b) => Some(b.as_slice()),
        _ => None,
    };
    let sniff_path = src.path.as_deref().or(match &src.data {
        IngestData::Path(p) => p.to_str(),
        _ => None,
    });
    let mime = resolve_mime(src.mime.as_ref(), sniff_bytes, sniff_path, is_text);
    let modality = modality_for_mime(&mime);

    match modality {
        Modality::Pdf => {
            let input = match src.data {
                IngestData::Bytes(b) => PdfInput::Bytes(b),
                IngestData::Path(p) => PdfInput::Path(p),
                IngestData::Text(_) => {
                    return Err(UnikoError::Unsupported("text payload typed as PDF".into()));
                }
            };
            let options = PdfIngestOptions {
                artifact_id: src.id.unwrap_or_else(uniko_store::id::new_id),
                extractor: None,
                source_path: src.path,
                session_id: context.session_id,
                triggered_by_message_id: context.triggered_by_message_id,
            };
            Ok(IngestOutcome::Pdf(ingest_pdf(kb, input, options).await?))
        }
        Modality::Text
        | Modality::Code
        | Modality::Markup
        | Modality::Structured
        | Modality::Document => {
            let content = match src.data {
                IngestData::Text(t) => t,
                IngestData::Bytes(b) => String::from_utf8_lossy(&b).into_owned(),
                IngestData::Path(p) => std::fs::read_to_string(&p)
                    .map_err(|e| UnikoError::Storage(format!("read {}: {e}", p.display())))?,
            };
            let mut metadata = src.metadata;
            // Hint the chunker via the legacy content_type token, unless the
            // caller already set one.
            metadata
                .entry("content_type".to_string())
                .or_insert_with(|| JsonValue::String(chunker_hint(modality).to_string()));
            let artifact = IngestArtifact {
                artifact_id: src.id.unwrap_or_else(uniko_store::id::new_id),
                content,
                kind: "document".to_string(),
                path: src.path,
                metadata,
                session_id: context.session_id,
                triggered_by_message_id: context.triggered_by_message_id,
                produced_by_action_id: None,
            };
            Ok(IngestOutcome::Artifact(
                ingest_artifact(kb, &artifact).await?,
            ))
        }
        Modality::Image | Modality::Audio | Modality::Video => match registry.get(modality) {
            Some(extractor) => Ok(IngestOutcome::Artifact(extractor.extract(kb, &src).await?)),
            None => Err(UnikoError::Unsupported(format!("{modality:?}"))),
        },
        // `Modality` is `#[non_exhaustive]`.
        _ => Err(UnikoError::Unsupported(format!("{modality:?}"))),
    }
}

/// Legacy chunker hint token for a text-family modality.
fn chunker_hint(modality: Modality) -> &'static str {
    match modality {
        Modality::Code => "code",
        Modality::Markup => "html",
        Modality::Structured => "json",
        _ => "text",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use async_trait::async_trait;
    use uniko_store::config::UnikoConfig;
    use uniko_store::storage::KnowledgeBase;

    use super::super::modality::{ModalityExtractor, ModalityRegistry};

    #[derive(Debug)]
    struct StubImage;

    #[async_trait]
    impl ModalityExtractor for StubImage {
        fn modality(&self) -> Modality {
            Modality::Image
        }
        async fn extract(
            &self,
            _kb: &KnowledgeBase,
            _src: &IngestSource,
        ) -> Result<ArtifactIngestResult, UnikoError> {
            Ok(ArtifactIngestResult {
                artifact_node_id: 42,
                artifact_id: "stub-image".into(),
                chunk_node_ids: vec![],
                was_deduplicated: false,
            })
        }
    }

    #[tokio::test]
    async fn registered_extractor_handles_its_modality() {
        let kb = KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .unwrap();
        let mut registry = ModalityRegistry::new();
        registry.register(Arc::new(StubImage));
        let png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        let outcome = ingest_source(
            &kb,
            &registry,
            IngestSource::bytes(png),
            IngestContext::default(),
        )
        .await
        .unwrap();
        match outcome {
            IngestOutcome::Artifact(r) => assert_eq!(r.artifact_node_id, 42),
            other => panic!("expected stub artifact, got {other:?}"),
        }
        kb.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn unregistered_modality_is_unsupported() {
        let kb = KnowledgeBase::in_memory(UnikoConfig::default())
            .await
            .unwrap();
        let registry = ModalityRegistry::new();
        let src = IngestSource::bytes(vec![1, 2, 3]).with_mime(Mime::parse("audio/mpeg").unwrap());
        let err = ingest_source(&kb, &registry, src, IngestContext::default())
            .await
            .unwrap_err();
        assert!(matches!(err, UnikoError::Unsupported(_)), "got {err:?}");
        kb.shutdown().await.unwrap();
    }

    #[test]
    fn explicit_mime_wins_over_sniffing() {
        let m = resolve_mime(
            Some(&Mime::parse("application/json").unwrap()),
            Some(b"%PDF-1.4"),
            Some("x.png"),
            false,
        );
        assert_eq!(m.essence(), "application/json");
    }

    #[test]
    fn sniffs_pdf_magic_bytes() {
        let m = resolve_mime(None, Some(b"%PDF-1.7\n%docs"), None, false);
        assert_eq!(m.essence(), "application/pdf");
        assert_eq!(modality_for_mime(&m), Modality::Pdf);
    }

    #[test]
    fn sniffs_png_magic_bytes() {
        let png = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0];
        let m = resolve_mime(None, Some(&png), None, false);
        assert_eq!(m.type_(), "image");
        assert_eq!(modality_for_mime(&m), Modality::Image);
    }

    #[test]
    fn falls_back_to_extension() {
        let m = resolve_mime(None, None, Some("notes.md"), false);
        assert_eq!(modality_for_mime(&m), Modality::Markup);
    }

    #[test]
    fn falls_back_to_text_then_octet_stream() {
        assert_eq!(resolve_mime(None, None, None, true).essence(), "text/plain");
        let bin = resolve_mime(None, None, None, false);
        assert_eq!(bin.essence(), "application/octet-stream");
        assert_eq!(modality_for_mime(&bin), Modality::Text);
    }
}
