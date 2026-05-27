//! Pluggable PDF text-extraction backend.
//!
//! [`PdfTextExtractor`] is the swap point. [`PdfExtractCrate`] is the
//! default impl, wrapping the `pdf-extract` crate inside
//! [`std::panic::catch_unwind`] so a malformed PDF that panics in the
//! underlying parser does not bring down the ingest worker
//! (pdf-extract issue #141 tracks ~50 known panics on adversarial input).

// Rust guideline compliant

use std::panic::{AssertUnwindSafe, catch_unwind};

/// Per-page extraction result. `page_number` is 1-indexed to match the
/// way humans (and PDF viewers) refer to pages.
#[derive(Debug, Clone)]
pub struct ExtractedPage {
    /// 1-indexed page number.
    pub page_number: u32,
    /// Plain-text content of the page.
    pub text: String,
}

/// Failure modes when extracting text from PDF bytes.
#[derive(Debug, thiserror::Error)]
pub enum PdfExtractError {
    /// The underlying parser reported a structural / decoding error.
    #[error("pdf parse error: {0}")]
    Parse(String),
    /// The underlying library panicked. Caught at the trait boundary
    /// so the caller can decide to skip the artifact rather than crash.
    #[error("pdf extractor panicked")]
    Panic,
    /// The PDF parsed but contained no extractable text on any page.
    #[error("pdf contained no extractable text")]
    Empty,
}

/// Pure-Rust PDF text extractor.
///
/// Implementations should not panic — they should catch panics from any
/// underlying library and return [`PdfExtractError::Panic`].
pub trait PdfTextExtractor: Send + Sync {
    /// Extract per-page text from `bytes`.
    ///
    /// # Errors
    /// Returns [`PdfExtractError::Parse`] for malformed PDFs,
    /// [`PdfExtractError::Panic`] if the underlying library panicked
    /// (caught via `catch_unwind`), or [`PdfExtractError::Empty`] when
    /// the document parsed but yielded no text.
    fn extract(&self, bytes: &[u8]) -> Result<Vec<ExtractedPage>, PdfExtractError>;
}

/// Default extractor — wraps the `pdf-extract` crate.
///
/// Per-page text via `pdf_extract::extract_text_from_mem_by_pages`,
/// invoked through [`std::panic::catch_unwind`] to contain library
/// panics on adversarial input.
#[derive(Debug, Default, Clone, Copy)]
pub struct PdfExtractCrate;

impl PdfTextExtractor for PdfExtractCrate {
    fn extract(&self, bytes: &[u8]) -> Result<Vec<ExtractedPage>, PdfExtractError> {
        // pdf-extract takes `&[u8]`. Borrow it across the catch_unwind
        // boundary via AssertUnwindSafe — the closure does not capture
        // any mutably-shared state.
        let result = catch_unwind(AssertUnwindSafe(|| {
            pdf_extract::extract_text_from_mem_by_pages(bytes)
        }));

        match result {
            Ok(Ok(pages)) => {
                if pages.iter().all(|p| p.trim().is_empty()) {
                    return Err(PdfExtractError::Empty);
                }
                Ok(pages
                    .into_iter()
                    .enumerate()
                    .map(|(i, text)| ExtractedPage {
                        page_number: u32::try_from(i + 1).unwrap_or(u32::MAX),
                        text,
                    })
                    .collect())
            }
            Ok(Err(e)) => Err(PdfExtractError::Parse(e.to_string())),
            Err(_) => Err(PdfExtractError::Panic),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_bytes_returns_parse_err() {
        let res = PdfExtractCrate.extract(b"this is not a pdf");
        match res {
            Err(PdfExtractError::Parse(_)) | Err(PdfExtractError::Panic) => {}
            other => panic!("expected Parse or Panic, got {other:?}"),
        }
    }

    #[test]
    fn empty_bytes_returns_parse_err() {
        let res = PdfExtractCrate.extract(b"");
        assert!(matches!(
            res,
            Err(PdfExtractError::Parse(_)) | Err(PdfExtractError::Panic)
        ));
    }

    #[test]
    fn extracts_text_from_real_pdf_fixture() {
        // Real round trip through `pdf-extract` against a tiny W3C WAI
        // test PDF — see `tests/fixtures/README.md`.
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/dummy.pdf"
        ))
        .expect("read dummy.pdf fixture");
        let pages = PdfExtractCrate.extract(&bytes).expect("extract");
        assert_eq!(pages.len(), 1, "dummy.pdf has one page");
        assert_eq!(pages[0].page_number, 1);
        assert!(
            pages[0].text.to_lowercase().contains("dummy"),
            "expected 'dummy' in extracted text, got: {:?}",
            pages[0].text
        );
    }
}
