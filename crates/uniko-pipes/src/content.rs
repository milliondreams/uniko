//! Content-type taxonomy: [`Mime`], [`Modality`], and [`ContentType`].
//!
//! Ingest used to thread bare strings (`"text"`, `"code"`, `"pdf"`) and
//! branch on them in scattered `match` arms. This module is the single
//! source of truth that maps a MIME type to a coarse [`Modality`] (the
//! routing key) while preserving the full MIME string for fidelity.
//!
//! The same [`Modality`] enum drives both ingest routing (which extractor /
//! chunker handles a blob) and recall's per-modality channels, so the two
//! ends of the pipeline share one vocabulary.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A validated MIME type, stored as its normalized string.
///
/// Wraps a `String` rather than exposing the third-party `mime::Mime` so the
/// public surface owns its types (M-DONT-LEAK-TYPES). Parameters (e.g.
/// `; charset=utf-8`) are preserved; [`essence`](Mime::essence) drops them.
///
/// # Examples
///
/// ```
/// use uniko_pipes::content::Mime;
///
/// let m = Mime::parse("text/markdown").unwrap();
/// assert_eq!(m.essence(), "text/markdown");
/// assert_eq!(m.type_(), "text");
/// assert!(Mime::parse("not-a-mime").is_err());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Mime(String);

/// Error parsing a [`Mime`].
#[derive(Debug, thiserror::Error)]
pub enum MimeError {
    /// The string is not a syntactically valid `type/subtype` MIME.
    #[error("invalid MIME type: {0:?}")]
    Invalid(String),
}

impl Mime {
    /// Parse and normalize a MIME string.
    ///
    /// # Errors
    ///
    /// Returns [`MimeError::Invalid`] when `s` is not a valid
    /// `type/subtype[; params]` MIME.
    pub fn parse(s: impl AsRef<str>) -> Result<Self, MimeError> {
        let raw = s.as_ref().trim();
        let parsed: mime::Mime = raw
            .parse()
            .map_err(|_| MimeError::Invalid(raw.to_string()))?;
        Ok(Self(parsed.to_string()))
    }

    /// `text/plain` — the universal fallback.
    #[must_use]
    pub fn text_plain() -> Self {
        Self("text/plain".to_string())
    }

    /// `application/octet-stream` — unknown binary.
    #[must_use]
    pub fn octet_stream() -> Self {
        Self("application/octet-stream".to_string())
    }

    /// The full normalized MIME string, including parameters.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The `type/subtype` essence, without parameters.
    #[must_use]
    pub fn essence(&self) -> &str {
        self.0.split(';').next().unwrap_or(&self.0).trim_end()
    }

    /// The top-level type (e.g. `text` in `text/html`).
    #[must_use]
    pub fn type_(&self) -> &str {
        self.essence().split('/').next().unwrap_or("")
    }

    /// The subtype (e.g. `html` in `text/html`).
    #[must_use]
    pub fn subtype(&self) -> &str {
        self.essence().split('/').nth(1).unwrap_or("")
    }
}

impl fmt::Display for Mime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for Mime {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl FromStr for Mime {
    type Err = MimeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl Serialize for Mime {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Mime {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::parse(&s).map_err(serde::de::Error::custom)
    }
}

/// Coarse content modality — the routing key for ingest and recall.
///
/// `#[non_exhaustive]` so new modalities are additive (M-FEATURES-ADDITIVE):
/// a host matching on `Modality` must keep a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Plain prose.
    Text,
    /// Source code (language-aware chunking).
    Code,
    /// Markup — HTML / Markdown.
    Markup,
    /// Structured data — CSV / JSON.
    Structured,
    /// A rich non-PDF document.
    Document,
    /// A PDF — bespoke tiered extraction pipeline.
    Pdf,
    /// A raster / vector image.
    Image,
    /// An audio stream.
    Audio,
    /// A video stream.
    Video,
}

/// A MIME paired with its derived [`Modality`].
///
/// `modality` is always [`modality_for_mime`] of `mime`; construct via
/// [`ContentType::classify`] so the two never drift.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentType {
    /// The full MIME type.
    pub mime: Mime,
    /// The coarse modality derived from `mime`.
    pub modality: Modality,
}

impl ContentType {
    /// Classify a MIME into its modality.
    #[must_use]
    pub fn classify(mime: Mime) -> Self {
        let modality = modality_for_mime(&mime);
        Self { mime, modality }
    }
}

/// Map a MIME to its coarse [`Modality`] — the single source of truth.
///
/// Unknown types fall back to [`Modality::Text`], preserving the historical
/// "unrecognized content type → text chunker" behavior.
#[must_use]
pub fn modality_for_mime(m: &Mime) -> Modality {
    let essence = m.essence();
    match essence {
        "application/pdf" => Modality::Pdf,
        "text/html" | "text/markdown" | "application/xhtml+xml" => Modality::Markup,
        "text/csv" | "application/json" | "application/x-ndjson" => Modality::Structured,
        "application/javascript" | "application/typescript" | "text/css" => Modality::Code,
        _ if essence.ends_with("+json") => Modality::Structured,
        // Source code is conventionally `text/x-<lang>` (x-python, x-rust).
        _ if m.type_() == "text" && m.subtype().starts_with("x-") => Modality::Code,
        _ if m.type_() == "image" => Modality::Image,
        _ if m.type_() == "audio" => Modality::Audio,
        _ if m.type_() == "video" => Modality::Video,
        _ if m.type_() == "text" => Modality::Text,
        // Unknown / arbitrary binary — fall back to text handling.
        _ => Modality::Text,
    }
}

/// Map a legacy bare content-type token to a synthetic [`Mime`].
///
/// Bridges the historical `content_type` strings (`"code"`, `"html"`,
/// `"csv"`, `"json"`, `"structured"`) onto the MIME taxonomy so
/// [`select_chunker`](crate) routing is unchanged. **Only** the exact legacy
/// special tokens are recognized; every other input (including `"text"`,
/// `"tool_result"`, and already-full MIME strings) maps to `text/plain`,
/// reproducing the old `match` default arm byte-for-byte.
#[must_use]
pub fn legacy_content_type_to_mime(content_type: &str) -> Mime {
    let synthetic = match content_type {
        "code" => "text/x-code",
        "html" => "text/html",
        "csv" => "text/csv",
        "json" | "structured" => "application/json",
        // "text", "tool_result", "error", "system", real MIMEs, unknown →
        // the old default arm (TextChunker).
        _ => "text/plain",
    };
    Mime::parse(synthetic).expect("static synthetic MIME is valid")
}

/// Best-effort MIME for a legacy artifact `kind` + detected `language`.
///
/// Lifted verbatim from the former `artifact::mime_for_kind` so artifact
/// ingest keeps emitting identical `:ArtifactContent.mime` strings.
#[must_use]
pub fn mime_for_kind(kind: &str, language: Option<&str>) -> Mime {
    let raw = match (kind, language) {
        (_, Some("python")) => "text/x-python",
        (_, Some("rust")) => "text/x-rust",
        (_, Some("javascript")) => "application/javascript",
        (_, Some("typescript")) => "application/typescript",
        (_, Some("tsx")) => "application/typescript",
        (_, Some("html")) => "text/html",
        (_, Some("css")) => "text/css",
        (_, Some("json")) => "application/json",
        (_, Some("csv")) => "text/csv",
        (_, Some("markdown")) => "text/markdown",
        ("markdown", _) => "text/markdown",
        ("html", _) => "text/html",
        ("code", _) => "text/plain",
        _ => "text/plain",
    };
    Mime::parse(raw).expect("static MIME table entry is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_garbage() {
        assert!(Mime::parse("not a mime").is_err());
        assert!(Mime::parse("text").is_err());
    }

    #[test]
    fn essence_drops_params() {
        let m = Mime::parse("text/plain; charset=utf-8").unwrap();
        assert_eq!(m.essence(), "text/plain");
        assert_eq!(m.type_(), "text");
        assert_eq!(m.subtype(), "plain");
    }

    #[test]
    fn modality_table() {
        let cases = [
            ("application/pdf", Modality::Pdf),
            ("text/html", Modality::Markup),
            ("text/markdown", Modality::Markup),
            ("text/csv", Modality::Structured),
            ("application/json", Modality::Structured),
            ("application/ld+json", Modality::Structured),
            ("text/x-python", Modality::Code),
            ("text/x-rust", Modality::Code),
            ("application/javascript", Modality::Code),
            ("image/png", Modality::Image),
            ("audio/mpeg", Modality::Audio),
            ("video/mp4", Modality::Video),
            ("text/plain", Modality::Text),
            ("application/octet-stream", Modality::Text),
        ];
        for (mime, want) in cases {
            let got = modality_for_mime(&Mime::parse(mime).unwrap());
            assert_eq!(got, want, "{mime}");
        }
    }

    #[test]
    fn legacy_tokens_map_like_the_old_match() {
        // Special tokens route to their modality...
        assert_eq!(
            modality_for_mime(&legacy_content_type_to_mime("code")),
            Modality::Code
        );
        assert_eq!(
            modality_for_mime(&legacy_content_type_to_mime("html")),
            Modality::Markup
        );
        for s in ["csv", "json", "structured"] {
            assert_eq!(
                modality_for_mime(&legacy_content_type_to_mime(s)),
                Modality::Structured,
                "{s}"
            );
        }
        // ...everything else falls to Text, like the old default arm.
        for s in [
            "text",
            "tool_result",
            "error",
            "system",
            "text/html",
            "application/octet-stream",
        ] {
            assert_eq!(
                modality_for_mime(&legacy_content_type_to_mime(s)),
                Modality::Text,
                "{s}"
            );
        }
    }

    #[test]
    fn mime_serde_roundtrips_as_string() {
        let m = Mime::parse("application/json").unwrap();
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(json, "\"application/json\"");
        let back: Mime = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }
}
