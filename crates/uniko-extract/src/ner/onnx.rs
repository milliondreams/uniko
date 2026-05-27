//! ONNX NER model integration.
//!
//! When the `onnx` feature is enabled, converts [`NlpResult`] entity
//! spans into [`RawEntity`] values.  Without the feature, returns an
//! error indicating the model is unavailable.

#[cfg(feature = "onnx")]
use super::types::{EntityType, ExtractionSource, RawEntity};

#[cfg(not(feature = "onnx"))]
use super::types::RawEntity;
#[cfg(not(feature = "onnx"))]
use uniko_store::UnikoError;

/// Convert NLP pipeline entity spans into raw entities.
///
/// Maps [`NerEntityType`](crate::nlp::types::NerEntityType) variants to
/// [`EntityType`] and computes byte offsets by searching for the entity
/// text in the source string.
#[cfg(feature = "onnx")]
pub fn entities_from_nlp_result(
    result: &crate::nlp::types::NlpResult,
    text: &str,
) -> Vec<RawEntity> {
    use crate::nlp::types::NerEntityType;

    result
        .entities
        .iter()
        .map(|span| {
            let entity_type = match span.entity_type {
                NerEntityType::Person => EntityType::Person,
                NerEntityType::Organization => EntityType::Organization,
                NerEntityType::Location => EntityType::Location,
                NerEntityType::Date => EntityType::Date,
                NerEntityType::Numeric => EntityType::Measurement,
                NerEntityType::Event
                | NerEntityType::Product
                | NerEntityType::WorkOfArt
                | NerEntityType::Group
                | NerEntityType::Misc => EntityType::Other,
            };

            // Find byte offsets in source text.
            let (start_byte, end_byte) = text
                .find(&span.text)
                .map(|start| (start, start + span.text.len()))
                .unwrap_or((0, 0));

            // Canonical name: title-case the surface form.
            let canonical = span
                .text
                .split_whitespace()
                .map(|w| {
                    let mut chars = w.chars();
                    match chars.next() {
                        Some(c) => c.to_uppercase().to_string() + &chars.as_str().to_lowercase(),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");

            RawEntity {
                surface_form: span.text.clone(),
                canonical_name: canonical,
                entity_type,
                confidence: span.confidence as f64,
                source: ExtractionSource::OnnxModel,
                start_byte,
                end_byte,
            }
        })
        .collect()
}

/// Extract entities using the ONNX NER model (stub).
///
/// # Errors
///
/// Always returns [`UnikoError::Pipeline`] when the `onnx` feature is
/// disabled.
#[cfg(not(feature = "onnx"))]
pub fn extract_entities_onnx(_text: &str) -> Result<Vec<RawEntity>, UnikoError> {
    Err(UnikoError::Pipeline("ONNX NER model not available".into()))
}
