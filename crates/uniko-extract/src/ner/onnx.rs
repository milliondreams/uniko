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

            RawEntity {
                surface_form: span.text.clone(),
                canonical_name: uniko_store::text::normalize_canonical(&span.text),
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

#[cfg(all(test, feature = "onnx"))]
mod tests {
    use super::*;
    use crate::ner::dedup::admit_entities;
    use crate::ner::types::EntityType;
    use crate::nlp::types::{NerEntityType, NerSpan, NlpResult, SentenceClass};

    fn span(text: &str, entity_type: NerEntityType, confidence: f32) -> NerSpan {
        NerSpan {
            text: text.to_string(),
            entity_type,
            start_word: 0,
            end_word: 1,
            confidence,
        }
    }

    fn nlp_with(entities: Vec<NerSpan>) -> NlpResult {
        NlpResult {
            words: Vec::new(),
            ner_indices: Vec::new(),
            pos_indices: Vec::new(),
            cls_index: 0,
            cls_confidence: 1.0,
            cls_probs: Vec::new(),
            entities,
            dep_arcs: Vec::new(),
            sentence_class: SentenceClass::Statement,
            srl_frames: Vec::new(),
        }
    }

    /// The key seam: `entities_from_nlp_result` is pure, so the ONNX-side
    /// temporal/numeric filter is testable WITHOUT running the model. A
    /// synthetic NER result with Date/Numeric/low-conf-Other spans must map
    /// + normalize correctly, then be dropped by admission, leaving only the
    /// real Person.
    #[test]
    fn onnx_date_numeric_other_mapped_then_dropped_by_admission() {
        let text = "Caroline met the team yesterday about 5 things at the meeting";
        let nlp = nlp_with(vec![
            span("Caroline", NerEntityType::Person, 0.95),
            span("yesterday", NerEntityType::Date, 0.90),
            span("5", NerEntityType::Numeric, 0.90),
            span("the meeting", NerEntityType::Event, 0.80), // -> Other, below gate
        ]);

        let raw = entities_from_nlp_result(&nlp, text);
        // Mapping + shared normalization (lowercased, no title_case).
        assert_eq!(raw.len(), 4);
        assert!(
            raw.iter()
                .any(|e| e.entity_type == EntityType::Date && e.canonical_name == "yesterday"),
            "Date span mapped + normalized"
        );
        assert!(
            raw.iter()
                .any(|e| e.entity_type == EntityType::Measurement),
            "Numeric -> Measurement"
        );

        // Admission drops Date, Measurement, and the sub-threshold Other;
        // keeps only the normalized Person.
        let kept = admit_entities(raw, true, 0.9);
        assert_eq!(kept.len(), 1, "only the real Person survives");
        assert_eq!(kept[0].entity_type, EntityType::Person);
        assert_eq!(kept[0].canonical_name, "caroline");
    }
}

