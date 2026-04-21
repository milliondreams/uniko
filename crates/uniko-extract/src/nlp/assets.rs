//! Embedded model assets loaded at compile time.
//!
//! The tokenizer and label maps are baked into the binary so that NLP
//! inference works without external file dependencies.  Only the ONNX
//! model itself is downloaded from HuggingFace on first use.

// Rust guideline compliant

use std::sync::OnceLock;

use serde::Deserialize;

/// Raw tokenizer.json bytes (RoBERTa BPE, ~3.5 MB).
static TOKENIZER_BYTES: &[u8] = include_bytes!("assets/tokenizer.json");

/// Raw label_maps.json string (~31 KB).
static LABEL_MAPS_JSON: &str = include_str!("assets/label_maps.json");

static TOKENIZER: OnceLock<tokenizers::Tokenizer> = OnceLock::new();
static LABELS: OnceLock<LabelMaps> = OnceLock::new();

/// Label-to-index mappings for each model head.
#[derive(Debug, Deserialize)]
pub struct LabelMaps {
    /// NER BIO labels (9): O, B-PER, I-PER, B-ORG, …
    pub ner_labels: Vec<String>,

    /// Universal POS tags (17): ADJ, ADP, ADV, …
    pub pos_labels: Vec<String>,

    /// dep2label tags (1440): "+1\@nsubj\@VERB", …
    pub dep_labels: Vec<String>,

    /// Sentence classification labels (7): statement, question, …
    pub cls_labels: Vec<String>,
}

/// Lazily parsed HuggingFace tokenizer from embedded bytes.
///
/// # Panics
///
/// Panics if the embedded `tokenizer.json` is invalid.  This should
/// never happen since it is a compile-time constant.
pub fn tokenizer() -> &'static tokenizers::Tokenizer {
    TOKENIZER.get_or_init(|| {
        tokenizers::Tokenizer::from_bytes(TOKENIZER_BYTES)
            .expect("embedded tokenizer.json must be valid")
    })
}

/// Lazily parsed label maps from embedded JSON.
///
/// # Panics
///
/// Panics if the embedded `label_maps.json` is invalid.
pub fn label_maps() -> &'static LabelMaps {
    LABELS.get_or_init(|| {
        serde_json::from_str(LABEL_MAPS_JSON).expect("embedded label_maps.json must be valid")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_maps_parse_correctly() {
        let maps = label_maps();
        assert_eq!(maps.ner_labels.len(), 37);
        assert_eq!(maps.pos_labels.len(), 17);
        assert_eq!(maps.cls_labels.len(), 9);
        assert!(maps.dep_labels.len() > 1000);
        assert_eq!(maps.ner_labels[0], "O");
        assert_eq!(maps.cls_labels[0], "inform");
    }

    #[test]
    fn tokenizer_loads() {
        let tok = tokenizer();
        let encoding = tok.encode("Hello world", false).unwrap();
        assert!(!encoding.get_ids().is_empty());
    }
}
