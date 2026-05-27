//! Embedded model assets loaded at compile time.
//!
//! The tokenizer and label maps are baked into the binary so that NLP
//! inference works without external file dependencies.  Only the ONNX
//! model itself is downloaded from HuggingFace on first use.

use std::sync::OnceLock;

use serde::Deserialize;

/// Raw tokenizer.json bytes (DeBERTa-v3 SPM, ~8 MB).
///
/// Truncation to `max_length=128` is encoded inside the JSON itself, so
/// `Tokenizer::from_bytes` picks it up automatically — no extra
/// `with_truncation` call needed at load time.
static TOKENIZER_BYTES: &[u8] = include_bytes!("assets/tokenizer.json");

/// Raw label_maps.json string (~31 KB).
static LABEL_MAPS_JSON: &str = include_str!("assets/label_maps.json");

static TOKENIZER: OnceLock<tokenizers::Tokenizer> = OnceLock::new();
static LABELS: OnceLock<LabelMaps> = OnceLock::new();

/// Label-to-index mappings for each model head.
#[derive(Debug, Deserialize)]
pub struct LabelMaps {
    /// NER BIO labels (37): O, B-PERSON, I-PERSON, B-ORG, …
    pub ner_labels: Vec<String>,

    /// Universal POS tags (17): ADJ, ADP, ADV, …
    pub pos_labels: Vec<String>,

    /// Universal Dependencies relation labels (53): root, nsubj, obj, …
    ///
    /// Indexed by the biaffine DEP head's `label_scores` argmax.
    pub dep_rel_labels: Vec<String>,

    /// Sentence-act classification labels (8): inform, request, question,
    /// confirm, reject, offer, social, status.
    pub cls_labels: Vec<String>,

    /// SRL BIO labels (42).
    ///
    /// The cascade ships an SRL head we don't currently consume; vendored
    /// for parity with the upstream model so the asset matches what the
    /// model was trained against.
    pub srl_labels: Vec<String>,
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
        assert_eq!(maps.dep_rel_labels.len(), 53);
        assert_eq!(maps.cls_labels.len(), 8);
        assert_eq!(maps.srl_labels.len(), 42);
        assert_eq!(maps.ner_labels[0], "O");
        assert_eq!(maps.cls_labels[0], "inform");
        assert_eq!(maps.dep_rel_labels[0], "root");
        assert!(maps.cls_labels.iter().any(|l| l == "status"));
    }

    #[test]
    fn tokenizer_loads() {
        let tok = tokenizer();
        let encoding = tok.encode("Hello world", false).unwrap();
        assert!(!encoding.get_ids().is_empty());
    }
}
