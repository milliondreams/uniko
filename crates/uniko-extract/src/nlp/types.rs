//! Data types for NLP pipeline output.
//!
//! All types derive `Serialize`/`Deserialize` so that [`NlpResult`] can be
//! passed between pipeline steps via `PipelineContext::metadata` as JSON.

// Rust guideline compliant

use serde::{Deserialize, Serialize};

/// Full NLP analysis result for a single text input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlpResult {
    /// Word-level tokens aligned from subword tokenization.
    pub words: Vec<String>,

    /// Per-word NER tag indices into `LabelMaps::ner_labels`.
    pub ner_indices: Vec<usize>,

    /// Per-word POS tag indices into `LabelMaps::pos_labels`.
    pub pos_indices: Vec<usize>,

    /// Per-word dep2label tag indices into `LabelMaps::dep_labels`.
    pub dep_indices: Vec<usize>,

    /// Sentence classification index into `LabelMaps::cls_labels`.
    pub cls_index: usize,

    /// Softmax confidence of the predicted sentence class.
    pub cls_confidence: f32,

    /// Named entity spans decoded from BIO tags.
    pub entities: Vec<NerSpan>,

    /// Dependency arcs decoded from dep2label tags.
    pub dep_arcs: Vec<DepArc>,

    /// Sentence-level classification result.
    pub sentence_class: SentenceClass,
}

/// A contiguous named-entity span merged from BIO tags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerSpan {
    /// Surface text of the entity.
    pub text: String,

    /// Semantic entity type.
    pub entity_type: NerEntityType,

    /// Start word index (inclusive).
    pub start_word: usize,

    /// End word index (exclusive).
    pub end_word: usize,

    /// Average softmax confidence across the span's tokens.
    pub confidence: f32,
}

/// Entity types produced by the NER head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NerEntityType {
    /// Named person.
    Person,
    /// Named organization (company, institution, team).
    Organization,
    /// Geographic location (city, country, region).
    Location,
    /// Miscellaneous named entity.
    Misc,
}

/// A dependency arc between two word-level tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepArc {
    /// Index of the dependent word (0-based).
    pub dependent: usize,

    /// Index of the head word (0-based). `usize::MAX` signals root.
    pub head: usize,

    /// Universal dependency relation label (e.g., "nsubj", "obj").
    pub relation: String,
}

/// Sentence-level classification from the CLS head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SentenceClass {
    /// Declarative factual statement.
    Statement,
    /// Interrogative sentence.
    Question,
    /// Question embedding a factual claim.
    QuestionFact,
    /// Imperative command or instruction.
    Command,
    /// Social greeting.
    Greeting,
    /// Filler or backchannel.
    Filler,
    /// Acknowledgment or confirmation.
    Acknowledgment,
}

impl SentenceClass {
    /// Parse from the label string in `label_maps.json`.
    pub fn from_label(label: &str) -> Self {
        match label {
            "statement" => Self::Statement,
            "question" => Self::Question,
            "question_fact" => Self::QuestionFact,
            "command" => Self::Command,
            "greeting" => Self::Greeting,
            "filler" => Self::Filler,
            "acknowledgment" => Self::Acknowledgment,
            _ => Self::Statement, // conservative fallback
        }
    }

    /// Whether this class indicates informative content.
    pub fn is_informative(self) -> bool {
        matches!(self, Self::Statement | Self::QuestionFact | Self::Command)
    }
}
