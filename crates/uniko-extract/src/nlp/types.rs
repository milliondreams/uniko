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

/// Entity types produced by the NER head (OntoNotes + CoNLL scheme).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NerEntityType {
    /// Named person.
    Person,
    /// Named organization (company, institution, team).
    Organization,
    /// Geographic location (city, country, region, facility, GPE).
    Location,
    /// Date or time expression.
    Date,
    /// Numeric value (money, percent, quantity, cardinal, ordinal).
    Numeric,
    /// Named event.
    Event,
    /// Product name.
    Product,
    /// Work of art (book, song, etc.).
    WorkOfArt,
    /// Nationality, religious, or political group.
    Group,
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
    ///
    /// Handles both the DistilRoBERTa labels (statement, question, etc.)
    /// and the DeBERTa labels (inform, correction, agreement, etc.).
    pub fn from_label(label: &str) -> Self {
        match label {
            // DistilRoBERTa scheme
            "statement" => Self::Statement,
            "question_fact" => Self::QuestionFact,
            "command" => Self::Command,
            "greeting" => Self::Greeting,
            "acknowledgment" => Self::Acknowledgment,
            // DeBERTa scheme (ISO 24617-2 inspired)
            "inform" => Self::Statement,
            "correction" => Self::Statement,
            "plan_commit" => Self::Command,
            "request" => Self::Command,
            "agreement" => Self::Acknowledgment,
            "feedback" => Self::Acknowledgment,
            "social" => Self::Greeting,
            // Shared
            "question" => Self::Question,
            "filler" => Self::Filler,
            _ => Self::Statement,
        }
    }

    /// Whether this class indicates informative content.
    pub fn is_informative(self) -> bool {
        matches!(self, Self::Statement | Self::QuestionFact | Self::Command)
    }
}
