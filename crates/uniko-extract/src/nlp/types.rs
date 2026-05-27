//! Data types for NLP pipeline output.
//!
//! All types derive `Serialize`/`Deserialize` so that [`NlpResult`] can be
//! passed between pipeline steps via `PipelineContext::metadata` as JSON.

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

    /// Sentence classification index into `LabelMaps::cls_labels`.
    pub cls_index: usize,

    /// Softmax confidence of the predicted sentence class.
    pub cls_confidence: f32,

    /// Full softmax probability vector over the 8 raw CLS labels
    /// (`inform`, `request`, `question`, `confirm`, `reject`, `offer`,
    /// `social`, `status`). Lets the gate consider multiple plausible
    /// labels per sentence instead of just the argmax.
    #[serde(default)]
    pub cls_probs: Vec<f32>,

    /// Named entity spans decoded from BIO tags.
    pub entities: Vec<NerSpan>,

    /// Dependency arcs decoded from the biaffine DEP head
    /// (`arc_scores` for head selection, `label_scores` for relation).
    pub dep_arcs: Vec<DepArc>,

    /// Sentence-level classification result.
    pub sentence_class: SentenceClass,

    /// Semantic-role-labelling frames, one per recognised predicate
    /// (typically VERB tokens). Empty when SRL is disabled via
    /// `UnikoConfig.nlp_srl_enabled = false` or when the cascade
    /// found no predicates. Each frame carries the predicate's word
    /// index plus a list of role-typed [`SrlArg`] spans
    /// (`ARG0`, `ARG1`, `ARGM-TMP`, `ARGM-LOC`, …).
    #[serde(default)]
    pub srl_frames: Vec<SrlFrame>,
}

/// One PropBank-style semantic role frame anchored on a predicate.
///
/// Produced by [`crate::nlp::decode::decode_srl_frame`] from the
/// model's `srl_logits` output for a specific `predicate_idx`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrlFrame {
    /// Word index of the predicate (matches the `predicate_idx` input
    /// fed to the model for this frame).
    pub predicate_idx: usize,
    /// Surface form of the predicate (typically the verb).
    pub predicate_word: String,
    /// Argument spans grouped by role. Empty for predicates with no
    /// recognised arguments (e.g. an isolated verb with no nsubj/obj).
    pub args: Vec<SrlArg>,
}

/// One argument of an [`SrlFrame`], identified by its PropBank role
/// label and the contiguous word span it covers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrlArg {
    /// Role label without the BIO prefix — e.g. `"ARG0"`, `"ARG1"`,
    /// `"ARGM-TMP"`, `"ARGM-LOC"`. The predicate itself is excluded
    /// from the args list (its `V` tag identifies it instead).
    pub role: String,
    /// Surface text of the argument span (words joined by single space).
    pub text: String,
    /// Start word index (inclusive).
    pub start_word: usize,
    /// End word index (exclusive).
    pub end_word: usize,
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

/// Sentence-level dialog-act classification from the CLS head.
///
/// Maps the model's 8-label dialog-act vocabulary onto a smaller set of
/// downstream-relevant categories. The model labels are ISO 24617-2
/// inspired: `inform`, `request`, `question`, `confirm`, `reject`,
/// `offer`, `social`, `status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SentenceClass {
    /// Declarative factual statement (`inform`, `status`).
    Statement,
    /// Interrogative sentence (`question`).
    Question,
    /// Imperative or proposal carrying actionable content (`request`, `offer`).
    Command,
    /// Social greeting / phatic turn (`social`).
    Greeting,
    /// Acknowledgment, confirmation, or rejection (`confirm`, `reject`).
    Acknowledgment,
}

impl SentenceClass {
    /// Parse from the label string in `label_maps.json`.
    pub fn from_label(label: &str) -> Self {
        match label {
            "inform" | "status" => Self::Statement,
            "request" | "offer" => Self::Command,
            "question" => Self::Question,
            "confirm" | "reject" => Self::Acknowledgment,
            "social" => Self::Greeting,
            _ => Self::Statement,
        }
    }

    /// Whether this class indicates informative content worth extracting
    /// observations from.
    ///
    /// `Statement` (inform/status) and `Command` (request/offer) carry
    /// propositional content. `Question` asks rather than asserts;
    /// `Acknowledgment` confirms or negates without adding new facts;
    /// `Greeting` is phatic.
    pub fn is_informative(self) -> bool {
        matches!(self, Self::Statement | Self::Command)
    }
}
