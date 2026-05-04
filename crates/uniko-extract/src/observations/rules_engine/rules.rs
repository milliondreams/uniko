//! Rule schema deserialised from YAML.

use std::path::Path;
use std::sync::OnceLock;

use serde::Deserialize;

const BUNDLED_RULES: &str = include_str!("../assets/english.yml");

/// Top-level rules document.
#[derive(Debug, Clone, Deserialize)]
pub struct Rules {
    #[serde(default)]
    pub subject_resolution: SubjectResolution,
    #[serde(default)]
    pub filters: Filters,
    #[serde(default)]
    pub phrase_collectors: std::collections::HashMap<String, PhraseCollector>,
    /// DEP-tree anchored patterns. Each runs against every word whose
    /// POS matches the anchor.
    #[serde(default)]
    pub patterns: Vec<Pattern>,
    /// SRL-frame anchored patterns. Each runs against every
    /// `SrlFrame` on `NlpResult.srl_frames` (one per recognised
    /// predicate). Captures select arguments by PropBank role label
    /// (`ARG0`, `ARG1`, `ARGM-TMP`, `ARGM-LOC`, …).
    #[serde(default)]
    pub srl_patterns: Vec<SrlPattern>,
}

/// Vocabulary that drives pronoun resolution. Lookup logic against
/// [`SentenceContext`](crate::ingest::context::SentenceContext) lives in
/// [`super::resolver`]; this struct only holds the words.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SubjectResolution {
    #[serde(default)]
    pub first_person: PronounSet,
    #[serde(default)]
    pub second_person: PronounSet,
    #[serde(default)]
    pub third_person_demonstrative: PronounSet,
    #[serde(default)]
    pub unresolvable: PronounSet,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PronounSet {
    #[serde(default)]
    pub tokens: Vec<String>,
    #[serde(default)]
    pub contraction_prefixes: Vec<String>,
}

impl PronounSet {
    pub fn matches(&self, lower: &str) -> bool {
        if self.tokens.iter().any(|t| t == lower) {
            return true;
        }
        self.contraction_prefixes
            .iter()
            .any(|p| lower.starts_with(p.as_str()))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Filters {
    #[serde(default = "default_min_content_words")]
    pub min_content_words: usize,
    #[serde(default = "default_subject_pos_required")]
    pub subject_pos_required: Vec<String>,
    #[serde(default = "default_true")]
    pub drop_when_subject_unresolvable: bool,
    /// Per-sentence CLS gate. The model's CLS head emits softmax
    /// probabilities over 8 raw labels (`inform`, `request`, `question`,
    /// `confirm`, `reject`, `offer`, `social`, `status`). A sentence
    /// passes the gate if **any** label whose probability exceeds
    /// `min_prob` is in `informative_labels`. Set `min_prob: 1.0` to
    /// fall back to argmax-only behaviour.
    #[serde(default)]
    pub cls_gate: ClsGate,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClsGate {
    #[serde(default = "default_cls_min_prob")]
    pub min_prob: f32,
    #[serde(default = "default_informative_labels")]
    pub informative_labels: Vec<String>,
}

impl Default for ClsGate {
    fn default() -> Self {
        Self {
            min_prob: default_cls_min_prob(),
            informative_labels: default_informative_labels(),
        }
    }
}

fn default_cls_min_prob() -> f32 {
    0.20
}

fn default_informative_labels() -> Vec<String> {
    vec![
        "inform".into(),
        "status".into(),
        "request".into(),
        "offer".into(),
    ]
}

impl Default for Filters {
    fn default() -> Self {
        Self {
            min_content_words: default_min_content_words(),
            subject_pos_required: default_subject_pos_required(),
            drop_when_subject_unresolvable: true,
            cls_gate: ClsGate::default(),
        }
    }
}

fn default_min_content_words() -> usize {
    3
}

fn default_subject_pos_required() -> Vec<String> {
    vec!["NOUN".into(), "PROPN".into(), "PRON".into()]
}

fn default_true() -> bool {
    true
}

/// Named phrase-collection strategy.
///
/// `include_relations` lists the DEP relations that are walked through
/// when gathering tokens around an anchor word. The special collector
/// `subtree` (no `include_relations`) walks every dependent.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct PhraseCollector {
    #[serde(default)]
    pub include_relations: Vec<String>,
    /// If true, walk every dependent ignoring `include_relations`.
    #[serde(default)]
    pub subtree: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Pattern {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "match")]
    pub match_: MatchSpec,
    pub template: String,
    #[serde(default)]
    pub quality: PatternQuality,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PatternQuality {
    #[serde(default)]
    pub min_modifier_words: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MatchSpec {
    pub anchor: AnchorSpec,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnchorSpec {
    /// Required POS for the anchor token. Single string or array.
    pub pos: StringOrList,
    #[serde(default)]
    pub children: Vec<ChildSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChildSpec {
    /// Required DEP relations for the child arc. Single string or array.
    pub dep: StringOrList,
    #[serde(default)]
    pub required: bool,
    /// Capture name. Tokens get exposed to the template as `{name}`.
    /// If `None`, the child only acts as a structural witness (e.g. `cop`).
    #[serde(default)]
    pub capture: Option<String>,
    /// Phrase collector to use for the captured span.
    #[serde(default = "default_collect")]
    pub collect: String,
    /// If true, multiple matching children are concatenated. If false,
    /// the first one wins.
    #[serde(default)]
    pub multi: bool,
}

fn default_collect() -> String {
    "noun_phrase".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrList {
    One(String),
    Many(Vec<String>),
}

impl StringOrList {
    pub fn contains(&self, s: &str) -> bool {
        match self {
            StringOrList::One(x) => x == s,
            StringOrList::Many(xs) => xs.iter().any(|x| x == s),
        }
    }
}

/// SRL-frame anchored extraction pattern.
///
/// Unlike [`Pattern`] (which walks DEP arcs from a POS-typed anchor
/// word), an `SrlPattern` matches against an entire `SrlFrame` from
/// the cascade. Every frame is tested against every pattern; captures
/// select arguments by role label, the predicate's surface text is
/// always available as `{predicate}`, and the speaker is the default
/// fallback for `{subject}`.
#[derive(Debug, Clone, Deserialize)]
pub struct SrlPattern {
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Argument captures, each mapping a PropBank role to a template
    /// variable name. Order matters only for human readability.
    #[serde(default)]
    pub captures: Vec<SrlCapture>,
    /// Render template, e.g. `"{subject} {predicate} {object}"`.
    /// `{predicate}` and `{subject}` are always available; other
    /// variables come from `captures[*].as_name`.
    pub template: String,
    /// Optional secondary templates conditional on which captures
    /// were filled. The matcher picks the longest template whose
    /// referenced captures are all non-empty. Lets a single pattern
    /// emit *"X gave Y"*, *"X gave Y on D"*, *"X gave Y at L"*, and
    /// *"X gave Y on D at L"* without four separate patterns.
    #[serde(default)]
    pub template_alternatives: Vec<String>,
}

/// One argument capture in an [`SrlPattern`].
#[derive(Debug, Clone, Deserialize)]
pub struct SrlCapture {
    /// PropBank role to match (`"ARG0"`, `"ARGM-TMP"`, etc.). Special
    /// value `"V"` captures the predicate's surface text — usually
    /// redundant with the auto-injected `{predicate}` variable.
    pub role: String,
    /// Template variable name to expose this argument under.
    #[serde(rename = "as")]
    pub as_name: String,
    /// When `true`, the pattern is dropped if this role is missing
    /// from the frame. When `false`, missing → empty `{var}` and the
    /// template's whitespace squeeze handles the gap.
    #[serde(default)]
    pub required: bool,
}

/// Bundled default rules — parsed once and cached.
pub fn load_bundled_rules() -> &'static Rules {
    static CACHE: OnceLock<Rules> = OnceLock::new();
    CACHE.get_or_init(|| {
        serde_yaml::from_str(BUNDLED_RULES).expect("bundled english.yml must parse")
    })
}

/// Parse a rules file from disk. Caller is responsible for caching.
pub fn load_rules_from_path(path: &Path) -> std::io::Result<Rules> {
    let text = std::fs::read_to_string(path)?;
    serde_yaml::from_str(&text).map_err(|e| std::io::Error::other(e.to_string()))
}
