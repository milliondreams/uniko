//! Data types for the entity extraction pipeline.

// Rust guideline compliant

use serde::{Deserialize, Serialize};

use uniko_store::NodeId;

/// How an entity was extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtractionSource {
    /// Rule-based regex pattern match.
    RuleBased,
    /// Tree-sitter AST extraction from code.
    CodeAst,
    /// ONNX NER model (future).
    OnnxModel,
    /// LLM-enhanced extraction (future).
    Llm,
}

/// Semantic type of an extracted entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// Named person.
    Person,
    /// HTTP/HTTPS URL.
    Url,
    /// Email address.
    Email,
    /// Date or time expression.
    Date,
    /// Numeric value with unit.
    Measurement,
    /// Preference object (from "I prefer X" patterns).
    Preference,
    /// Text in quotation marks.
    QuotedString,
    /// Code symbol: function, class, struct, enum, trait.
    CodeSymbol,
    /// Import or dependency reference from code.
    CodeImport,
    /// Uncategorized entity.
    Other,
}

impl EntityType {
    /// String stored in the graph `entity_type` property.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Url => "url",
            Self::Email => "email",
            Self::Date => "date",
            Self::Measurement => "measurement",
            Self::Preference => "preference",
            Self::QuotedString => "quoted_string",
            Self::CodeSymbol => "code_symbol",
            Self::CodeImport => "code_import",
            Self::Other => "other",
        }
    }
}

/// A raw entity extracted from text before deduplication.
#[derive(Debug, Clone)]
pub struct RawEntity {
    /// Surface form as it appeared in the text.
    pub surface_form: String,
    /// Normalized canonical name for dedup matching.
    pub canonical_name: String,
    /// Semantic entity type.
    pub entity_type: EntityType,
    /// Confidence score in \[0.0, 1.0\].
    pub confidence: f64,
    /// Extraction method.
    pub source: ExtractionSource,
    /// Byte offset in source text where the match starts.
    pub start_byte: usize,
    /// Byte offset in source text where the match ends.
    pub end_byte: usize,
}

/// Result of deduplication mapping to a graph node.
#[derive(Debug, Clone)]
pub struct EntityMatch {
    /// Canonical name of the entity.
    pub canonical_name: String,
    /// Graph node ID of the Entity node.
    pub node_id: NodeId,
    /// Whether the entity already existed in the graph.
    pub was_existing: bool,
    /// Number of mentions in the current content.
    pub mention_count: u32,
}
