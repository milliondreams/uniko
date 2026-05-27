//! LLM-enhanced entity extraction stub.
//!
//! Real implementation deferred until an LLM provider is available.

use super::types::RawEntity;

/// Enhance entity extraction using an LLM provider.
///
/// Currently returns an empty vector.  Real implementation will format
/// a prompt with the text + existing entities, call the provider through
/// the circuit breaker, and parse a structured JSON response.
pub async fn enhance_entities_llm(_text: &str, _existing: &[RawEntity]) -> Vec<RawEntity> {
    Vec::new()
}
