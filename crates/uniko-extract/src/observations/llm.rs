//! LLM-enhanced observation extraction stub.
//!
//! Real implementation deferred until an LLM provider is available.

// Rust guideline compliant

use super::types::RawObservation;

/// Enhance observation extraction using an LLM provider.
///
/// Currently returns an empty vector.  Real implementation will format
/// a prompt with the text + known entities and parse structured JSON
/// responses for additional observations, complex coreferences, and
/// implicit facts.
pub async fn enhance_observations_llm(
    _text: &str,
    _entities: &[(uniko_store::NodeId, String)],
) -> Vec<RawObservation> {
    Vec::new()
}
