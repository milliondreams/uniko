//! ONNX NER model stub.
//!
//! Real implementation deferred until a model asset is available.

// Rust guideline compliant

use super::types::RawEntity;
use uniko_store::UnikoError;

/// Extract entities using an ONNX NER model.
///
/// Currently returns an error; the model is not yet bundled.
///
/// # Errors
///
/// Always returns [`UnikoError::Pipeline`] in stub mode.
pub fn extract_entities_onnx(_text: &str) -> Result<Vec<RawEntity>, UnikoError> {
    Err(UnikoError::Pipeline(
        "ONNX NER model not available".into(),
    ))
}
