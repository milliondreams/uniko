//! Abductive reasoning: given a conclusion, find minimal supporting facts.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;
use crate::types::NodeId;

/// Result of abductive reasoning.
#[derive(Debug, Clone)]
pub struct AbductionResult {
    /// Facts that support the conclusion.
    pub supporting_facts: Vec<(NodeId, HashMap<String, Value>)>,
    /// Confidence in the abduction.
    pub confidence: f64,
    /// Human-readable explanation.
    pub explanation: String,
}

/// A derivation tree explaining how a rule derived a result.
#[derive(Debug, Clone)]
pub struct DerivationTree {
    /// Rule that produced the derivation.
    pub rule_name: String,
    /// Root node of the tree.
    pub root: DerivationNode,
}

/// A node in a derivation tree.
#[derive(Debug, Clone)]
pub struct DerivationNode {
    /// The MATCH pattern that was satisfied.
    pub pattern: String,
    /// Variable bindings at this node.
    pub bindings: HashMap<String, Value>,
    /// Sub-derivations.
    pub children: Vec<DerivationNode>,
}

impl KnowledgeBase {
    /// Find the minimal set of facts supporting a conclusion.
    ///
    /// Runs the ABDUCE program via Locy and returns the supporting facts.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on evaluation failure.
    pub async fn abduce(
        &self,
        program: &str,
        params: &HashMap<String, Value>,
    ) -> Result<AbductionResult> {
        let session = self.db.session();
        let mut builder = session.locy_with(program);
        for (k, v) in params {
            builder = builder.param(k.as_str(), v.clone());
        }
        let result = builder
            .run()
            .await
            .map_err(|e| UnikoError::Locy(e.to_string()))?;

        let mut supporting = Vec::new();
        if let Some(command_results) = result.rows() {
            for record in command_results {
                let nid = record
                    .get("vid")
                    .and_then(|v: &Value| v.as_i64())
                    .unwrap_or(0);
                supporting.push((nid, record.clone()));
            }
        }

        let count = supporting.len();
        Ok(AbductionResult {
            supporting_facts: supporting,
            confidence: 1.0,
            explanation: format!("Found {count} supporting facts"),
        })
    }
}
