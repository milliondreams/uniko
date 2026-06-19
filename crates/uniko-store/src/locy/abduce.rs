//! Abductive reasoning: given a conclusion, find the graph modifications that
//! would make it hold.

use std::collections::HashMap;

use uni_db::Value;
use uni_db::locy::CommandResult;

use crate::error::{Result, UnikoError};
use crate::storage::KnowledgeBase;

/// Result of abductive reasoning: ranked, validated graph modifications.
///
/// `ABDUCE <rule> WHERE …` asks "what changes to the graph would satisfy this
/// rule?" and returns each candidate modification, whether it was validated
/// against the goal, and a ranking cost.
#[derive(Debug, Clone)]
pub struct AbductionResult {
    /// The proposed modifications, in the engine's ranked order.
    pub modifications: Vec<AbducedModification>,
}

/// One proposed graph modification with its validation status and cost.
#[derive(Debug, Clone)]
pub struct AbducedModification {
    /// The modification, serialized (`add_edge` / `remove_edge` /
    /// `change_property` plus its variant-specific fields).
    pub modification: serde_json::Value,
    /// Whether applying it (via savepoint) satisfies the ABDUCE goal.
    pub validated: bool,
    /// Ranking cost (`RemoveEdge` = 1.0, `ChangeProperty` = 0.5, `AddEdge` = 1.5).
    pub cost: f64,
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
    /// Find the graph modifications that would make a conclusion hold.
    ///
    /// Runs the `ABDUCE` program via Locy and returns the proposed
    /// modifications (decoded from the `Abduce` command result).
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

        // ABDUCE output lands in a `CommandResult::Abduce`, not `rows()`.
        let modifications = result
            .command_results()
            .iter()
            .find_map(|cr| match cr {
                CommandResult::Abduce(abduction) => Some(abduction),
                _ => None,
            })
            .map(|abduction| {
                abduction
                    .modifications
                    .iter()
                    .map(|vm| AbducedModification {
                        // `Modification` derives `Serialize`; fall back to its
                        // `Debug` form if serialization ever fails.
                        modification: serde_json::to_value(&vm.modification).unwrap_or_else(|_| {
                            serde_json::Value::String(format!("{:?}", vm.modification))
                        }),
                        validated: vm.validated,
                        cost: vm.cost,
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(AbductionResult { modifications })
    }
}
