//! Locy logic programming runtime integration.
//!
//! Wraps uni-db's Locy engine for rule management, hypothetical reasoning
//! (ASSUME), and abductive reasoning (ABDUCE).

pub mod abduce;
pub mod assume;
pub mod rules;

use std::collections::HashMap;

use uni_db::Value;

/// A record (single row of bindings) from Locy or Cypher evaluation.
pub type Record = HashMap<String, Value>;

/// Metadata about a registered Locy rule.
#[derive(Debug, Clone)]
pub struct RuleInfo {
    /// Rule name.
    pub name: String,
}

pub use abduce::{AbductionResult, DerivationNode, DerivationTree};
pub use assume::AssumeBuilder;
