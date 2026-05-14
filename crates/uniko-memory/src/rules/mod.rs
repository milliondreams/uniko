//! Rule management: stdlib Locy rules and lifecycle.

// Rust guideline compliant

pub mod stdlib;

pub use stdlib::{is_stdlib_rule, register_stdlib_rules, relevance_decay_params};
