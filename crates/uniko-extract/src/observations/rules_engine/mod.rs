//! Data-driven observation extraction.
//!
//! Replaces the hardcoded [`extract_dep_observations`](crate::nlp::decode::extract_dep_observations)
//! by walking DEP trees against a YAML rule file. Each `Pattern` names an
//! anchor token (matched by POS) and a set of child arcs (matched by
//! relation). Captured spans are rendered through a template into the
//! final observation text.
//!
//! See `rules/english.yml` for the bundled default rule set.

pub mod matcher;
pub mod resolver;
pub mod rules;
pub mod template;

pub use matcher::extract_with_rules;
pub use rules::{Rules, load_bundled_rules, load_rules_from_path};
