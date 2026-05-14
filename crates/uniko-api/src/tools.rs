//! Agent-facing tools.
//!
//! These re-exports surface the subjective-state agent tools defined
//! in `uniko-memory` and downstream crates.  Pipelines handle what can
//! be inferred from messages; tools handle what only the agent can
//! decide to record.

// Rust guideline compliant

pub use uniko_memory::{RecordEpisodeParams, record_episode};
