//! Agent-facing tools.
//!
//! These re-exports surface the subjective-state agent tools defined
//! in `uniko-memory` and downstream crates.  Pipelines handle what can
//! be inferred from messages; tools handle what only the agent can
//! decide to record.

pub use uniko_memory::{
    RecordActionParams, RecordActionResult, RecordEpisodeParams, WorkingMemoryParams,
    nl_to_cypher::{is_safe_read_only, translate as translate_nl_to_cypher},
    policy::{Viewer, filter_bundle, visibility_admits},
    record_action, record_episode,
    rules::{
        AddRuleParams, DecayReport, RuleLifecycleConfig, add_rule, apply_decay_cycle,
        record_rule_match,
    },
    working_memory,
};
