//! # uniko-memory — Layer 4: Memory Management
//!
//! PipelineSystem orchestration, recall cascade (3-phase with coverage gating),
//! consolidation (fact derivation, contradiction, drift), and stdlib rules.
//!
//! Depends on `uniko-extract` only. This is the memory management brain.

// Bumped from the default 128 to clear E0275 overflow when checking
// `Send` on the `consolidation_worker.run()` future in
// `pipeline::mod::PipelineSystem::start`. The future captures
// uni-db's `Session`/`Transaction`, which transitively contain
// datafusion + sqlparser AST types whose `Vec<FunctionArg>` /
// `Vec<ObjectNamePart>` / etc. trait-checking chain hits the default
// limit. 256 is enough; uni-db's own crates set the same.
#![recursion_limit = "256"]

pub mod action;
pub mod consolidation;
pub mod episode;
pub mod llm_triples;
pub mod nl_to_cypher;
pub mod pipeline;
pub mod policy;
pub mod query;
pub mod recall;
pub mod rules;
pub(crate) mod value_convert;
pub mod working_memory;

#[doc(inline)]
pub use action::{RecordActionParams, RecordActionResult, record_action};
#[doc(inline)]
pub use episode::{RecordEpisodeParams, record_episode};
#[doc(inline)]
pub use pipeline::PipelineSystem;
#[doc(inline)]
pub use query::{
    GeneratedAnswer, QueryOutcome, QueryRecordOptions, RecordQueryEpisodeParams, answer_query,
    record_query_episode,
};
#[doc(inline)]
pub use working_memory::{WorkingMemoryParams, working_memory};

/// Sort `items` in descending order by the `f64` key returned by `score`.
///
/// `NaN` scores are treated as `Equal` so the sort is total. Used across
/// the recall cascade and working-memory ranking; centralised here so
/// the `partial_cmp(...).unwrap_or(Equal)` boilerplate lives in exactly
/// one place.
pub(crate) fn sort_by_score_desc<T>(items: &mut [T], score: impl Fn(&T) -> f64) {
    items.sort_by(|a, b| {
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
