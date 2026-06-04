//! Bench diagnostic and one-off investigation tools.
//!
//! Every member here is `#[ignore]`d (or an export utility) and is run
//! manually against a specific KB snapshot — none execute in CI. They are
//! consolidated into this single binary so the suite links once instead of
//! once per file.
//!
//! Run an individual tool, e.g.:
//! `cargo nextest run -p uniko-bench --test diagnostics graph_debug --run-ignored all --no-capture`

// Shared helpers live in `tests/common/mod.rs`; include them once for the
// whole binary so each member can reach them via `crate::common::*`.
#[path = "../common/mod.rs"]
mod common;

mod dump_obs_chunks;
mod export_schema;
mod graph_debug;
mod recall_debug;
mod review_observations;
mod single_hop_debug;
