//! Store performance / scaling microbenchmarks.
//!
//! All members are `#[ignore]`d and run manually — they measure edge-fetch
//! and growth scaling, some loading embedding models. Consolidated into one
//! binary so they link once.
//!
//! Run: `cargo nextest run -p uniko-store --test perf --run-ignored all`

mod get_edges_scaling_autoembed_repro;
mod get_edges_scaling_repro;
mod observed_in_growth_repro;
