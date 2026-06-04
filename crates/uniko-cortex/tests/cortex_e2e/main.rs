//! Cortex end-to-end integration tests.
//!
//! Consolidated binary covering the cortex knowledge-extraction surface:
//! procedural-memory promotion and topic detection. Each former
//! `tests/<name>.rs` file is included below as a module so the suite links
//! once instead of once per file.
//!
//! Run: `cargo nextest run -p uniko-cortex`

mod procedures_e2e;
mod topics_e2e;
