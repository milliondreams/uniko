//! Core store integration test suites.
//!
//! Consolidated binary covering schema, storage, search, traversal, blob,
//! BTIC, Locy, migration, and UNWIND-batch behavior. Each former
//! `tests/<name>.rs` file is included as a module so the suite links once
//! instead of once per file.
//!
//! Run: `cargo nextest run -p uniko-store --test store_tests`

mod blob_store_roundtrip;
mod btic_tests;
mod locy_tests;
mod migration_artifact_content;
mod schema_artifact_content;
mod schema_completeness;
mod schema_tests;
mod search_tests;
mod storage_tests;
mod traversal_tests;
mod unwind_batch_test;
