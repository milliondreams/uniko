//! Shared helpers for the bench integration tests.
//!
//! Each test file is its own crate, so this module is included via
//! `mod common;` from each test that needs it.  Helpers are marked
//! `#[allow(dead_code)]` because any given test file uses only a subset.

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use uni_db::ModelAliasSpec;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

/// Open the workspace KB at `kb_subpath` (relative to the workspace root)
/// using the canonical `config/catalog.json` and `config/schema.json`.
///
/// # Panics
///
/// Panics if the workspace root or KB cannot be opened — intended for
/// `#[ignore]`-gated diagnostic tests that run interactively.
pub async fn load_kb(kb_subpath: &str) -> Arc<KnowledgeBase> {
    let ws = workspace_root();
    std::env::set_current_dir(&ws).expect("cd to workspace root");
    let config = UnikoConfig {
        catalog_path: Some(ws.join("config/catalog.json")),
        schema_path: Some(ws.join("config/schema.json")),
        ..Default::default()
    };
    Arc::new(
        KnowledgeBase::open_with_xervo(ws.join(kb_subpath), config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("open KB"),
    )
}

/// Same as [`load_kb`] but omits `schema_path` so the KB falls back to
/// `register_schema()` programmatically.
///
/// # Panics
///
/// Panics on the same conditions as [`load_kb`].
pub async fn load_kb_no_schema(kb_subpath: &str) -> Arc<KnowledgeBase> {
    let ws = workspace_root();
    std::env::set_current_dir(&ws).expect("cd to workspace root");
    let config = UnikoConfig {
        catalog_path: Some(ws.join("config/catalog.json")),
        ..Default::default()
    };
    Arc::new(
        KnowledgeBase::open_with_xervo(ws.join(kb_subpath), config, Vec::<ModelAliasSpec>::new())
            .await
            .expect("open KB without schema.json"),
    )
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate dir has parent")
        .parent()
        .expect("crates dir has parent")
        .to_path_buf()
}
