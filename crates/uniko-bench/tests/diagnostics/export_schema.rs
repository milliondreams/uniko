//! One-time utility: export the uniko schema to config/schema.json.
//!
//! Run with:
//! `cargo nextest run -p uniko-bench --test diagnostics --run-ignored all \
//!  export_schema --nocapture`

use uni_db::ModelAliasSpec;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

// `#[ignore]` because this writes to a *tracked* file. It is a manual
// regeneration utility, not an assertion: without this attribute every
// `cargo nextest run --workspace` silently rewrote `config/schema.json`
// (bumping `schema_version` and every `created_at`), leaving dirty churn in
// the working tree and mutating an input other tests load via
// `UnikoConfig::schema_path`. Regenerate the snapshot deliberately, as its
// own commit — see `43a4784 "schema: regenerate snapshot from latest ingest
// run"`.
#[ignore = "writes the tracked config/schema.json; run deliberately"]
#[tokio::test]
async fn export_schema_json() {
    let config = UnikoConfig::default();
    let kb = KnowledgeBase::in_memory_with_xervo(config, Vec::<ModelAliasSpec>::new())
        .await
        .unwrap();

    kb.db()
        .save_schema("../../config/schema.json")
        .await
        .unwrap();

    eprintln!("Schema exported to config/schema.json");

    // Verify it's valid by loading it back
    let content = std::fs::read_to_string("../../config/schema.json").unwrap();
    let _: serde_json::Value = serde_json::from_str(&content).unwrap();
    eprintln!("Schema JSON is valid ({} bytes)", content.len());
}
