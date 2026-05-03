//! Repro: `SchemaBuilder::apply` appends duplicate index definitions
//! and re-runs synchronous rebuilds on every invocation, causing the
//! schema's `indexes` vector to grow **super-linearly** with the
//! number of applies (~`2^N` after N applies of an identical schema).
//!
//! Observed in this repro:
//! - 1 apply  → 1 index  (the canonical state)
//! - 2 applies → 2 indexes (BUG: expected 1)
//! - 10 applies → 2046 indexes (matches `f(N) = 2·f(N−1) + 2`)
//!
//! Why super-linear: each `SchemaBuilder::apply` performs two writes
//! per pending `AddIndex`. First, the explicit `manager.add_index(idx)`
//! at `uni-xervo-0.10/uni/src/api/schema.rs:181`. Then, after `save()`,
//! the loop at `:218-225` triggers `db.indexes().rebuild(label, false)`,
//! which calls `rebuild_indexes_for_label(label)` in
//! `uni-store/src/storage/index_manager.rs:610`. That function clones
//! every existing index for the label and calls `create_*_index(cfg)`
//! on each — and **every** `create_scalar_index` /
//! `create_vector_index` / `create_fts_index` ends with its own
//! `manager.add_index(...)` call (e.g. `index_manager.rs:427-428`).
//! So the full set of "already-present" indexes is re-appended on
//! every apply, on top of the freshly-pushed pending ones.
//!
//! Real-world impact: any consumer that calls `register_schema(...)`
//! more than once for the same logical schema (e.g. on every KB
//! reopen) ends up with N copies of every index after N opens. The
//! schema.json on disk grows linearly. Worse, `SchemaBuilder::apply`
//! at `uni-xervo-0.10/uni/src/api/schema.rs:218-225` then calls
//! `self.db.indexes().rebuild(&label, false).await?` synchronously
//! for every label that had any index added — so every reopen
//! rebuilds every label's indexes from scratch, scaling with the
//! number of duplicate entries. We hit this in production: a
//! `data/kb_llm/conv-26/catalog/schema.json` accumulated to
//! **60,969 indexes** after multiple bench reopens, making subsequent
//! KB opens take 4+ minutes (vs the few seconds expected on a tidy
//! schema).
//!
//! Compare with `add_label_with_desc`/`add_property_with_desc`/
//! `add_edge_type_with_desc`: the matching `SchemaChange` arms in
//! `SchemaBuilder::apply` (`schema.rs:148-153`, `:171-177`,
//! `:200-207`) all match `Err(e) if e.to_string().contains("already
//! exists") => {}` and silently skip. The `AddIndex` arm at
//! `schema.rs:179-187` has no such handling, and `add_index` itself
//! at `uni-common/src/core/schema.rs:1213-1218` unconditionally
//! pushes to the indexes vector with no name-collision check.
//!
//! Expected behaviour: applying a schema definition whose indexes
//! already exist (matched by `IndexDefinition::name()`) should be a
//! no-op for those indexes — neither growing the indexes vector nor
//! triggering a synchronous physical rebuild.

// Rust guideline compliant

use uni_db::{DataType, IndexType, ScalarType, Uni};

/// Define a small schema with one label, one property, and one
/// scalar index. The same definition is applied twice in a single
/// process to expose the duplicate-append bug without any disk I/O
/// or KB reopen.
async fn apply_canonical_schema(db: &Uni) {
    db.schema()
        .label("Foo")
        .property("name", DataType::String)
        .index("name", IndexType::Scalar(ScalarType::Hash))
        .done()
        .apply()
        .await
        .unwrap();
}

#[tokio::test]
async fn add_index_appends_duplicate_on_repeated_apply() {
    let db = Uni::in_memory().build().await.unwrap();

    // First apply: schema is fresh, expect exactly 1 index.
    apply_canonical_schema(&db).await;
    let indexes_after_first = db.schema_manager().schema().indexes.len();
    assert_eq!(
        indexes_after_first, 1,
        "after first apply, schema should have 1 index, got {indexes_after_first}",
    );

    // Second apply with the same definition: expected to be a no-op,
    // but `add_index` unconditionally appends a duplicate.
    apply_canonical_schema(&db).await;
    let indexes_after_second = db.schema_manager().schema().indexes.len();
    assert_eq!(
        indexes_after_second, 1,
        "after second identical apply, schema should still have 1 index, got {indexes_after_second} \
         (BUG: add_index appends without checking for name-collision)",
    );

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn repeated_apply_grows_indexes_linearly() {
    // Quantify the bug: applying the canonical schema N times leaves
    // the schema with N×K indexes (K = indexes per apply, here 1).
    // Demonstrates that the impact compounds with each reopen.
    let db = Uni::in_memory().build().await.unwrap();
    const APPLIES: usize = 10;

    for _ in 0..APPLIES {
        apply_canonical_schema(&db).await;
    }

    let total = db.schema_manager().schema().indexes.len();
    assert_eq!(
        total, 1,
        "after {APPLIES} identical applies, schema should still have 1 index, got {total} \
         (BUG: grows linearly with apply count)",
    );

    db.shutdown().await.unwrap();
}

#[tokio::test]
async fn duplicates_persist_across_reopen_on_disk() {
    // Variant that hits the production failure mode: bloat persists to
    // `catalog/schema.json` and reloads on every open, so subsequent
    // applies pile additional duplicates on top of what was loaded.
    let dir = std::env::temp_dir().join(format!(
        "uni-schema-bloat-repro-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.to_string_lossy().to_string();

    // First open: fresh DB, apply once, expect 1 index, shut down.
    {
        let db = Uni::open(&path).build().await.unwrap();
        apply_canonical_schema(&db).await;
        let n = db.schema_manager().schema().indexes.len();
        assert_eq!(n, 1, "first open: expected 1 index, got {n}");
        db.shutdown().await.unwrap();
    }

    // Second open: schema loaded from disk should be 1; another
    // identical apply should still be 1.
    {
        let db = Uni::open(&path).build().await.unwrap();
        let on_load = db.schema_manager().schema().indexes.len();
        assert_eq!(on_load, 1, "second open (post-load): expected 1 index, got {on_load}");

        apply_canonical_schema(&db).await;
        let after_apply = db.schema_manager().schema().indexes.len();
        assert_eq!(
            after_apply, 1,
            "second open + identical apply: expected 1 index, got {after_apply} \
             (BUG: persists to disk, so each reopen amplifies)",
        );
        db.shutdown().await.unwrap();
    }

    // Cleanup.
    let _ = std::fs::remove_dir_all(&dir);
}
