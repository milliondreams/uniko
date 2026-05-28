//! Integration test: verify workspace dependency layering.
//!
//! The workspace is not a strict linear chain — `uniko-cortex` and
//! `uniko-extract` are siblings that both build on `uniko-store`, and
//! `uniko-memory` sits above both (the consolidation worker calls
//! cortex P5/P6 post-cycle, see `crates/uniko-cortex/src/lib.rs`).
//! `uniko-api` re-exports the public surface from cortex + memory.
//!
//! Each test pins the set of intra-workspace dependencies a crate is
//! allowed to declare in `[dependencies]`. Add a crate to the allow
//! list deliberately — `uniko-bench` and dev-deps are out of scope.
//!
//! NOTE: this file lives at the workspace root and is not wired into
//! any cargo package, so it does not run as part of `cargo test`. The
//! pins are still load-bearing as machine-checkable documentation; to
//! enforce them in CI, move this file into `crates/uniko-api/tests/`
//! or add a workspace-root package.

fn read_cargo_toml(crate_path: &str) -> String {
    let path = format!("{crate_path}/Cargo.toml");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"))
}

/// Extract uniko-* dependency names from a Cargo.toml's [dependencies] section.
fn uniko_deps(content: &str) -> Vec<String> {
    let mut deps = Vec::new();
    let mut in_deps = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Track when we're in a [dependencies] or [dependencies.*] section.
        // Skip [dev-dependencies] and [build-dependencies].
        if trimmed.starts_with('[') {
            in_deps = trimmed == "[dependencies]" || trimmed.starts_with("[dependencies.");
            continue;
        }

        if in_deps && trimmed.starts_with("uniko-") {
            if let Some(name) = trimmed.split(['=', ' ', '.']).next() {
                deps.push(name.to_string());
            }
        }
    }
    deps
}

fn assert_only_deps(crate_name: &str, crate_path: &str, allowed: &[&str]) {
    let content = read_cargo_toml(crate_path);
    let deps = uniko_deps(&content);
    for dep in &deps {
        assert!(
            allowed.contains(&dep.as_str()),
            "{crate_name} has forbidden dependency: {dep} (allowed: {allowed:?})"
        );
    }
}

#[test]
fn test_uniko_store_no_uniko_deps() {
    assert_only_deps("uniko-store", "crates/uniko-store", &[]);
}

#[test]
fn test_uniko_pipes_depends_only_on_store() {
    assert_only_deps("uniko-pipes", "crates/uniko-pipes", &["uniko-store"]);
}

#[test]
fn test_uniko_extract_depends_on_pipes_store() {
    assert_only_deps(
        "uniko-extract",
        "crates/uniko-extract",
        &["uniko-pipes", "uniko-store"],
    );
}

#[test]
fn test_uniko_cortex_depends_only_on_store() {
    // Cortex is a sibling of extract — both build directly on store.
    // The consolidation worker in `uniko-memory` calls into cortex
    // (P5/P6), which is why memory depends on cortex but not the
    // other way around.
    assert_only_deps("uniko-cortex", "crates/uniko-cortex", &["uniko-store"]);
}

#[test]
fn test_uniko_memory_depends_on_cortex_extract_pipes_store() {
    assert_only_deps(
        "uniko-memory",
        "crates/uniko-memory",
        &["uniko-cortex", "uniko-extract", "uniko-pipes", "uniko-store"],
    );
}

#[test]
fn test_uniko_api_depends_only_on_cortex_memory() {
    // api is the public facade: it re-exports tools defined in
    // uniko-memory (`tools.rs`) and the cortex surface (`lib.rs`
    // wildcards `pub use uniko_cortex::*;`). It cannot drop the
    // direct memory dep because cortex sits below memory in the
    // graph (cycle would result).
    assert_only_deps(
        "uniko-api",
        "crates/uniko-api",
        &["uniko-cortex", "uniko-memory"],
    );
}

#[test]
fn test_uniko_py_depends_only_on_api() {
    assert_only_deps("uniko-py", "bindings/uniko-py", &["uniko-api"]);
}
