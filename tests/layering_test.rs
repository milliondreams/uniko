/// Integration test: verify strict linear dependency layering.
///
/// Each crate may only depend on the layer directly below it.
/// This test parses Cargo.toml files and checks that no forbidden
/// uniko-* dependencies are declared.

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

        // Track when we're in a [dependencies] or [dependencies.*] section
        if trimmed.starts_with('[') {
            in_deps = trimmed == "[dependencies]"
                || trimmed.starts_with("[dependencies.");
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
fn test_uniko_extract_depends_only_on_pipes() {
    assert_only_deps("uniko-extract", "crates/uniko-extract", &["uniko-pipes"]);
}

#[test]
fn test_uniko_memory_depends_only_on_extract() {
    assert_only_deps("uniko-memory", "crates/uniko-memory", &["uniko-extract"]);
}

#[test]
fn test_uniko_cortex_depends_only_on_memory() {
    assert_only_deps("uniko-cortex", "crates/uniko-cortex", &["uniko-memory"]);
}

#[test]
fn test_uniko_api_depends_only_on_cortex() {
    assert_only_deps("uniko-api", "crates/uniko-api", &["uniko-cortex"]);
}

#[test]
fn test_uniko_fs_depends_only_on_api() {
    assert_only_deps("uniko-fs", "crates/uniko-fs", &["uniko-api"]);
}

#[test]
fn test_uniko_shell_depends_only_on_api() {
    assert_only_deps("uniko-shell", "crates/uniko-shell", &["uniko-api"]);
}

#[test]
fn test_uniko_mcp_depends_only_on_api() {
    assert_only_deps("uniko-mcp", "crates/uniko-mcp", &["uniko-api"]);
}

#[test]
fn test_uniko_py_depends_only_on_api() {
    assert_only_deps("uniko-py", "bindings/uniko-py", &["uniko-api"]);
}
