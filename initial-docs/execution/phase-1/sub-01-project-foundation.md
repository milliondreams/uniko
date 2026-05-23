# Phase 1: Project Foundation

## Context

This phase establishes the complete Cargo workspace, all crate skeletons, inter-crate dependency wiring, shared infrastructure (error types, IDs, config), CI/CD, and external dependency configuration. Everything built here is foundational -- every subsequent phase depends on the workspace compiling cleanly with correct layering enforced at the dependency level.

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). It follows a 5-layer architecture: Store (L1, uniko-store) -> Pipes (L2, uniko-pipes) -> Extract (L3, uniko-extract) -> Memory (L4, uniko-memory) -> Cortex (L5, uniko-cortex) -> Integration (L6, uniko-fs/shell/mcp), plus uniko-api (facade) and uniko-py (Python binding). The previous 4-layer design was split because content processing steps (NER, chunking, embedding) need the Step trait and pipeline primitives (circuit breaker, retry, DLQ) but those belong at a lower layer than the memory management that orchestrates them. Separating uniko-pipes (generic pipeline machinery) from uniko-extract (content processing steps) and uniko-memory (orchestration, recall, consolidation) resolves this dependency inversion cleanly.

**Key principle:** Strict linear dependency -- each layer calls only the layer directly below it. This is enforced at the Cargo dependency level, not by convention.

## Prerequisites

- Rust toolchain (edition 2024, stable)
- Cargo installed with workspace support
- Access to uni-db crate (local path or registry)
- GitHub repository initialized
- Poetry installed (for Python binding tooling)

## Sub-phases

---

### 1.1 -- Cargo Workspace & Crate Skeletons

**Objective:** Create the root workspace and all 10 crate/binding directories with correct Cargo.toml files and module stubs.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `Cargo.toml` (root) | Config | Workspace definition with `members` list |
| `crates/uniko-store/Cargo.toml` | Config | L1 crate manifest |
| `crates/uniko-store/src/lib.rs` | Rust | L1 module root with public module declarations |
| `crates/uniko-pipes/Cargo.toml` | Config | L2 crate manifest |
| `crates/uniko-pipes/src/lib.rs` | Rust | L2 module root |
| `crates/uniko-extract/Cargo.toml` | Config | L3 crate manifest |
| `crates/uniko-extract/src/lib.rs` | Rust | L3 module root |
| `crates/uniko-memory/Cargo.toml` | Config | L4 crate manifest |
| `crates/uniko-memory/src/lib.rs` | Rust | L4 module root |
| `crates/uniko-cortex/Cargo.toml` | Config | L5 crate manifest |
| `crates/uniko-cortex/src/lib.rs` | Rust | L5 module root |
| `crates/uniko-api/Cargo.toml` | Config | Facade crate manifest |
| `crates/uniko-api/src/lib.rs` | Rust | Facade re-exports only |
| `crates/uniko-fs/Cargo.toml` | Config | L6 FS integration manifest |
| `crates/uniko-fs/src/lib.rs` | Rust | L6 FS module root |
| `crates/uniko-shell/Cargo.toml` | Config | L6 shell binary manifest |
| `crates/uniko-shell/src/main.rs` | Rust | L6 shell entry point |
| `crates/uniko-mcp/Cargo.toml` | Config | L6 MCP server manifest |
| `crates/uniko-mcp/src/lib.rs` | Rust | L6 MCP module root |
| `bindings/uniko-py/Cargo.toml` | Config | Python binding manifest |
| `bindings/uniko-py/src/lib.rs` | Rust | PyO3 module root |

#### Root `Cargo.toml` Workspace Members

```toml
[workspace]
members = [
    "crates/uniko-store",
    "crates/uniko-pipes",
    "crates/uniko-extract",
    "crates/uniko-memory",
    "crates/uniko-cortex",
    "crates/uniko-api",
    "crates/uniko-fs",
    "crates/uniko-shell",
    "crates/uniko-mcp",
    "bindings/uniko-py",
]
resolver = "3"
```

#### Inter-Crate Dependency Rules (Strict Linear Layering)

| Crate | Allowed Dependencies (uniko crates) | Forbidden |
|---|---|---|
| `uniko-store` | uni-db only | All other uniko crates |
| `uniko-pipes` | `uniko-store` only | All other uniko crates |
| `uniko-extract` | `uniko-pipes` only | `uniko-store` (access via uniko-pipes re-exports), `uniko-memory`, `uniko-cortex`, `uniko-api`, L6 crates |
| `uniko-memory` | `uniko-extract` only | `uniko-store`, `uniko-pipes` (access via re-exports), `uniko-cortex`, `uniko-api`, L6 crates |
| `uniko-cortex` | `uniko-memory` only | `uniko-store`, `uniko-pipes`, `uniko-extract` (access via re-exports), `uniko-api`, L6 crates |
| `uniko-api` | `uniko-cortex` only | Direct deps on L1/L2/L3/L4 crates |
| `uniko-fs` | `uniko-api` only | Direct deps on L1/L2/L3/L4/L5 crates |
| `uniko-shell` | `uniko-api` only | Direct deps on L1/L2/L3/L4/L5 crates |
| `uniko-mcp` | `uniko-api` only | Direct deps on L1/L2/L3/L4/L5 crates |
| `uniko-py` | `uniko-api` only | Direct deps on L1/L2/L3/L4/L5 crates |

#### Module Stubs per Crate

**uniko-store (`src/lib.rs`):**
```rust
pub mod schema;
pub mod storage;
pub mod search;
pub mod locy;
```

**uniko-pipes (`src/lib.rs`):**
```rust
pub mod step;
pub mod circuit_breaker;
pub mod retry;
pub mod cancel;
pub mod dead_letter;
pub mod health;
pub mod metrics;
pub mod types;
pub mod config;
```

**uniko-extract (`src/lib.rs`):**
```rust
pub mod ingest;
pub mod ner;
pub mod observations;
pub mod embedding;
```

**uniko-memory (`src/lib.rs`):**
```rust
pub mod pipeline;
pub mod recall;
pub mod rules;
pub mod consolidation;
```

**uniko-cortex (`src/lib.rs`):**
```rust
pub mod procedures;
pub mod topics;
pub mod reasoning;
```

**uniko-api (`src/lib.rs`):**
```rust
pub mod tools;

// Re-exports for downstream consumers
pub use uniko_cortex::*;
```

**uniko-fs (`src/lib.rs`):**
```rust
pub mod watcher;
pub mod shadow;
pub mod git;
```

**uniko-shell (`src/main.rs`):**
```rust
fn main() {
    // Semantic shell entry point
    todo!("Phase 5")
}
```

**uniko-mcp (`src/lib.rs`):**
```rust
pub mod server;
pub mod tools;
```

**uniko-py (`src/lib.rs`):**
```rust
use pyo3::prelude::*;

#[pymodule]
fn uniko(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
```

---

### 1.2 -- Shared Types & Error Handling

**Objective:** Define the foundational types, error enum, ID generation, and configuration struct that all crates depend on.

#### Files to Create

| File | Type | Contents |
|---|---|---|
| `crates/uniko-store/src/error.rs` | Rust | `UnikoError` enum |
| `crates/uniko-store/src/types.rs` | Rust | Shared type aliases and newtypes |
| `crates/uniko-store/src/id.rs` | Rust | UUID v7 generation, deterministic chunk IDs |
| `crates/uniko-store/src/config.rs` | Rust | `UnikoConfig` struct with spec defaults |

#### `error.rs` -- UnikoError

```rust
#[derive(Debug, thiserror::Error)]
pub enum UnikoError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("search error: {0}")]
    Search(String),

    #[error("schema error: {0}")]
    Schema(String),

    #[error("pipeline error: {0}")]
    Pipeline(String),

    #[error("locy error: {0}")]
    Locy(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("timeout after {0}ms")]
    Timeout(u64),

    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, UnikoError>;
```

**Requirements:**
- Derive `thiserror::Error` for all variants
- Implement `From<uni_db::Error>` for `UnikoError::Storage`
- Provide `Result<T>` type alias
- Each variant carries a `String` message (not inner errors, to avoid leaking uni-db types)

#### `types.rs` -- Shared Types

```rust
pub type NodeId = i64;          // uni-db internal node ID
pub type EdgeId = i64;          // uni-db internal edge ID
pub type Timestamp = chrono::DateTime<chrono::Utc>;
pub type AgentId = String;
pub type SessionId = String;
pub type GoalId = String;
pub type TaskId = String;
pub type EmbeddingVec = Vec<f32>;
```

**Requirements:**
- `NodeId` and `EdgeId` match uni-db's internal representation
- `Timestamp` uses `chrono::DateTime<Utc>` for all temporal fields
- String-based IDs for all domain identifiers (goal_id, task_id, etc.)
- `EmbeddingVec` as alias for `Vec<f32>`

#### `id.rs` -- ID Generation

```rust
/// Generate a new UUID v7 (time-sortable, monotonically increasing).
/// Used for all *_id fields when not caller-provided.
pub fn new_id() -> String;

/// Generate a deterministic chunk ID: `{parent_id}:{index}`.
/// Enables idempotent re-chunking.
pub fn chunk_id(parent_id: &str, index: usize) -> String;

/// Validate that a string is a valid UUID v7.
pub fn is_valid_id(id: &str) -> bool;
```

**Requirements:**
- Uses `uuid` crate with `v7` feature enabled
- `new_id()` returns lowercase hex string representation
- `chunk_id()` uses colon separator: `"{parent_id}:{index}"`
- Thread-safe (no global state)

#### `config.rs` -- UnikoConfig

```rust
pub struct UnikoConfig {
    // Pipeline capacities
    pub ingest_queue_capacity: usize,           // default: 200
    pub consolidation_queue_capacity: usize,    // default: 32

    // Consolidation triggers
    pub consolidation_threshold: u32,           // default: 20 (observations)
    pub consolidation_interval_secs: u64,       // default: 900 (15 min)

    // Retry policy
    pub retry_max_attempts: u32,                // default: 3
    pub retry_initial_delay_ms: u64,            // default: 500
    pub retry_max_delay_ms: u64,                // default: 30_000

    // Circuit breaker
    pub circuit_failure_threshold: u32,         // default: 5
    pub circuit_recovery_ms: u64,               // default: 60_000

    // Chunking thresholds
    pub message_chunk_threshold: usize,         // default: 1024 (tokens)
    pub action_output_artifact_threshold: usize, // default: 256 (tokens)

    // Chunk sizing
    pub max_chunk_tokens: usize,                // default: 512
    pub min_chunk_tokens: usize,                // default: 64

    // Memory decay
    pub half_life_days: f64,                    // default: 30.0
    pub prune_below: f64,                       // default: 0.05

    // Recall cascade thresholds
    pub phase1_coverage_threshold: f64,         // default: 0.75
    pub phase2_coverage_threshold: f64,         // default: 0.65
}
```

**Requirements:**
- Implement `Default` with all spec values
- Derive `Debug, Clone, Serialize, Deserialize`
- Builder pattern or `with_*` methods for overriding defaults
- Validate constraints (e.g., `min_chunk_tokens < max_chunk_tokens`, thresholds in [0, 1])

---

### 1.3 -- CI/CD & Dev Tooling

**Objective:** Set up automated quality gates and developer tooling configuration.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `.github/workflows/ci.yml` | YAML | CI pipeline definition |
| `rustfmt.toml` | Config | Formatting rules |
| `clippy.toml` | Config | Lint configuration |
| `deny.toml` | Config | Dependency audit rules |
| `.gitignore` | Config | Ignored files |

#### `.github/workflows/ci.yml` Steps

1. **cargo check** -- compilation verification across all crates
2. **cargo clippy -- -D warnings** -- lint with warnings-as-errors
3. **cargo fmt --check** -- formatting enforcement
4. **cargo nextest run** -- test execution with parallel runner (`-n auto` via nextest)
5. **cargo llvm-cov** -- coverage measurement with LLVM instrumentation

**Trigger:** push to `main`, all pull requests.

**Matrix:** Test on latest stable Rust. Crate matrix: `uniko-store`, `uniko-pipes`, `uniko-extract`, `uniko-memory`, `uniko-cortex`, `uniko-api`, `uniko-fs`, `uniko-shell`, `uniko-mcp`, `uniko-py`.

#### `rustfmt.toml`

```toml
edition = "2021"
max_width = 100
```

#### `clippy.toml`

```toml
cognitive-complexity-threshold = 25
```

#### `deny.toml`

- License allowlist: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unicode-3.0, Zlib
- Advisory database check enabled
- Duplicate crate detection enabled

#### `.gitignore`

```
target/
*.db
.env
*.swp
*.swo
.DS_Store
```

---

### 1.4 -- External Dependency Configuration

**Objective:** Declare all workspace-level dependencies in the root `Cargo.toml` and configure feature flags.

#### Root `Cargo.toml` `[workspace.dependencies]`

| Dependency | Version | Purpose | Primary Crate |
|---|---|---|---|
| `uni-db` | (local path or registry) | Embedded graph database | `uniko-store` |
| `serde` | 1.x, features = ["derive"] | Serialization | `uniko-store` |
| `serde_json` | 1.x | JSON handling | `uniko-store` |
| `chrono` | latest, features = ["serde"] | DateTime handling | `uniko-store` |
| `uuid` | 1.x, features = ["v7"] | UUID v7 generation | `uniko-store` |
| `thiserror` | latest | Error derive macro | `uniko-store` |
| `tokio` | 1.x, features = ["full"] | Async runtime | `uniko-pipes` |
| `tokio-util` | latest | `CancellationToken` | `uniko-pipes` |
| `metrics` | latest | Runtime metrics | `uniko-pipes` |
| `metrics-exporter-prometheus` | latest | Metrics export | `uniko-pipes` |
| `tracing` | latest | Structured logging | `uniko-pipes` |
| `tracing-subscriber` | latest, features = ["env-filter"] | Log output | `uniko-pipes` |
| `tree-sitter` | latest | Code parsing for AST-based chunking | `uniko-extract` |
| `tree-sitter-python` | latest | Python grammar | `uniko-extract` |
| `tree-sitter-rust` | latest | Rust grammar | `uniko-extract` |
| `tree-sitter-javascript` | latest | JavaScript grammar | `uniko-extract` |
| `tree-sitter-typescript` | latest | TypeScript grammar | `uniko-extract` |
| `ort` | latest | ONNX Runtime for local NER model | `uniko-extract` |
| `fastembed` | latest | Local embedding model | `uniko-extract` |
| `tiktoken-rs` | latest | Token counting for chunking | `uniko-extract` |
| `anyhow` | latest | Ad-hoc error handling | (various) |
| `proptest` | latest (dev) | Property-based testing | (dev) |
| `pyo3` | latest, features = ["extension-module"] | Python binding | `uniko-py` |

#### Feature Flags

Defined in root `Cargo.toml` and propagated to relevant crates:

| Flag | Purpose | Crates Affected |
|---|---|---|
| `llm` | Enables LLM-dependent code paths (NL-to-Cypher, LLM NER enhancement, observation LLM path, summarization, rule induction) | `uniko-extract`, `uniko-cortex` |
| `onnx` | Enables ONNX NER model loading via `ort` | `uniko-extract` |

**When `llm` is disabled:** All LLM-dependent pipeline steps skip cleanly with warnings. The system operates in offline/degraded mode (F72).

**When `onnx` is disabled:** NER falls back to rule-based extraction only (regex patterns for proper nouns, dates, numbers). No ONNX model loaded.

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_new_id_uniqueness` | `uniko-store/src/id.rs` | Two calls to `new_id()` never produce the same ID |
| `test_new_id_is_uuid_v7` | `uniko-store/src/id.rs` | Generated ID parses as valid UUID v7 |
| `test_new_id_monotonic` | `uniko-store/src/id.rs` | Sequential IDs are lexicographically ordered |
| `test_chunk_id_format` | `uniko-store/src/id.rs` | `chunk_id("abc", 3)` returns `"abc:3"` |
| `test_chunk_id_deterministic` | `uniko-store/src/id.rs` | Same inputs always produce same output |
| `test_config_defaults` | `uniko-store/src/config.rs` | `UnikoConfig::default()` matches all spec values |
| `test_config_validation` | `uniko-store/src/config.rs` | Invalid configs (min > max tokens) rejected |
| `test_error_display` | `uniko-store/src/error.rs` | All error variants produce readable messages |
| `test_error_from_unidb` | `uniko-store/src/error.rs` | uni-db errors convert to `UnikoError::Storage` |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_workspace_builds` | (cargo check) | All crates compile together |
| `test_dependency_layering` | `tests/layering_test.rs` | Cargo.toml files enforce strict linear deps |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_id_always_valid` | Any generated ID passes `is_valid_id()` |
| `proptest_chunk_id_no_collision` | Different (parent, index) pairs produce different chunk IDs |
| `proptest_config_roundtrip` | Serialize -> deserialize UnikoConfig preserves all fields |

### Build Validation

- `cargo check --workspace` succeeds
- `cargo clippy --workspace -- -D warnings` produces zero warnings
- `cargo fmt --workspace --check` produces zero diffs
- `cargo nextest run --workspace` all tests pass

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Crate-level doc comments | Each `src/lib.rs` | Purpose, layer, dependencies, usage |
| `UnikoError` doc comments | `error.rs` | When each variant is used |
| `UnikoConfig` doc comments | `config.rs` | What each field controls, spec reference |
| `id` module doc comments | `id.rs` | ADR-1 rationale for UUID v7 and deterministic chunk IDs |

---

## Review Checklist

- [ ] Root `Cargo.toml` lists all 10 workspace members
- [ ] Each crate has `Cargo.toml` + `src/lib.rs` (or `src/main.rs` for shell)
- [ ] `uniko-store` depends only on uni-db and external crates (no other uniko crates)
- [ ] `uniko-pipes` depends on `uniko-store` only (no other uniko crates)
- [ ] `uniko-extract` depends on `uniko-pipes` only (no direct `uniko-store`)
- [ ] `uniko-memory` depends on `uniko-extract` only (no direct `uniko-store`, `uniko-pipes`)
- [ ] `uniko-cortex` depends on `uniko-memory` only (no direct `uniko-store`, `uniko-pipes`, `uniko-extract`)
- [ ] `uniko-api` depends on `uniko-cortex` only (re-exports, no logic)
- [ ] L6 crates (`uniko-fs`, `uniko-shell`, `uniko-mcp`) depend on `uniko-api` only
- [ ] `uniko-py` depends on `uniko-api` only
- [ ] `UnikoError` has all 10 variants: Storage, Search, Schema, Pipeline, Locy, Config, Embedding, Llm, Timeout, Internal
- [ ] `UnikoConfig::default()` values match spec exactly (verified by test)
- [ ] UUID v7 generation uses `uuid` crate with `v7` feature
- [ ] Chunk IDs follow `{parent_id}:{index}` format
- [ ] CI runs check, clippy (-D warnings), fmt, nextest, llvm-cov
- [ ] `rustfmt.toml` sets edition 2021, max_width 100
- [ ] `clippy.toml` sets cognitive-complexity-threshold 25
- [ ] `deny.toml` has license allowlist and advisory check
- [ ] `.gitignore` covers target/, *.db, .env
- [ ] All workspace dependencies declared in root `[workspace.dependencies]`
- [ ] Feature flags `llm` and `onnx` defined and documented
- [ ] `cargo check --workspace` succeeds
- [ ] `cargo clippy --workspace -- -D warnings` clean
- [ ] `cargo fmt --workspace --check` clean
- [ ] All unit tests pass

---

## Definition of Done

1. **Workspace compiles:** `cargo check --workspace` succeeds with zero errors.
2. **Layering enforced:** No crate violates the strict 5-layer linear dependency chain (store -> pipes -> extract -> memory -> cortex -> api). Attempting to add `uniko-store` as a direct dependency of `uniko-cortex` would require changing `Cargo.toml` (the Cargo dependency graph is the enforcement mechanism).
3. **All stubs present:** Every crate (10 total) has its module stubs declared in `lib.rs`. Modules contain `// TODO` or empty structs/traits -- they compile but have no implementation.
4. **Shared types functional:** `UnikoError`, `UnikoConfig`, ID generation, and type aliases are implemented and tested.
5. **Config defaults match spec:** `UnikoConfig::default()` returns values matching the spec document exactly, verified by a dedicated test.
6. **CI green:** GitHub Actions CI passes all gates (check, clippy, fmt, test, coverage).
7. **Dev tooling configured:** rustfmt, clippy, cargo-deny all configured with project-specific rules.
8. **Feature flags work:** `cargo check --workspace --no-default-features` compiles (offline mode). `cargo check --workspace --features llm,onnx` compiles (full mode).
9. **No warnings:** Zero clippy warnings, zero rustfmt diffs.
10. **External dependencies pinned:** All workspace dependencies declared with version constraints in root `Cargo.toml`.
