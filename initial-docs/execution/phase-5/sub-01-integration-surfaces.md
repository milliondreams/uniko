# Phase 17: FS/Shell/Git Integration, Organization Support, & Python Binding

## Context

This phase builds Layer 4 integration surfaces — the shadow filesystem, git integration, semantic shell, organization/team multi-tenant support, and the Python binding via PyO3. These extend uniko beyond a core memory system into practical developer tooling. After Phase 16 has validated the core system against benchmarks, these integrations make uniko usable in real-world development workflows: file systems become knowledge-aware, git history becomes episodic memory, shell commands become semantic searches, organizations get access-controlled shared knowledge, and Python developers get native bindings.

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). The 4-layer architecture is: KnowledgeBase (L1, uniko-store) -> Extract (L2, uniko-extract) -> Pipes (uniko-pipes) + Memory (uniko-memory) + Cortex (L3, uniko-cortex) -> Integration (L4, uniko-fs/shell/mcp). This phase builds the L4 crates (`uniko-fs`, `uniko-shell`) and the Python binding (`uniko-py`), plus organization-level access control in `uniko-memory`.

**Key principle:** Integration surfaces depend on `uniko-api` only (the facade crate). They never reach into L1/L2/L3 directly. All functionality is accessed through the public `Uniko` API, ensuring that integration code is insulated from internal changes.

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 16 (Benchmarks) | Complete | Core system validated — all pipelines working, recall cascade proven, consolidation effective |
| Phase 11 (Agent Tools) | Complete | All 12 agent tools implemented: ingest_message, record_episode, recall, etc. |
| Phase 15 (MCP Server) | Complete | Full external API surface via MCP protocol |
| `uniko-api` facade | Complete | Public `Uniko` struct with all methods exposed |
| `notify` crate | Available | File system event watching (cross-platform) |
| `git2` crate | Available | Git repository access (libgit2 bindings) |
| `clap` crate | Available | Command-line argument parsing |
| `pyo3` crate | Available | Python <-> Rust FFI |
| `maturin` | Available | PyO3 build system for pip-installable packages |
| `tokio` 1.x | Available | Async runtime |

## Sub-phases

---

### 17.1 — Shadow Filesystem (uniko-fs)

**Objective:** Build a filesystem watcher that automatically syncs directory contents into uniko's knowledge graph. File creates, modifications, and deletions are detected in real-time and reflected as Artifact nodes with full pipeline processing (P1 chunking → P2 NER → P3 observations → P7 embeddings).

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-fs/src/lib.rs` | Rust | Module root, re-exports |
| `crates/uniko-fs/src/shadow.rs` | Rust | `ShadowFs` struct, sync + watch logic |
| `crates/uniko-fs/src/ignore.rs` | Rust | Gitignore + custom ignore pattern matching |
| `crates/uniko-fs/src/detect.rs` | Rust | File type detection, language identification |

#### `shadow.rs` — ShadowFs

```rust
/// Shadow filesystem: watches a directory tree and mirrors its contents
/// into uniko's knowledge graph as Artifact nodes.
///
/// Each file becomes an Artifact with:
///   - kind: detected from extension/content ("file", "document", "config", "image", etc.)
///   - path: relative to the watched root
///   - content: file text (for text files)
///   - language: detected programming language (for code)
///   - hash: SHA-256 content hash for change detection
///
/// On file modification, the existing Artifact is updated (not duplicated).
/// Chunks are re-generated, old chunks deleted, new chunks linked.
pub struct ShadowFs {
    /// File system event watcher.
    watcher: RecommendedWatcher,
    /// Uniko instance for graph operations.
    uniko: Arc<Uniko>,
    /// Root directory being watched.
    root: PathBuf,
    /// Ignore pattern matcher (gitignore + custom).
    ignore: IgnoreMatcher,
    /// Debounce duration to coalesce rapid file changes.
    debounce_ms: u64,
    /// Cancel token for graceful shutdown.
    cancel: CancellationToken,
}
```

Functions:

- `ShadowFs::new(uniko: Arc<Uniko>, root: PathBuf, config: ShadowFsConfig) -> Result<Self>` — Creates the watcher but does not start watching. Loads .gitignore and custom ignore patterns.
- `async fn sync(&self) -> Result<SyncStats>` — Full initial sync: walk the directory tree, ingest every non-ignored file as an Artifact. Returns statistics (files_synced, files_ignored, bytes_processed).
- `async fn watch(&mut self) -> Result<()>` — Start the file watcher. Runs until cancellation. Processes events in a loop with debouncing.
- `async fn handle_event(&self, event: Event) -> Result<()>` — Process a single filesystem event.
- `async fn ingest_file(&self, path: &Path) -> Result<NodeId>` — Read file, detect type/language, create or update Artifact node, trigger pipeline.
- `async fn remove_file(&self, path: &Path) -> Result<()>` — Delete Artifact node and all associated Chunks when a file is deleted.
- `async fn update_file(&self, path: &Path) -> Result<NodeId>` — Re-read file, compute hash, if changed: update Artifact content, re-chunk, re-embed.

```rust
pub struct ShadowFsConfig {
    /// Custom ignore patterns (in addition to .gitignore).
    pub ignore_patterns: Vec<String>,
    /// Debounce duration for coalescing rapid changes (ms).
    pub debounce_ms: u64,             // default: 200
    /// Maximum file size to ingest (bytes). Larger files are skipped.
    pub max_file_size: u64,           // default: 10_000_000 (10MB)
    /// Whether to follow symlinks.
    pub follow_symlinks: bool,        // default: false
    /// File extensions to include (empty = all non-ignored).
    pub include_extensions: Vec<String>,
}

pub struct SyncStats {
    pub files_synced: u64,
    pub files_ignored: u64,
    pub files_failed: u64,
    pub bytes_processed: u64,
    pub duration_ms: u64,
}
```

#### Event Handling

| Event Type | Action | Graph Operation |
|---|---|---|
| Create | Ingest new file | Create Artifact → P1 chunk → P2 NER → P3 observations → P7 embed |
| Modify | Update existing file | Update Artifact content/hash → re-chunk → re-process pipeline |
| Delete | Remove file | Delete Artifact + all HAS_CHUNK edges + Chunk nodes |
| Rename | Update path | Update Artifact.path property |

#### Debouncing

```
File events arrive from the OS watcher. Editors often generate multiple events
for a single save (write temp file, rename, delete old):

  Event: Create .file.tmp    → debounce buffer
  Event: Rename .file.tmp → file.rs  → debounce buffer
  Event: Delete file.rs~     → debounce buffer

After debounce_ms (200ms) with no new events for the same path:
  → Coalesce to single "Modify file.rs" event
  → Process once
```

#### Latency Target

File change detection + re-index: < 100ms for detecting the change. Full pipeline processing (P1-P7) runs async and may take longer, but the watcher must not block on pipeline completion.

#### `ignore.rs` — Ignore Pattern Matching

```rust
/// Matcher for determining which files to ignore during sync/watch.
pub struct IgnoreMatcher {
    /// Patterns loaded from .gitignore (if present).
    gitignore_patterns: Vec<Pattern>,
    /// Custom patterns from ShadowFsConfig.
    custom_patterns: Vec<Pattern>,
}
```

Functions:

- `IgnoreMatcher::new(root: &Path, custom_patterns: &[String]) -> Result<Self>` — Load .gitignore from root (and parent dirs), add custom patterns.
- `fn is_ignored(&self, path: &Path) -> bool` — Check if a path matches any ignore pattern.

Default ignored patterns (always applied):

```
.git/
target/
node_modules/
__pycache__/
*.pyc
.DS_Store
*.swp
*.swo
.env
```

#### `detect.rs` — File Type Detection

```rust
/// Detect the kind and language of a file from its path and content.
pub fn detect_file_type(path: &Path, content: &[u8]) -> FileType;

pub struct FileType {
    /// Artifact kind: "file", "document", "config", "image", "audio", "video"
    pub kind: String,
    /// Programming language (if code): "rust", "python", "javascript", etc.
    pub language: Option<String>,
    /// MIME type.
    pub mime_type: String,
    /// Whether content is text (readable) or binary.
    pub is_text: bool,
}
```

Language detection by extension:

| Extension | Language | Kind |
|---|---|---|
| `.rs` | rust | file |
| `.py` | python | file |
| `.js`, `.jsx` | javascript | file |
| `.ts`, `.tsx` | typescript | file |
| `.go` | go | file |
| `.java` | java | file |
| `.c`, `.h` | c | file |
| `.cpp`, `.hpp` | cpp | file |
| `.md`, `.rst`, `.txt` | — | document |
| `.json`, `.yaml`, `.yml`, `.toml` | — | config |
| `.png`, `.jpg`, `.gif`, `.svg` | — | image |
| `.mp3`, `.wav`, `.flac` | — | audio |
| `.mp4`, `.avi`, `.mov` | — | video |

---

### 17.2 — Git Integration (uniko-fs)

**Objective:** Map git repository history into uniko's episodic memory. Each commit becomes an Episode node, with relationships to changed files (Entities), and standard git operations (blame, log) are enhanced with memory context.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-fs/src/git.rs` | Rust | `GitIntegration` struct, commit mapping, enhanced blame |

#### `git.rs` — GitIntegration

```rust
/// Git repository integration for uniko.
/// Maps commits to Episodes, files to Entities, and enhances
/// standard git operations with memory context.
pub struct GitIntegration {
    /// libgit2 repository handle.
    repo: Repository,
    /// Uniko instance for graph operations.
    uniko: Arc<Uniko>,
}
```

Functions:

- `GitIntegration::new(repo_path: &Path, uniko: Arc<Uniko>) -> Result<Self>` — Open repository, validate it's a valid git repo.
- `async fn map_commits(&self, since: Option<DateTime<Utc>>) -> Result<CommitMapStats>` — Walk commit history (optionally since a date), create Episode nodes for each commit.
- `async fn map_single_commit(&self, commit: &Commit) -> Result<NodeId>` — Create one Episode from one commit.
- `async fn enhanced_blame(&self, path: &str) -> Result<Vec<EnhancedBlameLine>>` — Standard blame enriched with memory context.
- `async fn enhanced_log(&self, path: Option<&str>, limit: usize) -> Result<Vec<EnhancedLogEntry>>` — Git log enriched with episode and fact context.
- `async fn watch_new_commits(&self) -> Result<()>` — Watch for new commits (poll HEAD) and map them as they arrive.

#### Commit → Episode Mapping

```
For each commit:
  1. Create Episode node:
       episode_id: commit SHA (deterministic, enables idempotent re-mapping)
       action_type: "commit"
       outcome: "success"
       state: { "commit_message": "...", "author": "...", "date": "..." }
       delta: { "files_changed": [...], "insertions": N, "deletions": N }
       timestamp: commit.time()

  2. Create/lookup Participant for commit author:
       participant_id: author email
       kind: "human"
       name: author name

  3. Create RECORDED_BY edge: Episode → Participant

  4. For each changed file in the commit:
       a. Create/lookup Entity node:
            entity_id: file path (deterministic, enables dedup)
            name: file name
            entity_type: "file"
       b. Create MENTIONS edge: Episode → Entity

  5. Create FOLLOWED_BY edge to previous commit's Episode (if exists):
       gap_ms: time difference between commits in milliseconds
```

```rust
pub struct CommitMapStats {
    pub commits_mapped: u64,
    pub commits_skipped: u64,  // already mapped (idempotent check)
    pub entities_created: u64, // new file entities
    pub participants_created: u64,
    pub duration_ms: u64,
}
```

#### Enhanced Blame

```rust
pub struct EnhancedBlameLine {
    /// Standard blame fields.
    pub line_number: usize,
    pub content: String,
    pub commit_sha: String,
    pub author: String,
    pub date: DateTime<Utc>,
    /// Enhanced fields from uniko memory.
    pub episode_id: Option<String>,       // Episode node for this commit
    pub related_facts: Vec<String>,       // Facts mentioning this file
    pub commit_observations: Vec<String>, // Observations from the commit context
}
```

```
For each blame line:
  1. Standard blame: commit SHA, author, date, content
  2. Lookup Episode by episode_id = commit SHA
  3. Traverse Episode → MENTIONS → Entity (files) → ABOUT ← Fact
     → Collect Facts about this file
  4. Traverse Episode → IN_SESSION → Session → Observation
     → Collect Observations from the session context
  5. Return enriched blame line
```

#### Enhanced Log

```rust
pub struct EnhancedLogEntry {
    /// Standard log fields.
    pub commit_sha: String,
    pub author: String,
    pub date: DateTime<Utc>,
    pub message: String,
    pub files_changed: Vec<String>,
    /// Enhanced fields from uniko memory.
    pub episode_id: Option<String>,
    pub related_episodes: Vec<String>,    // Other episodes mentioning same entities
    pub facts_at_time: Vec<String>,       // Facts valid at commit time (BTIC query)
    pub task_context: Option<String>,     // Task this commit was part of (Episode → FOR_TASK → Task)
}
```

---

### 17.3 — Semantic Shell (uniko-shell)

**Objective:** Build a set of semantic shell commands that enhance standard Unix tools (grep, find, cat, diff, blame, ls, log) with uniko's knowledge graph. Each command queries the memory system and presents enriched output.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-shell/Cargo.toml` | Config | Binary crate, depends on `uniko-api`, `clap` |
| `crates/uniko-shell/src/main.rs` | Rust | Entry point, command dispatch |
| `crates/uniko-shell/src/commands/mod.rs` | Rust | Command module root |
| `crates/uniko-shell/src/commands/grep.rs` | Rust | `ug` — semantic grep |
| `crates/uniko-shell/src/commands/find.rs` | Rust | `uf` — semantic find |
| `crates/uniko-shell/src/commands/cat.rs` | Rust | `uc` — semantic cat |
| `crates/uniko-shell/src/commands/diff.rs` | Rust | `ud` — semantic diff |
| `crates/uniko-shell/src/commands/blame.rs` | Rust | `ub` — semantic blame |
| `crates/uniko-shell/src/commands/ls.rs` | Rust | `ul` — semantic ls |
| `crates/uniko-shell/src/commands/log.rs` | Rust | `ulog` — semantic log |
| `crates/uniko-shell/src/output.rs` | Rust | Output formatting (plain, JSON, colored) |

#### `main.rs` — Entry Point

```rust
#[derive(Parser)]
#[command(name = "uniko", about = "Semantic shell for uniko cognitive memory")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to uniko database.
    #[arg(long, env = "UNIKO_DB")]
    db: Option<PathBuf>,

    /// Output format.
    #[arg(long, default_value = "plain")]
    format: OutputFormat,
}

#[derive(Subcommand)]
enum Commands {
    /// Semantic grep: search using graph + vector + fulltext.
    #[command(alias = "ug")]
    Grep(GrepArgs),

    /// Semantic find: find files by semantic similarity.
    #[command(alias = "uf")]
    Find(FindArgs),

    /// Semantic cat: display file with knowledge context.
    #[command(alias = "uc")]
    Cat(CatArgs),

    /// Semantic diff: diff with provenance.
    #[command(alias = "ud")]
    Diff(DiffArgs),

    /// Semantic blame: enhanced blame with memory context.
    #[command(alias = "ub")]
    Blame(BlameArgs),

    /// Semantic ls: list files with knowledge annotations.
    #[command(alias = "ul")]
    Ls(LsArgs),

    /// Semantic log: git log enriched with episode context.
    #[command(alias = "ulog")]
    Log(LogArgs),
}
```

#### Command Specifications

**`ug` (uniko grep):**

```rust
pub struct GrepArgs {
    /// Search query (natural language or keywords).
    pub query: String,
    /// Maximum results to return.
    #[arg(short = 'n', default_value = "10")]
    pub limit: usize,
    /// Search scope: "all", "code", "docs", "messages", "facts".
    #[arg(short, default_value = "all")]
    pub scope: String,
}
```

Search flow:
1. `recall(query, budget=4096)` — Uses recall cascade (graph → vector → fulltext)
2. For each result in ContextBundle: display matching content with file path, line numbers, relevance score
3. Group results by source type (Chunk, Fact, Observation, Message)

**`uf` (uniko find):**

```rust
pub struct FindArgs {
    /// What to find (natural language description).
    pub query: String,
    /// File type filter.
    #[arg(short = 't')]
    pub file_type: Option<String>,
    /// Maximum results.
    #[arg(short = 'n', default_value = "20")]
    pub limit: usize,
}
```

Search flow:
1. Embed query → vector search on Artifact.text_embedding
2. Filter by kind/language if file_type specified
3. Display: file path, similarity score, brief description (from Artifact summary or first chunk)

**`uc` (uniko cat):**

```rust
pub struct CatArgs {
    /// File path to display.
    pub path: String,
    /// Show related entities.
    #[arg(long)]
    pub entities: bool,
    /// Show related facts.
    #[arg(long)]
    pub facts: bool,
    /// Show observations.
    #[arg(long)]
    pub observations: bool,
    /// Show all context (entities + facts + observations).
    #[arg(long, short = 'a')]
    pub all: bool,
}
```

Display:
1. Standard file content (with syntax highlighting if terminal supports it)
2. If `--entities` or `--all`: list entities mentioned in this file (Artifact → HAS_CHUNK → Chunk → MENTIONS → Entity)
3. If `--facts` or `--all`: list facts about entities in this file (Entity → ABOUT ← Fact)
4. If `--observations` or `--all`: list observations from this file (Chunk → OBSERVED_IN ← Observation)

**`ud` (uniko diff):**

```rust
pub struct DiffArgs {
    /// First file or commit.
    pub a: String,
    /// Second file or commit.
    pub b: String,
    /// Show provenance (why changes were made).
    #[arg(long)]
    pub provenance: bool,
}
```

Display:
1. Standard diff output
2. If `--provenance`: for each changed section, find Episodes that caused the change (Artifact → MODIFIED_BY → Action → IN_SESSION → Session), display commit message + session context

**`ub` (uniko blame):**

Uses `GitIntegration::enhanced_blame()` from 17.2.

**`ul` (uniko ls):**

```rust
pub struct LsArgs {
    /// Directory to list.
    #[arg(default_value = ".")]
    pub path: String,
    /// Show knowledge annotations (entity count, fact count).
    #[arg(long, short = 'k')]
    pub knowledge: bool,
}
```

Display:
1. Standard ls output (file name, size, date)
2. If `--knowledge`: for each file, show entity_count (how many entities extracted), fact_count (how many facts reference entities from this file), last_modified_episode (most recent Episode that modified this file)

**`ulog` (uniko log):**

Uses `GitIntegration::enhanced_log()` from 17.2.

#### `output.rs` — Output Formatting

```rust
pub enum OutputFormat {
    Plain,  // Human-readable text
    Json,   // Structured JSON
    Color,  // Colored terminal output
}

pub fn format_results(results: &[SearchResult], format: OutputFormat) -> String;
pub fn format_blame(blame: &[EnhancedBlameLine], format: OutputFormat) -> String;
pub fn format_log(entries: &[EnhancedLogEntry], format: OutputFormat) -> String;
```

---

### 17.4 — Organization & Access Control

**Objective:** Implement organization and team multi-tenant support with access control enforcement. Queries are scoped to the caller's organization by default. Facts can have visibility scope (agent, team, org, global), and cross-agent knowledge sharing is controlled.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/organization/mod.rs` | Rust | Module root, CRUD operations |
| `crates/uniko-memory/src/organization/access.rs` | Rust | Access control enforcement, scope filtering |
| `crates/uniko-memory/src/organization/sharing.rs` | Rust | Cross-agent knowledge sharing |

#### `mod.rs` — Organization & Team CRUD

```rust
/// Organization management operations.
/// Organizations are the top-level grouping for multi-tenant isolation.
pub struct OrgManager {
    kb: Arc<KnowledgeBase>,
}
```

Functions:

- `async fn create_org(&self, name: &str) -> Result<OrgId>` — Create Organization node.
- `async fn create_team(&self, org_id: &str, name: &str, purpose: &str) -> Result<TeamId>` — Create Team node with TEAM_IN_ORG edge.
- `async fn add_member(&self, org_id: &str, participant_id: &str, role: &str) -> Result<()>` — Create MEMBER_OF edge: Participant → Organization.
- `async fn add_to_team(&self, team_id: &str, participant_id: &str) -> Result<()>` — Create PART_OF_TEAM edge: Participant → Team.
- `async fn remove_member(&self, org_id: &str, participant_id: &str) -> Result<()>` — Remove MEMBER_OF edge.
- `async fn list_members(&self, org_id: &str) -> Result<Vec<Member>>` — List all members of an organization with roles.
- `async fn list_teams(&self, org_id: &str) -> Result<Vec<Team>>` — List all teams in an organization.
- `async fn get_participant_orgs(&self, participant_id: &str) -> Result<Vec<OrgMembership>>` — Get all organizations a participant belongs to.

```rust
pub struct Member {
    pub participant_id: String,
    pub name: String,
    pub role: String,
    pub joined_at: DateTime<Utc>,
}

pub struct OrgMembership {
    pub org_id: String,
    pub org_name: String,
    pub role: String,
    pub teams: Vec<String>,
}
```

#### `access.rs` — Access Control Enforcement

```rust
/// Access scope determines who can see a piece of knowledge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scope {
    /// Visible only to the creating agent.
    Agent,
    /// Visible to all members of the agent's team.
    Team(String),    // team_id
    /// Visible to all members of the organization.
    Org(String),     // org_id
    /// Visible to everyone (cross-organization).
    Global,
}

/// Access policy attached to Facts and Observations.
pub struct AccessPolicy {
    pub scope: Scope,
    pub created_by: String,   // participant_id of creator
    pub created_at: DateTime<Utc>,
}
```

Functions:

- `fn can_access(caller: &CallerContext, policy: &AccessPolicy) -> bool` — Check if a caller can access a piece of knowledge given its access policy.
- `async fn filter_by_access<T: HasAccessPolicy>(items: Vec<T>, caller: &CallerContext) -> Vec<T>` — Filter a list of items by caller's access permissions.
- `async fn scope_query(&self, query: &str, caller: &CallerContext) -> Result<String>` — Modify a graph query to include access control predicates.

```rust
/// Caller context for access control decisions.
pub struct CallerContext {
    pub participant_id: String,
    pub org_ids: Vec<String>,
    pub team_ids: Vec<String>,
}
```

Access control rules:

| Scope | Who Can Access | Query Predicate |
|---|---|---|
| Agent | Only the creating agent | `fact.created_by == caller.participant_id` |
| Team(id) | Any member of the specified team | `caller.team_ids CONTAINS team_id` |
| Org(id) | Any member of the specified organization | `caller.org_ids CONTAINS org_id` |
| Global | Anyone | No predicate (always accessible) |

Default scope: `Agent` — facts are private to the creating agent unless explicitly shared.

#### Query-Time Enforcement

```
Original query:
  MATCH (f:Fact)-[:ABOUT]->(e:Entity {name: "Alice"})
  RETURN f

Scoped query (for agent "agent-1" in org "org-1", team "team-alpha"):
  MATCH (f:Fact)-[:ABOUT]->(e:Entity {name: "Alice"})
  WHERE f.visibility = "global"
     OR f.created_by = "agent-1"
     OR (f.visibility = "org" AND f.org_id = "org-1")
     OR (f.visibility = "team" AND f.team_id = "team-alpha")
  RETURN f
```

#### `sharing.rs` — Cross-Agent Knowledge Sharing

```rust
/// Share a fact from one agent to a wider scope.
/// Creates a SHARED_FROM edge from the shared copy to the original.
pub async fn share_fact(
    kb: &KnowledgeBase,
    fact_id: &str,
    target_scope: Scope,
    shared_by: &str,
) -> Result<String>;  // Returns new fact_id of the shared copy

/// Retrieve facts shared with the caller's scope.
pub async fn get_shared_facts(
    kb: &KnowledgeBase,
    caller: &CallerContext,
    limit: usize,
) -> Result<Vec<Fact>>;

/// Promote a fact to global visibility.
/// Only org admins or the fact creator can do this.
pub async fn promote_to_global(
    kb: &KnowledgeBase,
    fact_id: &str,
    caller: &CallerContext,
) -> Result<()>;
```

Sharing flow:

```
1. Agent "agent-1" creates Fact F1 (scope: Agent, visibility: "agent")
2. share_fact(F1, Scope::Org("org-1"), "agent-1"):
   a. Create copy F1' with visibility: "org", org_id: "org-1"
   b. Create SHARED_FROM edge: F1' → F1 (with shared_by: "agent-1", shared_at: now)
   c. F1' inherits all properties from F1 (subject, predicate, object, confidence, valid_at)
3. Agent "agent-2" in org "org-1" queries:
   → Can see F1' (org-scoped), cannot see F1 (agent-scoped to agent-1)
```

---

### 17.5 — Python Binding (uniko-py)

**Objective:** Create a Python binding via PyO3 that exposes all Uniko methods as Python functions. The binding must be pip-installable, Python-idiomatic (context managers, iterators, dict returns), and support both sync and async usage.

#### Files

| File | Type | Purpose |
|---|---|---|
| `bindings/uniko-py/Cargo.toml` | Config | PyO3 cdylib crate |
| `bindings/uniko-py/src/lib.rs` | Rust | PyO3 module root, all Python-exposed types |
| `bindings/uniko-py/src/types.rs` | Rust | Python type conversions (Rust structs ↔ Python dicts) |
| `bindings/uniko-py/src/errors.rs` | Rust | Python exception mapping |
| `bindings/uniko-py/python/uniko/__init__.py` | Python | Package init, re-exports |
| `bindings/uniko-py/python/uniko/types.py` | Python | Type stubs for IDE support |
| `bindings/uniko-py/pyproject.toml` | Config | Maturin build configuration |
| `bindings/uniko-py/tests/test_uniko.py` | Python | Integration tests |

#### `Cargo.toml`

```toml
[package]
name = "uniko-py"
version = "0.1.0"
edition = "2024"

[lib]
name = "uniko"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { workspace = true, features = ["extension-module"] }
uniko-api = { path = "../../crates/uniko-api" }
tokio = { workspace = true }
serde_json = { workspace = true }
```

#### `pyproject.toml`

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "uniko"
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]

[tool.maturin]
features = ["pyo3/extension-module"]
```

#### `src/lib.rs` — PyO3 Module

```rust
use pyo3::prelude::*;

/// uniko: Cognitive memory system for AI agents.
///
/// Usage:
///     import uniko
///     u = uniko.Uniko("/path/to/db")
///     msg_id = u.ingest_message("Hello world", sender_id="user-1", session_id="s1")
///     ctx = u.recall("What was said?", budget=8192)
#[pymodule]
fn uniko(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUniko>()?;
    m.add_class::<PyContextBundle>()?;
    m.add_class::<PyRecallConfig>()?;
    Ok(())
}
```

#### `PyUniko` — Main Python Class

```rust
/// The main Uniko instance. Thread-safe, can be shared across threads.
#[pyclass(name = "Uniko")]
pub struct PyUniko {
    inner: Arc<Uniko>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyUniko {
    /// Create a new Uniko instance.
    ///
    /// Args:
    ///     db_path: Path to the database directory.
    ///     config: Optional configuration dict.
    ///
    /// Example:
    ///     u = uniko.Uniko("/tmp/my-memory")
    ///     u = uniko.Uniko("/tmp/my-memory", config={"consolidation_threshold": 10})
    #[new]
    #[pyo3(signature = (db_path, config=None))]
    fn new(db_path: &str, config: Option<PyObject>) -> PyResult<Self>;

    /// Ingest a message into memory.
    ///
    /// Args:
    ///     content: Message text content.
    ///     sender_id: ID of the message sender.
    ///     session_id: ID of the conversation session.
    ///     content_type: Type of content ("text", "code", etc.). Default: "text".
    ///     timestamp: ISO 8601 timestamp. Default: now.
    ///     addressed_to: ID of the message recipient.
    ///     goal_id: Associated goal ID.
    ///     task_id: Associated task ID.
    ///     metadata: Additional metadata dict.
    ///
    /// Returns:
    ///     str: The message ID.
    #[pyo3(signature = (content, *, sender_id, session_id, content_type="text", timestamp=None, addressed_to=None, goal_id=None, task_id=None, metadata=None))]
    fn ingest_message(
        &self,
        content: &str,
        sender_id: &str,
        session_id: &str,
        content_type: &str,
        timestamp: Option<&str>,
        addressed_to: Option<&str>,
        goal_id: Option<&str>,
        task_id: Option<&str>,
        metadata: Option<PyObject>,
    ) -> PyResult<String>;

    /// Recall relevant context for a query.
    ///
    /// Args:
    ///     query: Natural language query.
    ///     budget: Maximum token budget for context. Default: 8192.
    ///     scope: Search scope. Default: "all".
    ///
    /// Returns:
    ///     ContextBundle: The assembled context bundle (accessible as dict).
    #[pyo3(signature = (query, *, budget=8192, scope="all"))]
    fn recall(&self, query: &str, budget: usize, scope: &str) -> PyResult<PyContextBundle>;

    /// Record an episode (what happened, what changed).
    ///
    /// Args:
    ///     action_type: Type of episode ("investigate", "implement", etc.).
    ///     outcome: Result ("success", "failure", "partial", "inconclusive").
    ///     state: World context at time of episode (dict).
    ///     delta: What changed as a result (dict).
    ///     importance: Importance score (0.0-1.0).
    ///     participant_id: Who performed the action.
    ///     session_id: Session context.
    ///     task_id: Associated task.
    ///
    /// Returns:
    ///     str: The episode ID.
    #[pyo3(signature = (action_type, *, outcome=None, state=None, delta=None, importance=None, participant_id=None, session_id=None, task_id=None))]
    fn record_episode(
        &self,
        action_type: &str,
        outcome: Option<&str>,
        state: Option<PyObject>,
        delta: Option<PyObject>,
        importance: Option<f64>,
        participant_id: Option<&str>,
        session_id: Option<&str>,
        task_id: Option<&str>,
    ) -> PyResult<String>;

    /// Ingest an artifact (file, document, URL).
    #[pyo3(signature = (content, *, kind="file", path=None, language=None, mime_type=None, metadata=None))]
    fn ingest_artifact(
        &self,
        content: &str,
        kind: &str,
        path: Option<&str>,
        language: Option<&str>,
        mime_type: Option<&str>,
        metadata: Option<PyObject>,
    ) -> PyResult<String>;

    /// Record a tool call action.
    #[pyo3(signature = (action_type, *, input=None, output=None, status=None, participant_id=None, session_id=None))]
    fn record_action(
        &self,
        action_type: &str,
        input: Option<PyObject>,
        output: Option<PyObject>,
        status: Option<&str>,
        participant_id: Option<&str>,
        session_id: Option<&str>,
    ) -> PyResult<String>;

    /// Create or update a goal.
    #[pyo3(signature = (title, *, description=None, status="active", owner_id=None, deadline=None, metrics=None, guardrails=None))]
    fn create_goal(
        &self,
        title: &str,
        description: Option<&str>,
        status: &str,
        owner_id: Option<&str>,
        deadline: Option<&str>,
        metrics: Option<PyObject>,
        guardrails: Option<PyObject>,
    ) -> PyResult<String>;

    /// Create or update a task under a goal.
    #[pyo3(signature = (title, *, goal_id, description=None, status="pending", priority=None, assignee_id=None))]
    fn create_task(
        &self,
        title: &str,
        goal_id: &str,
        description: Option<&str>,
        status: &str,
        priority: Option<f64>,
        assignee_id: Option<&str>,
    ) -> PyResult<String>;

    /// Start a new session.
    #[pyo3(signature = (*, topic=None, task_id=None, goal_id=None, participant_ids=None))]
    fn start_session(
        &self,
        topic: Option<&str>,
        task_id: Option<&str>,
        goal_id: Option<&str>,
        participant_ids: Option<Vec<String>>,
    ) -> PyResult<String>;

    /// End a session.
    fn end_session(&self, session_id: &str) -> PyResult<()>;

    /// Share a fact to a wider scope.
    #[pyo3(signature = (fact_id, *, scope="global"))]
    fn share_fact(&self, fact_id: &str, scope: &str) -> PyResult<String>;

    /// Get shared facts visible to the current scope.
    #[pyo3(signature = (*, limit=50))]
    fn get_shared_facts(&self, limit: usize) -> PyResult<Vec<PyObject>>;

    /// Force a consolidation cycle.
    fn consolidate(&self, agent_id: &str) -> PyResult<()>;

    /// Get system health status.
    fn health(&self) -> PyResult<PyObject>;

    /// Shutdown the system gracefully.
    fn shutdown(&self) -> PyResult<()>;

    /// Context manager support: __enter__ returns self.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    /// Context manager support: __exit__ calls shutdown.
    fn __exit__(
        &self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> PyResult<bool>;
}
```

#### `PyContextBundle` — Context Bundle Wrapper

```rust
#[pyclass(name = "ContextBundle")]
pub struct PyContextBundle {
    inner: ContextBundle,
}

#[pymethods]
impl PyContextBundle {
    /// Total tokens in the context bundle.
    #[getter]
    fn tokens(&self) -> usize;

    /// Facts included in the context.
    #[getter]
    fn facts(&self) -> Vec<PyObject>;

    /// Observations included in the context.
    #[getter]
    fn observations(&self) -> Vec<PyObject>;

    /// Messages included in the context.
    #[getter]
    fn messages(&self) -> Vec<PyObject>;

    /// Chunks included in the context.
    #[getter]
    fn chunks(&self) -> Vec<PyObject>;

    /// Episodes included in the context.
    #[getter]
    fn episodes(&self) -> Vec<PyObject>;

    /// Convert to a Python dict.
    fn to_dict(&self) -> PyResult<PyObject>;

    /// Convert to a formatted string for LLM consumption.
    fn to_prompt(&self) -> String;

    /// String representation.
    fn __repr__(&self) -> String;

    /// Iteration support: iterate over all items in the bundle.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyContextBundleIter>;
}
```

#### `errors.rs` — Exception Mapping

```rust
/// Map UnikoError variants to Python exceptions.
///
/// UnikoError::Storage     → uniko.StorageError (subclass of RuntimeError)
/// UnikoError::Config      → uniko.ConfigError (subclass of ValueError)
/// UnikoError::Timeout     → uniko.TimeoutError (subclass of TimeoutError)
/// UnikoError::*           → uniko.UnikoError (subclass of Exception)
fn map_error(err: UnikoError) -> PyErr;
```

#### `tests/test_uniko.py` — Python Integration Tests

```python
import pytest
import uniko
import tempfile
import os

class TestBasicOperations:
    def setup_method(self):
        self.db_dir = tempfile.mkdtemp()
        self.u = uniko.Uniko(self.db_dir)

    def teardown_method(self):
        self.u.shutdown()

    def test_ingest_message(self):
        msg_id = self.u.ingest_message(
            "Hello, I'm working on the uniko project",
            sender_id="user-1",
            session_id="session-1",
        )
        assert isinstance(msg_id, str)
        assert len(msg_id) > 0

    def test_recall(self):
        self.u.ingest_message(
            "Alice prefers dark chocolate over milk chocolate",
            sender_id="user-1",
            session_id="session-1",
        )
        ctx = self.u.recall("What does Alice prefer?")
        assert ctx.tokens > 0
        assert isinstance(ctx.to_dict(), dict)

    def test_record_episode(self):
        ep_id = self.u.record_episode(
            "investigate",
            outcome="success",
            state={"query": "memory leak"},
            delta={"found": "buffer overflow in parser"},
            importance=0.8,
            participant_id="agent-1",
        )
        assert isinstance(ep_id, str)

    def test_context_manager(self):
        with uniko.Uniko(tempfile.mkdtemp()) as u:
            msg_id = u.ingest_message(
                "Test message",
                sender_id="user-1",
                session_id="s1",
            )
            assert isinstance(msg_id, str)
        # u.shutdown() called automatically

    def test_ingest_artifact(self):
        artifact_id = self.u.ingest_artifact(
            "def hello():\n    print('hello')",
            kind="file",
            path="/tmp/hello.py",
            language="python",
        )
        assert isinstance(artifact_id, str)

    def test_goal_task_session(self):
        goal_id = self.u.create_goal("Reduce latency by 50%")
        task_id = self.u.create_task("Profile hot paths", goal_id=goal_id)
        session_id = self.u.start_session(topic="Profiling", task_id=task_id)
        self.u.end_session(session_id)
        assert all(isinstance(x, str) for x in [goal_id, task_id, session_id])

    def test_recall_returns_context_bundle(self):
        self.u.ingest_message(
            "The meeting is scheduled for March 15 at 3pm",
            sender_id="user-1",
            session_id="s1",
        )
        ctx = self.u.recall("When is the meeting?")
        d = ctx.to_dict()
        assert "facts" in d or "observations" in d or "messages" in d

    def test_consolidation(self):
        for i in range(25):
            self.u.ingest_message(
                f"Alice mentioned preference #{i} for item-{i}",
                sender_id="user-1",
                session_id="s1",
            )
        self.u.consolidate("agent-1")
        # After consolidation, facts should exist
        ctx = self.u.recall("What are Alice's preferences?")
        assert ctx.tokens > 0

    def test_health(self):
        health = self.u.health()
        assert isinstance(health, dict)

    def test_error_handling(self):
        with pytest.raises(Exception):
            uniko.Uniko("/nonexistent/path/that/should/fail")
```

#### Async Support

For Python async usage, provide an async wrapper (optional, can use `pyo3-asyncio` or a sync-over-async pattern):

```python
# Future enhancement: native async support
# For now, all operations are synchronous (blocking)
# The Rust side runs tokio internally, Python sees sync calls
```

The initial binding uses synchronous wrappers around the async Rust code. The `Runtime` is created once in `PyUniko::new()` and used for all `block_on()` calls. This is safe because PyO3 releases the GIL during `block_on()`.

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_shadow_fs_sync_empty_dir` | `shadow.rs` | Sync on empty directory produces 0 files synced |
| `test_shadow_fs_sync_files` | `shadow.rs` | Sync on directory with files creates Artifact nodes |
| `test_shadow_fs_ignore_gitignore` | `shadow.rs` | Files matching .gitignore patterns are skipped |
| `test_shadow_fs_ignore_custom` | `shadow.rs` | Files matching custom ignore patterns are skipped |
| `test_shadow_fs_file_create_event` | `shadow.rs` | File creation event triggers Artifact creation |
| `test_shadow_fs_file_modify_event` | `shadow.rs` | File modification event updates existing Artifact |
| `test_shadow_fs_file_delete_event` | `shadow.rs` | File deletion event removes Artifact and Chunks |
| `test_shadow_fs_debounce` | `shadow.rs` | Rapid events for same path coalesced into single processing |
| `test_shadow_fs_max_file_size` | `shadow.rs` | Files exceeding max_file_size are skipped |
| `test_detect_file_type_rust` | `detect.rs` | `.rs` file → language="rust", kind="file" |
| `test_detect_file_type_markdown` | `detect.rs` | `.md` file → kind="document" |
| `test_detect_file_type_config` | `detect.rs` | `.toml` file → kind="config" |
| `test_detect_file_type_image` | `detect.rs` | `.png` file → kind="image", is_text=false |
| `test_ignore_default_patterns` | `ignore.rs` | `.git/`, `target/`, `node_modules/` always ignored |
| `test_ignore_gitignore_load` | `ignore.rs` | Patterns from `.gitignore` file loaded correctly |
| `test_git_map_single_commit` | `git.rs` | One commit → one Episode node with correct properties |
| `test_git_map_commit_edges` | `git.rs` | RECORDED_BY, MENTIONS, FOLLOWED_BY edges created |
| `test_git_map_idempotent` | `git.rs` | Mapping same commit twice creates one Episode (dedup by SHA) |
| `test_git_enhanced_blame_line` | `git.rs` | Blame line enriched with episode and fact context |
| `test_git_enhanced_log_entry` | `git.rs` | Log entry enriched with related episodes and task context |
| `test_org_create` | `organization/mod.rs` | Organization node created |
| `test_team_create` | `organization/mod.rs` | Team node created with TEAM_IN_ORG edge |
| `test_add_member` | `organization/mod.rs` | MEMBER_OF edge created with role |
| `test_access_agent_scope` | `organization/access.rs` | Agent-scoped facts visible only to creator |
| `test_access_team_scope` | `organization/access.rs` | Team-scoped facts visible to team members |
| `test_access_org_scope` | `organization/access.rs` | Org-scoped facts visible to org members |
| `test_access_global_scope` | `organization/access.rs` | Global facts visible to everyone |
| `test_cross_org_blocked` | `organization/access.rs` | Agent in org-A cannot see org-B scoped facts |
| `test_share_fact_org` | `organization/sharing.rs` | Fact shared to org creates copy with SHARED_FROM edge |
| `test_share_fact_global` | `organization/sharing.rs` | Promoted fact visible to all orgs |
| `test_python_ingest_message` | `test_uniko.py` | Python `ingest_message` returns valid message ID |
| `test_python_recall` | `test_uniko.py` | Python `recall` returns ContextBundle with data |
| `test_python_context_manager` | `test_uniko.py` | `with` statement works, shutdown called |
| `test_python_context_bundle_dict` | `test_uniko.py` | `to_dict()` returns valid Python dict |
| `test_python_error_mapping` | `test_uniko.py` | Rust errors map to Python exceptions |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_fs_sync_end_to_end` | integration | Sync directory → files become Artifacts → entities extracted → facts derived |
| `test_fs_watch_live_changes` | integration | Watch directory, create file, verify Artifact appears |
| `test_git_full_repo_mapping` | integration | Map real git repo history → Episodes with correct relationships |
| `test_shell_grep_returns_results` | integration | `ug "function"` returns matching code chunks |
| `test_shell_find_semantic` | integration | `uf "error handling"` returns relevant files by embedding similarity |
| `test_shell_cat_with_context` | integration | `uc --all file.rs` shows file + entities + facts |
| `test_org_isolation_full` | integration | Two agents in different orgs: each sees only their org's facts |
| `test_python_full_workflow` | integration | Python: ingest → consolidate → recall → verify facts in context |
| `test_python_pip_install` | integration | `pip install .` in bindings/uniko-py/ succeeds |

### Performance Tests

| Test | Target | What It Validates |
|---|---|---|
| `bench_fs_sync_100_files` | < 10s | Directory sync at moderate scale |
| `bench_fs_event_latency` | < 100ms detection | File change detection speed |
| `bench_git_map_1000_commits` | < 30s | Commit mapping at moderate scale |
| `bench_python_ingest_throughput` | > 50 msgs/sec | Python binding doesn't bottleneck |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| ShadowFs module doc | `uniko-fs/src/shadow.rs` | How shadow FS works, configuration, ignore patterns |
| GitIntegration module doc | `uniko-fs/src/git.rs` | Commit → Episode mapping, enhanced blame/log |
| Shell command docs | `uniko-shell/src/commands/*.rs` | Usage, flags, examples for each command |
| Organization module doc | `uniko-memory/src/organization/mod.rs` | Multi-tenant model, CRUD operations |
| Access control doc | `uniko-memory/src/organization/access.rs` | Scope types, enforcement rules, query scoping |
| Python README | `bindings/uniko-py/README.md` | Installation, quick start, API reference |
| Python type stubs | `bindings/uniko-py/python/uniko/types.py` | Type hints for IDE completion |

---

## Review Checklist

- [ ] `uniko-fs` depends on `uniko-api` only (no direct L1/L2/L3 deps)
- [ ] `uniko-shell` depends on `uniko-api` only
- [ ] `uniko-py` depends on `uniko-api` only
- [ ] ShadowFs: sync walks directory and creates Artifact nodes
- [ ] ShadowFs: watch detects create/modify/delete/rename events
- [ ] ShadowFs: .gitignore and custom patterns respected
- [ ] ShadowFs: debouncing coalesces rapid events
- [ ] ShadowFs: max_file_size enforced
- [ ] ShadowFs: binary files handled (kind detection, no text extraction)
- [ ] Git: commits mapped to Episodes with deterministic IDs (commit SHA)
- [ ] Git: RECORDED_BY, MENTIONS, FOLLOWED_BY edges wired correctly
- [ ] Git: mapping is idempotent (re-mapping same commits = no duplicates)
- [ ] Git: enhanced blame enriches lines with memory context
- [ ] Git: enhanced log enriches entries with episode and fact context
- [ ] Shell: all 7 commands implemented (ug, uf, uc, ud, ub, ul, ulog)
- [ ] Shell: each command parses args, queries uniko, formats output
- [ ] Shell: output formats supported (plain, JSON, colored)
- [ ] Shell: `ug` uses recall cascade for semantic search
- [ ] Shell: `uf` uses vector search on Artifact embeddings
- [ ] Shell: `uc --all` shows entities, facts, observations from file
- [ ] Org: Organization and Team CRUD operations work
- [ ] Org: MEMBER_OF, PART_OF_TEAM, TEAM_IN_ORG edges created correctly
- [ ] Access: Agent scope restricts facts to creator only
- [ ] Access: Team scope restricts facts to team members
- [ ] Access: Org scope restricts facts to org members
- [ ] Access: Global scope makes facts visible to all
- [ ] Access: cross-org queries blocked (agent in org-A cannot see org-B facts)
- [ ] Access: query-time filtering applied to all recall operations
- [ ] Sharing: share_fact creates copy with SHARED_FROM edge
- [ ] Sharing: promote_to_global requires admin or creator permission
- [ ] Python: PyO3 module compiles as cdylib
- [ ] Python: `pip install .` succeeds in bindings/uniko-py/
- [ ] Python: `import uniko` works
- [ ] Python: `Uniko(path)` constructor creates working instance
- [ ] Python: `ingest_message` returns message ID
- [ ] Python: `recall` returns ContextBundle
- [ ] Python: `record_episode` returns episode ID
- [ ] Python: `ingest_artifact` returns artifact ID
- [ ] Python: context manager (`with`) works
- [ ] Python: `ContextBundle.to_dict()` returns valid dict
- [ ] Python: error mapping (Rust errors → Python exceptions)
- [ ] Python: keyword-only arguments for optional params
- [ ] Python: type stubs provide IDE completion
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All Python tests pass

---

## Definition of Done

1. **ShadowFs functional:** `sync()` walks a directory and creates Artifact nodes for all non-ignored files. `watch()` detects file create/modify/delete events in real time and updates the graph accordingly. Debouncing coalesces rapid events. .gitignore and custom patterns respected.
2. **Git integration functional:** `map_commits()` converts git history into Episode nodes with correct edges (RECORDED_BY → author, MENTIONS → changed files, FOLLOWED_BY → previous commit). Mapping is idempotent. `enhanced_blame()` and `enhanced_log()` enrich standard git operations with memory context.
3. **Semantic shell functional:** All 7 commands (ug, uf, uc, ud, ub, ul, ulog) parse arguments, query the Uniko API, and produce formatted output. `ug` performs semantic search via recall cascade. `uf` performs vector search on Artifact embeddings. `uc --all` shows file content with entities, facts, and observations.
4. **Organization support functional:** Organization and Team CRUD operations create correct graph nodes and edges. Multi-tenant isolation enforced: queries scoped to caller's org by default. Access policies (Agent, Team, Org, Global) correctly filter query results.
5. **Access control enforced:** Agent-scoped facts visible only to creator. Team-scoped facts visible to team members. Org-scoped facts visible to org members. Global facts visible to all. Cross-org access correctly blocked. All recall operations apply access filtering.
6. **Knowledge sharing functional:** `share_fact()` creates org/global-scoped copies with SHARED_FROM provenance edges. `promote_to_global()` restricted to admins/creators. Shared facts visible to target scope.
7. **Python binding pip-installable:** `pip install .` in `bindings/uniko-py/` succeeds. `import uniko` works. All 12+ methods exposed with Python-idiomatic API (keyword arguments, context managers, dict returns, iterator support).
8. **Python tests pass:** All tests in `test_uniko.py` pass: ingest, recall, episode recording, artifact ingestion, goal/task/session lifecycle, context manager, error handling, consolidation.
9. **FS sync end-to-end validated:** Directory sync creates Artifacts, triggers pipeline (entities extracted, facts derived), recall finds content from synced files.
10. **All tests pass:** Rust unit tests, Rust integration tests, Python tests all green. `cargo nextest run -n auto` passes for all L4 crates. `pytest` passes for Python tests.
