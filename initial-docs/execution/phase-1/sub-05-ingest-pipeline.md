# Sub-Phase 5: Ingest Pipeline (P1) & Chunking

## Context

This sub-phase implements Pipeline 1 -- the synchronous ingest path for Messages and Artifacts. P1 is the entry point for all data entering uniko. Every node in the graph ultimately traces back to a message or an artifact processed by this pipeline. P1 creates the foundational graph structure (nodes, edges, session management, message ordering) that all downstream pipelines (P2-P8) operate on.

All ingest code lives in `uniko-extract` (content processing layer), implementing the `Step` trait from `uniko-pipes`.

The ingest pipeline handles:
- **Message ingest**: Create Message node with SENT_BY, ADDRESSED_TO, IN_SESSION, NEXT edges. Auto-create sessions. Chunk long messages (> 1024 tokens).
- **Artifact ingest**: Compute hash, detect MIME type, deduplicate by hash, create Artifact node, select chunker by content type, create Chunk nodes with HAS_CHUNK edges.
- **Session lifecycle**: Auto-create sessions for participant+goal combinations. End sessions on inactivity timeout, explicit signal, or goal terminal status.
- **Action output overflow**: Large Action outputs (> 256 tokens) overflow to Artifact nodes via PRODUCED edges.
- **Chunking strategies**: Text (recursive splitting), Code (tree-sitter AST), HTML (DOM sections), PDF (page extraction), CSV/JSON (schema-aware row grouping).

P1 runs synchronously within the IngestWorker for each item. After P1 completes, P2 (NER) runs immediately on the same item. P3 (observations) and P7a (auto-embed) are spawned asynchronously.

Latency targets: < 10ms for messages (NF1), < 100ms for text artifacts (NF5).

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Sub-phase 3: KnowledgeBase (uniko-store) | Complete | `Arc<KnowledgeBase>` with node/edge CRUD, index operations, embedding, search |
| Sub-phase 4: Pipeline Infrastructure | Complete | `Step` trait (from `uniko-pipes`), `PipelineContext`, `StepErrorPolicy`, channels, circuit breaker. `PipelineSystem` and `IngestWorker` from `uniko-memory`. |
| Sub-phase 2: Schema Types (uniko-store) | Complete | All node types (`Message`, `Artifact`, `Chunk`, `Session`, `Participant`, `Action`), edge types (`SENT_BY`, `IN_SESSION`, `NEXT`, `HAS_CHUNK`, `PRODUCED`, etc.) |
| `tree-sitter` crate + language grammars | Available | AST parsing for code chunking (Python, Rust, JS/TS, Go, Java, C/C++) |
| `tiktoken-rs` or equivalent | Available | Token counting for chunk size targeting |

## Sub-phases

---

### 5.1 -- Message Ingest Path

**Objective:** Implement the complete message ingest flow: create Message node, establish all edges, handle message ordering, check for chunking threshold, and queue for downstream pipelines.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/ingest/mod.rs` | New module root | Re-exports, `IngestStep` implementing `Step` trait |
| `crates/uniko-extract/src/ingest/message.rs` | New | `ingest_message` function, `IngestMessage` struct |

#### Structs

```rust
pub struct IngestMessage {
    pub content: String,
    pub sender_id: String,
    pub session_id: Option<String>,  // auto-created if None
    pub timestamp: DateTime<Utc>,
    pub content_type: String,        // "text", "code", "image", "tool_result", "error", "system"
    pub addressed_to: Option<Vec<String>>,  // participant IDs; inferred from session if None
    pub goal_id: Option<String>,     // used for session auto-creation
    pub task_id: Option<String>,     // used for session auto-creation
    pub message_id: Option<String>,  // caller-provided or UUID v7
}
```

#### Function: `ingest_message`

```rust
pub async fn ingest_message(
    kb: &KnowledgeBase,
    msg: IngestMessage,
) -> Result<ItemResult>
```

**Steps (executed sequentially per message):**

1. **Create Message node** -- Generate `message_id` (UUID v7 if not provided). Create node with content, content_type, timestamp. Embedding is auto-embedded by uni-db from `content` field.

2. **Create SENT_BY edge** -- `Message -[SENT_BY {role}]-> Participant`. Role derived from sender's `kind` field ("human" -> "user", "agent" -> "assistant", "service" -> "system"). Validate sender_id exists in graph; error if not.

3. **Create ADDRESSED_TO edges** -- If `addressed_to` is provided, create edges to each listed participant. If None, infer from session: all PARTICIPATED_IN participants except the sender.

4. **Create IN_SESSION edge** -- `Message -[IN_SESSION]-> Session`. If `session_id` is None, call `get_or_create_session()` (see 5.6). Create edge.

5. **Find previous Message and create NEXT edge** -- Query: `MATCH (m:Message)-[:IN_SESSION]->(s:Session {session_id: $sid}) RETURN m ORDER BY m.timestamp DESC LIMIT 1`. If found, create `prev_message -[NEXT {gap_ms}]-> new_message` where `gap_ms = new_timestamp - prev_timestamp` in milliseconds.

6. **Check length for chunking** -- Count tokens in `content`. If > 1024 tokens (configurable via `message_chunk_threshold`):
   - Chunk using text document strategy (see 5.2).
   - Create Chunk nodes + `Message -[HAS_CHUNK {index}]-> Chunk` edges.

7. **Queue for downstream pipelines** -- Return signals in `ItemResult` metadata indicating this item should be processed by P2 (NER), P3 (observations), and P7a (auto-embed). The IngestWorker step chain handles the routing.

**Latency target:** < 10ms for non-chunked messages.

---

### 5.2 -- Text & Markdown Chunking

**Objective:** Implement recursive splitting for text and markdown content targeting 400-512 tokens per chunk with sentence-boundary alignment and overlap.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/ingest/chunking/mod.rs` | New module root | `Chunker` trait, `ChunkConfig`, `ChunkData`, `select_chunker()` |
| `crates/uniko-extract/src/ingest/chunking/text.rs` | New | `TextChunker` implementing recursive splitting |

#### Trait and Types

```rust
pub trait Chunker: Send + Sync {
    fn chunk(&self, content: &str, config: &ChunkConfig) -> Result<Vec<ChunkData>>;
}

pub struct ChunkConfig {
    pub max_chunk_tokens: usize,                   // default: 512
    pub min_chunk_tokens: usize,                   // default: 64
    pub overlap_sentences: usize,                  // default: 1
    pub semantic_similarity_threshold: f64,         // default: 0.7
    pub code_split_level: String,                  // default: "function"
    pub message_chunk_threshold: usize,            // default: 1024
    pub action_output_artifact_threshold: usize,   // default: 256
}

pub struct ChunkData {
    pub text: String,
    pub index: usize,
    pub start: usize,          // byte offset in parent
    pub end: usize,            // byte offset end
    pub token_count: usize,
    pub chunk_type: String,    // "paragraph", "sentence_group", "function", etc.
    pub language: Option<String>,
    pub symbol_name: Option<String>,
    pub speaker: Option<String>,
    pub heading: Option<String>,
    pub mime_type: Option<String>,
}
```

#### `TextChunker` Algorithm

```rust
pub fn chunk_text(content: &str, config: &ChunkConfig) -> Vec<ChunkData>
```

**Recursive splitting algorithm:**

1. **Paragraph split** -- Split content at double-newline boundaries (`\n\n`).
2. **Sentence split** -- For each paragraph larger than `max_chunk_tokens`, split at sentence boundaries (`. `, `? `, `! `, `.\n`). Use a simple regex-based sentence boundary detector.
3. **Accumulate** -- Group consecutive sentences/paragraphs into chunks targeting 400-512 tokens. Never break mid-sentence.
4. **Overlap** -- Include `overlap_sentences` (default: 1) sentences from the end of the previous chunk at the start of the next chunk. This provides 10-20% overlap for context continuity.
5. **Markdown awareness** -- Track the nearest preceding heading (`# `, `## `, etc.) and include it in `ChunkData.heading` for each chunk.

**Token counting:** Use `tiktoken-rs` for accurate token counts when available. Fallback: word-count approximation (1 token per 0.75 words).

**Edge cases:**
- Content shorter than `min_chunk_tokens` (64): produce a single chunk.
- Single sentence longer than `max_chunk_tokens`: keep as-is (do not break mid-sentence). Log warning.
- Empty content: return empty vec.

#### Chunk Metadata

- `chunk_type`: "paragraph" (split at paragraph boundaries) or "sentence_group" (split at sentence boundaries within a paragraph).
- `heading`: nearest preceding heading for markdown. None for plain text without headings.

---

### 5.3 -- Code Chunking (tree-sitter)

**Objective:** Implement AST-based code chunking using tree-sitter. Extract meaningful code units (functions, classes, structs, import blocks) as individual chunks.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/ingest/chunking/code.rs` | New | `CodeChunker` using tree-sitter parsing |

#### Function

```rust
pub fn chunk_code(
    content: &str,
    language: &str,
    config: &ChunkConfig,
) -> Vec<ChunkData>
```

#### Algorithm

1. **Parse** -- Initialize tree-sitter parser for the detected language. Supported: Python, Rust, JavaScript, TypeScript, Go, Java, C, C++.
2. **Extract top-level declarations** -- Walk the AST tree for top-level nodes:
   - Functions/methods -> one chunk each. `chunk_type: "function"`, `symbol_name: "process_order"`.
   - Classes/structs/enums -> one chunk each (including all methods). `chunk_type: "class"` or `"struct"`.
   - Module-level statements (not inside any declaration) -> group into one chunk. `chunk_type: "module"`.
   - Import/use blocks -> group into one chunk. `chunk_type: "imports"`.
3. **Handle large functions** -- If a function/class exceeds `max_chunk_tokens`:
   - Split at block boundaries (nested function definitions, match arms, if/else blocks).
   - Each sub-block becomes a separate chunk with `chunk_type: "block"`.
   - Preserve the function signature as context in each sub-chunk.
4. **Language detection** -- If `language` is not provided, detect from file extension (via MIME type) or fall back to text chunking.

#### Chunk Metadata

- `chunk_type`: "function" | "class" | "struct" | "module" | "imports" | "block"
- `language`: "rust" | "python" | "javascript" | "typescript" | "go" | "java" | "c" | "cpp"
- `symbol_name`: The name of the function/class/struct. None for "module", "imports", "block" types.

#### Tree-sitter Node Mapping

| Language | Function | Class/Struct | Module | Imports |
|---|---|---|---|---|
| Python | `function_definition` | `class_definition` | top-level expressions | `import_statement`, `import_from_statement` |
| Rust | `function_item` | `struct_item`, `enum_item`, `impl_item` | top-level `let`, `const`, `static` | `use_declaration` |
| JS/TS | `function_declaration`, `arrow_function` | `class_declaration` | top-level expressions | `import_statement` |
| Go | `function_declaration`, `method_declaration` | `type_declaration` (struct) | top-level `var`, `const` | `import_declaration` |
| Java | `method_declaration` | `class_declaration`, `interface_declaration`, `enum_declaration` | -- | `import_declaration` |
| C/C++ | `function_definition` | `struct_specifier`, `class_specifier` | top-level declarations | `preproc_include` |

---

### 5.4 -- Additional Chunking Strategies

**Objective:** Implement chunking for HTML, PDF, and structured data (CSV/JSON). Each strategy extracts meaningful units appropriate to the content type.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/ingest/chunking/html.rs` | New | `HtmlChunker` -- DOM section extraction |
| `crates/uniko-extract/src/ingest/chunking/pdf.rs` | New | `PdfChunker` -- page extraction + text chunking |
| `crates/uniko-extract/src/ingest/chunking/structured.rs` | New | `StructuredChunker` -- CSV/JSON schema-aware row grouping |

#### HTML Chunker

```rust
pub fn chunk_html(content: &str, config: &ChunkConfig) -> Vec<ChunkData>
```

**Algorithm:**
1. Parse DOM (use `scraper` or `lol_html` crate).
2. Strip non-content elements: `<nav>`, `<footer>`, `<script>`, `<style>`, `<header>` (site header, not content header).
3. Split at content boundaries: `<article>`, `<section>`, `<h1>`-`<h6>`.
4. Tables -> one chunk per table. Header row stored in metadata.
5. Lists (`<ul>`, `<ol>`) -> one chunk per list, with context from the preceding paragraph.
6. Remaining prose -> recursive text splitting (reuse `TextChunker`).

**Chunk metadata:**
- `chunk_type`: "heading_section" | "table" | "list"
- `heading`: section heading text from the nearest `<h1>`-`<h6>`

#### PDF Chunker

```rust
pub fn chunk_pdf(content: &[u8], config: &ChunkConfig) -> Vec<ChunkData>
```

**Algorithm:**
1. Extract text per page (use `pdf-extract` or `lopdf` crate). Preserve reading order.
2. Detect multi-column layouts -> merge columns into single text flow.
3. Apply `TextChunker` to the merged text.
4. Tables -> extract separately (heuristic: rows of aligned text), one chunk per table.
5. Images -> extract as separate Artifact nodes (`kind: "image"`), linked via HAS_CHUNK or as child Artifacts.

**Chunk metadata:**
- `chunk_type`: "paragraph" | "table"
- `heading`: detected section heading (bold text at start of section, or text with larger font size)

#### Structured Data Chunker (CSV/JSON)

```rust
pub fn chunk_structured(content: &str, mime_type: &str, config: &ChunkConfig) -> Vec<ChunkData>
```

**Algorithm (CSV):**
1. Parse header row -> store as metadata (not a separate chunk).
2. Calculate `avg_row_tokens` from first 10 rows.
3. Compute `rows_per_chunk = max_chunk_tokens / avg_row_tokens`.
4. Group N rows per chunk. Each chunk includes the header as context prefix for self-contained understanding.

**Algorithm (JSON):**
1. If array of objects: each object or group of small objects = one chunk.
2. If nested object: each top-level key = one chunk (recursively split large values).
3. Schema (keys) stored as metadata.

**Chunk metadata:**
- `chunk_type`: "row_group"
- `heading`: column headers (CSV) or top-level keys (JSON) as comma-separated string

#### Chunker Selection

```rust
pub fn select_chunker(mime_type: &str) -> Box<dyn Chunker>
```

| MIME Type | Chunker |
|---|---|
| `text/plain`, `text/markdown` | `TextChunker` |
| `text/x-python`, `text/x-rust`, `text/javascript`, `text/typescript`, `text/x-go`, `text/x-java`, `text/x-c`, `text/x-c++` | `CodeChunker` |
| `text/html`, `application/xml`, `application/xhtml+xml` | `HtmlChunker` |
| `application/pdf` | `PdfChunker` |
| `text/csv`, `application/json` | `StructuredChunker` |
| `image/*` | No chunking (atomic) |
| `audio/*`, `video/*` | Deferred to Phase 6 (research) |
| Unknown | `TextChunker` (fallback) |

---

### 5.5 -- Artifact Ingest Path

**Objective:** Implement the complete artifact ingest flow: hash computation, MIME detection, deduplication, node creation, chunking, and downstream pipeline queueing.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/ingest/artifact.rs` | New | `ingest_artifact` function, `IngestArtifact` struct |

#### Struct

```rust
pub struct IngestArtifact {
    pub content: String,            // text content (None for binary)
    pub bytes: Option<Vec<u8>>,     // raw bytes (for binary artifacts)
    pub path: Option<String>,       // filesystem path, URL, or identifier
    pub mime_type: Option<String>,  // auto-detected if None
    pub kind: String,               // "file", "document", "url", "snippet", "config", "image", etc.
    pub language: Option<String>,   // for code: "rust", "python", etc.
    pub metadata: Option<Value>,    // arbitrary JSON metadata
    pub artifact_id: Option<String>, // caller-provided or UUID v7
}
```

#### Function: `ingest_artifact`

```rust
pub async fn ingest_artifact(
    kb: &KnowledgeBase,
    artifact: IngestArtifact,
) -> Result<ItemResult>
```

**Steps:**

1. **Compute hash** -- SHA-256 of content (text) or bytes (binary). Store as hex string.

2. **Detect MIME type** -- If `mime_type` is None, detect from:
   - File extension (from `path`)
   - Content sniffing (magic bytes for binary, heuristic for text)
   - Fallback: `application/octet-stream` (binary) or `text/plain` (text)

3. **Check deduplication** -- Query: `MATCH (a:Artifact {hash: $hash}) RETURN a`. If found:
   - Skip chunking and embedding entirely.
   - Optionally create a reference edge from the new context to the existing Artifact.
   - Return early with `StepOutcome::Skipped { reason: "duplicate hash" }`.

4. **Create Artifact node** -- Generate `artifact_id` (UUID v7 if not provided). Create node with path, content, mime_type, hash, size, kind, language, created_at, updated_at.

5. **Select chunker and create Chunks** -- Call `select_chunker(mime_type)` to get the appropriate chunker. Run chunker on content. For each `ChunkData`:
   - Deterministic `chunk_id`: `"{artifact_id}:{index}"` (enables idempotent re-chunking).
   - Create Chunk node with text, index, start, end, token_count, chunk_type, language, symbol_name, speaker, heading, mime_type.
   - Create `Artifact -[HAS_CHUNK {index}]-> Chunk` edge.

6. **Queue for downstream pipelines:**
   - Each Chunk -> P2 (NER) for entity extraction.
   - Each Chunk -> P7a (auto-embed on Chunk.text).
   - Artifact -> P7c (pool chunk embeddings -> Artifact.text_embedding).
   - If image/audio/video: Artifact -> P7b (multimodal embeddings).

**Latency target:** < 100ms for text artifacts.

---

### 5.6 -- Session Lifecycle Management

**Objective:** Implement automatic session creation, session end detection (inactivity, explicit, goal terminal), and session end side effects (summarization, re-embedding).

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/ingest/session.rs` | New | `get_or_create_session`, session end triggers, session lifecycle |

#### Functions

**`get_or_create_session`:**
```rust
pub async fn get_or_create_session(
    kb: &KnowledgeBase,
    participant_id: &str,
    goal_or_task: &str,   // goal_id or task_id
    topic: Option<&str>,
) -> Result<SessionId>
```

**Logic:**
1. Query for an active session (no `ended_at`) where this participant has a PARTICIPATED_IN edge and the session is FOR_TASK or FOR_GOAL matching `goal_or_task`.
2. If found: return existing session_id.
3. If not found: create new Session node with:
   - `session_id`: UUID v7
   - `topic`: provided or derived from goal/task title
   - `started_at`: now
   - `ended_at`: None
   - Create `Participant -[PARTICIPATED_IN {role: "initiator"}]-> Session` edge.
   - Create `Session -[FOR_TASK]-> Task` or `Session -[FOR_GOAL]-> Goal` edge.
   - Return new session_id.

**Session end triggers:**

| Trigger | Detection | Implementation |
|---|---|---|
| Explicit `end_session` tool call | Agent calls `end_session(session_id)` | Set `ended_at = now()`, queue side effects |
| Inactivity timeout | No message in session for 30 min (configurable) | Background task checks `last_message_timestamp` per active session |
| Goal/Task terminal status | Goal status -> "achieved" / "failed"; Task status -> "completed" / "failed" | Event hook on goal/task status update |

**On session end:**
1. Set `Session.ended_at = now()`.
2. Queue session for P7d (generate session summary from all messages).
3. Re-embed Session (now has topic + summary -- the embedding changes).

**Inactivity checker:**
- Background task runs every 5 minutes.
- Queries active sessions: `MATCH (s:Session) WHERE s.ended_at IS NULL`.
- For each: find latest message timestamp. If `now() - latest > inactivity_timeout`, end the session.
- Default `inactivity_timeout`: 30 minutes (configurable via `PipelineConfig`).

---

### 5.7 -- Action Output Overflow

**Objective:** When an Action's output exceeds 256 tokens, create an Artifact containing the full output and store only a summary in the Action node.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/ingest/overflow.rs` | New | `maybe_overflow_output` function |

#### Function

```rust
pub async fn maybe_overflow_output(
    kb: &KnowledgeBase,
    action: &mut Action,
) -> Result<Option<ArtifactId>>
```

**Logic:**

1. Count tokens in `action.output` (the JSON-serialized output).
2. If token_count <= 256: return None (no overflow needed).
3. If token_count > 256:
   a. Create Artifact node:
      - `kind`: "snippet"
      - `content`: full output text
      - `mime_type`: "text/plain" (or inferred from content)
      - `hash`: SHA-256 of content
   b. Create `Action -[PRODUCED]-> Artifact` edge.
   c. Generate summary of the output:
      - For now: truncate to first 200 tokens + "... [truncated, see artifact]".
      - Future: LLM-generated summary when available.
   d. Replace `action.output` with `{"summary": "<truncated>", "artifact_id": "<id>", "truncated": true}`.
   e. The created Artifact follows the normal artifact ingest path (chunk, embed).
   f. Return `Some(artifact_id)`.

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| **Message Ingest** | | |
| `test_ingest_message_creates_node` | `ingest/message.rs` | Message node created with correct fields |
| `test_ingest_message_sent_by_edge` | `ingest/message.rs` | SENT_BY edge created with correct role |
| `test_ingest_message_addressed_to_explicit` | `ingest/message.rs` | ADDRESSED_TO edges created for provided list |
| `test_ingest_message_addressed_to_inferred` | `ingest/message.rs` | ADDRESSED_TO edges inferred from session participants |
| `test_ingest_message_in_session_edge` | `ingest/message.rs` | IN_SESSION edge created |
| `test_ingest_message_next_edge` | `ingest/message.rs` | NEXT edge created with correct gap_ms |
| `test_ingest_message_next_edge_first_message` | `ingest/message.rs` | No NEXT edge for first message in session |
| `test_ingest_message_ordering` | `ingest/message.rs` | 10 messages create correct NEXT chain |
| `test_ingest_message_auto_session` | `ingest/message.rs` | Session auto-created when session_id is None |
| `test_ingest_message_long_content_chunked` | `ingest/message.rs` | Message > 1024 tokens creates Chunk nodes + HAS_CHUNK edges |
| `test_ingest_message_short_content_no_chunks` | `ingest/message.rs` | Message < 1024 tokens has no Chunk nodes |
| `test_ingest_message_uuid_generation` | `ingest/message.rs` | UUID v7 generated when message_id not provided |
| `test_ingest_message_caller_id_preserved` | `ingest/message.rs` | Caller-provided message_id used as-is |
| `test_ingest_message_latency` | `ingest/message.rs` | Completes in < 10ms (release mode, warm store) |
| **Text Chunking** | | |
| `test_chunk_text_basic` | `chunking/text.rs` | Short text produces single chunk |
| `test_chunk_text_paragraph_split` | `chunking/text.rs` | Multi-paragraph text splits at paragraph boundaries |
| `test_chunk_text_sentence_split` | `chunking/text.rs` | Long paragraph splits at sentence boundaries |
| `test_chunk_text_target_size` | `chunking/text.rs` | Chunks are within 400-512 token range |
| `test_chunk_text_no_mid_sentence_break` | `chunking/text.rs` | No chunk ends mid-sentence |
| `test_chunk_text_overlap` | `chunking/text.rs` | Adjacent chunks share overlap_sentences sentences |
| `test_chunk_text_markdown_headings` | `chunking/text.rs` | Chunks include nearest preceding heading in metadata |
| `test_chunk_text_empty` | `chunking/text.rs` | Empty content returns empty vec |
| `test_chunk_text_single_long_sentence` | `chunking/text.rs` | Single sentence > max_tokens kept intact with warning |
| `test_chunk_text_min_size` | `chunking/text.rs` | Content < min_chunk_tokens produces one chunk |
| **Code Chunking** | | |
| `test_chunk_code_python_functions` | `chunking/code.rs` | Python functions extracted as individual chunks |
| `test_chunk_code_python_classes` | `chunking/code.rs` | Python class with methods = one chunk |
| `test_chunk_code_python_imports` | `chunking/code.rs` | Import block grouped into one chunk |
| `test_chunk_code_rust_functions` | `chunking/code.rs` | Rust fn items extracted correctly |
| `test_chunk_code_rust_structs` | `chunking/code.rs` | Rust struct + impl = correct chunks with symbol_name |
| `test_chunk_code_js_arrow_functions` | `chunking/code.rs` | JS arrow functions detected |
| `test_chunk_code_large_function_split` | `chunking/code.rs` | Function > max_tokens split at block boundaries |
| `test_chunk_code_symbol_names` | `chunking/code.rs` | symbol_name correctly extracted for all chunk types |
| `test_chunk_code_language_metadata` | `chunking/code.rs` | language field set correctly per language |
| `test_chunk_code_unsupported_language` | `chunking/code.rs` | Falls back to text chunking for unsupported language |
| **HTML Chunking** | | |
| `test_chunk_html_strips_nav_footer` | `chunking/html.rs` | nav, footer, script, style elements removed |
| `test_chunk_html_section_split` | `chunking/html.rs` | Splits at article/section/heading boundaries |
| `test_chunk_html_tables` | `chunking/html.rs` | Tables extracted as separate chunks |
| `test_chunk_html_lists` | `chunking/html.rs` | Lists chunked with preceding paragraph context |
| **PDF Chunking** | | |
| `test_chunk_pdf_page_extraction` | `chunking/pdf.rs` | Text extracted from PDF pages in reading order |
| `test_chunk_pdf_table_extraction` | `chunking/pdf.rs` | Tables extracted separately |
| **Structured Chunking** | | |
| `test_chunk_csv_row_grouping` | `chunking/structured.rs` | Rows grouped by token budget |
| `test_chunk_csv_header_in_each` | `chunking/structured.rs` | Each chunk includes header context |
| `test_chunk_json_array` | `chunking/structured.rs` | JSON array of objects chunked per object/group |
| `test_chunk_json_nested` | `chunking/structured.rs` | Nested JSON split at top-level keys |
| **Chunker Selection** | | |
| `test_select_chunker_text` | `chunking/mod.rs` | text/plain -> TextChunker |
| `test_select_chunker_code` | `chunking/mod.rs` | text/x-python -> CodeChunker |
| `test_select_chunker_html` | `chunking/mod.rs` | text/html -> HtmlChunker |
| `test_select_chunker_unknown` | `chunking/mod.rs` | unknown/type -> TextChunker (fallback) |
| **Artifact Ingest** | | |
| `test_ingest_artifact_creates_node` | `ingest/artifact.rs` | Artifact node created with correct fields |
| `test_ingest_artifact_hash_computed` | `ingest/artifact.rs` | SHA-256 hash computed and stored |
| `test_ingest_artifact_mime_detection` | `ingest/artifact.rs` | MIME type auto-detected from extension |
| `test_ingest_artifact_dedup_same_hash` | `ingest/artifact.rs` | Duplicate hash -> skip, return single artifact |
| `test_ingest_artifact_creates_chunks` | `ingest/artifact.rs` | Chunks created with HAS_CHUNK edges |
| `test_ingest_artifact_deterministic_chunk_id` | `ingest/artifact.rs` | chunk_id = "{artifact_id}:{index}" |
| `test_ingest_artifact_idempotent_rechunk` | `ingest/artifact.rs` | Re-chunking same artifact produces same chunk_ids |
| `test_ingest_artifact_latency` | `ingest/artifact.rs` | Text artifact < 100ms (release mode) |
| **Session Lifecycle** | | |
| `test_session_auto_create` | `ingest/session.rs` | New session created for new participant+goal combo |
| `test_session_reuse_existing` | `ingest/session.rs` | Existing active session returned |
| `test_session_end_explicit` | `ingest/session.rs` | end_session sets ended_at |
| `test_session_end_inactivity` | `ingest/session.rs` | Session ends after 30 min inactivity |
| `test_session_end_goal_terminal` | `ingest/session.rs` | Session ends when goal status -> achieved/failed |
| `test_session_end_triggers_summary` | `ingest/session.rs` | Session end queues P7d summarization |
| **Action Overflow** | | |
| `test_overflow_small_output` | `ingest/overflow.rs` | Output <= 256 tokens -> no overflow |
| `test_overflow_large_output` | `ingest/overflow.rs` | Output > 256 tokens -> Artifact created |
| `test_overflow_produced_edge` | `ingest/overflow.rs` | PRODUCED edge created from Action to Artifact |
| `test_overflow_summary_in_action` | `ingest/overflow.rs` | Action.output replaced with summary JSON |

### Integration Tests

| Test | What It Validates |
|---|---|
| `test_message_roundtrip` | Ingest message -> verify all nodes and edges exist in graph -> query back |
| `test_message_ordering_chain` | Ingest 20 messages -> verify NEXT chain is complete and gap_ms values are correct |
| `test_artifact_chunk_embed_pipeline` | Ingest artifact -> chunks created -> chunk embeddings computed -> artifact text_embedding pooled |
| `test_long_message_chunking_pipeline` | Ingest 2000-token message -> chunks created -> chunks have correct sizes and overlap |
| `test_code_artifact_full_path` | Ingest Python file -> tree-sitter chunks -> symbol_name extracted -> searchable |
| `test_session_lifecycle_full` | Auto-create session -> send messages -> inactivity timeout -> session ended -> summary queued |
| `test_dedup_prevents_double_ingest` | Ingest same artifact twice -> single node, single set of chunks |
| `test_action_overflow_full_path` | Record action with large output -> Artifact created -> chunked -> searchable |

### Latency Benchmarks

| Operation | Target | Measurement |
|---|---|---|
| Message ingest (no chunking) | < 10ms | `test_ingest_message_latency` with `--release` |
| Text artifact (1000 tokens) | < 100ms | `test_ingest_artifact_latency` with `--release` |
| Code artifact (500 lines Python) | < 100ms | Custom benchmark |
| Chunk 10KB text | < 50ms | `test_chunk_text_target_size` timing |

---

## Documentation Plan

| Document | Content |
|---|---|
| Inline rustdoc on `ingest_message` | Full step sequence, edge types created, session auto-creation behavior |
| Inline rustdoc on `ingest_artifact` | Dedup behavior, chunker selection, downstream pipeline queueing |
| Inline rustdoc on `Chunker` trait | How to add a new chunking strategy |
| Inline rustdoc on `ChunkConfig` | All parameters with defaults, valid ranges, and effects |
| Inline rustdoc on `get_or_create_session` | Session lifecycle rules, inactivity timeout behavior |
| Inline rustdoc on `maybe_overflow_output` | Threshold, Artifact creation, summary format |
| Per-chunker module docs | Algorithm description, supported languages (code), edge cases |

---

## Review Checklist

- [ ] Message ingest creates all 5 edge types (SENT_BY, ADDRESSED_TO, IN_SESSION, NEXT, HAS_CHUNK)
- [ ] NEXT edges have correct `gap_ms` computed from timestamps
- [ ] Session auto-creation only occurs when no active session exists for the participant+goal combo
- [ ] Chunk sizes are within 400-512 token target range (text chunker)
- [ ] No chunk breaks mid-sentence (text chunker)
- [ ] Overlap sentences are present between adjacent chunks
- [ ] Code chunks have `symbol_name` set for functions/classes/structs
- [ ] Code chunks have `language` set correctly
- [ ] tree-sitter parsers handle all 8 supported languages
- [ ] `chunk_id` is deterministic: `"{artifact_id}:{index}"`
- [ ] Artifact dedup uses hash comparison, not content comparison
- [ ] MIME type detection has a fallback for unknown types
- [ ] Session inactivity timeout is configurable (default 30 min)
- [ ] Session end triggers P7d summarization and re-embedding
- [ ] Action overflow threshold is configurable (default 256 tokens)
- [ ] Action overflow creates PRODUCED edge, not HAS_CHUNK
- [ ] All node IDs are UUID v7 when not caller-provided
- [ ] No `unwrap()` on tree-sitter parse results (handle parse failures gracefully)
- [ ] HTML chunker strips all non-content elements before splitting
- [ ] PDF chunker handles empty pages and image-only pages gracefully

---

## Definition of Done

1. **All files created**: `ingest/mod.rs`, `ingest/message.rs`, `ingest/artifact.rs`, `ingest/session.rs`, `ingest/overflow.rs`, `ingest/chunking/mod.rs`, `ingest/chunking/text.rs`, `ingest/chunking/code.rs`, `ingest/chunking/html.rs`, `ingest/chunking/pdf.rs`, `ingest/chunking/structured.rs` exist in `crates/uniko-extract/src/`.
2. **Message ingest complete**: All 5 edge types created correctly. NEXT chain ordering verified with 20+ messages.
3. **All chunking strategies functional**: Text, Code (6+ languages), HTML, PDF, CSV, JSON chunkers produce valid chunks.
4. **Chunk size compliance**: 95% of text chunks fall within 400-512 token range. Zero mid-sentence breaks.
5. **Code chunk quality**: All functions/classes in test samples produce chunks with correct `symbol_name` and `language`.
6. **Artifact dedup works**: Duplicate hash produces single Artifact node, zero duplicate chunks.
7. **Session lifecycle complete**: Auto-create, inactivity timeout, explicit end, goal terminal status -- all 4 triggers work.
8. **Action overflow works**: Outputs > 256 tokens create Artifact with PRODUCED edge.
9. **Latency targets met**: Message < 10ms, text artifact < 100ms (release mode, warm store).
10. **All unit tests pass**: `cargo nextest run -p uniko-extract --lib ingest` passes with zero failures.
11. **All integration tests pass**: `cargo nextest run -p uniko-extract --test ingest_integration` passes.
12. **Clippy clean**: `cargo clippy -p uniko-extract -D warnings` passes.
13. **Documented**: All public types and functions have rustdoc.
14. **Step trait integration**: `IngestStep` implements the `Step` trait from `uniko-pipes`, with `error_policy: Abort` (ingest failure is critical).
