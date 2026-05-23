# Phase 15: MCP Server

## Context

This phase exposes all uniko agent tools as MCP (Model Context Protocol) tools via the `uniko-mcp` crate. MCP is a standardized protocol that allows LLM agents (Claude, GPT-4, Gemini, etc.) to discover and invoke tools through a JSON-RPC interface. By implementing an MCP server, any MCP-compatible agent can use uniko as its cognitive memory backend through standard tool calling -- no custom integration code required.

The MCP server is a Layer 4 integration surface. It depends on `uniko-api` (the facade crate) and through it has access to all Cortex operations: lifecycle management, knowledge operations, query tools, and reasoning capabilities. The server translates between MCP's JSON-RPC protocol and uniko's Rust API, handling serialization, error mapping, and timeout enforcement.

This implements requirement F67 (DIF): "Expose all operations as MCP tools for external LLM agents." The latency target is NF19: MCP tool call round-trip overhead < 50ms (the overhead is the MCP serialization/deserialization and dispatch, not the underlying tool execution time).

**Why MCP matters:** MCP is becoming the standard interface for LLM tool calling. Claude Code, Claude Desktop, VS Code with Copilot, and many other agent platforms support MCP servers natively. By shipping an MCP server, uniko becomes immediately usable by any agent on any platform without requiring a Rust dependency or custom SDK.

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 12 (MVP complete) | Complete | All agent tools functional via Cortex |
| Phase 13 (Procedural Memory) | Complete or in progress | Procedure and Topic tools available for MCP exposure |
| Phase 14 (Hypothetical Reasoning) | Complete or in progress | ASSUME, ABDUCE, NL-to-Cypher tools available for MCP exposure |
| `uniko-api` facade (Phase 1) | Available | Re-exports all Cortex operations as a single public API |
| `rmcp` or equivalent Rust MCP SDK | Available | MCP protocol implementation (JSON-RPC, tool schema, transport) |
| `schemars` crate | Available | JSON Schema generation from Rust types |
| `serde` / `serde_json` | Available | Serialization/deserialization |
| `tokio` | Available | Async runtime for server event loop |
| `clap` | Available | CLI argument parsing for binary |
| `tracing` | Available | Structured logging to stderr |

## Sub-phases

---

### 15.1 -- MCP Server Setup

**Objective:** Establish the MCP server infrastructure: protocol handling, transport layer, and server lifecycle.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-mcp/src/lib.rs` | Modified (from Phase 1 skeleton) | MCP server entry point, public API |
| `crates/uniko-mcp/src/server.rs` | New | `UnikoMcpServer` struct, server lifecycle, request routing |
| `crates/uniko-mcp/src/transport.rs` | New | Transport abstraction: stdio and SSE |

#### Structs and Functions

```rust
/// The uniko MCP server. Wraps a Uniko instance and exposes its tools
/// via the Model Context Protocol.
pub struct UnikoMcpServer {
    /// The uniko instance providing all cognitive memory operations.
    uniko: Arc<Uniko>,
    /// Registered MCP tools.
    tools: Vec<McpToolDefinition>,
    /// Per-tool timeout configuration.
    tool_timeouts: HashMap<String, Duration>,
    /// Default timeout for tool calls.
    default_timeout: Duration,
    /// Cancellation token for graceful shutdown.
    cancel: CancellationToken,
}

/// MCP transport mode.
pub enum Transport {
    /// Standard I/O (stdin/stdout). Primary transport for CLI integration.
    Stdio,
    /// Server-Sent Events over HTTP. For remote/networked access.
    Sse { host: String, port: u16 },
}
```

#### Server Lifecycle

- `UnikoMcpServer::new(uniko: Arc<Uniko>, config: McpConfig) -> Self` -- Creates the server, registers all tools, configures timeouts.
- `UnikoMcpServer::serve(transport: Transport) -> Result<()>` -- Main event loop. Reads JSON-RPC requests from transport, dispatches to tool handlers, writes responses.
- `UnikoMcpServer::shutdown(&self) -> Result<()>` -- Cancel the server's cancellation token, drain in-flight requests, close transport.

#### Protocol Implementation

The MCP protocol is JSON-RPC 2.0 with specific method conventions:

| MCP Method | Purpose | Handler |
|---|---|---|
| `initialize` | Client introduces itself, server returns capabilities | Return server info, tool list |
| `tools/list` | Client requests available tools | Return all registered tool definitions with JSON Schema |
| `tools/call` | Client invokes a tool | Dispatch to tool handler, return result |
| `notifications/initialized` | Client confirms initialization | Acknowledge, begin accepting tool calls |
| `ping` | Keep-alive | Return `pong` |

```rust
/// Handle an incoming JSON-RPC request.
async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse;

/// Route a tools/call request to the appropriate handler.
async fn dispatch_tool_call(&self, name: &str, args: Value) -> Result<Value>;
```

#### Transport: stdio

The primary transport for MCP. Used by Claude Code, Claude Desktop, and most MCP clients.

- **Input:** Read JSON-RPC requests from stdin, one per line (newline-delimited JSON).
- **Output:** Write JSON-RPC responses to stdout, one per line.
- **Logging:** All tracing/logging output goes to stderr (never stdout, which is the MCP transport).

```rust
pub async fn serve_stdio(server: &UnikoMcpServer, cancel: &CancellationToken) -> Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    let reader = BufReader::new(stdin);
    let mut writer = BufWriter::new(stdout);
    // Read lines from stdin, parse as JSON-RPC, handle, write response to stdout
}
```

#### Transport: SSE (optional)

For remote access and multi-client scenarios:

- HTTP server listening on configurable host:port.
- POST `/message` for client -> server requests.
- GET `/sse` for server -> client event stream.
- Session management via connection ID.

```rust
pub async fn serve_sse(
    server: &UnikoMcpServer,
    host: &str,
    port: u16,
    cancel: &CancellationToken,
) -> Result<()>;
```

#### McpConfig

```rust
pub struct McpConfig {
    /// Default timeout for tool calls (default: 30s).
    pub default_timeout_secs: u64,
    /// Per-tool timeout overrides.
    pub tool_timeouts: HashMap<String, u64>,
    /// Server name reported in initialize response.
    pub server_name: String,
    /// Server version reported in initialize response.
    pub server_version: String,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            default_timeout_secs: 30,
            tool_timeouts: HashMap::new(),
            server_name: "uniko-mcp".to_string(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}
```

---

### 15.2 -- Tool Registration & Schema Generation

**Objective:** Map each uniko agent tool to an MCP tool with auto-generated JSON Schema for parameters and descriptions from rustdoc comments.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-mcp/src/tools.rs` | New | Tool definitions, parameter types, schema generation |
| `crates/uniko-mcp/src/tools/lifecycle.rs` | New | Lifecycle tool implementations |
| `crates/uniko-mcp/src/tools/knowledge.rs` | New | Knowledge tool implementations |
| `crates/uniko-mcp/src/tools/query.rs` | New | Query tool implementations |
| `crates/uniko-mcp/src/tools/system.rs` | New | System tool implementations |

#### Tool Registry

All tools are registered at server creation time via a `ToolRegistry`:

```rust
pub struct McpToolDefinition {
    /// Tool name as exposed to MCP clients.
    pub name: String,
    /// Human-readable description (from rustdoc).
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
}

pub struct ToolRegistry {
    tools: Vec<McpToolDefinition>,
    handlers: HashMap<String, Box<dyn ToolHandler>>,
}

#[async_trait]
pub trait ToolHandler: Send + Sync {
    async fn handle(&self, uniko: &Uniko, args: Value) -> Result<Value>;
}

/// Register all uniko tools with the registry.
pub fn register_all_tools(registry: &mut ToolRegistry);
```

#### Tool Definitions

**Lifecycle Tools:**

| MCP Tool Name | Uniko Operation | Parameters | Returns |
|---|---|---|---|
| `uniko_create_goal` | Create a new Goal node | `title: string` (required), `description: string`, `owner_id: string`, `deadline: string`, `metrics: object`, `guardrails: object` | `{ goal_id: string }` |
| `uniko_update_goal` | Update Goal status/fields | `goal_id: string` (required), `status: string`, `title: string`, `description: string`, `metrics: object` | `{ updated: bool }` |
| `uniko_create_task` | Create a new Task node | `title: string` (required), `goal_id: string`, `description: string`, `priority: number`, `assigned_to: string` | `{ task_id: string }` |
| `uniko_update_task` | Update Task status/fields | `task_id: string` (required), `status: string`, `title: string`, `description: string`, `priority: number` | `{ updated: bool }` |
| `uniko_start_session` | Start a new Session | `topic: string`, `task_id: string`, `goal_id: string`, `participant_ids: string[]` | `{ session_id: string }` |
| `uniko_end_session` | End an active Session | `session_id: string` (required) | `{ ended: bool, summary: string }` |
| `uniko_create_organization` | Create Organization node | `name: string` (required) | `{ org_id: string }` |
| `uniko_create_team` | Create Team node | `name: string` (required), `org_id: string`, `purpose: string` | `{ team_id: string }` |
| `uniko_add_member` | Add Participant to Org/Team | `participant_id: string` (required), `org_id: string`, `team_id: string`, `role: string` | `{ added: bool }` |

**Knowledge Tools:**

| MCP Tool Name | Uniko Operation | Parameters | Returns |
|---|---|---|---|
| `uniko_record_episode` | Record an agent episode | `action_type: string` (required), `outcome: string`, `state: object`, `delta: object`, `importance: number`, `task_id: string` | `{ episode_id: string }` |
| `uniko_record_action` | Record an action (tool call) | `action_type: string` (required), `input: object`, `output: object`, `status: string`, `duration_ms: number` | `{ action_id: string }` |
| `uniko_add_observation` | Add an explicit observation | `content: string` (required), `subject: string`, `entity_ids: string[]` | `{ observation_id: string }` |
| `uniko_assert_fact` | Assert a fact (create or reinforce) | `subject: string` (required), `predicate: string` (required), `object: string`, `confidence: number` | `{ fact_id: string, reinforced: bool }` |
| `uniko_invalidate_fact` | Invalidate a fact (close BTIC interval) | `fact_id: string` (required), `reason: string` | `{ invalidated: bool }` |
| `uniko_add_rule` | Add an authored Locy rule | `name: string` (required), `source: string` (required), `natural_language: string` | `{ rule_id: string }` |
| `uniko_author_rule` | LLM-assisted rule authoring | `description: string` (required), `examples: object[]` | `{ rule_id: string, source: string }` |
| `uniko_share_fact` | Share a fact to global scope | `fact_id: string` (required) | `{ shared: bool }` |
| `uniko_shared_facts` | Retrieve globally shared facts | `predicate: string`, `subject: string`, `limit: number` | `{ facts: object[] }` |

**Query Tools:**

| MCP Tool Name | Uniko Operation | Parameters | Returns |
|---|---|---|---|
| `uniko_recall` | Recall context for a query | `query: string` (required), `limit: number`, `recency_days: number`, `min_reliability: number`, `include_procedures: bool`, `contrastive: bool` | `{ context: object }` (ContextBundle) |
| `uniko_search_entities` | Search entities by name/type | `query: string`, `entity_type: string`, `limit: number` | `{ entities: object[] }` |
| `uniko_search_facts` | Search facts by subject/predicate | `subject: string`, `predicate: string`, `query: string`, `limit: number` | `{ facts: object[] }` |
| `uniko_search_messages` | Search messages by content | `query: string` (required), `session_id: string`, `limit: number` | `{ messages: object[] }` |
| `uniko_working_memory` | Get working memory for a goal | `goal_id: string` (required) | `{ context: object }` (ContextBundle) |
| `uniko_assume` | Hypothetical reasoning | `mutations: object[]` (required), `query: string` (required), `query_type: string` | `{ results: object[], mutations_applied: number, query_time_ms: number }` |
| `uniko_abduce` | Abductive reasoning | `subject: string` (required), `predicate: string` (required), `object: string` (required), `max_depth: number` | `{ supporting_facts: object[], derivation_chain: object[], confidence: number }` |

**System Tools:**

| MCP Tool Name | Uniko Operation | Parameters | Returns |
|---|---|---|---|
| `uniko_health` | Pipeline health status | (none) | `PipelineHealth` |

#### JSON Schema Generation

Use `schemars` to auto-generate JSON Schema from Rust parameter types:

```rust
use schemars::JsonSchema;

/// Parameters for uniko_create_goal tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateGoalParams {
    /// The goal title (required).
    pub title: String,
    /// Detailed description of the goal.
    pub description: Option<String>,
    /// Participant ID of the goal owner.
    pub owner_id: Option<String>,
    /// Deadline as ISO 8601 datetime string.
    pub deadline: Option<String>,
    /// Target metrics as key-value pairs.
    pub metrics: Option<serde_json::Value>,
    /// Constraints: budget, compliance, risk.
    pub guardrails: Option<serde_json::Value>,
}
```

Each parameter struct derives `JsonSchema`, which generates the JSON Schema automatically. The `/// doc comments` become `description` fields in the schema. Required fields (non-Option) are marked as required in the schema.

```rust
/// Generate JSON Schema for a tool's parameters.
fn generate_schema<T: JsonSchema>() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap()
}
```

#### Tool Descriptions

Tool descriptions are derived from rustdoc comments on the parameter structs or handler functions. Each description should:
- Start with a verb (e.g., "Create a new goal", "Search entities by name")
- Be concise (one sentence)
- Mention what the tool returns

---

### 15.3 -- Request Handling & Serialization

**Objective:** Implement the request/response pipeline: deserialize MCP args, dispatch to Cortex tools, serialize results, and handle errors with appropriate MCP error codes.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-mcp/src/handler.rs` | New | Request handling, error mapping, timeout enforcement |

#### Structs and Functions

```rust
/// Handle a tool call: deserialize args, call Cortex, serialize result.
pub async fn handle_tool_call(
    server: &UnikoMcpServer,
    name: &str,
    args: Value,
) -> Result<Value>;

/// MCP error response.
pub struct McpError {
    pub code: i32,
    pub message: String,
    pub data: Option<Value>,
}
```

#### Request Processing Flow

1. **Lookup tool:** Find the registered handler by tool name.
   - If not found: return MCP error with code `-32601` (Method not found).

2. **Deserialize args:** Attempt to deserialize the JSON `args` into the expected parameter type.
   - If deserialization fails: return MCP error with code `-32602` (Invalid params) and include the deserialization error message.

3. **Timeout wrapper:** Wrap the tool execution in `tokio::time::timeout()` with the configured timeout (per-tool override or default 30s).

4. **Execute:** Call the corresponding Cortex/Uniko method.
   - Propagate any `UnikoError` to the error mapper.

5. **Serialize result:** Convert the Rust return type to JSON using serde.

6. **Return:** JSON-RPC response with result or error.

#### Error Mapping

| UnikoError Variant | MCP Error Code | Message |
|---|---|---|
| `Storage(msg)` | `-32000` (Server error) | `"Storage error: {msg}"` |
| `Search(msg)` | `-32000` | `"Search error: {msg}"` |
| `Schema(msg)` | `-32000` | `"Schema error: {msg}"` |
| `Pipeline(msg)` | `-32000` | `"Pipeline error: {msg}"` |
| `Locy(msg)` | `-32000` | `"Locy error: {msg}"` |
| `Config(msg)` | `-32602` (Invalid params) | `"Configuration error: {msg}"` |
| `Embedding(msg)` | `-32000` | `"Embedding error: {msg}"` |
| `Llm(msg)` | `-32000` | `"LLM error: {msg}"` |
| `Timeout(ms)` | `-32000` | `"Operation timed out after {ms}ms"` |
| `Internal(msg)` | `-32603` (Internal error) | `"Internal error: {msg}"` |

Additionally:
- Tool not found: `-32601` (Method not found)
- Invalid args: `-32602` (Invalid params)
- Timeout (from MCP layer): `-32000` with `"Tool execution timed out after {timeout}s"`

#### Timeout Enforcement

```rust
async fn execute_with_timeout(
    handler: &dyn ToolHandler,
    uniko: &Uniko,
    args: Value,
    timeout: Duration,
) -> Result<Value> {
    match tokio::time::timeout(timeout, handler.handle(uniko, args)).await {
        Ok(result) => result,
        Err(_) => Err(UnikoError::Timeout(timeout.as_millis() as u64)),
    }
}
```

Default timeout: 30s. Configurable per-tool via `McpConfig::tool_timeouts`:
- Knowledge tools (record_episode, assert_fact): 10s
- Query tools (recall, search): 30s
- Reasoning tools (assume, abduce): 30s
- System tools (health): 5s

---

### 15.4 -- MCP Integration Testing

**Objective:** End-to-end testing of the MCP server, simulating an MCP client to verify correct tool discovery, execution, error handling, and concurrent access.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-mcp/tests/mcp_integration.rs` | New | Integration tests simulating MCP client |
| `crates/uniko-mcp/tests/helpers.rs` | New | Test utilities: mock MCP client, fixture creation |

#### Test Helper: Mock MCP Client

```rust
/// A mock MCP client for testing. Sends JSON-RPC requests via in-memory channel.
struct MockMcpClient {
    server: UnikoMcpServer,
}

impl MockMcpClient {
    /// Send a tool call and return the result.
    async fn call_tool(&self, name: &str, args: Value) -> Result<Value>;

    /// List available tools.
    async fn list_tools(&self) -> Vec<McpToolDefinition>;

    /// Initialize the MCP session.
    async fn initialize(&self) -> Value;
}
```

#### Test Categories

**Tool Discovery Tests:**

| Test | What It Validates |
|---|---|
| `test_initialize_response` | Server returns valid initialize response with capabilities |
| `test_list_tools_returns_all` | All registered tools returned with schemas |
| `test_tool_schema_valid` | Each tool's input_schema is valid JSON Schema |
| `test_tool_descriptions_present` | Every tool has a non-empty description |

**Per-Tool Execution Tests (one per tool):**

| Test | What It Validates |
|---|---|
| `test_create_goal` | Valid params -> goal created, goal_id returned |
| `test_update_goal` | Valid params -> goal updated |
| `test_create_task` | Valid params -> task created, task_id returned |
| `test_update_task` | Valid params -> task updated |
| `test_start_session` | Valid params -> session started, session_id returned |
| `test_end_session` | Valid params -> session ended, summary returned |
| `test_create_organization` | Valid params -> org created |
| `test_create_team` | Valid params -> team created |
| `test_add_member` | Valid params -> member added |
| `test_record_episode` | Valid params -> episode recorded |
| `test_record_action` | Valid params -> action recorded |
| `test_add_observation` | Valid params -> observation added |
| `test_assert_fact` | Valid params -> fact asserted |
| `test_invalidate_fact` | Valid params -> fact invalidated |
| `test_add_rule` | Valid params -> rule added |
| `test_share_fact` | Valid params -> fact shared |
| `test_shared_facts` | Valid params -> shared facts returned |
| `test_recall` | Valid params -> context bundle returned |
| `test_search_entities` | Valid params -> matching entities returned |
| `test_search_facts` | Valid params -> matching facts returned |
| `test_search_messages` | Valid params -> matching messages returned |
| `test_working_memory` | Valid params -> working memory context returned |
| `test_assume` | Valid params -> assume result returned |
| `test_abduce` | Valid params -> abduce result returned |
| `test_health` | No params -> pipeline health returned |

**Error Handling Tests:**

| Test | What It Validates |
|---|---|
| `test_invalid_tool_name` | Unknown tool -> error -32601 (Method not found) |
| `test_invalid_params_missing_required` | Missing required field -> error -32602 with message |
| `test_invalid_params_wrong_type` | String where number expected -> error -32602 |
| `test_tool_timeout` | Slow operation -> timeout error with configured timeout |
| `test_pipeline_failure` | Cortex returns error -> error -32000 with message |
| `test_circuit_breaker_open` | LLM circuit open -> appropriate error for LLM-dependent tools |

**Concurrent Access Tests:**

| Test | What It Validates |
|---|---|
| `test_concurrent_tool_calls` | 10 simultaneous tool calls all succeed without interference |
| `test_concurrent_read_write` | Concurrent reads and writes don't deadlock |
| `test_concurrent_sessions` | Multiple simultaneous sessions don't corrupt state |

**Latency Tests:**

| Test | What It Validates |
|---|---|
| `test_mcp_overhead` | MCP serialization/deserialization + dispatch < 50ms (NF19) |
| `test_mcp_overhead_per_tool` | Measure overhead per tool category (lifecycle, knowledge, query) |

#### Latency Measurement

```rust
async fn measure_mcp_overhead(client: &MockMcpClient) {
    // Measure: time from JSON-RPC request receipt to tool handler entry
    // + time from tool handler return to JSON-RPC response send.
    // This excludes the actual tool execution time.
    let start = Instant::now();
    let result = client.call_tool("uniko_health", json!({})).await;
    let total = start.elapsed();
    // uniko_health is near-instantaneous, so total ~= MCP overhead
    assert!(total < Duration::from_millis(50), "MCP overhead: {:?}", total);
}
```

---

### 15.5 -- Binary Entry Point

**Objective:** Create the standalone `uniko-mcp` binary that initializes uniko, starts the MCP server, and handles signals for graceful shutdown.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-mcp/src/main.rs` | New | Binary entry point, CLI parsing, initialization |

#### CLI Interface

```
uniko-mcp [OPTIONS]

Options:
    --db-path <PATH>       Path to the uniko database directory [default: ./uniko.db]
    --config <PATH>        Path to configuration file (TOML) [optional]
    --transport <MODE>     Transport mode: stdio (default) or sse [default: stdio]
    --host <HOST>          Host for SSE transport [default: 127.0.0.1]
    --port <PORT>          Port for SSE transport [default: 3000]
    --log-level <LEVEL>    Log level: error, warn, info, debug, trace [default: info]
    --timeout <SECS>       Default tool timeout in seconds [default: 30]
    --version              Print version
    --help                 Print help
```

#### Implementation

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "uniko-mcp", about = "MCP server for uniko cognitive memory")]
struct Cli {
    #[arg(long, default_value = "./uniko.db")]
    db_path: PathBuf,

    #[arg(long)]
    config: Option<PathBuf>,

    #[arg(long, default_value = "stdio")]
    transport: String,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value = "3000")]
    port: u16,

    #[arg(long, default_value = "info")]
    log_level: String,

    #[arg(long, default_value = "30")]
    timeout: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // 1. Initialize tracing to stderr (not stdout)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(&cli.log_level)
        .init();

    // 2. Initialize Uniko
    let config = load_config(&cli)?;
    let uniko = Arc::new(Uniko::open(&cli.db_path, config)?);

    // 3. Create MCP server
    let mcp_config = McpConfig {
        default_timeout_secs: cli.timeout,
        ..Default::default()
    };
    let server = UnikoMcpServer::new(uniko.clone(), mcp_config);

    // 4. Set up signal handling
    let cancel = server.cancel.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        info!("Received SIGINT, initiating graceful shutdown");
        cancel.cancel();
    });

    // 5. Serve
    let transport = match cli.transport.as_str() {
        "stdio" => Transport::Stdio,
        "sse" => Transport::Sse { host: cli.host, port: cli.port },
        other => return Err(anyhow!("Unknown transport: {}", other)),
    };

    info!(transport = %cli.transport, db_path = %cli.db_path.display(), "Starting uniko MCP server");
    server.serve(transport).await?;

    // 6. Cleanup
    info!("MCP server shut down");
    Ok(())
}
```

#### Logging to stderr

All logging goes to stderr because stdout is the MCP transport (for stdio mode). This is critical -- any non-JSON-RPC output on stdout will break the MCP client.

```rust
// CORRECT: tracing output to stderr
tracing_subscriber::fmt()
    .with_writer(std::io::stderr)
    .init();

// NEVER: println!(), print!(), or any stdout output during server operation
```

#### Signal Handling

| Signal | Behavior |
|---|---|
| SIGINT (Ctrl+C) | Initiate graceful shutdown: cancel token -> drain in-flight requests -> close transport -> exit |
| SIGTERM | Same as SIGINT |

Graceful shutdown timeout: 10 seconds. If in-flight requests don't complete within 10 seconds, force exit.

#### Configuration File (optional)

If `--config` is provided, load a TOML file:

```toml
[uniko]
ingest_queue_capacity = 200
consolidation_threshold = 20
half_life_days = 30.0

[mcp]
default_timeout_secs = 30

[mcp.tool_timeouts]
uniko_recall = 60
uniko_assume = 30
uniko_health = 5
```

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_server_creation` | `server.rs` | UnikoMcpServer creates with default config |
| `test_server_creation_custom_config` | `server.rs` | Custom config values applied correctly |
| `test_tool_registration` | `tools.rs` | All expected tools registered |
| `test_tool_count` | `tools.rs` | Correct number of tools registered (26 tools) |
| `test_tool_schema_generation` | `tools.rs` | JSON Schema generated from parameter types |
| `test_tool_schema_required_fields` | `tools.rs` | Required fields marked correctly in schema |
| `test_tool_schema_optional_fields` | `tools.rs` | Optional fields not marked required |
| `test_handle_tool_call_valid` | `handler.rs` | Valid tool call returns result |
| `test_handle_tool_call_invalid_name` | `handler.rs` | Unknown tool returns error -32601 |
| `test_handle_tool_call_invalid_args` | `handler.rs` | Bad args returns error -32602 |
| `test_handle_tool_call_timeout` | `handler.rs` | Timeout returns error with timeout message |
| `test_error_mapping_storage` | `handler.rs` | UnikoError::Storage maps to -32000 |
| `test_error_mapping_config` | `handler.rs` | UnikoError::Config maps to -32602 |
| `test_error_mapping_internal` | `handler.rs` | UnikoError::Internal maps to -32603 |
| `test_error_mapping_timeout` | `handler.rs` | UnikoError::Timeout maps to -32000 |
| `test_transport_stdio_request_parsing` | `transport.rs` | JSON-RPC request parsed from stdin line |
| `test_transport_stdio_response_writing` | `transport.rs` | JSON-RPC response written as single line to stdout |
| `test_transport_stdio_invalid_json` | `transport.rs` | Invalid JSON on stdin returns parse error |
| `test_cli_default_args` | `main.rs` | Default CLI arguments applied correctly |
| `test_cli_custom_args` | `main.rs` | Custom CLI arguments override defaults |
| `test_config_file_loading` | `main.rs` | TOML config file loaded and applied |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_full_mcp_lifecycle` | `tests/mcp_integration.rs` | initialize -> list tools -> call tool -> verify result |
| `test_each_lifecycle_tool` | `tests/mcp_integration.rs` | Each lifecycle tool (create_goal, etc.) callable with correct response |
| `test_each_knowledge_tool` | `tests/mcp_integration.rs` | Each knowledge tool callable with correct response |
| `test_each_query_tool` | `tests/mcp_integration.rs` | Each query tool callable with correct response |
| `test_each_system_tool` | `tests/mcp_integration.rs` | Health tool callable with correct response |
| `test_error_handling_invalid_tool` | `tests/mcp_integration.rs` | Invalid tool name returns error -32601 |
| `test_error_handling_invalid_params` | `tests/mcp_integration.rs` | Invalid params returns error -32602 |
| `test_error_handling_timeout` | `tests/mcp_integration.rs` | Timeout triggers correct error response |
| `test_concurrent_calls` | `tests/mcp_integration.rs` | 10 simultaneous tool calls succeed without interference |
| `test_concurrent_read_write` | `tests/mcp_integration.rs` | Concurrent reads and writes don't deadlock or corrupt |
| `test_mcp_overhead_latency` | `tests/mcp_integration.rs` | MCP overhead < 50ms (NF19) |
| `test_graceful_shutdown` | `tests/mcp_integration.rs` | Server shuts down cleanly on cancel, in-flight requests complete |
| `test_shutdown_under_load` | `tests/mcp_integration.rs` | Shutdown during active tool calls completes gracefully |
| `test_recall_via_mcp` | `tests/mcp_integration.rs` | End-to-end: create goal + session + messages via MCP -> recall returns context |
| `test_assume_via_mcp` | `tests/mcp_integration.rs` | End-to-end: assert facts via MCP -> assume via MCP -> verify hypothetical results |

### Compatibility Tests

| Test | What It Validates |
|---|---|
| `test_claude_code_compatible` | MCP protocol messages match Claude Code's expected format |
| `test_initialize_handshake` | Initialize/initialized handshake completes correctly |
| `test_ping_pong` | Ping request returns pong response |
| `test_json_rpc_batch` | Batch JSON-RPC requests handled correctly (if supported) |

### Validation Criteria

- MCP server starts and accepts connections via stdio and SSE
- All 26 tools discoverable via `tools/list`
- Each tool callable with correct parameters and returns expected result
- Error handling: invalid tool, invalid params, timeout, pipeline failure
- Concurrent tool calls (10+ simultaneous) succeed without interference
- MCP overhead < 50ms (NF19) -- measured as serialization + dispatch, excluding tool execution
- Graceful shutdown: in-flight requests complete, server exits cleanly
- Logging goes to stderr only (stdout is MCP transport)
- Compatible with Claude Code and other MCP clients

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module doc | `lib.rs` | Overview of uniko-mcp, how to start the server, transport options |
| Module doc | `server.rs` | UnikoMcpServer lifecycle, protocol handling, configuration |
| Module doc | `tools.rs` | Complete tool reference table, how tools map to Cortex operations |
| Module doc | `handler.rs` | Request processing flow, error mapping, timeout behavior |
| Module doc | `transport.rs` | Stdio vs SSE transport, when to use each |
| Inline rustdoc on `McpToolDefinition` | `tools.rs` | How tool schemas are generated, description conventions |
| Inline rustdoc on `McpConfig` | `server.rs` | All configuration options with defaults |
| Inline rustdoc on `Cli` | `main.rs` | CLI argument documentation |
| User guide | `tools/lifecycle.rs` header | How lifecycle tools map to agent workflows |
| User guide | `tools/knowledge.rs` header | How knowledge tools map to memory operations |
| User guide | `tools/query.rs` header | How query tools map to recall/search operations |

---

## Review Checklist

- [ ] `UnikoMcpServer::new()` creates server with all tools registered
- [ ] `UnikoMcpServer::serve()` handles stdio transport (read stdin, write stdout)
- [ ] `UnikoMcpServer::serve()` handles SSE transport (HTTP endpoints)
- [ ] All logging goes to stderr (never stdout in stdio mode)
- [ ] JSON-RPC 2.0 protocol correctly implemented (id, method, params, result, error)
- [ ] `initialize` returns server capabilities and tool list
- [ ] `tools/list` returns all 26 tools with JSON Schema
- [ ] `tools/call` dispatches to correct handler
- [ ] `ping` returns pong
- [ ] All 9 lifecycle tools registered and functional
- [ ] All 9 knowledge tools registered and functional
- [ ] All 7 query tools registered and functional
- [ ] `uniko_health` system tool registered and functional
- [ ] JSON Schema auto-generated from Rust types via schemars
- [ ] Required parameters marked as required in JSON Schema
- [ ] Optional parameters not marked as required
- [ ] Tool descriptions are non-empty and start with a verb
- [ ] Error mapping: invalid tool -> -32601, invalid params -> -32602, internal -> -32603
- [ ] Timeout enforced per-tool (configurable, default 30s)
- [ ] Concurrent tool calls handled correctly (no deadlocks, no corruption)
- [ ] MCP overhead < 50ms (NF19, verified by benchmark test)
- [ ] Signal handling: SIGINT/SIGTERM trigger graceful shutdown
- [ ] Graceful shutdown: in-flight requests complete within 10s timeout
- [ ] CLI supports --db-path, --config, --transport, --host, --port, --log-level, --timeout
- [ ] TOML config file loading works when --config provided
- [ ] Default config values match spec
- [ ] Binary builds: `cargo build -p uniko-mcp --release` succeeds
- [ ] No `unwrap()` or `expect()` in production code paths
- [ ] No stdout output except JSON-RPC responses (stdio transport)

---

## Definition of Done

1. **Server starts and serves:** `uniko-mcp` binary starts, accepts MCP connections via stdio, and serves tool calls. The server reports its name, version, and tool list on initialize.
2. **All tools exposed:** All 26 tools (9 lifecycle, 9 knowledge, 7 query, 1 system) are discoverable via `tools/list` with valid JSON Schema for parameters.
3. **Tool execution correct:** Each tool can be called with valid parameters and returns the correct result. Results match what the direct Cortex API would return.
4. **Error handling robust:** Invalid tool names return -32601. Invalid parameters return -32602 with descriptive message. Internal errors return -32603. Timeouts return -32000 with timeout message.
5. **Concurrency safe:** 10+ simultaneous tool calls from the same client succeed without interference, deadlocks, or state corruption.
6. **Latency target met:** MCP serialization/deserialization + dispatch overhead < 50ms (NF19), measured by timing a near-instantaneous tool (uniko_health).
7. **Graceful shutdown works:** SIGINT/SIGTERM triggers orderly shutdown. In-flight requests complete within 10s. Server exits cleanly with exit code 0.
8. **Logging correct:** All tracing output goes to stderr. Stdout contains only JSON-RPC responses (stdio transport). No non-JSON output on stdout during operation.
9. **CLI functional:** All CLI arguments work: --db-path, --config, --transport, --host, --port, --log-level, --timeout. TOML config file loading works.
10. **Compatible with MCP clients:** Server protocol messages are compatible with Claude Code, Claude Desktop, and other standard MCP clients. Initialize handshake, tool discovery, and tool calling all work end-to-end.
11. **All tests pass:** `cargo nextest run -n auto -p uniko-mcp` passes with zero failures for all unit, integration, and compatibility tests.
12. **Binary builds:** `cargo build -p uniko-mcp --release` succeeds with no warnings.
13. **Clippy clean:** `cargo clippy -p uniko-mcp -- -D warnings` passes.
14. **Documented:** All public types, tool definitions, and CLI arguments have rustdoc with usage examples.
