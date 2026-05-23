# Sub-Phase 3: KnowledgeBase Layer (uniko-store) -- Storage & Search

## Context

This sub-phase builds the complete Layer 1 API in `uniko-store`: the KnowledgeBase struct that wraps uni-db and provides typed node/edge CRUD operations, vector search, fulltext search, hybrid search with Reciprocal Rank Fusion (RRF), graph traversal (including Personalized PageRank), and Locy runtime integration (rule execution, ASSUME/ABDUCE).

Layer 1 is the foundation that all higher layers depend on. It knows about nodes, edges, embeddings, and indexes -- but nothing about cognitive memory semantics. It does not know that a Message represents a communication event or that a Fact was consolidated from observations. That meaning lives in higher layers (Extract, Memory, Cortex). Layer 1 provides the storage and search primitives they need.

**Key performance constraints:**
- NF1: Store message (create node + edges) < 10ms
- NF2: Vector search (top-10) < 20ms
- NF3: Hybrid search (vector + FTS + graph) < 50ms
- NF4: Graph traversal (3-hop) < 5ms
- NF9: Single ASSUME (hypothetical reasoning) < 200ms

## Prerequisites

- **Sub-phase 2 complete:** All 16+ node types, 35+ edge types, indexes, and BTIC helpers are defined and tested. `register_schema()` works.
- uni-db API fully available: Database, Transaction, Node, Edge, Index, Btic, Vector, OpenCypher query execution, Locy rule execution.
- `uniko-store/src/storage/` and `uniko-store/src/search/` and `uniko-store/src/locy/` module directories exist (stubs from Phase 1).
- Shared types functional: `NodeId`, `EdgeId`, `UnikoError`, `Result<T>`, `UnikoConfig`.

## Sub-phases

---

### 3.1 -- Node & Edge CRUD Operations

**Objective:** Provide typed, transactional CRUD wrappers over uni-db for all node and edge types defined in Phase 2.

#### Files to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/storage/mod.rs` | Rust | `KnowledgeBase` struct, module root |
| `crates/uniko-store/src/storage/nodes.rs` | Rust | Generic node CRUD operations |
| `crates/uniko-store/src/storage/edges.rs` | Rust | Edge CRUD operations |
| `crates/uniko-store/src/storage/batch.rs` | Rust | Batch creation operations |

#### `storage/mod.rs` -- KnowledgeBase Struct

```rust
/// The Layer 1 storage engine. Wraps a uni-db Database instance and provides
/// typed CRUD operations, search, and Locy runtime access.
///
/// KnowledgeBase is the only entry point for graph operations in uniko.
/// All higher layers (Extract, Memory, Cortex) interact with the graph through this struct.
pub struct KnowledgeBase {
    db: Database,
    config: UnikoConfig,
}

impl KnowledgeBase {
    /// Create a new KnowledgeBase with an in-memory database.
    pub fn new_in_memory(config: UnikoConfig) -> Result<Self>;

    /// Create a new KnowledgeBase with a persistent database at the given path.
    pub fn new_persistent(path: &Path, config: UnikoConfig) -> Result<Self>;

    /// Open an existing KnowledgeBase from a persistent database.
    pub fn open(path: &Path, config: UnikoConfig) -> Result<Self>;

    /// Get a reference to the underlying database (for advanced operations).
    pub fn db(&self) -> &Database;

    /// Get the configuration.
    pub fn config(&self) -> &UnikoConfig;
}
```

#### `storage/nodes.rs` -- Generic Node CRUD

```rust
impl KnowledgeBase {
    /// Create a node of the given type with the given properties.
    /// Uses the *_id field as ext_id for MERGE semantics (upsert).
    ///
    /// Target: contributes to NF1 (< 10ms for message creation).
    pub fn create_node(
        &self,
        node_type: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<NodeId>;

    /// Get a node by its internal NodeId.
    /// Returns all properties as a HashMap.
    pub fn get_node(&self, id: NodeId) -> Result<Option<HashMap<String, Value>>>;

    /// Get a node by its external ID (e.g., message_id, entity_id).
    /// Uses the Hash index on the *_id field.
    pub fn get_node_by_ext_id(
        &self,
        node_type: &str,
        ext_id_field: &str,
        ext_id_value: &str,
    ) -> Result<Option<(NodeId, HashMap<String, Value>)>>;

    /// Update specific properties on a node.
    /// Only the provided properties are changed; others are preserved.
    pub fn update_node(
        &self,
        id: NodeId,
        properties: &HashMap<String, Value>,
    ) -> Result<()>;

    /// Delete a node and all its edges.
    pub fn delete_node(&self, id: NodeId) -> Result<()>;

    /// Merge (upsert) a node: if a node with the same ext_id exists, update it;
    /// otherwise, create a new node.
    ///
    /// This is the primary creation method. All *_id fields serve as ext_id
    /// for MERGE semantics (ADR-1).
    pub fn merge_node(
        &self,
        node_type: &str,
        ext_id_field: &str,
        ext_id_value: &str,
        properties: &HashMap<String, Value>,
    ) -> Result<NodeId>;

    /// Query nodes by type with optional property filters.
    /// Returns matching nodes with all their properties.
    pub fn query_nodes(
        &self,
        node_type: &str,
        filters: &HashMap<String, Value>,
        limit: Option<usize>,
    ) -> Result<Vec<(NodeId, HashMap<String, Value>)>>;
}
```

#### `storage/edges.rs` -- Edge CRUD

```rust
impl KnowledgeBase {
    /// Create an edge between two nodes.
    /// Properties are optional (e.g., role on SENT_BY, gap_ms on NEXT).
    pub fn create_edge(
        &self,
        edge_type: &str,
        from: NodeId,
        to: NodeId,
        properties: Option<&HashMap<String, Value>>,
    ) -> Result<EdgeId>;

    /// Get all edges of a given type from/to a node.
    /// Direction: Outgoing, Incoming, or Both.
    pub fn get_edges(
        &self,
        node_id: NodeId,
        edge_type: &str,
        direction: Direction,
    ) -> Result<Vec<EdgeRecord>>;

    /// Get all edges from/to a node regardless of type.
    pub fn get_all_edges(
        &self,
        node_id: NodeId,
        direction: Direction,
    ) -> Result<Vec<EdgeRecord>>;

    /// Delete an edge by its EdgeId.
    pub fn delete_edge(&self, id: EdgeId) -> Result<()>;

    /// Delete all edges of a given type between two specific nodes.
    pub fn delete_edges_between(
        &self,
        edge_type: &str,
        from: NodeId,
        to: NodeId,
    ) -> Result<u64>;

    /// Update properties on an existing edge.
    pub fn update_edge(
        &self,
        id: EdgeId,
        properties: &HashMap<String, Value>,
    ) -> Result<()>;
}

/// A record returned from edge queries.
pub struct EdgeRecord {
    pub id: EdgeId,
    pub edge_type: String,
    pub from: NodeId,
    pub to: NodeId,
    pub properties: HashMap<String, Value>,
}

/// Direction for edge traversal.
pub enum Direction {
    Outgoing,
    Incoming,
    Both,
}
```

#### `storage/batch.rs` -- Batch Operations

```rust
impl KnowledgeBase {
    /// Create multiple nodes in a single transaction.
    /// Returns the NodeIds of all created nodes.
    ///
    /// Significantly faster than individual create_node calls
    /// for bulk ingestion (e.g., chunking an artifact into 50 chunks).
    pub fn batch_create_nodes(
        &self,
        node_type: &str,
        items: &[HashMap<String, Value>],
    ) -> Result<Vec<NodeId>>;

    /// Create multiple edges in a single transaction.
    pub fn batch_create_edges(
        &self,
        edge_type: &str,
        edges: &[(NodeId, NodeId, Option<HashMap<String, Value>>)],
    ) -> Result<Vec<EdgeId>>;

    /// Execute a function within a transaction.
    /// If the function returns Ok, the transaction is committed.
    /// If it returns Err, the transaction is rolled back.
    pub fn transaction<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Transaction) -> Result<T>;
}
```

---

### 3.2 -- Vector Search

**Objective:** Provide vector similarity search across all embedding fields, supporting filtered queries and multi-type search.

#### File to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/search/mod.rs` | Rust | Search module root, shared types |
| `crates/uniko-store/src/search/vector.rs` | Rust | Vector search implementation |

#### `search/mod.rs` -- Shared Search Types

```rust
/// A single search result from any search method.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: NodeId,
    pub node_type: String,
    pub score: f64,           // similarity/relevance score
    pub properties: HashMap<String, Value>,
}

/// Filter predicate for search queries.
#[derive(Debug, Clone)]
pub enum Filter {
    Eq(String, Value),           // property == value
    Ne(String, Value),           // property != value
    Gt(String, Value),           // property > value
    Lt(String, Value),           // property < value
    Gte(String, Value),          // property >= value
    Lte(String, Value),          // property <= value
    In(String, Vec<Value>),      // property IN [values]
    And(Vec<Filter>),            // all must match
    Or(Vec<Filter>),             // any must match
    Not(Box<Filter>),            // negation
}
```

#### `search/vector.rs` -- Vector Search

```rust
impl KnowledgeBase {
    /// Search for nodes by vector similarity.
    ///
    /// Uses cosine similarity metric on the specified embedding field's
    /// HNSW index. Supports optional property filters applied alongside
    /// the vector search.
    ///
    /// Target: < 20ms for top-10 (NF2).
    ///
    /// # Arguments
    /// * `embedding` - Query vector
    /// * `node_type` - Which node type to search (e.g., "Fact", "Message")
    /// * `field` - Which embedding field (e.g., "embedding", "text_embedding")
    /// * `top_k` - Maximum number of results
    /// * `filter` - Optional property filter
    pub fn vector_search(
        &self,
        embedding: &[f32],
        node_type: &str,
        field: &str,
        top_k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>>;

    /// Search across multiple node types simultaneously.
    ///
    /// Performs vector search on each specified (node_type, field) pair
    /// and merges results by score. Useful for the recall cascade where
    /// Phase 1 searches Fact, Procedure, and Topic embeddings together.
    ///
    /// # Arguments
    /// * `embedding` - Query vector
    /// * `targets` - List of (node_type, field) pairs to search
    /// * `top_k` - Maximum total results across all types
    /// * `filter` - Optional per-target filters
    pub fn multi_type_vector_search(
        &self,
        embedding: &[f32],
        targets: &[(&str, &str)],
        top_k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>>;
}
```

**Implementation notes:**
- Cosine similarity is the distance metric for all vector indexes
- Filter predicates are pushed down to uni-db to avoid post-filtering
- Results are sorted by descending similarity score
- Empty embedding or zero-vector returns empty results
- Dimension mismatch between query and index returns `UnikoError::Search`

---

### 3.3 -- Fulltext Search

**Objective:** Provide BM25-ranked fulltext search on all Fulltext-indexed fields.

#### File to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/search/fulltext.rs` | Rust | Fulltext search implementation |

#### Functions

```rust
impl KnowledgeBase {
    /// Search for nodes using fulltext (BM25) ranking.
    ///
    /// Searches the Fulltext index on the specified field.
    /// Supports optional property filters.
    ///
    /// Target: contributes to NF3 (< 50ms hybrid total).
    ///
    /// # Arguments
    /// * `query` - Search query string (supports BM25 syntax)
    /// * `node_type` - Which node type to search
    /// * `field` - Which Fulltext-indexed field
    /// * `top_k` - Maximum results
    /// * `filter` - Optional property filter
    pub fn fulltext_search(
        &self,
        query: &str,
        node_type: &str,
        field: &str,
        top_k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>>;

    /// Search across multiple fulltext fields simultaneously.
    ///
    /// Useful for Phase 3 of recall cascade: searching Chunk.text
    /// and Message.content together.
    pub fn multi_field_fulltext_search(
        &self,
        query: &str,
        targets: &[(&str, &str)],  // (node_type, field) pairs
        top_k: usize,
        filter: Option<Filter>,
    ) -> Result<Vec<SearchResult>>;
}
```

**Implementation notes:**
- BM25 ranking provided by uni-db's fulltext index engine
- Empty query returns empty results
- Query terms are tokenized and matched against the inverted index
- Scores are BM25 relevance scores (not normalized to [0,1] at this layer -- normalization happens in hybrid search)

---

### 3.4 -- Hybrid Search & RRF

**Objective:** Combine vector search and fulltext search results using Reciprocal Rank Fusion (RRF) with tier-weighted scoring.

#### File to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/search/hybrid.rs` | Rust | Hybrid search with RRF fusion |

#### Functions

```rust
impl KnowledgeBase {
    /// Hybrid search combining vector similarity and fulltext BM25.
    ///
    /// 1. Runs vector_search on the embedding field
    /// 2. Runs fulltext_search on the text field
    /// 3. Normalizes scores per-source to [0,1] (min-max normalization)
    /// 4. Fuses results via RRF: score(item) = sum(1/(k + rank_i)) with k=60
    /// 5. Applies tier weights to final scores
    ///
    /// Target: < 50ms total (NF3).
    ///
    /// # Arguments
    /// * `query` - Text query (used for fulltext and embedding computation)
    /// * `embedding` - Pre-computed query embedding vector
    /// * `node_type` - Which node type to search
    /// * `top_k` - Maximum results after fusion
    pub fn hybrid_search(
        &self,
        query: &str,
        embedding: &[f32],
        node_type: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>>;

    /// Hybrid search with explicit tier weighting.
    ///
    /// Tier weights determine how results from different node types
    /// are weighted in the final ranking:
    ///   Semantic (Facts) = 1.0
    ///   Procedural (Procedures) = 0.9
    ///   Episodic (Episodes, Observations) = 0.7
    ///   KnowledgeBase (Chunks, Artifacts) = 0.5
    ///   Provenance (Actions, Messages) = 0.4
    pub fn hybrid_search_weighted(
        &self,
        query: &str,
        embedding: &[f32],
        targets: &[SearchTarget],
        top_k: usize,
    ) -> Result<Vec<SearchResult>>;
}

/// A search target with tier weight.
pub struct SearchTarget {
    pub node_type: String,
    pub embedding_field: String,
    pub fulltext_field: Option<String>,
    pub tier_weight: f64,
}

/// Default tier weights from the spec.
pub const TIER_WEIGHT_SEMANTIC: f64 = 1.0;
pub const TIER_WEIGHT_PROCEDURAL: f64 = 0.9;
pub const TIER_WEIGHT_EPISODIC: f64 = 0.7;
pub const TIER_WEIGHT_KB: f64 = 0.5;
pub const TIER_WEIGHT_PROVENANCE: f64 = 0.4;

/// RRF constant (standard value from literature).
pub const RRF_K: f64 = 60.0;
```

#### RRF Algorithm Detail

```
For each search source S_i:
  1. Execute search, get ranked results R_i
  2. Normalize scores in R_i to [0,1] via min-max: (score - min) / (max - min)

For each unique item across all R_i:
  rrf_score = sum over all sources i where item appears: 1 / (RRF_K + rank_in_source_i)

Final score = rrf_score * tier_weight_for_node_type

Sort by final score descending, take top_k.
```

---

### 3.5 -- Graph Traversal

**Objective:** Provide graph traversal primitives including bidirectional traversal, shortest path, and Personalized PageRank (PPR).

#### File to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/search/traversal.rs` | Rust | Graph traversal operations |

#### Functions

```rust
impl KnowledgeBase {
    /// Traverse the graph from a starting node.
    ///
    /// Follows edges up to the specified depth, optionally filtering by
    /// edge type and direction. Returns all visited nodes with their
    /// depth and path from the start node.
    ///
    /// Target: < 5ms for 3-hop (NF4).
    ///
    /// # Arguments
    /// * `start` - Starting node
    /// * `depth` - Maximum traversal depth (hops)
    /// * `direction` - Edge direction to follow
    /// * `edge_types` - Optional filter: only follow these edge types
    pub fn traverse(
        &self,
        start: NodeId,
        depth: usize,
        direction: Direction,
        edge_types: Option<&[&str]>,
    ) -> Result<Vec<TraversalResult>>;

    /// Find the shortest path between two nodes.
    ///
    /// Uses BFS. Returns None if no path exists.
    /// Optionally restricted to specific edge types.
    pub fn shortest_path(
        &self,
        from: NodeId,
        to: NodeId,
        edge_types: Option<&[&str]>,
    ) -> Result<Option<Path>>;

    /// Personalized PageRank (PPR) from seed nodes.
    ///
    /// Spreads activation from seed nodes across the graph, returning
    /// the top-k nodes by PageRank score. Used in Phase 3 of the recall
    /// cascade (HippoRAG-inspired) to discover multi-hop connections
    /// from query entities without expensive community summarization.
    ///
    /// Algorithm:
    ///   For each iteration:
    ///     For each node n:
    ///       score[n] = (1 - damping) * teleport[n]
    ///                + damping * sum(score[neighbor] / out_degree[neighbor])
    ///   Teleport probabilities: uniform over seed nodes, zero for non-seeds.
    ///   Converge when max score change < epsilon (1e-6) or max_iter reached.
    ///
    /// # Arguments
    /// * `seeds` - Starting nodes (uniform teleport probability)
    /// * `damping` - Damping factor (default 0.85)
    /// * `max_iter` - Maximum iterations (default 20)
    /// * `top_k` - Number of top-scoring nodes to return
    pub fn personalized_pagerank(
        &self,
        seeds: &[NodeId],
        damping: f64,
        max_iter: usize,
        top_k: usize,
    ) -> Result<Vec<(NodeId, f64)>>;
}

/// Result of a graph traversal step.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub node_id: NodeId,
    pub node_type: String,
    pub depth: usize,
    pub path: Vec<PathStep>,       // how we got here
    pub properties: HashMap<String, Value>,
}

/// A step in a traversal path.
#[derive(Debug, Clone)]
pub struct PathStep {
    pub edge_id: EdgeId,
    pub edge_type: String,
    pub from: NodeId,
    pub to: NodeId,
}

/// A complete path between two nodes.
#[derive(Debug, Clone)]
pub struct Path {
    pub steps: Vec<PathStep>,
    pub length: usize,
}
```

**Implementation notes:**
- BFS for traverse and shortest_path (guarantees shortest path)
- Visited set to avoid cycles in traversal
- PPR uses power iteration method
- PPR convergence: stop when max absolute score change < 1e-6 or max_iter reached
- PPR defaults: damping=0.85, max_iter=20 (from spec)
- Edge type filtering reduces the effective graph for traversal

---

### 3.6 -- Locy Runtime Integration

**Objective:** Provide Rust wrappers for uni-db's Locy logic programming features: rule creation, execution, ASSUME (hypothetical reasoning), ABDUCE (abductive reasoning), and rule explanation.

#### File to Create

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-store/src/locy/mod.rs` | Rust | Locy runtime wrapper |

#### Functions

```rust
impl KnowledgeBase {
    /// Create a Locy rule in the database.
    ///
    /// The rule is stored as executable Locy source code.
    /// Validates syntax before persisting.
    ///
    /// # Arguments
    /// * `name` - Rule name (must be unique)
    /// * `source` - Locy source code (CREATE RULE ... AS ...)
    pub fn create_rule(&self, name: &str, source: &str) -> Result<()>;

    /// Execute a named Locy rule with parameter injection.
    ///
    /// Parameters like $agent_id, $threshold, etc. are injected
    /// into the rule before execution.
    ///
    /// # Arguments
    /// * `name` - Rule name
    /// * `params` - Parameters to inject (e.g., {"agent_id": "agent-1"})
    pub fn execute_rule(
        &self,
        name: &str,
        params: &HashMap<String, Value>,
    ) -> Result<Vec<Record>>;

    /// Begin building an ASSUME (hypothetical reasoning) query.
    ///
    /// ASSUME forks the graph state, applies mutations, executes
    /// a query against the hypothetical state, then rolls back.
    /// Nothing is persisted.
    ///
    /// Target: < 200ms (NF9).
    ///
    /// # Example
    /// ```rust
    /// let results = kb.assume()
    ///     .add_fact("server", "port", "9090")
    ///     .then_query("MATCH (f:Fact) WHERE f.subject = 'server' RETURN f")
    ///     .run()?;
    /// ```
    pub fn assume(&self) -> AssumeBuilder;

    /// Find the minimal set of facts that support a conclusion.
    ///
    /// ABDUCE performs backward inference: given a conclusion,
    /// it searches for the smallest set of existing facts that
    /// would make the conclusion true.
    ///
    /// # Arguments
    /// * `conclusion` - The conclusion to explain (Cypher pattern or predicate)
    pub fn abduce(&self, conclusion: &str) -> Result<Vec<AbductionResult>>;

    /// Explain how a rule derived its results.
    ///
    /// Returns a derivation tree showing which graph patterns
    /// matched and which data contributed to each result.
    ///
    /// # Arguments
    /// * `name` - Rule name to explain
    pub fn explain_rule(&self, name: &str) -> Result<DerivationTree>;

    /// List all rules in the database.
    pub fn list_rules(&self) -> Result<Vec<RuleInfo>>;

    /// Delete a rule by name.
    pub fn delete_rule(&self, name: &str) -> Result<()>;
}
```

#### AssumeBuilder

```rust
/// Builder for hypothetical reasoning queries.
///
/// Creates a temporary fork of the graph state, applies mutations,
/// executes queries, and rolls back. The original graph is never modified.
pub struct AssumeBuilder<'a> {
    kb: &'a KnowledgeBase,
    mutations: Vec<Mutation>,
    query: Option<String>,
}

impl<'a> AssumeBuilder<'a> {
    /// Add a hypothetical fact to the assumed state.
    pub fn add_fact(self, subject: &str, predicate: &str, object: &str) -> Self;

    /// Remove a fact from the assumed state.
    pub fn remove_fact(self, fact_id: &str) -> Self;

    /// Add a hypothetical node.
    pub fn add_node(self, node_type: &str, properties: HashMap<String, Value>) -> Self;

    /// Add a hypothetical edge.
    pub fn add_edge(self, edge_type: &str, from: NodeId, to: NodeId) -> Self;

    /// Set the query to execute against the hypothetical state.
    /// Can be Cypher or a Locy rule name.
    pub fn then_query(self, query: &str) -> Self;

    /// Execute a Locy rule against the hypothetical state.
    pub fn then_rule(self, rule_name: &str, params: &HashMap<String, Value>) -> Self;

    /// Run the ASSUME: fork, mutate, query, rollback, return results.
    pub fn run(self) -> Result<Vec<Record>>;
}

/// Internal mutation for ASSUME.
enum Mutation {
    AddFact { subject: String, predicate: String, object: String },
    RemoveFact { fact_id: String },
    AddNode { node_type: String, properties: HashMap<String, Value> },
    AddEdge { edge_type: String, from: NodeId, to: NodeId },
}
```

#### Supporting Types

```rust
/// A result from abductive reasoning.
#[derive(Debug)]
pub struct AbductionResult {
    pub supporting_facts: Vec<(NodeId, HashMap<String, Value>)>,
    pub confidence: f64,
    pub explanation: String,
}

/// A derivation tree explaining how a rule produced its results.
#[derive(Debug)]
pub struct DerivationTree {
    pub rule_name: String,
    pub root: DerivationNode,
}

/// A node in a derivation tree.
#[derive(Debug)]
pub struct DerivationNode {
    pub pattern: String,                // the MATCH pattern that was satisfied
    pub bindings: HashMap<String, Value>, // variable bindings
    pub children: Vec<DerivationNode>,  // sub-patterns
}

/// Information about a registered rule.
#[derive(Debug)]
pub struct RuleInfo {
    pub name: String,
    pub source: String,
    pub created_at: Option<Timestamp>,
}

/// A record returned from rule execution (row of results).
pub type Record = HashMap<String, Value>;
```

---

## Test Plan

### 3.1 -- Node & Edge CRUD Tests

| Test | File | What It Validates |
|---|---|---|
| `test_create_and_get_node` | `tests/storage_tests.rs` | Create a Message node, get by NodeId, verify properties |
| `test_get_node_by_ext_id` | `tests/storage_tests.rs` | Lookup by message_id via Hash index |
| `test_update_node` | `tests/storage_tests.rs` | Update content, verify only changed properties affected |
| `test_delete_node` | `tests/storage_tests.rs` | Delete node, verify gone, verify edges removed |
| `test_merge_node_insert` | `tests/storage_tests.rs` | Merge with new ext_id creates node |
| `test_merge_node_update` | `tests/storage_tests.rs` | Merge with existing ext_id updates properties |
| `test_merge_idempotent` | `tests/storage_tests.rs` | Merge same data twice produces one node |
| `test_create_edge` | `tests/storage_tests.rs` | Create SENT_BY edge with role property |
| `test_get_edges_outgoing` | `tests/storage_tests.rs` | Get all outgoing edges from a node |
| `test_get_edges_incoming` | `tests/storage_tests.rs` | Get all incoming edges to a node |
| `test_delete_edge` | `tests/storage_tests.rs` | Delete edge, verify removal |
| `test_delete_edges_between` | `tests/storage_tests.rs` | Delete all NEXT edges between two messages |
| `test_batch_create_nodes` | `tests/storage_tests.rs` | Batch create 50 Chunk nodes in one transaction |
| `test_batch_create_edges` | `tests/storage_tests.rs` | Batch create HAS_CHUNK edges |
| `test_transaction_commit` | `tests/storage_tests.rs` | Transaction commits on success |
| `test_transaction_rollback` | `tests/storage_tests.rs` | Transaction rolls back on error |
| `test_query_nodes_with_filter` | `tests/storage_tests.rs` | Query Messages by content_type filter |
| `test_create_node_latency` | `tests/benchmarks.rs` | Create Message + 3 edges < 10ms (NF1) |

### 3.2 -- Vector Search Tests

| Test | File | What It Validates |
|---|---|---|
| `test_vector_search_basic` | `tests/search_tests.rs` | Insert 10 nodes with embeddings, search returns nearest |
| `test_vector_search_top_k` | `tests/search_tests.rs` | top_k=5 returns exactly 5 results |
| `test_vector_search_with_filter` | `tests/search_tests.rs` | Filter by status="active" narrows results |
| `test_vector_search_empty_results` | `tests/search_tests.rs` | No matching nodes returns empty vec |
| `test_vector_search_dimension_mismatch` | `tests/search_tests.rs` | Wrong dimension returns Search error |
| `test_multi_type_vector_search` | `tests/search_tests.rs` | Search across Fact + Procedure embeddings |
| `test_vector_search_cosine_ordering` | `tests/search_tests.rs` | Most similar embedding ranks first |
| `test_vector_search_latency` | `tests/benchmarks.rs` | Top-10 search < 20ms (NF2) |

### 3.3 -- Fulltext Search Tests

| Test | File | What It Validates |
|---|---|---|
| `test_fulltext_search_basic` | `tests/search_tests.rs` | Search "adoption" in Message.content finds matching messages |
| `test_fulltext_search_ranking` | `tests/search_tests.rs` | More relevant document ranks higher (BM25) |
| `test_fulltext_search_with_filter` | `tests/search_tests.rs` | Filter by session narrows results |
| `test_fulltext_search_no_results` | `tests/search_tests.rs` | Nonsense query returns empty |
| `test_multi_field_fulltext` | `tests/search_tests.rs` | Search across Chunk.text + Message.content |

### 3.4 -- Hybrid Search & RRF Tests

| Test | File | What It Validates |
|---|---|---|
| `test_hybrid_search_basic` | `tests/search_tests.rs` | Combines vector and fulltext results |
| `test_rrf_score_calculation` | `tests/search_tests.rs` | RRF scores computed correctly: 1/(k+rank) |
| `test_rrf_fusion_ordering` | `tests/search_tests.rs` | Item appearing in both sources ranks higher than single-source |
| `test_rrf_normalization` | `tests/search_tests.rs` | Per-source min-max normalization to [0,1] |
| `test_tier_weighted_scoring` | `tests/search_tests.rs` | Fact (weight 1.0) outranks Message (weight 0.4) at same RRF score |
| `test_hybrid_improves_over_vector_only` | `tests/search_tests.rs` | Hybrid recall >= vector-only recall on test set |
| `test_hybrid_improves_over_fulltext_only` | `tests/search_tests.rs` | Hybrid recall >= fulltext-only recall on test set |
| `test_hybrid_search_latency` | `tests/benchmarks.rs` | < 50ms total (NF3) |

### 3.5 -- Graph Traversal Tests

| Test | File | What It Validates |
|---|---|---|
| `test_traverse_depth_1` | `tests/traversal_tests.rs` | Returns immediate neighbors |
| `test_traverse_depth_3` | `tests/traversal_tests.rs` | Returns 3-hop neighborhood |
| `test_traverse_with_edge_filter` | `tests/traversal_tests.rs` | Only follows specified edge types |
| `test_traverse_direction` | `tests/traversal_tests.rs` | Outgoing vs Incoming vs Both |
| `test_traverse_cycle_handling` | `tests/traversal_tests.rs` | Cycles do not cause infinite loop |
| `test_shortest_path_exists` | `tests/traversal_tests.rs` | Finds shortest path between connected nodes |
| `test_shortest_path_not_exists` | `tests/traversal_tests.rs` | Returns None for disconnected nodes |
| `test_shortest_path_direct` | `tests/traversal_tests.rs` | Direct edge returns path of length 1 |
| `test_ppr_seed_nodes_rank_high` | `tests/traversal_tests.rs` | Seed nodes have high PPR scores |
| `test_ppr_connected_nodes_rank_high` | `tests/traversal_tests.rs` | Nodes well-connected to seeds rank higher |
| `test_ppr_convergence` | `tests/traversal_tests.rs` | PPR converges within max_iter |
| `test_ppr_damping_effect` | `tests/traversal_tests.rs` | Higher damping = more spread; lower = more local |
| `test_traverse_latency` | `tests/benchmarks.rs` | 3-hop traversal < 5ms (NF4) |

### 3.6 -- Locy Runtime Tests

| Test | File | What It Validates |
|---|---|---|
| `test_create_rule` | `tests/locy_tests.rs` | Rule creation succeeds with valid Locy source |
| `test_create_rule_invalid_syntax` | `tests/locy_tests.rs` | Invalid Locy source returns Locy error |
| `test_execute_rule_basic` | `tests/locy_tests.rs` | Simple rule returns expected results |
| `test_execute_rule_with_params` | `tests/locy_tests.rs` | $agent_id parameter injected correctly |
| `test_execute_rule_no_matches` | `tests/locy_tests.rs` | Rule with no matching data returns empty |
| `test_assume_add_fact` | `tests/locy_tests.rs` | ASSUME adds hypothetical fact, query finds it |
| `test_assume_rollback` | `tests/locy_tests.rs` | After ASSUME, original graph unchanged |
| `test_assume_remove_fact` | `tests/locy_tests.rs` | ASSUME removes existing fact, query confirms absence |
| `test_assume_with_rule` | `tests/locy_tests.rs` | ASSUME + Locy rule execution in hypothetical state |
| `test_assume_nested` | `tests/locy_tests.rs` | Nested ASSUME (for MCTS) works correctly |
| `test_abduce_basic` | `tests/locy_tests.rs` | Find supporting facts for a conclusion |
| `test_abduce_minimal` | `tests/locy_tests.rs` | Returns minimal (not all) supporting facts |
| `test_abduce_no_support` | `tests/locy_tests.rs` | Returns empty when conclusion has no support |
| `test_explain_rule` | `tests/locy_tests.rs` | Derivation tree shows matched patterns |
| `test_list_rules` | `tests/locy_tests.rs` | Lists all registered rules |
| `test_delete_rule` | `tests/locy_tests.rs` | Deletes rule, confirm not executable |
| `test_assume_latency` | `tests/benchmarks.rs` | Single ASSUME < 200ms (NF9) |

### Latency Benchmarks

All latency benchmarks should be run with `--release` profile on warm in-memory stores:

| Benchmark | Target | Measurement Method |
|---|---|---|
| NF1: Store message | < 10ms | Create Message node + SENT_BY + IN_SESSION + NEXT edges |
| NF2: Vector search top-10 | < 20ms | Search with 1K nodes indexed |
| NF3: Hybrid search | < 50ms | Vector + fulltext + fusion |
| NF4: 3-hop traversal | < 5ms | Traverse from node through 3 edge hops |
| NF9: ASSUME | < 200ms | Fork, add fact, query, rollback |

**Benchmark execution:**
```bash
cargo bench --release -p uniko-store
```

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| KnowledgeBase struct doc | `storage/mod.rs` | Purpose, creation methods, thread safety |
| Node CRUD docs | `storage/nodes.rs` | MERGE semantics explanation, ext_id convention, transaction behavior |
| Edge CRUD docs | `storage/edges.rs` | Direction enum, EdgeRecord structure, batch operations |
| Vector search docs | `search/vector.rs` | Cosine similarity, filter predicates, multi-type search |
| Fulltext search docs | `search/fulltext.rs` | BM25 ranking, supported query syntax |
| Hybrid search docs | `search/hybrid.rs` | RRF algorithm explanation, tier weights rationale, normalization |
| Traversal docs | `search/traversal.rs` | BFS algorithm, PPR convergence criteria, use in recall cascade |
| Locy docs | `locy/mod.rs` | Rule syntax overview, ASSUME semantics, ABDUCE semantics, parameter injection |
| Performance notes | Module-level doc comments | NF targets referenced at each function |

---

## Review Checklist

### Storage (3.1)
- [ ] `KnowledgeBase` struct wraps uni-db `Database` and `UnikoConfig`
- [ ] `new_in_memory()` and `new_persistent()` constructors work
- [ ] `create_node()` uses MERGE semantics via ext_id
- [ ] `get_node()` retrieves all properties
- [ ] `get_node_by_ext_id()` uses Hash index for O(1) lookup
- [ ] `update_node()` only changes specified properties
- [ ] `delete_node()` removes node and all edges
- [ ] `merge_node()` is idempotent (same data twice = one node)
- [ ] `create_edge()` supports optional properties (role, gap_ms, etc.)
- [ ] `get_edges()` supports Direction (Outgoing, Incoming, Both)
- [ ] `delete_edge()` works by EdgeId
- [ ] `batch_create_nodes()` creates all nodes in one transaction
- [ ] `batch_create_edges()` creates all edges in one transaction
- [ ] `transaction()` commits on Ok, rolls back on Err

### Vector Search (3.2)
- [ ] Cosine similarity metric used
- [ ] Filter predicates pushed to uni-db (not post-filter)
- [ ] `multi_type_vector_search()` merges results across node types
- [ ] Dimension mismatch returns clear error
- [ ] Top-10 search < 20ms on 1K nodes (NF2)

### Fulltext Search (3.3)
- [ ] BM25 ranking used
- [ ] Filter predicates supported
- [ ] `multi_field_fulltext_search()` works across fields
- [ ] Empty query returns empty results

### Hybrid Search (3.4)
- [ ] Per-source min-max normalization to [0,1]
- [ ] RRF formula: `1/(k + rank_i)` with k=60
- [ ] Tier weights match spec: Semantic=1.0, Procedural=0.9, Episodic=0.7, KB=0.5, Provenance=0.4
- [ ] Hybrid search < 50ms (NF3)
- [ ] Hybrid recall >= individual method recall

### Graph Traversal (3.5)
- [ ] `traverse()` respects depth limit
- [ ] `traverse()` handles cycles without infinite loop
- [ ] `traverse()` supports edge type filtering
- [ ] `shortest_path()` returns shortest path (BFS guarantee)
- [ ] `shortest_path()` returns None for disconnected nodes
- [ ] PPR with damping=0.85, max_iter=20 converges
- [ ] PPR seed nodes rank high in results
- [ ] 3-hop traversal < 5ms (NF4)

### Locy Runtime (3.6)
- [ ] `create_rule()` validates syntax before persisting
- [ ] `execute_rule()` injects parameters ($agent_id, $threshold, etc.)
- [ ] `assume()` builder: fork, mutate, query, rollback
- [ ] ASSUME does not modify original graph state
- [ ] Nested ASSUME works (for MCTS)
- [ ] `abduce()` returns minimal supporting facts
- [ ] `explain_rule()` produces derivation tree
- [ ] ASSUME < 200ms (NF9)

---

## Definition of Done

1. **KnowledgeBase functional:** All CRUD operations work correctly with MERGE semantics, transactions, and batch operations.
2. **Vector search operational:** Cosine similarity search returns correct top-k results with filter support. Multi-type search merges results correctly.
3. **Fulltext search operational:** BM25-ranked search returns relevant results. Multi-field search works.
4. **Hybrid search operational:** RRF fusion improves over individual methods. Per-source normalization and tier weighting produce correct final scores.
5. **Graph traversal operational:** BFS traversal with depth limit and edge filtering. Shortest path correctly found. PPR converges and produces meaningful scores.
6. **Locy runtime operational:** Rules can be created, executed with parameters, explained. ASSUME creates hypothetical state without modifying original. ABDUCE finds minimal supporting facts.
7. **Latency targets met:** NF1 < 10ms, NF2 < 20ms, NF3 < 50ms, NF4 < 5ms, NF9 < 200ms (all on warm in-memory stores with `--release` profile).
8. **All tests pass:** Unit tests, integration tests, and property-based tests all green with `cargo nextest run -n auto`.
9. **Benchmarks documented:** Latency benchmark results recorded and compared against NF targets.
10. **API documented:** Every public function has doc comments with purpose, arguments, return value, error conditions, and NF target reference where applicable.
