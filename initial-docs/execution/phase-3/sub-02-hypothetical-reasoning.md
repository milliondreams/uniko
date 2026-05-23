# Phase 14: Hypothetical Reasoning -- ASSUME/ABDUCE & NL-to-Cypher

## Context

This phase exposes three advanced reasoning capabilities as agent-facing APIs:

**ASSUME builder** enables hypothetical reasoning -- "what if?" queries. The agent forks the graph state, applies temporary mutations (add facts, entities, or edges), executes a query against the mutated state, collects results, and rolls back. The underlying state is never corrupted. This is unique to uniko -- no competitor offers database-internal hypothetical reasoning. ASSUME is powered by uni-db's Locy runtime, which natively supports `ASSUME { mutations } THEN { query }` semantics with automatic rollback.

**ABDUCE builder** enables abductive reasoning -- "why might this be true?" queries. Given a desired conclusion (subject, predicate, object), the system searches the existing graph for the minimal set of facts that would support or explain that conclusion. This is the inverse of deduction: instead of deriving conclusions from facts, we find facts that would justify a conclusion.

**NL-to-Cypher** enables natural language graph queries. Agents describe what they want in plain English, and the system translates to a valid Cypher query against the uniko schema. An LRU cache avoids repeated LLM calls for identical questions. Mutation-blocking ensures generated queries are read-only. This implements F61 (DIF).

**Contrastive retrieval** adds a mode to the recall cascade (Phase 10) where agents can learn from both successes and failures. When enabled, Phase 2 of the recall cascade retrieves failure-outcome episodes alongside success-outcome episodes for the same entities and action types. This implements F56 (DIF).

Together, these implement requirements F46, F47, F56, F61 and fulfill latency targets NF9 (ASSUME < 200ms) and NF13 (NL-to-Cypher 200-500ms).

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 12 (MVP complete) | Complete | All MVP pipelines, recall cascade, KnowledgeBase, agent tools |
| KnowledgeBase Locy runtime (Phase 3) | Operational | `ASSUME { } THEN { }` and `ABDUCE` execution via uni-db |
| Recall Cascade (Phase 10) | Operational | 3-phase recall with RecallContextBuilder, ContextBundle assembly |
| LLM provider | Required for NL-to-Cypher | Schema-aware prompt for Cypher translation |
| Schema registration (Phase 2) | Complete | SchemaInfo auto-generation from registered node/edge types |
| Fact nodes (Phase 9) | Available | Facts for ASSUME mutations and ABDUCE searches |
| Entity nodes (Phase 7) | Available | Entities for ASSUME creation and ABDUCE supporting evidence |
| Episode nodes (Phase 8) | Available | Episodes with outcome field for contrastive retrieval |

## Sub-phases

---

### 14.1 -- ASSUME Builder API

**Objective:** Provide a fluent builder API for hypothetical reasoning. Agents queue mutations (facts, entities, edges), specify a query, and execute -- all within a forked graph state that rolls back automatically.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/reasoning/mod.rs` | New module root | Re-exports for reasoning module |
| `crates/uniko-cortex/src/reasoning/assume.rs` | New | `AssumeBuilder`, `AssumeResult`, fork/mutate/query/rollback logic |

#### Structs and Functions

```rust
/// Builder for hypothetical reasoning queries.
///
/// Usage:
/// ```rust
/// let result = cortex.assume()
///     .assume_fact("auth_module", "uses", "oauth2")
///     .assume_entity("oauth2_provider", "service")
///     .assume_edge("auth_module", "DEPENDS_ON", "oauth2_provider")
///     .then_query("MATCH (e:Entity)<-[:DEPENDS_ON]-(m:Entity {name: 'auth_module'}) RETURN e.name")
///     .run()
///     .await?;
/// ```
pub struct AssumeBuilder<'a> {
    kb: &'a KnowledgeBase,
    mutations: Vec<AssumeMutation>,
    query: Option<AssumeQuery>,
}

/// A single mutation to apply in the forked state.
pub enum AssumeMutation {
    /// Add or modify a fact: (subject, predicate, value).
    Fact {
        subject: String,
        predicate: String,
        value: String,
    },
    /// Create a temporary entity.
    Entity {
        name: String,
        entity_type: String,
    },
    /// Create a temporary edge between two nodes.
    Edge {
        from: String,
        edge_type: String,
        to: String,
    },
}

/// The query to execute against the forked state.
pub enum AssumeQuery {
    /// A Locy rule to execute.
    Locy(String),
    /// A raw Cypher query to execute.
    Cypher(String),
}

/// Result of an ASSUME operation.
pub struct AssumeResult {
    /// Query results from the forked state.
    pub results: Vec<serde_json::Value>,
    /// Number of mutations that were applied.
    pub mutations_applied: usize,
    /// Time spent executing the query (ms).
    pub query_time_ms: u64,
}
```

#### Builder Methods

- `.assume_fact(subject: &str, predicate: &str, value: &str) -> &mut Self` -- Queue a fact mutation. The fact will be created (or modified if it already exists with the same subject+predicate) in the forked state.
- `.assume_entity(name: &str, entity_type: &str) -> &mut Self` -- Queue entity creation. The entity will exist only in the forked state.
- `.assume_edge(from: &str, edge_type: &str, to: &str) -> &mut Self` -- Queue edge creation. `from` and `to` are entity names (resolved to node IDs).
- `.then_query(locy_rule: &str) -> &mut Self` -- Set the query to run as a Locy rule execution. Mutually exclusive with `.then_cypher()`.
- `.then_cypher(cypher: &str) -> &mut Self` -- Set the query to run as raw Cypher. Mutually exclusive with `.then_query()`.
- `.run() -> Result<AssumeResult>` -- Execute the full ASSUME operation.

#### Execution Flow

`AssumeBuilder::run()` performs the following steps:

1. **Validate:** Ensure at least one mutation and exactly one query are specified.
2. **Build Locy ASSUME statement:** Translate the queued mutations and query into a single `ASSUME { mutations } THEN { query }` Locy statement:
   ```cypher
   ASSUME {
       CREATE (f:Fact {subject: 'auth_module', predicate: 'uses', object: 'oauth2'})
       CREATE (e:Entity {name: 'oauth2_provider', entity_type: 'service'})
       CREATE (auth)-[:DEPENDS_ON]->(oauth2)
   } THEN {
       MATCH (e:Entity)<-[:DEPENDS_ON]-(m:Entity {name: 'auth_module'})
       RETURN e.name
   }
   ```
3. **Execute** via KnowledgeBase Locy runtime. The runtime handles fork/rollback internally.
4. **Collect results** into `Vec<serde_json::Value>`.
5. **Measure** query_time_ms.
6. **Return** `AssumeResult`.

#### Error Handling

| Error Case | Behavior |
|---|---|
| No mutations specified | Return `Err(UnikoError::Pipeline("ASSUME requires at least one mutation"))` |
| No query specified | Return `Err(UnikoError::Pipeline("ASSUME requires a query"))` |
| Both `.then_query()` and `.then_cypher()` called | Return `Err(UnikoError::Pipeline("ASSUME cannot have both Locy and Cypher queries"))` |
| Entity name in `.assume_edge()` not found (and not being created) | Return `Err(UnikoError::Pipeline("Entity not found: {name}"))` |
| Locy runtime error | Propagate as `UnikoError::Locy(...)` |
| Timeout (> 200ms per NF9) | Return `Err(UnikoError::Timeout(200))` |

#### Chaining

Multiple mutations can be chained before the query:

```rust
cortex.assume()
    .assume_fact("server", "port", "9090")
    .assume_fact("server", "protocol", "https")
    .assume_entity("load_balancer", "service")
    .assume_edge("server", "BEHIND", "load_balancer")
    .then_query("MATCH (s:Entity {name: 'server'})-[:BEHIND]->(lb) RETURN lb.name")
    .run()
    .await?;
```

All mutations are applied atomically in the forked state before the query executes.

#### Latency Target

NF9: Single ASSUME < 200ms. Measured from `run()` call to return. The Locy runtime's fork/rollback is designed to be lightweight (copy-on-write snapshots in uni-db).

---

### 14.2 -- ABDUCE Builder API

**Objective:** Provide a builder API for abductive reasoning. Given a desired conclusion, find the minimal set of existing facts that would support it.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/reasoning/abduce.rs` | New | `AbduceBuilder`, `AbduceResult`, backward inference logic |

#### Structs and Functions

```rust
/// Builder for abductive reasoning queries.
///
/// Usage:
/// ```rust
/// let result = cortex.abduce()
///     .conclusion("auth_module", "is_secure", "true")
///     .max_depth(3)
///     .run()
///     .await?;
/// ```
pub struct AbduceBuilder<'a> {
    kb: &'a KnowledgeBase,
    conclusion_subject: Option<String>,
    conclusion_predicate: Option<String>,
    conclusion_object: Option<String>,
    max_depth: usize,
}

/// A single step in a derivation chain.
pub struct DerivationStep {
    /// The fact used at this step.
    pub fact: FactSummary,
    /// The rule that connected this fact to the next step (if any).
    pub rule: Option<String>,
    /// Depth level in the derivation tree (0 = closest to conclusion).
    pub depth: usize,
}

/// Summary of a fact for derivation chains.
pub struct FactSummary {
    pub fact_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f64,
}

/// Result of an ABDUCE operation.
pub struct AbduceResult {
    /// Facts that support the conclusion, ordered by relevance.
    pub supporting_facts: Vec<FactSummary>,
    /// Step-by-step derivation from supporting facts to conclusion.
    pub derivation_chain: Vec<DerivationStep>,
    /// Overall confidence in the abductive explanation (product of fact confidences).
    pub confidence: f64,
}
```

#### Builder Methods

- `.conclusion(subject: &str, predicate: &str, object: &str) -> &mut Self` -- Set the desired conclusion to explain.
- `.max_depth(depth: usize) -> &mut Self` -- Limit backward search depth (default: 3). Deeper searches explore more transitive connections but take longer.
- `.run() -> Result<AbduceResult>` -- Execute the ABDUCE operation.

#### Execution Flow

`AbduceBuilder::run()` performs:

1. **Validate:** Ensure conclusion is fully specified (subject + predicate + object).
2. **Build Locy ABDUCE statement:**
   ```cypher
   ABDUCE (f:Fact {subject: 'auth_module', predicate: 'is_secure', object: 'true'})
   MAX DEPTH 3
   ```
3. **Execute** via KnowledgeBase Locy runtime.
4. **Process results:** The Locy runtime returns supporting facts and derivation paths. Map to `DerivationStep` structs.
5. **Compute confidence:** Product of all supporting fact confidences, clamped to [0.0, 1.0].
6. **Return** `AbduceResult`.

#### Search Strategy

The ABDUCE operation searches backward from the conclusion:

1. **Direct support:** Find existing facts that directly match the conclusion's subject and predicate.
2. **Transitive support:** Follow rule derivation chains. If Rule R derives fact F from facts F1 and F2, and F matches the conclusion, then F1 and F2 are supporting facts at depth 1.
3. **Entity connections:** Follow ABOUT, MENTIONS, and DERIVED_FROM edges to find facts connected to the conclusion's entities.
4. **Depth limiting:** Stop at `max_depth` hops from the conclusion.

#### Error Handling

| Error Case | Behavior |
|---|---|
| Incomplete conclusion (missing subject/predicate/object) | Return `Err(UnikoError::Pipeline("ABDUCE requires subject, predicate, and object"))` |
| No supporting facts found | Return `Ok(AbduceResult { supporting_facts: vec![], derivation_chain: vec![], confidence: 0.0 })` |
| Locy runtime error | Propagate as `UnikoError::Locy(...)` |

---

### 14.3 -- NL-to-Cypher

**Objective:** Translate natural language questions about the knowledge graph into valid, read-only Cypher queries. Cache results, block mutations, and retry on parse failures.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/nl2cypher/mod.rs` | New module root | `nl_to_cypher()` function, `SchemaInfo`, cache |
| `crates/uniko-cortex/src/nl2cypher/schema.rs` | New | Schema introspection, `SchemaInfo` generation |
| `crates/uniko-cortex/src/nl2cypher/validator.rs` | New | Cypher mutation blocking, parse validation |

#### Structs and Functions

```rust
/// Translate a natural language query to Cypher.
///
/// # Example
/// ```rust
/// let cypher = nl_to_cypher(
///     "What entities does the auth module depend on?",
///     &schema_info,
///     &llm_provider,
/// ).await?;
/// // Returns: "MATCH (e:Entity)<-[:DEPENDS_ON]-(m:Entity {name: 'auth_module'}) RETURN e.name"
/// ```
pub async fn nl_to_cypher(
    query: &str,
    schema: &SchemaInfo,
    provider: &LlmProvider,
) -> Result<String>;

/// Schema metadata used for NL-to-Cypher prompting.
pub struct SchemaInfo {
    /// All registered node types with their properties.
    pub node_types: Vec<NodeTypeInfo>,
    /// All registered edge types with their properties.
    pub edge_types: Vec<EdgeTypeInfo>,
    /// All indexes (Hash, BTree, Fulltext, Vector).
    pub indexes: Vec<IndexInfo>,
}

pub struct NodeTypeInfo {
    pub name: String,
    pub properties: Vec<PropertyInfo>,
}

pub struct EdgeTypeInfo {
    pub name: String,
    pub source_type: String,
    pub target_type: String,
    pub properties: Vec<PropertyInfo>,
}

pub struct PropertyInfo {
    pub name: String,
    pub property_type: String,
    pub nullable: bool,
}

pub struct IndexInfo {
    pub node_type: String,
    pub property: String,
    pub index_type: String,  // "Hash", "BTree", "Fulltext", "Vector"
}
```

#### Schema Introspection

`SchemaInfo` is auto-generated from the registered schema:

```rust
/// Generate SchemaInfo from the database's registered schema.
pub fn introspect_schema(kb: &KnowledgeBase) -> Result<SchemaInfo>;
```

This function queries the KnowledgeBase for all registered node types, edge types, and indexes, producing a `SchemaInfo` that is included in the LLM prompt. The SchemaInfo is regenerated lazily (cached until schema changes).

#### LLM Prompt Template

```
You are a Cypher query generator for a knowledge graph with the following schema:

Node types:
{for each node_type: "- {name}: {property1} ({type}), {property2} ({type}), ..."}

Edge types:
{for each edge_type: "- ({source_type})-[:{name}]->({target_type}): {property1} ({type}), ..."}

Indexes:
{for each index: "- {node_type}.{property}: {index_type}"}

Rules:
1. Generate ONLY read queries. Never use CREATE, SET, DELETE, MERGE, or REMOVE.
2. Use MATCH and RETURN statements.
3. Use WHERE for filtering.
4. Use ORDER BY and LIMIT when appropriate.
5. Use the indexed properties for efficient queries.

User question: {query}

Respond with ONLY the Cypher query, no explanation.
```

#### LRU Cache

```rust
use lru::LruCache;
use std::sync::Mutex;

/// Cache for NL-to-Cypher translations.
struct NlCypherCache {
    cache: Mutex<LruCache<String, String>>,  // NL query -> Cypher
}

impl NlCypherCache {
    fn new() -> Self {
        Self {
            cache: Mutex::new(LruCache::new(
                std::num::NonZeroUsize::new(256).unwrap()
            )),
        }
    }
}
```

Cache key is the normalized natural language query (lowercased, trimmed). Cache hit returns immediately without LLM call.

#### Mutation Blocking

```rust
/// Validate that a Cypher query is read-only (no mutations).
pub fn is_read_only(cypher: &str) -> bool;

/// List of mutation keywords to block.
const MUTATION_KEYWORDS: &[&str] = &[
    "CREATE", "SET", "DELETE", "MERGE", "REMOVE",
    "DETACH DELETE", "FOREACH",
];
```

Parse the generated Cypher and check for mutation keywords (case-insensitive, respecting string literals). If a mutation keyword is found outside a string literal, reject the query with `Err(UnikoError::Pipeline("NL-to-Cypher generated a mutation query, which is forbidden"))`.

#### Retry Logic

If the LLM generates invalid Cypher (parse failure):

1. **First retry:** Include the parse error in the prompt: "The previous query had a syntax error: {error}. Please fix it."
2. **Second retry:** Include both the original error and the second attempt's error.
3. **After 2 retries:** Return `Err(UnikoError::Llm("Failed to generate valid Cypher after 3 attempts"))`.

```rust
const MAX_NL2CYPHER_RETRIES: usize = 2;
```

#### Execution Flow

1. **Normalize** query string (lowercase, trim).
2. **Cache check:** If cache hit, return cached Cypher.
3. **LLM call:** Generate Cypher from prompt with schema.
4. **Mutation check:** `is_read_only()`. If mutation detected, retry with instruction "generate only read queries."
5. **Parse validation:** Attempt to parse as valid Cypher. If parse fails, retry with error context.
6. **Cache store:** Store successful translation in LRU cache.
7. **Return** the validated Cypher string.

#### Latency Target

NF13: 200-500ms. This is LLM-dependent -- the LLM call dominates latency. Cache hits return in < 1ms. Without LLM, NL-to-Cypher is non-functional (returns error).

---

### 14.4 -- Contrastive Retrieval Mode

**Objective:** Extend the recall cascade's Phase 2 to retrieve failure-outcome episodes alongside success-outcome episodes, allowing agents to learn from both what worked and what didn't.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/recall/contrastive.rs` | New | Contrastive episode retrieval logic |
| `crates/uniko-memory/src/recall/cascade.rs` | Modified | Integration of contrastive mode into Phase 2 |

#### Structs and Functions

```rust
/// Retrieve failure episodes that contrast with success episodes.
///
/// For entities found in success episodes, find failure episodes
/// involving the same entities and similar action types.
pub async fn retrieve_contrastive_episodes(
    kb: &KnowledgeBase,
    success_episodes: &[Episode],
    limit: usize,
) -> Result<Vec<ContrastiveEpisode>>;

/// An episode with contrastive labeling.
pub struct ContrastiveEpisode {
    pub episode: Episode,
    /// "success" or "failure"
    pub tier_label: String,
    /// Which success episode(s) this contrasts with.
    pub contrasts_with: Vec<EpisodeId>,
}
```

#### Query Logic

When `.contrastive(true)` is set on `RecallContextBuilder`:

1. **Phase 2 executes normally:** Retrieves success-outcome episodes via vector search and graph traversal as usual.

2. **Contrastive extension:** For the entities mentioned in success episodes:
   ```cypher
   MATCH (e:Episode)-[:MENTIONS]->(ent:Entity)
   WHERE e.outcome = 'failure'
     AND ent.entity_id IN $success_entity_ids
     AND e.action_type IN $success_action_types
   RETURN e
   ORDER BY e.timestamp DESC
   LIMIT $limit
   ```

3. **Label results:** Success episodes are labeled with `tier_label: "success"`. Failure episodes are labeled with `tier_label: "failure"`.

4. **Include in ContextBundle:** Both success and failure episodes are included in the ContextBundle's episodic tier, with tier labeling preserved so the agent can distinguish them.

#### Integration with RecallContextBuilder

The existing `.contrastive(false)` method (from Phase 10) activates this behavior:

```rust
// In RecallContextBuilder
pub fn contrastive(mut self, enabled: bool) -> Self {
    self.contrastive_enabled = enabled;
    self
}
```

When `contrastive_enabled`:
- Phase 2 runs `retrieve_contrastive_episodes()` after its normal episode retrieval.
- Failure episodes are appended to the ContextBundle with appropriate labeling.
- Token budget is shared: contrastive episodes consume from the same budget as success episodes.

#### Contrastive Episode Limit

Default: retrieve up to `limit / 3` failure episodes (where `limit` is the total episode limit). This ensures success episodes remain dominant but failure context is available.

Example: if the recall limit is 15 items and 6 are episodes, up to 2 failure episodes are retrieved alongside 4 success episodes.

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_assume_single_fact` | `reasoning/assume.rs` | Fork -> add fact -> query -> verify fact visible -> verify rollback (fact not in main state) |
| `test_assume_multiple_mutations` | `reasoning/assume.rs` | Multiple facts + entities + edges applied atomically |
| `test_assume_entity_creation` | `reasoning/assume.rs` | Temporary entity visible in forked state, absent after rollback |
| `test_assume_edge_creation` | `reasoning/assume.rs` | Temporary edge visible in forked state, absent after rollback |
| `test_assume_then_locy` | `reasoning/assume.rs` | Locy rule executes correctly against forked state |
| `test_assume_then_cypher` | `reasoning/assume.rs` | Cypher query executes correctly against forked state |
| `test_assume_no_state_corruption` | `reasoning/assume.rs` | After ASSUME, main graph state is identical to before |
| `test_assume_chaining` | `reasoning/assume.rs` | Multiple chained assumptions all visible in single query |
| `test_assume_error_no_mutations` | `reasoning/assume.rs` | Error when no mutations specified |
| `test_assume_error_no_query` | `reasoning/assume.rs` | Error when no query specified |
| `test_assume_error_both_queries` | `reasoning/assume.rs` | Error when both Locy and Cypher specified |
| `test_assume_error_missing_entity` | `reasoning/assume.rs` | Error when edge references non-existent entity |
| `test_assume_result_fields` | `reasoning/assume.rs` | AssumeResult has correct mutations_applied count and query_time_ms |
| `test_abduce_direct_support` | `reasoning/abduce.rs` | Existing fact directly supporting conclusion is found |
| `test_abduce_transitive_support` | `reasoning/abduce.rs` | Facts connected via rules are found at correct depth |
| `test_abduce_no_support` | `reasoning/abduce.rs` | No supporting facts returns empty result with confidence 0.0 |
| `test_abduce_minimal_set` | `reasoning/abduce.rs` | Minimal (not redundant) set of supporting facts returned |
| `test_abduce_confidence_computation` | `reasoning/abduce.rs` | Confidence is product of fact confidences |
| `test_abduce_max_depth` | `reasoning/abduce.rs` | Search stops at max_depth |
| `test_abduce_derivation_chain` | `reasoning/abduce.rs` | DerivationSteps correctly trace from conclusion to supporting facts |
| `test_abduce_error_incomplete_conclusion` | `reasoning/abduce.rs` | Error when conclusion missing fields |
| `test_nl2cypher_basic` | `nl2cypher/mod.rs` | Simple NL query produces valid Cypher |
| `test_nl2cypher_entity_search` | `nl2cypher/mod.rs` | "Find all entities of type person" -> valid MATCH query |
| `test_nl2cypher_relationship` | `nl2cypher/mod.rs` | "What does X depend on?" -> correct edge traversal |
| `test_nl2cypher_temporal` | `nl2cypher/mod.rs` | "What happened last week?" -> BTree range query on timestamp |
| `test_nl2cypher_aggregation` | `nl2cypher/mod.rs` | "How many facts about X?" -> COUNT query |
| `test_nl2cypher_cache_hit` | `nl2cypher/mod.rs` | Repeat query returns cached result without LLM call |
| `test_nl2cypher_cache_capacity` | `nl2cypher/mod.rs` | Cache evicts LRU entry when full (256 entries) |
| `test_nl2cypher_mutation_blocking` | `nl2cypher/validator.rs` | Generated query with DELETE is rejected |
| `test_nl2cypher_mutation_blocking_create` | `nl2cypher/validator.rs` | Generated query with CREATE is rejected |
| `test_nl2cypher_mutation_blocking_set` | `nl2cypher/validator.rs` | Generated query with SET is rejected |
| `test_nl2cypher_mutation_in_string_literal` | `nl2cypher/validator.rs` | "DELETE" inside a string literal is NOT blocked |
| `test_nl2cypher_retry_on_parse_failure` | `nl2cypher/mod.rs` | Parse failure triggers retry with error context |
| `test_nl2cypher_retry_exhausted` | `nl2cypher/mod.rs` | After 2 retries, returns error |
| `test_nl2cypher_schema_introspection` | `nl2cypher/schema.rs` | SchemaInfo correctly lists all node types, edge types, indexes |
| `test_is_read_only_valid` | `nl2cypher/validator.rs` | Valid read query passes |
| `test_is_read_only_mutation` | `nl2cypher/validator.rs` | Mutation queries fail |
| `test_contrastive_retrieval_basic` | `recall/contrastive.rs` | Failure episodes for same entities returned |
| `test_contrastive_retrieval_action_type_match` | `recall/contrastive.rs` | Only failure episodes with matching action_types included |
| `test_contrastive_retrieval_labeling` | `recall/contrastive.rs` | Success episodes labeled "success", failure labeled "failure" |
| `test_contrastive_retrieval_limit` | `recall/contrastive.rs` | Failure episode count respects limit / 3 |
| `test_contrastive_disabled` | `recall/contrastive.rs` | No failure episodes when contrastive not enabled |
| `test_contrastive_no_failures` | `recall/contrastive.rs` | No error when no matching failure episodes exist |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_assume_full_lifecycle` | `tests/reasoning_integration.rs` | Create facts -> ASSUME new fact -> query shows combined results -> verify original state unchanged |
| `test_assume_with_recall` | `tests/reasoning_integration.rs` | ASSUME mutation affects recall results within forked state |
| `test_abduce_with_rules` | `tests/reasoning_integration.rs` | Create facts + rules -> ABDUCE finds facts via rule derivation chain |
| `test_nl2cypher_10_queries` | `tests/reasoning_integration.rs` | 10 diverse NL queries all produce valid, executable Cypher |
| `test_nl2cypher_execution` | `tests/reasoning_integration.rs` | Generated Cypher executes against real graph and returns results |
| `test_contrastive_in_recall` | `tests/reasoning_integration.rs` | RecallContextBuilder with `.contrastive(true)` returns both success and failure episodes in ContextBundle |
| `test_assume_concurrent` | `tests/reasoning_integration.rs` | Multiple concurrent ASSUME operations don't interfere |
| `test_assume_latency` | `tests/reasoning_integration.rs` | Single ASSUME completes in < 200ms (NF9) |
| `test_nl2cypher_latency` | `tests/reasoning_integration.rs` | NL-to-Cypher round-trip completes in 200-500ms (NF13) |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_assume_no_corruption` | For any set of random mutations and query, the main graph state is identical before and after ASSUME |
| `proptest_abduce_confidence_range` | Confidence is always in [0.0, 1.0] for any fact confidence inputs |
| `proptest_is_read_only_deterministic` | `is_read_only()` produces the same result for the same input across runs |
| `proptest_cache_consistency` | Cache hit returns the same result as a fresh LLM call would |

### Validation Criteria

- ASSUME produces correct results in forked state without corrupting main state
- Multiple concurrent ASSUME operations are isolated from each other
- ABDUCE finds minimal supporting fact sets with correct derivation chains
- NL-to-Cypher accuracy: 10+ diverse queries produce valid, correct Cypher
- NL-to-Cypher blocks all mutation keywords (CREATE, SET, DELETE, MERGE, REMOVE)
- NL-to-Cypher cache provides sub-millisecond hits on repeat queries
- Contrastive retrieval includes failure episodes with correct labeling
- ASSUME latency < 200ms (NF9)
- NL-to-Cypher latency 200-500ms (NF13)

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module doc | `reasoning/mod.rs` | Overview of reasoning capabilities (ASSUME, ABDUCE), when to use each |
| Module doc | `reasoning/assume.rs` | AssumeBuilder usage examples, mutation types, fork/rollback semantics, latency expectations |
| Module doc | `reasoning/abduce.rs` | AbduceBuilder usage examples, search strategy, confidence computation, depth limiting |
| Module doc | `nl2cypher/mod.rs` | NL-to-Cypher flow, cache behavior, retry logic, mutation blocking |
| Module doc | `nl2cypher/schema.rs` | SchemaInfo generation, how schema is provided to LLM |
| Module doc | `nl2cypher/validator.rs` | Mutation keyword list, string literal handling, validation logic |
| Module doc | `recall/contrastive.rs` | Contrastive retrieval concept, integration with Phase 2, labeling, budget allocation |
| Inline rustdoc on `AssumeBuilder` | `assume.rs` | Full fluent API documentation with chaining examples |
| Inline rustdoc on `AbduceBuilder` | `abduce.rs` | Full API documentation with derivation chain examples |
| Inline rustdoc on `SchemaInfo` | `schema.rs` | How schema info is auto-generated and used in prompts |

---

## Review Checklist

- [ ] `AssumeBuilder` supports `.assume_fact()`, `.assume_entity()`, `.assume_edge()`
- [ ] `AssumeBuilder` supports both `.then_query()` (Locy) and `.then_cypher()` (Cypher)
- [ ] Multiple mutations can be chained before query
- [ ] ASSUME translates to valid `ASSUME { } THEN { }` Locy statement
- [ ] Fork/rollback is handled by KnowledgeBase Locy runtime (no manual cleanup)
- [ ] No state corruption after ASSUME (verified by test)
- [ ] ASSUME error handling covers: no mutations, no query, both queries, missing entity
- [ ] ASSUME latency < 200ms (NF9, verified by benchmark test)
- [ ] `AbduceBuilder` requires subject, predicate, and object
- [ ] `max_depth` limits backward search depth (default: 3)
- [ ] ABDUCE returns minimal supporting fact set with derivation chain
- [ ] Confidence is product of fact confidences, clamped to [0.0, 1.0]
- [ ] Empty result returns confidence 0.0, not error
- [ ] NL-to-Cypher uses SchemaInfo auto-generated from registered schema
- [ ] LRU cache capacity = 256, keyed on normalized query
- [ ] Cache hit returns without LLM call
- [ ] Mutation blocking rejects CREATE, SET, DELETE, MERGE, REMOVE
- [ ] Mutation keywords inside string literals are NOT blocked
- [ ] Retry: up to 2 retries with error context in prompt
- [ ] After 2 retries, returns error (not infinite loop)
- [ ] NL-to-Cypher latency 200-500ms (NF13, verified by benchmark test)
- [ ] NL-to-Cypher returns error when LLM unavailable (not degraded -- F72 lists it as non-functional offline)
- [ ] Contrastive retrieval activated by `.contrastive(true)` on RecallContextBuilder
- [ ] Failure episodes match on same entities AND action_types as success episodes
- [ ] Tier labeling: "success" and "failure" preserved in ContextBundle
- [ ] Failure episode limit: <= limit / 3
- [ ] No error when contrastive enabled but no failure episodes exist
- [ ] All public types derive `Debug`
- [ ] No `unwrap()` or `expect()` on LLM calls

---

## Definition of Done

1. **ASSUME works correctly:** Forked state contains all mutations, query executes against forked state, results are correct, main state is unchanged after rollback. Verified with single and chained mutations.
2. **ASSUME is isolated:** Multiple concurrent ASSUME operations do not interfere with each other or with the main graph state.
3. **ASSUME meets latency target:** Single ASSUME completes in < 200ms (NF9) on warm in-memory store with < 10K nodes.
4. **ABDUCE finds supporting facts:** Given a conclusion, the system finds the minimal set of existing facts that would support it, with correct derivation chains and confidence computation.
5. **NL-to-Cypher produces valid queries:** 10+ diverse natural language questions are translated to valid, executable Cypher queries that return correct results.
6. **NL-to-Cypher is safe:** All generated queries are verified read-only. Mutation keywords are blocked. String literal false positives are handled.
7. **NL-to-Cypher caches effectively:** Repeat queries return cached results without LLM calls. Cache respects 256-entry capacity with LRU eviction.
8. **NL-to-Cypher retries gracefully:** Parse failures trigger up to 2 retries with error context. Third failure returns a clear error.
9. **Contrastive retrieval works:** When enabled, Phase 2 retrieves failure episodes alongside successes with correct labeling and budget allocation.
10. **Offline behavior correct:** ASSUME and ABDUCE work without LLM (they use Locy runtime). NL-to-Cypher returns error without LLM. Contrastive retrieval works without LLM.
11. **All unit tests pass:** `cargo nextest run -n auto -p uniko-cortex -p uniko-memory` passes with zero failures for all reasoning, NL-to-Cypher, and contrastive retrieval tests.
12. **Clippy clean:** `cargo clippy -p uniko-cortex -p uniko-memory -- -D warnings` passes.
13. **Documented:** All public types, builder methods, and error cases have rustdoc with usage examples.
