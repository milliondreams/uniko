# Phase 18: Rule Induction, MCTS Planning, & Multimodal Embedding

## Context

This phase implements the research-tier extensions that push uniko beyond what any existing cognitive memory system offers: automatic rule induction via LLM (Pipeline 8), Monte Carlo Tree Search planning over nested ASSUME/ABDUCE operations, multimodal embedding support (vision, audio, video), and audio/video chunking. These capabilities are experimental — they build on the proven core system (validated in Phase 16) and extend it into territory where no competitor has published results.

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). The 4-layer architecture is: KnowledgeBase (L1, uniko-store) -> Extract (L2, uniko-extract) -> Pipes (uniko-pipes) + Memory (uniko-memory) + Cortex (L3, uniko-cortex) -> Integration (L4). This phase adds capabilities to L2 (multimodal embedding in uniko-extract, audio/video chunking in uniko-extract) and L3 (rule induction in uniko-cortex, MCTS planning in uniko-cortex).

**Key principle:** These are research extensions — they must fail gracefully. If rule induction produces no useful rules, the system continues without them. If MCTS planning times out, it returns the best plan found so far. If multimodal models are unavailable, text-only embedding continues to work. No research extension can degrade the core system's performance.

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 16 (Benchmarks) | Complete | Core system validated — all pipelines working, recall proven, consolidation effective |
| Phase 14 (ASSUME/ABDUCE) | Complete | Hypothetical reasoning infrastructure for rule validation and MCTS simulation |
| Phase 13 (Procedural Memory P6) | Complete | Procedure nodes, effectiveness scoring, pattern detection |
| Phase 8 (Embedding/Summary P7) | Complete | Embedding infrastructure, auto-embed, pooling strategies |
| Phase 10 (Locy Rule Engine) | Complete | Rule evaluation, forward chaining, derivation trees |
| Rule node type (Phase 2) | Complete | Rule schema with lifecycle (candidate → active → demoted → pruned) |
| LLM provider | Available | For rule generation (GENERATE step) |
| CLIP/SigLIP model | Available (optional) | Image embedding |
| CLAP model | Available (optional) | Audio embedding |
| LanguageBind/InternVideo | Available (optional) | Video embedding |
| ImageBind/ONE-PEACE | Available (optional) | Unified multimodal embedding |
| Whisper model | Available (optional) | Audio transcription |
| `whisper-rs` or Whisper API | Available (optional) | Whisper Rust bindings or API client |
| `tokio` 1.x | Available | Async runtime |

## Sub-phases

---

### 18.1 — Pipeline 8: Rule Induction

**Objective:** Implement automatic rule induction — the system discovers patterns in its knowledge graph, generates candidate Locy rules via LLM, validates them against holdout data, and promotes successful rules to active status. This is the self-improvement loop: uniko learns its own consolidation rules from experience.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/consolidation/rule_induction.rs` | Rust | Main rule induction pipeline |
| `crates/uniko-cortex/src/consolidation/rule_mining.rs` | Rust | MINE step: pattern discovery |
| `crates/uniko-cortex/src/consolidation/rule_generation.rs` | Rust | GENERATE step: LLM rule creation |
| `crates/uniko-cortex/src/consolidation/rule_validation.rs` | Rust | VALIDATE step: ASSUME/ABDUCE testing |
| `crates/uniko-cortex/src/consolidation/rule_lifecycle.rs` | Rust | PERSIST, PROMOTE, MONITOR steps |

#### Pipeline Overview

```
MINE → GENERATE → VALIDATE → PERSIST → (after 3 cycles) PROMOTE → MONITOR

Trigger: after N consolidation cycles (default: 10), OR fact count grows > 20% since last induction
Latency target: < 30s per rule induction cycle (NF16)
```

#### `rule_mining.rs` — MINE Step

```rust
/// Discover statistical patterns in the knowledge graph that could become rules.
///
/// Queries the graph for:
///   - Correlations between entity types and outcomes
///   - Temporal patterns (X before Y within time window)
///   - Conditional patterns (state.X > threshold → outcome Y)
///   - Frequency patterns (entities that co-occur > N times)
///   - Causal chains (A → B → C in episode sequences)
pub async fn mine_patterns(
    kb: &KnowledgeBase,
    agent_id: &str,
    config: &MiningConfig,
) -> Result<Vec<Pattern>>;

/// A statistical pattern discovered in the knowledge graph.
pub struct Pattern {
    /// Natural language description of the pattern.
    pub description: String,
    /// Number of episodes/facts supporting this pattern.
    pub support: u32,
    /// Statistical confidence (0.0-1.0): how often the pattern holds.
    pub confidence: f64,
    /// Example episode IDs that exhibit this pattern.
    pub examples: Vec<String>,
    /// Pattern type for downstream processing.
    pub pattern_type: PatternType,
}

pub enum PatternType {
    /// Entity type X correlated with outcome Y.
    Correlation { entity_type: String, outcome: String },
    /// X happens before Y within a time window.
    Temporal { antecedent: String, consequent: String, window_hours: f64 },
    /// When condition C holds, outcome O follows.
    Conditional { condition: String, outcome: String },
    /// Entities A and B co-occur frequently.
    CoOccurrence { entity_a: String, entity_b: String, count: u32 },
    /// Causal chain: A leads to B leads to C.
    CausalChain { steps: Vec<String> },
}

pub struct MiningConfig {
    /// Minimum support count for a pattern to be considered.
    pub min_support: u32,              // default: 5
    /// Minimum confidence for a pattern to be considered.
    pub min_confidence: f64,           // default: 0.6
    /// Maximum patterns to discover per cycle.
    pub max_patterns: usize,           // default: 20
    /// Lookback window for temporal patterns (days).
    pub temporal_lookback_days: u64,   // default: 90
}
```

Mining queries (Cypher/Locy):

```
// Correlation: entity type → outcome
MATCH (ep:Episode)-[:MENTIONS]->(e:Entity)
WHERE ep.outcome IS NOT NULL
WITH e.entity_type AS etype, ep.outcome AS outcome, COUNT(*) AS cnt
WHERE cnt >= $min_support
RETURN etype, outcome, cnt, cnt * 1.0 / SUM(cnt) OVER (PARTITION BY etype) AS confidence

// Temporal: X before Y
MATCH (ep1:Episode)-[:FOLLOWED_BY]->(ep2:Episode)
WHERE ep1.action_type = $action_a AND ep2.action_type = $action_b
  AND ep2.timestamp - ep1.timestamp < duration($window_hours)
WITH ep1.action_type AS before, ep2.action_type AS after, COUNT(*) AS cnt
WHERE cnt >= $min_support
RETURN before, after, cnt

// Co-occurrence: entities that appear together
MATCH (ep:Episode)-[:MENTIONS]->(e1:Entity),
      (ep)-[:MENTIONS]->(e2:Entity)
WHERE e1.entity_id < e2.entity_id
WITH e1.name AS a, e2.name AS b, COUNT(DISTINCT ep) AS cnt
WHERE cnt >= $min_support
RETURN a, b, cnt
ORDER BY cnt DESC LIMIT $max_patterns
```

#### `rule_generation.rs` — GENERATE Step

```rust
/// Generate a candidate Locy rule from a discovered pattern using LLM.
///
/// Input: pattern description + existing rules + schema info
/// Output: Locy source code + natural language description
///
/// One LLM call per pattern. Gated by circuit breaker.
pub async fn generate_rule(
    pattern: &Pattern,
    schema: &SchemaInfo,
    existing_rules: &[Rule],
    provider: &LlmProvider,
) -> Result<CandidateRule>;

pub struct CandidateRule {
    /// Generated Locy source code.
    pub source: String,
    /// Natural language description of what the rule does.
    pub natural_language: String,
    /// The pattern this rule was generated from.
    pub source_pattern: Pattern,
    /// Parse validation: did the Locy source parse successfully?
    pub parse_valid: bool,
}

pub struct SchemaInfo {
    /// Available node types and their properties.
    pub node_types: Vec<NodeTypeInfo>,
    /// Available edge types.
    pub edge_types: Vec<EdgeTypeInfo>,
    /// Available predicates for Fact nodes.
    pub known_predicates: Vec<String>,
}
```

LLM prompt for rule generation:

```
You are generating a Locy rule for uniko's knowledge consolidation system.

Given this observed pattern:
{pattern.description}

Supporting evidence: {pattern.support} episodes with {pattern.confidence:.0%} confidence
Examples: {pattern.examples[0..3]}

Existing rules (avoid duplication):
{existing_rules.iter().map(|r| r.natural_language).join("\n")}

Schema information:
Node types: {schema.node_types}
Edge types: {schema.edge_types}
Known predicates: {schema.known_predicates}

Generate a Locy rule that captures this pattern. The rule should:
1. Have a clear MATCH pattern identifying the relevant graph structure
2. Use WHERE clauses for conditions
3. Produce new Facts or update existing ones via CREATE/MERGE
4. Be general enough to apply beyond the specific examples

Return JSON:
{
  "source": "<locy source code>",
  "natural_language": "<human-readable description of what the rule does>"
}
```

Post-generation validation:

```
1. Parse the Locy source → if parse fails, discard
2. Check for syntax errors → if errors, discard
3. Check that referenced node/edge types exist in schema → if not, discard
4. Check that the rule doesn't duplicate an existing rule:
   - Compare natural_language via embedding similarity > 0.85 → if duplicate, discard
```

#### `rule_validation.rs` — VALIDATE Step

```rust
/// Validate a candidate rule against holdout episodes using ASSUME/ABDUCE.
///
/// Process:
///   1. Select holdout episodes (20% of episodes since last induction)
///   2. ASSUME: apply rule to holdout episodes, check predictions
///   3. ABDUCE: attempt to falsify rule conclusions
///   4. Compute precision, recall, coverage
pub async fn validate_rule(
    rule: &CandidateRule,
    kb: &KnowledgeBase,
    holdout: &[String],  // episode_ids
) -> Result<ValidationResult>;

pub struct ValidationResult {
    /// How many holdout episodes the rule applied to.
    pub applicable_count: u32,
    /// How many applications produced correct predictions.
    pub correct_count: u32,
    /// How many applications produced incorrect predictions.
    pub incorrect_count: u32,
    /// Precision: correct / (correct + incorrect).
    pub precision: f64,
    /// Recall: correct / applicable.
    pub recall: f64,
    /// Coverage: applicable / total_holdout.
    pub coverage: f64,
    /// Whether ABDUCE found a falsification.
    pub falsified: bool,
    /// Falsification details (if any).
    pub falsification_reason: Option<String>,
    /// Overall score: precision * 0.4 + recall * 0.3 + novelty * 0.3.
    pub score: f64,
}
```

Validation flow:

```
For each holdout episode:
  1. ASSUME: fork KB state
     a. Apply rule to the episode's context (entities, observations, facts)
     b. Check: does the rule produce a prediction?
     c. Check: does the prediction match the actual outcome?
     d. Record: correct or incorrect
     e. Restore KB state

  2. ABDUCE (on the rule's conclusion):
     a. Take the rule's conclusion
     b. Search for counter-examples in the full KB
     c. If counter-example found: falsified = true
     d. Record falsification reason

Compute:
  precision = correct / (correct + incorrect)     // if no applications: 0.0
  recall = correct / applicable                   // if no applicable: 0.0
  coverage = applicable / total_holdout
  novelty = 1.0 - max_similarity_to_existing_rules  // higher if rule is different from existing
  score = precision * 0.4 + recall * 0.3 + novelty * 0.3
```

#### `rule_lifecycle.rs` — PERSIST, PROMOTE, MONITOR

**PERSIST — Store qualifying rules:**

```rust
/// Store a validated rule as a candidate Rule node in the graph.
///
/// Acceptance threshold: score >= 0.65
/// Creates Rule node with:
///   source_type: "induced"
///   status: "candidate"
///   confidence: validation_result.score
///   precision, recall, coverage from validation
pub async fn persist_rule(
    kb: &KnowledgeBase,
    rule: &CandidateRule,
    validation: &ValidationResult,
) -> Result<Option<String>>;  // Returns rule_id if accepted, None if below threshold
```

Acceptance criteria:

| Criterion | Threshold | Action if Below |
|---|---|---|
| Score (precision*0.4 + recall*0.3 + novelty*0.3) | >= 0.65 | Discard |
| Precision | >= 0.5 | Discard (too many false positives) |
| Coverage | >= 3 episodes | Discard (too specific) |
| Not falsified | falsified == false | Discard |

**PROMOTE — After successful validations:**

```rust
/// Promote a candidate rule to active status.
///
/// Triggered: after 3 successful validations in subsequent consolidation cycles.
/// A "successful validation" means the rule's predictions were correct
/// when applied to new episodes since the last validation.
pub async fn check_promotion(
    kb: &KnowledgeBase,
    rule_id: &str,
) -> Result<bool>;  // Returns true if promoted

/// Promote a specific rule from candidate to active.
pub async fn promote_rule(
    kb: &KnowledgeBase,
    rule_id: &str,
) -> Result<()>;
```

Promotion criteria:

```
1. Rule status = "candidate"
2. Rule has been validated in >= 3 subsequent consolidation cycles
3. Each validation: precision >= 0.5 AND not falsified
4. → Set status = "active"
5. Active rules are used by P4 consolidation for fact derivation
```

**MONITOR — Re-score existing induced rules:**

```rust
/// Monitor and maintain existing induced rules.
///
/// For each active induced rule:
///   1. Evaluate against new episodes since last scoring
///   2. Update precision, recall, coverage
///   3. Apply confidence decay if not recently validated
///   4. Demote if precision < 0.5
///   5. Prune if no coverage in 30 days
pub async fn monitor_rules(
    kb: &KnowledgeBase,
    agent_id: &str,
) -> Result<MonitorResult>;

pub struct MonitorResult {
    pub rules_evaluated: u32,
    pub rules_demoted: u32,
    pub rules_pruned: u32,
    pub rules_re_promoted: u32,
}
```

Rule lifecycle state machine:

```
                  score >= 0.65
  [discovered] ───────────────── [candidate]
                                     │
                            3 successful validations
                                     │
                                     ▼
                    ┌──────────── [active] ──────────────┐
                    │                                     │
              precision < 0.5                    confidence > 0.60
                    │                            (re-promotion)
                    ▼                                     │
                [demoted] ◄───────────────────────────────┘
                    │
            90 days inactive
                    │
                    ▼
               [pruned] (terminal)

Confidence decay (per missed consolidation cycle):
  stored_confidence = stored_confidence * 0.95

Re-promotion: if demoted rule's confidence > 0.60 after re-evaluation → active

Stdlib/authored rules: exempt from demotion/pruning/decay
```

#### Trigger Conditions

```rust
/// Determine whether rule induction should run.
pub fn should_run_induction(
    cycles_since_last: u32,
    fact_growth_pct: f64,
    config: &InductionConfig,
) -> bool;

pub struct InductionConfig {
    /// Run after N consolidation cycles.
    pub cycle_threshold: u32,          // default: 10
    /// Run when fact count grows by this percentage.
    pub fact_growth_threshold: f64,    // default: 0.20 (20%)
    /// Maximum rules to induce per cycle.
    pub max_rules_per_cycle: usize,    // default: 5
    /// Holdout fraction for validation.
    pub holdout_fraction: f64,         // default: 0.20 (20%)
}
```

---

### 18.2 — MCTS Planning (PlanBuilder)

**Objective:** Implement Monte Carlo Tree Search over the uniko knowledge graph, using nested ASSUME operations for simulation. The PlanBuilder provides a fluent API for constructing multi-step plans where each step is evaluated via Locy rules against the current and hypothetical knowledge states.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-cortex/src/reasoning/mcts.rs` | Rust | `PlanBuilder`, `MctsTree`, MCTS algorithm |
| `crates/uniko-cortex/src/reasoning/mcts_node.rs` | Rust | `MctsNode`, tree node operations |
| `crates/uniko-cortex/src/reasoning/simulation.rs` | Rust | Simulation via nested ASSUME |

#### `mcts.rs` — PlanBuilder (Fluent API)

```rust
/// Fluent API for constructing MCTS-based plans.
///
/// Usage:
///     let result = PlanBuilder::new(kb.clone())
///         .actions(vec!["batch_approve", "parallel_process", "escalate"])
///         .depth(3)
///         .simulations(50)
///         .score_rule("risk_propagation")
///         .exploration(1.41)
///         .state(json!({"pending_items": 100, "risk_level": 0.3}))
///         .cancel(cancel_token)
///         .run()
///         .await?;
pub struct PlanBuilder {
    kb: Arc<KnowledgeBase>,
    actions: Vec<String>,
    depth: usize,
    simulations: usize,
    score_rule: Option<String>,
    exploration: f64,
    state: serde_json::Value,
    cancel: Option<CancellationToken>,
    procedures: Option<Vec<String>>,  // procedure_ids to consider as actions
}

impl PlanBuilder {
    pub fn new(kb: Arc<KnowledgeBase>) -> Self {
        Self {
            kb,
            actions: vec![],
            depth: 3,
            simulations: 50,
            score_rule: None,
            exploration: 1.41,
            state: serde_json::Value::Null,
            cancel: None,
            procedures: None,
        }
    }

    /// Set available actions for planning.
    pub fn actions(mut self, actions: Vec<&str>) -> Self;

    /// Set maximum lookahead depth.
    pub fn depth(mut self, depth: usize) -> Self;

    /// Set number of rollout simulations per node.
    pub fn simulations(mut self, simulations: usize) -> Self;

    /// Set the Locy rule used for scoring states.
    pub fn score_rule(mut self, rule_name: &str) -> Self;

    /// Set the UCB1 exploration constant.
    pub fn exploration(mut self, c: f64) -> Self;

    /// Set the initial world state.
    pub fn state(mut self, state: serde_json::Value) -> Self;

    /// Set cancellation token for graceful abort.
    pub fn cancel(mut self, token: CancellationToken) -> Self;

    /// Use Procedure nodes as available actions (match by preconditions).
    pub fn use_procedures(mut self, procedure_ids: Vec<&str>) -> Self;

    /// Execute the MCTS search.
    pub async fn run(self) -> Result<PlanResult>;
}

/// Result of an MCTS planning run.
pub struct PlanResult {
    /// Best action sequence found.
    pub best_path: Vec<String>,
    /// Score of the best path.
    pub score: f64,
    /// Alternative paths with their scores (top-5).
    pub alternatives: Vec<(Vec<String>, f64)>,
    /// Tree statistics.
    pub tree_stats: TreeStats,
    /// Proof traces from Locy evaluation during simulation.
    pub proof_traces: Vec<DerivationTree>,
    /// Whether the search was cancelled early.
    pub cancelled: bool,
}

pub struct TreeStats {
    /// Total nodes in the search tree.
    pub total_nodes: usize,
    /// Total simulations completed.
    pub total_simulations: usize,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
    /// Time spent in selection (ms).
    pub selection_time_ms: u64,
    /// Time spent in simulation (ms).
    pub simulation_time_ms: u64,
    /// Time spent in backpropagation (ms).
    pub backprop_time_ms: u64,
}
```

#### `mcts_node.rs` — Tree Node

```rust
/// A node in the MCTS search tree.
pub struct MctsNode {
    /// Action that led to this node (None for root).
    pub action: Option<String>,
    /// State at this node.
    pub state: serde_json::Value,
    /// Number of times this node has been visited.
    pub visits: u32,
    /// Cumulative score from all simulations through this node.
    pub total_score: f64,
    /// Mean score: total_score / visits.
    pub mean_score: f64,
    /// Child nodes (one per available action).
    pub children: Vec<MctsNode>,
    /// Parent index (for backpropagation).
    pub parent_idx: Option<usize>,
    /// Depth in the tree.
    pub depth: usize,
}
```

Functions:

- `MctsNode::new(action: Option<String>, state: Value, depth: usize) -> Self`
- `fn ucb1(&self, parent_visits: u32, exploration: f64) -> f64` — UCB1 score: `mean_score + exploration * sqrt(ln(parent_visits) / visits)`. If visits == 0, return `f64::INFINITY` (always explore unvisited nodes).
- `fn best_child(&self, exploration: f64) -> Option<&MctsNode>` — Child with highest UCB1 score.
- `fn is_leaf(&self) -> bool` — No children.
- `fn is_fully_expanded(&self, available_actions: &[String]) -> bool` — All actions have been tried.
- `fn update(&mut self, score: f64)` — Increment visits, add to total_score, recompute mean_score.

#### MCTS Algorithm

```
fn run_mcts(root: MctsNode, config: MctsConfig) -> PlanResult:
  for sim in 0..config.simulations:
    if cancel.is_cancelled(): break

    // 1. SELECTION: traverse tree using UCB1
    node = root
    while !node.is_leaf() && node.is_fully_expanded(config.actions):
      node = node.best_child(config.exploration)

    // 2. EXPANSION: add new child for untried action
    if node.depth < config.depth:
      untried = config.actions - node.children.map(|c| c.action)
      action = untried.choose_random()
      new_state = apply_action(node.state, action)  // via ASSUME
      child = MctsNode::new(Some(action), new_state, node.depth + 1)
      node.children.push(child)
      node = &child

    // 3. SIMULATION: rollout from node to terminal depth
    score = simulate(node.state, config.depth - node.depth, config)

    // 4. BACKPROPAGATION: update scores up the tree
    while node is not root:
      node.update(score)
      node = node.parent
    root.update(score)

  // Extract best path
  best_path = extract_best_path(root)
  alternatives = extract_top_k_paths(root, 5)
  return PlanResult { best_path, alternatives, tree_stats, ... }
```

#### `simulation.rs` — Simulation via Nested ASSUME

```rust
/// Simulate a rollout from a state to a terminal depth.
///
/// Uses nested ASSUME operations:
///   1. Fork the KB state
///   2. Apply action to the forked state
///   3. Evaluate state quality via Locy score_rule
///   4. Repeat for remaining depth
///   5. Restore KB state
///
/// Returns the terminal state score.
pub async fn simulate(
    kb: &KnowledgeBase,
    state: &serde_json::Value,
    remaining_depth: usize,
    actions: &[String],
    score_rule: &str,
    cancel: &CancellationToken,
) -> Result<f64>;

/// Apply an action to a state and return the resulting state.
/// Uses ASSUME to fork KB, apply the action's effects, and compute new state.
async fn apply_action(
    kb: &KnowledgeBase,
    state: &serde_json::Value,
    action: &str,
) -> Result<serde_json::Value>;

/// Evaluate a state using a Locy rule.
/// Returns a score (0.0-1.0) representing state quality.
async fn evaluate_state(
    kb: &KnowledgeBase,
    state: &serde_json::Value,
    score_rule: &str,
) -> Result<f64>;
```

Simulation flow:

```
simulate(state, depth=3, actions=["approve", "escalate", "defer"]):
  current_state = state
  for d in 0..depth:
    if cancel.is_cancelled(): break
    action = random_choice(actions)
    ASSUME:
      fork KB
      apply action to current_state → new_state
      score = evaluate_state(new_state, score_rule)
      restore KB
    current_state = new_state
  return score
```

#### Cancellation Behavior

```
MCTS is cancellation-aware at every expansion:
  - Check cancel token before each simulation
  - If cancelled: stop immediately, return best result found so far
  - PlanResult.cancelled = true indicates partial results
  - Partial results are still valid — just fewer simulations than requested
```

#### Integration with Procedures

When `use_procedures()` is called:

```
1. Load Procedure nodes by IDs
2. For each Procedure: check precondition_rule against current state
3. Only applicable Procedures (preconditions satisfied) become available actions
4. Action = Procedure.name, effects = Procedure.steps
5. Effectiveness scores from Procedure.effectiveness inform initial UCB1 estimates
```

---

### 18.3 — Multimodal Embedding

**Objective:** Add support for embedding non-text content (images, audio, video) into the same vector space, enabling cross-modal search (e.g., text query finds relevant images). This extends the 5 embedding fields on the Artifact node type.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/embedding/multimodal.rs` | Rust | Multimodal embedding coordination |
| `crates/uniko-extract/src/embedding/image.rs` | Rust | CLIP/SigLIP image embedding |
| `crates/uniko-extract/src/embedding/audio.rs` | Rust | CLAP audio embedding |
| `crates/uniko-extract/src/embedding/video.rs` | Rust | LanguageBind/InternVideo video embedding |
| `crates/uniko-extract/src/embedding/unified.rs` | Rust | ImageBind/ONE-PEACE unified embedding |

#### `multimodal.rs` — Coordination

```rust
/// Multimodal embedding coordinator.
/// Dispatches content to the appropriate modality-specific embedding model
/// and optionally runs the unified model for cross-modal search.
pub struct MultimodalEmbedder {
    /// Image embedding model (CLIP/SigLIP).
    image_model: Option<Arc<ImageModel>>,
    /// Audio embedding model (CLAP).
    audio_model: Option<Arc<AudioModel>>,
    /// Video embedding model (LanguageBind/InternVideo).
    video_model: Option<Arc<VideoModel>>,
    /// Unified multimodal model (ImageBind/ONE-PEACE).
    unified_model: Option<Arc<UnifiedModel>>,
}

impl MultimodalEmbedder {
    /// Create a new multimodal embedder with available models.
    /// Models that are None will be skipped — text-only embedding continues.
    pub fn new(config: MultimodalConfig) -> Result<Self>;

    /// Embed content based on its modality.
    /// Returns modality-specific embedding + optional unified embedding.
    pub async fn embed(
        &self,
        content: &[u8],
        modality: Modality,
    ) -> Result<MultimodalEmbedding>;

    /// Check which modalities are available.
    pub fn available_modalities(&self) -> Vec<Modality>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

pub struct MultimodalEmbedding {
    /// Modality-specific embedding.
    pub modality_embedding: Vec<f32>,
    /// Modality-specific embedding dimension.
    pub modality_dim: usize,
    /// Unified embedding (if unified model available).
    pub unified_embedding: Option<Vec<f32>>,
    /// Unified embedding dimension.
    pub unified_dim: Option<usize>,
    /// Which modality this embedding represents.
    pub modality: Modality,
}

pub struct MultimodalConfig {
    /// Path to CLIP/SigLIP model weights.
    pub image_model_path: Option<PathBuf>,
    /// Path to CLAP model weights.
    pub audio_model_path: Option<PathBuf>,
    /// Path to video model weights.
    pub video_model_path: Option<PathBuf>,
    /// Path to unified model weights.
    pub unified_model_path: Option<PathBuf>,
}
```

#### `image.rs` — Image Embedding (CLIP/SigLIP)

```rust
/// Image embedding using CLIP or SigLIP model.
/// Produces a 768-dimensional vector for each image.
pub struct ImageModel {
    /// ONNX runtime session for the vision encoder.
    session: ort::Session,
    /// Image preprocessor (resize, normalize).
    preprocessor: ImagePreprocessor,
}

impl ImageModel {
    /// Load model from ONNX weights.
    pub fn load(model_path: &Path) -> Result<Self>;

    /// Embed a single image.
    ///
    /// Input: raw image bytes (PNG, JPEG, etc.)
    /// Output: 768-dimensional f32 vector
    ///
    /// Pipeline:
    ///   1. Decode image bytes → pixel array
    ///   2. Resize to model input size (224x224 for CLIP)
    ///   3. Normalize pixels (ImageNet mean/std)
    ///   4. Run through vision encoder
    ///   5. L2-normalize output vector
    pub async fn embed_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>>;
}

struct ImagePreprocessor {
    target_size: (u32, u32),  // (224, 224) for CLIP
    mean: [f32; 3],           // ImageNet normalization
    std: [f32; 3],
}
```

Integration with Artifact node:

```
When Artifact.kind == "image":
  1. Read file content as bytes
  2. image_model.embed_image(bytes) → 768d vector
  3. Store in Artifact.image_embedding
  4. If unified model available:
     unified_model.embed(bytes, Modality::Image) → 1024d vector
     Store in Artifact.multimodal_embedding
```

#### `audio.rs` — Audio Embedding (CLAP)

```rust
/// Audio embedding using CLAP model.
/// Produces a 512-dimensional vector for each audio clip.
pub struct AudioModel {
    session: ort::Session,
    preprocessor: AudioPreprocessor,
}

impl AudioModel {
    pub fn load(model_path: &Path) -> Result<Self>;

    /// Embed audio content.
    ///
    /// Input: raw audio bytes (WAV, MP3, FLAC, etc.)
    /// Output: 512-dimensional f32 vector
    ///
    /// Pipeline:
    ///   1. Decode audio → PCM samples
    ///   2. Resample to model sample rate (48kHz for CLAP)
    ///   3. Compute mel spectrogram
    ///   4. Run through audio encoder
    ///   5. L2-normalize output vector
    pub async fn embed_audio(&self, audio_bytes: &[u8]) -> Result<Vec<f32>>;
}

struct AudioPreprocessor {
    target_sample_rate: u32,   // 48000 for CLAP
    n_mels: usize,             // 64
    hop_length: usize,         // 480
    n_fft: usize,              // 1024
}
```

#### `video.rs` — Video Embedding (LanguageBind/InternVideo)

```rust
/// Video embedding using LanguageBind or InternVideo model.
/// Produces a 768-dimensional vector for each video.
pub struct VideoModel {
    vision_session: ort::Session,
    audio_model: Option<Arc<AudioModel>>,
    config: VideoEmbedConfig,
}

impl VideoModel {
    pub fn load(model_path: &Path, audio_model: Option<Arc<AudioModel>>) -> Result<Self>;

    /// Embed a video file.
    ///
    /// Pipeline:
    ///   1. Sample N frames uniformly from the video
    ///   2. Embed each frame using the vision encoder
    ///   3. Pool frame embeddings (mean pooling)
    ///   4. If audio track present:
    ///      a. Extract audio track
    ///      b. Embed audio using AudioModel
    ///      c. Concatenate + project to 768d
    ///   5. L2-normalize final vector
    pub async fn embed_video(&self, video_path: &Path) -> Result<Vec<f32>>;
}

pub struct VideoEmbedConfig {
    /// Number of frames to sample.
    pub num_frames: usize,          // default: 8
    /// Frame sampling strategy.
    pub sampling: FrameSampling,
}

pub enum FrameSampling {
    /// Sample frames uniformly across the video.
    Uniform,
    /// Sample frames at scene boundaries.
    SceneBoundary,
}
```

#### `unified.rs` — Unified Multimodal Embedding (ImageBind/ONE-PEACE)

```rust
/// Unified multimodal embedding using ImageBind or ONE-PEACE.
/// Maps ALL modalities (text, image, audio, video) into a shared
/// 1024-dimensional vector space, enabling cross-modal search.
///
/// Cross-modal search example:
///   Text query: "sunset over ocean"
///   → embed as text → 1024d vector
///   → KNN search on Artifact.multimodal_embedding
///   → Returns images, videos, audio of sunsets/oceans
pub struct UnifiedModel {
    text_session: ort::Session,
    image_session: ort::Session,
    audio_session: ort::Session,
    video_session: ort::Session,
}

impl UnifiedModel {
    pub fn load(model_dir: &Path) -> Result<Self>;

    /// Embed any modality into the unified vector space.
    ///
    /// All modalities produce 1024-dimensional vectors that are
    /// directly comparable via cosine similarity.
    pub async fn embed_multimodal(
        &self,
        content: &[u8],
        modality: Modality,
    ) -> Result<Vec<f32>>;

    /// Embed text into the unified vector space.
    /// This enables text-query → multimodal-result search.
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
}
```

#### Cross-Modal Search Integration

```
Text query: "meeting about quarterly results"

Search flow:
  1. Embed query text with unified model → 1024d vector
  2. KNN search on Artifact.multimodal_embedding (all modalities)
  3. Results may include:
     - Text documents about quarterly results
     - Audio recordings of the meeting
     - Video of the presentation
     - Slides (images) from the meeting
  4. Also search text-specific embeddings (Artifact.text_embedding, Chunk.embedding)
  5. Merge results by RRF across all embedding spaces
```

Integration with P7c (artifact pooling):

```
After modality-specific embedding is computed:
  1. If unified model available:
     Run unified model on the same content → multimodal_embedding
  2. Store both:
     - Modality-specific: Artifact.image_embedding / audio_embedding / video_embedding
     - Unified: Artifact.multimodal_embedding
  3. P7c artifact pooling: for text artifacts, pool chunk embeddings → text_embedding
     For multimodal artifacts: modality-specific + unified embeddings stored directly
```

---

### 18.4 — Audio/Video Chunking

**Objective:** Implement chunking strategies for audio and video content, creating Chunk nodes that can be searched, embedded, and linked to entities. Audio is chunked by speaker turns (with transcription and diarization). Video is chunked by scene boundaries (with transcript alignment and keyframe extraction).

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-extract/src/chunking/audio.rs` | Rust | Audio transcription, diarization, chunking |
| `crates/uniko-extract/src/chunking/video.rs` | Rust | Scene detection, transcript alignment, keyframe extraction |
| `crates/uniko-extract/src/chunking/transcription.rs` | Rust | Whisper transcription wrapper |
| `crates/uniko-extract/src/chunking/diarization.rs` | Rust | Speaker diarization wrapper |

#### `transcription.rs` — Whisper Transcription

```rust
/// Transcribe audio to text using Whisper model.
pub struct Transcriber {
    /// Whisper model (via whisper-rs or API).
    model: WhisperModel,
}

impl Transcriber {
    pub fn load(model_path: &Path) -> Result<Self>;

    /// Transcribe audio bytes to text with timestamps.
    ///
    /// Returns: list of segments, each with text, start (ms), end (ms).
    pub async fn transcribe(&self, audio_bytes: &[u8]) -> Result<Vec<TranscriptSegment>>;
}

pub struct TranscriptSegment {
    /// Transcribed text.
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
    /// Confidence score (0.0-1.0).
    pub confidence: f64,
}
```

#### `diarization.rs` — Speaker Diarization

```rust
/// Identify who spoke when in an audio recording.
pub struct Diarizer {
    /// Diarization model (via pyannote bindings or custom).
    model: DiarizationModel,
}

impl Diarizer {
    pub fn load(model_path: &Path) -> Result<Self>;

    /// Diarize audio: identify speaker turns.
    ///
    /// Returns: list of speaker segments.
    pub async fn diarize(&self, audio_bytes: &[u8]) -> Result<Vec<SpeakerSegment>>;
}

pub struct SpeakerSegment {
    /// Speaker identifier (e.g., "SPEAKER_00", "SPEAKER_01").
    pub speaker: String,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}
```

#### `audio.rs` — Audio Chunking

```rust
/// Chunk audio content into speaker turns or fixed segments.
///
/// Strategy:
///   1. Transcribe audio → transcript segments
///   2. If diarization available: identify speaker turns
///   3. Align transcript to speaker turns → one chunk per speaker turn
///   4. If no diarization: fixed 30s segments aligned to sentence boundaries
///   5. Large turns (> max_chunk_tokens): split at sentence boundaries
pub async fn chunk_audio(
    audio_bytes: &[u8],
    transcriber: &Transcriber,
    diarizer: Option<&Diarizer>,
    config: &AudioChunkConfig,
) -> Result<Vec<AudioChunk>>;

pub struct AudioChunk {
    /// Chunk text (transcribed).
    pub text: String,
    /// Chunk type.
    pub chunk_type: AudioChunkType,
    /// Speaker (if diarization available).
    pub speaker: Option<String>,
    /// Start time in the original audio (ms).
    pub start_ms: u64,
    /// End time in the original audio (ms).
    pub end_ms: u64,
    /// Token count of the text.
    pub token_count: usize,
}

pub enum AudioChunkType {
    /// One speaker's continuous speech.
    SpeakerTurn,
    /// Fixed-duration segment (no diarization).
    AudioSegment,
}

pub struct AudioChunkConfig {
    /// Fixed segment duration when no diarization (ms).
    pub segment_duration_ms: u64,        // default: 30_000 (30s)
    /// Maximum chunk size in tokens.
    pub max_chunk_tokens: usize,         // default: 512
    /// Minimum chunk size in tokens (merge small turns).
    pub min_chunk_tokens: usize,         // default: 32
}
```

Chunking flow:

```
Input: audio file (WAV/MP3/FLAC)

Case 1: Diarization available
  1. Transcribe → [TranscriptSegment { text, start, end }]
  2. Diarize → [SpeakerSegment { speaker, start, end }]
  3. Align: for each SpeakerSegment, collect overlapping TranscriptSegments
  4. One AudioChunk per SpeakerSegment:
       text = concatenated transcript text within the turn
       speaker = speaker ID
       chunk_type = SpeakerTurn
       start_ms, end_ms from SpeakerSegment
  5. If turn > max_chunk_tokens: split at sentence boundaries within the turn
  6. If turn < min_chunk_tokens and next turn is same speaker: merge

Case 2: No diarization
  1. Transcribe → [TranscriptSegment]
  2. Group segments into 30s windows, aligned to sentence boundaries:
     - Start a window at 0ms
     - Add segments until window duration >= 30s
     - Find the nearest sentence boundary (period/question mark) for clean break
     - Start next window after the break
  3. One AudioChunk per window:
       text = concatenated transcript text
       speaker = None
       chunk_type = AudioSegment
       start_ms, end_ms from window boundaries
```

Graph integration:

```
Audio Artifact → HAS_CHUNK → Chunk (for each AudioChunk)

Chunk properties:
  chunk_id: deterministic from artifact_id + index
  text: transcribed text
  chunk_type: "speaker_turn" or "audio_segment"
  speaker: speaker ID (if available)
  start: start_ms (stored as Int64)
  end: end_ms (stored as Int64)
  token_count: computed
  embedding: auto-embedded from text (P7a)

After chunk creation:
  → P2 NER runs on chunk text → Entity + MENTIONS edges
  → P3 Observation extraction runs on chunk text → Observation + edges
  → P7 audio embedding on original audio → Artifact.audio_embedding
```

#### `video.rs` — Video Chunking

```rust
/// Chunk video content by scene boundaries with transcript alignment.
///
/// Strategy:
///   1. Detect scene boundaries via frame-level analysis
///   2. Extract audio track → transcribe + diarize
///   3. Align transcript to scenes
///   4. One chunk per scene with transcript text
///   5. Extract keyframe per scene → store as child Artifact
pub async fn chunk_video(
    video_path: &Path,
    transcriber: &Transcriber,
    diarizer: Option<&Diarizer>,
    config: &VideoChunkConfig,
) -> Result<Vec<VideoChunk>>;

pub struct VideoChunk {
    /// Chunk text (transcript aligned to this scene).
    pub text: String,
    /// Scene index.
    pub scene_index: usize,
    /// Speaker(s) in this scene (from transcript alignment).
    pub speakers: Vec<String>,
    /// Start time of the scene (ms).
    pub start_ms: u64,
    /// End time of the scene (ms).
    pub end_ms: u64,
    /// Keyframe image bytes (JPEG).
    pub keyframe: Vec<u8>,
    /// Token count of the text.
    pub token_count: usize,
}

pub struct VideoChunkConfig {
    /// Cosine distance threshold for scene boundary detection.
    pub scene_threshold: f64,        // default: 0.5
    /// Minimum scene duration (ms). Merge shorter scenes.
    pub min_scene_duration_ms: u64,  // default: 2_000 (2s)
    /// Maximum scene duration (ms). Split longer scenes.
    pub max_scene_duration_ms: u64,  // default: 60_000 (60s)
    /// Maximum chunk size in tokens.
    pub max_chunk_tokens: usize,     // default: 512
    /// Keyframe quality (JPEG quality 1-100).
    pub keyframe_quality: u8,        // default: 85
}

/// Detect scene boundaries in a video.
///
/// Algorithm:
///   1. Extract frames at regular intervals (e.g., every 500ms)
///   2. Compute visual feature embedding for each frame (using image model)
///   3. Compute cosine distance between consecutive frame embeddings
///   4. Frame pairs where distance > scene_threshold = scene boundary
///   5. Merge boundaries closer than min_scene_duration
pub async fn detect_scene_boundaries(
    video_path: &Path,
    image_model: &ImageModel,
    config: &VideoChunkConfig,
) -> Result<Vec<SceneBoundary>>;

pub struct SceneBoundary {
    /// Time of the boundary (ms).
    pub time_ms: u64,
    /// Cosine distance that triggered this boundary.
    pub distance: f64,
}

/// Extract a keyframe from a scene.
///
/// Selects the frame closest to the temporal midpoint of the scene.
/// Encodes as JPEG at the configured quality.
pub fn extract_keyframe(
    video_path: &Path,
    start_ms: u64,
    end_ms: u64,
    quality: u8,
) -> Result<Vec<u8>>;
```

Chunking flow:

```
Input: video file (MP4/AVI/MOV)

1. Scene detection:
   - Extract frames every 500ms
   - Compute CLIP embedding for each frame
   - Cosine distance between consecutive frames
   - Distance > 0.5 = scene boundary
   - Merge scenes shorter than 2s with neighbors

2. Audio extraction:
   - Extract audio track using ffmpeg/gstreamer
   - Transcribe audio → TranscriptSegments
   - If diarization available: identify speaker turns

3. Transcript alignment:
   - For each scene [start_ms, end_ms]:
     - Collect TranscriptSegments overlapping this time range
     - Concatenate text → scene transcript
     - Identify speakers active in this scene

4. Chunk creation:
   - For each scene:
     VideoChunk {
       text: aligned transcript,
       scene_index: sequential,
       speakers: speakers active in scene,
       start_ms, end_ms,
       keyframe: extract_keyframe(video, start, end, quality),
       token_count: count_tokens(text),
     }

5. Large scenes (> max_chunk_tokens): split at sentence boundaries within transcript
```

Graph integration:

```
Video Artifact → HAS_CHUNK → Chunk (for each VideoChunk)

Chunk properties:
  chunk_id: deterministic from artifact_id + scene_index
  text: scene transcript
  chunk_type: "scene"
  speaker: primary speaker (if single), None (if multiple)
  start: start_ms
  end: end_ms
  token_count: computed
  embedding: auto-embedded from text

Keyframe storage:
  For each scene with a keyframe:
    Create child Artifact:
      kind: "image"
      path: "{parent_artifact_path}#scene-{scene_index}"
      content: None (binary)
      mime_type: "image/jpeg"
    Create HAS_CHUNK edge: parent Artifact → keyframe Artifact? 
    Or: CREATED_BY edge: keyframe Artifact → Action (the ingest action)
    
    Embed keyframe: image_model.embed_image(keyframe_bytes) → image_embedding
    Store in keyframe Artifact.image_embedding

After chunk creation:
  → P2 NER runs on scene transcript → Entity + MENTIONS edges
  → P3 Observation extraction on transcript → Observation + edges
  → P7 video embedding on full video → Artifact.video_embedding
  → P7 audio embedding on audio track → Artifact.audio_embedding
```

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_mine_correlation_pattern` | `rule_mining.rs` | Correlation between entity type and outcome detected |
| `test_mine_temporal_pattern` | `rule_mining.rs` | "X before Y" temporal pattern detected |
| `test_mine_cooccurrence_pattern` | `rule_mining.rs` | Frequently co-occurring entities detected |
| `test_mine_min_support_filter` | `rule_mining.rs` | Patterns below min_support are excluded |
| `test_mine_min_confidence_filter` | `rule_mining.rs` | Patterns below min_confidence are excluded |
| `test_generate_rule_valid_locy` | `rule_generation.rs` | Generated Locy source parses successfully |
| `test_generate_rule_rejects_duplicate` | `rule_generation.rs` | Rule similar to existing (embedding > 0.85) is discarded |
| `test_generate_rule_validates_schema` | `rule_generation.rs` | Rule referencing non-existent node type is discarded |
| `test_validate_rule_correct_predictions` | `rule_validation.rs` | Rule with correct predictions gets high precision |
| `test_validate_rule_falsified` | `rule_validation.rs` | ABDUCE finds counter-example → falsified = true |
| `test_validate_score_formula` | `rule_validation.rs` | score = precision*0.4 + recall*0.3 + novelty*0.3 |
| `test_persist_rule_accepted` | `rule_lifecycle.rs` | Rule with score >= 0.65 persisted as candidate |
| `test_persist_rule_rejected` | `rule_lifecycle.rs` | Rule with score < 0.65 discarded |
| `test_promote_after_3_validations` | `rule_lifecycle.rs` | Candidate promoted to active after 3 successful cycles |
| `test_promote_blocked_by_failure` | `rule_lifecycle.rs` | Candidate not promoted if validation fails |
| `test_monitor_demote_low_precision` | `rule_lifecycle.rs` | Active rule demoted when precision drops < 0.5 |
| `test_monitor_prune_inactive` | `rule_lifecycle.rs` | Demoted rule pruned after 90 days no coverage |
| `test_monitor_re_promote` | `rule_lifecycle.rs` | Demoted rule re-promoted when confidence > 0.60 |
| `test_confidence_decay` | `rule_lifecycle.rs` | Confidence decays by 0.95^missed_cycles |
| `test_stdlib_exempt_from_demotion` | `rule_lifecycle.rs` | Stdlib rules never demoted/pruned |
| `test_should_run_induction_cycle` | `rule_induction.rs` | Triggers after N consolidation cycles |
| `test_should_run_induction_growth` | `rule_induction.rs` | Triggers when fact count grows > 20% |
| `test_ucb1_unvisited` | `mcts_node.rs` | Unvisited nodes return f64::INFINITY |
| `test_ucb1_formula` | `mcts_node.rs` | UCB1 = mean + c * sqrt(ln(parent_visits) / visits) |
| `test_ucb1_exploration_vs_exploitation` | `mcts_node.rs` | Higher exploration constant prefers less-visited nodes |
| `test_mcts_best_child` | `mcts_node.rs` | best_child returns node with highest UCB1 |
| `test_mcts_node_update` | `mcts_node.rs` | Update increments visits, updates mean_score |
| `test_plan_builder_defaults` | `mcts.rs` | Default PlanBuilder has depth=3, simulations=50, exploration=1.41 |
| `test_plan_builder_fluent_api` | `mcts.rs` | Chained builder calls produce correct configuration |
| `test_mcts_cancellation` | `mcts.rs` | Cancelled MCTS returns best result found so far |
| `test_mcts_returns_alternatives` | `mcts.rs` | PlanResult contains top-5 alternative paths |
| `test_simulation_via_assume` | `simulation.rs` | Simulation forks KB, applies action, evaluates, restores |
| `test_image_embed_dimensions` | `image.rs` | Image embedding produces 768d vector |
| `test_image_embed_normalized` | `image.rs` | Image embedding is L2-normalized |
| `test_audio_embed_dimensions` | `audio.rs` | Audio embedding produces 512d vector |
| `test_video_embed_dimensions` | `video.rs` | Video embedding produces 768d vector |
| `test_video_frame_sampling` | `video.rs` | N frames sampled uniformly from video |
| `test_unified_embed_dimensions` | `unified.rs` | Unified embedding produces 1024d vector |
| `test_unified_cross_modal` | `unified.rs` | Text and image of same concept have high cosine similarity |
| `test_transcribe_audio` | `transcription.rs` | Whisper produces text segments with timestamps |
| `test_diarize_speakers` | `diarization.rs` | Multiple speakers identified with time ranges |
| `test_audio_chunk_speaker_turns` | `audio.rs` | Audio chunked by speaker turns (with diarization) |
| `test_audio_chunk_fixed_segments` | `audio.rs` | Audio chunked by fixed 30s segments (no diarization) |
| `test_audio_chunk_large_turn_split` | `audio.rs` | Large speaker turns split at sentence boundaries |
| `test_audio_chunk_small_turn_merge` | `audio.rs` | Small turns from same speaker merged |
| `test_scene_boundary_detection` | `video.rs` | Scene boundaries detected at high cosine distance frames |
| `test_scene_merge_short` | `video.rs` | Scenes shorter than min_duration merged |
| `test_video_chunk_transcript_alignment` | `video.rs` | Transcript text aligned to correct scene time ranges |
| `test_keyframe_extraction` | `video.rs` | Keyframe extracted at scene midpoint, encoded as JPEG |
| `test_video_chunk_multiple_speakers` | `video.rs` | Multiple speakers per scene correctly attributed |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_rule_induction_full_cycle` | `rule_induction.rs` | MINE → GENERATE → VALIDATE → PERSIST → MONITOR complete |
| `test_rule_induction_to_promotion` | `rule_induction.rs` | Candidate rule promoted after 3 successful validations |
| `test_induced_rule_used_in_consolidation` | `rule_induction.rs` | Active induced rule produces new Facts during P4 |
| `test_mcts_produces_valid_plan` | `mcts.rs` | MCTS returns executable action sequence |
| `test_mcts_with_procedures` | `mcts.rs` | Procedures used as actions with precondition matching |
| `test_mcts_score_rule_evaluation` | `mcts.rs` | Locy score rule evaluates states during simulation |
| `test_cross_modal_search` | `multimodal.rs` | Text query finds relevant images via unified embedding |
| `test_audio_ingest_end_to_end` | `audio.rs` | Audio file → transcribe → chunk → embed → searchable via text |
| `test_video_ingest_end_to_end` | `video.rs` | Video file → scenes → chunks → keyframes → searchable via text |
| `test_video_keyframe_as_artifact` | `video.rs` | Keyframes stored as child Artifacts with image_embedding |
| `test_multimodal_graceful_degradation` | `multimodal.rs` | Missing models → text-only embedding works, no errors |
| `test_rule_induction_graceful_failure` | `rule_induction.rs` | LLM unavailable → induction skips GENERATE, no errors |

### Performance Tests

| Test | Target | What It Validates |
|---|---|---|
| `bench_mine_patterns` | < 5s per cycle | Pattern mining at moderate KB size (1000 episodes) |
| `bench_rule_induction_cycle` | < 30s total (NF16) | Complete MINE → GENERATE → VALIDATE cycle |
| `bench_mcts_50_simulations` | < 10s | 50 simulations with depth 3, 5 actions |
| `bench_image_embed` | < 100ms per image | CLIP embedding inference speed |
| `bench_audio_transcribe_30s` | < 5s | 30-second audio transcription |
| `bench_scene_detection_60s` | < 10s | Scene boundary detection for 60-second video |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_ucb1_always_positive` | UCB1 score is always >= 0 for visited nodes |
| `proptest_ucb1_monotone_exploration` | Higher exploration constant → higher UCB1 for less-visited nodes |
| `proptest_mcts_visits_sum` | Sum of child visits <= parent visits |
| `proptest_rule_score_bounded` | Score is always in [0.0, 1.0] |
| `proptest_embedding_normalized` | All modality embeddings have L2 norm ~= 1.0 |
| `proptest_audio_chunks_cover_full_duration` | Union of chunk time ranges covers full audio duration |
| `proptest_video_chunks_cover_full_duration` | Union of scene time ranges covers full video duration |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Rule induction overview | `consolidation/rule_induction.rs` | Full pipeline description, trigger conditions, lifecycle |
| Mining patterns doc | `consolidation/rule_mining.rs` | Pattern types, mining queries, confidence computation |
| Rule lifecycle doc | `consolidation/rule_lifecycle.rs` | State machine (candidate → active → demoted → pruned), decay formula |
| PlanBuilder API doc | `reasoning/mcts.rs` | Fluent API usage examples, parameter tuning guide |
| MCTS algorithm doc | `reasoning/mcts.rs` | UCB1 formula, selection/expansion/simulation/backprop steps |
| Multimodal embedding doc | `embedding/multimodal.rs` | Supported modalities, model requirements, cross-modal search |
| Audio chunking doc | `chunking/audio.rs` | Speaker turn vs fixed segment strategies, diarization integration |
| Video chunking doc | `chunking/video.rs` | Scene detection algorithm, transcript alignment, keyframe extraction |

---

## Review Checklist

### Rule Induction (P8)

- [ ] MINE: discovers correlation, temporal, conditional, co-occurrence, causal chain patterns
- [ ] MINE: min_support and min_confidence filters applied
- [ ] MINE: max_patterns cap enforced
- [ ] GENERATE: LLM produces valid Locy source (parses successfully)
- [ ] GENERATE: duplicate rules detected (embedding similarity > 0.85) and discarded
- [ ] GENERATE: rules referencing non-existent schema types discarded
- [ ] GENERATE: gated by circuit breaker (LLM unavailable → skip)
- [ ] VALIDATE: ASSUME forks KB state, applies rule, checks predictions
- [ ] VALIDATE: ABDUCE searches for counter-examples (falsification)
- [ ] VALIDATE: score formula = precision*0.4 + recall*0.3 + novelty*0.3
- [ ] PERSIST: acceptance threshold >= 0.65 enforced
- [ ] PERSIST: precision >= 0.5 required
- [ ] PERSIST: coverage >= 3 episodes required
- [ ] PERSIST: falsified rules rejected
- [ ] PERSIST: Rule node created with source_type="induced", status="candidate"
- [ ] PROMOTE: after 3 successful validations → status="active"
- [ ] PROMOTE: failed validation resets validation count
- [ ] MONITOR: active rules re-evaluated against new episodes
- [ ] MONITOR: precision < 0.5 → demoted
- [ ] MONITOR: 90 days no coverage → pruned (terminal)
- [ ] MONITOR: confidence decay = stored_confidence * 0.95^missed_cycles
- [ ] MONITOR: demoted rule re-promoted when confidence > 0.60
- [ ] Stdlib/authored rules exempt from demotion/pruning/decay
- [ ] Trigger: runs after N consolidation cycles (default 10) OR fact growth > 20%
- [ ] Latency: < 30s per induction cycle (NF16)

### MCTS Planning

- [ ] PlanBuilder fluent API: actions, depth, simulations, score_rule, exploration, state, cancel
- [ ] UCB1: `mean_score + exploration * sqrt(ln(parent_visits) / visits)`
- [ ] UCB1: unvisited nodes return f64::INFINITY
- [ ] Selection: traverse tree using UCB1
- [ ] Expansion: add child for untried action
- [ ] Simulation: nested ASSUME (fork → apply → evaluate → restore)
- [ ] Backpropagation: update scores up the tree
- [ ] Cancellation: check token at each expansion, return best so far
- [ ] PlanResult: best_path, score, alternatives (top-5), tree_stats, proof_traces
- [ ] Procedure integration: load Procedures, check preconditions, use as actions
- [ ] TreeStats: total_nodes, total_simulations, max_depth_reached, timing

### Multimodal Embedding

- [ ] Image: CLIP/SigLIP → 768d vector, L2-normalized
- [ ] Audio: CLAP → 512d vector, L2-normalized
- [ ] Video: LanguageBind/InternVideo → 768d vector (frame sampling + audio + pooling)
- [ ] Unified: ImageBind/ONE-PEACE → 1024d vector (all modalities same space)
- [ ] Cross-modal: text query → unified embedding → KNN on multimodal_embedding → image/audio/video results
- [ ] Graceful degradation: missing models → text-only embedding works
- [ ] Artifact.image_embedding, audio_embedding, video_embedding, multimodal_embedding populated correctly
- [ ] Integration with P7c artifact pooling

### Audio/Video Chunking

- [ ] Audio transcription via Whisper produces segments with timestamps
- [ ] Speaker diarization identifies speakers with time ranges
- [ ] Audio chunking with diarization: one chunk per speaker turn
- [ ] Audio chunking without diarization: fixed 30s segments at sentence boundaries
- [ ] Large speaker turns split at sentence boundaries (> max_chunk_tokens)
- [ ] Small turns from same speaker merged (< min_chunk_tokens)
- [ ] Chunk metadata: chunk_type, speaker, start (ms), end (ms)
- [ ] Video scene detection: frame cosine distance > threshold
- [ ] Short scenes (< min_duration) merged with neighbors
- [ ] Audio track extracted from video for transcription
- [ ] Transcript aligned to scene time ranges
- [ ] Keyframe extracted per scene at midpoint
- [ ] Keyframe stored as child Artifact (kind: "image")
- [ ] Keyframe Artifact has image_embedding
- [ ] Chunk → P2 NER → P3 Observations pipeline runs on transcript text
- [ ] Audio/video content searchable via text queries after chunking

---

## Definition of Done

1. **Rule induction functional:** Complete MINE → GENERATE → VALIDATE → PERSIST → PROMOTE → MONITOR pipeline works end-to-end. Patterns discovered from knowledge graph, LLM generates valid Locy rules, ASSUME/ABDUCE validates against holdout data, qualifying rules persisted and promoted after 3 successful cycles. Active induced rules produce new Facts during P4 consolidation.
2. **Rule lifecycle enforced:** Candidate → Active (after 3 validations). Active → Demoted (precision < 0.5). Demoted → Pruned (90 days inactive). Confidence decay (0.95^missed_cycles). Re-promotion (confidence > 0.60). Stdlib exempt from lifecycle management.
3. **Rule induction latency:** < 30s per complete induction cycle (NF16), verified by performance test.
4. **MCTS planning functional:** PlanBuilder fluent API works. MCTS produces valid action sequences with scores. UCB1 balances exploration and exploitation. Nested ASSUME simulates action effects via Locy evaluation. Cancellation returns best result found so far. Procedure integration works (preconditions matched).
5. **MCTS correctness:** UCB1 formula produces correct values. Backpropagation correctly updates all ancestors. Visit counts sum correctly. At least 5 alternative paths provided in PlanResult.
6. **Multimodal embedding functional:** Image (768d CLIP), audio (512d CLAP), video (768d LanguageBind), unified (1024d ImageBind/ONE-PEACE) embeddings computed and stored in correct Artifact fields. L2-normalized. Cross-modal search works: text query → relevant images/audio/video.
7. **Graceful degradation:** Missing multimodal models → text-only embedding continues without errors. LLM unavailable → rule induction skips GENERATE step without errors. MCTS cancelled → returns partial results. No research extension degrades core system performance.
8. **Audio chunking functional:** Whisper transcription produces text with timestamps. Speaker diarization identifies speakers. Chunking with diarization: one chunk per speaker turn. Without diarization: fixed 30s segments. Large turns split, small turns merged. Chunks are searchable via text queries.
9. **Video chunking functional:** Scene boundaries detected via frame cosine distance. Transcript aligned to scenes. Keyframes extracted per scene. Keyframes stored as child Artifacts with image_embedding. Video content searchable via text queries through transcript chunks.
10. **All tests pass:** Unit, integration, performance, and property-based tests green for all 4 sub-phases. `cargo nextest run -n auto` passes. No regressions in benchmark scores from Phase 16.
