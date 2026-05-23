# Phase 9: Consolidation Pipeline (P4)

## Context

This phase implements Pipeline 4 — the consolidation engine. This is the "brain" of the cognitive memory system: it derives Facts from Observations, reinforces existing Facts with new evidence, detects and resolves contradictions via BTIC invalidation, detects entity drift, applies memory decay, executes Locy rules, and records consolidation cycles for full provenance.

Consolidation is a background process that never blocks agent operations. It runs periodically (every 15 minutes) or when a threshold of new observations is reached (20). It operates per-agent — each agent has independent consolidation over their own observations and facts.

The consolidation pipeline transforms the system from a passive observation store into an active knowledge system. Without P4, the system has observations but no derived knowledge. With P4, repeated observations crystallize into Facts, contradictions are detected and resolved, entity drift is flagged, and the recall cascade finds compiled knowledge in Phase 1 (Compact) instead of falling through to raw messages in Phase 3 (Broaden).

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). The pipeline chain is P1 (Ingest) -> P2 (NER) -> P3 (Observations) -> P4 (Consolidation). P7 (Embedding/Summary) runs alongside. Consolidation derives Facts from Observations using BTIC temporal intervals for validity tracking.

**Key principle:** Consolidation is conservative. It requires >= 3 observations from >= 2 distinct sessions before deriving a Fact (preventing premature crystallization from a single conversational tangent). Contradictions require > 40% contradicting evidence before invalidation. These thresholds prevent oscillation while ensuring the knowledge graph evolves with new evidence.

## Prerequisites

- **Phase 7 (Observation P3) complete** — P4 consumes observations produced by P3. Observations must exist with OBSERVED_IN, ABOUT, and embedding fields populated.
- **Phase 8 (Embedding P7) complete** — P4 uses embedding cosine similarity for observation clustering and fact matching. All relevant nodes must be embedded.
- **Phase 3 (Schema) complete** — Fact, Observation, Entity, Episode, Rule, ConsolidationCycle node types defined with all fields including BTIC `valid_at` on Facts.
- **Phase 4 (KnowledgeBase L1) complete** — Graph CRUD, vector search, BTIC operations (`btic.contains`, `btic.overlaps`), Locy rule execution.
- **Pipeline management infrastructure** — Step trait, error policies, consolidation worker, bounded channel (32 capacity), semaphore (4 concurrent agent consolidations).
- **Stdlib rules defined** — `relevance_decay`, `episode_pattern_detector`, `sequence_detector`, `contradiction_detector` rules exist as Rule nodes (from Phase 2 or deferred to this phase).

## Sub-phases

---

### 9.1 — ConsolidationStep Trait & Orchestration

**Objective:** Define the step trait for consolidation sub-steps and the orchestrator that executes them in order. Each consolidation cycle runs a fixed sequence of steps, each of which can independently succeed, skip, or fail.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/mod.rs` | Rust | ConsolidationStep trait, orchestrator, context |

#### Core Types

```rust
/// Trait for individual consolidation steps.
/// Each step performs one phase of the consolidation cycle.
/// Steps are executed in a fixed order by the orchestrator.
pub trait ConsolidationStep: Send + Sync {
    /// Human-readable name for logging and metrics.
    fn name(&self) -> &str;

    /// Whether this step should run in the current cycle.
    /// Steps can skip based on context (e.g., no new observations → skip derivation).
    fn should_run(&self, ctx: &ConsolidationContext) -> bool;

    /// Execute the step. May modify the context (add derived facts, flags, etc.).
    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;

    /// Error handling policy. Default: Skip (step failure doesn't abort the cycle).
    fn error_policy(&self) -> StepErrorPolicy {
        StepErrorPolicy::Skip
    }
}

/// Outcome of a consolidation step execution.
pub enum StepOutcome {
    /// Step completed successfully with a summary of what was done.
    Completed {
        facts_created: u32,
        facts_reinforced: u32,
        facts_invalidated: u32,
        drift_alerts: u32,
    },
    /// Step was skipped (should_run returned false, or precondition unmet).
    Skipped { reason: String },
    /// Step failed but error_policy allowed continuation.
    Failed { error: String },
}

/// Shared context passed through all consolidation steps in a cycle.
/// Accumulated state that each step reads and writes.
pub struct ConsolidationContext {
    /// The agent whose knowledge base is being consolidated.
    pub agent_id: String,

    /// Observations loaded since last consolidation cycle.
    pub new_observations: Vec<Observation>,

    /// Existing active Facts (WHERE btic.contains(valid_at, now())).
    pub active_facts: Vec<Fact>,

    /// Active Locy rules (status = "active").
    pub active_rules: Vec<Rule>,

    /// Recent Episodes for pattern detection (after decay pruning).
    pub recent_episodes: Vec<Episode>,

    /// Observations grouped by subject entity (populated in grouping step).
    pub observation_groups: HashMap<String, Vec<Observation>>,

    /// Classification of each observation (populated in pattern detection step).
    pub observation_classifications: HashMap<NodeId, ObservationClass>,

    /// Contradiction flags from P3 and from this cycle's detection.
    pub contradiction_flags: Vec<ContradictionFlag>,

    /// Counters for the ConsolidationCycle record.
    pub counters: CycleCounters,

    /// Reference to KnowledgeBase for graph operations.
    pub kb: Arc<KnowledgeBase>,

    /// Reference to embedding model for similarity computations.
    pub embed_model: Arc<EmbedModel>,

    /// Configuration parameters.
    pub config: Arc<UnikoConfig>,
}

/// Classification of an observation relative to existing facts.
pub enum ObservationClass {
    /// Supports an existing fact.
    Reinforcing { fact_id: NodeId, similarity: f64 },
    /// No matching fact exists — candidate for new fact derivation.
    Novel,
    /// Conflicts with an existing fact.
    Contradicting { fact_id: NodeId, similarity: f64 },
}

/// Counters accumulated during a consolidation cycle.
pub struct CycleCounters {
    pub observations_processed: u32,
    pub episodes_involved: u32,
    pub facts_created: u32,
    pub facts_reinforced: u32,
    pub facts_invalidated: u32,
    pub procedures_promoted: u32,
    pub drift_alerts: u32,
    pub rules_applied: u32,
}
```

#### Orchestration Order

The consolidator executes steps in this fixed order:

```
1. Load          — load observations, facts, rules, episodes (9.2)
2. Decay         — apply memory decay, prune low-importance episodes (9.2)
3. Group         — group observations by subject entity (9.3)
4. Detect        — classify observations as reinforcing/novel/contradicting (9.3)
5. Derive        — create new Facts from novel observation clusters (9.4)
6. Reinforce     — strengthen existing Facts with reinforcing observations (9.5)
7. Contradict    — handle contradictions, invalidate via BTIC (9.6)
8. Drift         — detect entity drift from invalidation patterns (9.7)
9. Rules         — execute active Locy rules (9.8)
10. Record       — create ConsolidationCycle node with edges (9.9)
```

```rust
/// The consolidation orchestrator. Runs the fixed step sequence
/// for a single agent's consolidation cycle.
pub struct ConsolidationOrchestrator {
    steps: Vec<Box<dyn ConsolidationStep>>,
}

impl ConsolidationOrchestrator {
    /// Create a new orchestrator with all steps in order.
    pub fn new() -> Self;

    /// Run a complete consolidation cycle for one agent.
    /// Returns the ConsolidationCycle node ID for provenance.
    pub async fn run_cycle(
        &self,
        kb: Arc<KnowledgeBase>,
        agent_id: &str,
        config: Arc<UnikoConfig>,
        embed_model: Arc<EmbedModel>,
    ) -> Result<NodeId>;
}
```

---

### 9.2 — Loading & Memory Decay

**Objective:** Load all data needed for consolidation, then apply memory decay to prune stale episodes before pattern detection begins.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/load.rs` | Rust | Data loading and memory decay |

#### Loading

```rust
/// Load all data needed for a consolidation cycle.
///
/// Queries:
///   1. Observations created since last ConsolidationCycle for this agent
///   2. Active Facts: WHERE btic.contains(valid_at, now())
///   3. Active Rules: WHERE status = "active"
///   4. Recent Episodes: within configurable lookback window
pub struct LoadStep;

impl ConsolidationStep for LoadStep {
    fn name(&self) -> &str { "load" }

    fn should_run(&self, _ctx: &ConsolidationContext) -> bool {
        true // always runs
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}
```

#### Loading Queries

| Data | Query Pattern | Stored In |
|---|---|---|
| New observations | `MATCH (o:Observation) WHERE o._created_at > $last_cycle_completed_at` | `ctx.new_observations` |
| Active facts | `MATCH (f:Fact) WHERE btic.contains(f.valid_at, now())` | `ctx.active_facts` |
| Active rules | `MATCH (r:Rule) WHERE r.status = "active"` | `ctx.active_rules` |
| Recent episodes | `MATCH (e:Episode) WHERE e.timestamp > $lookback` | `ctx.recent_episodes` |
| Last cycle | `MATCH (c:ConsolidationCycle {agent_id: $agent_id}) ORDER BY c.completed_at DESC LIMIT 1` | Used for `$last_cycle_completed_at` |

#### Memory Decay

```rust
/// Apply memory decay to episodes before pattern detection.
///
/// Formula: decayed_importance = importance * exp(-ln(2) / half_life * age_days)
///
/// Parameters:
///   half_life_days: 30.0 (from UnikoConfig)
///   prune_below: 0.05 (from UnikoConfig)
///
/// Episodes with decayed importance below prune_below are excluded from
/// pattern detection (but not deleted from the graph).
pub struct DecayStep;

impl ConsolidationStep for DecayStep {
    fn name(&self) -> &str { "decay" }

    fn should_run(&self, ctx: &ConsolidationContext) -> bool {
        !ctx.recent_episodes.is_empty()
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}

/// Compute the decayed importance of an episode.
///
/// importance * exp(-ln(2) / half_life_days * age_days)
///
/// Example:
///   importance = 0.8, age = 30 days, half_life = 30 days
///   → 0.8 * exp(-0.693 / 30 * 30) = 0.8 * 0.5 = 0.4
///
///   importance = 0.8, age = 90 days, half_life = 30 days
///   → 0.8 * exp(-0.693 / 30 * 90) = 0.8 * 0.125 = 0.1
///
///   importance = 0.2, age = 60 days, half_life = 30 days
///   → 0.2 * exp(-0.693 / 30 * 60) = 0.2 * 0.25 = 0.05 → pruned at threshold 0.05
fn compute_decayed_importance(
    importance: f64,
    age_days: f64,
    half_life_days: f64,
) -> f64;
```

---

### 9.3 — Observation Grouping & Pattern Detection

**Objective:** Group observations by subject entity, cluster them by semantic similarity, and classify each observation as reinforcing, novel, or contradicting relative to existing Facts.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/patterns.rs` | Rust | Grouping, clustering, classification |

#### Functions

```rust
/// Group observations by their subject entity.
/// Each group contains all observations about the same entity.
pub struct GroupStep;

impl ConsolidationStep for GroupStep {
    fn name(&self) -> &str { "group" }

    fn should_run(&self, ctx: &ConsolidationContext) -> bool {
        !ctx.new_observations.is_empty()
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}

/// Classify observations as reinforcing, novel, or contradicting.
///
/// For each observation in each group:
///   1. Find existing Facts with matching subject (via ABOUT → Entity)
///   2. Compute cosine similarity between observation embedding and fact embedding
///   3. Classify:
///      - cosine > 0.7 with matching fact → REINFORCING
///      - cosine < 0.3 with matching fact → CONTRADICTING
///      - no matching fact (or 0.3 <= cosine <= 0.7) → NOVEL
pub struct DetectStep;

impl ConsolidationStep for DetectStep {
    fn name(&self) -> &str { "detect" }

    fn should_run(&self, ctx: &ConsolidationContext) -> bool {
        !ctx.observation_groups.is_empty()
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}

/// Cluster observations within a subject group by semantic similarity.
///
/// Algorithm:
///   1. Compute pairwise cosine similarity between observation embeddings
///   2. Group observations where cosine > 0.7 (same pattern)
///   3. Each cluster represents a potential Fact
///
/// Returns clusters as Vec<Vec<&Observation>> — each inner Vec is a cluster
/// of semantically similar observations about the same subject.
fn cluster_observations(
    observations: &[Observation],
) -> Vec<Vec<&Observation>>;

/// Match an observation against existing Facts for its subject.
///
/// Returns the best-matching Fact (if any) and the similarity score.
fn match_against_facts(
    observation: &Observation,
    facts: &[Fact],
) -> Option<(NodeId, f64)>;
```

#### Classification Thresholds

| Cosine Similarity | Same Subject | Classification |
|---|---|---|
| > 0.7 | Yes | REINFORCING — supports existing fact |
| 0.3 - 0.7 | Yes | NOVEL — related but different enough to be a new fact |
| < 0.3 | Yes | CONTRADICTING — conflicts with existing fact |
| Any | No matching fact | NOVEL |

---

### 9.4 — Fact Derivation from Novel Observations

**Objective:** When enough novel observations cluster together, derive a new Fact. This is the core knowledge crystallization mechanism.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/derive.rs` | Rust | Fact derivation from observation clusters |

#### Derivation Thresholds

| Threshold | Value | Rationale |
|---|---|---|
| Minimum observations | >= 3 | Avoid deriving facts from single or paired observations |
| Minimum distinct sessions | >= 2 | Prevent premature crystallization from a single conversational tangent |

```rust
/// Derive new Facts from clusters of novel observations.
///
/// For each observation cluster with:
///   - >= 3 observations
///   - from >= 2 distinct sessions (checked via OBSERVED_IN → Message → IN_SESSION)
///
/// Derive a Fact with:
///   subject: common entity
///   predicate: extracted from observation content
///   object: extracted from observation content
///   confidence: observation_count / (observation_count + 2) (Laplace smoothing)
///   valid_at: BTIC [earliest_observed_at, ∞)
pub struct DeriveStep;

impl ConsolidationStep for DeriveStep {
    fn name(&self) -> &str { "derive" }

    fn should_run(&self, ctx: &ConsolidationContext) -> bool {
        ctx.observation_classifications.values()
            .any(|c| matches!(c, ObservationClass::Novel))
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}
```

#### Predicate Extraction

```rust
/// Extract predicate from observation content using verb-frame patterns.
///
/// Patterns:
///   "X is Y"         → predicate: "is"
///   "X has Y"        → predicate: "has"
///   "X attended Y"   → predicate: "attended"
///   "X wants to Y"   → predicate: "wants_to"
///   "X prefers Y"    → predicate: "prefers"
///   "X works at Y"   → predicate: "works_at"
///   "X lives in Y"   → predicate: "lives_in"
///   "X started Y"    → predicate: "started"
///   "X completed Y"  → predicate: "completed"
///   "X pursuing Y"   → predicate: "pursuing"
///
/// If no pattern matches: use LLM one-shot prompt (if available)
///   "Given the observation '{content}', extract: subject, predicate, object."
/// If LLM unavailable: predicate = "related_to" (generic fallback)
///
/// Predicates are normalized to snake_case verb forms.
fn extract_predicate(observation_content: &str, subject: &str) -> (String, String);
```

#### Confidence Calculation

```rust
/// Compute confidence using Laplace smoothing.
///
/// Formula: observation_count / (observation_count + 2)
///
/// Examples:
///   3 observations → 3/5 = 0.60
///   5 observations → 5/7 = 0.71
///   10 observations → 10/12 = 0.83
///   20 observations → 20/22 = 0.91
///   50 observations → 50/52 = 0.96 (but capped at 0.95 by reinforcement)
fn compute_confidence(observation_count: u32) -> f64;
```

#### BTIC Construction

```rust
/// Construct the BTIC valid_at interval for a new Fact.
///
/// lo = earliest observation.observed_at in the cluster
/// hi = ∞ (open, active — the fact is currently believed true)
/// certainty = "approximate" if observation_count < 10, "definite" if >= 10
/// granularity = finest granularity from observation timestamps
///   (day if all observations have day-level timestamps, month if some are month-level)
fn construct_btic(
    observations: &[Observation],
) -> Btic;
```

#### Edge Creation

For each derived Fact:

| Edge | Direction | Target | Properties |
|---|---|---|---|
| SUPPORTED_BY | Fact -> Observation | Each observation in the cluster | `weight: f64` (1.0 / observation_count for equal weight) |
| DERIVED_FROM | Fact -> Episode | Episodes that contain these observations (via OBSERVED_DURING) | — |
| ABOUT | Fact -> Entity | The subject Entity node | — |
| DERIVED_BY | Fact -> Rule | If a Locy rule triggered this derivation (from step 9.8) | — |

```rust
/// Create a Fact node and wire all edges.
///
/// 1. Create Fact node with: fact_id, subject, predicate, object,
///    confidence, observation_count, valid_at (BTIC), visibility ("agent")
/// 2. Compute embedding via P7b: subject + " " + predicate + " " + object
/// 3. Create SUPPORTED_BY edges to each observation
/// 4. Create DERIVED_FROM edges to related Episodes
/// 5. Create ABOUT edge to subject Entity
/// 6. Update ctx.counters.facts_created
async fn create_fact(
    ctx: &mut ConsolidationContext,
    subject: &str,
    predicate: &str,
    object: &str,
    observations: &[&Observation],
) -> Result<NodeId>;
```

#### No-Derivation Guard

```rust
/// Check that a cluster meets derivation thresholds.
/// Returns false if:
///   - cluster has < 3 observations
///   - observations come from < 2 distinct sessions
fn meets_derivation_threshold(
    observations: &[&Observation],
    kb: &KnowledgeBase,
) -> bool;
```

---

### 9.5 — Fact Reinforcement

**Objective:** When new observations support existing Facts, increase the Fact's confidence and add SUPPORTED_BY edges.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/reinforce.rs` | Rust | Fact reinforcement logic |

#### Functions

```rust
/// Reinforce existing Facts with new supporting observations.
///
/// For each observation classified as REINFORCING:
///   1. Compute new confidence
///   2. Increment observation_count
///   3. Add SUPPORTED_BY edge
///   4. Update certainty if crossing 10-observation threshold
pub struct ReinforceStep;

impl ConsolidationStep for ReinforceStep {
    fn name(&self) -> &str { "reinforce" }

    fn should_run(&self, ctx: &ConsolidationContext) -> bool {
        ctx.observation_classifications.values()
            .any(|c| matches!(c, ObservationClass::Reinforcing { .. }))
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}
```

#### Reinforcement Formula

```rust
/// Compute the reinforced confidence value.
///
/// Formula: new_conf = old_conf + (1 - old_conf) * 0.1 * count
/// Capped at 0.95 (facts never reach 1.0 — epistemic humility).
///
/// Examples:
///   old_conf = 0.60, count = 1 → 0.60 + 0.40 * 0.1 * 1 = 0.64
///   old_conf = 0.60, count = 3 → 0.60 + 0.40 * 0.1 * 3 = 0.72
///   old_conf = 0.90, count = 2 → 0.90 + 0.10 * 0.1 * 2 = 0.92
///   old_conf = 0.94, count = 5 → 0.94 + 0.06 * 0.1 * 5 = 0.97 → capped at 0.95
fn compute_reinforced_confidence(
    old_confidence: f64,
    reinforcing_count: u32,
) -> f64;
```

#### Certainty Threshold

When a Fact's `observation_count` crosses 10 (after incrementing):
- Update BTIC certainty from "approximate" to "definite"
- This is a one-time transition per Fact

```rust
/// Update Fact after reinforcement.
///
/// 1. Set new confidence (computed by formula, capped at 0.95)
/// 2. Increment observation_count by reinforcing_count
/// 3. Create SUPPORTED_BY edges to new observations
/// 4. If observation_count crosses 10: update BTIC certainty to "definite"
async fn reinforce_fact(
    ctx: &mut ConsolidationContext,
    fact_id: NodeId,
    reinforcing_observations: &[&Observation],
) -> Result<()>;
```

---

### 9.6 — Contradiction Detection & BTIC Invalidation

**Objective:** When contradicting observations accumulate past the 40% threshold, invalidate the old Fact by closing its BTIC interval and derive a new Fact from the contradicting evidence.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/contradiction.rs` | Rust | Contradiction resolution and BTIC invalidation |

#### Functions

```rust
/// Handle contradictions: count evidence, invalidate when threshold
/// exceeded, derive replacement facts, detect oscillation.
pub struct ContradictStep;

impl ConsolidationStep for ContradictStep {
    fn name(&self) -> &str { "contradict" }

    fn should_run(&self, ctx: &ConsolidationContext) -> bool {
        ctx.observation_classifications.values()
            .any(|c| matches!(c, ObservationClass::Contradicting { .. }))
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}
```

#### Contradiction Threshold

```rust
/// Determine whether a Fact should be invalidated based on evidence ratios.
///
/// For a given Fact:
///   total_observations = reinforcing_count + contradicting_count
///   contradiction_ratio = contradicting_count / total_observations
///
/// If contradiction_ratio > 0.40 → invalidate
/// If contradiction_ratio <= 0.40 → fact survives, contradicting observations noted
///
/// The 40% threshold is intentionally high to prevent oscillation from
/// noisy or ambiguous observations.
fn should_invalidate(
    reinforcing_count: u32,
    contradicting_count: u32,
) -> bool;
```

#### BTIC Invalidation

```rust
/// Invalidate a Fact by closing its BTIC interval.
///
/// Before: valid_at = [2023-05-25, ∞)     — active, believed true
/// After:  valid_at = [2023-05-25, now())  — was true during this window only
///
/// Steps:
///   1. Close BTIC interval: set hi = now()
///   2. Create INVALIDATES edge from new Fact to old Fact with reason
///   3. Derive new Fact from contradicting observations (via DeriveStep)
///   4. Update ctx.counters.facts_invalidated
async fn invalidate_fact(
    ctx: &mut ConsolidationContext,
    fact_id: NodeId,
    reason: &str,
    contradicting_observations: &[&Observation],
) -> Result<()>;
```

#### Oscillation Detection

```rust
/// Detect oscillation: same subject+predicate invalidated repeatedly.
///
/// Query: count Fact nodes with same subject and predicate that have
/// been invalidated (BTIC hi != ∞) in the entire history.
///
/// If count >= 3:
///   1. Create special "unstable" observation:
///      subject = entity_name
///      content = "Entity shows systematic oscillation on predicate '{predicate}'"
///   2. Flag for human review (stored as observation with confidence = 0.0)
///   3. Do NOT derive a new Fact (the evidence is unreliable)
///
/// This prevents the system from endlessly flip-flopping between
/// contradicting facts about the same topic.
fn detect_oscillation(
    kb: &KnowledgeBase,
    subject: &str,
    predicate: &str,
) -> Result<bool>;
```

---

### 9.7 — Drift Detection

**Objective:** Detect entities whose knowledge is systematically changing — multiple invalidations in a short period indicate the real-world entity is drifting, and recall should be more aggressive for queries involving it.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/drift.rs` | Rust | Entity drift detection and alerting |

#### Functions

```rust
/// Detect entity drift: entities with high invalidation rates.
///
/// For each entity that had facts invalidated in this cycle:
///   1. Count total invalidations in the last 30 days
///   2. If count >= 4: create drift alert
///   3. Flag entity for recall cascade override
pub struct DriftStep;

impl ConsolidationStep for DriftStep {
    fn name(&self) -> &str { "drift" }

    fn should_run(&self, ctx: &ConsolidationContext) -> bool {
        ctx.counters.facts_invalidated > 0
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}
```

#### Drift Detection Logic

```rust
/// Count invalidations for an entity in the last 30 days.
///
/// Query:
///   MATCH (f:Fact)-[:ABOUT]->(e:Entity {entity_id: $entity_id})
///   WHERE f.valid_at.hi IS NOT NULL     -- fact was invalidated
///     AND f.valid_at.hi > $thirty_days_ago
///   RETURN count(f)
async fn count_recent_invalidations(
    kb: &KnowledgeBase,
    entity_id: &str,
    lookback_days: u32,
) -> Result<u32>;
```

#### Drift Alert

```rust
/// Create a drift alert for an entity.
///
/// 1. Create Observation:
///    subject: entity_name
///    content: "Entity shows systematic drift"
///    observed_at: now()
///    confidence: 1.0 (this is a system-generated observation, not extracted)
///
/// 2. Flag entity for recall cascade override:
///    When this entity appears in a recall query, force Phase 2+ even if
///    Phase 1 coverage is sufficient (F58). This ensures the recall cascade
///    re-evaluates episodic memory for drifting entities instead of relying
///    on potentially-stale compiled knowledge.
///
/// 3. Update ctx.counters.drift_alerts
async fn create_drift_alert(
    ctx: &mut ConsolidationContext,
    entity_id: &str,
    entity_name: &str,
    invalidation_count: u32,
) -> Result<()>;
```

#### Drift Configuration

| Parameter | Default | Description |
|---|---|---|
| `drift_lookback_days` | 30 | Window for counting invalidations |
| `drift_threshold` | 4 | Minimum invalidations to trigger drift |

#### Latency Target

- < 100ms for drift detection step (NF12)
- Drift detection is lightweight — just graph queries and node creation

---

### 9.8 — Locy Rule Execution in Consolidation

**Objective:** Execute active Locy rules against the current graph state as part of each consolidation cycle. Rules can derive additional Facts, detect patterns, and validate existing Facts.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/rules.rs` | Rust | Locy rule execution within consolidation |

#### Functions

```rust
/// Execute active Locy rules against the current graph state.
///
/// For each active Rule:
///   1. Inject parameters: $agent_id, $promotion_threshold, $contradiction_threshold
///   2. Execute rule via KnowledgeBase Locy API
///   3. Process results: create Facts, detect patterns, validate existing Facts
///   4. Track DERIVED_BY edges from any created Facts to the Rule
pub struct RulesStep;

impl ConsolidationStep for RulesStep {
    fn name(&self) -> &str { "rules" }

    fn should_run(&self, ctx: &ConsolidationContext) -> bool {
        !ctx.active_rules.is_empty()
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}
```

#### Stdlib Rules

| Rule | Purpose | Execution Order |
|---|---|---|
| `relevance_decay` | Apply decay to fact confidence based on age and evidence recency | First (before other rules see stale data) |
| `episode_pattern_detector` | Detect recurring patterns in episode outcomes | After decay |
| `sequence_detector` | Detect temporal sequences in entity mentions | After episode patterns |
| `contradiction_detector` | Detect contradictions that the embedding-based check missed | Last (validates results of other steps) |

```rust
/// Execute a single Locy rule and process its results.
///
/// Injected parameters available to all rules:
///   $agent_id: String — current agent
///   $promotion_threshold: f64 — from config (default 0.7)
///   $contradiction_threshold: f64 — from config (default 0.4)
///   $now: DateTime — current timestamp
///
/// Rule results can:
///   - Derive additional Facts (risk propagation, transitive relationships)
///   - Detect patterns the clustering missed
///   - Validate existing Facts against new evidence
///   - Flag entities for review
async fn execute_rule(
    ctx: &mut ConsolidationContext,
    rule: &Rule,
) -> Result<StepOutcome>;
```

#### DERIVED_BY Edge Tracking

When a Locy rule creates a Fact:
- Create `DERIVED_BY` edge from the new Fact to the Rule
- Set `Fact.source_rule` to the Rule's `rule_id`
- This provides full provenance: "Why does this Fact exist?" → "Because Rule X detected this pattern"

---

### 9.9 — ConsolidationCycle Recording

**Objective:** Record the consolidation cycle as a ConsolidationCycle node with edges to all affected nodes, providing full provenance and auditability.

#### Files

| File | Type | Purpose |
|---|---|---|
| `crates/uniko-memory/src/consolidation/cycle.rs` | Rust | Cycle recording and provenance edges |

#### Functions

```rust
/// Record the consolidation cycle as a graph node with provenance edges.
pub struct RecordStep;

impl ConsolidationStep for RecordStep {
    fn name(&self) -> &str { "record" }

    fn should_run(&self, _ctx: &ConsolidationContext) -> bool {
        true // always record, even if nothing happened
    }

    async fn execute(&self, ctx: &mut ConsolidationContext) -> Result<StepOutcome>;
}
```

#### ConsolidationCycle Node

```rust
/// Create the ConsolidationCycle node.
///
/// Fields (from CycleCounters):
///   cycle_id: new_id()
///   agent_id: ctx.agent_id
///   started_at: cycle start timestamp
///   completed_at: now()
///   observations_processed: ctx.counters.observations_processed
///   episodes_involved: ctx.counters.episodes_involved
///   facts_created: ctx.counters.facts_created
///   facts_reinforced: ctx.counters.facts_reinforced
///   facts_invalidated: ctx.counters.facts_invalidated
///   procedures_promoted: ctx.counters.procedures_promoted
///   drift_alerts: ctx.counters.drift_alerts
async fn create_cycle_node(
    ctx: &ConsolidationContext,
    started_at: DateTime<Utc>,
) -> Result<NodeId>;
```

#### Provenance Edges

| Edge | Direction | Target | Purpose |
|---|---|---|---|
| PROCESSED | ConsolidationCycle -> Observation | Each observation consumed | What was processed |
| INVOLVED | ConsolidationCycle -> Episode | Each episode involved | What experiences informed this cycle |
| CREATED | ConsolidationCycle -> Fact | Each fact derived | What was learned |
| INVALIDATED | ConsolidationCycle -> Fact | Each fact closed | What was unlearned |
| PROMOTED | ConsolidationCycle -> Procedure | Each procedure promoted (from P5 if combined) | What procedures emerged |
| APPLIED_RULE | ConsolidationCycle -> Rule | Each rule executed | What rules were used |

#### Per-Agent Tracking

The consolidation worker maintains per-agent observation counts for trigger decisions:

```rust
/// Track per-agent observation counts for consolidation triggering.
///
/// The consolidation worker receives ObservationsReady notifications
/// from P3. It accumulates counts per agent and triggers consolidation when:
///   - count >= consolidation_threshold (default 20), OR
///   - timer >= consolidation_interval_secs (default 900 = 15 min)
///
/// After a consolidation cycle completes, the counter resets for that agent.
pub struct AgentConsolidationTracker {
    /// Per-agent observation counts since last consolidation.
    counts: HashMap<String, u32>,
    /// Per-agent last consolidation timestamp.
    last_consolidated: HashMap<String, DateTime<Utc>>,
}
```

#### Latency Target

- < 5s per agent for the entire consolidation cycle (NF15)
- Individual steps should be much faster; the 5s budget is for the full sequence

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_decay_formula` | `consolidation/load.rs` | `importance=0.8, age=30d, half_life=30d → 0.4` |
| `test_decay_prune_threshold` | `consolidation/load.rs` | `importance=0.2, age=60d → 0.05 → pruned` |
| `test_decay_fresh_episode` | `consolidation/load.rs` | `age=0 → importance unchanged` |
| `test_clustering_same_pattern` | `consolidation/patterns.rs` | Observations with cosine > 0.7 cluster together |
| `test_clustering_different_patterns` | `consolidation/patterns.rs` | Observations with cosine < 0.7 form separate clusters |
| `test_classification_reinforcing` | `consolidation/patterns.rs` | Observation matching fact (cosine > 0.7) classified as REINFORCING |
| `test_classification_novel` | `consolidation/patterns.rs` | Observation with no matching fact classified as NOVEL |
| `test_classification_contradicting` | `consolidation/patterns.rs` | Observation conflicting with fact (cosine < 0.3) classified as CONTRADICTING |
| `test_confidence_laplace` | `consolidation/derive.rs` | `3 obs → 0.60`, `5 obs → 0.71`, `10 obs → 0.83` |
| `test_derivation_threshold_met` | `consolidation/derive.rs` | 3 observations from 2 sessions → fact derived |
| `test_derivation_threshold_not_met_count` | `consolidation/derive.rs` | 2 observations from 2 sessions → no fact (< 3 obs) |
| `test_derivation_threshold_not_met_sessions` | `consolidation/derive.rs` | 5 observations from 1 session → no fact (< 2 sessions) |
| `test_predicate_extraction_is` | `consolidation/derive.rs` | "Caroline is a social worker" → predicate: "is" |
| `test_predicate_extraction_attended` | `consolidation/derive.rs` | "Caroline attended the group" → predicate: "attended" |
| `test_predicate_extraction_fallback` | `consolidation/derive.rs` | Unmatched pattern → predicate: "related_to" |
| `test_reinforcement_formula` | `consolidation/reinforce.rs` | `old=0.60, count=1 → 0.64` |
| `test_reinforcement_cap` | `consolidation/reinforce.rs` | `old=0.94, count=5 → 0.95 (capped)` |
| `test_reinforcement_certainty_upgrade` | `consolidation/reinforce.rs` | Crossing 10 observations → certainty "approximate" → "definite" |
| `test_contradiction_threshold` | `consolidation/contradiction.rs` | `reinforcing=6, contradicting=4 → 40% → invalidate` |
| `test_contradiction_below_threshold` | `consolidation/contradiction.rs` | `reinforcing=8, contradicting=2 → 20% → no invalidation` |
| `test_oscillation_detection` | `consolidation/contradiction.rs` | Same subject+predicate invalidated 3 times → oscillation flagged |
| `test_drift_threshold` | `consolidation/drift.rs` | 4 invalidations in 30 days → drift alert |
| `test_drift_below_threshold` | `consolidation/drift.rs` | 3 invalidations in 30 days → no drift alert |
| `test_drift_outside_window` | `consolidation/drift.rs` | 5 invalidations but > 30 days old → no drift alert |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_load_observations_since_last_cycle` | `consolidation/load.rs` | Only new observations loaded (since last ConsolidationCycle) |
| `test_load_active_facts` | `consolidation/load.rs` | Only facts with open BTIC intervals loaded |
| `test_decay_prunes_episodes` | `consolidation/load.rs` | Old low-importance episodes excluded from context |
| `test_fact_derivation_creates_node` | `consolidation/derive.rs` | Fact node created with correct fields in uni-db |
| `test_fact_edges_supported_by` | `consolidation/derive.rs` | SUPPORTED_BY edges from Fact to each Observation |
| `test_fact_edges_derived_from` | `consolidation/derive.rs` | DERIVED_FROM edges from Fact to related Episodes |
| `test_fact_edges_about` | `consolidation/derive.rs` | ABOUT edge from Fact to subject Entity |
| `test_fact_btic_active` | `consolidation/derive.rs` | New fact has BTIC [earliest_observed_at, ∞) |
| `test_fact_embedding` | `consolidation/derive.rs` | Fact.embedding computed via P7b (subject + predicate + object) |
| `test_reinforce_updates_confidence` | `consolidation/reinforce.rs` | Fact.confidence updated in graph |
| `test_reinforce_adds_supported_by` | `consolidation/reinforce.rs` | New SUPPORTED_BY edges added |
| `test_btic_invalidation` | `consolidation/contradiction.rs` | Invalidated fact has BTIC [lo, now()) — hi is closed |
| `test_invalidates_edge` | `consolidation/contradiction.rs` | INVALIDATES edge created with reason |
| `test_replacement_fact_derived` | `consolidation/contradiction.rs` | New fact derived from contradicting observations after invalidation |
| `test_oscillation_creates_unstable_obs` | `consolidation/contradiction.rs` | "unstable" observation created on 3rd invalidation |
| `test_drift_creates_alert` | `consolidation/drift.rs` | Drift observation created with correct content |
| `test_drift_flags_entity` | `consolidation/drift.rs` | Entity flagged for recall cascade override (F58) |
| `test_rule_execution` | `consolidation/rules.rs` | Active Locy rule executes and results processed |
| `test_rule_derived_by_edge` | `consolidation/rules.rs` | Fact derived by rule has DERIVED_BY edge |
| `test_cycle_recording` | `consolidation/cycle.rs` | ConsolidationCycle node created with correct counters |
| `test_cycle_provenance_edges` | `consolidation/cycle.rs` | PROCESSED, CREATED, INVALIDATED, APPLIED_RULE edges all created |
| `test_orchestrator_full_cycle` | `consolidation/mod.rs` | Complete cycle: load → decay → group → detect → derive → reinforce → contradict → drift → rules → record |

### End-to-End Tests

| Test | What It Validates |
|---|---|
| `test_message_to_fact_pipeline` | Full pipeline: Message → P1 → P2 → P3 → P4 → Fact. Message ingested, entities extracted, observations created, consolidation derives a Fact. |
| `test_contradiction_lifecycle` | Old observations → Fact derived → new contradicting observations → Fact invalidated → new Fact derived from contradicting evidence. |
| `test_drift_triggers_recall_override` | Entity with 4 invalidations → drift alert → recall query for that entity forces Phase 2+ |
| `test_consolidation_idempotent` | Running consolidation twice with no new observations → second cycle has zero changes |

### Performance Tests

| Test | Target | What It Validates |
|---|---|---|
| `bench_consolidation_cycle` | < 5s per agent (NF15) | Full cycle within latency budget |
| `bench_drift_detection` | < 100ms (NF12) | Drift step within latency target |
| `bench_clustering_100_observations` | < 500ms | Clustering scales to typical observation batch |
| `bench_decay_1000_episodes` | < 50ms | Decay computation is efficient |

### Property-Based Tests

| Test | What It Validates |
|---|---|
| `proptest_confidence_monotonic` | Reinforcement always increases confidence (never decreases) |
| `proptest_confidence_bounded` | Confidence is always in (0.0, 0.95] after reinforcement |
| `proptest_laplace_bounded` | Laplace smoothing is always in (0.0, 1.0) |
| `proptest_decay_monotonic` | Decayed importance is always <= original importance |
| `proptest_decay_non_negative` | Decayed importance is always >= 0.0 |
| `proptest_btic_valid_interval` | After invalidation, BTIC hi > BTIC lo |
| `proptest_derivation_requires_threshold` | Facts only derived when >= 3 obs from >= 2 sessions |

### Validation Criteria

| Metric | Target | Source |
|---|---|---|
| Compression ratio (observations to facts) | > 0.1 (at least 1 fact per 10 observations) | Spec benchmark flow |
| Contradictions invalidate correctly | 100% of facts with > 40% contradicting evidence are invalidated | F38 |
| Drift flags entities | 100% of entities with >= 4 invalidations in 30 days are flagged | F39 |
| No premature crystallization | 0 facts from < 3 observations or < 2 sessions | Conservative threshold |
| Consolidation latency | < 5s per agent | NF15 |
| Oscillation detected | 100% of subject+predicate pairs invalidated >= 3 times | System integrity |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Module-level doc comment | `consolidation/mod.rs` | P4 overview, step sequence, trigger mechanism, relationship to P3 |
| `ConsolidationStep` trait doc | `consolidation/mod.rs` | Trait contract, error_policy semantics, context mutation rules |
| `ConsolidationContext` doc | `consolidation/mod.rs` | What each field holds, lifecycle through the step sequence |
| Memory decay doc | `consolidation/load.rs` | Formula derivation, half-life semantics, pruning rationale |
| Derivation threshold doc | `consolidation/derive.rs` | Why >= 3 observations from >= 2 sessions, anti-premature-crystallization rationale |
| BTIC invalidation doc | `consolidation/contradiction.rs` | Before/after examples, relationship to Allen's interval algebra |
| Oscillation detection doc | `consolidation/contradiction.rs` | Why 3-invalidation threshold, what "unstable" means |
| Drift detection doc | `consolidation/drift.rs` | How drift interacts with recall cascade override (F58) |
| Locy rule execution doc | `consolidation/rules.rs` | Available parameters, stdlib rules, how DERIVED_BY provenance works |
| ConsolidationCycle doc | `consolidation/cycle.rs` | What each counter means, how to query "why does this fact exist?" |

---

## Review Checklist

### Orchestration (9.1)
- [ ] `ConsolidationStep` trait defined with `name`, `should_run`, `execute`, `error_policy`
- [ ] `ConsolidationContext` holds: observations, facts, rules, episodes, groups, classifications, counters
- [ ] Orchestrator runs steps in correct order: load → decay → group → detect → derive → reinforce → contradict → drift → rules → record
- [ ] Each step can independently skip or fail without aborting the cycle
- [ ] error_policy defaults to Skip

### Loading & Decay (9.2)
- [ ] Observations loaded since last ConsolidationCycle (not all observations)
- [ ] Active facts filtered by `btic.contains(valid_at, now())`
- [ ] Active rules filtered by `status = "active"`
- [ ] Memory decay formula: `importance * exp(-ln(2) / half_life * age_days)`
- [ ] Default half_life_days = 30.0
- [ ] Episodes below prune_below (0.05) excluded from pattern detection
- [ ] Decay applied BEFORE pattern detection

### Grouping & Pattern Detection (9.3)
- [ ] Observations grouped by subject entity
- [ ] Semantic clustering: cosine > 0.7 = same pattern
- [ ] Classification: REINFORCING (cosine > 0.7 with fact), NOVEL (no match), CONTRADICTING (cosine < 0.3 with fact)

### Fact Derivation (9.4)
- [ ] Threshold: >= 3 observations from >= 2 distinct sessions
- [ ] No facts derived from < 3 observations (verified by test)
- [ ] No facts derived from 1 session (verified by test)
- [ ] Confidence: Laplace smoothing `observation_count / (observation_count + 2)`
- [ ] BTIC valid_at: `[earliest_observed_at, ∞)`
- [ ] Certainty: "approximate" if < 10 obs, "definite" if >= 10
- [ ] Predicate extraction: verb-frame patterns with snake_case normalization
- [ ] Edges: SUPPORTED_BY → Observations, DERIVED_FROM → Episodes, ABOUT → Entity
- [ ] Fact embedded via P7b: subject + predicate + object

### Reinforcement (9.5)
- [ ] Formula: `new_conf = old_conf + (1 - old_conf) * 0.1 * count`
- [ ] Capped at 0.95
- [ ] observation_count incremented
- [ ] SUPPORTED_BY edges added to new observations
- [ ] Certainty updated at 10-observation threshold crossing

### Contradiction (9.6)
- [ ] Threshold: contradicting > 40% of total observations → invalidate
- [ ] BTIC invalidation: hi set to now() (interval closed)
- [ ] INVALIDATES edge created with reason
- [ ] New Fact derived from contradicting observations
- [ ] Oscillation detection: >= 3 invalidations of same subject+predicate
- [ ] Oscillation creates "unstable" observation, flags for review
- [ ] Oscillation does NOT derive a new Fact

### Drift Detection (9.7)
- [ ] Count invalidations per entity in last 30 days
- [ ] Threshold: >= 4 invalidations → drift alert
- [ ] Drift observation created: "Entity shows systematic drift"
- [ ] Entity flagged for recall cascade override (F58: force Phase 2+)
- [ ] Drift detection latency < 100ms (NF12)

### Locy Rules (9.8)
- [ ] Active rules loaded and executed
- [ ] Parameters injected: $agent_id, $promotion_threshold, $contradiction_threshold
- [ ] DERIVED_BY edge from rule-derived Facts to Rule
- [ ] Stdlib rules: relevance_decay, episode_pattern_detector, sequence_detector, contradiction_detector
- [ ] relevance_decay runs first

### Cycle Recording (9.9)
- [ ] ConsolidationCycle node created with all counters
- [ ] PROCESSED edges to consumed Observations
- [ ] CREATED edges to derived Facts
- [ ] INVALIDATED edges to closed Facts
- [ ] APPLIED_RULE edges to executed Rules
- [ ] Per-agent observation tracking for trigger
- [ ] Consolidation latency < 5s per agent (NF15)

### General
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All end-to-end tests pass
- [ ] All property-based tests pass
- [ ] Performance benchmarks within targets

---

## Definition of Done

1. **Orchestration works:** ConsolidationStep trait implemented by all 9 steps. Orchestrator runs them in correct order. Steps can independently skip or fail.
2. **Loading correct:** Only new observations (since last cycle) loaded. Active facts filtered by BTIC. Active rules filtered by status.
3. **Memory decay applied:** Episodes below pruning threshold (0.05) excluded from pattern detection. Decay formula verified by unit tests.
4. **Pattern detection functional:** Observations grouped by subject, clustered by semantic similarity (cosine > 0.7), and classified as reinforcing/novel/contradicting.
5. **Fact derivation conservative:** Facts only derived from >= 3 observations from >= 2 distinct sessions. No premature crystallization. Confidence follows Laplace smoothing. BTIC intervals correctly set.
6. **Reinforcement works:** Existing Facts strengthened by new evidence. Confidence formula capped at 0.95. Certainty upgrades at 10-observation threshold.
7. **Contradiction resolution correct:** Facts invalidated when > 40% evidence contradicts. BTIC intervals properly closed. INVALIDATES edges created. Replacement facts derived from contradicting evidence.
8. **Oscillation detected:** Same subject+predicate invalidated >= 3 times → "unstable" observation created, no new Fact derived.
9. **Drift detection functional:** Entities with >= 4 invalidations in 30 days flagged. Drift observation created. Recall cascade override triggered (F58).
10. **Locy rules execute:** All active rules run per cycle. DERIVED_BY provenance tracked. Stdlib rules available.
11. **Full provenance recorded:** ConsolidationCycle node with all counters and edges to affected Observations, Facts, and Rules.
12. **End-to-end pipeline works:** Message → P1 → P2 → P3 → P4 → Fact pipeline produces correct Facts from messages.
13. **Latency within targets:** Full cycle < 5s per agent (NF15). Drift detection < 100ms (NF12).
14. **All tests pass:** Unit, integration, end-to-end, property-based, and performance tests green.
