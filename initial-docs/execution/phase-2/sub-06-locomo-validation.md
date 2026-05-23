# Phase 13: LoCoMo/LongMemEval MVP Validation

## Context

This phase is the gate between MVP (Phases 1-2) and differentiator features (Phases 3+). The spec is explicit: "Phases 1-2 are the MVP. Prove LoCoMo/LongMemEval uplift before investing in Phase 3+." This is NOT the full benchmark harness (that's Phase 4/sub-02, which builds infrastructure for all 5 benchmarks with parameter tuning and competitive comparison). This is a focused validation run — just enough code to load LoCoMo data, ingest it through the full pipeline, query it, and score it. If the scores are insufficient, we iterate on Phase 2 (tune consolidation, improve NER, adjust recall cascade thresholds) before moving on.

The purpose is pragmatic: every component built in Phases 1-2 — NER (P2), Observations (P3), Consolidation (P4), Embedding (P7), Recall Cascade — must prove it works on real benchmark data, not just synthetic unit tests. LoCoMo is the primary benchmark: 10 conversations, 19 sessions each, 5882 turns, 1986 questions across 5 types (single-hop, temporal, multi-hop, open-ended, adversarial). LongMemEval is used selectively to validate 3 critical mechanisms: knowledge updates (BTIC invalidation), temporal reasoning (observed_at correctness), and abstention (correctly reporting "no information").

The validation run also establishes the MVP's baseline numbers: `phase1_only_pct`, `compression_ratio`, `assembly_latency`, and per-question-type accuracy. These become the improvement targets for Phases 3-4. If scores are too low, the decision gate at the end of this phase sends us back to iterate on Phase 2 components rather than forward to Phase 3.

uniko is a cognitive memory system for AI agents built in Rust on uni-db (embedded graph database). The pipeline chain is P1 (Ingest) -> P2 (NER) -> P3 (Observations) -> P4 (Consolidation). P7 (Embedding/Summary) runs alongside. Consolidation derives Facts from Observations using BTIC temporal intervals for validity tracking. The recall cascade is a 3-phase system: Phase 1 (Compact: graph traversal over Facts/Procedures/Topics), Phase 2 (Expand: vector search over Episodes/Observations), Phase 3 (Broaden: fulltext search over Messages/Chunks), with coverage-gated early exit.

**Key principle:** This validation run uses the simplest viable scoring metric — `context_contains_answer` (binary: does the recall ContextBundle contain the answer text?) — because the goal is to validate the memory architecture, not the LLM answer generation. No LLM is needed for this phase. If the context bundle contains the answer, the architecture works; if it doesn't, no amount of prompt engineering will fix it.

## Prerequisites

| Dependency | Status Required | What It Provides |
|---|---|---|
| Phase 12 (Public API Facade) | Complete | `Uniko` entry point, `IngestBuilder`, `recall()`, `shutdown()` |
| Phase 10 (Recall Cascade) | Complete | `RecallContextBuilder`, 3-phase cascade, `ContextBundle` assembly |
| Phase 9 (Consolidation P4) | Complete | Fact derivation from Observations, BTIC invalidation, contradiction detection |
| Phase 8 (Embedding P7) | Complete | All node types embedded for vector search |
| Phase 7 (Observation P3) | Complete | Observations extracted from messages |
| Phase 6 (NER P2) | Complete | Entities extracted, MENTIONS edges created |
| Phase 5 (Ingest P1) | Complete | Messages ingested with session/participant structure |
| Phases 1-4 (Foundation) | Complete | Schema, types, KnowledgeBase, pipeline infrastructure |
| LoCoMo dataset | Downloaded | 10 conversations, 19 sessions each, 5882 turns, 1986 questions |
| LongMemEval dataset (subset) | Downloaded | Questions for knowledge updates, temporal reasoning, abstention |
| `tokio` 1.x | Available | Async runtime |
| `serde_json` | Available | Parsing benchmark data files (JSON) |

## Sub-phases

---

### 13.1 — Minimal Benchmark Runner

**Objective:** Build just enough benchmark infrastructure to load LoCoMo data, ingest conversations, run queries, and compute scores. This is NOT the full harness from Phase 16 — no `Benchmark` trait, no `BenchmarkHarness`, no multi-benchmark orchestration. Just a focused test binary.

#### Files

| File | Type | Purpose |
|---|---|---|
| `tests/benchmarks/locomo_runner.rs` | Rust | Self-contained LoCoMo validation test binary |
| `tests/benchmarks/mod.rs` | Rust | Module root for benchmark tests |
| `tests/benchmarks/scoring.rs` | Rust | `context_contains_answer` scoring function |
| `tests/benchmarks/data_loader.rs` | Rust | LoCoMo/LongMemEval dataset parsing |

#### Core Types

```rust
/// A single LoCoMo question with its expected answer and type classification.
pub struct LoCoMoQuestion {
    pub id: String,
    pub text: String,
    pub question_type: QuestionType,
    pub expected_answer: String,
    pub conversation_id: String,
}

pub enum QuestionType {
    SingleHop,
    Temporal,
    MultiHop,
    OpenEnded,
    Adversarial,
}

/// Result of scoring a single question.
pub struct QuestionResult {
    pub question_id: String,
    pub question_type: QuestionType,
    pub passed: bool,               // context_contains_answer result
    pub assembly_latency_ms: u64,   // time to assemble ContextBundle
    pub phase1_only: bool,          // was Phase 1 of recall sufficient?
    pub context_tokens: usize,      // tokens in the assembled ContextBundle
}

/// Aggregate results for a question type or overall.
pub struct AggregateScore {
    pub total: usize,
    pub passed: usize,
    pub accuracy: f64,              // passed / total
    pub mean_assembly_latency_ms: f64,
    pub phase1_only_pct: f64,
    pub mean_context_tokens: f64,
}
```

#### Scoring Function

```rust
/// Binary context-contains-answer scoring.
///
/// Returns true if the ContextBundle's rendered text contains the expected
/// answer text after normalization (lowercase, strip punctuation, collapse
/// whitespace).
///
/// This is deliberately simple: no F1, no token overlap, no fuzzy matching.
/// If the answer text is in the context, the memory architecture found it.
/// If it isn't, the architecture failed regardless of how good the LLM is.
///
/// Normalization steps:
///   1. Lowercase both strings
///   2. Remove punctuation (keep alphanumeric and whitespace)
///   3. Collapse multiple whitespace to single space
///   4. Trim leading/trailing whitespace
///   5. Check if normalized context contains normalized answer
pub fn context_contains_answer(context: &str, expected_answer: &str) -> bool;

/// Normalize text for comparison: lowercase, strip punctuation,
/// collapse whitespace, trim.
fn normalize_for_comparison(text: &str) -> String;
```

#### Data Loading

```rust
/// Load the LoCoMo dataset from disk.
///
/// Expected directory structure:
///   data/locomo/
///     conversations/
///       conv_01.json  ... conv_10.json
///     questions.json
///
/// Each conversation JSON contains:
///   { "id": "...", "participants": ["Alice", "Bob"],
///     "sessions": [{ "id": "...", "turns": [
///       { "sender": "Alice", "content": "...", "timestamp": "..." }, ...
///     ]}, ...] }
///
/// Questions JSON contains:
///   [{ "id": "...", "text": "...", "type": "single-hop",
///      "expected_answer": "...", "conversation_id": "..." }, ...]
pub fn load_locomo_dataset(data_dir: &Path) -> Result<LoCoMoDataset>;

pub struct LoCoMoDataset {
    pub conversations: Vec<LoCoMoConversation>,
    pub questions: Vec<LoCoMoQuestion>,
}

pub struct LoCoMoConversation {
    pub id: String,
    pub participants: Vec<String>,
    pub sessions: Vec<LoCoMoSession>,
}

pub struct LoCoMoSession {
    pub id: String,
    pub turns: Vec<LoCoMoTurn>,
}

pub struct LoCoMoTurn {
    pub sender: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}
```

#### Runner Entry Point

```rust
/// Run the complete LoCoMo validation.
///
/// Steps:
///   1. Load dataset from data_dir
///   2. Create Uniko instance (in-memory)
///   3. Ingest all conversations (13.2)
///   4. Run all questions (13.3)
///   5. Aggregate and report results (13.5)
///
/// Returns the full result set for analysis.
pub async fn run_locomo_validation(
    data_dir: &Path,
    config: UnikoConfig,
) -> Result<ValidationReport>;

pub struct ValidationReport {
    pub locomo_results: Vec<QuestionResult>,
    pub locomo_aggregate: AggregateScore,
    pub per_type_scores: HashMap<QuestionType, AggregateScore>,
    pub ingest_stats: IngestStats,
    pub longmemeval_results: Option<LongMemEvalResults>,
}
```

---

### 13.2 — LoCoMo Ingestion

**Objective:** Ingest all 10 LoCoMo conversations through the full pipeline (P1 -> P2 -> P3 -> P7a), then run post-ingestion pipelines (P4, P7d, P7b) to derive Facts, generate summaries, and compute embeddings. Verify that the pipeline produces reasonable entity/observation/fact counts.

#### Ingestion Flow

```
For each of 10 conversations:
  1. Create Participant nodes (2 per conversation)
     → Node type: Participant with participant_id and name
     → 20 Participant nodes total

  2. For each of 19 sessions:
     a. Create Session node linked to conversation context
     b. For each turn in the session:
        - Ingest as Message via IngestBuilder:
          uniko.ingest()
              .in_session(&session_id)
              .message_with_timestamp(turn.content, &turn.sender, turn.timestamp)
              .run()
              .await?;
        - Pipeline fires:
          P1: Message node created, SENT_BY → Participant, IN_SESSION → Session
          P2: NER extracts entities, creates Entity nodes, MENTIONS edges
          P3: Observation extraction, creates Observation nodes
          P7a: Auto-embed Message and Observation nodes
     c. Wait for all pipeline steps to complete for the session

  3. After ALL sessions for a conversation are ingested:
     a. Run P4 (Consolidation):
        - Trigger consolidation manually or wait for threshold (20 observations)
        - Consolidation derives Facts from Observations
        - Contradiction detection runs (relevant for adversarial questions later)
        - BTIC intervals set on all derived Facts
     b. Run P7d (Summarization):
        - Generate session summaries for each of the 19 sessions
        - Summary nodes created with embedding
     c. Run P7b (Computed embeddings):
        - Embed Facts (subject + predicate + object)
        - Embed Topics (from entity clustering)
        - Embed any other nodes that need computed embeddings

  4. Wait for all background pipelines to drain
```

#### Expected Counts (Per Conversation, Approximate)

| Metric | Expected Range | Rationale |
|---|---|---|
| Participant nodes | 2 | Each conversation has exactly 2 speakers |
| Session nodes | 19 | LoCoMo defines 19 sessions per conversation |
| Message nodes | ~588 | ~5882 turns / 10 conversations |
| Entity nodes (unique) | ~50 | Named people, places, organizations, events mentioned |
| Observation nodes | ~100 | Factual statements, preferences, events extracted from messages |
| Fact nodes (derived) | ~30 | Observations that meet derivation threshold (>= 3 obs from >= 2 sessions) |
| Summary nodes | 19 | One per session |
| MENTIONS edges | ~200 | Message → Entity links |
| SENT_BY edges | ~588 | One per message |
| IN_SESSION edges | ~588 | One per message |
| SUPPORTED_BY edges | ~90+ | Fact → Observation links (avg 3 obs per fact) |

#### Verification Assertions

```rust
/// Verify that ingestion produced reasonable output for one conversation.
///
/// These are sanity checks, not exact counts. The goal is to catch
/// catastrophic failures (e.g., NER producing 0 entities, or consolidation
/// deriving 0 facts) rather than validate exact numbers.
async fn verify_conversation_ingestion(
    uniko: &Uniko,
    conversation_id: &str,
) -> Result<IngestionVerification>;

pub struct IngestionVerification {
    pub participant_count: usize,     // expected: 2
    pub session_count: usize,         // expected: 19
    pub message_count: usize,         // expected: ~588
    pub entity_count: usize,          // expected: >= 20 (at least some entities found)
    pub observation_count: usize,     // expected: >= 30 (at least some observations)
    pub fact_count: usize,            // expected: >= 5 (at least some facts derived)
    pub summary_count: usize,         // expected: 19 (one per session)
}

/// Minimum thresholds that must be met for the ingestion to be considered
/// successful. Below these, the pipeline is fundamentally broken.
const MIN_ENTITIES_PER_CONVERSATION: usize = 20;
const MIN_OBSERVATIONS_PER_CONVERSATION: usize = 30;
const MIN_FACTS_PER_CONVERSATION: usize = 5;
```

#### Aggregate Ingestion Stats

```rust
pub struct IngestStats {
    pub total_participants: usize,    // expected: 20
    pub total_sessions: usize,        // expected: 190
    pub total_messages: usize,        // expected: 5882
    pub total_entities: usize,        // expected: ~500 (unique across all conversations)
    pub total_observations: usize,    // expected: ~1000
    pub total_facts: usize,           // expected: ~300
    pub total_summaries: usize,       // expected: 190
    pub total_ingest_time_ms: u64,    // wall-clock time for all ingestion
    pub consolidation_cycles: usize,  // number of P4 cycles triggered
}
```

---

### 13.3 — LoCoMo Question Answering

**Objective:** Run all 1986 LoCoMo questions through recall, score each one using `context_contains_answer`, and aggregate results per question type and overall. The token budget is fixed at 8192 tokens — this is the constraint that forces the memory architecture to matter (at 128K budget, raw content dump would work fine).

#### Query Flow

```
For each of 1986 questions:
  1. Construct recall query:
     let bundle = uniko.recall(&question.text, 8192, None).await?;

  2. Render ContextBundle to text:
     let context_text = bundle.render();
     // render() concatenates all items in the bundle into a single
     // text representation, respecting token budget

  3. Score:
     let passed = context_contains_answer(&context_text, &question.expected_answer);

  4. Record metrics:
     QuestionResult {
         question_id: question.id,
         question_type: question.question_type,
         passed,
         assembly_latency_ms: bundle.assembly_latency_ms,
         phase1_only: bundle.phase1_sufficient,
         context_tokens: bundle.total_tokens,
     }
```

#### Token Budget Rationale

```
Budget: 8192 tokens

Why 8K:
  - At 128K tokens, you can dump the entire conversation into context.
    No memory architecture needed — brute force works.
  - At 8K tokens, you must SELECT the right information. This is where
    the recall cascade, entity graph, and compiled facts matter.
  - If the system can answer questions with only 8K tokens of context
    (from a conversation with 20K+ tokens), it proves the architecture
    is doing useful compression and selection.
  - Mem0 reported 91.6% on LoCoMo — we target 75%+ at 8K budget.
```

#### Per-Question-Type Analysis

| Question Type | What It Tests | What Failure Means |
|---|---|---|
| **Single-hop** | Can Phase 1 find a specific Fact or Observation? | Recall cascade doesn't reach the right node. Fix: improve entity matching or graph traversal. |
| **Temporal** | Are `observed_at` timestamps correct? Are events ordered? | P3 isn't extracting temporal information correctly, or recall doesn't respect time ordering. Fix: improve temporal extraction in P3. |
| **Multi-hop** | Does graph traversal cross entity boundaries? | Phase 1 graph traversal is too shallow, or entity deduplication is failing. Fix: increase traversal depth or fix entity merging. |
| **Open-ended** | Does recall aggregate multiple facts about an entity? | ContextBundle doesn't include enough items, or coverage scoring exits too early. Fix: lower Phase 1 coverage threshold. |
| **Adversarial** | Do SENT_BY edges prevent false attribution? | Graph structure isn't enforcing speaker attribution. Fix: ensure recall includes provenance in context, or improve SENT_BY edge creation. |

#### Scoring Edge Cases

```rust
/// Handle scoring edge cases:
///
/// 1. Empty context: bundle has 0 items → passed = false
///    (abstention is correct behavior for questions with no data,
///     but LoCoMo questions always have data, so empty = failure)
///
/// 2. Partial answer: expected is "hiking and pottery", context contains
///    "hiking" but not "pottery" → passed = false
///    (context_contains_answer is strict substring match after normalization)
///
/// 3. Answer spread across items: the answer text appears across multiple
///    ContextBundle items but not in any single item → passed depends on
///    whether render() concatenates items (it should)
///
/// 4. Normalized match: expected "New York City", context has "new york city"
///    → passed = true (normalization handles case)
```

---

### 13.4 — LongMemEval Validation (Subset)

**Objective:** Run a targeted subset of LongMemEval to validate 3 critical MVP mechanisms: knowledge updates (BTIC invalidation), temporal reasoning (observed_at correctness), and abstention (correctly reporting "no information"). This is NOT the full 500-question LongMemEval run — just enough to confirm these mechanisms work on real benchmark data.

#### Scope

| Ability | Questions | Purpose | What Failure Means |
|---|---|---|---|
| Knowledge Updates | ~30 | Validate BTIC invalidation works on real data | P4 contradiction detection or BTIC interval closing is broken |
| Temporal Reasoning | ~30 | Validate observed_at timestamps are correct | P3 temporal extraction or BTIC interval construction is wrong |
| Abstention | ~30 | Validate system reports "no information" for unknown topics | Recall cascade returns false positives, or abstention flag logic is broken |

Total: ~90 questions (vs full LongMemEval's 500).

#### Knowledge Updates Validation

```
Test pattern:
  1. Ingest conversation containing initial fact:
     "Alice lives in New York"
     → P3 creates Observation → P4 derives Fact(subject="Alice", predicate="lives_in", object="New York", valid_at=[t1, ∞))

  2. Ingest later message with update:
     "Alice moved to San Francisco last month"
     → P3 creates contradicting Observation → P4 detects contradiction
     → Old Fact invalidated: valid_at=[t1, t2)
     → New Fact created: valid_at=[t2, ∞)

  3. Query: "Where does Alice live?"
     → recall() should find current Fact (valid_at contains now) = "San Francisco"
     → context_contains_answer(context, "San Francisco") should be true
     → context should NOT prominently contain "New York" as current answer

Validation criteria:
  - Old fact BTIC interval is closed (hi != ∞)
  - New fact BTIC interval is open (hi == ∞)
  - Recall returns current fact, not outdated one
  - Score: >= 80% of knowledge update questions answered with current value
```

#### Temporal Reasoning Validation

```
Test pattern:
  1. Ingest conversation with temporal references:
     "We met last Tuesday to discuss the project" (sent 2026-04-14)
     "The deadline was set for next Friday" (sent 2026-04-14)

  2. Query: "When did you meet to discuss the project?"
     → Observation.observed_at should reflect the resolved date
     → Context should contain temporal information

  3. Query: "What happened before the deadline was set?"
     → Recall should use session ordering and BTIC intervals to find
        events preceding the deadline-setting message

Validation criteria:
  - observed_at dates are plausible (not default/zero dates)
  - Temporal ordering queries return correctly ordered events
  - Score: >= 70% of temporal questions answered correctly
```

#### Abstention Validation

```
Test pattern:
  1. Ingest conversation about specific topics (e.g., cooking, hiking)

  2. Query about unrelated topic: "What did Alice say about quantum physics?"
     → recall() should return low-confidence or empty ContextBundle
     → ContextBundle.abstention should be true
     → context_contains_answer(context, any_fabricated_answer) should be false

  3. Score: 1.0 if system correctly abstains (returns empty or very low-confidence
     context), 0.0 if system returns fabricated content

Validation criteria:
  - ContextBundle has abstention == true for unrelated queries
  - Context items (if any) have low relevance scores
  - Score: >= 80% of abstention questions correctly abstained
```

#### LongMemEval Data Loading

```rust
/// Load a subset of LongMemEval focusing on 3 abilities.
///
/// Expected directory structure:
///   data/longmemeval/
///     knowledge_updates.json
///     temporal_reasoning.json
///     abstention.json
///
/// Each file contains:
///   { "conversations": [...], "questions": [...] }
pub fn load_longmemeval_subset(data_dir: &Path) -> Result<LongMemEvalSubset>;

pub struct LongMemEvalSubset {
    pub knowledge_update_questions: Vec<LongMemEvalQuestion>,
    pub temporal_questions: Vec<LongMemEvalQuestion>,
    pub abstention_questions: Vec<LongMemEvalQuestion>,
    pub conversations: Vec<LongMemEvalConversation>,
}

pub struct LongMemEvalQuestion {
    pub id: String,
    pub text: String,
    pub ability: LongMemEvalAbility,
    pub expected_answer: String,
    pub conversation_id: String,
}

pub enum LongMemEvalAbility {
    KnowledgeUpdate,
    TemporalReasoning,
    Abstention,
}

pub struct LongMemEvalResults {
    pub knowledge_update_score: AggregateScore,
    pub temporal_score: AggregateScore,
    pub abstention_score: AggregateScore,
    pub overall: AggregateScore,
}
```

---

### 13.5 — MVP Benchmark Report

**Objective:** Produce a comprehensive report documenting actual scores vs targets, per-question-type breakdown, key system metrics, gap analysis, and a go/no-go decision for Phase 3.

#### Report Structure

```rust
pub struct MvpBenchmarkReport {
    // LoCoMo scores
    pub locomo_overall: AggregateScore,
    pub locomo_by_type: HashMap<QuestionType, AggregateScore>,

    // LongMemEval subset scores
    pub longmemeval: LongMemEvalResults,

    // System metrics
    pub system_metrics: SystemMetrics,

    // Ingestion stats
    pub ingest_stats: IngestStats,

    // Gap analysis
    pub gaps: Vec<GapAnalysis>,

    // Decision
    pub decision: GateDecision,
}

pub struct SystemMetrics {
    /// Percentage of recalls satisfied by Phase 1 alone.
    /// Higher is better — means compiled knowledge is working.
    pub phase1_only_pct: f64,

    /// Ratio of context_bundle_tokens to total_stored_tokens.
    /// Must exceed 0.1 — proves recall is selective, not dumping.
    pub compression_ratio: f64,

    /// Mean time to assemble a ContextBundle (ms).
    /// Target: < 100ms.
    pub mean_assembly_latency_ms: f64,

    /// 95th percentile assembly latency (ms).
    pub p95_assembly_latency_ms: f64,

    /// Total tokens stored in the system after full ingestion.
    pub total_stored_tokens: usize,

    /// Entity deduplication ratio: unique entities / total entity mentions.
    /// Lower is better — means dedup is working.
    pub entity_dedup_ratio: f64,

    /// Fact compression: facts / observations.
    /// Expected range: 0.1-0.5 (10-50 facts per 100 observations).
    pub fact_observation_ratio: f64,
}

pub struct GapAnalysis {
    /// Which question type or ability is weak.
    pub area: String,

    /// Actual score achieved.
    pub actual: f64,

    /// Target score.
    pub target: f64,

    /// Gap: target - actual.
    pub gap: f64,

    /// Likely root cause.
    pub diagnosis: String,

    /// Recommended fix.
    pub recommendation: String,
}

pub enum GateDecision {
    /// Scores sufficient. Proceed to Phase 3.
    Proceed {
        justification: String,
    },
    /// Scores insufficient. Iterate on Phase 2 before proceeding.
    Iterate {
        areas_to_improve: Vec<String>,
        estimated_effort: String,
    },
}
```

#### Score Targets

| Metric | Target | Pass Condition |
|---|---|---|
| LoCoMo overall | 75%+ | `locomo_overall.accuracy >= 0.75` |
| LoCoMo single-hop | 80%+ | Simplest question type should score highest |
| LoCoMo temporal | 65%+ | Temporal reasoning is harder; accept lower |
| LoCoMo multi-hop | 60%+ | Multi-hop graph traversal is the hardest path |
| LoCoMo open-ended | 70%+ | Aggregation should work with entity graph |
| LoCoMo adversarial | 75%+ | SENT_BY edges must prevent false attribution |
| LongMemEval knowledge updates | 80%+ | BTIC invalidation must work reliably |
| LongMemEval temporal reasoning | 70%+ | Temporal extraction must be functional |
| LongMemEval abstention | 80%+ | Must not hallucinate answers for unknown topics |
| phase1_only_pct | > 0% | At least some recalls should be satisfied by Phase 1 |
| compression_ratio | > 0.1 | Recall must be selective |
| mean_assembly_latency | < 100ms | Assembly must be fast |

#### Report Output

```rust
/// Generate the MVP benchmark report as structured output.
///
/// Outputs:
///   1. Console summary: one-line pass/fail per metric
///   2. Detailed JSON: full results for programmatic analysis
///   3. Gap analysis: per-area diagnosis and recommendations
///   4. Decision gate: proceed to Phase 3 or iterate Phase 2
pub fn generate_report(
    locomo_results: &[QuestionResult],
    longmemeval_results: &LongMemEvalResults,
    ingest_stats: &IngestStats,
) -> MvpBenchmarkReport;
```

#### Console Output Format

```
=== MVP Benchmark Validation Report ===

LoCoMo Results (1986 questions, 8K token budget):
  Overall:      78.3%  [PASS >= 75%]
  Single-hop:   84.1%  [PASS >= 80%]
  Temporal:     68.2%  [PASS >= 65%]
  Multi-hop:    63.5%  [PASS >= 60%]
  Open-ended:   73.8%  [PASS >= 70%]
  Adversarial:  79.1%  [PASS >= 75%]

LongMemEval Subset (90 questions):
  Knowledge Updates:    85.0%  [PASS >= 80%]
  Temporal Reasoning:   72.3%  [PASS >= 70%]
  Abstention:           88.7%  [PASS >= 80%]

System Metrics:
  phase1_only_pct:      23.4%  [PASS > 0%]
  compression_ratio:    0.15   [PASS > 0.1]
  mean_assembly_latency: 45ms  [PASS < 100ms]
  p95_assembly_latency:  82ms  [PASS < 100ms]

Gap Analysis:
  [WEAK] Multi-hop (63.5% vs 60% target): close to threshold
    Diagnosis: Phase 1 graph traversal limited to 2-hop depth
    Recommendation: Increase PPR max_hops to 3, or improve entity dedup

Decision: PROCEED to Phase 3
  All targets met. Multi-hop is the weakest area — monitor in Phase 4.
```

#### Decision Gate Logic

```rust
/// Determine the gate decision based on scores vs targets.
///
/// PROCEED if ALL of:
///   - LoCoMo overall >= 75%
///   - All per-type scores >= their respective targets
///   - LongMemEval knowledge updates >= 80%
///   - LongMemEval abstention >= 80%
///   - phase1_only_pct > 0%
///   - compression_ratio > 0.1
///   - mean_assembly_latency < 100ms
///
/// ITERATE if ANY target is missed.
/// The iterate decision includes:
///   - Which areas failed
///   - Diagnosed root causes
///   - Recommended fixes (tune consolidation, improve NER, adjust recall cascade)
///   - Estimated effort to fix
fn evaluate_gate(report: &MvpBenchmarkReport) -> GateDecision;
```

---

## Test Plan

### Unit Tests

| Test | File | What It Validates |
|---|---|---|
| `test_context_contains_answer_exact` | `scoring.rs` | Exact substring match → true |
| `test_context_contains_answer_case_insensitive` | `scoring.rs` | "New York" in context, "new york" expected → true |
| `test_context_contains_answer_missing` | `scoring.rs` | Answer not in context → false |
| `test_context_contains_answer_partial` | `scoring.rs` | Only part of multi-part answer present → false |
| `test_context_contains_answer_punctuation` | `scoring.rs` | "it's" vs "its" normalized → true |
| `test_normalize_for_comparison` | `scoring.rs` | Strips punctuation, lowercases, collapses whitespace |
| `test_load_locomo_conversations` | `data_loader.rs` | 10 conversations parsed with correct participant/session/turn counts |
| `test_load_locomo_questions` | `data_loader.rs` | 1986 questions parsed with correct types |
| `test_load_longmemeval_subset` | `data_loader.rs` | ~90 questions parsed across 3 abilities |
| `test_aggregate_score_computation` | `locomo_runner.rs` | AggregateScore correctly computes accuracy, mean latency, phase1_only_pct |
| `test_gate_decision_all_pass` | `locomo_runner.rs` | All targets met → GateDecision::Proceed |
| `test_gate_decision_locomo_fail` | `locomo_runner.rs` | LoCoMo overall < 75% → GateDecision::Iterate |
| `test_gate_decision_latency_fail` | `locomo_runner.rs` | Assembly latency > 100ms → GateDecision::Iterate |
| `test_question_type_classification` | `data_loader.rs` | Question types correctly parsed from dataset |

### Integration Tests

| Test | File | What It Validates |
|---|---|---|
| `test_locomo_single_conversation_ingest` | `locomo_runner.rs` | Ingest 1 conversation → entities/observations/facts created within expected ranges |
| `test_locomo_single_question_score` | `locomo_runner.rs` | Single known question → recall → score → correct result |
| `test_locomo_adversarial_attribution` | `locomo_runner.rs` | Adversarial question checks SENT_BY edges → correct speaker identified |
| `test_locomo_temporal_ordering` | `locomo_runner.rs` | Temporal question → events ordered correctly by observed_at |
| `test_longmemeval_knowledge_update` | `locomo_runner.rs` | Contradicting info → BTIC invalidation → current value in context |
| `test_longmemeval_abstention` | `locomo_runner.rs` | Unrelated query → ContextBundle.abstention == true |
| `test_ingestion_verification_thresholds` | `locomo_runner.rs` | Verify entity/observation/fact counts meet minimum thresholds |
| `test_8k_budget_enforcement` | `locomo_runner.rs` | ContextBundle never exceeds 8192 tokens |

### End-to-End Tests

| Test | What It Validates |
|---|---|
| `test_full_locomo_validation` | All 10 conversations ingested, all 1986 questions scored, overall accuracy computed. This is the main validation run — it may take 30+ minutes. Run with `--release` and `--ignored` flag. |
| `test_full_longmemeval_subset` | All ~90 LongMemEval subset questions scored across 3 abilities. Run with `--release` and `--ignored` flag. |
| `test_mvp_gate_decision` | Full validation run → report generated → gate decision computed. The ultimate pass/fail for the MVP. |

### Performance Tests

| Test | Target | What It Validates |
|---|---|---|
| `bench_single_recall_8k` | < 100ms | Single recall at 8K budget completes within latency target |
| `bench_ingest_conversation` | < 60s per conversation | One conversation (588 messages) ingested within time budget |
| `bench_consolidation_after_ingest` | < 30s | Consolidation cycle after full conversation ingest within budget |

---

## Documentation Plan

| Artifact | Location | Contents |
|---|---|---|
| Runner module doc | `tests/benchmarks/locomo_runner.rs` | Overview of the validation run, how to execute, what results mean |
| Scoring doc | `tests/benchmarks/scoring.rs` | `context_contains_answer` rationale, normalization rules, edge cases |
| Data loader doc | `tests/benchmarks/data_loader.rs` | Expected data formats, directory structure, parsing conventions |
| Inline comments | All files | Why 8K token budget, why binary scoring, why these thresholds |
| Report interpretation | `locomo_runner.rs` | How to read the console output, what each metric means, what to do when a metric fails |

---

## Review Checklist

### Runner Infrastructure (13.1)
- [ ] `locomo_runner.rs` exists with `run_locomo_validation` entry point
- [ ] `scoring.rs` implements `context_contains_answer` with normalization
- [ ] `data_loader.rs` loads LoCoMo (10 conversations, 1986 questions) and LongMemEval subset (~90 questions)
- [ ] All types defined: `LoCoMoQuestion`, `QuestionResult`, `AggregateScore`, `QuestionType`
- [ ] `QuestionType` enum has all 5 variants: SingleHop, Temporal, MultiHop, OpenEnded, Adversarial

### Ingestion (13.2)
- [ ] All 10 conversations ingested with correct participant/session structure
- [ ] Pipeline fires for each message: P1 → P2 → P3 → P7a
- [ ] P4 consolidation runs after all sessions per conversation
- [ ] P7d summaries generated for each session
- [ ] P7b computed embeddings generated for Facts, Topics
- [ ] Entity count per conversation >= 20
- [ ] Observation count per conversation >= 30
- [ ] Fact count per conversation >= 5
- [ ] Summary count per conversation == 19

### Question Answering (13.3)
- [ ] All 1986 questions run through `recall(question, budget=8192)`
- [ ] Token budget fixed at 8192 (not configurable — this is the validation constraint)
- [ ] `context_contains_answer` used for scoring (binary, not F1)
- [ ] Per-question metrics recorded: passed, assembly_latency_ms, phase1_only, context_tokens
- [ ] Results aggregated per question type and overall
- [ ] No LLM calls — scoring is against ContextBundle text, not LLM-generated answers

### LongMemEval Validation (13.4)
- [ ] Knowledge update questions: BTIC invalidation produces current value
- [ ] Temporal reasoning questions: observed_at timestamps are plausible
- [ ] Abstention questions: ContextBundle.abstention == true for unrelated queries
- [ ] ~90 questions total (not full 500)
- [ ] Per-ability scores computed

### Report (13.5)
- [ ] Console output shows pass/fail per metric with actual vs target
- [ ] Gap analysis identifies weakest areas with diagnosis and recommendation
- [ ] Gate decision: PROCEED if all targets met, ITERATE if any missed
- [ ] Gate decision includes specific areas to improve and recommended fixes
- [ ] `phase1_only_pct` tracked and reported
- [ ] `compression_ratio` tracked and reported
- [ ] `mean_assembly_latency` and `p95_assembly_latency` tracked and reported

### General
- [ ] Scoring normalization handles case, punctuation, whitespace
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] End-to-end tests pass with `--release`
- [ ] No LLM dependency — validation works purely on context bundle content
- [ ] Results are deterministic (no randomness in recall or scoring)
- [ ] Report clearly states whether to proceed to Phase 3 or iterate Phase 2

---

## Definition of Done

1. **Minimal runner functional:** `locomo_runner.rs` loads LoCoMo data, ingests all 10 conversations, queries all 1986 questions at 8K token budget, and produces scored results. No LLM required — scoring uses `context_contains_answer` directly on ContextBundle text.

2. **Ingestion verified:** All 10 conversations ingested through full pipeline (P1 → P2 → P3 → P4 → P7). Per-conversation verification confirms minimum thresholds: >= 20 entities, >= 30 observations, >= 5 facts, 19 summaries. Total: ~5882 messages, ~500 entities, ~1000 observations, ~300 facts.

3. **LoCoMo scored:** All 1986 questions scored. Overall accuracy >= 75%. Per-type breakdown computed: single-hop >= 80%, temporal >= 65%, multi-hop >= 60%, open-ended >= 70%, adversarial >= 75%.

4. **LongMemEval subset validated:** ~90 questions across 3 abilities scored. Knowledge updates >= 80% (BTIC invalidation works). Temporal reasoning >= 70% (observed_at correct). Abstention >= 80% (no false positives).

5. **System metrics established:** `phase1_only_pct` > 0% (Phase 1 compiled knowledge is being used). `compression_ratio` > 0.1 (recall is selective). `mean_assembly_latency` < 100ms (recall is fast).

6. **Gap analysis complete:** Weakest areas identified with diagnosed root causes and recommended fixes. Specific enough to act on (e.g., "multi-hop fails because graph traversal depth is 2; increase to 3" not "multi-hop needs improvement").

7. **Gate decision rendered:** Clear PROCEED or ITERATE decision based on all targets. If ITERATE: specific areas to improve, specific pipeline components to tune, estimated effort.

8. **Baseline numbers recorded:** All metrics documented as the Phase 2 MVP baseline. These become the improvement targets for Phase 3-4 optimizations. Specifically: LoCoMo per-type scores, phase1_only_pct, compression_ratio, and assembly latency become the reference points.

9. **No forward dependencies:** The validation runner has NO dependency on Phase 3+ features (procedural memory, ASSUME/ABDUCE, MCP server). It uses only Phase 1-2 (MVP) components via the `uniko-api` facade.

10. **Decision gate respected:** If scores are insufficient, Phase 2 is iterated before proceeding. The specific iteration plan is documented in the gap analysis. This phase blocks Phase 3 until all targets are met.
