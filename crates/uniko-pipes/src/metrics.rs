//! Pipeline metric registration and emission helpers.
//!
//! All metrics use the `uniko.` prefix and the [`metrics`] crate for
//! Counter, Gauge, and Histogram types.

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, histogram};

/// Register all pipeline metrics with the global recorder.
///
/// Safe to call multiple times — descriptions are idempotent.
pub fn register_pipeline_metrics() {
    describe_counter!(
        "uniko.ingest.items_total",
        "Total items submitted to ingest"
    );
    describe_counter!("uniko.ingest.items_failed", "Items that failed at a step");
    describe_histogram!(
        "uniko.ingest.duration_ms",
        "End-to-end ingest processing time per item"
    );

    describe_counter!(
        "uniko.consolidation.cycles_total",
        "Total consolidation cycles"
    );
    describe_counter!(
        "uniko.consolidation.facts_derived",
        "Facts created during consolidation"
    );
    describe_counter!(
        "uniko.consolidation.facts_invalidated",
        "Facts invalidated during consolidation"
    );
    describe_histogram!(
        "uniko.consolidation.duration_ms",
        "Consolidation cycle duration"
    );

    describe_gauge!(
        "uniko.recall.phase1_only_pct",
        "Recalls satisfied by Phase 1 alone"
    );
    describe_histogram!(
        "uniko.recall.assembly_ms",
        "Context bundle assembly latency"
    );

    describe_counter!(
        "uniko.cortex.topic_cycles_total",
        "Total topic-detection sweeps (P6)"
    );
    describe_histogram!(
        "uniko.cortex.topic_cycle_ms",
        "Topic-detection sweep duration"
    );
    describe_counter!(
        "uniko.cortex.topics_created_total",
        "Topic nodes created by P6"
    );
    describe_counter!(
        "uniko.cortex.procedure_cycles_total",
        "Total procedure-promotion sweeps (P5)"
    );
    describe_histogram!(
        "uniko.cortex.procedure_cycle_ms",
        "Procedure-promotion sweep duration"
    );
    describe_counter!(
        "uniko.cortex.procedures_promoted_total",
        "Procedures promoted by P5"
    );
}

// ── Emission helpers ────────────────────────────────────────────────

/// Increment the ingest item counter.
pub fn emit_ingest_item() {
    counter!("uniko.ingest.items_total").increment(1);
}

/// Increment the ingest failure counter for a step.
pub fn emit_ingest_failure(step: &str) {
    counter!("uniko.ingest.items_failed", "step" => step.to_string()).increment(1);
}

/// Record ingest duration in milliseconds.
pub fn emit_ingest_duration(ms: f64) {
    histogram!("uniko.ingest.duration_ms").record(ms);
}

/// Increment the consolidation cycle counter.
pub fn emit_consolidation_cycle(agent_id: &str) {
    counter!("uniko.consolidation.cycles_total", "agent_id" => agent_id.to_string()).increment(1);
}

/// Record consolidation duration in milliseconds.
pub fn emit_consolidation_duration(ms: f64) {
    histogram!("uniko.consolidation.duration_ms").record(ms);
}

/// Increment the topic-detection (P6) cycle counter.
pub fn emit_topic_cycle(agent_id: &str) {
    counter!("uniko.cortex.topic_cycles_total", "agent_id" => agent_id.to_string()).increment(1);
}

/// Record topic-detection sweep duration in milliseconds.
pub fn emit_topic_cycle_duration(ms: f64) {
    histogram!("uniko.cortex.topic_cycle_ms").record(ms);
}

/// Increment the topics-created counter.
pub fn emit_topics_created(n: usize) {
    counter!("uniko.cortex.topics_created_total").increment(n as u64);
}

/// Increment the procedure-promotion (P5) cycle counter.
pub fn emit_procedure_cycle(agent_id: &str) {
    counter!("uniko.cortex.procedure_cycles_total", "agent_id" => agent_id.to_string())
        .increment(1);
}

/// Record procedure-promotion sweep duration in milliseconds.
pub fn emit_procedure_cycle_duration(ms: f64) {
    histogram!("uniko.cortex.procedure_cycle_ms").record(ms);
}

/// Increment the procedures-promoted counter.
pub fn emit_procedures_promoted(n: usize) {
    counter!("uniko.cortex.procedures_promoted_total").increment(n as u64);
}
