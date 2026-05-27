//! Candidate contradiction detection between observations.
//!
//! # Why This Is a Stub
//!
//! The original plan was to flag observations where cosine similarity
//! between embeddings is < 0.3 for same-subject pairs.  This approach
//! is **unreliable** because low embedding similarity does NOT imply
//! contradiction:
//!
//! - "Caroline works at the hospital" vs "Caroline likes jazz"
//!   → low similarity, but these are **unrelated**, not contradictory.
//! - "Caroline works at the hospital" vs "Caroline works night shifts"
//!   → moderate similarity, but these are **compatible**.
//! - "Caroline works at the hospital" vs "Caroline works at the law firm"
//!   → THIS is a real contradiction (same relation slot, different value).
//!
//! Embedding similarity alone cannot distinguish "unrelated" from
//! "contradictory."  Real contradiction detection requires:
//!
//! 1. **Predicate extraction** — parse observations into structured
//!    `(subject, predicate, object)` triples.  E.g., "Caroline works at
//!    the hospital" → `(Caroline, works_at, hospital)`.  This is ADR-3
//!    from the spec (verb-frame patterns).
//!
//! 2. **Same-slot comparison** — compare only observations with the
//!    same normalized predicate (e.g., both `works_at`).  Different
//!    predicates are unrelated, not contradictory.
//!
//! 3. **Value conflict detection** — within the same slot, check for
//!    mutually exclusive values: `works_at: hospital` vs
//!    `works_at: law_firm`.  Even then, mark as "candidate" because
//!    a person can hold multiple jobs unless the schema constrains
//!    the relation to single-valued.
//!
//! 4. **Embedding as secondary signal** — cosine similarity can be
//!    used as a tie-breaker or confidence booster, but NOT as the
//!    primary contradiction test.
//!
//! # When This Ships
//!
//! Proper contradiction detection ships with **P4 (consolidation)**
//! in Phase 2, when:
//! - Facts have structured `(subject, predicate, object)` triples
//! - ADR-3 predicate extraction is implemented
//! - BTIC temporal intervals track fact validity
//! - The `contradiction_detector` stdlib Locy rule operates on Facts
//!
//! P3's role is to **extract observations** — contradiction resolution
//! is P4's responsibility by design (spec F38).

use super::types::ContradictionFlag;

/// Check a new observation against existing ones for contradictions.
///
/// **Currently returns an empty vector.**  Proper contradiction
/// detection requires predicate-aware comparison (same subject +
/// same relation slot + different value), which ships with P4
/// consolidation.  See module-level docs for the full rationale.
///
/// When implemented, this function should:
/// 1. Query observations with the same subject.
/// 2. Filter to same normalized predicate / relation slot.
/// 3. Compare values for mutual exclusivity.
/// 4. Optionally use embedding similarity as a secondary signal.
/// 5. Return `ContradictionFlag` only for likely conflicts.
pub async fn check_contradictions(
    _kb: &uniko_store::KnowledgeBase,
    _observation_id: uniko_store::NodeId,
    _subject: &str,
) -> Vec<ContradictionFlag> {
    Vec::new()
}
