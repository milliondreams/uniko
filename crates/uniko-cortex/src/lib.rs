//! # uniko-cortex — Layer 5: Higher Reasoning
//!
//! Procedure promotion, topic detection, MCTS planning, rule induction,
//! working memory traversal, and NL-to-Cypher translation.
//!
//! Depends on `uniko-store` only.  Sibling of `uniko-memory` — the
//! consolidation worker calls into cortex (P5/P6) post-cycle.

pub mod procedures;
pub mod topics;

#[doc(inline)]
pub use procedures::{
    LifecycleConfig, MatchedProcedure, PromotionReport, match_procedures, promote_procedures_once,
    record_procedure_use,
};
#[doc(inline)]
pub use topics::{TopicConfig, TopicReport, detect_topics_once, detect_topics_once_with_llm};
