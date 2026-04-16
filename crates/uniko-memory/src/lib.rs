//! # uniko-memory — Layer 4: Memory Management
//!
//! PipelineSystem orchestration, recall cascade (3-phase with coverage gating),
//! consolidation (fact derivation, contradiction, drift), and stdlib rules.
//!
//! Depends on `uniko-extract` only. This is the memory management brain.

pub mod consolidation;
pub mod pipeline;
pub mod recall;
pub mod rules;
