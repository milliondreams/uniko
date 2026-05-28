//! # uniko-api — Public Facade
//!
//! Re-exports from the cognitive stack for downstream consumers.
//! Contains no logic.

pub mod tools;

// Wildcard re-export is intentional: this is the public facade for
// external consumers, and `uniko-cortex` owns the surface they need.
// Narrowing this list risks silently dropping items downstream relies
// on without compile-time feedback inside this workspace (no internal
// consumer uses `uniko_api`).
pub use uniko_cortex::*;
