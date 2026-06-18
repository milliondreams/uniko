//! # uniko-pipes — Layer 2: Pipeline Infrastructure
//!
//! Generic pipeline machinery: Step trait, circuit breaker, dead-letter
//! queue, cancellation tokens, health checks, and metrics.
//!
//! Depends on `uniko-store` only. Content processing steps live in `uniko-extract`.

pub mod cancel;
pub mod circuit_breaker;
pub mod config;
pub mod content;
pub mod dead_letter;
pub mod health;
pub mod metrics;
pub mod step;
pub mod types;

// Re-exports for convenience.
#[doc(inline)]
pub use cancel::ShutdownCoordinator;
#[doc(inline)]
pub use circuit_breaker::{CircuitBreaker, CircuitState};
#[doc(inline)]
pub use config::PipelineConfig;
#[doc(inline)]
pub use content::{ContentType, Mime, MimeError, Modality, modality_for_mime};
#[doc(inline)]
pub use step::{PipelineContext, Step};
#[doc(inline)]
pub use types::{
    ConsolidationTask, IngestArtifact, IngestData, IngestMessage, IngestPdf, IngestSource,
    IngestTask, ItemResult, ObservationsReady, PdfInput, StepErrorPolicy, StepOutcome,
};
