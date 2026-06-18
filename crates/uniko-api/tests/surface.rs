//! Surface guard for the public facade.
//!
//! The point of `uniko-api` is that an end user touches intent types only.
//! This test pins the *positive* surface (the facade types must stay
//! reachable); the `compile_fail` doctests on `uniko_api::tools` pin the
//! *negative* surface (internals must stay hidden).

/// Compile-time proof the facade entry points are exported through the
/// `tools` namespace.
#[test]
fn facade_entry_points_are_exported() {
    fn assert_exists<T>() {}

    assert_exists::<uniko_api::tools::Uniko>();
    assert_exists::<uniko_api::tools::UnikoBuilder>();
    assert_exists::<uniko_api::tools::Session>();
    assert_exists::<uniko_api::tools::Turn>();
    assert_exists::<uniko_api::tools::Document>();
    assert_exists::<uniko_api::tools::PdfSource>();
    assert_exists::<uniko_api::tools::Agent>();
    assert_exists::<uniko_api::tools::LlmSpec>();
    assert_exists::<uniko_api::tools::RecallScope>();
    assert_exists::<uniko_api::tools::ContextBundle>();
    assert_exists::<uniko_api::tools::QueryOutcome>();
    // Method return / field types must be nameable.
    assert_exists::<uniko_api::tools::FactUpsert>();
    assert_exists::<uniko_api::tools::GeneratedAnswer>();
    assert_exists::<uniko_api::tools::RecallTier>();
    assert_exists::<uniko_api::tools::AtomicIngestResult>();
    assert_exists::<uniko_api::tools::ArtifactIngestResult>();
    assert_exists::<uniko_api::tools::PdfIngestResult>();
    // Logic / query surface types.
    assert_exists::<uniko_api::tools::Record>();
    assert_exists::<uniko_api::tools::Value>();
    assert_exists::<uniko_api::tools::AbductionResult>();
    assert_exists::<uniko_api::tools::AssumeBuilder<'static>>();
    // Error / id types must be nameable for handling and storing results.
    assert_exists::<uniko_api::tools::UnikoError>();
    assert_exists::<uniko_api::tools::NodeId>();
}
