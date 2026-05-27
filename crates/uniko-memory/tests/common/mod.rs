//! Shared fixtures for uniko-memory integration tests.
//!
//! Exposes [`kb_with_chat`] which builds an in-memory KB *with*
//! `chat/default` registered as the OpenAI `gpt-5-nano` chat model.
//! Tests that exercise LLM-dependent code paths must use this
//! constructor; tests that only exercise local code can keep using
//! [`KnowledgeBase::in_memory`].
//!
//! `require_openai_key` is the single chokepoint that lets a test
//! announce "I need a real LLM" — when `OPENAI_API_KEY` is unset
//! we panic with a clear message so a missing key fails loud
//! instead of letting the test pass vacuously.

use uni_db::{ModelAliasSpec, ModelTask, WarmupPolicy};
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

/// Alias the NL→Cypher translator (and other generative paths) use
/// for the chat model.  Matches the default in
/// `uniko_memory::nl_to_cypher::translate`.
pub const CHAT_ALIAS: &str = "chat/default";

/// Default OpenAI model id used by tests.
pub const DEFAULT_CHAT_MODEL: &str = "gpt-5-nano";

/// Fail loudly if `OPENAI_API_KEY` is not set.
///
/// Use this at the top of every test that calls a remote chat model;
/// it converts a missing-env-var into an unambiguous test failure
/// instead of a silent skip.
pub fn require_openai_key() {
    if std::env::var("OPENAI_API_KEY").is_err() {
        panic!(
            "OPENAI_API_KEY is not set. \
             This test exercises gpt-5-nano via Xervo and cannot \
             pass without a real API key.  Set OPENAI_API_KEY in the \
             environment or skip the test."
        );
    }
}

/// Build a `ModelAliasSpec` for `gpt-5-nano` on OpenAI under
/// `chat/default`.  Default temperature 0 so generations are
/// deterministic for cache-equality assertions.
#[must_use]
pub fn chat_alias_spec() -> ModelAliasSpec {
    ModelAliasSpec {
        alias: CHAT_ALIAS.to_string(),
        task: ModelTask::Generate,
        provider_id: "remote/openai".to_string(),
        model_id: DEFAULT_CHAT_MODEL.to_string(),
        revision: None,
        warmup: WarmupPolicy::Lazy,
        required: false,
        timeout: None,
        load_timeout: None,
        retry: None,
        options: serde_json::json!({
            "api_key_env": "OPENAI_API_KEY",
        }),
    }
}

/// Construct an in-memory KB with the default embed + NLP catalog
/// plus a registered `chat/default` alias pointing at gpt-5-nano.
///
/// # Panics
///
/// Panics if `OPENAI_API_KEY` is missing or if the underlying KB
/// cannot be constructed.
pub async fn kb_with_chat() -> KnowledgeBase {
    require_openai_key();
    KnowledgeBase::in_memory_with_xervo(UnikoConfig::default(), vec![chat_alias_spec()])
        .await
        .expect("in-memory KB with chat/default")
}

/// Construct an in-memory KB without a chat alias.  Equivalent to
/// `KnowledgeBase::in_memory(UnikoConfig::default())` — kept here so
/// tests have one canonical constructor per fixture flavour.
pub async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}
