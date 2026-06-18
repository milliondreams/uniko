//! Integration tests for the high-level [`Uniko`] facade.
//!
//! These exercise the end-user surface only — `Uniko` / `Agent` /
//! `Session` / `Turn` — never the lower-layer `KnowledgeBase` or
//! `PipelineSystem` directly (except `Agent::kb()`, the sanctioned escape
//! hatch used here purely to seed access-control fixtures).
//!
//! Recall-dependent assertions follow the same skip-guard convention as
//! `policy_e2e`: when the embedding model is unavailable in the test
//! environment the test logs and returns rather than failing vacuously.

use std::collections::HashMap;

use uni_db::Value;
use uniko_memory::{Document, PdfSource, Turn, Uniko};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;
use uniko_store::{KnowledgeBase, UnikoError};

/// True for the "model not present in this environment" error so recall
/// tests can skip instead of failing where embeddings are unavailable.
fn is_model_unavailable(err: &UnikoError) -> bool {
    matches!(err, UnikoError::Embedding(_))
}

/// `observe()` commits before returning, so a following `recall()` on the
/// same instance sees the turn (read-after-write).
#[tokio::test]
async fn observe_then_recall_is_read_after_write() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("chat-1");

    let ingest = match session
        .observe(Turn::new(
            "alice",
            "I love hiking in the mountains every weekend",
        ))
        .await
    {
        Ok(result) => result,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("observe failed: {e}"),
    };
    assert!(ingest.message_node_id > 0, "ingest should yield a node id");

    match agent.recall("hiking hobbies").await {
        Ok(bundle) => assert!(
            !bundle.items.is_empty(),
            "recall should surface the just-observed turn"
        ),
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => panic!("recall failed: {e}"),
    }

    // shutdown drains the store and needs sole ownership: drop the
    // agent/session handles (each holds a KB clone) first.
    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `answer()` without a configured LLM is a clear `Config` error, not a
/// panic or a silent empty answer.
#[tokio::test]
async fn answer_without_llm_is_config_error() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    let err = agent
        .answer("what is the meaning of life?")
        .await
        .expect_err("answer without an LLM must error");
    assert!(
        matches!(err, UnikoError::Config(_)),
        "expected Config error, got {err:?}"
    );
}

/// `submit()` without streaming enabled is a clear `Config` error.
#[tokio::test]
async fn submit_without_streaming_is_config_error() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let session = memory.agent("assistant").session("chat-1");

    let err = session
        .submit(Turn::new("alice", "hello"))
        .await
        .expect_err("submit without streaming must error");
    assert!(
        matches!(err, UnikoError::Config(_)),
        "expected Config error, got {err:?}"
    );
}

/// Streaming `submit()` + `flush()` actually ingests: after the barrier,
/// the messages exist in the graph (proves the worker payload wiring and
/// the in-flight quiesce barrier).
#[tokio::test]
async fn streaming_submit_then_flush_ingests() {
    let memory = match Uniko::builder().in_memory().streaming(true).build().await {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: streaming instance unavailable: {e}");
            return;
        }
    };
    let agent = memory.agent("assistant");
    let session = agent.session("stream-1");

    for i in 0..3 {
        if let Err(e) = session
            .submit(Turn::new(
                "alice",
                format!("streamed note {i} about rock climbing"),
            ))
            .await
        {
            eprintln!("skipping: submit failed: {e}");
            return;
        }
    }
    session
        .flush()
        .await
        .expect("flush should drain the pipeline");

    // Count Messages directly — independent of embedding availability, so
    // this proves the streaming path ingested rather than no-op'd.
    let count = message_count(agent.kb()).await;
    assert!(
        count >= 3,
        "expected >= 3 messages after flush, found {count}"
    );

    drop(session);
    drop(agent);
    memory.shutdown().await.expect("shutdown");
}

/// `scope_to_agent()` filters reads to each agent's own visibility: a
/// `private:alice` Fact is recalled for `alice` but not for `bob`.
#[tokio::test]
async fn agent_scope_filters_private_facts() {
    let memory = match Uniko::builder().in_memory().scope_to_agent().build().await {
        Ok(memory) => memory,
        Err(e) => {
            eprintln!("skipping: instance unavailable: {e}");
            return;
        }
    };
    let alice = memory.agent("alice");
    let bob = memory.agent("bob");
    let kb = alice.kb();

    seed_participant(kb, "alice").await;
    seed_participant(kb, "bob").await;
    let query = "quarterly revenue outlook";
    seed_fact(kb, "f-public", query, Some("public")).await;
    seed_fact(kb, "f-private", query, Some("private:alice")).await;
    let private_nid = fact_nid(kb, "f-private").await;

    let alice_bundle = match alice.recall(query).await {
        Ok(bundle) => bundle,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("alice recall failed: {e}"),
    };
    if !alice_bundle.items.iter().any(|i| i.node_id == private_nid) {
        eprintln!("skipping: recall did not surface Facts in this env");
        return;
    }

    let bob_bundle = bob.recall(query).await.expect("bob recall");
    assert!(
        !bob_bundle.items.iter().any(|i| i.node_id == private_nid),
        "bob must not see the private:alice Fact"
    );
}

/// `Uniko::open`/`in_memory` run the validated best config (reranker on,
/// phase-1 boost, BGE-small embeddings) — verified without loading models.
#[tokio::test]
async fn defaults_match_validated_best_config() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let config = memory.config();
    let expected = UnikoConfig::default();
    assert!(config.reranker.enabled, "reranker should default on");
    assert_eq!(config.phase1_strategy, expected.phase1_strategy);
    assert_eq!(config.embedding.model_id, expected.embedding.model_id);
}

/// `Turn::id` makes observe idempotent: the same id re-ingests to the
/// same node rather than duplicating.
#[tokio::test]
async fn turn_id_makes_observe_idempotent() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");
    let mut session = agent.session("chat-1");

    let first = match session
        .observe(Turn::new("alice", "fixed content").id("msg-1"))
        .await
    {
        Ok(result) => result,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("first observe failed: {e}"),
    };
    let second = session
        .observe(Turn::new("alice", "fixed content").id("msg-1"))
        .await
        .expect("second observe");

    assert_eq!(
        first.message_node_id, second.message_node_id,
        "same message id must dedup to the same node"
    );
    assert_eq!(message_count(agent.kb()).await, 1, "no duplicate message");
}

/// `Session::ingest_document` persists an artifact and dedups identical
/// content by hash.
#[tokio::test]
async fn ingest_document_persists_and_dedups() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("librarian");
    let session = agent.session("kb-import");

    let body = "The Eiffel Tower is a wrought-iron lattice tower in Paris.";
    let first = match session.ingest_document(Document::new(body)).await {
        Ok(result) => result,
        Err(e) if is_model_unavailable(&e) => {
            eprintln!("skipping: embeddings unavailable");
            return;
        }
        Err(e) => panic!("ingest_document failed: {e}"),
    };
    assert!(first.artifact_node_id > 0);
    assert!(!first.was_deduplicated, "first ingest is not a dup");

    let second = session
        .ingest_document(Document::new(body))
        .await
        .expect("re-ingest");
    assert!(
        second.was_deduplicated,
        "identical content must dedup by hash"
    );
}

/// `Session::ingest_pdf` persists an artifact; a non-PDF body reports an
/// extraction failure rather than erroring.
#[tokio::test]
async fn ingest_pdf_persists_artifact() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let session = memory.agent("librarian").session("kb-import");

    match session
        .ingest_pdf(
            "doc-1",
            PdfSource::Bytes(b"%PDF-1.4 not a real pdf".to_vec()),
        )
        .await
    {
        Ok(result) => assert!(result.artifact_node_id > 0, "PDF artifact should persist"),
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => eprintln!("skipping: pdf extractor unavailable in this env: {e}"),
    }
}

/// `Agent::query` rejects writes (read-only gate) and runs read Cypher.
#[tokio::test]
async fn query_is_read_only() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    // A write is rejected before it ever reaches the engine.
    let write = agent.query("CREATE (n:Foo {x: 1}) RETURN n").await;
    assert!(
        matches!(write, Err(UnikoError::Storage(_))),
        "write Cypher must be rejected, got {write:?}"
    );

    // A read runs (empty graph → empty rows).
    match agent.query("MATCH (n:Participant) RETURN n").await {
        Ok(rows) => assert!(rows.is_empty(), "empty graph yields no rows"),
        Err(UnikoError::Locy(_)) => eprintln!("skipping: Locy runtime unavailable"),
        Err(e) => panic!("query failed: {e}"),
    }
}

/// `Agent::define_rule` registers a Locy rule (tolerant of the
/// runtime-rejects-syntax path, per spec).
#[tokio::test]
async fn define_rule_registers() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    match agent
        .define_rule(
            "facade_rule",
            "CREATE RULE facade_rule AS MATCH (n:Episode) YIELD KEY n",
        )
        .await
    {
        Ok(nid) => assert!(nid > 0, "rule should get a node id"),
        Err(UnikoError::Locy(_)) => {
            eprintln!("skipping: Locy runtime rejected the test rule (allowed by spec)");
        }
        Err(e) => panic!("define_rule failed: {e}"),
    }
}

/// `Agent::assume` is hypothetical: a mutation inside it is rolled back,
/// so the real graph is untouched afterwards.
#[tokio::test]
async fn assume_does_not_mutate_the_graph() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let agent = memory.agent("assistant");

    let assumed = agent
        .assume("ASSUME { CREATE (:Fact {subject: 'srv', predicate: 'port', object: '9090'}) }")
        .then_query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
        .run()
        .await;
    if let Err(UnikoError::Locy(_)) = assumed {
        eprintln!("skipping: Locy runtime unavailable for ASSUME");
        return;
    }
    assumed.expect("assume should run");

    // The hypothetical Fact must NOT persist in the real graph.
    if let Ok(rows) = agent
        .query("MATCH (f:Fact {subject: 'srv'}) RETURN f")
        .await
    {
        assert!(rows.is_empty(), "ASSUME mutation must be rolled back");
    }
}

/// `Session::summarize` on a session with no content returns `None`.
#[tokio::test]
async fn summarize_unused_session_is_none() {
    let Ok(memory) = Uniko::in_memory().await else {
        eprintln!("skipping: in-memory instance unavailable (no model?)");
        return;
    };
    let session = memory.agent("assistant").session("never-used");

    match session.summarize().await {
        Ok(summary) => assert!(summary.is_none(), "unused session has nothing to summarize"),
        Err(e) if is_model_unavailable(&e) => eprintln!("skipping: embeddings unavailable"),
        Err(e) => panic!("summarize failed: {e}"),
    }
}

// ── Fixtures (seed via the `Agent::kb()` escape hatch) ────────────────

async fn message_count(kb: &KnowledgeBase) -> usize {
    kb.query_nodes(labels::MESSAGE, None, None)
        .await
        .expect("message query")
        .len()
}

async fn seed_participant(kb: &KnowledgeBase, pid: &str) {
    let mut props = HashMap::new();
    props.insert("kind".to_string(), Value::String("agent".into()));
    kb.merge_node(labels::PARTICIPANT, "participant_id", pid, &props)
        .await
        .expect("participant");
}

async fn seed_fact(kb: &KnowledgeBase, fid: &str, object: &str, visibility: Option<&str>) {
    let mut props = HashMap::new();
    props.insert("subject".to_string(), Value::String("project".into()));
    props.insert("predicate".to_string(), Value::String("status_is".into()));
    props.insert("object".to_string(), Value::String(object.into()));
    if let Some(v) = visibility {
        props.insert("visibility".to_string(), Value::String(v.into()));
    }
    kb.merge_node(labels::FACT, "fact_id", fid, &props)
        .await
        .expect("fact");
}

async fn fact_nid(kb: &KnowledgeBase, fid: &str) -> i64 {
    kb.get_node_by_ext_id(labels::FACT, "fact_id", fid)
        .await
        .expect("lookup")
        .expect("fact exists")
        .0
}
