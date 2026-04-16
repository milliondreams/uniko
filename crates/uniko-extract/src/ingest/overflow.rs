//! Action output overflow: large outputs become Artifact nodes.

// Rust guideline compliant

use std::collections::HashMap;

use uni_db::Value;

use sha2::Digest;

use uniko_store::{KnowledgeBase, NodeId};

use super::chunking::count_tokens;

/// If `output` exceeds `threshold` tokens, store it as an Artifact.
///
/// Creates an Artifact node with `kind = "action_output"` and a
/// `PRODUCED` edge from the Action.  Returns `None` if the output
/// fits within the threshold.
///
/// # Errors
///
/// Returns a storage error if graph operations fail.
#[expect(dead_code, reason = "used when record_action tool ships in Phase 2")]
pub(crate) async fn maybe_overflow_action_output(
    kb: &KnowledgeBase,
    action_node_id: NodeId,
    output: &str,
    threshold: usize,
) -> uniko_store::Result<Option<NodeId>> {
    if count_tokens(output) <= threshold {
        return Ok(None);
    }

    // Create the overflow Artifact.
    let artifact_id = uniko_store::id::new_id();
    let hash = hex::encode(sha2::Sha256::digest(output.as_bytes()));

    let mut props = HashMap::new();
    props.insert("artifact_id".into(), Value::String(artifact_id));
    props.insert("kind".into(), Value::String("action_output".into()));
    props.insert("content".into(), Value::String(output.to_string()));
    props.insert("hash".into(), Value::String(hash));
    props.insert("size".into(), Value::Int(output.len() as i64));
    let artifact_nid = kb.create_node("Artifact", &props).await?;

    // Link: Action -[PRODUCED]-> Artifact.
    kb.create_edge("PRODUCED", action_node_id, artifact_nid, &HashMap::new())
        .await?;

    Ok(Some(artifact_nid))
}
