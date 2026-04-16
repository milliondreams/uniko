//! Integration tests for graph traversal: BFS, shortest path, and PPR.

use std::collections::HashMap;
use uni_db::Value;
use uniko_store::config::UnikoConfig;
use uniko_store::storage::edges::Direction;
use uniko_store::storage::KnowledgeBase;

async fn test_kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

/// Create a chain: A -> B -> C -> D via NEXT edges.
async fn create_chain(kb: &KnowledgeBase) -> Vec<i64> {
    let mut ids = Vec::new();
    for i in 0..4u8 {
        let mut props = HashMap::new();
        props.insert(
            "message_id".into(),
            Value::String(format!("trav-{i}")),
        );
        props.insert(
            "content".into(),
            Value::String(format!("msg {i}")),
        );
        props.insert(
            "timestamp".into(),
            Value::String("2024-01-01T00:00:00Z".into()),
        );
        let nid = kb.create_node("Message", &props).await.unwrap();
        ids.push(nid);
    }
    for i in 0..3 {
        kb.create_edge("NEXT", ids[i], ids[i + 1], &HashMap::new())
            .await
            .unwrap();
    }
    ids
}

#[tokio::test]
async fn test_traverse_depth_1() {
    let kb = test_kb().await;
    let ids = create_chain(&kb).await;

    let results = kb
        .traverse(ids[0], 1, Direction::Outgoing, Some(&["NEXT"]))
        .await
        .unwrap();

    // Should find only B (depth 1 from A).
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].node_id, ids[1]);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_traverse_depth_3() {
    let kb = test_kb().await;
    let ids = create_chain(&kb).await;

    let results = kb
        .traverse(ids[0], 3, Direction::Outgoing, Some(&["NEXT"]))
        .await
        .unwrap();

    // Should find B, C, D (3 nodes reachable within depth 3).
    assert_eq!(results.len(), 3);
    let found_ids: Vec<i64> = results.iter().map(|r| r.node_id).collect();
    assert!(found_ids.contains(&ids[1]));
    assert!(found_ids.contains(&ids[2]));
    assert!(found_ids.contains(&ids[3]));

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_traverse_incoming() {
    let kb = test_kb().await;
    let ids = create_chain(&kb).await;

    // Traverse incoming from D should find C, B, A.
    let results = kb
        .traverse(ids[3], 3, Direction::Incoming, Some(&["NEXT"]))
        .await
        .unwrap();

    assert_eq!(results.len(), 3);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_shortest_path_exists() {
    let kb = test_kb().await;
    let ids = create_chain(&kb).await;

    let path = kb
        .shortest_path(ids[0], ids[3], Some(&["NEXT"]), 10)
        .await
        .unwrap();

    assert!(path.is_some());
    assert_eq!(path.unwrap().length, 3);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_shortest_path_not_exists() {
    let kb = test_kb().await;
    let ids = create_chain(&kb).await;

    // No path from D -> A (edges are directed).
    let path = kb
        .shortest_path(ids[3], ids[0], Some(&["NEXT"]), 10)
        .await
        .unwrap();

    assert!(path.is_none());

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_shortest_path_direct() {
    let kb = test_kb().await;
    let ids = create_chain(&kb).await;

    let path = kb
        .shortest_path(ids[0], ids[1], Some(&["NEXT"]), 10)
        .await
        .unwrap();

    assert!(path.is_some());
    assert_eq!(path.unwrap().length, 1);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_ppr_basic() {
    let kb = test_kb().await;
    let ids = create_chain(&kb).await;

    let result = kb
        .personalized_pagerank(&[ids[0]], 0.85, 20, 10)
        .await
        .unwrap();

    // Should produce scores for all reachable nodes.
    assert!(!result.scores.is_empty());

    // Seed node should have a high score.
    let seed_score = result
        .scores
        .iter()
        .find(|(nid, _)| *nid == ids[0])
        .map(|(_, s)| *s)
        .unwrap_or(0.0);
    assert!(seed_score > 0.0);

    kb.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_ppr_empty_seeds() {
    let kb = test_kb().await;

    let result = kb
        .personalized_pagerank(&[], 0.85, 20, 10)
        .await
        .unwrap();

    assert!(result.scores.is_empty());
    assert!(result.converged);

    kb.shutdown().await.unwrap();
}
