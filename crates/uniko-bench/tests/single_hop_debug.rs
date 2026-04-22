//! Debug retrieval for remaining single-hop misses.
//!
//! Run: cargo nextest run -p uniko-bench --test single_hop_debug --nocapture --run-ignored all

use std::sync::Arc;
use uni_db::ModelAliasSpec;
use uniko_memory::recall::{recall, RecallConfig};
use uniko_store::config::UnikoConfig;
use uniko_store::KnowledgeBase;

async fn load_kb() -> Arc<KnowledgeBase> {
    let ws = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap().to_path_buf();
    std::env::set_current_dir(&ws).expect("cd");
    let config = UnikoConfig {
        catalog_path: Some(ws.join("config/catalog.json")),
        schema_path: Some(ws.join("config/schema.json")),
        ..Default::default()
    };
    Arc::new(
        KnowledgeBase::open_with_xervo(ws.join("data/kb/conv-30"), config, Vec::<ModelAliasSpec>::new())
            .await.expect("open KB"),
    )
}

#[tokio::test]
#[ignore]
async fn debug_single_hop_misses() {
    let kb = load_kb().await;
    let config = RecallConfig { limit: 15, ..Default::default() };

    let misses: &[(&str, &str)] = &[
        ("How do Jon and Gina both like to destress?", "by dancing"),
        ("Why did Jon decide to start his dance studio?", "lost his job, passionate"),
        ("What is Gina's favorite style of dance?", "Contemporary"),
        ("What is Jon's favorite style of dance?", "Contemporary"),
        ("What do the dancers in the photo represent?", "performing at the festival"),
        ("What does Gina say about the dancers in the photo?", "graceful"),
        ("What did Gina design for her store?", "space, furniture, decor"),
        ("What did Jon and Gina compare their entrepreneurial journeys to?", "dancing together"),
        ("What does Jon's dance make him?", "happy"),
        ("What does Jon tell Gina he won't do?", "quit"),
        ("How does Jon feel about the opening night of his dance studio?", "excited"),
    ];

    for (question, gold) in misses {
        let bundle = recall(&kb, question, &config).await.unwrap();

        eprintln!("\nQ: {question}");
        eprintln!("Gold: {gold}");
        eprintln!("Results ({}):", bundle.items.len());

        for (i, item) in bundle.items.iter().enumerate() {
            let preview = item.content.replace('\n', " ");
            let short = if preview.len() > 120 { format!("{}...", &preview[..120]) } else { preview };
            eprintln!("  #{:2} [{:7}] {:.4} | {}", i+1, item.node_type, item.score, short);
        }
    }
}
