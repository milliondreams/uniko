//! Side-by-side fact comparison between two KBs.
//!
//! Use after running ingest twice — once with the SRL/DEP triple
//! source and once with `--extract-triples-llm-alias` — to see how
//! much cleaner the LLM triples are.  Reports per-KB Fact counts,
//! mean cluster size, and the top-N most-supported triples.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let kb_a: PathBuf = args
        .next()
        .expect("usage: compare_facts <srl_kb> <llm_kb>")
        .into();
    let kb_b: PathBuf = args
        .next()
        .expect("usage: compare_facts <srl_kb> <llm_kb>")
        .into();

    println!("\n══ A: {} ══", kb_a.display());
    let a = summarize(&kb_a).await?;

    println!("\n══ B: {} ══", kb_b.display());
    let b = summarize(&kb_b).await?;

    println!("\n══ Predicate vocabulary delta ══");
    let only_a: BTreeSet<_> = a.predicates.difference(&b.predicates).collect();
    let only_b: BTreeSet<_> = b.predicates.difference(&a.predicates).collect();
    println!(
        "  shared:        {}",
        a.predicates.intersection(&b.predicates).count()
    );
    println!(
        "  only in A:     {} (e.g. {:?})",
        only_a.len(),
        only_a.iter().take(8).collect::<Vec<_>>()
    );
    println!(
        "  only in B:     {} (e.g. {:?})",
        only_b.len(),
        only_b.iter().take(8).collect::<Vec<_>>()
    );

    Ok(())
}

struct Summary {
    predicates: BTreeSet<String>,
}

async fn summarize(path: &PathBuf) -> anyhow::Result<Summary> {
    let kb = open_kb(path, UnikoConfig::default(), &[]).await?;
    let session = kb.db().session();

    for (label, q) in [
        (
            "Facts                  ",
            "MATCH (f:Fact) RETURN count(f) AS n",
        ),
        (
            "Observations (total)   ",
            "MATCH (o:Observation) RETURN count(o) AS n",
        ),
        (
            "Observations (w/triple)",
            "MATCH (o:Observation) WHERE o.predicate IS NOT NULL RETURN count(o) AS n",
        ),
        (
            "Cycles                 ",
            "MATCH (c:ConsolidationCycle) RETURN count(c) AS n",
        ),
        (
            "Distinct predicates    ",
            "MATCH (f:Fact) RETURN count(DISTINCT f.predicate) AS n",
        ),
    ] {
        let r = session.query_with(q).fetch_all().await?;
        let n: i64 = r
            .rows()
            .first()
            .and_then(|row| row.get("n").ok())
            .unwrap_or(-1);
        println!("  {label}: {n}");
    }

    println!("  --- Top-15 most-supported facts ---");
    let r = session
        .query_with(
            "MATCH (f:Fact) RETURN f.subject AS s, f.predicate AS p, f.object AS o, \
             f.observation_count AS n \
             ORDER BY f.observation_count DESC LIMIT 15",
        )
        .fetch_all()
        .await?;
    for row in r.rows() {
        let s: String = row.get("s").unwrap_or_default();
        let p: String = row.get("p").unwrap_or_default();
        let o: String = row.get("o").unwrap_or_default();
        let n: i64 = row.get("n").unwrap_or(0);
        println!("    {n:>3}× ({s} | {p} | {o})");
    }

    let mut predicates: BTreeSet<String> = BTreeSet::new();
    let r = session
        .query_with("MATCH (f:Fact) RETURN DISTINCT f.predicate AS p")
        .fetch_all()
        .await?;
    for row in r.rows() {
        if let Ok(p) = row.get::<String>("p") {
            predicates.insert(p);
        }
    }
    let _ = HashMap::<(), ()>::new(); // silence unused-import for HashMap if it stays

    drop(session);
    if let Ok(kb_owned) = std::sync::Arc::try_unwrap(kb) {
        kb_owned.shutdown().await.ok();
    }
    Ok(Summary { predicates })
}
