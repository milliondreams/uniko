//! Repro for slow `UNWIND ... MATCH WHERE id(n)=u.nid SET ...` in uni-db.
//!
//! Designed to drop into `uni-db/crates/uni/examples/` and run with:
//!   cargo run --release --example uni_db_update_slow
//!
//! What it does:
//!   1. In-memory uni-db, single Entity label, bulk-inserts N=4000 nodes.
//!   2. Runs the UPDATE Cypher at varying batch sizes (1, 3, 10, 100, 1000).
//!   3. For each: reports wall, exec (from QueryMetrics), and per-operator
//!      stats from .profile().
//!
//! What the output shows (on our machine):
//!   - Per-row cost is non-monotonic: 1.9 ms at batch=1, 12 ms at batch=3
//!     (17× regression), then amortises down to 3.8 ms at batch=1000.
//!   - In ProfileOutput, MutationSetExec and GraphScanExec both report
//!     time=0 ms, hiding the dominant ops. Only FilterExec & HashJoinExec
//!     have visible time. Accounted op time at batch=1000 is ~67 ms of
//!     a 3770 ms run — 98% of the cost is in operators that don't expose
//!     timing.

// mimalloc as global allocator — measured ~3x throughput on uni-db's
// concurrent_mutations benchmark (uni-db commit 65399a2b).
#[global_allocator]
static GLOBAL: uni_db::MiMalloc = uni_db::MiMalloc;

use std::collections::HashMap;
use std::time::Instant;

use uni_db::{Uni, Value};

use uniko_bench::now_value;

const UPDATE_CYPHER: &str = "\
    UNWIND $updates AS u \
    MATCH (n:Entity) WHERE id(n) = u.nid \
    SET n.frequency = u.new_frequency, \
        n.last_seen = $now, \
        n.confidence = u.new_confidence";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let tmp = tempfile::tempdir()?;
    let db = Uni::open(tmp.path().to_string_lossy().to_string())
        .build()
        .await?;
    let session = db.session();

    // ── Schema
    {
        let tx = session.tx().await?;
        tx.execute(
            "CREATE LABEL Entity (\
               entity_id STRING NOT NULL, \
               name STRING NOT NULL, \
               frequency INT, \
               last_seen DATETIME, \
               confidence FLOAT)",
        )
        .await?;
        // Production-like schema noise: 25 sibling labels populated with
        // some data, so the planner has to deal with a non-trivial catalog.
        for i in 0..25 {
            tx.execute(&format!("CREATE LABEL Sibling{i} (x INT)"))
                .await?;
        }
        tx.commit().await?;
        let tx = session.tx().await?;
        for i in 0..25 {
            let label = format!("Sibling{i}");
            let mut rows = Vec::with_capacity(100);
            for j in 0..100 {
                let mut h = HashMap::new();
                h.insert("x".into(), Value::Int(j));
                rows.push(h);
            }
            tx.bulk_insert_vertices(&label, rows).await?;
        }
        tx.commit().await?;
    }

    // ── Insert 4000 entities (this is fast; not what we're measuring)
    const N: usize = 4000;
    let all_vids: Vec<i64> = {
        let tx = session.tx().await?;
        let mut rows: Vec<HashMap<String, Value>> = Vec::with_capacity(N);
        for i in 0..N {
            let mut h = HashMap::new();
            h.insert("entity_id".into(), Value::String(format!("e:{i}")));
            h.insert("name".into(), Value::String(format!("entity_{i}")));
            h.insert("frequency".into(), Value::Int(1));
            h.insert("last_seen".into(), now_value());
            h.insert("confidence".into(), Value::Float(0.5));
            rows.push(h);
        }
        let vids = tx.bulk_insert_vertices("Entity", rows).await?;
        tx.commit().await?;
        vids.iter().map(|v| v.as_u64() as i64).collect()
    };
    println!("Setup: {N} Entity nodes inserted.\n");

    // ── Measure UPDATE wall + exec_time across batch sizes
    println!("## Wall + exec_time vs batch size (median of 5 iters)");
    println!(
        "{:>6} {:>10} {:>10} {:>10}",
        "batch", "wall_ms", "exec_ms", "ms/row"
    );
    for &batch in &[1usize, 3, 10, 100, 1000] {
        let updates: Vec<Value> = all_vids[..batch]
            .iter()
            .enumerate()
            .map(|(i, &vid)| {
                let mut m = HashMap::new();
                m.insert("nid".into(), Value::Int(vid));
                m.insert("new_frequency".into(), Value::Int((i as i64) + 2));
                m.insert("new_confidence".into(), Value::Float(0.7));
                Value::Map(m)
            })
            .collect();

        let mut walls = Vec::new();
        let mut execs = Vec::new();
        for _ in 0..5 {
            let tx = session.tx().await?;
            let t = Instant::now();
            let result = tx
                .execute_with(UPDATE_CYPHER)
                .param("updates", Value::List(updates.clone()))
                .param("now", now_value())
                .run()
                .await?;
            walls.push(t.elapsed().as_secs_f64() * 1000.0);
            execs.push(result.metrics().exec_time.as_secs_f64() * 1000.0);
            tx.commit().await?;
        }
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let w = med(walls);
        let e = med(execs);
        println!(
            "{:>6} {:>10.2} {:>10.2} {:>10.3}",
            batch,
            w,
            e,
            w / batch as f64
        );
    }

    // ── Per-operator breakdown via .profile()
    println!("\n## .profile() per-op breakdown");
    for &batch in &[3usize, 1000] {
        let updates: Vec<Value> = all_vids[..batch]
            .iter()
            .enumerate()
            .map(|(i, &vid)| {
                let mut m = HashMap::new();
                m.insert("nid".into(), Value::Int(vid));
                m.insert("new_frequency".into(), Value::Int((i as i64) + 2));
                m.insert("new_confidence".into(), Value::Float(0.7));
                Value::Map(m)
            })
            .collect();
        let tx = session.tx().await?;
        let (_res, profile) = tx
            .execute_with(UPDATE_CYPHER)
            .param("updates", Value::List(updates))
            .param("now", now_value())
            .profile()
            .await?;
        println!(
            "--- batch={batch} profile total={} ms peak_mem={} B ---",
            profile.total_time_ms, profile.peak_memory_bytes
        );
        let mut accounted = 0.0_f64;
        for (i, op) in profile.runtime_stats.iter().enumerate() {
            accounted += op.time_ms;
            println!(
                "  [{i}] {:<24} rows={:>6}  time={:>9.3} ms",
                op.operator, op.actual_rows, op.time_ms
            );
        }
        let total = profile.total_time_ms as f64;
        println!(
            "  → accounted op time = {:.2} ms of {:.0} ms profile total ({:.1}% unaccounted)",
            accounted,
            total,
            100.0 * (1.0 - accounted / total)
        );
    }

    Ok(())
}
