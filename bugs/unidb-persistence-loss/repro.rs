//! Reproduction: nodes created late in a persistent KB's write sequence
//! are silently dropped on shutdown + reopen.
//!
//! ## Summary
//!
//! When many writes happen early in a persistent uni-db session and a
//! small number of writes follow, the late writes return `Ok` and
//! `tx.commit()` succeeds but the rows are absent after `shutdown()`
//! and `Uni::open()`-ing the same path.
//!
//! ## Observed in uniko2
//!
//! - 415 Fact upserts during consolidation persist correctly across shutdown.
//! - 1 ConsolidationCycle node + ~100 PROCESSED edges created at the
//!   end of the same consolidation call (immediately before shutdown)
//!   do NOT persist.
//! - 1 Participant node + ~200 Episode nodes created in a subsequent
//!   phase, also before shutdown, also do NOT persist.
//!
//! ## Build & run
//!
//!     cd bugs/unidb-persistence-loss
//!     cargo run --release
//!
//! Each scenario reports PASS / FAIL.  A failure means the late writes
//! were lost.

use std::error::Error;

use tempfile::TempDir;
use uni_db::{
    DataType, IndexType, ScalarType, Uni, Value, VectorAlgo, VectorIndexCfg, VectorMetric,
};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut failures = 0usize;

    if std::env::var("ONLY_LABEL_BUG").is_ok() {
        println!("\n=== Scenario 8 only ===");
        if !label_match_vs_id_match().await? {
            failures += 1;
        }
        println!("\n=== Scenario 9 only ===");
        if !label_match_with_vector_index().await? {
            failures += 1;
        }
        println!("\n=== Scenario 10 only ===");
        if !string_into_datetime_column().await? {
            failures += 1;
        }
        println!("\n--- {failures} failing scenario(s) ---");
        return Ok(());
    }

    println!("\n=== Scenario 0: minimal (no indexes) — 1 CREATE then shutdown ===");
    if !minimal_create().await? {
        failures += 1;
    }

    println!("\n=== Scenario 0a: same as 0 but WITH a Hash index on the key ===");
    if !minimal_with_hash_index().await? {
        failures += 1;
    }

    println!("\n=== Scenario 0b: minimal with Hash index, MERGE upsert ===");
    if !minimal_merge().await? {
        failures += 1;
    }

    println!("\n=== Scenario 0c: 0a but with a BTree index instead of Hash ===");
    if !minimal_with_btree_index().await? {
        failures += 1;
    }

    println!("\n=== Scenario 0d: 0a but property called 'name' not 'ext_id' ===");
    if !minimal_hash_on_name().await? {
        failures += 1;
    }

    println!("\n=== Scenario 0e: Hash on String, BUT a second property exists ===");
    if !minimal_two_props_hash().await? {
        failures += 1;
    }

    println!("\n=== Scenario 0f: Hash on Int64 (not String) ===");
    if !minimal_hash_int64().await? {
        failures += 1;
    }

    println!("\n=== Scenario 1: single late write, no preceding bulk ===");
    if !baseline_single_late().await? {
        failures += 1;
    }

    println!("\n=== Scenario 2: 400 bulk writes, then 1 late write + edges ===");
    if !bulk_then_late().await? {
        failures += 1;
    }

    println!("\n=== Scenario 3: 1 late write first, then 400 bulk writes ===");
    if !late_then_bulk().await? {
        failures += 1;
    }

    println!("\n=== Scenario 4: bulk then late, with explicit flush() ===");
    if !explicit_flush().await? {
        failures += 1;
    }

    println!("\n=== Scenario 5: bulk via MERGE, late via MERGE ===");
    if !merge_path().await? {
        failures += 1;
    }

    println!("\n=== Scenario 6: bulk, late, second-session re-open without shutdown ===");
    if !no_shutdown_call().await? {
        failures += 1;
    }

    println!("\n=== Scenario 7: minimal — 1 CREATE then shutdown then reopen ===");
    if !minimal_create().await? {
        failures += 1;
    }

    println!("\n=== Scenario 8: label-anchored MATCH misses vertices visible to unlabeled MATCH ===");
    if !label_match_vs_id_match().await? {
        failures += 1;
    }

    println!("\n=== Scenario 9: scenario-8 + Vector property + vector index (Episode-like) ===");
    if !label_match_with_vector_index().await? {
        failures += 1;
    }

    println!("\n=== Scenario 10: String value into DataType::DateTime non-nullable column ===");
    if !string_into_datetime_column().await? {
        failures += 1;
    }

    println!("\n--- {failures} failing scenario(s) ---");
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}

async fn apply_schema(db: &Uni) -> Result<(), Box<dyn Error>> {
    // Mirror uniko-store property naming exactly: `<label>_id`, not the
    // bare `ext_id` literal.  The literal `ext_id` collides with a
    // Lance-internal column and trips an unrelated flush() failure
    // (see scenarios 0a-0e in this repro for the isolated case).
    db.schema()
        .label("Bulk")
        .property("bulk_id", DataType::String)
        .property_nullable("payload", DataType::String)
        .index("bulk_id", IndexType::Scalar(ScalarType::Hash))
        .done()
        .label("Late")
        .property("cycle_id", DataType::String)
        .property_nullable("agent_id", DataType::String)
        .property_nullable("created_count", DataType::Int64)
        .index("cycle_id", IndexType::Scalar(ScalarType::Hash))
        .done()
        .edge_type("LINKED", &["Late"], &["Bulk"])
        .done()
        .apply()
        .await?;
    Ok(())
}

async fn count(db: &Uni, cypher: &str) -> Result<i64, Box<dyn Error>> {
    let r = db.session().query_with(cypher).fetch_all().await?;
    Ok(r.rows().first().and_then(|row| row.get::<i64>("c").ok()).unwrap_or(-1))
}

/// Open a persistent KB the same way uniko-store does: explicitly empty
/// `xervo_catalog`.  Whether this matters durability-wise is itself one
/// of the questions this repro answers.
async fn open_db(path: &str) -> Result<Uni, Box<dyn Error>> {
    Ok(Uni::open(path).xervo_catalog(vec![]).build().await?)
}

/// Scenario 1: trivial baseline — one Late write, nothing else.
///
/// Note: schema is *not* re-applied on reopen.  If a re-apply truncates
/// or rebuilds tables, that is also a bug but a separate one — we want
/// to isolate the durability question first.
async fn baseline_single_late() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        let tx = db.session().tx().await?;
        tx.execute_with(
            "CREATE (n:Late {cycle_id: 'late-1', agent_id: 'agent', created_count: 0})",
        )
        .run()
        .await?;
        tx.commit().await?;
        db.shutdown().await?;
    }

    let db = open_db(&path).await?;
    // No schema re-apply — the schema lives in the store.
    let post = count(&db, "MATCH (n:Late) RETURN count(n) AS c").await?;
    db.shutdown().await?;

    let pass = post == 1;
    println!(
        "  Late after reopen: {post}  ({})",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

/// Scenario 2: the failing case — bulk first, then a late audit node
/// + linked edges, then shutdown.  Mirrors uniko-store's consolidation.
async fn bulk_then_late() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();
    let bulk_count = 400usize;
    let mut bulk_vids: Vec<i64> = Vec::with_capacity(bulk_count);

    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        let session = db.session();

        // 400 individual Bulk creations, each in its own tx (mirrors
        // upsert_fact_by_triple loop).
        for i in 0..bulk_count {
            let tx = session.tx().await?;
            let r = tx
                .query_with(
                    "CREATE (n:Bulk {bulk_id: $ext, payload: 'p'}) RETURN id(n) AS vid",
                )
                .param("ext", Value::String(format!("bulk-{i}")))
                .fetch_all()
                .await?;
            let vid: i64 = r.rows().first().unwrap().get("vid")?;
            bulk_vids.push(vid);
            tx.commit().await?;
        }

        // One Late node + LINKED edges to every Bulk node.
        let tx = session.tx().await?;
        let r = tx
            .query_with(
                "CREATE (c:Late {cycle_id: 'cycle-1', agent_id: 'agent', created_count: $n}) \
                 RETURN id(c) AS vid",
            )
            .param("n", Value::Int(bulk_count as i64))
            .fetch_all()
            .await?;
        let cycle_vid: i64 = r.rows().first().unwrap().get("vid")?;

        for &bvid in &bulk_vids {
            tx.execute_with(
                "MATCH (c:Late), (b:Bulk) WHERE id(c) = $cv AND id(b) = $bv \
                 CREATE (c)-[:LINKED]->(b)",
            )
            .param("cv", cycle_vid)
            .param("bv", bvid)
            .run()
            .await?;
        }
        tx.commit().await?;

        // In-session visibility (sanity check the writes happened).
        let pre_bulk = count(&db, "MATCH (b:Bulk) RETURN count(b) AS c").await?;
        let pre_late = count(&db, "MATCH (c:Late) RETURN count(c) AS c").await?;
        let pre_edges =
            count(&db, "MATCH ()-[r:LINKED]->() RETURN count(r) AS c").await?;
        println!("  pre-shutdown: bulk={pre_bulk} late={pre_late} edges={pre_edges}");

        db.shutdown().await?;
    }

    // Reopen and inspect.
    let db = open_db(&path).await?;
    // No schema re-apply on reopen.
    let post_bulk = count(&db, "MATCH (b:Bulk) RETURN count(b) AS c").await?;
    let post_late = count(&db, "MATCH (c:Late) RETURN count(c) AS c").await?;
    let post_edges =
        count(&db, "MATCH ()-[r:LINKED]->() RETURN count(r) AS c").await?;
    db.shutdown().await?;

    let pass = post_bulk as usize == bulk_count
        && post_late == 1
        && post_edges as usize == bulk_count;
    println!(
        "  post-reopen:   bulk={post_bulk} late={post_late} edges={post_edges}  ({})",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

/// Scenario 3: invert the order — late write first, then bulk.
///
/// If position-in-stream matters, this should pass even when scenario
/// 2 fails.  Helps triangulate the cause.
async fn late_then_bulk() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        let session = db.session();

        let tx = session.tx().await?;
        tx.execute_with(
            "CREATE (c:Late {cycle_id: 'cycle-early', agent_id: 'agent', created_count: 0})",
        )
        .run()
        .await?;
        tx.commit().await?;

        for i in 0..400usize {
            let tx = session.tx().await?;
            tx.execute_with("CREATE (n:Bulk {bulk_id: $ext, payload: 'p'})")
                .param("ext", Value::String(format!("bulk-{i}")))
                .run()
                .await?;
            tx.commit().await?;
        }

        db.shutdown().await?;
    }

    let db = open_db(&path).await?;
    // No schema re-apply on reopen.
    let post_late = count(&db, "MATCH (c:Late) RETURN count(c) AS c").await?;
    let post_bulk = count(&db, "MATCH (b:Bulk) RETURN count(b) AS c").await?;
    db.shutdown().await?;
    let pass = post_late == 1 && post_bulk == 400;
    println!(
        "  post-reopen: late={post_late} bulk={post_bulk}  ({})",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

/// Scenario 4: bulk + late, with explicit `flush()` before shutdown.
async fn explicit_flush() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        let session = db.session();
        for i in 0..400usize {
            let tx = session.tx().await?;
            tx.execute_with("CREATE (n:Bulk {bulk_id: $ext, payload: 'p'})")
                .param("ext", Value::String(format!("bulk-{i}")))
                .run()
                .await?;
            tx.commit().await?;
        }
        let tx = session.tx().await?;
        tx.execute_with(
            "CREATE (c:Late {cycle_id: 'cycle-1', agent_id: 'agent', created_count: 0})",
        )
        .run()
        .await?;
        tx.commit().await?;

        if let Err(e) = db.flush().await { println!("  flush errored: {e}  (FAIL)"); return Ok(false); }
        db.shutdown().await?;
    }

    let db = open_db(&path).await?;
    // No schema re-apply on reopen.
    let post_late = count(&db, "MATCH (c:Late) RETURN count(c) AS c").await?;
    db.shutdown().await?;
    let pass = post_late == 1;
    println!(
        "  post-reopen: late={post_late}  ({})",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

/// Scenario 5: use `MERGE` (the codepath uniko-store's `merge_node`
/// uses internally) rather than `CREATE`.
async fn merge_path() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        let session = db.session();
        for i in 0..400usize {
            let tx = session.tx().await?;
            tx.execute_with(
                "MERGE (n:Bulk {bulk_id: $ext}) ON CREATE SET n.payload = 'p'",
            )
            .param("ext", Value::String(format!("bulk-{i}")))
            .run()
            .await?;
            tx.commit().await?;
        }
        let tx = session.tx().await?;
        tx.execute_with(
            "MERGE (c:Late {cycle_id: 'cycle-1'}) \
             ON CREATE SET c.agent_id = 'agent', c.created_count = 400",
        )
        .run()
        .await?;
        tx.commit().await?;
        db.shutdown().await?;
    }

    let db = open_db(&path).await?;
    // No schema re-apply on reopen.
    let post_late = count(&db, "MATCH (c:Late) RETURN count(c) AS c").await?;
    let post_bulk = count(&db, "MATCH (b:Bulk) RETURN count(b) AS c").await?;
    db.shutdown().await?;
    let pass = post_late == 1 && post_bulk == 400;
    println!(
        "  post-reopen: late={post_late} bulk={post_bulk}  ({})",
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(pass)
}

/// Scenario 7: bare-minimum repro — single label, single property, no
/// indexes, single CREATE, shutdown, reopen, count.
async fn minimal_create() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("Tiny")
            .property("x", DataType::String)
            .done()
            .apply()
            .await?;
        let tx = db.session().tx().await?;
        tx.execute_with("CREATE (n:Tiny {x: 'hello'})").run().await?;
        tx.commit().await?;
        println!(
            "  pre-shutdown: tiny={}",
            count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?
        );
        if let Err(e) = db.flush().await { println!("  flush errored: {e}  (FAIL)"); return Ok(false); }
        db.shutdown().await?;
    }

    let db = open_db(&path).await?;
    let n = count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?;
    db.shutdown().await?;
    println!(
        "  post-reopen:  tiny={n}  ({})",
        if n == 1 { "PASS" } else { "FAIL" }
    );
    Ok(n == 1)
}

/// Scenario 0a: same as 0 but the property has a Hash index.  If this
/// fails while 0 passes, the Hash index is the trigger.
async fn minimal_with_hash_index() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("Tiny")
            .property("ext_id", DataType::String)
            .index("ext_id", IndexType::Scalar(ScalarType::Hash))
            .done()
            .apply()
            .await?;
        let tx = db.session().tx().await?;
        tx.execute_with("CREATE (n:Tiny {ext_id: 'hello'})")
            .run()
            .await?;
        tx.commit().await?;
        println!(
            "  pre-shutdown: tiny={}",
            count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?
        );
        if let Err(e) = db.flush().await {
            println!("  flush errored: {e}  (FAIL)");
            return Ok(false);
        }
        if let Err(e) = db.shutdown().await {
            println!("  shutdown errored: {e}  (FAIL)");
            return Ok(false);
        }
    }

    match open_db(&path).await {
        Ok(db) => {
            let n = count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?;
            db.shutdown().await?;
            let pass = n == 1;
            println!(
                "  post-reopen:  tiny={n}  ({})",
                if pass { "PASS" } else { "FAIL" }
            );
            Ok(pass)
        }
        Err(e) => {
            println!("  reopen errored: {e}  (FAIL)");
            Ok(false)
        }
    }
}

/// Scenario 0b: minimal with Hash index, MERGE upsert.  Mirrors how
/// uniko-store's `merge_node` writes.  Useful if MERGE has a different
/// codepath than raw CREATE.
async fn minimal_merge() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("Tiny")
            .property("ext_id", DataType::String)
            .index("ext_id", IndexType::Scalar(ScalarType::Hash))
            .done()
            .apply()
            .await?;
        let tx = db.session().tx().await?;
        tx.execute_with("MERGE (n:Tiny {ext_id: 'hello'})")
            .run()
            .await?;
        tx.commit().await?;
        if let Err(e) = db.flush().await { println!("  flush errored: {e}  (FAIL)"); return Ok(false); }
        db.shutdown().await?;
    }

    match open_db(&path).await {
        Ok(db) => {
            let n = count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?;
            db.shutdown().await?;
            let pass = n == 1;
            println!(
                "  post-reopen:  tiny={n}  ({})",
                if pass { "PASS" } else { "FAIL" }
            );
            Ok(pass)
        }
        Err(e) => {
            println!("  reopen errored: {e}  (FAIL)");
            Ok(false)
        }
    }
}

/// Scenario 0c: same as 0a but BTree index instead of Hash.
async fn minimal_with_btree_index() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("Tiny")
            .property("ext_id", DataType::String)
            .index("ext_id", IndexType::Scalar(ScalarType::BTree))
            .done()
            .apply()
            .await?;
        let tx = db.session().tx().await?;
        tx.execute_with("CREATE (n:Tiny {ext_id: 'hello'})")
            .run()
            .await?;
        tx.commit().await?;
        if let Err(e) = db.flush().await { println!("  flush errored: {e}  (FAIL)"); return Ok(false); }
        db.shutdown().await?;
    }

    match open_db(&path).await {
        Ok(db) => {
            let n = count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?;
            db.shutdown().await?;
            let pass = n == 1;
            println!(
                "  post-reopen:  tiny={n}  ({})",
                if pass { "PASS" } else { "FAIL" }
            );
            Ok(pass)
        }
        Err(e) => {
            println!("  reopen errored: {e}  (FAIL)");
            Ok(false)
        }
    }
}

/// Scenario 0d: 0a but property called `name` (not `ext_id`).  If the
/// `ext_id` field name itself is special in lance (reserved?), this
/// passes; if the bug is generic, this also fails.
async fn minimal_hash_on_name() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("Tiny")
            .property("name", DataType::String)
            .index("name", IndexType::Scalar(ScalarType::Hash))
            .done()
            .apply()
            .await?;
        let tx = db.session().tx().await?;
        tx.execute_with("CREATE (n:Tiny {name: 'alice'})")
            .run()
            .await?;
        tx.commit().await?;
        if let Err(e) = db.flush().await { println!("  flush errored: {e}  (FAIL)"); return Ok(false); }
        db.shutdown().await?;
    }

    match open_db(&path).await {
        Ok(db) => {
            let n = count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?;
            db.shutdown().await?;
            let pass = n == 1;
            println!(
                "  post-reopen:  tiny={n}  ({})",
                if pass { "PASS" } else { "FAIL" }
            );
            Ok(pass)
        }
        Err(e) => {
            println!("  reopen errored: {e}  (FAIL)");
            Ok(false)
        }
    }
}

/// Scenario 0e: 0a but with a second property alongside the indexed one.
async fn minimal_two_props_hash() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("Tiny")
            .property("ext_id", DataType::String)
            .property("payload", DataType::String)
            .index("ext_id", IndexType::Scalar(ScalarType::Hash))
            .done()
            .apply()
            .await?;
        let tx = db.session().tx().await?;
        tx.execute_with("CREATE (n:Tiny {ext_id: 'a', payload: 'b'})")
            .run()
            .await?;
        tx.commit().await?;
        if let Err(e) = db.flush().await {
            println!("  flush errored: {e}  (FAIL)");
            return Ok(false);
        }
        db.shutdown().await?;
    }

    match open_db(&path).await {
        Ok(db) => {
            let n = count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?;
            db.shutdown().await?;
            let pass = n == 1;
            println!(
                "  post-reopen:  tiny={n}  ({})",
                if pass { "PASS" } else { "FAIL" }
            );
            Ok(pass)
        }
        Err(e) => {
            println!("  reopen errored: {e}  (FAIL)");
            Ok(false)
        }
    }
}

/// Scenario 0f: Hash on an Int64 column.  If this passes, the bug is
/// specific to Hash-on-String.
async fn minimal_hash_int64() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("Tiny")
            .property("k", DataType::Int64)
            .index("k", IndexType::Scalar(ScalarType::Hash))
            .done()
            .apply()
            .await?;
        let tx = db.session().tx().await?;
        tx.execute_with("CREATE (n:Tiny {k: 42})").run().await?;
        tx.commit().await?;
        if let Err(e) = db.flush().await {
            println!("  flush errored: {e}  (FAIL)");
            return Ok(false);
        }
        db.shutdown().await?;
    }

    match open_db(&path).await {
        Ok(db) => {
            let n = count(&db, "MATCH (t:Tiny) RETURN count(t) AS c").await?;
            db.shutdown().await?;
            let pass = n == 1;
            println!(
                "  post-reopen:  tiny={n}  ({})",
                if pass { "PASS" } else { "FAIL" }
            );
            Ok(pass)
        }
        Err(e) => {
            println!("  reopen errored: {e}  (FAIL)");
            Ok(false)
        }
    }
}

/// Scenario 10: minimal isolation of the conv-26 bench bug.
///
/// `CREATE (n:X { ts: "2026-05-13T..." })` where `X.ts` is declared as a
/// non-nullable `DataType::DateTime`.  uni-db accepts the CREATE, the
/// transaction commits with `Ok`, and the row is visible in-session.
/// But a background "Post-commit flush check" fails:
///
///     WARN: Post-commit flush check failed (non-critical):
///       Column 'ts' is declared as non-nullable but contains null values
///
/// After shutdown + reopen, `MATCH (n:X) RETURN count(n)` returns 0 —
/// the row is invisible to label-anchored MATCH despite being reachable
/// via `MATCH (n) WHERE id(n)=$vid` (which returns the node with
/// `labels(n)=["X"]`).
async fn string_into_datetime_column() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();
    const N: usize = 50;

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("X")
            .property("x_id", DataType::String)
            .property("ts", DataType::DateTime) // non-nullable
            .index("x_id", IndexType::Scalar(ScalarType::Hash))
            .done()
            .apply()
            .await?;

        let session = db.session();
        for i in 0..N {
            let tx = session.tx().await?;
            // BUG SHAPE: callers (e.g. uniko2 before the fix) accidentally
            // write the DateTime as a String via to_rfc3339().  uni-db
            // accepts this, the commit returns Ok, but the post-commit
            // flush silently fails because the String → DateTime coercion
            // produces null and the column is non-nullable.
            tx.execute_with(
                "CREATE (n:X {x_id: $id, ts: $ts})",
            )
            .param("id", Value::String(format!("x-{i}")))
            .param("ts", Value::String("2026-05-13T20:00:00.000Z".into()))
            .run()
            .await?;
            tx.commit().await?; // returns Ok — caller thinks the write succeeded
        }

        let pre = count(&db, "MATCH (n:X) RETURN count(n) AS c").await?;
        println!("  pre-shutdown: MATCH (n:X) = {pre}  (expected {N})");
        db.shutdown().await?;
    }

    let db = open_db(&path).await?;
    db.schema()
        .label("X")
        .property("x_id", DataType::String)
        .property("ts", DataType::DateTime)
        .index("x_id", IndexType::Scalar(ScalarType::Hash))
        .done()
        .apply()
        .await?;

    let post_label = count(&db, "MATCH (n:X) RETURN count(n) AS c").await?;
    let post_unlabeled =
        count(&db, "MATCH (n) WHERE n.x_id IS NOT NULL RETURN count(n) AS c").await?;
    println!(
        "  post-reopen:  MATCH (n:X) = {post_label}  |  MATCH (n) WHERE n.x_id IS NOT NULL = {post_unlabeled}"
    );

    let bug = post_label == 0 && post_unlabeled > 0
        || (post_label as usize) < N && (post_unlabeled as usize) >= N;
    db.shutdown().await?;

    if bug {
        println!("  ❌ BUG REPRODUCED: label-scan returns 0/{N} but rows are persisted (visible to unlabeled MATCH)");
        Ok(false)
    } else if post_label as usize == N {
        println!("  ✓ no discrepancy (post_label = N, behavior may have been fixed upstream)");
        Ok(true)
    } else {
        println!("  ⚠ unexpected state — manual inspection needed");
        Ok(true)
    }
}

/// Scenario 9: like scenario 8, but with a `Vector(N)` property and vector
/// index on the late-label, then `update_node`-style SET that writes the
/// vector.  This is what `record_episode` + `embed_episode` does in uniko.
///
/// The vector index materialization on flush is the strongest remaining
/// candidate for the trigger, since scenario 8 (without vector) doesn't
/// reproduce.
async fn label_match_with_vector_index() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    const DIM: usize = 384;
    const N_BULK: usize = 400;
    const N_EPISODES: usize = 200;

    let vec_cfg = VectorIndexCfg {
        algorithm: VectorAlgo::Flat,
        metric: VectorMetric::Cosine,
        embedding: None,
    };

    {
        let db = open_db(&path).await?;
        db.schema()
            .label("Bulk")
            .property("bulk_id", DataType::String)
            .property_nullable("payload", DataType::String)
            .index("bulk_id", IndexType::Scalar(ScalarType::Hash))
            .done()
            .label("Episode")
            .property("episode_id", DataType::String)
            .property("action_type", DataType::String)
            .property_nullable(
                "embedding",
                DataType::Vector { dimensions: DIM },
            )
            .index("episode_id", IndexType::Scalar(ScalarType::Hash))
            .index("embedding", IndexType::Vector(vec_cfg))
            .done()
            .apply()
            .await?;

        let session = db.session();

        // Phase A: bulk Bulks (mimic Observations).
        for i in 0..N_BULK {
            let tx = session.tx().await?;
            tx.execute_with("CREATE (n:Bulk {bulk_id: $b, payload: 'p'})")
                .param("b", Value::String(format!("bulk-{i}")))
                .run()
                .await?;
            tx.commit().await?;
        }

        // Phase B: N_EPISODES vertices, each = CREATE then labelless SET of vector.
        let dummy_vec: Vec<f32> = (0..DIM).map(|i| (i as f32) * 0.001).collect();
        for i in 0..N_EPISODES {
            let id = format!("ep-{i:04}");
            // tx 1: CREATE
            let tx = session.tx().await?;
            tx.execute_with(
                "CREATE (e:Episode {episode_id: $eid, action_type: 'retrieve'})",
            )
            .param("eid", Value::String(id.clone()))
            .run()
            .await?;
            tx.commit().await?;
            // tx 2: labelless SET of vector (exactly mirrors update_node)
            let tx = session.tx().await?;
            let r = tx
                .query_with("MATCH (e:Episode {episode_id: $eid}) RETURN id(e) AS vid")
                .param("eid", Value::String(id.clone()))
                .fetch_all()
                .await?;
            let vid: i64 = r.rows().first().unwrap().get("vid")?;
            tx.execute_with("MATCH (n) WHERE id(n) = $v SET n.embedding = $vec")
                .param("v", vid)
                .param("vec", Value::Vector(dummy_vec.clone()))
                .run()
                .await?;
            tx.commit().await?;
        }

        let pre = count(&db, "MATCH (e:Episode) RETURN count(e) AS c").await?;
        println!("  pre-shutdown: MATCH (e:Episode)={pre}  (expected {N_EPISODES})");
        db.shutdown().await?;
    }

    let db = open_db(&path).await?;
    // Re-apply schema to mirror uniko's open path.
    db.schema()
        .label("Bulk")
        .property("bulk_id", DataType::String)
        .property_nullable("payload", DataType::String)
        .index("bulk_id", IndexType::Scalar(ScalarType::Hash))
        .done()
        .label("Episode")
        .property("episode_id", DataType::String)
        .property("action_type", DataType::String)
        .property_nullable(
            "embedding",
            DataType::Vector { dimensions: DIM },
        )
        .index("episode_id", IndexType::Scalar(ScalarType::Hash))
        .index(
            "embedding",
            IndexType::Vector(VectorIndexCfg {
                algorithm: VectorAlgo::Flat,
                metric: VectorMetric::Cosine,
                embedding: None,
            }),
        )
        .done()
        .apply()
        .await?;

    let post_label = count(&db, "MATCH (e:Episode) RETURN count(e) AS c").await?;
    let post_unlabeled = count(
        &db,
        "MATCH (n) WHERE n.episode_id IS NOT NULL RETURN count(n) AS c",
    )
    .await?;
    println!("  post-reopen: MATCH (e:Episode)={post_label}  |  unlabeled-by-episode_id={post_unlabeled}  (expected {N_EPISODES})");

    let bug = post_label < post_unlabeled || post_label < N_EPISODES as i64;
    db.shutdown().await?;

    if bug {
        println!("  ❌ BUG REPRODUCED: label-anchored MATCH (e:Episode) returns {post_label}, unlabeled finds {post_unlabeled}");
        Ok(false)
    } else {
        println!("  ✓ no discrepancy");
        Ok(true)
    }
}

/// Scenario 8: the actual conv-26 bug — `MATCH (n:Label)` returns 0 rows
/// for nodes that are visible via `MATCH (n) WHERE id(n)=$vid` and whose
/// `labels(n)` confirms they have that label.
///
/// Trigger pattern (matches the uniko-bench question loop):
/// 1. Bulk-write many vertices of label `Bulk`.
/// 2. Create one `Late` vertex + edges from it to many Bulks.
/// 3. Create N more `Late` vertices, each followed by an `update_node`
///    setting a property (mimics `embed_episode` setting `embedding`).
/// 4. Shutdown + reopen.
/// 5. Compare `MATCH (l:Late) RETURN count(l)` vs the count of distinct
///    source-VIDs reachable via `MATCH (l)-[r:LINKED]->(b) RETURN id(l)`.
///
/// If counts differ, label-anchored MATCH is broken for this workload.
async fn label_match_vs_id_match() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    const N_BULK: usize = 400;
    const N_LATE: usize = 50;

    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        let session = db.session();

        // Phase A: bulk Bulks.
        for i in 0..N_BULK {
            let tx = session.tx().await?;
            tx.execute_with("CREATE (n:Bulk {bulk_id: $ext, payload: 'p'})")
                .param("ext", Value::String(format!("bulk-{i}")))
                .run()
                .await?;
            tx.commit().await?;
        }

        // Phase B: one Late + edges to first 100 Bulks (the cycle pattern).
        let tx = session.tx().await?;
        tx.execute_with(
            "CREATE (c:Late {cycle_id: 'cycle-1', agent_id: 'agent', created_count: 100})",
        )
        .run()
        .await?;
        tx.commit().await?;

        for i in 0..100 {
            let tx = session.tx().await?;
            tx.execute_with(
                "MATCH (c:Late {cycle_id: 'cycle-1'}), (b:Bulk {bulk_id: $bid}) \
                 CREATE (c)-[:LINKED]->(b)",
            )
            .param("bid", Value::String(format!("bulk-{i}")))
            .run()
            .await?;
            tx.commit().await?;
        }

        // Phase C: 50 more Late vertices, each followed by an update_node-style
        // SET (this matches the Episode → embedding pattern).  Each vertex is
        // created via MERGE in a separate tx, then re-opened in a fresh tx and
        // its `agent_id` overwritten — mirroring uniko's two-tx merge_node
        // followed by an update_node call.
        for i in 0..N_LATE {
            let id = format!("late-{i:03}");
            // tx 1: create (no MERGE because we don't need uniqueness here)
            let tx = session.tx().await?;
            tx.execute_with(
                "CREATE (c:Late {cycle_id: $cid, agent_id: 'init', created_count: 0})",
            )
            .param("cid", Value::String(id.clone()))
            .run()
            .await?;
            tx.commit().await?;
            // tx 2: labelless SET — exactly what uniko-store's update_node does
            let tx = session.tx().await?;
            let r = tx
                .query_with("MATCH (c:Late {cycle_id: $cid}) RETURN id(c) AS vid")
                .param("cid", Value::String(id.clone()))
                .fetch_all()
                .await?;
            let vid: i64 = r.rows().first().unwrap().get("vid")?;
            tx.execute_with("MATCH (n) WHERE id(n) = $v SET n.agent_id = 'updated'")
                .param("v", vid)
                .run()
                .await?;
            tx.commit().await?;
        }

        // In-session probes BEFORE shutdown.
        let lbl_count = count(&db, "MATCH (l:Late) RETURN count(l) AS c").await?;
        let edge_src_count = count(
            &db,
            "MATCH (l)-[:LINKED]->(b) RETURN count(DISTINCT id(l)) AS c",
        )
        .await?;
        println!("  pre-shutdown: MATCH (l:Late)={lbl_count}  |  distinct LINKED sources={edge_src_count}  |  expected Late=51, edge_sources=1");

        db.shutdown().await?;
    }

    let db = open_db(&path).await?;
    apply_schema(&db).await?;

    let lbl_count = count(&db, "MATCH (l:Late) RETURN count(l) AS c").await?;

    // Get count of distinct source VIDs in the LINKED adjacency.
    let r = db
        .session()
        .query_with("MATCH (l)-[:LINKED]->(b) RETURN DISTINCT id(l) AS vid")
        .fetch_all()
        .await?;
    let distinct_edge_sources = r.rows().len() as i64;

    // For each edge source, check labels(n) and label-anchored match.
    let mut id_says_late = 0i64;
    let mut label_match_finds = 0i64;
    for row in r.rows() {
        let vid: i64 = row.get("vid")?;
        let r2 = db
            .session()
            .query_with("MATCH (n) WHERE id(n) = $v RETURN labels(n) AS lbls")
            .param("v", vid)
            .fetch_all()
            .await?;
        if let Some(row2) = r2.rows().first() {
            let lbls: Vec<String> = row2.get("lbls").unwrap_or_default();
            if lbls.iter().any(|l| l == "Late") {
                id_says_late += 1;
            }
        }
        let r3 = db
            .session()
            .query_with("MATCH (n:Late) WHERE id(n) = $v RETURN id(n) AS vid")
            .param("v", vid)
            .fetch_all()
            .await?;
        if !r3.rows().is_empty() {
            label_match_finds += 1;
        }
    }

    println!("  post-reopen: MATCH (l:Late)={lbl_count}  |  distinct LINKED sources={distinct_edge_sources}");
    println!("    of {distinct_edge_sources} edge-source VIDs: labels(n) says 'Late' for {id_says_late}, MATCH (:Late) finds {label_match_finds}");

    let bug_reproduced = id_says_late > label_match_finds || lbl_count < distinct_edge_sources;
    db.shutdown().await?;

    if bug_reproduced {
        println!("  ❌ BUG REPRODUCED: label-anchored match disagrees with labels(n) or with edge sources");
        Ok(false)
    } else {
        println!("  ✓ no discrepancy");
        Ok(true)
    }
}

/// Scenario 6: bulk + late, then DROP the Uni handle without calling
/// `shutdown()`.  Documents the durability contract — what survives
/// when the writer dies uncleanly?
async fn no_shutdown_call() -> Result<bool, Box<dyn Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().to_str().unwrap().to_string();

    {
        let db = open_db(&path).await?;
        apply_schema(&db).await?;
        let session = db.session();
        for i in 0..400usize {
            let tx = session.tx().await?;
            tx.execute_with("CREATE (n:Bulk {bulk_id: $ext, payload: 'p'})")
                .param("ext", Value::String(format!("bulk-{i}")))
                .run()
                .await?;
            tx.commit().await?;
        }
        let tx = session.tx().await?;
        tx.execute_with(
            "CREATE (c:Late {cycle_id: 'cycle-1', agent_id: 'agent', created_count: 0})",
        )
        .run()
        .await?;
        tx.commit().await?;
        // intentionally NO shutdown — drop the handle.
        drop(db);
    }

    // This may fail to open if the WAL has no snapshot manifest; that
    // would itself be a useful signal.
    let open_result = Uni::open(&path).build().await;
    match open_result {
        Ok(db) => {
            apply_schema(&db).await?;
            let post_late = count(&db, "MATCH (c:Late) RETURN count(c) AS c").await?;
            let post_bulk = count(&db, "MATCH (b:Bulk) RETURN count(b) AS c").await?;
            db.shutdown().await?;
            println!("  post-reopen-without-shutdown: late={post_late} bulk={post_bulk}");
            Ok(true)
        }
        Err(e) => {
            println!("  reopen failed: {e}  (this is fine — documents the contract)");
            Ok(true)
        }
    }
}
