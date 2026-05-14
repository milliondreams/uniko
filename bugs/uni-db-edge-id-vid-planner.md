# uni-db planner: `MATCH ()-[r]->() WHERE id(r) = $eid` resolves `id(r)` to `r._vid` instead of `r._eid`

## Symptom

```
Storage("Query error: DataFusion planning failed:
        Schema error: No field named \"r._vid\". Did you mean 'r._eid'?.")
```

## Repro

```rust
use uni_db::{Uni, DataType, Value};

#[tokio::main]
async fn main() {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("A").property("name", DataType::String).done()
        .edge_type("REL", &["A"], &["A"]).done()
        .apply().await.unwrap();

    let session = db.session();
    let tx = session.tx().await.unwrap();

    let result = tx
        .execute_with(
            "CREATE (a:A {name: 'x'})-[r:REL]->(b:A {name: 'y'}) \
             RETURN id(r) AS eid",
        )
        .run().await.unwrap();
    tx.commit().await.unwrap();
    let eid: i64 = 0; // recover from result rows in real test

    let tx = session.tx().await.unwrap();
    // ↓ This fails — planner looks for `r._vid`.
    let result = tx
        .execute_with("MATCH ()-[r]->() WHERE id(r) = $eid DELETE r")
        .param("eid", eid)
        .run()
        .await;
    println!("{result:?}");
}
```

## Expected
`id(r)` on a relationship variable should resolve to the edge id (`_eid`), not the node vid.

## Affected uniko2 sites
- `crates/uniko-store/src/storage/edges.rs::delete_edge` (line 250)
- `crates/uniko-store/src/storage/edges.rs::update_edge` (line 310)

## Workaround
None known yet — pending uni-db fix or confirmed alternate syntax.
