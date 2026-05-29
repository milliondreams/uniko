# uni-db: server-side primitives that would eliminate uniko-store's RMW lock layer

## Context

`uniko-store` currently serializes read-modify-write operations against
the same logical row using an in-process striped `tokio::sync::Mutex`
(`crates/uniko-store/src/locks.rs::StripedLocks`).  This works because
uniko is single-process today, but it is upstream-fixable: every site
that uses the lock would collapse to a single Cypher statement if uni-db
exposed the primitives below.

Five candidate primitives were investigated in `../uni/` (commit at the
time of writing: `b52b184` in `uniko2` / current `main` in `uni`).
None of them currently cover the RMW case:

| Primitive | Status | Source |
|---|---|---|
| CRDT types (`GCounter`, `PNCounter`, `LWWRegister`, …) | Passive data only — incrementing still requires read-merge-write | `uni-common/src/core/schema.rs:77-86` |
| Atomic SET expressions (`n.count += 1`) | SET evaluates against the row binding passed in, not server-side state | `uni-query/src/query/executor/read.rs:1087-1100` |
| `MERGE` serializability | Per-L0-buffer uniqueness only; not enforced across concurrent transactions | `uni-query/src/query/df_graph/mutation_common.rs:719-760`, `uni-store/src/runtime/l0.rs:134` |
| Transaction isolation | Explicitly last-writer-wins on L0 merge | `uni-store/src/storage/manager.rs:1964, 2070` |
| Row-level locks | None.  Lance-level locks are per-table only | `uni-store/src/backend/lance.rs:99, 282` |

This file collects the three primitive requests in priority order.
Each request lists the workspace site(s) that would collapse on
adoption.

---

## Request 1 — Server-side atomic SET expressions

### What we need

```cypher
MATCH (n:Fact {fact_id: $fid})
SET n.observation_count = n.observation_count + $delta,
    n.confidence       = laplace_smoothing(n.observation_count)
```

…where the right-hand side of SET evaluates against the *committed*
state of `n.observation_count` at write time, not against the snapshot
the caller read.  Equivalent to PostgreSQL's `UPDATE … SET col = col + 1`
or Neo4j's `SET n.col = coalesce(n.col, 0) + 1`.

### Sites this would simplify

- `crates/uniko-store/src/operations/facts.rs::upsert_fact_by_triple` —
  the lookup, Rust-side `prior_count + observation_count` arithmetic,
  conditional certainty upgrade, and `update_node` write would collapse
  into one `MERGE … ON CREATE SET … ON MATCH SET …` statement.
- `crates/uniko-store/src/operations/facts.rs::record_entity_invalidation`
  — same shape: read `invalidation_count`, increment, write back.
- `crates/uniko-store/src/operations/facts.rs::batch_upsert_facts` —
  the whole Phase 1 / Phase 3 split exists only because the SET side
  needs `prior_count + input.observation_count`.  One `UNWIND $rows AS r
  MERGE (n:Fact {fact_id: r.fact_id}) ON MATCH SET
  n.observation_count = n.observation_count + r.observation_count`
  would replace it.
- `crates/uniko-store/src/storage/kb_stats.rs::bump_modality_presence`
  — `SET s.has_image_content = s.has_image_content OR $v` would replace
  the read-then-merge map and the `kb_stats_lock` mutex entirely.

### Repro for the gap

```rust
use uni_db::{Uni, DataType, Value};

#[tokio::main]
async fn main() {
    let db = Uni::in_memory().build().await.unwrap();
    db.schema()
        .label("Counter").property("n", DataType::Int).done()
        .apply().await.unwrap();

    // Seed.
    {
        let tx = db.session().tx().await.unwrap();
        tx.execute_with("CREATE (:Counter {id: 'x', n: 0})")
            .run().await.unwrap();
        tx.commit().await.unwrap();
    }

    // Two concurrent transactions, each "SET n = n + 1".
    let d1 = std::sync::Arc::new(db);
    let d2 = d1.clone();
    let a = tokio::spawn(async move {
        let tx = d1.session().tx().await.unwrap();
        tx.execute_with("MATCH (c:Counter {id: 'x'}) SET c.n = c.n + 1")
            .run().await.unwrap();
        tx.commit().await.unwrap();
    });
    let b = tokio::spawn(async move {
        let tx = d2.session().tx().await.unwrap();
        tx.execute_with("MATCH (c:Counter {id: 'x'}) SET c.n = c.n + 1")
            .run().await.unwrap();
        tx.commit().await.unwrap();
    });
    let _ = tokio::join!(a, b);

    let session = d2.session();
    let r = session.query_with("MATCH (c:Counter {id: 'x'}) RETURN c.n AS n")
        .fetch_all().await.unwrap();
    let n: i64 = r.rows().first().unwrap().get("n").unwrap();
    assert_eq!(n, 2, "want 2 (atomic), got {n} (last-writer-wins)");
}
```

The assert fails today (`n == 1`).

---

## Request 2 — Serializable `MERGE` on a unique key

### What we need

```cypher
MERGE (n:Entity {entity_id: $eid})
ON CREATE SET n.created_at = $now
```

…with the guarantee that two concurrent callers cannot both observe
"not present" and each commit a fresh row.  Either:

- a unique constraint (`CREATE CONSTRAINT FOR (n:Entity) REQUIRE
  n.entity_id IS UNIQUE`) enforced across L0 buffers / on commit, or
- transactional MERGE that takes a row-key lock at MATCH time.

### Sites this would simplify

- `crates/uniko-store/src/storage/nodes.rs::merge_node` — the function
  currently does a two-step `get_node_by_ext_id` → `update_node` /
  `create_node` because uni-db's `MERGE … ON CREATE SET …` evaluates
  NOT-NULL constraints before `ON CREATE SET` runs (`storage/blob.rs:88-105`
  notes the same workaround for `:ArtifactContent`).  Both sites
  collapse if MERGE were both correct under contention and tolerant of
  NOT-NULL `SET` ordering.

### Repro for the gap

```rust
let db = std::sync::Arc::new(Uni::in_memory().build().await.unwrap());
db.schema()
    .label("E").property("eid", DataType::String).done()
    .apply().await.unwrap();

let mut handles = Vec::new();
for _ in 0..16 {
    let d = db.clone();
    handles.push(tokio::spawn(async move {
        let tx = d.session().tx().await.unwrap();
        tx.execute_with("MERGE (e:E {eid: 'shared'})")
            .run().await.unwrap();
        tx.commit().await.unwrap();
    }));
}
for h in handles { let _ = h.await; }

let session = db.session();
let r = session.query_with("MATCH (e:E {eid: 'shared'}) RETURN count(e) AS c")
    .fetch_all().await.unwrap();
let c: i64 = r.rows().first().unwrap().get("c").unwrap();
assert_eq!(c, 1, "want 1, got {c} duplicate rows");
```

The assert fails today (`c` is some integer > 1 under contention).

---

## Request 3 — Row-level pessimistic locking primitive

### What we need

Either:

- `MATCH (n) WHERE id(n) = $vid SET LOCK` — explicit row lock acquired
  inside a transaction and released on commit/abort, or
- `MATCH (n) … FOR UPDATE` — Postgres-style locking-MATCH clause.

This is a fallback for sites whose RMW isn't a simple counter and so
won't fit Request 1's atomic-SET pattern.

### Sites this would simplify

- Future sites that read a row, compute a non-trivial Rust-side update
  (e.g. property-graph schema migrations, audit-trail enrichment),
  and write back.  None in `uniko-store` today, but the pattern keeps
  recurring as the workspace grows; without a primitive, every new
  such site needs to acquire an `rmw_locks` stripe.

---

## Adoption order

1. **Request 1** has the broadest immediate impact — four sites in
   `uniko-store` would simplify, including the `kb_stats_lock` global
   mutex.  Estimated diff to `uniko-store`: ~150 LOC deleted.
2. **Request 2** closes the `merge_node` race specifically, plus the
   blob-store NOT-NULL workaround at `storage/blob.rs:88-105`.
3. **Request 3** is a safety net for future work; not blocking today.

When any of these land in uni-db, the corresponding site in
`uniko-store` should drop its `self.rmw_locks.lock(...)` acquisition
and switch to the new server-side primitive.  The regression tests in
`crates/uniko-store/tests/storage_tests.rs` (the four
`test_*_concurrent_no_*` cases) will continue to assert the
non-regression.
