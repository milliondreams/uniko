//! Integration tests for BTIC helpers with a real database.

use chrono::{TimeZone, Utc};
use uni_db::Uni;
use uni_db::common::uni_btic::btic::POS_INF;
use uni_db::common::uni_btic::{Certainty, Granularity};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::btic::*;
use uniko_store::schema::register_schema;
use uniko_store::storage::embed_catalog;

async fn test_db() -> Uni {
    let config = UnikoConfig::default();
    let db = Uni::in_memory()
        .xervo_catalog(embed_catalog(&config))
        .build()
        .await
        .expect("in-memory db");
    register_schema(&db, &config)
        .await
        .expect("schema registration");
    db
}

fn dt(y: i32, m: u32, d: u32) -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
}

/// Store a Fact with a BTIC valid_at interval and read it back.
#[tokio::test]
async fn test_fact_with_btic_roundtrip() {
    let db = test_db().await;
    let session = db.session();

    let active = btic_active(dt(2024, 3, 15));

    let tx = session.tx().await.unwrap();
    tx.execute_with(
        "CREATE (:Fact {fact_id: 'f-btic', subject: 'user', predicate: 'likes', object: 'Rust', valid_at: $btic})",
    )
    .param("btic", uni_db::common::Value::Temporal(
        uni_db::common::TemporalValue::Btic {
            lo: active.lo(),
            hi: active.hi(),
            meta: active.meta(),
        },
    ))
    .run()
    .await
    .unwrap();
    tx.commit().await.unwrap();

    let result = session
        .query("MATCH (f:Fact {fact_id: 'f-btic'}) RETURN f.valid_at")
        .await
        .unwrap();
    assert_eq!(result.len(), 1);

    db.shutdown().await.unwrap();
}

// ── BTIC unit-level tests (pure functions, no DB required) ──

#[test]
fn test_btic_active_creation() {
    let observed = dt(2024, 6, 15);
    let b = btic_active(observed);
    assert_eq!(b.lo(), observed.timestamp_millis());
    assert_eq!(b.hi(), POS_INF);
    assert_eq!(b.lo_certainty(), Certainty::Approximate);
    assert_eq!(b.lo_granularity(), Granularity::Day);
}

#[test]
fn test_btic_invalidation() {
    let active = btic_active(dt(2024, 1, 1));
    let now = dt(2024, 6, 15);
    let closed = btic_invalidate(&active, now);
    assert_eq!(closed.lo(), active.lo());
    assert_eq!(closed.hi(), now.timestamp_millis());
    assert!(closed.hi() < POS_INF);
}

#[test]
fn test_btic_contains_active() {
    let b = btic_active(dt(2024, 1, 1));
    assert!(btic_contains(&b, dt(2024, 6, 15)));
    assert!(btic_contains(&b, dt(2025, 12, 31)));
}

#[test]
fn test_btic_contains_invalidated() {
    let b = btic_invalidate(&btic_active(dt(2024, 1, 1)), dt(2024, 6, 15));
    assert!(btic_contains(&b, dt(2024, 3, 1)));
    assert!(!btic_contains(&b, dt(2024, 7, 1)));
}

#[test]
fn test_btic_contains_before_start() {
    let b = btic_active(dt(2024, 6, 1));
    assert!(!btic_contains(&b, dt(2024, 5, 31)));
}

#[test]
fn test_btic_overlaps_same_period() {
    let a = btic_invalidate(&btic_active(dt(2024, 1, 1)), dt(2024, 6, 1));
    let b = btic_invalidate(&btic_active(dt(2024, 3, 1)), dt(2024, 9, 1));
    assert!(btic_overlaps(&a, &b));
    assert!(btic_overlaps(&b, &a));
}

#[test]
fn test_btic_overlaps_sequential() {
    let a = btic_invalidate(&btic_active(dt(2024, 1, 1)), dt(2024, 3, 1));
    let b = btic_invalidate(&btic_active(dt(2024, 6, 1)), dt(2024, 9, 1));
    assert!(!btic_overlaps(&a, &b));
}

#[test]
fn test_btic_before() {
    let a = btic_invalidate(&btic_active(dt(2024, 1, 1)), dt(2024, 3, 1));
    let b = btic_invalidate(&btic_active(dt(2024, 6, 1)), dt(2024, 9, 1));
    assert!(btic_before(&a, &b));
    assert!(!btic_before(&b, &a));
}

#[test]
fn test_btic_certainty_upgrade() {
    let b = btic_active(dt(2024, 1, 1));
    assert_eq!(b.lo_certainty(), Certainty::Approximate);
    let upgraded = btic_upgrade_certainty(&b);
    assert_eq!(upgraded.lo_certainty(), Certainty::Definite);
    assert_eq!(upgraded.lo(), b.lo());
    assert_eq!(upgraded.hi(), b.hi());
}

#[test]
fn test_btic_granularity() {
    let b = btic_active(dt(2024, 1, 1));
    let (lo_g, hi_g) = btic_granularity(&b);
    assert_eq!(lo_g, Granularity::Day);
    assert_eq!(hi_g, Granularity::Millisecond); // sentinel
}
