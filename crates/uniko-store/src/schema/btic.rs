//! BTIC temporal helpers for Fact validity intervals.
//!
//! Provides domain-specific constructors and predicates layered over
//! uni-db's native [`Btic`] type.  All functions are pure — no I/O.

use chrono::{DateTime, Utc};
use uni_db::common::uni_btic::btic::POS_INF;
use uni_db::common::uni_btic::{Btic, Certainty, Granularity};

/// Observation count threshold for upgrading certainty from
/// [`Certainty::Approximate`] to [`Certainty::Definite`].
pub const CERTAINTY_THRESHOLD: u64 = 10;

/// Create an active BTIC interval `[observed_at, +∞)`.
///
/// Certainty defaults to [`Certainty::Approximate`] (< 10 observations).
/// Lo granularity defaults to [`Granularity::Day`].
/// Hi bound is `POS_INF` with zeroed granularity/certainty per INV-6.
pub fn btic_active(observed_at: DateTime<Utc>) -> Btic {
    let lo = observed_at.timestamp_millis();
    let meta = Btic::build_meta(
        Granularity::Day,
        Granularity::Millisecond, // sentinel hi → code 0
        Certainty::Approximate,
        Certainty::Definite, // sentinel hi → code 0
    );
    Btic::new(lo, POS_INF, meta).expect("valid active BTIC interval")
}

/// Close a BTIC interval by setting hi = `now`.
///
/// Returns a new [`Btic`] with the same lo but hi closed to `now`,
/// making the fact valid only in `[lo, now)`.
pub fn btic_invalidate(interval: &Btic, now: DateTime<Utc>) -> Btic {
    let hi = now.timestamp_millis();
    let meta = Btic::build_meta(
        interval.lo_granularity(),
        Granularity::Day,
        interval.lo_certainty(),
        Certainty::Definite,
    );
    Btic::new(interval.lo(), hi, meta).expect("invalidation must produce valid BTIC (lo < hi)")
}

/// Check if a point in time falls within a BTIC interval.
///
/// Uses half-open semantics: `lo <= point < hi`.
pub fn btic_contains(interval: &Btic, point: DateTime<Utc>) -> bool {
    let ms = point.timestamp_millis();
    interval.lo() <= ms && ms < interval.hi()
}

/// Check if two BTIC intervals overlap (Allen's algebra).
///
/// Two intervals overlap if their intersection is non-empty:
/// `a.lo < b.hi && b.lo < a.hi`.
pub fn btic_overlaps(a: &Btic, b: &Btic) -> bool {
    a.lo() < b.hi() && b.lo() < a.hi()
}

/// Check if interval `a` ends strictly before interval `b` begins.
pub fn btic_before(a: &Btic, b: &Btic) -> bool {
    a.hi() <= b.lo()
}

/// Upgrade lo-bound certainty from [`Certainty::Approximate`] to
/// [`Certainty::Definite`].
///
/// Call when `observation_count` crosses [`CERTAINTY_THRESHOLD`].
/// Returns the interval unchanged if already [`Certainty::Definite`].
pub fn btic_upgrade_certainty(interval: &Btic) -> Btic {
    if interval.lo_certainty() == Certainty::Definite {
        return *interval;
    }
    let meta = Btic::build_meta(
        interval.lo_granularity(),
        interval.hi_granularity(),
        Certainty::Definite,
        interval.hi_certainty(),
    );
    Btic::new(interval.lo(), interval.hi(), meta)
        .expect("certainty upgrade must preserve validity")
}

/// Get the granularity of both bounds `(lo, hi)`.
pub fn btic_granularity(interval: &Btic) -> (Granularity, Granularity) {
    (interval.lo_granularity(), interval.hi_granularity())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn dt(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

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
        let observed = dt(2024, 1, 1);
        let active = btic_active(observed);
        let now = dt(2024, 6, 15);
        let closed = btic_invalidate(&active, now);
        assert_eq!(closed.lo(), observed.timestamp_millis());
        assert_eq!(closed.hi(), now.timestamp_millis());
        assert!(closed.hi() < POS_INF);
    }

    #[test]
    fn test_btic_contains_active() {
        let observed = dt(2024, 1, 1);
        let b = btic_active(observed);
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
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_datetime() -> impl Strategy<Value = DateTime<Utc>> {
        // 2000-01-01 to 2030-12-31 in epoch millis
        (946_684_800_000i64..1_924_991_999_000i64).prop_map(|ms| {
            DateTime::from_timestamp_millis(ms).unwrap()
        })
    }

    proptest! {
        #[test]
        fn prop_containment_consistent(
            lo in arb_datetime(),
            point in arb_datetime(),
        ) {
            let b = btic_active(lo);
            let contains = btic_contains(&b, point);
            if point >= lo {
                prop_assert!(contains, "active interval must contain point >= lo");
            } else {
                prop_assert!(!contains, "active interval must not contain point < lo");
            }
        }

        #[test]
        fn prop_before_and_overlaps_exclusive(
            lo_a in arb_datetime(),
            lo_b in arb_datetime(),
        ) {
            // Create two finite intervals with 30-day width
            let hi_a = lo_a + chrono::Duration::days(30);
            let hi_b = lo_b + chrono::Duration::days(30);
            let a = btic_invalidate(&btic_active(lo_a), hi_a);
            let b = btic_invalidate(&btic_active(lo_b), hi_b);
            // before and overlaps should be mutually exclusive
            if btic_before(&a, &b) {
                prop_assert!(!btic_overlaps(&a, &b));
            }
        }

        #[test]
        fn prop_upgrade_is_idempotent(lo in arb_datetime()) {
            let b = btic_active(lo);
            let u1 = btic_upgrade_certainty(&b);
            let u2 = btic_upgrade_certainty(&u1);
            prop_assert_eq!(u1.lo(), u2.lo());
            prop_assert_eq!(u1.hi(), u2.hi());
            prop_assert_eq!(u1.meta(), u2.meta());
        }
    }
}
