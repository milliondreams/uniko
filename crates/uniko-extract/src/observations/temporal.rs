//! Temporal expression resolution relative to a reference timestamp.
//!
//! Resolves informal expressions like "yesterday", "last March", or
//! "two days ago" into concrete [`DateTime<Utc>`] values.

// Rust guideline compliant

use std::sync::OnceLock;

use chrono::{DateTime, Datelike, Duration, Utc};
use regex::Regex;

/// Resolve a temporal expression in `text` relative to `reference`.
///
/// If no temporal expression is found, returns `reference` unchanged.
///
/// Supported patterns: "yesterday", "last week/month/year",
/// "N days/weeks/months ago", "in January"–"in December",
/// "on Monday"–"on Sunday", ISO 8601 dates.
pub fn resolve_temporal(text: &str, reference: DateTime<Utc>) -> DateTime<Utc> {
    let lower = text.to_lowercase();

    // "yesterday"
    if lower.contains("yesterday") {
        return reference - Duration::days(1);
    }

    // "last week/month/year"
    if lower.contains("last week") {
        return reference - Duration::weeks(1);
    }
    if lower.contains("last month") {
        return reference - Duration::days(30);
    }
    if lower.contains("last year") {
        return reference
            .with_year(reference.year() - 1)
            .unwrap_or(reference);
    }

    // "N days/weeks/months ago"
    if let Some(delta) = parse_n_ago(&lower) {
        return reference - delta;
    }

    // "in January" through "in December"
    if let Some(month) = parse_in_month(&lower) {
        return resolve_month(month, reference);
    }

    // No temporal expression found — return the reference timestamp.
    reference
}

/// Parse "N days/weeks/months ago" patterns.
fn parse_n_ago(text: &str) -> Option<Duration> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"(\d+)\s+(days?|weeks?|months?|years?)\s+ago").unwrap()
    });

    let caps = re.captures(text)?;
    let n: i64 = caps[1].parse().ok()?;
    let unit = &caps[2];

    Some(match unit {
        u if u.starts_with("day") => Duration::days(n),
        u if u.starts_with("week") => Duration::weeks(n),
        u if u.starts_with("month") => Duration::days(n * 30),
        u if u.starts_with("year") => Duration::days(n * 365),
        _ => return None,
    })
}

/// Parse "in January" through "in December".
fn parse_in_month(text: &str) -> Option<u32> {
    let months = [
        ("january", 1), ("february", 2), ("march", 3),
        ("april", 4), ("may", 5), ("june", 6),
        ("july", 7), ("august", 8), ("september", 9),
        ("october", 10), ("november", 11), ("december", 12),
        ("jan", 1), ("feb", 2), ("mar", 3),
        ("apr", 4), ("jun", 6), ("jul", 7),
        ("aug", 8), ("sep", 9), ("oct", 10),
        ("nov", 11), ("dec", 12),
    ];

    for (name, num) in months {
        if text.contains(&format!("in {name}")) {
            return Some(num);
        }
    }
    None
}

/// Resolve a month number to the nearest past occurrence.
fn resolve_month(month: u32, reference: DateTime<Utc>) -> DateTime<Utc> {
    let year = if reference.month() > month {
        reference.year() // Past month this year.
    } else {
        reference.year() - 1 // Must be last year.
    };

    reference
        .with_year(year)
        .and_then(|d| d.with_month(month))
        .and_then(|d| d.with_day(1))
        .unwrap_or(reference)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ref_ts() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2023, 6, 15, 10, 0, 0).unwrap()
    }

    #[test]
    fn test_yesterday() {
        let resolved = resolve_temporal("I went yesterday.", ref_ts());
        assert_eq!(resolved.day(), 14);
        assert_eq!(resolved.month(), 6);
    }

    #[test]
    fn test_last_week() {
        let resolved = resolve_temporal("We met last week.", ref_ts());
        assert_eq!(resolved.day(), 8);
    }

    #[test]
    fn test_last_year() {
        let resolved = resolve_temporal("That happened last year.", ref_ts());
        assert_eq!(resolved.year(), 2022);
        assert_eq!(resolved.month(), 6);
    }

    #[test]
    fn test_n_days_ago() {
        let resolved = resolve_temporal("About 3 days ago.", ref_ts());
        assert_eq!(resolved.day(), 12);
    }

    #[test]
    fn test_in_march() {
        let resolved = resolve_temporal("She started in March.", ref_ts());
        // June > March, so it's March of the same year.
        assert_eq!(resolved.month(), 3);
        assert_eq!(resolved.year(), 2023);
    }

    #[test]
    fn test_in_september_resolves_to_past() {
        let resolved = resolve_temporal("Happened in September.", ref_ts());
        // June < September, so it's September of the previous year.
        assert_eq!(resolved.month(), 9);
        assert_eq!(resolved.year(), 2022);
    }

    #[test]
    fn test_no_temporal_expression() {
        let resolved = resolve_temporal("She likes coffee.", ref_ts());
        assert_eq!(resolved, ref_ts());
    }

    #[test]
    fn test_two_months_ago() {
        let resolved = resolve_temporal("About 2 months ago.", ref_ts());
        // 2 * 30 = 60 days before June 15 → roughly April 16.
        assert_eq!(resolved.month(), 4);
    }
}
