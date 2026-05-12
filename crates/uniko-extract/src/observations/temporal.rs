//! Temporal expression resolution relative to a reference timestamp.
//!
//! Resolves informal expressions like "yesterday", "last March",
//! "two days ago", "Last Fri", "next Monday", "two weekends ago" into
//! concrete [`DateTime<Utc>`] values.
//!
//! The reference timestamp is the message timestamp; relative
//! expressions resolve against it.  When no temporal expression is
//! recognised, the reference is returned unchanged — callers can detect
//! a no-match by comparing the result to the reference.

// Rust guideline compliant

use std::sync::OnceLock;

use chrono::{DateTime, Datelike, Duration, Utc, Weekday};
use regex::Regex;

/// Resolve a temporal expression in `text` relative to `reference`.
///
/// Returns the resolved [`DateTime<Utc>`], or the unchanged `reference`
/// when no temporal expression is found.
///
/// Supported patterns (matched in priority order):
/// - Generic relative: `today`, `tomorrow`, `yesterday`
/// - Numeric relative: `N days|weeks|months|years ago`, `in N days|weeks|months|years`
/// - Period relative: `last week|month|year`, `next week|month|year`
/// - Weekday relative: `last|this|next Monday..Sunday` (and 3-letter
///   abbreviations: `Mon`, `Tue`, ..., `Sun`; also `Fri` / `Friday`)
/// - Weekend relative: `last|this|next weekend`, `N weekends ago`
/// - Month relative: `in January..December` (resolves to nearest past
///   occurrence)
pub fn resolve_temporal(text: &str, reference: DateTime<Utc>) -> DateTime<Utc> {
    let lower = text.to_lowercase();

    // ── Highest-priority literal anchors ──────────────────────────
    if lower.contains("yesterday") {
        return reference - Duration::days(1);
    }
    if lower.contains("tomorrow") {
        return reference + Duration::days(1);
    }
    // Match "today" only as a whole word so "todays" / "today's"
    // (which usually appear in unrelated contexts) don't accidentally
    // trigger.  Also reject "today's" specifically.
    if has_word(&lower, "today") {
        return reference;
    }

    // ── Weekend handling — checked before generic "last X" / "next X"
    // because "last weekend" / "next weekend" need a different
    // resolution semantics than "last week".
    if let Some(delta) = parse_weekend(&lower) {
        return reference + delta;
    }

    // ── "N {unit} ago" and "in N {unit}" ──────────────────────────
    if let Some(delta) = parse_n_ago(&lower) {
        return reference - delta;
    }
    if let Some(delta) = parse_in_n(&lower) {
        return reference + delta;
    }

    // ── Weekday relative ("last Friday", "Last Fri", "this Tuesday",
    //                       "next Monday") ─────────────────────────
    if let Some(resolved) = parse_relative_weekday(&lower, reference) {
        return resolved;
    }

    // ── "last week|month|year" / "next week|month|year" ────────────
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
    if lower.contains("next week") {
        return reference + Duration::weeks(1);
    }
    if lower.contains("next month") {
        return reference + Duration::days(30);
    }
    if lower.contains("next year") {
        return reference
            .with_year(reference.year() + 1)
            .unwrap_or(reference);
    }

    // ── "in {month}" → nearest past occurrence ─────────────────────
    if let Some(month) = parse_in_month(&lower) {
        return resolve_month(month, reference);
    }

    // No temporal expression found — return the reference timestamp.
    reference
}

/// Check whether `text` contains `word` as a whole token.
///
/// Avoids accidental substring matches: `has_word("todays plan", "today")`
/// → `false`; `has_word("the today report", "today")` → `true`.
fn has_word(text: &str, word: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric())
        .any(|tok| tok == word)
}

/// Parse "N days|weeks|months|years ago" patterns.
///
/// `N` may be a literal digit ("2 weeks ago") or a small English word
/// numeral ("two weeks ago") up to ten — matches how the LoCoMo corpus
/// expresses these (`"two weekends ago we camped"`).
fn parse_n_ago(text: &str) -> Option<Duration> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?|years?|weekends?)\s+ago",
        )
        .unwrap()
    });

    let caps = re.captures(text)?;
    let n: i64 = parse_count(&caps[1])?;
    let unit = &caps[2];

    duration_for_unit(unit, n)
}

/// Parse "in N days|weeks|months|years" patterns (forward in time).
fn parse_in_n(text: &str) -> Option<Duration> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"in\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(days?|weeks?|months?|years?|weekends?)\b",
        )
        .unwrap()
    });

    let caps = re.captures(text)?;
    let n: i64 = parse_count(&caps[1])?;
    let unit = &caps[2];

    duration_for_unit(unit, n)
}

/// Parse a digit string or small English word numeral (`"one"..."ten"`)
/// to a count.  Returns `None` for unrecognised words.
fn parse_count(s: &str) -> Option<i64> {
    if let Ok(n) = s.parse::<i64>() {
        return Some(n);
    }
    Some(match s {
        "one" => 1,
        "two" => 2,
        "three" => 3,
        "four" => 4,
        "five" => 5,
        "six" => 6,
        "seven" => 7,
        "eight" => 8,
        "nine" => 9,
        "ten" => 10,
        _ => return None,
    })
}

/// Shared duration calculation for `N {unit}` patterns.
fn duration_for_unit(unit: &str, n: i64) -> Option<Duration> {
    Some(match unit {
        u if u.starts_with("day") => Duration::days(n),
        u if u.starts_with("week") && !u.starts_with("weekend") => Duration::weeks(n),
        u if u.starts_with("weekend") => Duration::days(n * 7),
        u if u.starts_with("month") => Duration::days(n * 30),
        u if u.starts_with("year") => Duration::days(n * 365),
        _ => return None,
    })
}

/// Parse "last weekend", "this weekend", "next weekend".
///
/// Returns the signed [`Duration`] offset from `reference`: -7d for
/// last, 0 for this, +7d for next.  None when no weekend phrase matches.
fn parse_weekend(text: &str) -> Option<Duration> {
    if text.contains("last weekend") {
        Some(-Duration::days(7))
    } else if text.contains("this weekend") {
        Some(Duration::zero())
    } else if text.contains("next weekend") {
        Some(Duration::days(7))
    } else {
        None
    }
}

/// Parse "last Friday", "this Tuesday", "next Monday", and abbreviated
/// 3-letter weekday forms ("Last Fri", "next Wed", "this Sat").
///
/// Resolution semantics:
/// - `last <weekday>` — the most recent past occurrence of that weekday
///   (strictly before reference; if reference is on that weekday, goes
///   back 7 days).
/// - `this <weekday>` — the occurrence within the current week.  When
///   ambiguous (e.g. ref is Tuesday, query is "this Friday"), resolves
///   forward.
/// - `next <weekday>` — the next future occurrence.
///
/// Standalone weekday names without `last/this/next` resolve to the
/// most recent past occurrence (treated as `last`).
fn parse_relative_weekday(text: &str, reference: DateTime<Utc>) -> Option<DateTime<Utc>> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(
            r"(?P<prefix>last|this|next)?\s*(?P<wd>monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
        )
        .unwrap()
    });

    let caps = re.captures(text)?;
    let weekday = parse_weekday(caps.name("wd")?.as_str())?;
    let prefix = caps.name("prefix").map(|m| m.as_str()).unwrap_or("last");

    let ref_weekday = reference.weekday();
    let ref_num = ref_weekday.num_days_from_monday() as i64;
    let target_num = weekday.num_days_from_monday() as i64;

    let delta_days = match prefix {
        "last" => {
            let diff = (ref_num - target_num).rem_euclid(7);
            if diff == 0 { -7 } else { -diff }
        }
        "next" => {
            let diff = (target_num - ref_num).rem_euclid(7);
            if diff == 0 { 7 } else { diff }
        }
        "this" => {
            // Same calendar week.  "This Friday" when today is Tuesday
            // means +3 days.  When today is Friday, means 0.  When
            // today is Saturday, means -1 (the past Friday).
            target_num - ref_num
        }
        _ => return None,
    };

    Some(reference + Duration::days(delta_days))
}

/// Map weekday name or 3-letter abbreviation to [`Weekday`].
///
/// Recognises full English names, 3-letter abbreviations, and the
/// 4-letter `tues`/`thur`/`thurs` variants commonly written informally.
fn parse_weekday(s: &str) -> Option<Weekday> {
    match s {
        "monday" | "mon" => Some(Weekday::Mon),
        "tuesday" | "tue" | "tues" => Some(Weekday::Tue),
        "wednesday" | "wed" => Some(Weekday::Wed),
        "thursday" | "thu" | "thur" | "thurs" => Some(Weekday::Thu),
        "friday" | "fri" => Some(Weekday::Fri),
        "saturday" | "sat" => Some(Weekday::Sat),
        "sunday" | "sun" => Some(Weekday::Sun),
        _ => None,
    }
}

/// Parse "in January" through "in December".
fn parse_in_month(text: &str) -> Option<u32> {
    let months = [
        ("january", 1),
        ("february", 2),
        ("march", 3),
        ("april", 4),
        ("may", 5),
        ("june", 6),
        ("july", 7),
        ("august", 8),
        ("september", 9),
        ("october", 10),
        ("november", 11),
        ("december", 12),
        ("jan", 1),
        ("feb", 2),
        ("mar", 3),
        ("apr", 4),
        ("jun", 6),
        ("jul", 7),
        ("aug", 8),
        ("sep", 9),
        ("oct", 10),
        ("nov", 11),
        ("dec", 12),
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

    /// Reference: Thursday, 2023-06-15.
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
    fn test_today() {
        let resolved = resolve_temporal("I'm doing this today.", ref_ts());
        assert_eq!(resolved, ref_ts());
    }

    #[test]
    fn test_tomorrow() {
        let resolved = resolve_temporal("I'll go tomorrow.", ref_ts());
        assert_eq!(resolved.day(), 16);
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
    fn test_next_week() {
        let resolved = resolve_temporal("Catching up next week.", ref_ts());
        assert_eq!(resolved.day(), 22);
    }

    #[test]
    fn test_next_year() {
        let resolved = resolve_temporal("See you next year.", ref_ts());
        assert_eq!(resolved.year(), 2024);
    }

    #[test]
    fn test_n_days_ago() {
        let resolved = resolve_temporal("About 3 days ago.", ref_ts());
        assert_eq!(resolved.day(), 12);
    }

    #[test]
    fn test_in_n_days() {
        let resolved = resolve_temporal("Catching up in 5 days.", ref_ts());
        assert_eq!(resolved.day(), 20);
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

    #[test]
    fn test_last_friday() {
        // Ref is Thursday 2023-06-15.  Last Friday = 2023-06-09.
        let resolved = resolve_temporal("Last Friday I baked pies.", ref_ts());
        assert_eq!(resolved.weekday(), Weekday::Fri);
        assert_eq!(resolved.day(), 9);
    }

    #[test]
    fn test_last_fri_abbrev() {
        let resolved = resolve_temporal("Last Fri I went hiking.", ref_ts());
        assert_eq!(resolved.weekday(), Weekday::Fri);
        assert_eq!(resolved.day(), 9);
    }

    #[test]
    fn test_this_friday_forward() {
        // Ref is Thursday; this Friday is tomorrow.
        let resolved = resolve_temporal("Going on this Friday.", ref_ts());
        assert_eq!(resolved.weekday(), Weekday::Fri);
        assert_eq!(resolved.day(), 16);
    }

    #[test]
    fn test_next_monday() {
        // Ref is Thursday; next Monday = 2023-06-19.
        let resolved = resolve_temporal("Meet next Monday.", ref_ts());
        assert_eq!(resolved.weekday(), Weekday::Mon);
        assert_eq!(resolved.day(), 19);
    }

    #[test]
    fn test_last_tuesday_when_ref_is_same_day_minus_one_week() {
        // Ref is Thursday; last Tuesday = 2023-06-13.
        let resolved = resolve_temporal("I called last Tuesday.", ref_ts());
        assert_eq!(resolved.weekday(), Weekday::Tue);
        assert_eq!(resolved.day(), 13);
    }

    #[test]
    fn test_last_thursday_with_same_weekday_ref_goes_back_one_week() {
        // Ref is Thursday 2023-06-15; last Thursday = 2023-06-08.
        let resolved = resolve_temporal("Last Thursday I started.", ref_ts());
        assert_eq!(resolved.weekday(), Weekday::Thu);
        assert_eq!(resolved.day(), 8);
    }

    #[test]
    fn test_two_weekends_ago() {
        let resolved = resolve_temporal("Two weekends ago we camped.", ref_ts());
        // 2 weekends = 14 days back.
        assert_eq!(resolved.day(), 1);
    }

    #[test]
    fn test_last_weekend() {
        let resolved = resolve_temporal("Last weekend was busy.", ref_ts());
        assert_eq!(resolved.day(), 8);
    }

    #[test]
    fn test_next_weekend() {
        let resolved = resolve_temporal("Free next weekend?", ref_ts());
        assert_eq!(resolved.day(), 22);
    }

    #[test]
    fn test_has_word_rejects_substrings() {
        assert!(has_word("the today report", "today"));
        assert!(!has_word("todays plan", "today"));
        assert!(!has_word("yesterdays trip", "today"));
    }
}
