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

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc, Weekday};
use regex::Regex;

/// Granularity of a resolved temporal expression.
///
/// Used to derive a `[lo, hi)` retrieval window — "last May" widens to
/// the whole month, "yesterday" to one day, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalGranularity {
    Day,
    Week,
    Month,
    Year,
}

/// A temporal expression resolved to a concrete point + its semantic
/// granularity.  See [`resolve_temporal_with_granularity`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTemporal {
    /// The resolved point in time — usually the start of the period
    /// implied by the phrase (e.g. "last May" → 2025-05-01).
    pub point: DateTime<Utc>,
    /// Semantic granularity of the original phrase.
    pub granularity: TemporalGranularity,
}

impl ResolvedTemporal {
    /// Convert this resolved phrase into a half-open `[lo, hi)`
    /// retrieval window aligned to its [`granularity`](Self::granularity).
    ///
    /// - `Day`:   `[start-of-day(point), start-of-next-day(point))`
    /// - `Week`:  `[point, point + 7d)`
    /// - `Month`: `[first-of-month, first-of-next-month)`
    /// - `Year`:  `[Jan 1, Jan 1 next year)`
    #[must_use]
    pub fn to_range(&self) -> (DateTime<Utc>, DateTime<Utc>) {
        match self.granularity {
            TemporalGranularity::Day => {
                let day = self.point.date_naive();
                let lo = day.and_hms_opt(0, 0, 0).unwrap().and_utc();
                let hi = day
                    .succ_opt()
                    .unwrap_or(day)
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc();
                (lo, hi)
            }
            TemporalGranularity::Week => (self.point, self.point + Duration::days(7)),
            TemporalGranularity::Month => {
                let y = self.point.year();
                let m = self.point.month();
                let lo = NaiveDate::from_ymd_opt(y, m, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc();
                let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
                let hi = NaiveDate::from_ymd_opt(ny, nm, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc();
                (lo, hi)
            }
            TemporalGranularity::Year => {
                let y = self.point.year();
                let lo = NaiveDate::from_ymd_opt(y, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc();
                let hi = NaiveDate::from_ymd_opt(y + 1, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc();
                (lo, hi)
            }
        }
    }
}

/// Resolve a temporal expression in `text` relative to `reference`.
///
/// Returns the resolved [`DateTime<Utc>`], or the unchanged `reference`
/// when no temporal expression is found.
///
/// Thin wrapper over [`resolve_temporal_with_granularity`]; callers that
/// also need the granularity (e.g. to derive a retrieval window) should
/// use that instead.
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
    resolve_temporal_with_granularity(text, reference)
        .map(|r| r.point)
        .unwrap_or(reference)
}

/// Resolve a temporal expression in `text` relative to `reference`, also
/// returning its semantic [`TemporalGranularity`].
///
/// Returns `None` when no temporal expression is recognised — distinct
/// from [`resolve_temporal`], which returns `reference` on no-match.
/// The granularity lets callers derive a `[lo, hi)` retrieval window
/// via [`ResolvedTemporal::to_range`].
pub fn resolve_temporal_with_granularity(
    text: &str,
    reference: DateTime<Utc>,
) -> Option<ResolvedTemporal> {
    let lower = text.to_lowercase();

    // ── Highest-priority literal anchors ──────────────────────────
    if lower.contains("yesterday") {
        return Some(ResolvedTemporal {
            point: reference - Duration::days(1),
            granularity: TemporalGranularity::Day,
        });
    }
    if lower.contains("tomorrow") {
        return Some(ResolvedTemporal {
            point: reference + Duration::days(1),
            granularity: TemporalGranularity::Day,
        });
    }
    if has_word(&lower, "today") {
        return Some(ResolvedTemporal {
            point: reference,
            granularity: TemporalGranularity::Day,
        });
    }

    // ── Weekend handling ──────────────────────────────────────────
    if let Some(delta) = parse_weekend(&lower) {
        return Some(ResolvedTemporal {
            point: reference + delta,
            granularity: TemporalGranularity::Week,
        });
    }

    // ── "N {unit} ago" and "in N {unit}" ──────────────────────────
    if let Some((delta, gran)) = parse_n_ago_with_gran(&lower) {
        return Some(ResolvedTemporal {
            point: reference - delta,
            granularity: gran,
        });
    }
    if let Some((delta, gran)) = parse_in_n_with_gran(&lower) {
        return Some(ResolvedTemporal {
            point: reference + delta,
            granularity: gran,
        });
    }

    // ── Weekday relative ──────────────────────────────────────────
    if let Some(point) = parse_relative_weekday(&lower, reference) {
        return Some(ResolvedTemporal {
            point,
            granularity: TemporalGranularity::Day,
        });
    }

    // ── "last week|month|year" / "next week|month|year" ───────────
    if lower.contains("last week") {
        return Some(ResolvedTemporal {
            point: reference - Duration::weeks(1),
            granularity: TemporalGranularity::Week,
        });
    }
    if lower.contains("last month") {
        return Some(ResolvedTemporal {
            point: reference - Duration::days(30),
            granularity: TemporalGranularity::Month,
        });
    }
    if lower.contains("last year") {
        return Some(ResolvedTemporal {
            point: reference
                .with_year(reference.year() - 1)
                .unwrap_or(reference),
            granularity: TemporalGranularity::Year,
        });
    }
    if lower.contains("next week") {
        return Some(ResolvedTemporal {
            point: reference + Duration::weeks(1),
            granularity: TemporalGranularity::Week,
        });
    }
    if lower.contains("next month") {
        return Some(ResolvedTemporal {
            point: reference + Duration::days(30),
            granularity: TemporalGranularity::Month,
        });
    }
    if lower.contains("next year") {
        return Some(ResolvedTemporal {
            point: reference
                .with_year(reference.year() + 1)
                .unwrap_or(reference),
            granularity: TemporalGranularity::Year,
        });
    }

    // ── "in {month}" → nearest past occurrence ─────────────────────
    if let Some(month) = parse_in_month(&lower) {
        return Some(ResolvedTemporal {
            point: resolve_month(month, reference),
            granularity: TemporalGranularity::Month,
        });
    }

    // No temporal expression found.
    None
}

/// Like [`parse_n_ago`] but also returns the granularity inferred from
/// the unit string.  Used by [`resolve_temporal_with_granularity`].
fn parse_n_ago_with_gran(text: &str) -> Option<(Duration, TemporalGranularity)> {
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
    let dur = duration_for_unit(unit, n)?;
    Some((dur, granularity_for_unit(unit)))
}

/// Like [`parse_in_n`] but also returns the granularity.
fn parse_in_n_with_gran(text: &str) -> Option<(Duration, TemporalGranularity)> {
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
    let dur = duration_for_unit(unit, n)?;
    Some((dur, granularity_for_unit(unit)))
}

/// Map a unit substring (as captured by the N-unit regex) to a
/// [`TemporalGranularity`].  Weekends collapse to `Week`.
fn granularity_for_unit(unit: &str) -> TemporalGranularity {
    if unit.starts_with("day") {
        TemporalGranularity::Day
    } else if unit.starts_with("weekend") || unit.starts_with("week") {
        TemporalGranularity::Week
    } else if unit.starts_with("month") {
        TemporalGranularity::Month
    } else if unit.starts_with("year") {
        TemporalGranularity::Year
    } else {
        // Conservative fallback — should be unreachable given the regex.
        TemporalGranularity::Day
    }
}

/// Check whether `text` contains `word` as a whole token.
///
/// Avoids accidental substring matches: `has_word("todays plan", "today")`
/// → `false`; `has_word("the today report", "today")` → `true`.
fn has_word(text: &str, word: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric())
        .any(|tok| tok == word)
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
        // Left `\b` on the weekday group is load-bearing: without it,
        // the alternation matches as a *substring* anywhere in the
        // input.  `"next Whitsun"` would match `"sun"` (trailing 3
        // chars of `Whitsun`), `"frisat"` would match `"fri"`, etc.
        // The left word boundary blocks all such embedded matches.
        Regex::new(
            r"(?P<prefix>last|this|next)?\s*\b(?P<wd>monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\b",
        )
        .unwrap()
    });

    let caps = re.captures(text)?;
    let wd_match = caps.name("wd")?;
    // Reject hyphen-attached matches like "Monday-morning-quarterback".
    // Regex `\b` treats `-` as a word boundary, so the alternation
    // happily matches inside compound nouns.  A weekday that is part of
    // a hyphenated compound is not a temporal anchor.
    let bytes = text.as_bytes();
    let lhs_hyphen = wd_match.start().checked_sub(1).and_then(|i| bytes.get(i)) == Some(&b'-');
    let rhs_hyphen = bytes.get(wd_match.end()) == Some(&b'-');
    if lhs_hyphen || rhs_hyphen {
        return None;
    }
    let weekday = parse_weekday(wd_match.as_str())?;
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

    // ── granularity-aware resolver + to_range ─────────────────────

    #[test]
    fn test_granularity_yesterday_returns_day() {
        let r = resolve_temporal_with_granularity("it was yesterday", ref_ts()).unwrap();
        assert_eq!(r.granularity, TemporalGranularity::Day);
    }

    #[test]
    fn test_granularity_last_may_returns_month() {
        // ref is mid-March 2024 — "in may" resolves to most recent past May (May 2023).
        let r = resolve_temporal_with_granularity("happened in may", ref_ts()).unwrap();
        assert_eq!(r.granularity, TemporalGranularity::Month);
        assert_eq!(r.point.month(), 5);
    }

    #[test]
    fn test_granularity_last_year_returns_year() {
        let r = resolve_temporal_with_granularity("last year", ref_ts()).unwrap();
        assert_eq!(r.granularity, TemporalGranularity::Year);
    }

    #[test]
    fn test_granularity_no_match_returns_none() {
        assert!(resolve_temporal_with_granularity("just a normal sentence", ref_ts()).is_none());
    }

    #[test]
    fn test_to_range_day_is_24h_aligned_to_midnight() {
        let r = ResolvedTemporal {
            point: chrono::DateTime::parse_from_rfc3339("2025-05-15T14:23:00Z")
                .unwrap()
                .with_timezone(&Utc),
            granularity: TemporalGranularity::Day,
        };
        let (lo, hi) = r.to_range();
        assert_eq!(lo.to_rfc3339(), "2025-05-15T00:00:00+00:00");
        assert_eq!(hi.to_rfc3339(), "2025-05-16T00:00:00+00:00");
        assert_eq!((hi - lo).num_hours(), 24);
    }

    #[test]
    fn test_to_range_week_is_7d() {
        let r = ResolvedTemporal {
            point: chrono::DateTime::parse_from_rfc3339("2025-05-15T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            granularity: TemporalGranularity::Week,
        };
        let (lo, hi) = r.to_range();
        assert_eq!((hi - lo).num_days(), 7);
    }

    #[test]
    fn test_to_range_month_spans_calendar_month() {
        let r = ResolvedTemporal {
            point: chrono::DateTime::parse_from_rfc3339("2025-05-15T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            granularity: TemporalGranularity::Month,
        };
        let (lo, hi) = r.to_range();
        assert_eq!(lo.to_rfc3339(), "2025-05-01T00:00:00+00:00");
        assert_eq!(hi.to_rfc3339(), "2025-06-01T00:00:00+00:00");
    }

    #[test]
    fn test_to_range_month_december_rolls_to_next_year() {
        let r = ResolvedTemporal {
            point: chrono::DateTime::parse_from_rfc3339("2025-12-15T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            granularity: TemporalGranularity::Month,
        };
        let (lo, hi) = r.to_range();
        assert_eq!(lo.to_rfc3339(), "2025-12-01T00:00:00+00:00");
        assert_eq!(hi.to_rfc3339(), "2026-01-01T00:00:00+00:00");
    }

    #[test]
    // ── Regression tests for substring weekday false-positives ──────

    /// `"next Whitsun"` once matched as `"next … sun"` (trailing 3 chars
    /// of `"Whitsun"`) because the weekday alternation lacked a left
    /// word boundary.  The regex now anchors with `\b` on the left, so
    /// this phrase produces no temporal match.
    #[test]
    fn test_next_whitsun_does_not_match_weekday_substring() {
        let reference = chrono::DateTime::parse_from_rfc3339("2023-05-08T14:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        assert!(
            resolve_temporal_with_granularity("next Whitsun", reference).is_none(),
            "embedded 'sun' inside 'Whitsun' must not match the weekday alternation",
        );
    }

    /// Concatenated weekday abbreviations without spaces or word
    /// boundaries between them must not be picked up as a date either.
    #[test]
    fn test_concatenated_weekday_abbrevs_do_not_match() {
        let reference = chrono::DateTime::parse_from_rfc3339("2023-05-08T14:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        assert!(
            resolve_temporal_with_granularity("frisat", reference).is_none(),
            "'fri' embedded in 'frisat' must not match — both sides are word chars",
        );
    }

    /// Hyphen-attached compounds like "Monday-morning-quarterback"
    /// contain a weekday token that satisfies regex `\b` (hyphen is a
    /// word boundary), but they are not temporal expressions.  The
    /// post-match hyphen check rejects these.
    #[test]
    fn test_hyphen_attached_weekday_compounds_do_not_match() {
        let reference = chrono::DateTime::parse_from_rfc3339("2023-05-08T14:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        assert!(
            resolve_temporal_with_granularity("monday-morning-quarterback", reference).is_none(),
            "Monday inside a hyphenated compound noun is not a date",
        );
        assert!(
            resolve_temporal_with_granularity("the-friday-special", reference).is_none(),
            "Friday inside a hyphenated compound is not a date",
        );
        assert!(
            resolve_temporal_with_granularity("nine-to-five-mon", reference).is_none(),
            "weekday at end of hyphenated compound is not a date",
        );
    }

    /// Regression-safety: the intended `"last Friday"` path still works
    /// after the left-`\b` tightening.
    #[test]
    fn test_last_friday_still_resolves_after_word_boundary_fix() {
        // 2023-05-08 is Monday; "last Friday" → 2023-05-05.
        let reference = chrono::DateTime::parse_from_rfc3339("2023-05-08T14:00:00+00:00")
            .unwrap()
            .with_timezone(&Utc);
        let resolved =
            resolve_temporal_with_granularity("last Friday", reference).expect("must resolve");
        assert_eq!(
            resolved.point.format("%Y-%m-%d").to_string(),
            "2023-05-05",
            "last Friday from a Monday must resolve to the prior Friday",
        );
    }

    #[test]
    fn test_to_range_year_spans_calendar_year() {
        let r = ResolvedTemporal {
            point: chrono::DateTime::parse_from_rfc3339("2025-05-15T10:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            granularity: TemporalGranularity::Year,
        };
        let (lo, hi) = r.to_range();
        assert_eq!(lo.to_rfc3339(), "2025-01-01T00:00:00+00:00");
        assert_eq!(hi.to_rfc3339(), "2026-01-01T00:00:00+00:00");
    }
}
