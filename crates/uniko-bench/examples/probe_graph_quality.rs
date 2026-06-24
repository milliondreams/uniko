//! Read-only **graph-quality probe** for the M0 diagnosis (Phase A).
//!
//! Quantifies the two graph-quality gaps M0 was scoped around, directly
//! from an already-ingested KB — no model runtime / GPU needed (it reads
//! stored `Entity.embedding` vectors and computes cosine in-process):
//!
//!   1. **ABOUT-edge coverage** — how many Observations have zero ABOUT
//!      edges, broken down by subject class (pronoun / empty / other).
//!      Only the *non-pronoun* "other" bucket is potentially fixable by
//!      better observation→entity linking (1b); pronoun/empty subjects are
//!      filtered or unlinkable by design.
//!   2. **Entity near-duplication** — first confirms `Entity.embedding` is
//!      actually populated (the whole 1a premise), then buckets
//!      distinct-name entity pairs by cosine to size the fuzzy-merge
//!      opportunity *and* its precision risk (sample pairs are printed for
//!      a manual read).
//!   3. **SUPPORTED_BY weight distribution** — confirms whether the
//!      weights are still the inert constant 1.0 (pre-1c) or informative.
//!
//! Usage: `cargo run --example probe_graph_quality -p uniko-bench -- <kb_dir> [max_entities]`

use std::path::PathBuf;

use uniko_bench::bench_config::BenchConfig;
use uniko_bench::open_kb;
use uniko_store::config::UnikoConfig;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Pronoun / indefinite / interrogative tokens that the observation
/// resolver either drops or cannot anchor to a named entity. Mirrors the
/// classes handled in `nlp/decode.rs`.
fn is_pronoun(s: &str) -> bool {
    matches!(
        s.trim().to_lowercase().as_str(),
        "i" | "you"
            | "he"
            | "she"
            | "it"
            | "we"
            | "they"
            | "this"
            | "that"
            | "these"
            | "those"
            | "me"
            | "him"
            | "her"
            | "us"
            | "them"
            | "my"
            | "your"
            | "his"
            | "its"
            | "our"
            | "their"
            | "myself"
            | "yourself"
            | "himself"
            | "herself"
            | "itself"
            | "ourselves"
            | "themselves"
            | "who"
            | "whom"
            | "which"
            | "what"
            | "someone"
            | "somebody"
            | "anyone"
            | "anybody"
            | "everyone"
            | "everybody"
            | "nobody"
            | "no one"
            | "others"
            | "one"
            | "there"
            | "here"
    )
}

/// A name that looks like a temporal expression (date/time reference) —
/// these belong on observation temporal anchors, not as `Entity` nodes.
fn is_temporal(s: &str) -> bool {
    let t = s.trim().trim_end_matches(['.', ',', '!', '?', ';', ':']).to_lowercase();
    const MONTHS: [&str; 12] = [
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december",
    ];
    const DAYS: [&str; 9] = [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today",
        "tomorrow",
    ];
    const SEASONS: [&str; 5] = ["summer", "winter", "spring", "fall", "autumn"];
    const REL: [&str; 8] = [
        "yesterday", "tonight", "morning", "evening", "afternoon", "noon", "midnight", "weekend",
    ];
    let words: Vec<&str> = t.split_whitespace().collect();
    let first = words.first().copied().unwrap_or("");
    MONTHS.contains(&t.as_str())
        || DAYS.contains(&t.as_str())
        || SEASONS.contains(&t.as_str())
        || REL.contains(&t.as_str())
        || words.iter().any(|w| MONTHS.contains(w) || DAYS.contains(w) || REL.contains(w))
        // "last X" / "next X" / "this X" / "X ago" / "last fri"
        || matches!(first, "last" | "next" | "this")
        || t.ends_with(" ago")
        || t.starts_with("last fri")
        // bare year or numeric date fragments
        || (t.chars().all(|c| c.is_ascii_digit() || c == '/' || c == '-') && t.len() >= 2)
}

/// A name that looks like a greeting / discourse fragment, not an entity.
fn is_greeting(s: &str) -> bool {
    let first = s
        .trim()
        .to_lowercase()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    matches!(
        first.as_str(),
        "hey" | "hi" | "hello" | "thanks" | "thank" | "bye" | "goodbye"
            | "ok" | "okay" | "yeah" | "yep" | "yes" | "nope" | "sure" | "well"
            | "oh" | "wow" | "hmm" | "please" | "sorry" | "congrats" | "congratulations"
    )
}

/// Trailing punctuation that should be normalised out of a canonical name.
fn has_trailing_punct(s: &str) -> bool {
    s.trim().ends_with(['.', ',', '!', '?', ';', ':'])
}

fn pct(n: i64, total: i64) -> f64 {
    if total <= 0 {
        0.0
    } else {
        100.0 * n as f64 / total as f64
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: probe_graph_quality <kb_dir> <bench_config.json> [max_entities]")
        .into();
    // The KB schema (embedding dims, Bytes columns) is fixed at ingest by
    // the bench-config, so opening with bare defaults hits a schema-type
    // mismatch. Build the same UnikoConfig the bench did (main.rs:209-215).
    let bench_config: PathBuf = std::env::args()
        .nth(2)
        .expect("usage: probe_graph_quality <kb_dir> <bench_config.json> [max_entities]")
        .into();
    let max_entities: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);

    let mut config = UnikoConfig::default();
    BenchConfig::load(&bench_config)
        .and_then(|bc| bc.apply_to_uniko_config(&mut config))
        .map_err(|e| anyhow::anyhow!("loading bench config {}: {e}", bench_config.display()))?;

    let kb = open_kb(&path, config, &[]).await?;
    let session = kb.db().session();

    // Small inline scalar helper closure isn't possible across await, so
    // repeat the 3-line fetch pattern (same as probe_kb.rs).
    macro_rules! scalar {
        ($q:expr) => {{
            let r = session.query_with($q).fetch_all().await?;
            r.rows()
                .first()
                .and_then(|row| row.get::<i64>("n").ok())
                .unwrap_or(-1)
        }};
    }

    println!("== ABOUT-edge coverage ==");
    let total_obs = scalar!("MATCH (o:Observation) RETURN count(o) AS n");
    let obs_with_about =
        scalar!("MATCH (o:Observation) WHERE (o)-[:ABOUT]->() RETURN count(o) AS n");
    let zero = (total_obs - obs_with_about).max(0);
    println!("observations total            : {total_obs}");
    println!(
        "observations with >=1 ABOUT   : {obs_with_about}  ({:.1}%)",
        pct(obs_with_about, total_obs)
    );
    println!(
        "observations with 0 ABOUT     : {zero}  ({:.1}%)",
        pct(zero, total_obs)
    );

    // Breakdown of 0-ABOUT observations by subject class.
    let rows = session
        .query_with(
            "MATCH (o:Observation) WHERE NOT (o)-[:ABOUT]->() \
             RETURN coalesce(o.subject, '') AS subj",
        )
        .fetch_all()
        .await?;
    let (mut pron, mut empty, mut other) = (0i64, 0i64, 0i64);
    let mut other_samples: Vec<String> = Vec::new();
    for r in rows.rows() {
        let s: String = r.get("subj").unwrap_or_default();
        if s.trim().is_empty() {
            empty += 1;
        } else if is_pronoun(&s) {
            pron += 1;
        } else {
            other += 1;
            if other_samples.len() < 30 {
                other_samples.push(s);
            }
        }
    }
    println!(
        "  0-ABOUT subject classes: pronoun/indefinite={pron}  empty={empty}  other(non-pronoun)={other}"
    );
    println!("  (only 'other' is potentially fixable by better obs->entity linking — 1b)");

    // Speaker is auto-ABOUT'd to every observation, so ">=1 ABOUT" is
    // trivially ~100%. The substantive question is whether observations
    // link to a named :Entity (not just the :Participant speaker).
    let about_entity =
        scalar!("MATCH (o:Observation)-[:ABOUT]->(:Entity) RETURN count(DISTINCT o) AS n");
    let about_part =
        scalar!("MATCH (o:Observation)-[:ABOUT]->(:Participant) RETURN count(DISTINCT o) AS n");
    let no_entity = (total_obs - about_entity).max(0);
    println!(
        "  observations ABOUT a named Entity : {about_entity}  ({:.1}%)",
        pct(about_entity, total_obs)
    );
    println!("  observations ABOUT a Participant  : {about_part}");
    println!(
        "  observations with NO Entity link (speaker-only): {no_entity}  ({:.1}%)  <- real 1b surface",
        pct(no_entity, total_obs)
    );

    println!("\n== Consolidation artifacts ==");
    let facts = scalar!("MATCH (f:Fact) RETURN count(f) AS n");
    let cycles = scalar!("MATCH (c:ConsolidationCycle) RETURN count(c) AS n");
    let supported =
        scalar!("MATCH (:Fact)-[r:SUPPORTED_BY]->(:Observation) RETURN count(r) AS n");
    let obs_triple =
        scalar!("MATCH (o:Observation) WHERE o.predicate IS NOT NULL RETURN count(o) AS n");
    println!(
        "Facts={facts}  ConsolidationCycles={cycles}  SUPPORTED_BY={supported}  \
         Observations-with-triple={obs_triple}"
    );
    if supported == 0 {
        println!("  (no SUPPORTED_BY edges -> consolidation hasn't produced Facts here; \
                  1c needs a consolidated KB)");
    }
    if !other_samples.is_empty() {
        println!("  sample non-pronoun 0-ABOUT subjects:");
        for s in &other_samples {
            println!("    {s:?}");
        }
    }

    println!("\n== Entity duplication ==");
    let total_ent = scalar!("MATCH (e:Entity) RETURN count(e) AS n");
    let ent_emb = scalar!("MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN count(e) AS n");
    println!("entities total                : {total_ent}");
    println!(
        "entities with embedding       : {ent_emb}  ({:.1}%)",
        pct(ent_emb, total_ent)
    );
    if ent_emb == 0 {
        println!(
            "  !! Entity.embedding NEVER populated — fuzzy-merge (1a) needs an embedding \
             backfill first."
        );
    }

    // ── Entity precision: how many "entities" are actually noise? ──
    {
        let rows = session
            .query_with(&format!(
                "MATCH (e:Entity) RETURN e.name AS name, coalesce(e.entity_type, '') AS typ \
                 LIMIT {max_entities}"
            ))
            .fetch_all()
            .await?;
        let mut by_type: std::collections::BTreeMap<String, i64> = Default::default();
        let (mut temporal, mut greeting, mut punct, mut clean) = (0i64, 0i64, 0i64, 0i64);
        let mut temporal_s: Vec<String> = Vec::new();
        let mut greeting_s: Vec<String> = Vec::new();
        let mut punct_s: Vec<String> = Vec::new();
        for r in rows.rows() {
            let name: String = r.get("name").unwrap_or_default();
            let typ: String = r.get("typ").unwrap_or_default();
            *by_type.entry(if typ.is_empty() { "<none>".into() } else { typ }).or_insert(0) += 1;
            if is_temporal(&name) {
                temporal += 1;
                if temporal_s.len() < 20 { temporal_s.push(name); }
            } else if is_greeting(&name) {
                greeting += 1;
                if greeting_s.len() < 20 { greeting_s.push(name); }
            } else if has_trailing_punct(&name) {
                punct += 1;
                if punct_s.len() < 20 { punct_s.push(name); }
            } else {
                clean += 1;
            }
        }
        let noise = temporal + greeting + punct;
        let loaded = temporal + greeting + punct + clean;
        println!("  entity_type histogram:");
        for (t, c) in &by_type {
            println!("    {t:<14} {c}");
        }
        println!(
            "  NOISE: temporal={temporal} greeting={greeting} trailing-punct={punct}  \
             => {noise}/{loaded} ({:.1}%) look like non-entities; clean={clean}",
            pct(noise, loaded)
        );
        if !temporal_s.is_empty() { println!("    temporal e.g.: {temporal_s:?}"); }
        if !greeting_s.is_empty() { println!("    greeting e.g.: {greeting_s:?}"); }
        if !punct_s.is_empty() { println!("    trailing-punct e.g.: {punct_s:?}"); }
    }

    // Name-based duplication sizing (independent of embeddings): distinct-id
    // entity pairs that are normalized-equal (=> cross-type, since same
    // name+type collapses to one entity_id) or substring-contained. This
    // sizes the fuzzy-merge opportunity even with embeddings absent.
    {
        let rows = session
            .query_with(&format!(
                "MATCH (e:Entity) RETURN e.name AS name, coalesce(e.entity_type, '') AS typ \
                 LIMIT {max_entities}"
            ))
            .fetch_all()
            .await?;
        let mut ents: Vec<String> = Vec::new();
        for r in rows.rows() {
            let name: String = r.get("name").unwrap_or_default();
            let norm = name.trim().to_lowercase();
            if !norm.is_empty() {
                ents.push(norm);
            }
        }
        let n = ents.len();
        let (mut norm_eq, mut substr_pairs) = (0u64, 0u64);
        let mut samples: Vec<(String, String)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let (a, b) = (&ents[i], &ents[j]);
                if a == b {
                    norm_eq += 1;
                    if samples.len() < 25 {
                        samples.push((a.clone(), b.clone()));
                    }
                } else if a.len() > 3
                    && b.len() > 3
                    && (a.contains(b.as_str()) || b.contains(a.as_str()))
                {
                    substr_pairs += 1;
                    if samples.len() < 25 {
                        samples.push((a.clone(), b.clone()));
                    }
                }
            }
        }
        println!("  name-based near-dup pairs (no embeddings needed):");
        println!("    normalized-equal (cross-type): {norm_eq}");
        println!("    substring-contained pairs    : {substr_pairs}");
        for (a, b) in &samples {
            println!("    ~  {a:?}  <>  {b:?}");
        }
    }
    if ent_emb > 0 {
        let rows = session
            .query_with(&format!(
                "MATCH (e:Entity) WHERE e.embedding IS NOT NULL \
                 RETURN id(e) AS id, e.name AS name, coalesce(e.entity_type, '') AS typ, \
                 e.embedding AS emb LIMIT {max_entities}"
            ))
            .fetch_all()
            .await?;
        struct Ent {
            name: String,
            typ: String,
            emb: Vec<f32>,
        }
        let mut ents: Vec<Ent> = Vec::new();
        for r in rows.rows() {
            if let Ok(emb) = r.get::<Vec<f32>>("emb") {
                ents.push(Ent {
                    name: r.get("name").unwrap_or_default(),
                    typ: r.get("typ").unwrap_or_default(),
                    emb,
                });
            }
        }
        let n = ents.len();
        println!(
            "  loaded {n} entities with embedding (cap {max_entities}); pairwise cosine over \
             distinct-name pairs..."
        );
        let (mut high, mut mid) = (0u64, 0u64);
        let (mut high_same, mut high_cross) = (0u64, 0u64);
        let mut samples_high: Vec<(String, String, f32, bool)> = Vec::new();
        let mut samples_mid: Vec<(String, String, f32, bool)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                // Identical names already share an entity_id and merge via
                // exact match, so only distinct names matter for fuzzy merge.
                if ents[i].name.eq_ignore_ascii_case(&ents[j].name) {
                    continue;
                }
                let c = cosine(&ents[i].emb, &ents[j].emb);
                let same_type = ents[i].typ == ents[j].typ;
                if c >= 0.92 {
                    high += 1;
                    if same_type {
                        high_same += 1;
                    } else {
                        high_cross += 1;
                    }
                    if samples_high.len() < 40 {
                        samples_high.push((
                            ents[i].name.clone(),
                            ents[j].name.clone(),
                            c,
                            same_type,
                        ));
                    }
                } else if c >= 0.85 {
                    mid += 1;
                    if samples_mid.len() < 40 {
                        samples_mid.push((
                            ents[i].name.clone(),
                            ents[j].name.clone(),
                            c,
                            same_type,
                        ));
                    }
                }
            }
        }
        println!("  near-duplicate pairs (distinct names):");
        println!("    cosine >= 0.92  : {high}  (same-type {high_same}, cross-type {high_cross})");
        println!("    cosine 0.85-0.92: {mid}");
        println!("  --- sample >=0.92 pairs (manual precision read; CROSS = different type) ---");
        for (a, b, c, st) in &samples_high {
            println!("    {c:.3} [{}]  {a}  <>  {b}", if *st { "same" } else { "CROSS" });
        }
        println!("  --- sample 0.85-0.92 pairs ---");
        for (a, b, c, st) in &samples_mid {
            println!("    {c:.3} [{}]  {a}  <>  {b}", if *st { "same" } else { "CROSS" });
        }
    }

    println!("\n== SUPPORTED_BY weights ==");
    let rows = session
        .query_with("MATCH (:Fact)-[r:SUPPORTED_BY]->(:Observation) RETURN r.weight AS w LIMIT 100000")
        .fetch_all()
        .await?;
    let ws: Vec<f64> = rows
        .rows()
        .iter()
        .filter_map(|r| r.get::<f64>("w").ok())
        .collect();
    if ws.is_empty() {
        println!("  no SUPPORTED_BY edges (or no weight property present)");
    } else {
        let n = ws.len() as f64;
        let mean = ws.iter().sum::<f64>() / n;
        let min = ws.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = ws.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let std = (ws.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n).sqrt();
        let constant_one = ws.iter().all(|w| (*w - 1.0).abs() < 1e-9);
        println!(
            "  edges sampled={}  mean={mean:.4}  min={min:.4}  max={max:.4}  std={std:.4}",
            ws.len()
        );
        println!("  all exactly 1.0 (inert, pre-1c): {constant_one}");
    }

    drop(session);
    if let Ok(kb_owned) = std::sync::Arc::try_unwrap(kb) {
        kb_owned.shutdown().await.ok();
    }
    Ok(())
}
