//! Unit tests for SRL-anchored observation extraction.
//!
//! Each fixture builds a synthetic `SrlFrame` and calls
//! `extract_with_rules` with empty DEP arcs but populated frames,
//! verifying the bundled `srl_action*` patterns produce the expected
//! `DepObservation` content. No ONNX inference required — these tests
//! are pure rule-engine assertions.
//!
//! Two failure modes that the dmas-memory bench surfaced are covered
//! here as forward-looking specifications:
//!
//! - **Bug 1 — destination roles dropped.** PropBank tags the
//!   destination of motion verbs ("go", "travel", "move") as ARG2 or
//!   ARG4, recipients of transfer verbs ("give", "send") as ARG2, and
//!   sources/start points as ARG3.  The bundled rules
//!   (`assets/english.yml`) only capture ARG0/ARG1/ARGM-TMP/ARGM-LOC,
//!   so `"I went to the LGBTQ support group yesterday"` collapses to
//!   `"I went yesterday"`.  Tests `srl_action_captures_arg*_*`
//!   pin the expected post-fix shape.
//!
//! - **Bug 2 — pronoun resolution skipped for SRL.**
//!   `try_match_srl` in `rules_engine/matcher.rs` writes the raw ARG0
//!   surface form into the `{agent}` capture without calling
//!   `resolve_subject`.  DEP-extracted observations rewrite "I" →
//!   speaker; SRL-extracted ones don't.  Tests
//!   `srl_action_*_resolves_*_pronoun` pin the expected post-fix
//!   shape.

#![cfg(feature = "onnx")]

use uniko_extract::ingest::context::SentenceContext;
use uniko_extract::nlp::types::{SrlArg, SrlFrame};
use uniko_extract::observations::rules_engine::{extract_with_rules, load_bundled_rules};

// ── Fixture helpers ─────────────────────────────────────────────────

fn frame(predicate: &str, args: &[(&str, &str)]) -> SrlFrame {
    SrlFrame {
        predicate_idx: 0,
        predicate_word: predicate.to_string(),
        args: args
            .iter()
            .enumerate()
            .map(|(i, (role, text))| SrlArg {
                role: (*role).to_string(),
                text: (*text).to_string(),
                start_word: i,
                end_word: i + 1,
            })
            .collect(),
    }
}

/// Run extraction with `Caroline` as the speaker and `Melanie` as the
/// other participant, matching the conventional fixture used across
/// observation tests.
fn run(frames: &[SrlFrame]) -> Vec<String> {
    run_for(frames, "Caroline", &["Melanie"])
}

/// Same as [`run`] but lets callers override the participant lineup —
/// needed for second-person resolution where "you" should resolve to
/// the *other* speaker.
fn run_for(frames: &[SrlFrame], speaker: &str, other_speakers: &[&str]) -> Vec<String> {
    let rules = load_bundled_rules();
    let others: Vec<String> = other_speakers.iter().map(|s| (*s).to_string()).collect();
    let mut ctx = SentenceContext::new(speaker, others);
    extract_with_rules(rules, &[], &[], &[], &[], frames, speaker, &mut ctx)
        .into_iter()
        .map(|o| o.content)
        .collect()
}

// ── Existing behaviour — basic SVO and modifier templates ───────────

#[test]
fn srl_action_emits_clean_svo() {
    // "Caroline bought a yellow dress"
    let f = frame(
        "bought",
        &[("ARG0", "Caroline"), ("ARG1", "a yellow dress")],
    );
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t == "Caroline bought a yellow dress"),
        "expected the basic SVO observation, got: {texts:?}",
    );
}

#[test]
fn srl_action_includes_argm_tmp_and_loc() {
    // "She gave me the necklace yesterday in her kitchen"
    let f = frame(
        "gave",
        &[
            ("ARG0", "She"),
            ("ARG1", "the necklace"),
            ("ARGM-TMP", "yesterday"),
            ("ARGM-LOC", "in her kitchen"),
        ],
    );
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t.contains("She")
            && t.contains("gave")
            && t.contains("the necklace")
            && t.contains("yesterday")
            && t.contains("in her kitchen")),
        "expected fully-modified observation, got: {texts:?}",
    );
}

#[test]
fn srl_action_skipped_when_no_arg0() {
    // No ARG0 → required-capture failure → no observation
    let f = frame("ran", &[("ARG1", "the marathon")]);
    let texts = run(&[f]);
    assert!(
        texts.is_empty(),
        "expected no observations when ARG0 missing, got: {texts:?}",
    );
}

#[test]
fn srl_dedupes_identical_text_within_frame() {
    let f1 = frame("ate", &[("ARG0", "Caroline"), ("ARG1", "lunch")]);
    let f2 = frame("ate", &[("ARG0", "Caroline"), ("ARG1", "lunch")]);
    let texts = run(&[f1, f2]);
    let count = texts.iter().filter(|t| t == &"Caroline ate lunch").count();
    assert_eq!(count, 1, "expected dedup to one entry, got: {texts:?}");
}

// ── Bug 2 — pronoun resolution on SRL frames ────────────────────────

#[test]
fn srl_action_resolves_first_person_arg0_to_speaker() {
    // "I bought a yellow dress" by Caroline → "Caroline bought a yellow dress"
    let f = frame("bought", &[("ARG0", "I"), ("ARG1", "a yellow dress")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t == "Caroline bought a yellow dress"),
        "expected 'I' to resolve to speaker 'Caroline', got: {texts:?}",
    );
    assert!(
        !texts.iter().any(|t| t.starts_with("I ")),
        "expected no raw 'I' subject in output, got: {texts:?}",
    );
}

#[test]
fn srl_action_resolves_first_person_we_to_speaker() {
    // "We picked the kids up" by Caroline → resolves to speaker
    let f = frame("picked", &[("ARG0", "We"), ("ARG1", "the kids up")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t.starts_with("Caroline ")),
        "expected 'We' to resolve to speaker 'Caroline', got: {texts:?}",
    );
}

#[test]
fn srl_action_resolves_second_person_to_other_speaker() {
    // Caroline addresses Melanie: "You bought a yellow dress" → "Melanie bought ..."
    let f = frame("bought", &[("ARG0", "You"), ("ARG1", "a yellow dress")]);
    let texts = run_for(&[f], "Caroline", &["Melanie"]);
    assert!(
        texts.iter().any(|t| t == "Melanie bought a yellow dress"),
        "expected 'You' to resolve to other_speakers[0] 'Melanie', got: {texts:?}",
    );
}

#[test]
fn srl_action_temporal_only_resolves_arg0_pronoun() {
    // "I went yesterday" by Caroline → "Caroline went yesterday"
    let f = frame("went", &[("ARG0", "I"), ("ARGM-TMP", "yesterday")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t == "Caroline went yesterday"),
        "expected 'I' to resolve to 'Caroline' in temporal-only pattern, got: {texts:?}",
    );
}

#[test]
fn srl_action_locative_only_resolves_arg0_pronoun() {
    // "I went home" by Caroline → "Caroline went home"
    let f = frame("went", &[("ARG0", "I"), ("ARGM-LOC", "home")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t == "Caroline went home"),
        "expected 'I' to resolve to 'Caroline' in locative-only pattern, got: {texts:?}",
    );
}

// ── Pattern matching — intransitive shapes still fire correctly ─────

/// Regression-only: confirm a frame matching just `srl_action_temporal_only`
/// still yields *some* observation.  The deeper assertion — that "I" is
/// resolved to the speaker name — lives in
/// `srl_action_temporal_only_resolves_arg0_pronoun` below.  Kept as a
/// separate test so a future refactor that breaks pattern dispatch
/// surfaces here even if the pronoun-resolution test stays green.
#[test]
fn srl_action_temporal_only_emits_some_observation() {
    let f = frame("went", &[("ARG0", "I"), ("ARGM-TMP", "yesterday")]);
    let texts = run(&[f]);
    assert!(
        !texts.is_empty(),
        "expected at least one observation from temporal-only pattern, got nothing",
    );
}

/// Mirror of `srl_action_temporal_only_emits_some_observation` for the
/// locative pattern.  Deeper assertion in
/// `srl_action_locative_only_resolves_arg0_pronoun`.
#[test]
fn srl_action_locative_only_emits_some_observation() {
    let f = frame("went", &[("ARG0", "I"), ("ARGM-LOC", "home")]);
    let texts = run(&[f]);
    assert!(
        !texts.is_empty(),
        "expected at least one observation from locative-only pattern, got nothing",
    );
}

// ── Bug 1 — destination/recipient/source roles (ARG2/ARG3/ARG4) ─────

#[test]
fn srl_action_captures_arg4_destination_for_motion_verbs() {
    // PropBank go.01: ARG0=walker, ARG4=destination.
    // "I went to the LGBTQ support group" → "Caroline went to the LGBTQ support group"
    let f = frame(
        "went",
        &[("ARG0", "I"), ("ARG4", "to the LGBTQ support group")],
    );
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t.contains("LGBTQ support group")),
        "expected ARG4 destination to surface in observation, got: {texts:?}",
    );
    assert!(
        texts.iter().any(|t| t.starts_with("Caroline ")),
        "expected pronoun resolution alongside destination, got: {texts:?}",
    );
}

#[test]
fn srl_action_captures_arg4_with_temporal_anchor() {
    // The canonical regression — "I went to the LGBTQ support group yesterday".
    // Both pronoun and destination must survive, and temporal anchor
    // must appear in the rendered content.
    let f = frame(
        "went",
        &[
            ("ARG0", "I"),
            ("ARG4", "to the LGBTQ support group"),
            ("ARGM-TMP", "yesterday"),
        ],
    );
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t.contains("Caroline")
            && t.contains("LGBTQ support group")
            && t.contains("yesterday")),
        "expected combined ARG0 + ARG4 + ARGM-TMP rendering, got: {texts:?}",
    );
}

#[test]
fn srl_action_captures_arg2_recipient_for_transfer_verbs() {
    // PropBank give.01: ARG0=giver, ARG1=thing given, ARG2=recipient.
    // "Caroline gave Melanie a necklace" — ARG2 must appear in output.
    let f = frame(
        "gave",
        &[
            ("ARG0", "Caroline"),
            ("ARG1", "a necklace"),
            ("ARG2", "to Melanie"),
        ],
    );
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t.contains("a necklace") && t.contains("Melanie")),
        "expected ARG2 recipient to surface alongside theme, got: {texts:?}",
    );
}

#[test]
fn srl_action_captures_arg3_source_for_motion_verbs() {
    // PropBank come.01: ARG0=mover, ARG3=start point.
    // "Caroline came from the kitchen" — ARG3 source must surface.
    let f = frame("came", &[("ARG0", "Caroline"), ("ARG3", "from the kitchen")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t.contains("from the kitchen")),
        "expected ARG3 source to surface in observation, got: {texts:?}",
    );
}

#[test]
fn srl_action_arg4_destination_falls_back_to_intransitive_when_only_dest_present() {
    // Pure motion verb with no ARG1 but a destination — must still
    // produce an observation that includes the destination.  Today's
    // rules would dispatch this to `srl_action_temporal_only` /
    // `srl_action_locative_only` and lose the ARG4 text.
    let f = frame("went", &[("ARG0", "Caroline"), ("ARG4", "to the park")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t.contains("to the park") || t.contains("the park")),
        "expected ARG4-only motion verb to retain destination, got: {texts:?}",
    );
}

// ── Light-verb / quality gates — must still reject empty SVO ────────

#[test]
fn srl_action_drops_light_verb_only_with_no_complement() {
    // "I do" — light verb, no theme, no temporal, no location, no
    // destination.  Should be dropped by the light-verb filter.
    let f = frame("do", &[("ARG0", "I")]);
    let texts = run(&[f]);
    assert!(
        texts.is_empty(),
        "expected light-verb-only frame with no complement to drop, got: {texts:?}",
    );
}

#[test]
fn srl_action_keeps_light_verb_with_destination() {
    // "I went to the park" — destination IS the specifying complement,
    // so the observation should survive despite "went" alone being a
    // light motion verb.
    let f = frame("went", &[("ARG0", "I"), ("ARG4", "to the park")]);
    let texts = run(&[f]);
    assert!(
        !texts.is_empty(),
        "expected destination to count as specifying complement, got nothing",
    );
}
