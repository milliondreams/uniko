//! Unit tests for SRL-anchored observation extraction.
//!
//! Each fixture builds a synthetic `SrlFrame` and calls
//! `extract_with_rules` with empty DEP arcs but populated frames,
//! verifying the bundled `srl_action*` patterns produce the expected
//! `DepObservation` content. No ONNX inference required — these tests
//! are pure rule-engine assertions.

#[cfg(feature = "onnx")]
use uniko_extract::ingest::context::SentenceContext;
#[cfg(feature = "onnx")]
use uniko_extract::nlp::types::{SrlArg, SrlFrame};
#[cfg(feature = "onnx")]
use uniko_extract::observations::rules_engine::{extract_with_rules, load_bundled_rules};

#[cfg(feature = "onnx")]
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

#[cfg(feature = "onnx")]
fn run(frames: &[SrlFrame]) -> Vec<String> {
    let rules = load_bundled_rules();
    let mut ctx = SentenceContext::new("Caroline", vec!["Melanie".into()]);
    extract_with_rules(
        rules,
        &[], // no words
        &[], // no POS
        &[], // no DEP arcs — DEP patterns can't fire
        &[], // no POS labels
        frames,
        "Caroline",
        &mut ctx,
    )
    .into_iter()
    .map(|o| o.content)
    .collect()
}

#[cfg(feature = "onnx")]
#[test]
fn srl_action_emits_clean_svo() {
    // "Caroline bought a yellow dress"
    let f = frame("bought", &[("ARG0", "Caroline"), ("ARG1", "a yellow dress")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t == "Caroline bought a yellow dress"),
        "expected the basic SVO observation, got: {texts:?}",
    );
}

#[cfg(feature = "onnx")]
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
    // The fully-modified template wins since both time + location are present.
    assert!(
        texts.iter().any(|t| t.contains("She")
            && t.contains("gave")
            && t.contains("the necklace")
            && t.contains("yesterday")
            && t.contains("in her kitchen")),
        "expected fully-modified observation, got: {texts:?}",
    );
}

#[cfg(feature = "onnx")]
#[test]
fn srl_action_temporal_only_fires_on_intransitive_verb() {
    // "I went yesterday" — no ARG1
    let f = frame("went", &[("ARG0", "I"), ("ARGM-TMP", "yesterday")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t == "I went yesterday"),
        "expected temporal-only observation, got: {texts:?}",
    );
}

#[cfg(feature = "onnx")]
#[test]
fn srl_action_locative_only_fires_on_intransitive_verb() {
    // "I went home"
    let f = frame("went", &[("ARG0", "I"), ("ARGM-LOC", "home")]);
    let texts = run(&[f]);
    assert!(
        texts.iter().any(|t| t == "I went home"),
        "expected locative-only observation, got: {texts:?}",
    );
}

#[cfg(feature = "onnx")]
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

#[cfg(feature = "onnx")]
#[test]
fn srl_dedupes_identical_text_within_frame() {
    // Two frames producing the same text — dedup by text in matcher.
    let f1 = frame("ate", &[("ARG0", "Caroline"), ("ARG1", "lunch")]);
    let f2 = frame("ate", &[("ARG0", "Caroline"), ("ARG1", "lunch")]);
    let texts = run(&[f1, f2]);
    let count = texts
        .iter()
        .filter(|t| t == &"Caroline ate lunch")
        .count();
    assert_eq!(count, 1, "expected dedup to one entry, got: {texts:?}");
}
