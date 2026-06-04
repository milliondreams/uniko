//! Verifies that cross-modal recall channels stay dormant in a
//! text-only corpus.
//!
//! Phase 3 lands the plumbing (`ModalityPresence` cache,
//! `RecallConfig` toggles, `modality::*_channel_active` predicates)
//! but does NOT activate the channels in default text-only flows. The
//! Track B ingest paths flip both the corpus-side flag and the config
//! toggle when they land.
//!
//! This test is the headline acceptance gate for Phase 3: with no
//! image / audio / video content in the KB, the activation predicates
//! must return `false` regardless of whether the per-channel config
//! toggles are flipped on.
//
use uniko_memory::recall::RecallConfig;
use uniko_memory::recall::modality::{
    RecallCounters, audio_channel_active, image_channel_active, multimodal_channel_active,
};
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

#[tokio::test]
async fn dormant_in_text_only_corpus_with_toggles_off() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");
    let presence = kb
        .read_modality_presence()
        .await
        .expect("read modality_presence");

    // Fresh KB → singleton was written by `init_kb_stats`; every
    // modality flag defaults to `false`.
    assert!(!presence.has_image_content);
    assert!(!presence.has_audio_content);
    assert!(!presence.has_video_content);
    assert!(!presence.has_multimodal_indexed);

    let cfg = RecallConfig::default();
    // RecallConfig defaults: all four channels off.
    assert!(!cfg.enable_image_channel);
    assert!(!cfg.enable_audio_channel);
    assert!(!cfg.enable_video_channel);
    assert!(!cfg.enable_multimodal_channel);

    // Activation predicates: both signals required → all dormant.
    assert!(!image_channel_active(&presence, cfg.enable_image_channel));
    assert!(!audio_channel_active(&presence, cfg.enable_audio_channel));
    assert!(!multimodal_channel_active(
        &presence,
        cfg.enable_multimodal_channel
    ));

    let counters = RecallCounters::new();
    assert_eq!(counters.total(), 0);
    kb.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn dormant_in_text_only_corpus_even_with_toggles_on() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");
    let presence = kb
        .read_modality_presence()
        .await
        .expect("read modality_presence");

    // Flip every channel toggle on. Presence flags remain false in a
    // text-only KB, so the activation predicates MUST still return
    // `false` — the corpus-side gate is what keeps channels dormant
    // in the common case.
    assert!(!image_channel_active(&presence, true));
    assert!(!audio_channel_active(&presence, true));
    assert!(!multimodal_channel_active(&presence, true));

    kb.shutdown().await.expect("shutdown");
}

#[tokio::test]
async fn bump_modality_presence_activates_the_corresponding_channel() {
    let kb = KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB");

    // Pre-bump: image dormant.
    let presence = kb.read_modality_presence().await.unwrap();
    assert!(!image_channel_active(&presence, true));

    // Bump: flip image flag.
    kb.bump_modality_presence("image").await.unwrap();
    let presence = kb.read_modality_presence().await.unwrap();
    assert!(presence.has_image_content);
    // Channel now eligible to fire — but only when the toggle is on.
    assert!(image_channel_active(&presence, true));
    assert!(!image_channel_active(&presence, false));
    // Audio still dormant.
    assert!(!audio_channel_active(&presence, true));

    kb.shutdown().await.expect("shutdown");
}
