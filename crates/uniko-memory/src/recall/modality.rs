//! Cross-modal recall plumbing — landed dormant in Phase 3.
//!
//! The cross-modal channels (image, audio, multimodal) sit on top of
//! the existing recall cascade as additional per-modality vector
//! candidates. They activate automatically when:
//!
//! 1. The KB's `:KnowledgeBaseStats.modality_presence` flag for that
//!    modality is `true` (any `:Artifact.<modality>_embedding` is
//!    populated), AND
//! 2. The matching toggle on [`crate::recall::RecallConfig`] is on.
//!
//! In Phase 3 both conditions default to `false` in text-only corpora,
//! so the channels never fire. Track B (xervo-dependent) ingest paths
//! flip the corpus-side flag via `KnowledgeBase::bump_modality_presence`
//! and at the same time provide a reason to flip the config toggle.
//!
//! A [`RecallCounters`] handle is exposed for tests: instrument the
//! counters at recall start, assert image/audio/video remain at zero
//! for text-only corpora.
//
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub use uniko_store::storage::kb_stats::ModalityPresence;

/// Test-only counter handle threaded through recall to verify that
/// cross-modal channels stay dormant in text-only corpora.
///
/// Wrapped in `Arc` so the caller can clone-share between the recall
/// invocation and the post-call assertion. All increments use
/// `Ordering::Relaxed` because the counters are read after the recall
/// future has fully resolved.
#[derive(Debug, Clone, Default)]
pub struct RecallCounters {
    image_channel_fires: Arc<AtomicUsize>,
    audio_channel_fires: Arc<AtomicUsize>,
    multimodal_channel_fires: Arc<AtomicUsize>,
}

impl RecallCounters {
    /// Construct a fresh counter handle. All counts start at zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the image channel counter. Called inside the dormant
    /// branch when the image channel actually fires.
    pub fn bump_image(&self) {
        self.image_channel_fires.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the audio channel counter.
    pub fn bump_audio(&self) {
        self.audio_channel_fires.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the multimodal channel counter.
    pub fn bump_multimodal(&self) {
        self.multimodal_channel_fires
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Read the image channel counter.
    #[must_use]
    pub fn image(&self) -> usize {
        self.image_channel_fires.load(Ordering::Relaxed)
    }

    /// Read the audio channel counter.
    #[must_use]
    pub fn audio(&self) -> usize {
        self.audio_channel_fires.load(Ordering::Relaxed)
    }

    /// Read the multimodal channel counter.
    #[must_use]
    pub fn multimodal(&self) -> usize {
        self.multimodal_channel_fires.load(Ordering::Relaxed)
    }

    /// Total cross-modal channel fires across all modalities. Useful
    /// for one-shot "did anything cross-modal fire?" assertions.
    #[must_use]
    pub fn total(&self) -> usize {
        self.image() + self.audio() + self.multimodal()
    }
}

/// Decision: does the image channel run for this query against this
/// corpus? `false` for text-only corpora (presence flag off) or when
/// the channel toggle is off.
///
/// Centralised so test instrumentation hits exactly one branch. The
/// actual vector-search call is gated on this returning `true`.
#[must_use]
pub fn image_channel_active(presence: &ModalityPresence, image_channel_enabled: bool) -> bool {
    presence.has_image_content && image_channel_enabled
}

/// Mirror of [`image_channel_active`] for audio.
#[must_use]
pub fn audio_channel_active(presence: &ModalityPresence, audio_channel_enabled: bool) -> bool {
    presence.has_audio_content && audio_channel_enabled
}

/// Mirror of [`image_channel_active`] for the multimodal channel
/// (Cohere v4 / Gemini Embed 2 mixed embedding column).
#[must_use]
pub fn multimodal_channel_active(
    presence: &ModalityPresence,
    multimodal_channel_enabled: bool,
) -> bool {
    presence.has_multimodal_indexed && multimodal_channel_enabled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counters_default_zero() {
        let c = RecallCounters::new();
        assert_eq!(c.total(), 0);
        assert_eq!(c.image(), 0);
    }

    #[test]
    fn test_counters_bump() {
        let c = RecallCounters::new();
        c.bump_image();
        c.bump_audio();
        assert_eq!(c.image(), 1);
        assert_eq!(c.audio(), 1);
        assert_eq!(c.total(), 2);
    }

    #[test]
    fn test_dormant_in_text_only_corpus() {
        // Default ModalityPresence — all flags false.
        let p = ModalityPresence::default();
        // Channel toggle on, but presence false: channel does NOT activate.
        assert!(!image_channel_active(&p, true));
        assert!(!audio_channel_active(&p, true));
        assert!(!multimodal_channel_active(&p, true));
    }

    #[test]
    fn test_active_requires_both_signals() {
        let p = ModalityPresence {
            has_image_content: true,
            ..ModalityPresence::default()
        };
        assert!(image_channel_active(&p, true));
        assert!(!image_channel_active(&p, false)); // toggle gates it off
        assert!(!audio_channel_active(&p, true)); // presence gates it off
    }
}
