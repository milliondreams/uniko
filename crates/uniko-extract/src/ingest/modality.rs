//! Extension point for non-text modalities (image / audio / video).
//!
//! Text, code, markup, structured, and PDF ingest are built in. Image,
//! audio, and video are routed to a [`ModalityExtractor`] registered in a
//! [`ModalityRegistry`]; with no extractor registered, ingest of those
//! modalities returns [`UnikoError::Unsupported`]. No ASR/VLM/captioning
//! model ships here — this is the seam a host plugs one into.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uniko_pipes::content::Modality;
use uniko_store::{KnowledgeBase, UnikoError};

use uniko_pipes::types::IngestSource;

use super::artifact::ArtifactIngestResult;

/// Extracts a single non-text modality into the graph as an artifact.
///
/// Implement for a concrete modality (e.g. an image captioner) and register
/// it in a [`ModalityRegistry`]; the unified ingest dispatch then routes
/// matching MIME types to it.
#[async_trait]
pub trait ModalityExtractor: Send + Sync + std::fmt::Debug {
    /// The modality this extractor handles.
    fn modality(&self) -> Modality;

    /// Extract `src` into the graph, returning the created artifact.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError`] on an extraction or write failure.
    async fn extract(
        &self,
        kb: &KnowledgeBase,
        src: &IngestSource,
    ) -> Result<ArtifactIngestResult, UnikoError>;
}

/// A registry of [`ModalityExtractor`]s keyed by [`Modality`].
///
/// The unified ingest dispatch consults this for Image/Audio/Video; an empty
/// registry (the [`Default`]) yields [`UnikoError::Unsupported`] for them.
#[derive(Debug, Default, Clone)]
pub struct ModalityRegistry {
    extractors: HashMap<Modality, Arc<dyn ModalityExtractor>>,
}

impl ModalityRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `extractor` for its declared [`Modality`], replacing any
    /// previous entry for that modality.
    pub fn register(&mut self, extractor: Arc<dyn ModalityExtractor>) {
        self.extractors.insert(extractor.modality(), extractor);
    }

    /// The extractor for `modality`, if registered.
    #[must_use]
    pub fn get(&self, modality: Modality) -> Option<&Arc<dyn ModalityExtractor>> {
        self.extractors.get(&modality)
    }
}
