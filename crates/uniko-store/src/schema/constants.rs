//! String constants for node labels, edge types, and schema configuration.
//!
//! Centralising names here prevents typos and enables programmatic
//! completeness checks in tests.

/// Default embedding vector dimensions (fastembed all-MiniLM-L6-v2).
pub const DEFAULT_VECTOR_DIM: usize = 384;

/// HNSW `m` parameter — max bi-directional links per node.
pub const HNSW_M: u32 = 16;

/// HNSW `ef_construction` parameter — search width during build.
pub const HNSW_EF_CONSTRUCTION: u32 = 100;

/// All 20 node labels, ordered by schema layer.
pub mod labels {
    // Layer 0 — Participants
    pub const PARTICIPANT: &str = "Participant";

    // Layer 1 — Goals, Tasks, Sessions
    pub const GOAL: &str = "Goal";
    pub const TASK: &str = "Task";
    pub const SESSION: &str = "Session";

    // Layer 2 — Episodic Memory
    pub const MESSAGE: &str = "Message";
    pub const ACTION: &str = "Action";
    pub const EPISODE: &str = "Episode";

    // Layer 3 — Artifacts & Chunks
    pub const ARTIFACT: &str = "Artifact";
    pub const CHUNK: &str = "Chunk";

    // Layer 4 — Semantic Memory
    pub const ENTITY: &str = "Entity";
    pub const OBSERVATION: &str = "Observation";
    pub const FACT: &str = "Fact";
    pub const TOPIC: &str = "Topic";
    pub const SUMMARY: &str = "Summary";

    // Layer 5 — Procedural Memory
    pub const PROCEDURE: &str = "Procedure";
    pub const RULE: &str = "Rule";

    // Layer 6 — Meta-Memory
    pub const CONSOLIDATION_CYCLE: &str = "ConsolidationCycle";
    pub const DEAD_LETTER: &str = "DeadLetter";

    // Layer 7 — Organization
    pub const ORGANIZATION: &str = "Organization";
    pub const TEAM: &str = "Team";

    /// Every node label in layer order.
    pub const ALL: &[&str] = &[
        PARTICIPANT,
        GOAL,
        TASK,
        SESSION,
        MESSAGE,
        ACTION,
        EPISODE,
        ARTIFACT,
        CHUNK,
        ENTITY,
        OBSERVATION,
        FACT,
        TOPIC,
        SUMMARY,
        PROCEDURE,
        RULE,
        CONSOLIDATION_CYCLE,
        DEAD_LETTER,
        ORGANIZATION,
        TEAM,
    ];
}

/// All edge type names.
pub mod edges {
    // Layer 1 edges
    pub const OWNED_BY: &str = "OWNED_BY";
    pub const PARENT_GOAL: &str = "PARENT_GOAL";
    pub const PART_OF: &str = "PART_OF";
    pub const ASSIGNED_TO: &str = "ASSIGNED_TO";
    pub const DEPENDS_ON: &str = "DEPENDS_ON";
    pub const SUBTASK_OF: &str = "SUBTASK_OF";
    pub const FOR_TASK: &str = "FOR_TASK";
    pub const FOR_GOAL: &str = "FOR_GOAL";
    pub const PARTICIPATED_IN: &str = "PARTICIPATED_IN";

    // Layer 2 edges
    pub const SENT_BY: &str = "SENT_BY";
    pub const ADDRESSED_TO: &str = "ADDRESSED_TO";
    pub const IN_SESSION: &str = "IN_SESSION";
    pub const NEXT: &str = "NEXT";
    pub const PERFORMED_BY: &str = "PERFORMED_BY";
    pub const TRIGGERED_BY: &str = "TRIGGERED_BY";
    pub const PRODUCED: &str = "PRODUCED";
    pub const NEXT_ACTION: &str = "NEXT_ACTION";
    pub const RECORDED_BY: &str = "RECORDED_BY";
    pub const INVOLVES: &str = "INVOLVES";
    pub const MENTIONS: &str = "MENTIONS";
    pub const FOLLOWED_BY: &str = "FOLLOWED_BY";

    // Layer 3 edges
    pub const HAS_CHUNK: &str = "HAS_CHUNK";
    pub const CREATED_BY: &str = "CREATED_BY";
    pub const MODIFIED_BY: &str = "MODIFIED_BY";

    // Layer 4 edges
    pub const OBSERVED_IN: &str = "OBSERVED_IN";
    pub const OBSERVED_DURING: &str = "OBSERVED_DURING";
    pub const ABOUT: &str = "ABOUT";
    pub const SUPPORTED_BY: &str = "SUPPORTED_BY";
    pub const DERIVED_BY: &str = "DERIVED_BY";
    pub const DERIVED_FROM: &str = "DERIVED_FROM";
    pub const INVALIDATES: &str = "INVALIDATES";
    pub const SHARED_FROM: &str = "SHARED_FROM";
    pub const BELONGS_TO: &str = "BELONGS_TO";
    pub const SUMMARIZES: &str = "SUMMARIZES";

    // Layer 5 edges
    pub const OPERATES_ON: &str = "OPERATES_ON";
    pub const USED_IN: &str = "USED_IN";
    pub const SUPERSEDES: &str = "SUPERSEDES";
    pub const COVERS: &str = "COVERS";

    // Layer 6 edges
    pub const PROCESSED: &str = "PROCESSED";
    pub const INVOLVED: &str = "INVOLVED";
    pub const CREATED: &str = "CREATED";
    pub const INVALIDATED: &str = "INVALIDATED";
    pub const PROMOTED: &str = "PROMOTED";
    pub const APPLIED_RULE: &str = "APPLIED_RULE";

    // Layer 7 edges
    pub const MEMBER_OF: &str = "MEMBER_OF";
    pub const PART_OF_TEAM: &str = "PART_OF_TEAM";
    pub const TEAM_IN_ORG: &str = "TEAM_IN_ORG";

    /// Every edge type name.
    pub const ALL: &[&str] = &[
        OWNED_BY,
        PARENT_GOAL,
        PART_OF,
        ASSIGNED_TO,
        DEPENDS_ON,
        SUBTASK_OF,
        FOR_TASK,
        FOR_GOAL,
        PARTICIPATED_IN,
        SENT_BY,
        ADDRESSED_TO,
        IN_SESSION,
        NEXT,
        PERFORMED_BY,
        TRIGGERED_BY,
        PRODUCED,
        NEXT_ACTION,
        RECORDED_BY,
        INVOLVES,
        MENTIONS,
        FOLLOWED_BY,
        HAS_CHUNK,
        CREATED_BY,
        MODIFIED_BY,
        OBSERVED_IN,
        OBSERVED_DURING,
        ABOUT,
        SUPPORTED_BY,
        DERIVED_BY,
        DERIVED_FROM,
        INVALIDATES,
        SHARED_FROM,
        BELONGS_TO,
        SUMMARIZES,
        OPERATES_ON,
        USED_IN,
        SUPERSEDES,
        COVERS,
        PROCESSED,
        INVOLVED,
        CREATED,
        INVALIDATED,
        PROMOTED,
        APPLIED_RULE,
        MEMBER_OF,
        PART_OF_TEAM,
        TEAM_IN_ORG,
    ];
}
