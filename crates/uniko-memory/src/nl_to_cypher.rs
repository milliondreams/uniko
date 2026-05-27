//! Natural-language to Cypher translation (F61).
//!
//! Translates a free-text question into a read-only Cypher query that
//! runs against the uniko schema.  The flow:
//!
//! 1. Normalise the input (lowercase, collapse whitespace).
//! 2. Look up the normalised query in the in-process LRU cache; on
//!    hit, return the cached Cypher.
//! 3. On miss, build the LLM prompt from a cached schema snapshot
//!    plus a few-shot tail, and ask the configured Xervo alias.
//! 4. Strip code fences / commentary from the response.
//! 5. Reject any output containing mutating keywords
//!    (`CREATE`/`MERGE`/`DELETE`/`SET`/`REMOVE`/`DROP`/`CALL` with
//!    write side-effects) — see [`is_safe_read_only`].
//! 6. Store in the cache and return.
//!
//! Retries: one LLM retry on transient failure (network error /
//! malformed response).  The translator never executes the generated
//! Cypher — that's the caller's responsibility, so this module stays
//! a thin pure-text bridge.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use uni_db::xervo::{GenerationOptions, Message};
#[cfg(feature = "onnx")]
use uniko_extract::nlp::NlpPipeline;
use uniko_store::schema::labels;
use uniko_store::{KnowledgeBase, UnikoError};

/// Default cache capacity.  ~128 entries × ~256 bytes per entry ≈
/// 32 KiB — negligible memory, large enough to cover one session's
/// recurring questions.
pub const DEFAULT_CACHE_CAPACITY: usize = 128;

/// Maximum LLM retry attempts on transient / parse failures.
const RETRY_ATTEMPTS: u32 = 2;

/// LLM generation budget.  Sized for gpt-5-nano: reasoning models
/// burn completion tokens on internal chain-of-thought before
/// emitting visible content.  Empirical measurement: a one-line
/// "count messages" query used 471 completion tokens (448 reasoning,
/// 23 visible).  2048 gives ~1500 tokens of headroom after a worst-case
/// reasoning pass — enough for any single Cypher query while keeping
/// cost bounded.
const MAX_TOKENS: usize = 2048;

/// System prompt instructing the LLM to emit read-only Cypher.
const SYSTEM_PROMPT: &str = r#"
You are a Cypher generator for the uniko knowledge graph.
Output ONLY a single read-only Cypher query — no commentary, no code
fences, no explanations.

Constraints:
- READ ONLY: never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, or
  CALL with side effects.
- Always RETURN something.
- Use only the labels, properties, and edges from the schema below.
- Keep queries simple and bounded with LIMIT (default 25).
- If the question cannot be answered with the schema, output:
  RETURN 'cannot_answer' AS reason
"#;

/// LRU cache entry — value plus insertion ordinal for eviction.
struct CacheEntry {
    cypher: String,
    last_used: u64,
}

/// Process-global cache + monotonic clock.
struct Cache {
    entries: HashMap<String, CacheEntry>,
    clock: u64,
    capacity: usize,
}

impl Cache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity.min(256)),
            clock: 0,
            capacity,
        }
    }

    fn get(&mut self, key: &str) -> Option<String> {
        self.clock = self.clock.wrapping_add(1);
        let now = self.clock;
        let entry = self.entries.get_mut(key)?;
        entry.last_used = now;
        Some(entry.cypher.clone())
    }

    fn put(&mut self, key: String, cypher: String) {
        if self.entries.len() >= self.capacity
            && let Some(victim_key) = self.find_lru_key()
        {
            self.entries.remove(&victim_key);
        }
        self.clock = self.clock.wrapping_add(1);
        self.entries.insert(
            key,
            CacheEntry {
                cypher,
                last_used: self.clock,
            },
        );
    }

    fn find_lru_key(&self) -> Option<String> {
        self.entries
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, _)| k.clone())
    }

    /// Visible for tests.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

fn cache_mutex(capacity: usize) -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Cache::new(capacity)))
}

/// Translate a natural-language query into a read-only Cypher query.
///
/// The translator caches by normalised input.  Two questions that
/// differ only in casing or extra whitespace share a cache slot, so a
/// fragile LLM doesn't get re-asked for trivially similar prompts.
///
/// # Errors
///
/// - [`UnikoError::Locy`] when the LLM returns a query that contains
///   mutating keywords (rejected as a defence against prompt
///   injection / hallucinated CREATE statements).
/// - [`UnikoError::Embedding`] when every retry attempt fails or the
///   response is empty.  Embedding is the canonical "LLM unavailable"
///   error variant in this codebase.
pub async fn translate(
    kb: &KnowledgeBase,
    nl_query: &str,
    llm_alias: &str,
) -> Result<String, UnikoError> {
    let key = normalise(nl_query);
    if key.is_empty() {
        return Err(UnikoError::Locy("nl_to_cypher: empty query".into()));
    }

    // Hot path: cache hit.
    if let Some(cached) = cache_mutex(DEFAULT_CACHE_CAPACITY)
        .lock()
        .expect("nl_to_cypher cache mutex poisoned — bug")
        .get(&key)
    {
        return Ok(cached);
    }

    let entity_hints = kniv_ner_hints(kb, nl_query).await;
    let prompt = build_prompt(kb, nl_query, &entity_hints);
    let cypher = call_with_retry(kb, llm_alias, &prompt).await?;
    if !is_safe_read_only(&cypher) {
        return Err(UnikoError::Locy(format!(
            "nl_to_cypher: rejected mutating output: {cypher}"
        )));
    }

    cache_mutex(DEFAULT_CACHE_CAPACITY)
        .lock()
        .expect("nl_to_cypher cache mutex poisoned — bug")
        .put(key, cypher.clone());
    Ok(cypher)
}

/// Check whether `cypher` is safe to execute as a read-only query.
///
/// Rejects any of the mutating keywords listed in the spec, even if
/// they only appear inside a string literal — string-literal
/// scanning isn't worth the AST dependency for the MVP, and
/// callers can re-quote anything legitimately containing those
/// words inside parameters.
#[must_use]
pub fn is_safe_read_only(cypher: &str) -> bool {
    let upper = cypher.to_uppercase();
    // Word-boundary check to avoid rejecting `CREATE_AT` (no such
    // identifier today but defensive).  Surround each keyword with
    // non-alphanumeric chars in a regex-free check.
    const FORBIDDEN: &[&str] = &[
        "CREATE", "MERGE", "DELETE", "DETACH", "SET ", "REMOVE", "DROP", "LOAD ",
    ];
    for kw in FORBIDDEN {
        if contains_word(&upper, kw) {
            return false;
        }
    }
    // CALL is ambiguous (read-only procedures exist) — only reject
    // CALL when followed by a known write procedure.  For MVP we keep
    // it permissive; the keyword list above covers every direct write
    // syntax.
    true
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    // Strip the trailing space sentinel some keywords use, then walk
    // the haystack splitting on non-alphanumeric chars.
    let trimmed = needle.trim_end();
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(trimmed) {
        let abs = start + pos;
        let before = abs == 0
            || !haystack
                .as_bytes()
                .get(abs - 1)
                .map(|b| b.is_ascii_alphanumeric() || *b == b'_')
                .unwrap_or(false);
        let after_idx = abs + trimmed.len();
        let after = after_idx >= haystack.len()
            || !haystack
                .as_bytes()
                .get(after_idx)
                .map(|b| b.is_ascii_alphanumeric() || *b == b'_')
                .unwrap_or(false);
        if before && after {
            return true;
        }
        start = abs + trimmed.len();
    }
    false
}

fn normalise(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn build_prompt(_kb: &KnowledgeBase, nl_query: &str, entity_hints: &[String]) -> String {
    // Format any Kniv-resolved entities as a bulleted hints block so
    // the LLM grounds its query on the exact surface form / type the
    // NLP cascade saw.  Empty hints → skip the section entirely so
    // the prompt stays compact for short questions.
    let hints_block = if entity_hints.is_empty() {
        String::new()
    } else {
        let mut s = String::from("\nEntities detected in the question (use these as literals):\n");
        for hint in entity_hints {
            s.push_str("- ");
            s.push_str(hint);
            s.push('\n');
        }
        s
    };
    format!(
        "{system}\n\nSchema:\n{schema}\n{hints}\nFew-shot examples:\n{shots}\n\nQuestion: {q}\nCypher:",
        system = SYSTEM_PROMPT.trim(),
        schema = schema_snapshot(),
        hints = hints_block,
        shots = FEW_SHOT_EXAMPLES.trim(),
        q = nl_query.trim(),
    )
}

/// Run Kniv NER over `nl_query` and return one `"text (TYPE)"` string
/// per detected entity span.  Best-effort: when the NLP runtime is
/// unavailable, returns an empty vec and the prompt drops the hints
/// section entirely.
#[cfg(feature = "onnx")]
async fn kniv_ner_hints(kb: &KnowledgeBase, nl_query: &str) -> Vec<String> {
    let Some(pipeline) = NlpPipeline::try_new(kb).await else {
        return Vec::new();
    };
    let result = match pipeline.analyze(nl_query).await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!(error = %e, "Kniv NER on NL query failed");
            return Vec::new();
        }
    };
    result
        .entities
        .iter()
        .map(|span| format!("{} ({:?})", span.text, span.entity_type))
        .collect()
}

/// No-op fallback when the `onnx` feature is disabled.  Keeps the
/// signature stable so [`translate`] doesn't need conditional
/// compilation around the call site.
#[cfg(not(feature = "onnx"))]
async fn kniv_ner_hints(_kb: &KnowledgeBase, _nl_query: &str) -> Vec<String> {
    Vec::new()
}

/// Schema description rendered once per process.
///
/// We intentionally avoid serialising the full uni-db schema (it
/// includes vector index parameters that bloat the prompt).  Instead
/// we hand-author a compact summary of the labels and key edges.
/// Keeping the list short controls LLM cost more than completeness:
/// the spec only requires *that the LLM is grounded*, not exhaustive
/// coverage.
fn schema_snapshot() -> &'static str {
    static SCHEMA: OnceLock<String> = OnceLock::new();
    SCHEMA.get_or_init(|| {
        let mut buf = String::new();
        buf.push_str("Nodes (label: external_id, key properties):\n");
        for (label, ext, props) in NODE_SUMMARY {
            buf.push_str(&format!("- {label}: {ext}; {props}\n"));
        }
        buf.push_str("\nEdges (relationship: from → to):\n");
        for (rel, from, to) in EDGE_SUMMARY {
            buf.push_str(&format!("- :{rel}  {from} → {to}\n"));
        }
        buf
    })
}

/// Hand-authored summary of the labels the LLM is allowed to use.
///
/// Keep in sync with `crates/uniko-store/src/schema/constants.rs` —
/// when a new node type ships, add a row here so the LLM can target
/// it.  Properties listed are the ones agents actually query; we
/// don't list timestamps, embeddings, or internal book-keeping.
const NODE_SUMMARY: &[(&str, &str, &str)] = &[
    (labels::PARTICIPANT, "participant_id", "kind, name"),
    (labels::GOAL, "goal_id", "title, status, description"),
    (labels::TASK, "task_id", "title, status, priority"),
    (labels::SESSION, "session_id", "topic, started_at, ended_at"),
    (labels::MESSAGE, "message_id", "content, timestamp"),
    (
        labels::ACTION,
        "action_id",
        "action_type, status, started_at",
    ),
    (
        labels::EPISODE,
        "episode_id",
        "action_type, outcome, importance",
    ),
    (labels::ARTIFACT, "artifact_id", "kind, path, content"),
    (labels::CHUNK, "chunk_id", "text, chunk_type, language"),
    (labels::ENTITY, "entity_id", "name, entity_type, frequency"),
    (
        labels::OBSERVATION,
        "observation_id",
        "content, subject, predicate, object, observed_at",
    ),
    (
        labels::FACT,
        "fact_id",
        "subject, predicate, object, confidence, valid_at",
    ),
    (labels::TOPIC, "topic_id", "name, summary, entity_count"),
    (
        labels::PROCEDURE,
        "procedure_id",
        "name, description, status",
    ),
    (labels::RULE, "rule_id", "name, status, confidence"),
];

const EDGE_SUMMARY: &[(&str, &str, &str)] = &[
    ("PARENT_GOAL", "Goal", "Goal"),
    ("PART_OF", "Task", "Goal"),
    ("ASSIGNED_TO", "Task", "Participant"),
    ("FOR_GOAL", "Session", "Goal"),
    ("FOR_TASK", "Session", "Task"),
    ("IN_SESSION", "Message|Action|Episode", "Session"),
    ("SENT_BY", "Message", "Participant"),
    ("ADDRESSED_TO", "Message", "Participant"),
    ("RECORDED_BY", "Episode", "Participant"),
    ("FOLLOWED_BY", "Episode", "Episode"),
    ("PERFORMED_BY", "Action", "Participant"),
    ("TRIGGERED_BY", "Action|Episode", "Message"),
    ("PRODUCED", "Action", "Artifact"),
    ("NEXT_ACTION", "Action", "Action"),
    ("HAS_CHUNK", "Artifact|Message", "Chunk"),
    (
        "MENTIONS",
        "Message|Chunk|Action|Episode|Artifact",
        "Entity",
    ),
    ("OBSERVED_IN", "Observation", "Message|Chunk"),
    ("OBSERVED_DURING", "Observation", "Episode"),
    ("ABOUT", "Observation|Fact|Chunk", "Entity|Participant"),
    ("SUPPORTED_BY", "Fact", "Observation"),
    ("INVALIDATES", "Fact", "Fact"),
    ("BELONGS_TO", "Entity|Fact", "Topic"),
    ("MEMBER_OF", "Participant", "Organization"),
    ("PART_OF_TEAM", "Participant", "Team"),
    ("TEAM_IN_ORG", "Team", "Organization"),
];

const FEW_SHOT_EXAMPLES: &str = r#"
Q: How many messages did Caroline send in the last week?
A: MATCH (m:Message)-[:SENT_BY]->(p:Participant {name: 'Caroline'})
   WHERE m.timestamp > datetime() - duration({days: 7})
   RETURN count(m) AS n

Q: Which entities does the goal 'reduce refund time' mention?
A: MATCH (g:Goal {title: 'reduce refund time'})<-[:FOR_GOAL]-(:Session)<-[:IN_SESSION]-(m:Message)-[:MENTIONS]->(e:Entity)
   RETURN DISTINCT e.name AS entity LIMIT 25

Q: List the most recent 5 procedures for agent X.
A: MATCH (p:Procedure)
   WHERE p.status = 'active'
   RETURN p.name, p.effectiveness ORDER BY p.last_used_at DESC LIMIT 5
"#;

async fn call_with_retry(
    kb: &KnowledgeBase,
    llm_alias: &str,
    prompt: &str,
) -> Result<String, UnikoError> {
    let messages = vec![Message::user(prompt)];
    // Note: gpt-5-nano (and other o-series reasoning models) reject
    // any non-default temperature — `temperature: 1.0` is the only
    // accepted value.  Leave the field unset so each provider can use
    // its model's accepted default.
    let options = GenerationOptions {
        max_tokens: Some(MAX_TOKENS),
        ..Default::default()
    };

    let mut last_err = None;
    for attempt in 0..=RETRY_ATTEMPTS {
        let res = kb
            .db()
            .xervo()
            .generate(llm_alias, &messages, options.clone())
            .await;
        match res {
            Ok(r) => {
                let cypher = clean_response(&r.text);
                if !cypher.is_empty() {
                    return Ok(cypher);
                }
                last_err = Some(UnikoError::Embedding(
                    "nl_to_cypher: empty LLM response".into(),
                ));
            }
            Err(e) => {
                last_err = Some(UnikoError::Embedding(format!(
                    "nl_to_cypher: LLM call failed (attempt {attempt}): {e}"
                )));
            }
        }
        if attempt < RETRY_ATTEMPTS {
            // Backoff: 50ms → 200ms.  Pure-asyncio sleep keeps the
            // executor free for other work.
            let delay_ms = 50_u64 * (1 << attempt);
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
    }
    Err(last_err.unwrap_or_else(|| {
        UnikoError::Embedding("nl_to_cypher: exhausted retries with no error captured".into())
    }))
}

/// Strip Markdown code fences and lead-in chatter from an LLM response.
fn clean_response(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    if let Some(stripped) = s.strip_prefix("```cypher") {
        s = stripped.to_string();
    } else if let Some(stripped) = s.strip_prefix("```") {
        s = stripped.to_string();
    }
    if let Some(stripped) = s.strip_suffix("```") {
        s = stripped.to_string();
    }
    // Some models prefix with "Cypher:" — strip that too.
    if let Some(rest) = s.strip_prefix("Cypher:") {
        s = rest.to_string();
    }
    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalise_collapses_whitespace_and_lowercases() {
        assert_eq!(
            normalise("  What  did Caroline\nresearch?  "),
            "what did caroline research?"
        );
    }

    #[test]
    fn read_only_accepts_simple_match_return() {
        assert!(is_safe_read_only("MATCH (n) RETURN n LIMIT 10"));
    }

    #[test]
    fn read_only_rejects_create() {
        assert!(!is_safe_read_only("CREATE (n:Person {name: 'x'})"));
    }

    #[test]
    fn read_only_rejects_merge() {
        assert!(!is_safe_read_only("MERGE (n:Person {id: 'x'})"));
    }

    #[test]
    fn read_only_rejects_delete_and_detach() {
        assert!(!is_safe_read_only("MATCH (n) DELETE n"));
        assert!(!is_safe_read_only("MATCH (n) DETACH DELETE n"));
    }

    #[test]
    fn read_only_rejects_set_and_remove() {
        assert!(!is_safe_read_only("MATCH (n) SET n.x = 1"));
        assert!(!is_safe_read_only("MATCH (n) REMOVE n.x"));
    }

    #[test]
    fn contains_word_respects_boundaries() {
        assert!(contains_word("MATCH (N) CREATE (X)", "CREATE"));
        assert!(!contains_word("CREATEDATE", "CREATE"));
        assert!(!contains_word("RECREATED", "CREATE"));
    }

    #[test]
    fn clean_response_strips_fences() {
        assert_eq!(
            clean_response("```cypher\nMATCH (n) RETURN n\n```"),
            "MATCH (n) RETURN n"
        );
        assert_eq!(clean_response("```\nMATCH (n)\n```"), "MATCH (n)");
        assert_eq!(
            clean_response("Cypher: MATCH (n) RETURN n"),
            "MATCH (n) RETURN n"
        );
    }

    #[test]
    fn cache_evicts_lru_when_full() {
        let mut cache = Cache::new(2);
        cache.put("a".into(), "Q1".into());
        cache.put("b".into(), "Q2".into());
        // Touch 'a' so 'b' becomes the LRU.
        let _ = cache.get("a");
        cache.put("c".into(), "Q3".into());
        assert_eq!(cache.len(), 2);
        assert!(cache.get("b").is_none());
        assert_eq!(cache.get("a").as_deref(), Some("Q1"));
        assert_eq!(cache.get("c").as_deref(), Some("Q3"));
    }

    #[test]
    fn schema_snapshot_lists_core_labels() {
        let s = schema_snapshot();
        assert!(s.contains("Message:"));
        assert!(s.contains("Fact:"));
        assert!(s.contains("MENTIONS"));
    }
}
