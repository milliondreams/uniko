//! Integration tests for the agent-facing tools added for the MVP
//! gap-remediation increment 2: `create_goal`, `create_task`,
//! `assert_fact`, `invalidate_fact`, `add_observation`, episode
//! provenance edges, session auto-close, F59 summaries, and the `Agent`
//! facade.
//!
//! Embedding-dependent tools follow the `episode_e2e` convention: an
//! [`UnikoError::Embedding`] is treated as "structural path validated,
//! embedder unavailable in this env" and skips the test rather than
//! failing it.

use std::collections::HashMap;

use chrono::{Duration, Utc};
use uni_db::Value;

use uniko_memory::{
    AddObservationParams, Agent, AssertFactParams, CreateGoalParams, CreateTaskParams,
    InvalidateFactParams, RecordEpisodeParams, add_observation, assert_fact, create_goal,
    create_task, generate_session_summary, invalidate_fact, record_episode,
};
use uniko_store::config::UnikoConfig;
use uniko_store::schema::constants::labels;
use uniko_store::types::datetime_value;
use uniko_store::{KnowledgeBase, UnikoError};

async fn kb() -> KnowledgeBase {
    KnowledgeBase::in_memory(UnikoConfig::default())
        .await
        .expect("in-memory KB")
}

async fn seed_participant(kb: &KnowledgeBase, agent_id: &str) {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("kind".into(), Value::String("agent".into()));
    props.insert("name".into(), Value::String(format!("agent-{agent_id}")));
    kb.merge_node(labels::PARTICIPANT, "participant_id", agent_id, &props)
        .await
        .expect("participant created");
}

async fn seed_message(
    kb: &KnowledgeBase,
    message_id: &str,
    content: &str,
    ts: chrono::DateTime<Utc>,
) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("message_id".into(), Value::String(message_id.into()));
    props.insert("content".into(), Value::String(content.into()));
    props.insert("timestamp".into(), datetime_value(ts));
    kb.merge_node(labels::MESSAGE, "message_id", message_id, &props)
        .await
        .expect("message created")
}

async fn seed_session(
    kb: &KnowledgeBase,
    session_id: &str,
    started_at: chrono::DateTime<Utc>,
) -> i64 {
    let mut props: HashMap<String, Value> = HashMap::new();
    props.insert("session_id".into(), Value::String(session_id.into()));
    props.insert("started_at".into(), datetime_value(started_at));
    kb.merge_node(labels::SESSION, "session_id", session_id, &props)
        .await
        .expect("session created")
}

/// `true` when the result is an embedding failure (skip the test).
fn skip_embedding<T>(res: &Result<T, UnikoError>) -> bool {
    if let Err(UnikoError::Embedding(_)) = res {
        eprintln!("skipping: embedding unavailable in this env");
        true
    } else {
        false
    }
}

// ── create_goal ─────────────────────────────────────────────────────

#[tokio::test]
async fn create_goal_wires_owned_by() {
    let kb = kb().await;
    seed_participant(&kb, "agent-1").await;

    let res = create_goal(
        &kb,
        "agent-1",
        CreateGoalParams {
            goal_id: Some("goal-1".into()),
            title: "Ship the release".into(),
            description: Some("v6 launch".into()),
            ..Default::default()
        },
    )
    .await;
    if skip_embedding(&res) {
        return;
    }
    let goal = res.expect("create_goal");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (g:Goal)-[:OWNED_BY]->(p:Participant) WHERE id(g) = $gid \
             RETURN p.participant_id AS pid",
        )
        .param("gid", goal)
        .fetch_all()
        .await
        .expect("query");
    let pid: String = rows
        .rows()
        .first()
        .expect("OWNED_BY edge")
        .get("pid")
        .expect("pid");
    assert_eq!(pid, "agent-1");
}

// ── create_task ─────────────────────────────────────────────────────

#[tokio::test]
async fn create_task_wires_part_of_and_assigned_to() {
    let kb = kb().await;
    seed_participant(&kb, "agent-1").await;

    let goal_res = create_goal(
        &kb,
        "agent-1",
        CreateGoalParams {
            goal_id: Some("goal-2".into()),
            title: "Parent goal".into(),
            ..Default::default()
        },
    )
    .await;
    if skip_embedding(&goal_res) {
        return;
    }
    goal_res.expect("create_goal");

    let task_res = create_task(
        &kb,
        "agent-1",
        CreateTaskParams {
            task_id: Some("task-1".into()),
            title: "Do the thing".into(),
            goal_id: Some("goal-2".into()),
            ..Default::default()
        },
    )
    .await;
    if skip_embedding(&task_res) {
        return;
    }
    let task = task_res.expect("create_task");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (t:Task)-[:ASSIGNED_TO]->(p:Participant) WHERE id(t) = $tid \
             MATCH (t)-[:PART_OF]->(g:Goal) \
             RETURN p.participant_id AS pid, g.goal_id AS gid",
        )
        .param("tid", task)
        .fetch_all()
        .await
        .expect("query");
    let row = rows.rows().first().expect("task edges");
    assert_eq!(row.get::<String>("pid").expect("pid"), "agent-1");
    assert_eq!(row.get::<String>("gid").expect("gid"), "goal-2");
}

// ── assert_fact / invalidate_fact ───────────────────────────────────

#[tokio::test]
async fn assert_fact_creates_then_invalidate_closes_btic() {
    let kb = kb().await;

    let res = assert_fact(
        &kb,
        AssertFactParams {
            subject: "user".into(),
            predicate: "prefers".into(),
            object: Some("dark mode".into()),
            ..Default::default()
        },
    )
    .await;
    if skip_embedding(&res) {
        return;
    }
    let upsert = res.expect("assert_fact");
    assert!(upsert.was_created, "first assertion should create the Fact");

    // Read the fact's external id and confirm its BTIC is open.
    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (f:Fact) WHERE id(f) = $fid \
             RETURN f.fact_id AS fact_id, btic_is_unbounded(f.valid_at) AS open",
        )
        .param("fid", upsert.node_id)
        .fetch_all()
        .await
        .expect("query");
    let row = rows.rows().first().expect("fact row");
    let fact_id: String = row.get("fact_id").expect("fact_id");
    assert!(row.get::<bool>("open").expect("open"), "BTIC starts open");

    invalidate_fact(
        &kb,
        InvalidateFactParams {
            fact_id: fact_id.clone(),
            reason: Some("user changed preference".into()),
            ..Default::default()
        },
    )
    .await
    .expect("invalidate_fact");

    let rows2 = session
        .query_with("MATCH (f:Fact {fact_id: $fid}) RETURN btic_is_unbounded(f.valid_at) AS open")
        .param("fid", fact_id)
        .fetch_all()
        .await
        .expect("query");
    let still_open: bool = rows2
        .rows()
        .first()
        .expect("fact row")
        .get("open")
        .expect("open");
    assert!(!still_open, "BTIC must be closed after invalidate_fact");
}

// ── add_observation ─────────────────────────────────────────────────

#[tokio::test]
async fn add_observation_anchors_to_message() {
    let kb = kb().await;
    seed_participant(&kb, "agent-1").await;
    seed_message(&kb, "msg-1", "Caroline plays clarinet", Utc::now()).await;

    let res = add_observation(
        &kb,
        "agent-1",
        AddObservationParams {
            message_id: "msg-1".into(),
            content: "Caroline plays clarinet".into(),
            subject: "Caroline".into(),
            predicate: Some("plays".into()),
            object: Some("clarinet".into()),
            ..Default::default()
        },
    )
    .await;
    if skip_embedding(&res) {
        return;
    }
    let obs = res.expect("add_observation");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (o:Observation)-[:OBSERVED_IN]->(m:Message) WHERE id(o) = $oid \
             RETURN m.message_id AS mid",
        )
        .param("oid", obs)
        .fetch_all()
        .await
        .expect("query");
    let mid: String = rows
        .rows()
        .first()
        .expect("OBSERVED_IN edge")
        .get("mid")
        .expect("mid");
    assert_eq!(mid, "msg-1");
}

// ── record_episode provenance (TRIGGERED_BY / INVOLVES) ──────────────

#[tokio::test]
async fn record_episode_wires_triggered_by_and_involves() {
    let kb = kb().await;
    seed_participant(&kb, "agent-1").await;
    seed_message(&kb, "msg-2", "go fetch the docs", Utc::now()).await;

    // Seed an Action to involve.
    let mut action_props: HashMap<String, Value> = HashMap::new();
    action_props.insert("action_id".into(), Value::String("act-1".into()));
    action_props.insert("action_type".into(), Value::String("retrieve".into()));
    kb.merge_node(labels::ACTION, "action_id", "act-1", &action_props)
        .await
        .expect("action created");

    let res = record_episode(
        &kb,
        "agent-1",
        RecordEpisodeParams {
            action_type: "retrieve".into(),
            outcome: Some("success".into()),
            triggered_by_message_id: Some("msg-2".into()),
            involved_action_ids: vec!["act-1".into()],
            ..Default::default()
        },
    )
    .await;
    if skip_embedding(&res) {
        return;
    }
    let episode = res.expect("record_episode");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (e:Episode)-[:TRIGGERED_BY]->(m:Message) WHERE id(e) = $eid \
             MATCH (e)-[:INVOLVES]->(a:Action) \
             RETURN m.message_id AS mid, a.action_id AS aid",
        )
        .param("eid", episode)
        .fetch_all()
        .await
        .expect("query");
    let row = rows.rows().first().expect("episode provenance edges");
    assert_eq!(row.get::<String>("mid").expect("mid"), "msg-2");
    assert_eq!(row.get::<String>("aid").expect("aid"), "act-1");
}

// ── session auto-close (F14) ────────────────────────────────────────

#[tokio::test]
async fn close_inactive_sessions_stamps_ended_at() {
    let kb = kb().await;
    let now = Utc::now();
    let two_hours_ago = now - Duration::hours(2);

    let sess_nid = seed_session(&kb, "sess-1", two_hours_ago).await;
    let msg_nid = seed_message(&kb, "msg-old", "stale message", two_hours_ago).await;
    kb.create_edge("IN_SESSION", msg_nid, sess_nid, &HashMap::new())
        .await
        .expect("IN_SESSION edge");

    // 1h inactivity window: last activity (2h ago) is well past the cutoff.
    let closed = kb
        .close_inactive_sessions(3_600, now)
        .await
        .expect("close_inactive_sessions");
    assert_eq!(closed, vec!["sess-1".to_string()]);

    let session = kb.db().session();
    let rows = session
        .query_with("MATCH (s:Session) WHERE id(s) = $sid RETURN s.ended_at IS NOT NULL AS closed")
        .param("sid", sess_nid)
        .fetch_all()
        .await
        .expect("query");
    let is_closed: bool = rows
        .rows()
        .first()
        .expect("session")
        .get("closed")
        .expect("closed");
    assert!(is_closed, "ended_at must be set after auto-close");
}

#[tokio::test]
async fn close_inactive_sessions_skips_recent_sessions() {
    let kb = kb().await;
    let now = Utc::now();

    let sess_nid = seed_session(&kb, "sess-fresh", now).await;
    let msg_nid = seed_message(&kb, "msg-fresh", "just now", now).await;
    kb.create_edge("IN_SESSION", msg_nid, sess_nid, &HashMap::new())
        .await
        .expect("IN_SESSION edge");

    let closed = kb
        .close_inactive_sessions(3_600, now)
        .await
        .expect("close_inactive_sessions");
    assert!(closed.is_empty(), "an active session must not be closed");
}

// ── F59 session summary ─────────────────────────────────────────────

#[tokio::test]
async fn generate_session_summary_creates_node_and_edge() {
    let kb = kb().await;
    let now = Utc::now();
    let sess_nid = seed_session(&kb, "sess-2", now).await;
    let msg_nid = seed_message(&kb, "msg-3", "User likes pizza and jazz.", now).await;
    kb.create_edge("IN_SESSION", msg_nid, sess_nid, &HashMap::new())
        .await
        .expect("IN_SESSION edge");

    let res = generate_session_summary(&kb, "sess-2", now, None).await;
    if skip_embedding(&res) {
        return;
    }
    let summary = res
        .expect("generate_session_summary")
        .expect("summary produced");

    let session = kb.db().session();
    let rows = session
        .query_with(
            "MATCH (sum:Summary)-[:SUMMARIZES]->(s:Session) WHERE id(sum) = $sid \
             RETURN s.session_id AS sess, sum.level AS level, sum.text AS text",
        )
        .param("sid", summary)
        .fetch_all()
        .await
        .expect("query");
    let row = rows.rows().first().expect("SUMMARIZES edge");
    assert_eq!(row.get::<String>("sess").expect("sess"), "sess-2");
    assert_eq!(row.get::<String>("level").expect("level"), "session");
    assert!(
        row.get::<String>("text").expect("text").contains("pizza"),
        "extractive summary should carry the transcript content"
    );
}

// ── Agent facade ────────────────────────────────────────────────────

#[tokio::test]
async fn agent_facade_delegates_create_goal() {
    let kb = kb().await;
    seed_participant(&kb, "agent-x").await;

    let agent = Agent::new(kb.clone(), "agent-x");
    assert_eq!(agent.agent_id(), "agent-x");

    let res = agent
        .create_goal(CreateGoalParams {
            goal_id: Some("goal-facade".into()),
            title: "Facade goal".into(),
            ..Default::default()
        })
        .await;
    if skip_embedding(&res) {
        return;
    }
    let goal = res.expect("agent.create_goal");

    let session = kb.db().session();
    let rows = session
        .query_with("MATCH (g:Goal) WHERE id(g) = $gid RETURN g.goal_id AS gid")
        .param("gid", goal)
        .fetch_all()
        .await
        .expect("query");
    assert_eq!(
        rows.rows()
            .first()
            .expect("goal")
            .get::<String>("gid")
            .expect("gid"),
        "goal-facade"
    );
}
