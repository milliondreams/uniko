//! Session maintenance: inactivity auto-close (F14).
//!
//! Backs the consolidation worker's session-maintenance sweep.  A
//! Session has no explicit "ended" signal from the agent, so one is
//! inferred: when its most recent message is older than the configured
//! inactivity window, the Session is stamped with `ended_at` (the time
//! of that last message) and becomes eligible for F59 summarisation.

use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::error::Result;
use crate::storage::KnowledgeBase;
use crate::types::{NodeId, datetime_from_value, datetime_value};

impl KnowledgeBase {
    /// Close every open Session whose latest message predates the
    /// inactivity window, returning the closed `session_id`s.
    ///
    /// A Session is "open" when `ended_at IS NULL`.  Its `ended_at` is
    /// set to the timestamp of its most recent message (when decodable;
    /// otherwise `now`), so the close reflects when activity actually
    /// stopped rather than when the sweep ran.  A zero
    /// `inactivity_secs` disables the sweep and returns an empty list.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::error::UnikoError::Storage)
    /// when the query or an update fails.
    pub async fn close_inactive_sessions(
        &self,
        inactivity_secs: u64,
        now: DateTime<Utc>,
    ) -> Result<Vec<String>> {
        if inactivity_secs == 0 {
            return Ok(Vec::new());
        }
        let cutoff = now - chrono::Duration::seconds(inactivity_secs as i64);

        let session = self.db.session();
        // Project one row per (open session, message) and aggregate the
        // latest timestamp per session in Rust. We intentionally avoid a
        // Cypher `max(m.timestamp)` aggregate here: uni-db serialises an
        // aggregated DateTime as `LargeBinary`, which then fails the
        // typed comparison/readback. A bare `m.timestamp` projection
        // round-trips as a proper `Timestamp`.
        let cypher = "MATCH (m:Message)-[:IN_SESSION]->(s:Session) \
                      WHERE s.ended_at IS NULL \
                      RETURN id(s) AS nid, s.session_id AS sid, m.timestamp AS ts";
        let result = session.query_with(cypher).fetch_all().await?;

        // Latest message timestamp per session node.
        let mut latest: HashMap<i64, (String, DateTime<Utc>)> = HashMap::new();
        for row in result.rows() {
            let Ok(nid) = row.get::<i64>("nid") else {
                continue;
            };
            let Ok(sid) = row.get::<String>("sid") else {
                continue;
            };
            let Some(ts) = row
                .value("ts")
                .and_then(|v| datetime_from_value(v, "Message.timestamp").ok())
            else {
                continue;
            };
            latest
                .entry(nid)
                .and_modify(|cur| {
                    if ts > cur.1 {
                        cur.1 = ts;
                    }
                })
                .or_insert((sid, ts));
        }

        let mut closed = Vec::new();
        for (nid, (sid, last)) in latest {
            if last > cutoff {
                continue;
            }
            // ended_at reflects when activity actually stopped.
            let mut props: HashMap<String, crate::Value> = HashMap::new();
            props.insert("ended_at".into(), datetime_value(last));
            self.update_node(nid as NodeId, &props).await?;
            closed.push(sid);
        }
        Ok(closed)
    }
}
