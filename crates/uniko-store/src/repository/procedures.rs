//! Procedure-lifecycle reads (P5): active-procedure candidates and a
//! counter snapshot. Backs `uniko_cortex::procedures`.

use crate::error::Result;
use crate::storage::KnowledgeBase;

/// A candidate Procedure for state matching: external id, name, the raw
/// `precondition_rule` (the caller evaluates it), and effectiveness.
#[derive(Debug, Clone)]
pub struct ProcedureCandidate {
    pub procedure_id: String,
    pub name: String,
    pub precondition_rule: String,
    pub effectiveness: f64,
}

/// A point-in-time snapshot of a Procedure's counters + status.
#[derive(Debug, Clone, Default)]
pub struct ProcedureSnapshot {
    pub use_count: i64,
    pub success_count: i64,
    pub failure_count: i64,
    pub status: String,
}

impl KnowledgeBase {
    /// Procedures in `status`, ordered by effectiveness desc.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn fetch_procedures_by_status(
        &self,
        status: &str,
    ) -> Result<Vec<ProcedureCandidate>> {
        let session = self.db.session();
        let cypher = "MATCH (p:Procedure) WHERE p.status = $st \
                      RETURN p.procedure_id AS pid, p.name AS name, \
                             coalesce(p.precondition_rule, '') AS pre, \
                             coalesce(p.effectiveness, 0.0) AS eff \
                      ORDER BY eff DESC";
        let result = session
            .query_with(cypher)
            .param("st", status)
            .fetch_all()
            .await?;
        let mut out = Vec::new();
        for row in result.rows() {
            let Ok(pid) = row.get::<String>("pid") else {
                continue;
            };
            out.push(ProcedureCandidate {
                procedure_id: pid,
                name: row.get("name").unwrap_or_default(),
                precondition_rule: row.get("pre").unwrap_or_default(),
                effectiveness: row.get("eff").unwrap_or(0.0),
            });
        }
        Ok(out)
    }

    /// Snapshot the counters + status of Procedure `procedure_id`, or
    /// `None` if absent.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Storage`](crate::UnikoError::Storage) on
    /// query failure.
    pub async fn read_procedure_snapshot(
        &self,
        procedure_id: &str,
    ) -> Result<Option<ProcedureSnapshot>> {
        let session = self.db.session();
        let cypher = "MATCH (p:Procedure) WHERE p.procedure_id = $pid \
                      RETURN coalesce(p.use_count, 0) AS uc, \
                             coalesce(p.success_count, 0) AS sc, \
                             coalesce(p.failure_count, 0) AS fc, \
                             coalesce(p.status, '') AS st";
        let result = session
            .query_with(cypher)
            .param("pid", procedure_id)
            .fetch_all()
            .await?;
        let Some(row) = result.rows().first() else {
            return Ok(None);
        };
        Ok(Some(ProcedureSnapshot {
            use_count: row.get("uc").unwrap_or(0),
            success_count: row.get("sc").unwrap_or(0),
            failure_count: row.get("fc").unwrap_or(0),
            status: row.get("st").unwrap_or_default(),
        }))
    }
}
