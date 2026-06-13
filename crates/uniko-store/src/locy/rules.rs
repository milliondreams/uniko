//! Locy rule creation, execution, and lifecycle management.

use std::collections::HashMap;

use uni_db::Value;

use crate::error::{Result, UnikoError};
use crate::locy::{Record, RuleInfo};
use crate::storage::KnowledgeBase;

impl KnowledgeBase {
    /// Register a Locy rule from source code.
    ///
    /// The `source` should be valid Locy syntax (e.g. a `CREATE RULE`
    /// statement or raw horn clauses).
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] if the source has syntax errors.
    pub async fn create_rule(&self, source: &str) -> Result<()> {
        self.db
            .rules()
            .register(source)
            .await
            .map_err(|e| UnikoError::Locy(e.to_string()))
    }

    /// Execute a Locy program and return the result rows.
    ///
    /// Parameters like `$agent_id` are injected via `params`.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on evaluation failure.
    pub async fn execute_rule(
        &self,
        program: &str,
        params: &HashMap<String, Value>,
    ) -> Result<Vec<Record>> {
        let session = self.db.session();
        let mut builder = session.locy_with(program);
        for (k, v) in params {
            builder = builder.param(k.as_str(), v.clone());
        }
        let result = builder
            .run()
            .await
            .map_err(|e| UnikoError::Locy(e.to_string()))?;

        // rows() returns Option<&Vec<Record>>; clone-collect or empty.
        Ok(result.rows().map(|rs| rs.to_vec()).unwrap_or_default())
    }

    /// List all registered Locy rules.
    pub fn list_rules(&self) -> Vec<RuleInfo> {
        self.db
            .rules()
            .list()
            .into_iter()
            .map(|name| RuleInfo { name })
            .collect()
    }

    /// Remove a Locy rule by name.
    ///
    /// Returns `true` if the rule was found and removed.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on failure.
    pub async fn delete_rule(&self, name: &str) -> Result<bool> {
        self.db
            .rules()
            .remove(name)
            .await
            .map_err(|e| UnikoError::Locy(e.to_string()))
    }

    /// Explain how a rule derives its results.
    ///
    /// Returns the raw `Debug` rendering of the runtime stats — this
    /// is a developer-facing diagnostic, not a user-stable string, so
    /// the shape may change when the stats struct evolves.
    ///
    /// # Errors
    ///
    /// Returns [`UnikoError::Locy`] on failure.
    pub async fn explain_rule(&self, program: &str) -> Result<String> {
        let session = self.db.session();
        let result = session
            .locy_with(program)
            .run()
            .await
            .map_err(|e| UnikoError::Locy(e.to_string()))?;

        Ok(format!("{:?}", result.stats()))
    }
}
