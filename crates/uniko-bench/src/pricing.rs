//! Per-model USD pricing for the bench's cost columns.
//!
//! Pricing is read from a hand-maintained CSV (`pricing.csv` at the
//! crate root by default).  The CSV format is intentionally small —
//! the workspace has no `csv` crate dep, so this module parses the
//! file by hand.  Columns:
//!
//! ```text
//! model_id,provider,input_per_million_usd,output_per_million_usd,embedding_per_million_usd,notes
//! ```
//!
//! Any cost column may be blank — a blank value means "unpriced" and
//! returns `None` from the corresponding `cost_*` accessor (no
//! fabricated zeros).  Unknown models trigger a one-shot warning so
//! the operator notices missing entries without flooding logs.

// Rust guideline compliant

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};

/// Per-million-token rates for one model.
#[derive(Debug, Clone, Default)]
pub struct Rates {
    /// USD per 1 000 000 input tokens, or `None` when unpriced.
    pub input_per_m: Option<f64>,
    /// USD per 1 000 000 output tokens, or `None` when unpriced.
    pub output_per_m: Option<f64>,
    /// USD per 1 000 000 embedding tokens, or `None` when unpriced.
    pub embedding_per_m: Option<f64>,
}

/// In-memory pricing table loaded from `pricing.csv`.
#[derive(Debug)]
pub struct Pricing {
    rates: HashMap<String, Rates>,
    warned: Mutex<HashSet<String>>,
}

impl Pricing {
    /// Build an empty pricing table — every cost lookup returns `None`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            rates: HashMap::new(),
            warned: Mutex::new(HashSet::new()),
        }
    }

    /// Load the pricing table from a CSV file.
    ///
    /// Lines starting with `#` are treated as comments.  The first
    /// non-comment line is consumed as the header.  Blank cost cells
    /// are interpreted as `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the header is
    /// missing the expected columns.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading pricing CSV at {}", path.display()))?;

        let mut lines = raw
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'));

        let header = lines
            .next()
            .with_context(|| format!("pricing CSV {} has no header row", path.display()))?;
        let cols: Vec<&str> = header.split(',').map(str::trim).collect();
        let idx = |name: &str| {
            cols.iter().position(|c| *c == name).with_context(|| {
                format!("pricing CSV {} missing required column {name:?}", path.display())
            })
        };
        let i_model = idx("model_id")?;
        let i_input = idx("input_per_million_usd")?;
        let i_output = idx("output_per_million_usd")?;
        let i_embed = idx("embedding_per_million_usd")?;

        let mut rates: HashMap<String, Rates> = HashMap::new();
        for (lineno, line) in lines.enumerate() {
            let row: Vec<&str> = line.split(',').map(str::trim).collect();
            let Some(model) = row.get(i_model) else { continue };
            if model.is_empty() {
                continue;
            }
            let parse = |raw: &str, kind: &str| -> Result<Option<f64>> {
                if raw.is_empty() {
                    return Ok(None);
                }
                let v: f64 = raw.parse().with_context(|| {
                    format!(
                        "pricing CSV {} line {}: invalid {kind} value {raw:?}",
                        path.display(),
                        lineno + 2 // +2: 1-based, plus the header line
                    )
                })?;
                Ok(Some(v))
            };
            let entry = Rates {
                input_per_m: parse(row.get(i_input).copied().unwrap_or(""), "input")?,
                output_per_m: parse(row.get(i_output).copied().unwrap_or(""), "output")?,
                embedding_per_m: parse(row.get(i_embed).copied().unwrap_or(""), "embedding")?,
            };
            rates.insert((*model).to_string(), entry);
        }

        tracing::info!(
            path = %path.display(),
            models = rates.len(),
            "pricing table loaded",
        );
        Ok(Self {
            rates,
            warned: Mutex::new(HashSet::new()),
        })
    }

    /// Cost in USD for `tokens` input tokens on `model`.
    ///
    /// Returns `None` when the model is unknown or when its input rate
    /// is blank in the CSV.  Logs a single warning per unknown model.
    #[must_use]
    pub fn cost_input(&self, model: &str, tokens: u64) -> Option<f64> {
        let rates = self.lookup(model)?;
        rates.input_per_m.map(|r| token_cost(r, tokens))
    }

    /// Cost in USD for `tokens` output tokens on `model`.
    #[must_use]
    pub fn cost_output(&self, model: &str, tokens: u64) -> Option<f64> {
        let rates = self.lookup(model)?;
        rates.output_per_m.map(|r| token_cost(r, tokens))
    }

    /// Cost in USD for `tokens` embedding tokens on `model`.
    #[must_use]
    pub fn cost_embedding(&self, model: &str, tokens: u64) -> Option<f64> {
        let rates = self.lookup(model)?;
        rates.embedding_per_m.map(|r| token_cost(r, tokens))
    }

    fn lookup(&self, model: &str) -> Option<&Rates> {
        if let Some(r) = self.rates.get(model) {
            return Some(r);
        }
        // Warn-once on the first miss; the table is small enough that a
        // missing model_id is almost always a config error worth flagging.
        if let Ok(mut warned) = self.warned.lock()
            && warned.insert(model.to_string())
        {
            tracing::warn!(
                model,
                "no pricing entry for model; cost columns will be null. add a row to pricing.csv to fix"
            );
        }
        None
    }
}

fn token_cost(rate_per_m: f64, tokens: u64) -> f64 {
    rate_per_m * (tokens as f64) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_csv(body: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(body.as_bytes()).expect("write");
        f
    }

    #[test]
    fn loads_priced_and_unpriced_rows() {
        let f = write_csv(
            "model_id,provider,input_per_million_usd,output_per_million_usd,embedding_per_million_usd,notes\n\
             gpt-4o-mini,openai,0.15,0.60,,\n\
             gpt-5,openai,,,,fill in\n",
        );
        let p = Pricing::load(f.path()).expect("load");
        let cost_in = p.cost_input("gpt-4o-mini", 1_000_000).expect("priced");
        let cost_out = p.cost_output("gpt-4o-mini", 500_000).expect("priced");
        assert!((cost_in - 0.15).abs() < 1e-9, "expected 0.15, got {cost_in}");
        assert!((cost_out - 0.30).abs() < 1e-9, "expected 0.30, got {cost_out}");
        assert!(p.cost_input("gpt-5", 100).is_none(), "blank rate -> None");
    }

    #[test]
    fn unknown_model_returns_none() {
        let f = write_csv("model_id,provider,input_per_million_usd,output_per_million_usd,embedding_per_million_usd,notes\n");
        let p = Pricing::load(f.path()).expect("load");
        assert!(p.cost_input("nonexistent", 1000).is_none());
    }

    #[test]
    fn empty_table_is_safe() {
        let p = Pricing::empty();
        assert!(p.cost_input("anything", 1000).is_none());
        assert!(p.cost_output("anything", 1000).is_none());
        assert!(p.cost_embedding("anything", 1000).is_none());
    }
}
