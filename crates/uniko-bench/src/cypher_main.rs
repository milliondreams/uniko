//! `uniko-cypher` — interactive / one-shot Cypher console for uniko KBs.
//!
//! Opens an existing KnowledgeBase and runs Cypher queries against it
//! without recompiling. Works in three modes:
//!
//! ```bash
//! # 1. One-shot via --query
//! uniko-cypher --kb data/kb_llm/conv-26 --query "MATCH (m:Message) RETURN count(m) AS n"
//!
//! # 2. One-shot from stdin (UNIX-pipeline friendly)
//! echo "MATCH (m:Message) RETURN count(m) AS n" | uniko-cypher --kb data/kb_llm/conv-26
//!
//! # 3. Interactive REPL (open KB once, run many queries)
//! uniko-cypher --kb data/kb_llm/conv-26 --interactive
//! ```
//!
//! Output formats: `--format pretty` (default, multi-line table-ish),
//! `--format json` (one line per row, machine-readable), `--format csv`
//! (header + rows). All runs share the same KB-open path as the bench
//! (`KnowledgeBase::open_with_xervo`) so embedder/NLP catalog and
//! vector-index dimensions are consistent with how the data was
//! ingested.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use uni_db::ModelAliasSpec;
use uniko_store::KnowledgeBase;
use uniko_store::config::UnikoConfig;

/// Run Cypher queries against a uniko KnowledgeBase.
#[derive(Parser)]
#[command(name = "uniko-cypher", about = "Quick Cypher console for uniko KBs")]
struct Cli {
    /// Path to the KB directory (e.g. `data/kb_llm/conv-26`).
    #[arg(long, short = 'k')]
    kb: PathBuf,

    /// Cypher query to run. If omitted, reads from stdin.
    #[arg(long, short = 'q')]
    query: Option<String>,

    /// Read query from a file (alternative to --query / stdin).
    #[arg(long, short = 'f')]
    file: Option<PathBuf>,

    /// Drop into a REPL after opening the KB. End each query with `;`
    /// or a blank line. `:quit` / `:q` to exit.
    #[arg(long, short = 'i')]
    interactive: bool,

    /// Output format: pretty (default), json, csv.
    #[arg(long, default_value = "pretty")]
    format: String,

    /// Path to xervo model catalog JSON file (matches the bench's flag).
    #[arg(long)]
    catalog: Option<PathBuf>,

    /// Path to schema JSON file (matches the bench's flag).
    #[arg(long)]
    schema: Option<PathBuf>,

    /// Embedding preset for opening the KB. MUST match what was used
    /// at ingest time, otherwise the schema's vector-index dimensions
    /// won't align with the on-disk vectors.
    /// One of: `nomic` | `minilm` | `bge-small` | `bge-large`.
    #[arg(long, default_value = "nomic")]
    embedding: String,

    /// Suppress the timing footer.
    #[arg(long)]
    quiet: bool,

    /// Pre-warm xervo models on KB open. Off by default since the
    /// Cypher CLI is a data-inspection tool — skipping prefetch saves
    /// the multi-minute model load. Pass when you plan to run queries
    /// using `similar_to(...)` and want a hot embedder ready.
    #[arg(long)]
    prefetch: bool,

    /// Run the query under PROFILE — prints the logical plan + per-operator
    /// runtime stats instead of result rows. Useful for confirming index
    /// usage on hot reads (e.g. ext-id lookups).
    #[arg(long)]
    profile: bool,

    /// Treat the query as a write — open a Transaction, run the query
    /// inside it, and DROP the tx without committing. Combine with
    /// --profile to inspect SET/CREATE/DELETE plans without mutating
    /// the KB.
    #[arg(long)]
    write: bool,

    /// JSON object of parameters to bind, e.g.
    /// `'{"updates": [{"nid": 64, "freq": 5}], "now": "2026-01-01T00:00:00Z"}'`.
    /// Values are converted via `serde_json::Value → uni_common::Value`.
    #[arg(long)]
    params_json: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Default to error-only so the REPL prompt isn't drowned in
    // dataset-init warnings on KB open. Override via RUST_LOG.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "error".parse().unwrap()),
        )
        .init();

    let cli = Cli::parse();

    let mut config = UnikoConfig {
        catalog_path: cli.catalog.clone(),
        schema_path: cli.schema.clone(),
        ..Default::default()
    };
    config.embedding = match cli.embedding.as_str() {
        "nomic" => uniko_store::config::EmbeddingConfig::nomic_v15(),
        "minilm" => uniko_store::config::EmbeddingConfig::minilm_l6_v2(),
        "bge-small" => uniko_store::config::EmbeddingConfig::bge_small_en_v15(),
        "bge-large" => uniko_store::config::EmbeddingConfig::bge_large_en_v15(),
        other => anyhow::bail!(
            "unknown --embedding preset {other:?}; expected one of: nomic, minilm, bge-small, bge-large"
        ),
    };

    let kb = if cli.prefetch {
        KnowledgeBase::open_with_xervo(&cli.kb, config, Vec::<ModelAliasSpec>::new()).await
    } else {
        KnowledgeBase::open_with_xervo_no_prefetch(&cli.kb, config, Vec::<ModelAliasSpec>::new())
            .await
    }
    .with_context(|| format!("opening KB at {}", cli.kb.display()))?;

    if cli.interactive {
        run_repl(&kb, &cli.format, cli.quiet).await?;
    } else {
        // Source the query from --query, --file, or stdin.
        let query = match (&cli.query, &cli.file) {
            (Some(q), None) => q.clone(),
            (None, Some(p)) => {
                std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?
            }
            (Some(_), Some(_)) => anyhow::bail!("--query and --file are mutually exclusive"),
            (None, None) => read_stdin()?,
        };
        if query.trim().is_empty() {
            anyhow::bail!("no query provided (pass --query, --file, or pipe via stdin)");
        }
        let params = parse_params_json(cli.params_json.as_deref())?;
        if cli.profile {
            if cli.write {
                run_profile_write(&kb, &query, &params).await?;
            } else {
                run_profile(&kb, &query, &params).await?;
            }
        } else {
            run_one(&kb, &query, &cli.format, cli.quiet).await?;
        }
    }

    Ok(())
}

fn read_stdin() -> Result<String> {
    let stdin = std::io::stdin();
    let mut buf = String::new();
    for line in stdin.lock().lines() {
        let line = line.context("reading stdin")?;
        buf.push_str(&line);
        buf.push('\n');
    }
    Ok(buf)
}

async fn run_repl(kb: &KnowledgeBase, format: &str, quiet: bool) -> Result<()> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut buf = String::new();
    println!("uniko-cypher REPL — end queries with `;` or blank line; `:q` to exit");
    loop {
        let prompt = if buf.is_empty() { "> " } else { ".. " };
        write!(stdout, "{prompt}")?;
        stdout.flush()?;
        let mut line = String::new();
        let bytes = stdin.lock().read_line(&mut line)?;
        if bytes == 0 {
            // EOF
            break;
        }
        let trimmed = line.trim();
        if trimmed == ":q" || trimmed == ":quit" {
            break;
        }
        if trimmed.is_empty() {
            // Blank line — execute accumulated buffer if non-empty.
            if !buf.trim().is_empty() {
                let q = std::mem::take(&mut buf);
                if let Err(e) = run_one(kb, &q, format, quiet).await {
                    eprintln!("ERROR: {e}");
                }
            }
            continue;
        }
        buf.push_str(&line);
        if trimmed.ends_with(';') {
            // Execute and reset.
            let q = std::mem::take(&mut buf);
            // Strip trailing semicolon — uni-db parser doesn't accept it.
            let q = q.trim().trim_end_matches(';').to_string();
            if !q.is_empty()
                && let Err(e) = run_one(kb, &q, format, quiet).await
            {
                eprintln!("ERROR: {e}");
            }
        }
    }
    Ok(())
}

fn parse_params_json(s: Option<&str>) -> Result<Vec<(String, uni_db::Value)>> {
    let Some(s) = s else {
        return Ok(Vec::new());
    };
    let v: serde_json::Value = serde_json::from_str(s).context("parsing --params-json")?;
    let obj = v
        .as_object()
        .context("--params-json must be a JSON object at the top level")?;
    let mut out = Vec::with_capacity(obj.len());
    for (k, val) in obj {
        out.push((k.clone(), uni_db::Value::from(val.clone())));
    }
    Ok(out)
}

async fn run_profile_write(
    kb: &KnowledgeBase,
    query: &str,
    params: &[(String, uni_db::Value)],
) -> Result<()> {
    let session = kb.db().session();
    let tx = session
        .tx()
        .await
        .map_err(|e| anyhow::anyhow!("tx open: {e}"))?;
    let start = std::time::Instant::now();
    let mut builder = tx.execute_with(query);
    for (k, v) in params {
        builder = builder.param(k.as_str(), v.clone());
    }
    let (result, profile) = builder
        .profile()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let elapsed = start.elapsed();
    // Drop tx without commit — no mutation lands.
    drop(tx);

    println!("=== PLAN ===");
    println!("{:#?}", profile.explain);
    println!("\n=== RUNTIME (per operator) ===");
    for op in &profile.runtime_stats {
        println!(
            "  {:<30} rows={:<8} time_ms={:<10.3} mem={:<10} idx_hits={:?} idx_misses={:?}",
            op.operator,
            op.actual_rows,
            op.time_ms,
            op.memory_bytes,
            op.index_hits,
            op.index_misses,
        );
    }
    println!(
        "\n=== TOTALS === total_ms={} peak_mem={} write_result={:?} wall_elapsed_ms={:.2}",
        profile.total_time_ms,
        profile.peak_memory_bytes,
        result,
        elapsed.as_secs_f64() * 1000.0
    );
    Ok(())
}

async fn run_profile(
    kb: &KnowledgeBase,
    query: &str,
    params: &[(String, uni_db::Value)],
) -> Result<()> {
    let session = kb.db().session();
    let start = std::time::Instant::now();
    let mut builder = session.query_with(query);
    for (k, v) in params {
        builder = builder.param(k.as_str(), v.clone());
    }
    let (result, profile) = builder
        .profile()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let elapsed = start.elapsed();

    println!("=== PLAN ===");
    println!("{:#?}", profile.explain);
    println!("\n=== RUNTIME (per operator) ===");
    for op in &profile.runtime_stats {
        println!(
            "  {:<30} rows={:<8} time_ms={:<10.3} mem={:<10} idx_hits={:?} idx_misses={:?}",
            op.operator,
            op.actual_rows,
            op.time_ms,
            op.memory_bytes,
            op.index_hits,
            op.index_misses,
        );
    }
    println!(
        "\n=== TOTALS === total_ms={} peak_mem={} returned_rows={} wall_elapsed_ms={:.2}",
        profile.total_time_ms,
        profile.peak_memory_bytes,
        result.rows().len(),
        elapsed.as_secs_f64() * 1000.0
    );
    Ok(())
}

async fn run_one(kb: &KnowledgeBase, query: &str, format: &str, quiet: bool) -> Result<()> {
    let session = kb.db().session();
    let start = std::time::Instant::now();
    let result = session
        .query_with(query)
        .fetch_all()
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let elapsed = start.elapsed();

    let columns = result.columns().to_vec();
    let rows = result.rows();

    match format {
        "json" => print_json(&columns, rows)?,
        "csv" => print_csv(&columns, rows)?,
        _ => print_pretty(&columns, rows)?,
    }

    if !quiet {
        eprintln!(
            "({} row{} in {:.1}ms)",
            rows.len(),
            if rows.len() == 1 { "" } else { "s" },
            elapsed.as_secs_f64() * 1000.0
        );
    }
    Ok(())
}

fn print_pretty(columns: &[String], rows: &[uni_db::Row]) -> Result<()> {
    if columns.is_empty() {
        return Ok(());
    }
    // For each row print one line per column: `col=value`. Compact and
    // readable for short results; long results with many columns get
    // unwieldy but that's what --format json is for.
    for (i, row) in rows.iter().enumerate() {
        if rows.len() > 1 {
            println!("--- row {i} ---");
        }
        for (col_idx, col) in columns.iter().enumerate() {
            let v = row.values().get(col_idx);
            let s = v
                .map(|val| serde_json::to_string(val).unwrap_or_else(|_| format!("{val:?}")))
                .unwrap_or_else(|| "<missing>".to_string());
            // Truncate long values to avoid flooding the terminal.
            let display = if s.len() > 500 {
                format!("{}...({} chars)", &s[..500], s.len())
            } else {
                s
            };
            println!("  {col}: {display}");
        }
    }
    Ok(())
}

fn print_json(columns: &[String], rows: &[uni_db::Row]) -> Result<()> {
    for row in rows {
        let mut map = serde_json::Map::with_capacity(columns.len());
        for (i, col) in columns.iter().enumerate() {
            let v = row.values().get(i);
            let json_val = v
                .map(|val| serde_json::to_value(val).unwrap_or(serde_json::Value::Null))
                .unwrap_or(serde_json::Value::Null);
            map.insert(col.clone(), json_val);
        }
        println!("{}", serde_json::Value::Object(map));
    }
    Ok(())
}

fn print_csv(columns: &[String], rows: &[uni_db::Row]) -> Result<()> {
    // Header
    println!("{}", columns.join(","));
    for row in rows {
        let cells: Vec<String> = columns
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let v = row.values().get(i);
                let s = v
                    .map(|val| serde_json::to_string(val).unwrap_or_default())
                    .unwrap_or_default();
                // Naive CSV quoting: wrap in quotes if contains comma or quote.
                if s.contains(',') || s.contains('"') {
                    format!("\"{}\"", s.replace('"', "\"\""))
                } else {
                    s
                }
            })
            .collect();
        println!("{}", cells.join(","));
    }
    Ok(())
}
