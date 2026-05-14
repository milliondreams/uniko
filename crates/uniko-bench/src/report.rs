//! Benchmark result aggregation and reporting.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::Serialize;

use crate::data::QuestionCategory;
use crate::query::QueryResult;

/// Per-category evaluation results.
#[derive(Debug, Serialize)]
pub struct CategoryReport {
    pub name: String,
    pub count: usize,
    pub evidence_hit_rate: f64,
    pub avg_f1: f64,
    pub avg_judge: Option<f64>,
    pub avg_recall_latency_ms: f64,
    pub avg_generation_latency_ms: f64,
}

/// Per-conversation rollup — same shape as the overall report but
/// scoped to a single LoCoMo `sample_id`.  Lets downstream analysis
/// pinpoint which conversations are pulling the headline down (or up).
#[derive(Debug, Serialize)]
pub struct ConversationReport {
    pub sample_id: String,
    pub count: usize,
    pub evidence_hit_rate: f64,
    pub avg_f1: f64,
    pub avg_judge: Option<f64>,
    pub avg_recall_latency_ms: f64,
    pub avg_generation_latency_ms: f64,
    /// Per-category split inside this conversation.
    pub categories: Vec<CategoryReport>,
}

/// Full benchmark report.
#[derive(Debug, Serialize)]
pub struct BenchmarkReport {
    pub overall_evidence_hit_rate: f64,
    pub overall_f1: f64,
    pub overall_judge: Option<f64>,
    pub total_questions: usize,
    pub total_conversations: usize,
    pub categories: Vec<CategoryReport>,
    pub avg_recall_latency_ms: f64,
    pub avg_generation_latency_ms: f64,
    /// Per-conversation breakdown.  Empty for runs that processed a
    /// single conversation (rolled up into the overall figures).
    pub conversations: Vec<ConversationReport>,
}

/// Per-question detail for JSON output.
#[derive(Debug, Serialize)]
pub struct QuestionDetail {
    pub sample_id: String,
    pub question_index: usize,
    pub question: String,
    pub gold_answer: String,
    pub predicted_answer: String,
    pub category: String,
    pub evidence_found: usize,
    pub evidence_total: usize,
    pub f1: f64,
    pub judge: Option<f64>,
    pub recall_latency_ms: u64,
    pub generation_latency_ms: u64,
    /// Recall cascade indicators carried from `ContextBundle`.
    pub phase1_only: bool,
    pub coverage: f64,
    pub total_tokens: usize,
    /// Full bundle of retrieved items (node_type / score / content).
    /// Always populated; useful for offline failure-mode analysis.
    pub recall_bundle: Vec<crate::query::RecalledItem>,
}

/// Aggregator state collected over a sub-slice of results (used for
/// both the overall rollup and each per-conversation rollup).
struct Accum {
    f1_sum: f64,
    judge_scores: Vec<f64>,
    recall_ms_sum: u64,
    gen_ms_sum: u64,
    evidence_found: usize,
    evidence_total: usize,
    count: usize,
    by_category: HashMap<QuestionCategory, Accum>,
}

impl Accum {
    fn new() -> Self {
        Self {
            f1_sum: 0.0,
            judge_scores: Vec::new(),
            recall_ms_sum: 0,
            gen_ms_sum: 0,
            evidence_found: 0,
            evidence_total: 0,
            count: 0,
            by_category: HashMap::new(),
        }
    }

    fn observe(&mut self, qr: &QueryResult, f1: f64, judge: Option<f64>) {
        self.f1_sum += f1;
        if let Some(j) = judge {
            self.judge_scores.push(j);
        }
        self.recall_ms_sum += qr.recall_latency_ms;
        self.gen_ms_sum += qr.generation_latency_ms;
        self.evidence_found += qr.evidence_found;
        self.evidence_total += qr.evidence_total;
        self.count += 1;
        let cat = self
            .by_category
            .entry(qr.category)
            .or_insert_with(Accum::new);
        cat.f1_sum += f1;
        if let Some(j) = judge {
            cat.judge_scores.push(j);
        }
        cat.recall_ms_sum += qr.recall_latency_ms;
        cat.gen_ms_sum += qr.generation_latency_ms;
        cat.evidence_found += qr.evidence_found;
        cat.evidence_total += qr.evidence_total;
        cat.count += 1;
    }

    fn evidence_hit_rate(&self) -> f64 {
        if self.evidence_total > 0 {
            self.evidence_found as f64 / self.evidence_total as f64
        } else {
            0.0
        }
    }

    fn avg_f1(&self) -> f64 {
        if self.count > 0 {
            self.f1_sum / self.count as f64
        } else {
            0.0
        }
    }

    fn avg_judge(&self) -> Option<f64> {
        if self.judge_scores.is_empty() {
            None
        } else {
            Some(self.judge_scores.iter().sum::<f64>() / self.judge_scores.len() as f64)
        }
    }

    fn avg_recall_ms(&self) -> f64 {
        if self.count > 0 {
            self.recall_ms_sum as f64 / self.count as f64
        } else {
            0.0
        }
    }

    fn avg_gen_ms(&self) -> f64 {
        if self.count > 0 {
            self.gen_ms_sum as f64 / self.count as f64
        } else {
            0.0
        }
    }

    fn categories(&self) -> Vec<CategoryReport> {
        [
            QuestionCategory::SingleHop,
            QuestionCategory::MultiHop,
            QuestionCategory::Temporal,
            QuestionCategory::OpenDomain,
            QuestionCategory::Adversarial,
        ]
        .iter()
        .filter_map(|cat| {
            let acc = self.by_category.get(cat)?;
            Some(CategoryReport {
                name: cat.name().to_string(),
                count: acc.count,
                evidence_hit_rate: acc.evidence_hit_rate(),
                avg_f1: acc.avg_f1(),
                avg_judge: acc.avg_judge(),
                avg_recall_latency_ms: acc.avg_recall_ms(),
                avg_generation_latency_ms: acc.avg_gen_ms(),
            })
        })
        .collect()
    }
}

/// Aggregate query results into a benchmark report.
pub fn aggregate(
    results: &[(QueryResult, f64, Option<f64>)],
    num_conversations: usize,
) -> BenchmarkReport {
    let mut overall = Accum::new();
    let mut by_sample: BTreeMap<String, Accum> = BTreeMap::new();

    for (qr, f1, judge) in results {
        overall.observe(qr, *f1, *judge);
        by_sample
            .entry(qr.sample_id.clone())
            .or_insert_with(Accum::new)
            .observe(qr, *f1, *judge);
    }

    // Per-conversation rollup, sorted by sample_id for stable output.
    let conversations: Vec<ConversationReport> = by_sample
        .into_iter()
        .map(|(sample_id, acc)| ConversationReport {
            sample_id,
            count: acc.count,
            evidence_hit_rate: acc.evidence_hit_rate(),
            avg_f1: acc.avg_f1(),
            avg_judge: acc.avg_judge(),
            avg_recall_latency_ms: acc.avg_recall_ms(),
            avg_generation_latency_ms: acc.avg_gen_ms(),
            categories: acc.categories(),
        })
        .collect();

    BenchmarkReport {
        overall_evidence_hit_rate: overall.evidence_hit_rate(),
        overall_f1: overall.avg_f1(),
        overall_judge: overall.avg_judge(),
        total_questions: overall.count,
        total_conversations: num_conversations,
        categories: overall.categories(),
        avg_recall_latency_ms: overall.avg_recall_ms(),
        avg_generation_latency_ms: overall.avg_gen_ms(),
        conversations,
    }
}

/// Print the report as a formatted table to stdout.
pub fn print_report(report: &BenchmarkReport) {
    print_report_with_name(report, "LoCoMo");
}

/// Print the report with a custom benchmark name.
pub fn print_report_with_name(report: &BenchmarkReport, name: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  {name} Benchmark Results");
    println!(
        "  {} conversations, {} questions",
        report.total_conversations, report.total_questions
    );
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  {:<14} {:>6} {:>10} {:>8} {:>8} {:>10}",
        "Category", "Count", "Evidence%", "F1", "Judge", "Recall ms"
    );
    println!("  ─────────────────────────────────────────────────────────────");

    for cat in &report.categories {
        let judge_str = cat
            .avg_judge
            .map(|j| format!("{j:.3}"))
            .unwrap_or_else(|| "N/A".to_string());
        println!(
            "  {:<14} {:>6} {:>9.1}% {:>8.3} {:>8} {:>10.0}",
            cat.name,
            cat.count,
            cat.evidence_hit_rate * 100.0,
            cat.avg_f1,
            judge_str,
            cat.avg_recall_latency_ms,
        );
    }

    println!("  ─────────────────────────────────────────────────────────────");
    let judge_str = report
        .overall_judge
        .map(|j| format!("{j:.3}"))
        .unwrap_or_else(|| "N/A".to_string());
    println!(
        "  {:<14} {:>6} {:>9.1}% {:>8.3} {:>8} {:>10.0}",
        "Overall",
        report.total_questions,
        report.overall_evidence_hit_rate * 100.0,
        report.overall_f1,
        judge_str,
        report.avg_recall_latency_ms,
    );
    println!("═══════════════════════════════════════════════════════════════");

    // Per-conversation breakdown when more than one sample is present.
    if report.conversations.len() > 1 {
        println!();
        println!("  Per-conversation breakdown:");
        println!(
            "  {:<14} {:>6} {:>10} {:>8} {:>8} {:>10}",
            "sample_id", "Count", "Evidence%", "F1", "Judge", "Recall ms",
        );
        println!("  ─────────────────────────────────────────────────────────────");
        for c in &report.conversations {
            let judge_str = c
                .avg_judge
                .map(|j| format!("{j:.3}"))
                .unwrap_or_else(|| "N/A".to_string());
            println!(
                "  {:<14} {:>6} {:>9.1}% {:>8.3} {:>8} {:>10.0}",
                c.sample_id,
                c.count,
                c.evidence_hit_rate * 100.0,
                c.avg_f1,
                judge_str,
                c.avg_recall_latency_ms,
            );
        }
        println!("  ─────────────────────────────────────────────────────────────");
    }
    println!();
}

/// Write per-question details to a JSON file.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_json(
    results: &[(QueryResult, f64, Option<f64>)],
    report: &BenchmarkReport,
    path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let details: Vec<QuestionDetail> = results
        .iter()
        .map(|(qr, f1, judge)| QuestionDetail {
            sample_id: qr.sample_id.clone(),
            question_index: qr.question_index,
            question: qr.question.clone(),
            gold_answer: qr.gold_answer.clone(),
            predicted_answer: qr.predicted_answer.clone(),
            category: qr.category.name().to_string(),
            evidence_found: qr.evidence_found,
            evidence_total: qr.evidence_total,
            f1: *f1,
            judge: *judge,
            recall_latency_ms: qr.recall_latency_ms,
            generation_latency_ms: qr.generation_latency_ms,
            phase1_only: qr.phase1_only,
            coverage: qr.coverage,
            total_tokens: qr.total_tokens,
            recall_bundle: qr.recall_bundle.clone(),
        })
        .collect();

    #[derive(Serialize)]
    struct FullOutput<'a> {
        summary: &'a BenchmarkReport,
        questions: Vec<QuestionDetail>,
    }

    let output = FullOutput {
        summary: report,
        questions: details,
    };

    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(path, json)?;
    Ok(())
}
