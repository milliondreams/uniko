//! Benchmark result aggregation and reporting.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde::Serialize;

use crate::data::QuestionCategory;
use crate::pricing::Pricing;
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
    /// Mean answer-LLM input tokens per question with provider usage.
    pub avg_answer_input_tokens: f64,
    /// Mean answer-LLM output tokens per question with provider usage.
    pub avg_answer_output_tokens: f64,
    /// Mean USD cost per question for the answer + judge LLM calls.
    pub avg_query_cost_usd: f64,
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
    pub avg_answer_input_tokens: f64,
    pub avg_answer_output_tokens: f64,
    pub avg_query_cost_usd: f64,
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
    /// Total USD cost of answer-LLM calls across every question.
    ///
    /// This is the **usage cost** — what a production customer of
    /// uniko would pay.  Headline cost numbers should derive from
    /// this field, not from `total_query_cost_usd`.
    pub total_answer_cost_usd: f64,
    /// Total USD cost of judge-LLM calls across every question.
    ///
    /// This is **evaluator overhead** — bench-harness cost only.
    /// Customers never run the judge.  Reported separately so it
    /// doesn't inflate the headline usage cost.
    pub total_judge_cost_usd: f64,
    /// **Deprecated.** Sum of answer + judge cost.  Conflates usage
    /// cost (answer) with evaluator overhead (judge); use
    /// `total_answer_cost_usd` for production cost reporting.
    /// Retained so older JSON consumers don't break.
    pub total_query_cost_usd: f64,
    /// Mean per-question usage cost (answer LLM only).
    pub cost_per_question_usd: f64,
    /// Mean answer-LLM input tokens per question.
    pub avg_answer_input_tokens: f64,
    /// Mean answer-LLM output tokens per question.
    pub avg_answer_output_tokens: f64,
    /// Mean judge-LLM input tokens per question that ran a judge.
    pub avg_judge_input_tokens: f64,
    /// Mean judge-LLM output tokens per question that ran a judge.
    pub avg_judge_output_tokens: f64,
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
    pub answer_input_tokens: Option<u64>,
    pub answer_output_tokens: Option<u64>,
    pub answer_model: String,
    pub judge_latency_ms: Option<u64>,
    pub judge_input_tokens: Option<u64>,
    pub judge_output_tokens: Option<u64>,
    pub judge_model: Option<String>,
    pub answer_cost_usd: f64,
    pub judge_cost_usd: f64,
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
    answer_in_sum: u64,
    answer_in_count: usize,
    answer_out_sum: u64,
    answer_out_count: usize,
    judge_in_sum: u64,
    judge_in_count: usize,
    judge_out_sum: u64,
    judge_out_count: usize,
    answer_cost_sum: f64,
    judge_cost_sum: f64,
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
            answer_in_sum: 0,
            answer_in_count: 0,
            answer_out_sum: 0,
            answer_out_count: 0,
            judge_in_sum: 0,
            judge_in_count: 0,
            judge_out_sum: 0,
            judge_out_count: 0,
            answer_cost_sum: 0.0,
            judge_cost_sum: 0.0,
            by_category: HashMap::new(),
        }
    }

    fn observe(&mut self, qr: &QueryResult, f1: f64, judge: Option<f64>, pricing: &Pricing) {
        self.observe_inner(qr, f1, judge, pricing);
        let cat = self
            .by_category
            .entry(qr.category)
            .or_insert_with(Accum::new);
        cat.observe_inner(qr, f1, judge, pricing);
    }

    fn observe_inner(&mut self, qr: &QueryResult, f1: f64, judge: Option<f64>, pricing: &Pricing) {
        self.f1_sum += f1;
        if let Some(j) = judge {
            self.judge_scores.push(j);
        }
        self.recall_ms_sum += qr.recall_latency_ms;
        self.gen_ms_sum += qr.generation_latency_ms;
        self.evidence_found += qr.evidence_found;
        self.evidence_total += qr.evidence_total;
        self.count += 1;
        if let Some(t) = qr.answer_input_tokens {
            self.answer_in_sum += t;
            self.answer_in_count += 1;
        }
        if let Some(t) = qr.answer_output_tokens {
            self.answer_out_sum += t;
            self.answer_out_count += 1;
        }
        if let Some(t) = qr.judge_input_tokens {
            self.judge_in_sum += t;
            self.judge_in_count += 1;
        }
        if let Some(t) = qr.judge_output_tokens {
            self.judge_out_sum += t;
            self.judge_out_count += 1;
        }
        self.answer_cost_sum += answer_cost(qr, pricing);
        self.judge_cost_sum += judge_cost(qr, pricing);
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

    fn avg_answer_in(&self) -> f64 {
        if self.answer_in_count > 0 {
            self.answer_in_sum as f64 / self.answer_in_count as f64
        } else {
            0.0
        }
    }

    fn avg_answer_out(&self) -> f64 {
        if self.answer_out_count > 0 {
            self.answer_out_sum as f64 / self.answer_out_count as f64
        } else {
            0.0
        }
    }

    fn avg_judge_in(&self) -> f64 {
        if self.judge_in_count > 0 {
            self.judge_in_sum as f64 / self.judge_in_count as f64
        } else {
            0.0
        }
    }

    fn avg_judge_out(&self) -> f64 {
        if self.judge_out_count > 0 {
            self.judge_out_sum as f64 / self.judge_out_count as f64
        } else {
            0.0
        }
    }

    fn avg_query_cost(&self) -> f64 {
        if self.count > 0 {
            (self.answer_cost_sum + self.judge_cost_sum) / self.count as f64
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
                avg_answer_input_tokens: acc.avg_answer_in(),
                avg_answer_output_tokens: acc.avg_answer_out(),
                avg_query_cost_usd: acc.avg_query_cost(),
            })
        })
        .collect()
    }
}

/// USD cost of the answer LLM call carried on `qr` under `pricing`.
fn answer_cost(qr: &QueryResult, pricing: &Pricing) -> f64 {
    if qr.answer_model.is_empty() {
        return 0.0;
    }
    let in_cost = qr
        .answer_input_tokens
        .and_then(|t| pricing.cost_input(&qr.answer_model, t))
        .unwrap_or(0.0);
    let out_cost = qr
        .answer_output_tokens
        .and_then(|t| pricing.cost_output(&qr.answer_model, t))
        .unwrap_or(0.0);
    in_cost + out_cost
}

/// USD cost of the judge LLM call carried on `qr` under `pricing`.
fn judge_cost(qr: &QueryResult, pricing: &Pricing) -> f64 {
    let Some(model) = qr.judge_model.as_deref() else {
        return 0.0;
    };
    let in_cost = qr
        .judge_input_tokens
        .and_then(|t| pricing.cost_input(model, t))
        .unwrap_or(0.0);
    let out_cost = qr
        .judge_output_tokens
        .and_then(|t| pricing.cost_output(model, t))
        .unwrap_or(0.0);
    in_cost + out_cost
}

/// Aggregate query results into a benchmark report.
///
/// Backwards-compatible wrapper around [`aggregate_with_pricing`] that
/// supplies an empty pricing table — every cost field comes out as 0.
#[must_use]
pub fn aggregate(
    results: &[(QueryResult, f64, Option<f64>)],
    num_conversations: usize,
) -> BenchmarkReport {
    aggregate_with_pricing(results, num_conversations, &Pricing::empty())
}

/// Aggregate query results into a benchmark report with USD cost
/// rollups computed from `pricing`.
#[must_use]
pub fn aggregate_with_pricing(
    results: &[(QueryResult, f64, Option<f64>)],
    num_conversations: usize,
    pricing: &Pricing,
) -> BenchmarkReport {
    let mut overall = Accum::new();
    let mut by_sample: BTreeMap<String, Accum> = BTreeMap::new();

    for (qr, f1, judge) in results {
        overall.observe(qr, *f1, *judge, pricing);
        by_sample
            .entry(qr.sample_id.clone())
            .or_insert_with(Accum::new)
            .observe(qr, *f1, *judge, pricing);
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
            avg_answer_input_tokens: acc.avg_answer_in(),
            avg_answer_output_tokens: acc.avg_answer_out(),
            avg_query_cost_usd: acc.avg_query_cost(),
            categories: acc.categories(),
        })
        .collect();

    let total_answer_cost_usd = overall.answer_cost_sum;
    let total_judge_cost_usd = overall.judge_cost_sum;
    // `cost_per_question_usd` is the production usage figure —
    // answer-LLM only.  `total_query_cost_usd` retains the legacy
    // sum semantics for backward compatibility with older JSON
    // consumers but should not be used for headline reporting.
    let total_query_cost_usd = total_answer_cost_usd + total_judge_cost_usd;
    let cost_per_question_usd = if overall.count > 0 {
        total_answer_cost_usd / overall.count as f64
    } else {
        0.0
    };

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
        total_answer_cost_usd,
        total_judge_cost_usd,
        total_query_cost_usd,
        cost_per_question_usd,
        avg_answer_input_tokens: overall.avg_answer_in(),
        avg_answer_output_tokens: overall.avg_answer_out(),
        avg_judge_input_tokens: overall.avg_judge_in(),
        avg_judge_output_tokens: overall.avg_judge_out(),
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

    // Cost / token block — populated when `aggregate_with_pricing`
    // was used and the pricing table covered the models in play.
    if report.total_query_cost_usd > 0.0
        || report.avg_answer_input_tokens > 0.0
        || report.avg_judge_input_tokens > 0.0
    {
        println!();
        // Usage cost = answer LLM only.  That's what a production
        // customer of uniko would pay; the judge is bench-harness
        // overhead and is reported separately below.
        println!("  Usage cost (answer LLM):");
        println!(
            "  {:<32} ${:>10.4}",
            "Total (answer)", report.total_answer_cost_usd
        );
        println!(
            "  {:<32} ${:>10.6}",
            "Per question (mean)", report.cost_per_question_usd
        );
        println!(
            "  {:<32} {:>10.1}",
            "Answer input tokens (avg)", report.avg_answer_input_tokens
        );
        println!(
            "  {:<32} {:>10.1}",
            "Answer output tokens (avg)", report.avg_answer_output_tokens
        );

        // Evaluator overhead — bench-only, not part of usage cost.
        // Reported so the operator can size the bench's own bill,
        // but kept visually separate from the headline so nobody
        // mistakes judge spend for production cost.
        println!();
        println!("  Evaluator overhead (bench-only; not customer-visible):");
        println!(
            "  {:<32} ${:>10.4}",
            "Judge cost (total)", report.total_judge_cost_usd
        );
        println!(
            "  {:<32} {:>10.1}",
            "Judge input tokens (avg)", report.avg_judge_input_tokens
        );
        println!(
            "  {:<32} {:>10.1}",
            "Judge output tokens (avg)", report.avg_judge_output_tokens
        );
    }
    println!();
}

/// Write per-question details to a JSON file.
///
/// Backwards-compatible wrapper around [`write_json_with_pricing`]
/// that supplies an empty pricing table — per-question cost columns
/// come out as 0.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_json(
    results: &[(QueryResult, f64, Option<f64>)],
    report: &BenchmarkReport,
    path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    write_json_with_pricing(results, report, path, &Pricing::empty())
}

/// Write per-question details to a JSON file with USD cost columns
/// derived from `pricing`.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_json_with_pricing(
    results: &[(QueryResult, f64, Option<f64>)],
    report: &BenchmarkReport,
    path: impl AsRef<Path>,
    pricing: &Pricing,
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
            answer_input_tokens: qr.answer_input_tokens,
            answer_output_tokens: qr.answer_output_tokens,
            answer_model: qr.answer_model.clone(),
            judge_latency_ms: qr.judge_latency_ms,
            judge_input_tokens: qr.judge_input_tokens,
            judge_output_tokens: qr.judge_output_tokens,
            judge_model: qr.judge_model.clone(),
            answer_cost_usd: answer_cost(qr, pricing),
            judge_cost_usd: judge_cost(qr, pricing),
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
