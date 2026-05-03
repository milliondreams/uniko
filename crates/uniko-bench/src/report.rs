//! Benchmark result aggregation and reporting.

use std::collections::HashMap;
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
}

/// Per-question detail for JSON output.
#[derive(Debug, Serialize)]
pub struct QuestionDetail {
    pub question: String,
    pub gold_answer: String,
    pub predicted_answer: String,
    pub category: String,
    pub evidence_found: usize,
    pub evidence_total: usize,
    pub f1: f64,
    pub judge: Option<f64>,
    pub recall_latency_ms: u64,
    /// Full bundle of retrieved items (node_type / score / content).
    /// Always populated; useful for offline failure-mode analysis.
    pub recall_bundle: Vec<crate::query::RecalledItem>,
}

/// Aggregate query results into a benchmark report.
pub fn aggregate(
    results: &[(QueryResult, f64, Option<f64>)],
    num_conversations: usize,
) -> BenchmarkReport {
    struct CatAccum {
        f1_sum: f64,
        judge_scores: Vec<f64>,
        recall_ms_sum: u64,
        gen_ms_sum: u64,
        evidence_found: usize,
        evidence_total: usize,
        count: usize,
    }

    let mut by_category: HashMap<QuestionCategory, CatAccum> = HashMap::new();

    for (qr, f1, judge) in results {
        let acc = by_category.entry(qr.category).or_insert(CatAccum {
            f1_sum: 0.0,
            judge_scores: Vec::new(),
            recall_ms_sum: 0,
            gen_ms_sum: 0,
            evidence_found: 0,
            evidence_total: 0,
            count: 0,
        });
        acc.f1_sum += f1;
        if let Some(j) = judge {
            acc.judge_scores.push(*j);
        }
        acc.recall_ms_sum += qr.recall_latency_ms;
        acc.gen_ms_sum += qr.generation_latency_ms;
        acc.evidence_found += qr.evidence_found;
        acc.evidence_total += qr.evidence_total;
        acc.count += 1;
    }

    let categories: Vec<CategoryReport> = [
        QuestionCategory::SingleHop,
        QuestionCategory::MultiHop,
        QuestionCategory::Temporal,
        QuestionCategory::OpenDomain,
        QuestionCategory::Adversarial,
    ]
    .iter()
    .filter_map(|cat| {
        let acc = by_category.get(cat)?;
        let avg_judge = if acc.judge_scores.is_empty() {
            None
        } else {
            Some(acc.judge_scores.iter().sum::<f64>() / acc.judge_scores.len() as f64)
        };
        let evidence_hit_rate = if acc.evidence_total > 0 {
            acc.evidence_found as f64 / acc.evidence_total as f64
        } else {
            0.0
        };

        Some(CategoryReport {
            name: cat.name().to_string(),
            count: acc.count,
            evidence_hit_rate,
            avg_f1: acc.f1_sum / acc.count as f64,
            avg_judge,
            avg_recall_latency_ms: acc.recall_ms_sum as f64 / acc.count as f64,
            avg_generation_latency_ms: acc.gen_ms_sum as f64 / acc.count as f64,
        })
    })
    .collect();

    let total = results.len();
    let overall_f1 = if total > 0 {
        results.iter().map(|(_, f, _)| f).sum::<f64>() / total as f64
    } else {
        0.0
    };
    let overall_judge = {
        let judged: Vec<f64> = results.iter().filter_map(|(_, _, j)| *j).collect();
        if judged.is_empty() {
            None
        } else {
            Some(judged.iter().sum::<f64>() / judged.len() as f64)
        }
    };
    let avg_recall = if total > 0 {
        results
            .iter()
            .map(|(qr, _, _)| qr.recall_latency_ms as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };
    let avg_gen = if total > 0 {
        results
            .iter()
            .map(|(qr, _, _)| qr.generation_latency_ms as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };

    let total_evidence_found: usize = results.iter().map(|(qr, _, _)| qr.evidence_found).sum();
    let total_evidence_total: usize = results.iter().map(|(qr, _, _)| qr.evidence_total).sum();
    let overall_evidence_hit_rate = if total_evidence_total > 0 {
        total_evidence_found as f64 / total_evidence_total as f64
    } else {
        0.0
    };

    BenchmarkReport {
        overall_evidence_hit_rate,
        overall_f1,
        overall_judge,
        total_questions: total,
        total_conversations: num_conversations,
        categories,
        avg_recall_latency_ms: avg_recall,
        avg_generation_latency_ms: avg_gen,
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
            question: qr.question.clone(),
            gold_answer: qr.gold_answer.clone(),
            predicted_answer: qr.predicted_answer.clone(),
            category: qr.category.name().to_string(),
            evidence_found: qr.evidence_found,
            evidence_total: qr.evidence_total,
            f1: *f1,
            judge: *judge,
            recall_latency_ms: qr.recall_latency_ms,
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
