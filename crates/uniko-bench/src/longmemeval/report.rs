//! LongMemEval benchmark result aggregation and reporting.

use std::collections::HashMap;
use std::path::Path;

use serde::Serialize;

use super::data::LmeQuestionType;
use super::query::LmeQueryResult;

/// Per-category evaluation results for LongMemEval.
#[derive(Debug, Serialize)]
pub struct LmeCategoryReport {
    pub name: String,
    pub long_name: String,
    pub count: usize,
    /// Fraction of questions where context_contains_answer is true.
    pub context_contains_rate: f64,
    /// Mean Recall@5 across questions in this category.
    pub avg_recall_at_5: f64,
    /// Mean NDCG@5 across questions in this category.
    pub avg_ndcg_at_5: f64,
    /// Mean judge score (if evaluated).
    pub avg_judge: Option<f64>,
    pub avg_recall_latency_ms: f64,
    pub avg_generation_latency_ms: f64,
}

/// Full LongMemEval benchmark report.
#[derive(Debug, Serialize)]
pub struct LmeBenchmarkReport {
    pub benchmark_name: String,
    pub total_questions: usize,
    pub total_items: usize,
    /// Overall context_contains_answer rate.
    pub overall_context_contains_rate: f64,
    /// Overall mean Recall@5.
    pub overall_recall_at_5: f64,
    /// Overall mean NDCG@5.
    pub overall_ndcg_at_5: f64,
    /// Overall mean judge score.
    pub overall_judge: Option<f64>,
    pub categories: Vec<LmeCategoryReport>,
    pub avg_recall_latency_ms: f64,
    pub avg_generation_latency_ms: f64,
}

/// Per-question detail for JSON output.
#[derive(Debug, Serialize)]
pub struct LmeQuestionDetail {
    pub question_id: String,
    pub question: String,
    pub question_type: String,
    pub is_abstention: bool,
    pub gold_answer: String,
    pub predicted_answer: String,
    pub context_contains_answer: bool,
    pub recall_at_5: f64,
    pub ndcg_at_5: f64,
    pub judge: Option<f64>,
    pub recall_latency_ms: u64,
    pub generation_latency_ms: u64,
}

/// Aggregate query results into a LongMemEval benchmark report.
pub fn aggregate_lme(
    results: &[(LmeQueryResult, Option<f64>)],
    num_items: usize,
) -> LmeBenchmarkReport {
    struct CatAccum {
        context_contains_count: usize,
        recall_at_5_sum: f64,
        ndcg_at_5_sum: f64,
        judge_scores: Vec<f64>,
        recall_ms_sum: u64,
        gen_ms_sum: u64,
        count: usize,
    }

    let mut by_type: HashMap<LmeQuestionType, CatAccum> = HashMap::new();

    for (qr, judge) in results {
        let acc = by_type.entry(qr.question_type).or_insert(CatAccum {
            context_contains_count: 0,
            recall_at_5_sum: 0.0,
            ndcg_at_5_sum: 0.0,
            judge_scores: Vec::new(),
            recall_ms_sum: 0,
            gen_ms_sum: 0,
            count: 0,
        });

        if qr.context_contains_answer {
            acc.context_contains_count += 1;
        }
        acc.recall_at_5_sum += qr.recall_at_5;
        acc.ndcg_at_5_sum += qr.ndcg_at_5;
        if let Some(j) = judge {
            acc.judge_scores.push(*j);
        }
        acc.recall_ms_sum += qr.recall_latency_ms;
        acc.gen_ms_sum += qr.generation_latency_ms;
        acc.count += 1;
    }

    // Build per-category reports in display order.
    let category_order = [
        LmeQuestionType::SingleSessionUser,
        LmeQuestionType::SingleSessionAssistant,
        LmeQuestionType::SingleSessionPreference,
        LmeQuestionType::MultiSession,
        LmeQuestionType::TemporalReasoning,
        LmeQuestionType::KnowledgeUpdate,
    ];

    let categories: Vec<LmeCategoryReport> = category_order
        .iter()
        .filter_map(|qt| {
            let acc = by_type.get(qt)?;
            let avg_judge = if acc.judge_scores.is_empty() {
                None
            } else {
                Some(acc.judge_scores.iter().sum::<f64>() / acc.judge_scores.len() as f64)
            };

            Some(LmeCategoryReport {
                name: qt.name().to_string(),
                long_name: qt.long_name().to_string(),
                count: acc.count,
                context_contains_rate: acc.context_contains_count as f64 / acc.count as f64,
                avg_recall_at_5: acc.recall_at_5_sum / acc.count as f64,
                avg_ndcg_at_5: acc.ndcg_at_5_sum / acc.count as f64,
                avg_judge,
                avg_recall_latency_ms: acc.recall_ms_sum as f64 / acc.count as f64,
                avg_generation_latency_ms: acc.gen_ms_sum as f64 / acc.count as f64,
            })
        })
        .collect();

    let total = results.len();
    let overall_context_contains = if total > 0 {
        results
            .iter()
            .filter(|(qr, _)| qr.context_contains_answer)
            .count() as f64
            / total as f64
    } else {
        0.0
    };
    let overall_recall_5 = if total > 0 {
        results.iter().map(|(qr, _)| qr.recall_at_5).sum::<f64>() / total as f64
    } else {
        0.0
    };
    let overall_ndcg_5 = if total > 0 {
        results.iter().map(|(qr, _)| qr.ndcg_at_5).sum::<f64>() / total as f64
    } else {
        0.0
    };
    let overall_judge = {
        let judged: Vec<f64> = results.iter().filter_map(|(_, j)| *j).collect();
        if judged.is_empty() {
            None
        } else {
            Some(judged.iter().sum::<f64>() / judged.len() as f64)
        }
    };
    let avg_recall = if total > 0 {
        results
            .iter()
            .map(|(qr, _)| qr.recall_latency_ms as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };
    let avg_gen = if total > 0 {
        results
            .iter()
            .map(|(qr, _)| qr.generation_latency_ms as f64)
            .sum::<f64>()
            / total as f64
    } else {
        0.0
    };

    LmeBenchmarkReport {
        benchmark_name: "LongMemEval".to_string(),
        total_questions: total,
        total_items: num_items,
        overall_context_contains_rate: overall_context_contains,
        overall_recall_at_5: overall_recall_5,
        overall_ndcg_at_5: overall_ndcg_5,
        overall_judge,
        categories,
        avg_recall_latency_ms: avg_recall,
        avg_generation_latency_ms: avg_gen,
    }
}

/// Print the LongMemEval report as a formatted table to stdout.
pub fn print_lme_report(report: &LmeBenchmarkReport) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  {} Benchmark Results", report.benchmark_name);
    println!(
        "  {} items, {} questions",
        report.total_items, report.total_questions
    );
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!(
        "  {:<6} {:>6} {:>10} {:>8} {:>8} {:>8} {:>10}",
        "Type", "Count", "Contains%", "R@5", "NDCG@5", "Judge", "Recall ms"
    );
    println!("  ─────────────────────────────────────────────────────────────────────────");

    for cat in &report.categories {
        let judge_str = cat
            .avg_judge
            .map(|j| format!("{j:.3}"))
            .unwrap_or_else(|| "N/A".to_string());
        println!(
            "  {:<6} {:>6} {:>9.1}% {:>8.3} {:>8.3} {:>8} {:>10.0}",
            cat.name,
            cat.count,
            cat.context_contains_rate * 100.0,
            cat.avg_recall_at_5,
            cat.avg_ndcg_at_5,
            judge_str,
            cat.avg_recall_latency_ms,
        );
    }

    println!("  ─────────────────────────────────────────────────────────────────────────");
    let judge_str = report
        .overall_judge
        .map(|j| format!("{j:.3}"))
        .unwrap_or_else(|| "N/A".to_string());
    println!(
        "  {:<6} {:>6} {:>9.1}% {:>8.3} {:>8.3} {:>8} {:>10.0}",
        "ALL",
        report.total_questions,
        report.overall_context_contains_rate * 100.0,
        report.overall_recall_at_5,
        report.overall_ndcg_at_5,
        judge_str,
        report.avg_recall_latency_ms,
    );
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!();
}

/// Write per-question details to a JSON file.
pub fn write_lme_json(
    results: &[(LmeQueryResult, Option<f64>)],
    report: &LmeBenchmarkReport,
    path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let details: Vec<LmeQuestionDetail> = results
        .iter()
        .map(|(qr, judge)| LmeQuestionDetail {
            question_id: qr.question_id.clone(),
            question: qr.question.clone(),
            question_type: qr.question_type.name().to_string(),
            is_abstention: qr.is_abstention,
            gold_answer: qr.gold_answer.clone(),
            predicted_answer: qr.predicted_answer.clone(),
            context_contains_answer: qr.context_contains_answer,
            recall_at_5: qr.recall_at_5,
            ndcg_at_5: qr.ndcg_at_5,
            judge: *judge,
            recall_latency_ms: qr.recall_latency_ms,
            generation_latency_ms: qr.generation_latency_ms,
        })
        .collect();

    #[derive(Serialize)]
    struct FullOutput<'a> {
        summary: &'a LmeBenchmarkReport,
        questions: Vec<LmeQuestionDetail>,
    }

    let output = FullOutput {
        summary: report,
        questions: details,
    };

    let json = serde_json::to_string_pretty(&output)?;
    std::fs::write(path, json)?;
    Ok(())
}
