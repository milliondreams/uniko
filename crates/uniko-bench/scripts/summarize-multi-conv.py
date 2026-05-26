#!/usr/bin/env python3
"""Aggregate per-conversation bench JSON summaries into a single report.

Reads every `<prefix>_<conv-id>.json` produced by `run-all-convs.sh`
and prints accuracy + token + cost rollups that match the in-tree
report style.  Cost numbers are corrected to count only the answer
LLM (production usage cost) — judge cost is reported separately as
evaluator overhead, never folded into the headline.

This separation matters because the bench's pre-fix Rust binary may
report `total_query_cost_usd = answer + judge`, which conflates
production usage with bench-harness overhead.  Use this script when
you want the customer-facing cost figure.

Usage:
    ./summarize-multi-conv.py <prefix>
    # e.g. ./summarize-multi-conv.py data/locomo_gemini31
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output-prefix>", file=sys.stderr)
        return 2
    prefix = sys.argv[1]
    files = sorted(glob.glob(f"{prefix}_conv-*.json"))
    if not files:
        print(f"no files match {prefix}_conv-*.json", file=sys.stderr)
        return 1

    # ----- accumulators ----------------------------------------------
    total_q = 0
    total_judged = 0
    judge_sum = 0.0
    f1_sum = 0.0
    ev_found = 0
    ev_total = 0
    recall_ms_sum = 0
    gen_ms_sum = 0
    judge_ms_sum = 0
    judge_ms_count = 0

    answer_in_sum = 0
    answer_in_count = 0
    answer_out_sum = 0
    answer_out_count = 0
    judge_in_sum = 0
    judge_in_count = 0
    judge_out_sum = 0
    judge_out_count = 0

    answer_cost_sum = 0.0
    judge_cost_sum = 0.0

    # per-category accuracy + token rollups
    cat_sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "ev_found": 0, "ev_total": 0, "f1": 0.0, "judge": 0.0, "judged": 0}
    )

    per_conv: list[dict] = []

    for path in files:
        body = json.loads(Path(path).read_text())
        summary = body["summary"]
        questions = body["questions"]
        per_conv.append({"file": path, "summary": summary, "n": len(questions)})

        for q in questions:
            total_q += 1
            f1_sum += q.get("f1") or 0.0
            ev_found += q.get("evidence_found") or 0
            ev_total += q.get("evidence_total") or 0
            recall_ms_sum += q.get("recall_latency_ms") or 0
            gen_ms_sum += q.get("generation_latency_ms") or 0
            if (jm := q.get("judge_latency_ms")) is not None:
                judge_ms_sum += jm
                judge_ms_count += 1
            if (j := q.get("judge")) is not None:
                judge_sum += j
                total_judged += 1

            if (t := q.get("answer_input_tokens")) is not None:
                answer_in_sum += t
                answer_in_count += 1
            if (t := q.get("answer_output_tokens")) is not None:
                answer_out_sum += t
                answer_out_count += 1
            if (t := q.get("judge_input_tokens")) is not None:
                judge_in_sum += t
                judge_in_count += 1
            if (t := q.get("judge_output_tokens")) is not None:
                judge_out_sum += t
                judge_out_count += 1

            answer_cost_sum += q.get("answer_cost_usd") or 0.0
            judge_cost_sum += q.get("judge_cost_usd") or 0.0

            cat = q.get("category", "?")
            c = cat_sums[cat]
            c["count"] += 1
            c["ev_found"] += q.get("evidence_found") or 0
            c["ev_total"] += q.get("evidence_total") or 0
            c["f1"] += q.get("f1") or 0.0
            if (j := q.get("judge")) is not None:
                c["judge"] += j
                c["judged"] += 1

    n_convs = len(files)

    # ----- printable headline ---------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  LoCoMo bench rollup — {n_convs} conversations, {total_q} questions")
    print(f"  ({len(files)} files: {Path(prefix).name}_conv-*.json)")
    print(f"{'=' * 64}")
    print(
        f"  {'Category':<14} {'Count':>6} {'Evidence%':>10} {'F1':>8} {'Judge':>8}"
    )
    print(f"  {'-' * 60}")
    cat_order = ["Single-hop", "Multi-hop", "Temporal", "Open-domain", "Adversarial"]
    for cat in cat_order:
        if cat not in cat_sums:
            continue
        c = cat_sums[cat]
        ev_rate = (c["ev_found"] / c["ev_total"] * 100.0) if c["ev_total"] else 0.0
        avg_f1 = c["f1"] / c["count"] if c["count"] else 0.0
        avg_judge = (c["judge"] / c["judged"]) if c["judged"] else None
        judge_str = f"{avg_judge:.3f}" if avg_judge is not None else "N/A"
        print(
            f"  {cat:<14} {c['count']:>6} {ev_rate:>9.1f}% {avg_f1:>8.3f} {judge_str:>8}"
        )
    print(f"  {'-' * 60}")
    overall_ev = (ev_found / ev_total * 100.0) if ev_total else 0.0
    overall_f1 = f1_sum / total_q if total_q else 0.0
    overall_judge = judge_sum / total_judged if total_judged else 0.0
    print(
        f"  {'Overall':<14} {total_q:>6} {overall_ev:>9.1f}% {overall_f1:>8.3f} {overall_judge:>8.3f}"
    )
    print(f"{'=' * 64}")

    # ----- production usage cost (answer LLM only) ------------------
    cost_per_q = (answer_cost_sum / total_q) if total_q else 0.0
    print()
    print("  Usage cost (answer LLM only):")
    print(f"    Total (answer)              ${answer_cost_sum:10.4f}")
    print(f"    Per question (mean)         ${cost_per_q:10.6f}")
    if answer_in_count:
        print(f"    Answer input tokens (avg)   {answer_in_sum / answer_in_count:>10.1f}")
    if answer_out_count:
        print(f"    Answer output tokens (avg)  {answer_out_sum / answer_out_count:>10.1f}")

    # ----- evaluator overhead (judge, separated) -------------------
    print()
    print("  Evaluator overhead (bench-only; not customer-visible):")
    print(f"    Judge cost (total)          ${judge_cost_sum:10.4f}")
    if judge_in_count:
        print(f"    Judge input tokens (avg)    {judge_in_sum / judge_in_count:>10.1f}")
    if judge_out_count:
        print(f"    Judge output tokens (avg)   {judge_out_sum / judge_out_count:>10.1f}")
    if judge_ms_count:
        print(f"    Judge latency (avg ms)      {judge_ms_sum / judge_ms_count:>10.1f}")
    print()

    # ----- per-conversation table ----------------------------------
    print(f"  Per-conversation breakdown:")
    print(f"    {'sample_id':<14} {'Count':>6} {'Judge':>8} {'Recall ms':>10}")
    print(f"    {'-' * 50}")
    for entry in per_conv:
        s = entry["summary"]
        oj = s.get("overall_judge")
        oj_s = f"{oj:.3f}" if oj is not None else "N/A"
        print(
            f"    {Path(entry['file']).stem.split('_')[-1]:<14} "
            f"{entry['n']:>6} {oj_s:>8} {s.get('avg_recall_latency_ms', 0):>10.0f}"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
