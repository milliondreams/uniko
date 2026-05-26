#!/usr/bin/env python3
"""Merge LoCoMo gemini-3.1 outputs back into one JSON.

Combines:
  - 6 full-conv files: data/locomo_gemini31_{conv}.json (conv-26/30/41/43/47/49)
  - 20 per-category files: data/locomo_gemini31_{conv}_cat{N}.json (conv-42/44/48/50 x cats 1-5)

Re-aggregates summary stats over all questions. The `questions` array
is the source of truth — every summary field is derived from it,
mirroring `report::aggregate_with_pricing` in the bench's Rust source.

Output: data/locomo_gemini31_merged.json
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # uniko2/
DATA = ROOT / "data"
OUT = DATA / "locomo_gemini31_merged.json"

FULL_CONVS = ["conv-26", "conv-30", "conv-41", "conv-43", "conv-47", "conv-49"]
SPLIT_CONVS = ["conv-42", "conv-44", "conv-48", "conv-50"]
CATEGORIES_ORDERED = ["Single-hop", "Multi-hop", "Temporal", "Open-domain", "Adversarial"]


def _avg(xs, default=None):
    """Average of non-None entries; returns `default` (None unless overridden)
    when every entry is None. Use a numeric default (e.g. 0.0) only for fields
    where 'missing' should be treated as 'zero' (sums)."""
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else default


def _sum(xs):
    return sum(x for x in xs if x is not None)


def aggregate_questions(questions: list[dict]) -> dict:
    """Compute the summary block matching the bench's Rust aggregator."""
    n = len(questions)
    if n == 0:
        return {
            "overall_evidence_hit_rate": 0.0,
            "overall_f1": 0.0,
            "overall_judge": 0.0,
            "total_questions": 0,
            "total_conversations": 0,
            "categories": [],
            "conversations": [],
        }

    # Convs nested
    by_conv = defaultdict(list)
    for q in questions:
        by_conv[q["sample_id"]].append(q)
    conv_blocks = []
    for sid in sorted(by_conv.keys()):
        qs = by_conv[sid]
        # Per-conv categories
        by_cat = defaultdict(list)
        for q in qs:
            by_cat[q["category"]].append(q)
        cat_blocks = []
        for cat in CATEGORIES_ORDERED:
            if cat not in by_cat:
                continue
            cqs = by_cat[cat]
            cat_blocks.append(_category_block(cat, cqs))
        conv_blocks.append({
            "sample_id": sid,
            "count": len(qs),
            "evidence_hit_rate": _hit_rate(qs),
            "avg_f1": _avg([q.get("f1") for q in qs]),
            "avg_judge": _avg([q.get("judge") for q in qs]),
            "avg_recall_latency_ms": _avg([q.get("recall_latency_ms") for q in qs]),
            "avg_generation_latency_ms": _avg([q.get("generation_latency_ms") for q in qs]),
            "avg_answer_input_tokens": _avg([q.get("answer_input_tokens") for q in qs]),
            "avg_answer_output_tokens": _avg([q.get("answer_output_tokens") for q in qs]),
            "avg_query_cost_usd": _avg([
                (q.get("answer_cost_usd") or 0.0) + (q.get("judge_cost_usd") or 0.0)
                for q in qs
            ]),
            "categories": cat_blocks,
        })

    # Overall categories
    by_overall_cat = defaultdict(list)
    for q in questions:
        by_overall_cat[q["category"]].append(q)
    overall_cat_blocks = [
        _category_block(c, by_overall_cat[c]) for c in CATEGORIES_ORDERED if c in by_overall_cat
    ]

    total_answer_cost = _sum(q.get("answer_cost_usd") for q in questions)
    total_judge_cost = _sum(q.get("judge_cost_usd") for q in questions)
    total_query_cost = total_answer_cost + total_judge_cost

    return {
        "overall_evidence_hit_rate": _hit_rate(questions),
        "overall_f1": _avg([q.get("f1") for q in questions]),
        "overall_judge": _avg([q.get("judge") for q in questions]),
        "total_questions": n,
        "total_conversations": len(by_conv),
        "categories": overall_cat_blocks,
        "avg_recall_latency_ms": _avg([q.get("recall_latency_ms") for q in questions]),
        "avg_generation_latency_ms": _avg([q.get("generation_latency_ms") for q in questions]),
        "conversations": conv_blocks,
        "total_answer_cost_usd": total_answer_cost,
        "total_judge_cost_usd": total_judge_cost,
        "total_query_cost_usd": total_query_cost,
        "cost_per_question_usd": total_query_cost / n if n else 0.0,
        "avg_answer_input_tokens": _avg([q.get("answer_input_tokens") for q in questions]),
        "avg_answer_output_tokens": _avg([q.get("answer_output_tokens") for q in questions]),
        "avg_judge_input_tokens": _avg([q.get("judge_input_tokens") for q in questions]),
        "avg_judge_output_tokens": _avg([q.get("judge_output_tokens") for q in questions]),
    }


def _hit_rate(qs):
    hits = sum(1 for q in qs if (q.get("evidence_found") or 0) > 0)
    return hits / len(qs) if qs else 0.0


def _category_block(name, qs):
    return {
        "name": name,
        "count": len(qs),
        "evidence_hit_rate": _hit_rate(qs),
        "avg_f1": _avg([q.get("f1") for q in qs]),
        "avg_judge": _avg([q.get("judge") for q in qs]),
        "avg_recall_latency_ms": _avg([q.get("recall_latency_ms") for q in qs]),
        "avg_generation_latency_ms": _avg([q.get("generation_latency_ms") for q in qs]),
        "avg_answer_input_tokens": _avg([q.get("answer_input_tokens") for q in qs]),
        "avg_answer_output_tokens": _avg([q.get("answer_output_tokens") for q in qs]),
        "avg_query_cost_usd": _avg([
            (q.get("answer_cost_usd") or 0.0) + (q.get("judge_cost_usd") or 0.0) for q in qs
        ]),
    }


def main():
    all_questions = []
    files_loaded = []

    for conv in FULL_CONVS:
        path = DATA / f"locomo_gemini31_{conv}.json"
        if not path.exists():
            print(f"[warn] missing full-conv file: {path}", file=sys.stderr)
            continue
        with path.open() as f:
            d = json.load(f)
        all_questions.extend(d["questions"])
        files_loaded.append((path.name, len(d["questions"])))

    for conv in SPLIT_CONVS:
        for cat in range(1, 6):
            path = DATA / f"locomo_gemini31_{conv}_cat{cat}.json"
            if not path.exists():
                print(f"[warn] missing per-cat file: {path}", file=sys.stderr)
                continue
            with path.open() as f:
                d = json.load(f)
            all_questions.extend(d["questions"])
            files_loaded.append((path.name, len(d["questions"])))

    summary = aggregate_questions(all_questions)

    out = {"summary": summary, "questions": all_questions}
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"[ok] merged {len(files_loaded)} files into {OUT.relative_to(ROOT)}")
    print(f"[ok] total questions: {summary['total_questions']} across {summary['total_conversations']} conversations")
    print()
    def fmt(v, w=6, p=4):
        return f"{v:.{p}f}" if v is not None else " N/A ".rjust(w)

    print(f"  overall judge:         {fmt(summary['overall_judge'])}  (over questions with non-null judge)")
    print(f"  overall f1:            {fmt(summary['overall_f1'])}")
    print(f"  overall evidence hit:  {fmt(summary['overall_evidence_hit_rate'])}")
    print(f"  total cost:            ${summary['total_query_cost_usd']:.4f}")
    print(f"  cost/question:         ${summary['cost_per_question_usd']:.6f}")
    print()
    print("  by category:")
    for cat in summary["categories"]:
        print(f"    {cat['name']:14s} n={cat['count']:4d}  judge={fmt(cat['avg_judge'])}  f1={fmt(cat['avg_f1'])}  hit={fmt(cat['evidence_hit_rate'])}")
    print()
    print("  by conversation:")
    for c in summary["conversations"]:
        print(f"    {c['sample_id']:8s} n={c['count']:4d}  judge={fmt(c['avg_judge'])}  f1={fmt(c['avg_f1'])}")


if __name__ == "__main__":
    main()
