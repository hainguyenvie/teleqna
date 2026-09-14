#!/usr/bin/env python3
"""Paired comparison of the memorisation-probe arms.

Every arm runs the same rows, so the comparison is paired and McNemar's test
applies. That matters: on 600 rows an unpaired comparison has a ~2.0pp standard
error per arm and cannot resolve a 4pp gap, while the paired test only looks at
the rows where the two arms disagree and resolves it comfortably.

Exact binomial two-sided p on the discordant pairs — no chi-square
approximation, no continuity-correction argument, and it stays valid when the
discordant count is small, which is the case this is most likely to be in.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(run_dir: Path) -> dict[str, bool]:
    path = run_dir / "results.jsonl"
    out: dict[str, bool] = {}
    for line in path.open(encoding="utf-8"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not record.get("error"):
            out[record["sample_id"]] = bool(record["correct"])
    return out


def binom_two_sided(b: int, c: int) -> float:
    """P(|deviation| >= observed) under H0: a discordant pair is 50/50."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict:
    shared = sorted(set(a) & set(b))
    only_a = sum(1 for k in shared if a[k] and not b[k])
    only_b = sum(1 for k in shared if b[k] and not a[k])
    acc_a = sum(a[k] for k in shared) / len(shared) if shared else 0.0
    acc_b = sum(b[k] for k in shared) / len(shared) if shared else 0.0
    return {
        "paired_rows": len(shared),
        "accuracy_baseline": round(acc_a, 4),
        "accuracy_variant": round(acc_b, 4),
        "delta_pp": round(100 * (acc_b - acc_a), 2),
        "baseline_only_correct": only_a,
        "variant_only_correct": only_b,
        "discordant": only_a + only_b,
        "p_value": round(binom_two_sided(only_a, only_b), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True,
                        help="dir holding one subdir per arm")
    parser.add_argument("--baseline", default="probe_orig")
    parser.add_argument("--arms", nargs="+",
                        default=["probe_perm", "probe_distract", "probe_paraphrase"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base = load(args.results_root / args.baseline)
    report = {"baseline": args.baseline, "baseline_rows": len(base), "comparisons": {}}
    for arm in args.arms:
        run_dir = args.results_root / arm
        if not (run_dir / "results.jsonl").exists():
            report["comparisons"][arm] = {"error": "missing results.jsonl"}
            continue
        report["comparisons"][arm] = mcnemar(base, load(run_dir))

    for name in ("tsguess_distractor", "tsguess_gold"):
        path = args.results_root / name / "summary.json"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            report[name] = {
                key: summary[key]
                for key in ("total", "exact_match", "exact_match_rate", "mean_token_f1",
                            "mean_question_overlap_floor", "high_f1_rate_ge_0.8", "verdict")
                if key in summary
            }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
