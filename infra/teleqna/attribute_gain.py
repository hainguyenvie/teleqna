#!/usr/bin/env python3
"""Decide whether an optimised prompt earned its points or found the artefact.

An optimiser maximises the metric. On teleqna the single strongest signal in the
data is not telecom knowledge — it is distractor shape: "All of the above" is
gold in 89.81% of the 824 rows where it appears, "None of the above" in 0.31% of
647. A no-knowledge policy built from those cues scores 39.72%, and it already
beats the thinking baseline on 764 rows. Any optimiser pointed at accuracy will
find that gradient, and a prompt that encodes it will show a real gain on this
benchmark while learning nothing that transfers to the other six columns.

So the delta is never read on its own. It is split four ways:

  by subject          a gain concentrated in Standards specifications would
                      contradict the baseline finding that the subject is a
                      knowledge gap — treat it as suspicious, not as success.
  shape-cued rows     gain on the 824 "All of the above" rows against gain on
                      the 8,000+ rows with no shape cue. The second number is
                      the one that means anything.
  heuristic agreement did the model move *toward* the no-knowledge policy? A
                      rising agreement rate with a flat knowledge-win count is
                      the signature of artefact-chasing.
  knowledge wins      rows the shape heuristic gets wrong and the model gets
                      right. This is the only count that cannot be explained by
                      the artefact, and it is the headline.

Baseline results are subset to the held-out ids rather than re-run, so the
comparison is paired on identical rows at zero extra cost.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ALL_ABOVE = re.compile(r"(?i)^all of the above")
NONE_ABOVE = re.compile(r"(?i)^none of the above")
BOTH_OF = re.compile(r"(?i)^both ")


def heuristic_index(row: dict[str, Any]) -> int:
    choices = row["choices"]
    for index, choice in enumerate(choices):
        if ALL_ABOVE.search(choice):
            return index
    for index, choice in enumerate(choices):
        if BOTH_OF.search(choice):
            return index
    live = [i for i, c in enumerate(choices) if not NONE_ABOVE.search(c)] or list(
        range(len(choices))
    )
    return max(live, key=lambda i: len(choices[i]))


def load_results(run_dir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for line in (run_dir / "results.jsonl").open(encoding="utf-8"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not record.get("error"):
            out[record["sample_id"]] = bool(record["correct"])
    return out


def binom_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def mcnemar(base: dict[str, bool], new: dict[str, bool], ids: list[str]) -> dict:
    shared = [i for i in ids if i in base and i in new]
    only_base = sum(1 for i in shared if base[i] and not new[i])
    only_new = sum(1 for i in shared if new[i] and not base[i])
    n = len(shared) or 1
    return {
        "rows": len(shared),
        "baseline": round(sum(base[i] for i in shared) / n, 4),
        "optimized": round(sum(new[i] for i in shared) / n, 4),
        "delta_pp": round(100 * (sum(new[i] for i in shared) - sum(base[i] for i in shared)) / n, 2),
        "baseline_only": only_base,
        "optimized_only": only_new,
        "p_value": round(binom_two_sided(only_base, only_new), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="full test.jsonl")
    parser.add_argument("--heldout", type=Path, required=True, help="splits/heldout.jsonl")
    parser.add_argument("--baseline", type=Path, required=True, help="baseline run dir (any scale)")
    parser.add_argument("--optimized", type=Path, required=True, help="optimised run dir")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = {r["sample_id"]: r for r in (json.loads(l) for l in args.data.open(encoding="utf-8"))}
    heldout_ids = [json.loads(l)["sample_id"] for l in args.heldout.open(encoding="utf-8")]
    base = load_results(args.baseline)
    new = load_results(args.optimized)

    missing = [i for i in heldout_ids if i not in base or i not in new]
    ids = [i for i in heldout_ids if i in base and i in new]

    report: dict[str, Any] = {
        "heldout_rows": len(heldout_ids),
        "paired_rows": len(ids),
        "unpaired_dropped": len(missing),
        "overall": mcnemar(base, new, ids),
    }

    by_subject: dict[str, list[str]] = defaultdict(list)
    for sample_id in ids:
        by_subject[rows[sample_id].get("subject") or "<none>"].append(sample_id)
    report["by_subject"] = {
        subject: mcnemar(base, new, subject_ids)
        for subject, subject_ids in sorted(by_subject.items(), key=lambda kv: -len(kv[1]))
    }

    cued = [i for i in ids if any(ALL_ABOVE.search(c) or BOTH_OF.search(c)
                                  for c in rows[i]["choices"])]
    uncued = [i for i in ids if i not in set(cued)]
    report["shape_cued_rows"] = mcnemar(base, new, cued)
    report["no_shape_cue_rows"] = mcnemar(base, new, uncued)

    def split_vs_heuristic(results: dict[str, bool]) -> dict[str, int]:
        counts = {"both_right": 0, "heuristic_only": 0, "model_only": 0, "both_wrong": 0}
        for sample_id in ids:
            row = rows[sample_id]
            h = heuristic_index(row) == int(row["answer"])
            m = results[sample_id]
            key = ("both_right" if h and m else "heuristic_only" if h
                   else "model_only" if m else "both_wrong")
            counts[key] += 1
        return counts

    base_split, new_split = split_vs_heuristic(base), split_vs_heuristic(new)
    agree_base = (base_split["both_right"] + base_split["both_wrong"]) / max(1, len(ids))
    agree_new = (new_split["both_right"] + new_split["both_wrong"]) / max(1, len(ids))
    report["vs_shape_heuristic"] = {
        "baseline": base_split,
        "optimized": new_split,
        "knowledge_wins_baseline": base_split["model_only"],
        "knowledge_wins_optimized": new_split["model_only"],
        "knowledge_wins_delta": new_split["model_only"] - base_split["model_only"],
        "outcome_agreement_baseline": round(agree_base, 4),
        "outcome_agreement_optimized": round(agree_new, 4),
    }

    gained = report["overall"]["optimized_only"] - report["overall"]["baseline_only"]
    knowledge_gained = report["vs_shape_heuristic"]["knowledge_wins_delta"]
    if gained <= 0:
        verdict = "no net gain to attribute"
    elif knowledge_gained >= 0.7 * gained:
        verdict = ("gain is mostly knowledge — it lands on rows the shape policy gets wrong; "
                   "this is the kind that should transfer")
    elif knowledge_gained <= 0.3 * gained:
        verdict = ("gain is mostly artefact — it lands on rows the shape policy already solves; "
                   "expect it not to transfer, and do not report it as telecom capability")
    else:
        verdict = "gain is mixed — report the no-shape-cue number alongside the headline"
    report["verdict"] = verdict

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
