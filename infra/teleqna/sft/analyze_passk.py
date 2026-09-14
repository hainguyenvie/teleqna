#!/usr/bin/env python3
"""Read a profile_pass_rate.py output and report the ladder GRPO is bounded by.

Four numbers, and the distance between them is the whole decision:

  pass@1    mean sampled accuracy at T=1 — what the policy is worth per rollout
  vote@k    majority over the k sampled letters — the free ensembling baseline
  pass@k    correct in ANY sampled branch — the absolute ceiling of any method
            that reweights branches it already produces, GRPO included
  band      rows with 1 <= correct <= k-1 — the only rows that carry a GRPO
            gradient at all; everything else has zero advantage in-group

If pass@k sits close to greedy, the "information is already in there" story is
false at fixed choice order and GRPO is optimising a channel with no headroom.
Truncation is reported alongside because a rollout that runs out of tokens
scores zero and is indistinguishable from a wrong answer in the count.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--greedy", type=float, default=None,
                    help="greedy accuracy of the same model on the same rows, "
                         "for the pass@1-vs-greedy sanity check")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.profile.open(encoding="utf-8")]
    k = rows[0]["k"]
    n = len(rows)

    tot_correct = sum(r["correct"] for r in rows)
    tot_unparsed = sum(r["unparsed"] for r in rows)
    oracle = sum(1 for r in rows if r["correct"] >= 1)
    allright = sum(1 for r in rows if r["correct"] == k)
    band = sum(1 for r in rows if 1 <= r["correct"] <= k - 1)

    votes = 0
    have_letters = all("letters" in r for r in rows)
    if have_letters:
        for r in rows:
            c = collections.Counter(x for x in r["letters"] if x)
            if c and c.most_common(1)[0][0] == r["answer"]:
                votes += 1

    hist = collections.Counter(r["correct"] for r in rows)
    by_subject: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_subject[r.get("subject", "?")].append(r)

    summary = {
        "rows": n, "k": k,
        "pass@1": round(tot_correct / (n * k), 4),
        "vote@k": round(votes / n, 4) if have_letters else None,
        f"pass@{k}": round(oracle / n, 4),
        "greedy_ref": args.greedy,
        "always_right": allright, "always_wrong": n - oracle,
        "trainable_band": band, "trainable_frac": round(band / n, 4),
        "unparsed_rollouts": round(tot_unparsed / (n * k), 4),
        "pass@1_ci95": [round(x, 4) for x in wilson(tot_correct, n * k)],
        f"pass@{k}_ci95": [round(x, 4) for x in wilson(oracle, n)],
        "headroom_pass@k_minus_pass@1": round(oracle / n - tot_correct / (n * k), 4),
        "by_subject": {
            s: {"n": len(v),
                "pass@1": round(sum(r["correct"] for r in v) / (len(v) * k), 4),
                f"pass@{k}": round(sum(1 for r in v if r["correct"] >= 1) / len(v), 4),
                "band": round(sum(1 for r in v if 1 <= r["correct"] <= k - 1) / len(v), 4)}
            for s, v in sorted(by_subject.items())},
        "hist_correct_out_of_k": {str(c): hist.get(c, 0) for c in range(k + 1)},
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
