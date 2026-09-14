#!/usr/bin/env python3
"""Keep the GRPO prompts whose sampled pass-rate lies in the learnable band.

Default band 1..k-1 keeps every prompt with any within-group disagreement.
Tighten with --lo/--hi when the profile shows a big lucky-guess mass at the
bottom (with 4 choices, a pure guesser averages k/4 correct, so a prompt
sitting at exactly that level is more likely luck than partial competence)."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lo", type=int, default=1, help="min correct, inclusive")
    ap.add_argument("--hi", type=int, default=None,
                    help="max correct, inclusive (default k-1)")
    ap.add_argument("--max-unparsed", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.profile.open(encoding="utf-8")]
    k = rows[0]["k"]
    hi = args.hi if args.hi is not None else k - 1
    keep, drops = [], collections.Counter()
    for r in rows:
        if r["correct"] < args.lo:
            drops["too_hard"] += 1
        elif r["correct"] > hi:
            drops["too_easy"] += 1
        elif r["unparsed"] > args.max_unparsed:
            drops["unparsed"] += 1
        else:
            keep.append(r)
    keep.sort(key=lambda r: abs(r["correct"] - k / 2))   # most informative first
    if args.limit:
        keep = keep[:args.limit]

    with args.out.open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps({kk: r[kk] for kk in
                                ("prompt", "answer", "n_choices", "src", "band")},
                               ensure_ascii=False) + "\n")
    print(json.dumps({
        "kept": len(keep), "band": f"{args.lo}..{hi} of {k}",
        "drops": dict(drops),
        "by_correct": dict(sorted(collections.Counter(r["correct"] for r in keep).items())),
        "by_src": dict(collections.Counter(r["src"] for r in keep)),
        "by_orig_band": dict(collections.Counter(r["band"] for r in keep)),
    }, indent=2))


if __name__ == "__main__":
    main()
