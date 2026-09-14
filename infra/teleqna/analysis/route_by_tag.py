#!/usr/bin/env python3
"""Price routing on a signal the model can actually see: the question's own tags.

route_by_subject.py routes on the `subject` field, which the Inspect harness
keeps in metadata and never shows the model — so any gain there is capped by a
classifier nobody has built yet. The tags are different: `[3GPP Release 17]`,
`[IEEE 802.11]`, `[TCP/IP]` are inside the question string the model is handed,
so a router keyed on them is deterministic and free, with no classification
error to discount.

Reported per bucket, per arm, with the same max-over-arms bias correction as the
subject table. Buckets are coarse on purpose: a router that splits the benchmark
five ways on 1,000 dev rows is choosing among 200-row estimates, and the whole
lesson of the subject table is that that is not enough resolution to pick a
winner.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

TAGS = [
    ("3gpp", re.compile(r"(?i)\[3gpp\b|\brelease\s+\d{1,2}\b")),
    ("ieee", re.compile(r"(?i)\[ieee\b|\b802\.\d")),
    ("tcpip", re.compile(r"(?i)\[tcp/ip\]|\brfc\s*\d+")),
]


def bucket(q: str) -> str:
    for name, rx in TAGS:
        if rx.search(q):
            return name
    return "untagged"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--include", default="")
    args = ap.parse_args()

    buckets = {}
    for line in open(args.dev, encoding="utf-8"):
        r = json.loads(line)
        buckets[r["sample_id"]] = bucket(r["question"])
    order = ["3gpp", "ieee", "tcpip", "untagged"]
    sizes = {b: sum(1 for v in buckets.values() if v == b) for b in order}
    print("dev-1000 buckets: " + "  ".join(f"{b}={sizes[b]}" for b in order))

    arms = {}
    for d in args.dir:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.basename(f)[:-5]
            if args.include and not any(w and w in name
                                        for w in args.include.split(",")):
                continue
            try:
                j = json.load(open(f))
            except Exception:
                continue
            if not isinstance(j, dict) or j.get("summary", {}).get("total") != 1000:
                continue
            hit = {b: [0, 0] for b in order}
            for row in j.get("results", []):
                b = buckets.get(row.get("sample_id"))
                if b is None:
                    continue
                hit[b][1] += 1
                hit[b][0] += bool(row.get("correct"))
            arms[name] = {"overall": j["summary"]["accuracy"], "hit": hit}
    if not arms:
        raise SystemExit("no arms matched")

    names = sorted(arms, key=lambda k: -arms[k]["overall"])
    width = max(len(x) for x in names) + 2
    print(f"\n{'arm':<{width}}{'overall':>9}" +
          "".join(f"{b:>12s}" for b in order))
    for k in names:
        row = ""
        for b in order:
            c, n = arms[k]["hit"][b]
            row += f"{(c/n*100 if n else 0):>12.2f}"
        print(f"{k:<{width}}{arms[k]['overall']*100:>9.2f}{row}")

    total = sum(sizes.values())
    routed, bias_tot = 0.0, 0.0
    print("\nper-bucket winners (selected on this table — biased):")
    for b in order:
        if not sizes[b]:
            continue
        win = max(arms, key=lambda a: (arms[a]["hit"][b][0] /
                                       max(arms[a]["hit"][b][1], 1)))
        c, n = arms[win]["hit"][b]
        acc = c / max(n, 1)
        routed += acc * sizes[b]
        se = math.sqrt(max(acc * (1 - acc), 1e-9) / max(n, 1))
        bias_tot += se * math.sqrt(2 * math.log(max(len(arms), 2))) * sizes[b]
        print(f"  {b:<10s} n={n:<5d} {acc*100:6.2f}  <- {win}")
    routed /= total
    bias = bias_tot / total
    best = names[0]
    print(f"\noracle tag routing        {routed*100:6.2f}")
    print(f"selection bias estimate   {bias*100:6.2f}")
    print(f"bias-corrected            {(routed - bias)*100:6.2f}")
    print(f"vs best single ({best})   "
          f"{(routed - bias - arms[best]['overall'])*100:+6.2f}")


if __name__ == "__main__":
    main()
