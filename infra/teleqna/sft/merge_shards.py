#!/usr/bin/env python3
"""Glue eval_dev.py shards back into one dev-1000 result.

Sharding only splits the row list; each shard keeps the unsharded run's batch
boundaries (eval_dev.py refuses an --offset that is not a multiple of --batch),
so concatenating them is the same evaluation, not an approximation. This checks
that: the shards must tile the row range with no gap and no overlap, or it
refuses rather than reporting an accuracy over a hole.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect", type=int, default=1000,
                     help="rows the shards must add up to")
    args = ap.parse_args()

    parts = []
    for p in args.shards:
        d = json.loads(p.read_text())
        parts.append((d["summary"]["offset"], d))
    parts.sort()

    at, results = 0, []
    for off, d in parts:
        if off != at:
            raise SystemExit(f"shards do not tile: expected offset {at}, got {off}")
        results += d["results"]
        at += d["summary"]["total"]
    if at != args.expect:
        raise SystemExit(f"shards cover {at} rows, expected {args.expect}")

    head = parts[0][1]["summary"]
    correct = sum(r["correct"] for r in results)
    # per-shard by_subject carries n and acc; n*acc is an integer count, so the
    # subjects recombine exactly without needing the rows to carry a label.
    by_subject: dict[str, list[int]] = {}
    for off, d in parts:
        for s, v in d["summary"].get("by_subject", {}).items():
            tot = by_subject.setdefault(s, [0, 0])
            tot[0] += v["n"]
            tot[1] += round(v["n"] * v["acc"])

    summary = {
        "adapter": head["adapter"], "lora_scale": head["lora_scale"],
        "sharded_from": [str(p) for p in args.shards],
        "total": len(results), "correct": correct,
        "accuracy": round(correct / len(results), 4),
    }
    if by_subject:
        summary["by_subject"] = {
            s: {"n": n, "acc": round(c / n, 4)}
            for s, (n, c) in sorted(by_subject.items())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results},
                                    ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
