#!/usr/bin/env python3
"""Merge the two filtered pools into one item file with a stable id.

Two independent filters ran over the same generated items and they keep
different things, so the union has to be built deliberately rather than
concatenated:

  novel      passed the closed-book probe: grounded, not already known to the
             student. This is the knowledge branch - the 858 rows the teacher
             fixes by holding the passage.
  mined      passed the two behavioural passes: a distractor the student picks
             closed-book and abandons once it sees the evidence. This is the
             discrimination branch - the 63% of the residual that no extra
             corpus reaches.

An item can be in both, and when it is, the mined distractors are the better
ones: they were validated against the student's own confusion, while the
generated set's came from a 122B whose sense of confusable was measured to be
weaker by 8.86 points. So mined wins the merge on the distractor field only -
question, answer and evidence come from the item that passed the novelty probe,
which is the one whose grounding was checked.

item_id is content-addressed rather than positional. expand_views.py falls back
to a running counter when an item has no id, and a counter would renumber on any
change to the input order, so the views could not be joined back to their
evidence afterwards.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def iid(r: dict) -> str:
    h = hashlib.sha256()
    h.update((str(r.get("chunk_id", "")) + "|" + (r.get("question") or "")
              + "|" + (r.get("answer") or "")).encode("utf-8"))
    return h.hexdigest()[:16]


def load(paths: list[Path]) -> dict:
    out = {}
    for p in paths:
        if not p.exists():
            print(f"  MISSING {p}", flush=True)
            continue
        n = 0
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if not (r.get("question") and r.get("answer") and r.get("evidence")):
                continue
            out[iid(r)] = r
            n += 1
        print(f"  {p.name}: {n}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel", type=Path, action="append", default=[])
    ap.add_argument("--mined", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    print("novel:", flush=True)
    novel = load(args.novel)
    print("mined:", flush=True)
    mined = load(args.mined)

    stats = collections.Counter()
    merged = {}
    for k, r in novel.items():
        r = dict(r)
        r["item_id"] = k
        if k in mined and mined[k].get("distractors"):
            r["distractors"] = mined[k]["distractors"]
            r["why_wrong"] = r.get("why_wrong") or []
            r["neg_origin"] = "mined_31b"
            stats["novel_with_mined_negatives"] += 1
        else:
            r["neg_origin"] = "generated_122b"
            stats["novel_only"] += 1
        merged[k] = r
    for k, r in mined.items():
        if k in merged:
            continue
        r = dict(r)
        r["item_id"] = k
        r["neg_origin"] = "mined_31b"
        # Not novelty-checked: the student may already know this fact. Kept
        # anyway because its value is the distractor, not the fact, and the
        # discrimination branch is the one with no other source of data.
        r["novelty_checked"] = False
        merged[k] = r
        stats["mined_only"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {"items_out": len(merged), "novel_in": len(novel),
              "mined_in": len(mined), "overlap": len(set(novel) & set(mined)),
              "breakdown": dict(stats.most_common())}
    args.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
