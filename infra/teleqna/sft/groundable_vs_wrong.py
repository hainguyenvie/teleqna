#!/usr/bin/env python3
"""Cross groundability against the model's actual errors.

67.03% of rows have a window that alone carries the answer, so a grounded fact
can be built for two thirds of the benchmark. That number is worthless on its
own: the 122B already answers 80.90% correctly, and a training set aimed at
rows it never gets wrong buys nothing but the risk of disturbing them.

The cell that decides whether the no-RAG track can work at all is
`wrong AND groundable` - rows the model fails today and for which the corpus
holds a quotable span. That is the addressable set, and its size is the honest
upper bound on what any amount of synthetic data and any training objective
can recover from this corpus.

Two cells are worth naming as well:

  wrong AND NOT groundable   nothing in this corpus can fix these. Either the
                             fact is absent, or it is split across windows and
                             no single span states it. More generation cannot
                             reach them; a different corpus might.
  right AND groundable       the regression surface. Format anchors and replay
                             exist for exactly these rows, since the measured
                             failure mode of this repo is a fine-tune that
                             learns a fact and loses the answer contract.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")


def official(completion: str) -> str:
    m = STRICT.findall(completion or "") or LOOSE.findall(completion or "")
    return m[-1].strip().rstrip(".").upper() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True,
                    help="single_window_recall.py .rows.jsonl")
    ap.add_argument("--eval", type=Path, required=True,
                    help="eval_dev_vllm.py result json holding the BASE arm")
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--field", default="base_completion",
                    help="which stored completion is the no-context arm")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ground = {}
    for line in args.rows.open(encoding="utf-8"):
        r = json.loads(line)
        ground[r["sample_id"]] = r

    gold, nch = {}, {}
    for line in args.gold.open(encoding="utf-8"):
        r = json.loads(line)
        gold[r["sample_id"]] = chr(65 + int(r["answer"]))
        nch[r["sample_id"]] = len(r["choices"])

    j = json.load(args.eval.open(encoding="utf-8"))
    res = j["results"] if isinstance(j, dict) else j
    have = sorted({k for r in res[:1] for k in r})
    print("stored fields:", have, flush=True)

    cells = collections.Counter()
    by_sub = collections.defaultdict(collections.Counter)
    depth_of_addressable = collections.Counter()
    addressable = []
    missing = 0

    for r in res:
        sid = r.get("sample_id")
        g = gold.get(sid)
        gr = ground.get(sid)
        if g is None or gr is None:
            missing += 1
            continue
        comp = r.get(args.field)
        if comp is None:
            missing += 1
            continue
        valid = {chr(65 + i) for i in range(nch[sid])}
        pred = official(comp)
        ok = (pred in valid) and pred == g
        key = ("right" if ok else "wrong",
               "groundable" if gr["single"] else "not-groundable")
        cells[key] += 1
        by_sub[gr["subject"]][key] += 1
        if not ok and gr["single"]:
            depth_of_addressable[gr["best_window"]] += 1
            addressable.append(sid)

    n = sum(cells.values())
    cum, depth = 0, {}
    for d in sorted(depth_of_addressable):
        cum += depth_of_addressable[d]
        depth[d + 1] = round(cum / max(len(addressable), 1), 4)

    report = {
        "eval": str(args.eval), "field": args.field, "n_scored": n,
        "unmatched": missing,
        "cells": {f"{a}/{b}": v for (a, b), v in sorted(cells.items())},
        "base_accuracy": round(
            sum(v for (a, _), v in cells.items() if a == "right") / n, 4),
        "addressable": len(addressable),
        "addressable_frac_of_all": round(len(addressable) / n, 4),
        "addressable_frac_of_wrong": round(
            len(addressable) / max(sum(v for (a, _), v in cells.items()
                                       if a == "wrong"), 1), 4),
        "unreachable_from_this_corpus": sum(
            v for (a, b), v in cells.items()
            if a == "wrong" and b == "not-groundable"),
        "regression_surface": sum(
            v for (a, b), v in cells.items()
            if a == "right" and b == "groundable"),
        "addressable_by_depth": {k: depth[k] for k in sorted(depth)
                                 if k in (1, 2, 4, 8, 16, 24, 32)},
        "by_subject": {s: {f"{a}/{b}": v for (a, b), v in sorted(c.items())}
                       for s, c in sorted(by_sub.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    args.out.with_suffix(".ids.txt").write_text("\n".join(addressable) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "by_subject"},
                     indent=1))


if __name__ == "__main__":
    main()
