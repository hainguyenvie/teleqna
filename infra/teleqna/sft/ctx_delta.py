#!/usr/bin/env python3
"""Split the context gain by whether the context actually held the answer.

A single delta - 80.90 to 86.90 - hides two opposite effects that matter
separately for the training track. Where retrieval succeeded, the passage is
in front of the model and the gain measures how much a *correct* window is
worth: that is the quantity the student must inherit. Where retrieval failed,
the model is reading irrelevant text, and any loss there is crowding, an
artefact of serving that a trained student will never pay.

Only the first number is the target. Reporting the net figure as "what context
buys" would understate the distillation ceiling by however much crowding cost,
and crowding is precisely the thing the no-RAG route deletes.

`fixed` and `broken` are printed because the net can be small while the churn
is large, and a training set built from windows should be aimed at the rows
that context fixes, not at the rows that happen to net out.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
STOP = {
    "the", "and", "for", "that", "with", "this", "are", "was", "which", "from",
    "has", "have", "not", "can", "may", "shall", "will", "its", "their", "than",
    "when", "what", "which", "where", "how", "why", "who", "does", "did", "any",
    "all", "one", "two", "following", "above", "below", "used", "use", "using",
    "purpose", "main", "key", "type", "types", "based", "into", "such", "other",
}
STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
RULE = "\n\n---\n\n"


def terms(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text.lower()) if t not in STOP}


def official(completion: str) -> str:
    m = STRICT.findall(completion or "") or LOOSE.findall(completion or "")
    return m[-1].strip().rstrip(".").upper() if m else ""


def load(path: Path) -> dict[str, str]:
    j = json.load(path.open(encoding="utf-8"))
    res = j["results"] if isinstance(j, dict) else j
    return {r["sample_id"]: r.get("completion", "") for r in res}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-data", type=Path, required=True,
                    help="the jsonl whose question field carries the windows")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--ctx", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gold, nch, sub = {}, {}, {}
    for line in args.gold.open(encoding="utf-8"):
        r = json.loads(line)
        gold[r["sample_id"]] = chr(65 + int(r["answer"]))
        nch[r["sample_id"]] = len(r["choices"])
        sub[r["sample_id"]] = r.get("subject", "?")

    base, ctx = load(args.base), load(args.ctx)

    cells = collections.defaultdict(lambda: collections.Counter())
    skipped = 0
    for line in args.rag_data.open(encoding="utf-8"):
        row = json.loads(line)
        sid = row["sample_id"]
        if sid not in base or sid not in ctx or sid not in gold:
            skipped += 1
            continue
        gt = terms(row["choices"][int(row["answer"])])
        if not gt:
            skipped += 1     # "All of the above" and friends: metric undefined
            continue
        block = row["question"].split(RULE)[0]
        hit = len(gt & terms(block)) >= 0.8 * len(gt)

        valid = {chr(65 + i) for i in range(nch[sid])}
        b = official(base[sid]) in valid and official(base[sid]) == gold[sid]
        c = official(ctx[sid]) in valid and official(ctx[sid]) == gold[sid]

        k = "answer_in_ctx" if hit else "no_answer_in_ctx"
        cells[k]["n"] += 1
        cells[k]["base"] += b
        cells[k]["ctx"] += c
        cells[k]["fixed"] += (c and not b)
        cells[k]["broken"] += (b and not c)

    report = {"rag_data": str(args.rag_data), "base": str(args.base),
              "ctx": str(args.ctx), "skipped_metric_undefined": skipped,
              "cells": {}}
    tot = collections.Counter()
    for k, v in cells.items():
        tot.update(v)
        report["cells"][k] = {
            "n": v["n"],
            "base_acc": round(100 * v["base"] / v["n"], 2),
            "ctx_acc": round(100 * v["ctx"] / v["n"], 2),
            "delta": round(100 * (v["ctx"] - v["base"]) / v["n"], 2),
            "fixed": v["fixed"], "broken": v["broken"]}
    report["overall"] = {
        "n": tot["n"],
        "base_acc": round(100 * tot["base"] / tot["n"], 2),
        "ctx_acc": round(100 * tot["ctx"] / tot["n"], 2),
        "delta": round(100 * (tot["ctx"] - tot["base"]) / tot["n"], 2),
        "fixed": tot["fixed"], "broken": tot["broken"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
