#!/usr/bin/env python3
"""One table over every leakage arm: context depth and distilled answer key.

Reads a directory of `eval_dev_vllm.py` outputs and prints, per arm, the
overall score, the per-subject breakdown, and the score restricted to the rows
the base model gets wrong. That last column is the one that matters: an arm can
gain three points overall while fixing nothing the model could not already do,
and only the wrong-row slice tells the two apart.

Arms are compared against whichever arm is named by --ref, never across serving
stacks. The haystack sweep runs on a vLLM nightly and re-measures its own base
for exactly that reason.

**Everything this script reports is leakage.** Its output is a discount table,
not a results table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

SUBJECTS = ["Lexicon", "Research overview", "Research publications",
            "Standards overview", "Standards specifications"]


def load(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    return {r["sample_id"]: bool(r["correct"]) for r in d["results"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--order", default="",
                    help="comma-separated arm names, in the order to print; "
                         "anything not listed is appended alphabetically")
    ap.add_argument("--ref", default="base", help="arm to take deltas against")
    ap.add_argument("--split", type=Path, required=True,
                    help="the eval split, for subject labels")
    ap.add_argument("--wrong-ids", type=Path, default=None,
                    help="one sample_id per line: rows the base model misses")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.split.open(encoding="utf-8")]
    subject = {r["sample_id"]: r["subject"] for r in rows}

    arms: dict[str, dict] = {}
    for p in sorted(glob.glob(str(args.dir / "*_base_think.json"))):
        name = os.path.basename(p)[: -len("_base_think.json")]
        arms[name] = load(p)
    # Adapter arms land as <tag>_<armname>_think.json
    for p in sorted(glob.glob(str(args.dir / "*_think.json"))):
        base = os.path.basename(p)[: -len("_think.json")]
        if base.endswith("_base"):
            continue
        arms[base] = load(p)
    if not arms:
        raise SystemExit(f"no result files in {args.dir}")

    wrong = None
    if args.wrong_ids and args.wrong_ids.exists():
        wrong = {w for w in args.wrong_ids.read_text().split() if w}

    order = [a for a in args.order.split(",") if a in arms]
    order += sorted(a for a in arms if a not in order)

    def pct(ids, res):
        hit = [res[i] for i in ids if i in res]
        return 100 * sum(hit) / len(hit) if hit else float("nan")

    all_ids = list(subject)
    ref = arms.get(args.ref)
    header = f"{'arm':<12}{'overall':>9}{'Δ':>7}"
    for s in SUBJECTS:
        header += f"{s.split()[-1][:6]:>8}"
    if wrong:
        header += f"{'on-wrong':>10}"
    print(header)
    print("-" * len(header))
    table = []
    for name in order:
        res = arms[name]
        o = pct(all_ids, res)
        d = o - pct(all_ids, ref) if ref else float("nan")
        line = f"{name:<12}{o:9.2f}{d:+7.2f}"
        rec = {"arm": name, "overall": round(o, 2), "delta": round(d, 2),
               "by_subject": {}}
        for s in SUBJECTS:
            v = pct([i for i in all_ids if subject[i] == s], res)
            line += f"{v:8.1f}"
            rec["by_subject"][s] = round(v, 2)
        if wrong:
            w = pct(list(wrong), res)
            line += f"{w:10.1f}"
            rec["on_wrong_rows"] = round(w, 2)
        print(line)
        table.append(rec)
    if args.out:
        args.out.write_text(json.dumps(
            {"ref": args.ref, "note": "answer-key leakage arms; not submittable",
             "table": table}, indent=1))


if __name__ == "__main__":
    main()
