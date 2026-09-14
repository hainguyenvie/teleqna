#!/usr/bin/env python3
"""Mix knowledge rows with format-anchor rows, so learning a fact does not cost the answer.

This exists because of a measured failure, not a precaution. Exp B trained on
the benchmark's own explanations - the perfect corpus, one clean sentence per
fact - and the result was:

    base        75.10   unparsed  10
    fact_all    66.30   unparsed  96      <- the model stopped emitting ANSWER: X
    fact_held   68.40   unparsed  36

Nine percent of rows scored zero not because the model was wrong but because it
had been taught to answer in prose. On the 865 rows where every arm still
produced a parsable answer the knowledge signal was +2.89 (p=0.089), against
+22.2 for the same facts delivered in context. The format damage was three
times the size of the thing being measured.

The fix is the oldest one in the continual-learning literature and the one the
prompt-distillation work reports as decisive: train the new thing alongside the
old behaviour rather than instead of it. Here the anchor is harness-format MCQ
rows built from the Tele-Data grounded set - real corpus material, not test
data - so every optimiser step that pulls toward free prose is balanced by one
that pulls toward `ANSWER: X`.

Anchors are shared between arms by construction. Both the treatment and the
control get the *same* anchor rows, so the only thing that differs between them
is still the 1,000 facts belonging to the split being scored. Anything else
would confound the comparison the whole experiment exists to make.

Interleaving is deterministic and stratified: a shuffle that happened to put
all the anchors in the last epoch would leave the format damage in place for
most of training and then paper over it at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knowledge", type=Path, required=True,
                    help="the rows carrying the facts being injected")
    ap.add_argument("--anchor", type=Path, required=True,
                    help="harness-format rows that preserve the answer contract")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="anchor rows per knowledge row")
    ap.add_argument("--anchor-seed", default="replay-v2",
                    help="fixes which anchors are drawn; keep identical across "
                         "arms or the arms stop being comparable")
    args = ap.parse_args()

    know = read(args.knowledge)
    anchor_pool = read(args.anchor)
    want = int(len(know) * args.ratio)
    if want > len(anchor_pool):
        raise SystemExit(f"need {want} anchors, pool has {len(anchor_pool)}")
    # Deterministic draw by hash of a stable key, not by position: the anchor
    # file is written in corpus order, and a prefix would be one spec series.
    anchor_pool.sort(key=lambda r: hashlib.sha256(
        (args.anchor_seed + "|" + str(r.get("sample_id"))).encode()).hexdigest())
    anchors = anchor_pool[:want]

    rows = []
    for r in know:
        rows.append({**r, "role": "knowledge"})
    for r in anchors:
        rows.append({**r, "role": "anchor"})
    # Stratified interleave: sort by a hash that ignores role, so the two kinds
    # are evenly spread through every epoch.
    rows.sort(key=lambda r: hashlib.sha256(
        ("mix|" + str(r["sample_id"])).encode()).hexdigest())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_know = sum(1 for r in rows if r["role"] == "knowledge")
    report = {"knowledge_rows": n_know, "anchor_rows": len(rows) - n_know,
              "total": len(rows), "ratio": args.ratio,
              "anchor_source": str(args.anchor),
              "note": "knowledge rows are answer-key derived; not submittable"}
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
