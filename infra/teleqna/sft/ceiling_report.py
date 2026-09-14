#!/usr/bin/env python3
"""Read the Phase-0.1 ceiling arms and report what is actually spendable.

Three numbers come out of the pair of runs, and only the third one is a target:

  raw ceiling      accuracy with the row's own explanation in context. An upper
                   bound on every knowledge route, teacher or scaffold, but
                   inflated wherever the explanation restates the gold option -
                   no retriever returns the answer key.
  control          accuracy with another row's explanation, same subject. If
                   this rises above base, part of the raw ceiling is the model
                   being made careful by prose rather than being told anything.
  honest ceiling   the raw ceiling read on rows where the explanation does NOT
                   lexically point at the gold option (`margin` below the
                   threshold). This is the stratum a real corpus resembles: the
                   passage a fact came from, not a restatement of the answer.

Also reported: the ceiling on the 160 rows where base got 0/8 sampled rollouts
right. Those are where the whole gain has to come from, and if the ceiling is
low *there* while high overall, the headroom is in rows already nearly solved
and is not worth a corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {r["sample_id"]: bool(r["correct"]) for r in d["results"]}, d["summary"]


def rate(ids, res) -> tuple[float, int]:
    hit = [res[i] for i in ids if i in res]
    return (round(100 * sum(hit) / len(hit), 2) if hit else float("nan"), len(hit))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expl", type=Path, required=True)
    ap.add_argument("--shuf", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--strata", type=Path, required=True)
    ap.add_argument("--always-wrong", type=Path, default=None,
                    help="one sample_id per line: the 0/8 rows from the pass@k profile")
    ap.add_argument("--margin", type=float, default=0.1,
                    help="rows at or below this give-away margin form the honest stratum")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    expl, s_expl = load(args.expl)
    shuf, s_shuf = load(args.shuf)
    base, s_base = load(args.base)
    strata = json.loads(args.strata.read_text())["by_id"]

    ids = [i for i in strata if i in expl and i in shuf and i in base]
    low = [i for i in ids if strata[i]["margin"] <= args.margin]
    high = [i for i in ids if strata[i]["margin"] > args.margin]

    rows = []
    def line(name, subset):
        b, n = rate(subset, base)
        s, _ = rate(subset, shuf)
        e, _ = rate(subset, expl)
        rows.append({"stratum": name, "n": n, "base": b, "shuffled": s,
                     "own_explanation": e, "headroom": round(e - b, 2),
                     "control_drift": round(s - b, 2)})

    line("all", ids)
    line(f"give-away margin <= {args.margin}", low)
    line(f"give-away margin > {args.margin}", high)
    for subj in sorted({strata[i]["subject"] for i in ids}):
        line(f"  subject: {subj}", [i for i in ids if strata[i]["subject"] == subj])
    if args.always_wrong and args.always_wrong.exists():
        aw = [l.strip() for l in args.always_wrong.read_text().split() if l.strip()]
        line("base 0/8 rows (pass@k always-wrong)", [i for i in aw if i in expl])

    w = max(len(r["stratum"]) for r in rows)
    print(f"{'stratum'.ljust(w)}  {'n':>5} {'base':>7} {'shuf':>7} {'expl':>7} "
          f"{'headroom':>9} {'ctrl':>6}")
    for r in rows:
        print(f"{r['stratum'].ljust(w)}  {r['n']:5d} {r['base']:7.2f} "
              f"{r['shuffled']:7.2f} {r['own_explanation']:7.2f} "
              f"{r['headroom']:9.2f} {r['control_drift']:6.2f}")

    if args.out:
        args.out.write_text(json.dumps(
            {"margin_threshold": args.margin,
             "summaries": {"base": s_base, "shuf": s_shuf, "expl": s_expl},
             "table": rows}, indent=1))


if __name__ == "__main__":
    main()
