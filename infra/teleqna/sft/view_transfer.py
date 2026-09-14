#!/usr/bin/env python3
"""Does a trained fact survive being asked a different way?

S1's residual is 1,646 errors, of which 773 are fixed by putting eight retrieved
chunks in front of the model. The corpus holds those answers and the retriever
finds them, yet a run that trained on facts mined from that same corpus does not
know them. Coverage cannot explain it: 90.40% of the residual was aimed at.

The cheapest explanation left is that the training installed the row rather than
the fact. Each fact was trained as exactly one `mcq0` row, and `expand_views.py`
also emits the same fact as `mcq1`/`mcq2` (identical question, options rotated)
and as `qa`, `statement`, `reverse` and `cloze` (different framings entirely).
So the probe writes itself:

  seen-form      the mcq0 row the model actually trained on
  rotated        same question, options permuted -- tests letter binding
  reframed       qa / statement / reverse -- tests whether the fact generalises

against untrained facts as the control. If S1 beats base on seen-form, ties on
rotated and ties on reframed, then one epoch bought surface form and nothing
transferable, and no quantity of further facts in this shape will convert. If it
holds up on reframed, the facts did install and the failure is elsewhere.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

REFRAMED = {"qa", "statement", "reverse"}
ROTATED = {"mcq1", "mcq2"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--trained", type=Path, required=True,
                    help="the arm's training file, to read which facts it saw")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-cell", type=int, default=500)
    args = ap.parse_args()

    seen = set()
    for line in args.trained.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("role") != "anchor" and r.get("fact_id"):
            seen.add(r["fact_id"])
    print(f"facts the arm trained on: {len(seen)}")

    byfact: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in args.views.open(encoding="utf-8"):
        r = json.loads(line)
        v = r.get("view")
        if v == "mcq0" or v in ROTATED or v in REFRAMED:
            byfact[r["fact_id"]][v] = r
    print(f"facts with views on disk: {len(byfact)}")

    trained = sorted(f for f in byfact if f in seen)
    control = sorted(f for f in byfact if f not in seen)
    rng = random.Random(20260812)
    rng.shuffle(trained)
    rng.shuffle(control)
    print(f"trained {len(trained)}  control {len(control)}")

    rows = []
    for pool, tag in ((trained, "trained"), (control, "control")):
        for cell, views in (("seen", ["mcq0"]), ("rotated", sorted(ROTATED)),
                            ("reframed", sorted(REFRAMED))):
            n = 0
            for fid in pool:
                if n >= args.per_cell:
                    break
                for v in views:
                    r = byfact[fid].get(v)
                    if r:
                        rows.append({"sample_id": f"{fid}::{v}::{tag}",
                                     "prompt": r["prompt"],
                                     "completion": r["completion"],
                                     "cell": cell, "pool": tag, "view": v,
                                     "fact_id": fid})
                        n += 1
                        break
            print(f"  {tag:8s} {cell:9s} {n}")

    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
