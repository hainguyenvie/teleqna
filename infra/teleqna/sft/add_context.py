#!/usr/bin/env python3
"""Attach the teacher's context to views that were written for plain SFT.

expand_views.py emits {prompt, completion, sample_id, fact_id, view} - the shape
every earlier trainer on this track wanted, because all of them were
cross-entropy. train_ctxdistill.py needs one more field: the passage the teacher
reads while the student does not. Without it every row is a CE anchor and the
run silently degenerates into the very SFT that was already measured to be worth
+4..6 at best, which is the failure this whole campaign exists to get past. The
trainer refuses to start if no row carries context, so this join is not optional.

The context is the `evidence` span, not the whole retrieved window. The span is
what the static gate proved is verbatim in the source, it is what the answer was
shown to sit inside, and it is short enough that the teacher sequence stays
under MAXLEN once the prompt is added. A full 800-word window would push many
rows past 2048 tokens, and a row whose teacher sequence overflows is dropped
back to a CE anchor - so a longer context would buy nothing and lose rows.

Views whose fact has no evidence keep no context and train as CE anchors, which
is the same treatment make_replay_mix.py's anchors get.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--items", type=Path, action="append", required=True,
                    help="the item jsonl the views were expanded from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key", default="item_id")
    args = ap.parse_args()

    ev = {}
    for p in args.items:
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            k = str(r.get(args.key) or r.get("id") or "")
            if k and r.get("evidence"):
                ev[k] = r["evidence"]
    print(f"evidence spans indexed: {len(ev)}", flush=True)

    n, hit, by_view = 0, 0, collections.Counter()
    with args.out.open("w", encoding="utf-8") as fh:
        for line in args.views.open(encoding="utf-8"):
            r = json.loads(line)
            n += 1
            c = ev.get(str(r.get("fact_id")))
            if c:
                r["context"] = c
                hit += 1
                by_view[r.get("view", "?")] += 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    frac = hit / max(n, 1)
    print(json.dumps({"views": n, "with_context": hit,
                      "frac": round(frac, 4),
                      "by_view": dict(by_view.most_common())}, indent=1))
    # A near-total miss means the join key is wrong, not that the data is thin.
    # Failing loudly here costs a second; failing quietly costs the whole run,
    # because the trainer would still start on whatever fraction did join.
    if frac < 0.5:
        raise SystemExit(f"only {frac:.1%} of views joined to an evidence span; "
                         f"the --key is probably wrong")


if __name__ == "__main__":
    main()
