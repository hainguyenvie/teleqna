#!/usr/bin/env python3
"""Build M1: answer-key supervision with a self-replay anchor.

The target is the ot-full number, and training on ot-full is permitted, so the
shortest path is direct supervision from `answerkey_mcq_all.jsonl` -- the
benchmark's own explanation followed by its own gold letter, in the harness
prompt byte-for-byte, 10,000/10,000 labels verified correct. The earlier
answer-key campaign failed with this same data at a 1:1 ratio against
`anchor_mcq.jsonl`; today's controlled contrast says that anchor was the fault,
not the ratio, so the anchor is replaced rather than the recipe.

Two deliberate holes in the coverage.

  clean holdout   500 questions, stratified by subject, are excluded from every
                  row this file emits -- answer key and anchor alike. They cost
                  about half a point of the headline score and they are the only
                  way to keep an honest read of what generalises, which is what
                  a private test will actually measure.

  anchor overlap  the anchor keeps only questions outside the holdout that the
                  base already answers correctly. Those rows carry the base's
                  own wording, so voice and format stay anchored to what the
                  harness saw at score time while the answer-key rows move the
                  content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import cycle, islice
from pathlib import Path


def hsh(salt: str, s: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}|{s}".encode()).digest()[:8], "big")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerkey", type=Path, required=True)
    ap.add_argument("--selfreplay", type=Path, required=True)
    ap.add_argument("--otfull", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holdout-out", type=Path, required=True)
    ap.add_argument("--holdout", type=int, default=500)
    args = ap.parse_args()

    meta = {}
    for line in args.otfull.open(encoding="utf-8"):
        r = json.loads(line)
        meta[r["sample_id"]] = r.get("subject", "?")

    # stratified clean holdout: proportional per subject, deterministic
    bysub = defaultdict(list)
    for q, s in meta.items():
        bysub[s].append(q)
    hold = set()
    for s, qs in sorted(bysub.items()):
        qs = sorted(qs, key=lambda q: hsh("holdout-v1", q))
        hold.update(qs[:max(1, round(args.holdout * len(qs) / len(meta)))])
    print(f"clean holdout {len(hold)} questions over {len(bysub)} subjects")
    args.holdout_out.write_text(json.dumps(sorted(hold), indent=1))

    know = []
    for line in args.answerkey.open(encoding="utf-8"):
        r = json.loads(line)
        q = r["sample_id"].split("::")[0]
        if q in hold:
            continue
        r["role"] = "knowledge"
        r["fact_id"] = q
        know.append(r)
    print(f"answer-key rows: {len(know)}")

    anchor_pool = []
    for line in args.selfreplay.open(encoding="utf-8"):
        r = json.loads(line)
        q = r["sample_id"].split("::")[0]
        if q in hold:
            continue
        r = dict(r)
        r["role"] = "anchor"
        r["fact_id"] = q
        r.pop("context", None)
        anchor_pool.append(r)
    print(f"anchor pool: {len(anchor_pool)} (base's own correct answers)")

    # one anchor row per knowledge row, so neither side dominates the gradient
    anchor = []
    for i, r in enumerate(islice(cycle(anchor_pool), len(know))):
        r = dict(r)
        r["sample_id"] = f"{r['sample_id']}#{i // len(anchor_pool)}"
        anchor.append(r)

    rows = know + anchor
    rows.sort(key=lambda r: hsh("m1-shuffle", r["sample_id"]))
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"M1: {len(rows)} rows ({len(know)} knowledge + {len(anchor)} anchor, "
          f"{len(anchor)/len(anchor_pool):.1f}x cycling) -> {args.out}")


if __name__ == "__main__":
    main()
