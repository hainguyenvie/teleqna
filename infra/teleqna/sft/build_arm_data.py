#!/usr/bin/env python3
"""Build the three retention arms.

What the audit found. The first run spent its 100k-row budget on 6.76 views of
12,996 facts and anchored them with `anchor_mcq.jsonl`, a file that is 94% the
same generated schema as the treatment. Measured against base on dev1000 it
gained +14.6pp on the 178 questions its facts were aimed at and lost 11.7pp on
the 822 it was not, for -7.0 overall. So the data works; what failed is
retention, and the anchor that was supposed to provide it was a sample of the
treatment rather than of the behaviour being preserved.

Two changes here, and one of them is the variable under test.

  breadth   one mcq view per fact instead of 6.76, so the same row budget buys
            40,000 facts instead of 12,996. The three mcq views of a fact are
            the same question with the options rotated, so the redundancy was
            buying letter-position invariance -- worth something, but not 5x
            the fact coverage. Every arm gets this; it is not a variable.

  anchor    S1 and S3 anchor on the base model's own correct answers, S2 on
            the old `anchor_mcq.jsonl`. Both pools are cut to the SAME number
            of unique rows and cycled to the SAME length, so S1 - S2 is the
            source of the anchor and nothing else: not its size, not how often
            a row repeats, not the ratio, not the learning rate.

  lr        S1 - S3 is the learning rate alone, 1e-4 against 5e-5.

The anchors are drawn from heldout9000, never from dev1000, so a recovery on
dev1000 has to come from the anchor generalising rather than from the model
having been shown those answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import cycle, islice
from pathlib import Path


def rd(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def hsh(salt: str, s: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}|{s}".encode()).digest()[:8],
                          "big")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--selfreplay", type=Path, required=True)
    ap.add_argument("--oldanchor", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--facts", type=int, default=40000)
    ap.add_argument("--anchor-rows", type=int, default=40000)
    args = ap.parse_args()

    # one view per fact: mcq0, the shape the benchmark actually asks for
    know = []
    for line in args.views.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("view") != "mcq0":
            continue
        r["role"] = "knowledge"
        know.append(r)
    know.sort(key=lambda r: hsh("facts-v3", r["fact_id"]))
    know = know[:args.facts]
    n_facts = len({r["fact_id"] for r in know})
    print(f"knowledge: {len(know)} rows over {n_facts} facts")

    self_pool = rd(args.selfreplay)
    old_pool = rd(args.oldanchor)
    n_uniq = min(len(self_pool), len(old_pool))
    # Same unique count and the same cycle length for both, so the contrast is
    # the source of the rows and not how thin the pool was spread.
    self_pool.sort(key=lambda r: hsh("anchor-v3", str(r.get("sample_id"))))
    old_pool.sort(key=lambda r: hsh("anchor-v3", str(r.get("sample_id"))))
    self_pool, old_pool = self_pool[:n_uniq], old_pool[:n_uniq]
    print(f"anchor pools cut to {n_uniq} unique rows each, cycled to "
          f"{args.anchor_rows} ({args.anchor_rows / n_uniq:.1f}x)")

    def cycled(pool):
        out = []
        for i, r in enumerate(islice(cycle(pool), args.anchor_rows)):
            r = dict(r)
            r["role"] = "anchor"
            r["sample_id"] = f"{r.get('sample_id', 'anchor')}#{i // len(pool)}"
            r.pop("context", None)
            out.append(r)
        return out

    # S3 differs from S1 only in the learning rate, so it reads S1's file.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, pool in (("S1", self_pool), ("S2", old_pool)):
        rows = know + cycled(pool)
        out = args.out_dir / f"arm_{name}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
