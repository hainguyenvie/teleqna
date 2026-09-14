#!/usr/bin/env python3
"""Carve a train / val / held-out split for prompt optimisation.

teleqna has no training split — the 10,000 rows *are* the test set. Any
optimisation signal therefore has to be bought by burning rows: whatever the
optimiser sees is no longer clean for reporting. There is no way around that
short of generating a synthetic dev set from the source specifications, which
is the right long-term answer and needs the corpus first.

So the deal made here, explicitly:

  train    400 rows   the optimiser's search signal
  val      600 rows   candidate selection, disjoint from train
  heldout  9,000 rows never seen by any optimiser — the only number worth quoting

1,000 rows are spent. The held-out 9,000 still gives a standard error of about
0.46 points, which resolves anything worth acting on. Every arm of this track
that reports an optimised number reports it on the held-out split, and says so.

Splits are stratified by subject and deterministic given --seed, so the same
rows are burned every time and the accounting stays honest across runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def rng_for(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--val", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row.get("subject") or "<none>"].append(row)

    total = len(rows)
    train: list[dict] = []
    val: list[dict] = []
    for subject in sorted(by_subject):
        pool = by_subject[subject]
        shuffled = pool[:]
        rng_for("split", subject, str(args.seed)).shuffle(shuffled)
        n_train = max(1, round(args.train * len(pool) / total))
        n_val = max(1, round(args.val * len(pool) / total))
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])

    burned = {r["sample_id"] for r in train} | {r["sample_id"] for r in val}
    heldout = [r for r in rows if r["sample_id"] not in burned]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("heldout", heldout)):
        part.sort(key=lambda r: r["sample_index"])
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in part:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Disjointness is the entire point of this file; assert it rather than trust it.
    ids = [{r["sample_id"] for r in part} for part in (train, val, heldout)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "splits overlap"
    assert sum(len(p) for p in (train, val, heldout)) == total, "rows lost"

    print(json.dumps({
        "seed": args.seed,
        "train": len(train), "val": len(val), "heldout": len(heldout),
        "burned_rows": len(burned),
        "heldout_stderr_at_74pct": round((0.74 * 0.26 / len(heldout)) ** 0.5, 4),
        "subject_balance": {
            name: dict(Counter(r["subject"] for r in part))
            for name, part in (("train", train), ("val", val), ("heldout", heldout))
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
