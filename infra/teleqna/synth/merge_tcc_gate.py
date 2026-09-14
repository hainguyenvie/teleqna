#!/usr/bin/env python3
"""Join a gated TCC (literature) pool with its student-difficulty-gate
results into the schema build_train_set.py's --extra expects: each kept row
plus a merged-in 'student_correct' bool (None if the request/parse failed)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    correct = {}
    for line in args.results.open(encoding="utf-8"):
        row = json.loads(line)
        correct[row["sample_id"]] = bool(row["correct"])

    n = 0
    with args.out.open("w", encoding="utf-8") as out:
        for line in args.kept.open(encoding="utf-8"):
            row = json.loads(line)
            row["student_correct"] = correct.get(row["sample_id"])
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows -> {args.out}")


if __name__ == "__main__":
    main()
