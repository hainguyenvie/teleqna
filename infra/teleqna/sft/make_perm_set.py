#!/usr/bin/env python3
"""ot-full with the answer options rotated, as another way of asking the same
question.

The point is not a permutation oracle -- that number was already dismissed on
this project, and correctly: counting a question as recoverable because *some*
ordering landed right is 1-(1-p)^4, which is 68% at p=0.25 and measures the
instability it claims to expose.

What is being built here is the opposite statistic. A question is counted only
if the model answers it correctly in at least 6 of 8 samples, which a coin flip
over five options reaches about once in ten thousand tries. That makes the
per-mode set close to guess-free, and the union over modes an honest floor on
what the weights already hold. plain and cot already disagree on 871 such
questions in the two directions; rotating the options tests whether that spread
is about wording or about position.

Rotation is by a fixed shift rather than a shuffle so the transform is exactly
invertible and the gold letter moves with its own text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shift", type=int, required=True)
    args = ap.parse_args()

    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for line in args.data.open(encoding="utf-8"):
            row = json.loads(line)
            ch = list(row["choices"])
            nc = len(ch)
            s = args.shift % nc
            # position i of the rotated list holds original choice (i - s)
            rot = [ch[(i - s) % nc] for i in range(nc)]
            gold = (int(row["answer"]) + s) % nc
            assert rot[gold] == ch[int(row["answer"])]
            prompt = TEMPLATE.format(
                letters=",".join(chr(65 + i) for i in range(nc)),
                question=row["question"],
                choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(rot)))
            f.write(json.dumps({
                "prompt": [{"role": "user", "content": prompt}],
                "answer": chr(65 + gold),
                "n_choices": nc,
                "sample_id": row["sample_id"],
                "subject": row.get("subject", "?"),
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} prompts rotated by {args.shift} -> {args.out}")


if __name__ == "__main__":
    main()
