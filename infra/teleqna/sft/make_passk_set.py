#!/usr/bin/env python3
"""Convert real teleqna rows (dev-1000 shape) into the profiler's prompt shape.

Why this exists. The 83.85% "oracle" this track has been quoting is the
any-branch ceiling over **four choice-order permutations**, not over sampled
reasoning traces. Those are different quantities and only the second one is
GRPO's headroom:

  * a permutation oracle counts a row as recoverable if *some ordering* of the
    options happens to land right. On a knife-edge row that is a coin flip
    repeated four times — 1-(1-p)^4 is 68% at p=0.25 — so the number is
    inflated by exactly the instability it is meant to expose.
  * a sampled-trace oracle at fixed choice order counts a row as recoverable if
    *some reasoning path* reaches the gold letter. That is the pass@k GRPO
    converts into pass@1, and nobody has measured it on the benchmark's own
    rows.

The prompt built here is byte-identical to eval_dev.py's TEMPLATE (which is
run_baseline.py's, which is the Inspect harness's), so pass@1 from the profiler
is directly comparable with the greedy dev-1000 numbers already on file.
Choice order is left exactly as the benchmark ships it — permuting here would
reintroduce the confound this measurement exists to remove.
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
    args = ap.parse_args()

    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for line in args.data.open(encoding="utf-8"):
            row = json.loads(line)
            nc = len(row["choices"])
            prompt = TEMPLATE.format(
                letters=",".join(chr(65 + i) for i in range(nc)),
                question=row["question"],
                choices="\n".join(f"{chr(65+i)}) {c}"
                                  for i, c in enumerate(row["choices"])))
            f.write(json.dumps({
                "prompt": [{"role": "user", "content": prompt}],
                "answer": chr(65 + int(row["answer"])),
                "n_choices": nc,
                "sample_id": row["sample_id"],
                "subject": row.get("subject", "?"),
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} prompts -> {args.out}")


if __name__ == "__main__":
    main()
