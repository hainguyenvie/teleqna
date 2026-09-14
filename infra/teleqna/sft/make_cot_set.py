#!/usr/bin/env python3
"""Rebuild ot-full with a reasoning instruction in the plain channel.

The no-think pass@8 on this model is 0.73 points above a coin flip on every
question it does not already answer 8/8, which says the eight samples are eight
draws on one answer distribution rather than eight reasoning paths. There is
nothing for a branch-selecting method to select.

That is a statement about the channel, not about the model: this base's think
channel is broken (52/1000 parsed on ot-lite against 732 no-think), so it has
been scored its whole life in a mode where it answers immediately. This file
asks for the reasoning in the ordinary assistant turn instead, keeping the final
line in the harness's exact format so the same parser grades both runs and the
numbers stay comparable with everything already measured.

The word cap is in the prompt on purpose. An unbounded request on a model whose
long-form channel is damaged is how the think arm ended up with 941 unparsed
rows, and a rollout that never reaches the answer line scores zero.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PLAIN = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)

COT = (
    "Answer the following multiple choice question. Reason briefly, in at most "
    "80 words, weighing the options against each other. Then end your response "
    "with a final line of exactly the form 'ANSWER: $LETTER' (without quotes) "
    "where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--style", choices=["plain", "cot"], default="cot")
    args = ap.parse_args()

    tmpl = COT if args.style == "cot" else PLAIN
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for line in args.data.open(encoding="utf-8"):
            row = json.loads(line)
            nc = len(row["choices"])
            prompt = tmpl.format(
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
    print(f"wrote {n} {args.style} prompts -> {args.out}")


if __name__ == "__main__":
    main()
