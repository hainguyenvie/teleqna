#!/usr/bin/env python3
"""Turn gated grounded items into a closed-book SFT file for train_teleqna.py.

Two decisions carry this file, and both are corrections of what failed before.

**The prompt is the harness prompt, verbatim.** Every training row is shaped
exactly like a benchmark row — same template, same letter list, same option
formatting as run_baseline.py and eval_dev.py. Nothing is learned in a format
the model will never see again.

**The completion is the reasoning, not the letter.** train_v3 supervised
"ANSWER: X" and nothing else: five tokens, one bit, and it cost 3.6 points at
its gentlest dose. Four DPO recipes worked the same channel for a best of +1.4
at p=0.17. Here the target states the spec fact that settles the question, then
says what each wrong option actually is, and only then emits the letter. The
distractor probe measured +15.83pp sitting in option discrimination; this is
that discrimination written down rather than compressed into a letter.

The document is never shown. The model is being trained to produce, from its
weights alone, what validate_grounded_qa.py confirmed it could only produce
with the specification in front of it.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)

_WS = re.compile(r"\s+")
# The evidence is a raw markdown span; asterisks and stray table pipes read as
# noise in a completion the model is asked to imitate.
_MD = re.compile(r"[*_`|]+")


def clean(s: str) -> str:
    return _WS.sub(" ", _MD.sub(" ", s or "")).strip().rstrip(".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-why", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=500,
                    help="prompt+completion words; the trainer refuses rather "
                         "than truncates anything past MAXLEN tokens")
    args = ap.parse_args()

    rows, letters_hist = [], collections.Counter()
    dropped_long = [0]
    for line in args.items.open(encoding="utf-8"):
        it = json.loads(line)
        options = [it["answer"]] + list(it["distractors"])[:3]
        n = len(options)
        if n < 2:
            continue
        # Same deterministic rotation the other builders use, so gold letters
        # stay balanced without a random seed anyone has to remember.
        digest = hashlib.sha256(f"distill_v1:{it['item_id']}".encode()).digest()
        order = [(i + digest[0] % n) % n for i in range(n)]
        shuffled = [options[j] for j in order]
        gold = order.index(0)

        prompt = TEMPLATE.format(
            letters=",".join(chr(65 + i) for i in range(n)),
            question=it["question"],
            choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(shuffled)))

        parts = [f"{clean(it['evidence'])}."]
        for pos, opt in enumerate(shuffled):
            if pos == gold:
                continue
            try:
                why = it["why_wrong"][it["distractors"].index(opt)]
            except (ValueError, IndexError):
                continue
            parts.append(f"{chr(65+pos)}) {clean(opt)} is wrong: {clean(why)}.")
        parts.append(f"ANSWER: {chr(65 + gold)}")
        completion = " ".join(parts[:1 + args.max_why + 1])

        # train_teleqna.py hard-fails on any row over MAXLEN rather than
        # truncating, and carves its loss-curve slice by hashing `sample_id`.
        # Both are its contract, so meet it here instead of at step 0 of a run.
        if len(prompt.split()) + len(completion.split()) > args.max_words:
            dropped_long[0] += 1
            continue

        letters_hist[chr(65 + gold)] += 1
        rows.append({"prompt": prompt, "completion": completion,
                     "sample_id": it["item_id"], "spec": it.get("spec")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lens = sorted(len(r["completion"].split()) for r in rows)
    report = {
        "rows": len(rows),
        "dropped_too_long": dropped_long[0],
        "gold_letters": dict(letters_hist),
        "completion_words": {
            "median": lens[len(lens) // 2] if lens else 0,
            "p90": lens[int(len(lens) * 0.9)] if lens else 0,
            "max": lens[-1] if lens else 0},
        "by_spec": dict(collections.Counter(
            r.get("spec") for r in rows).most_common(20)),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
