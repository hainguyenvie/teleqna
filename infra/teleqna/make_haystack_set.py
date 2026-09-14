#!/usr/bin/env python3
"""Build the haystack arms: the right note, buried among K-1 wrong ones.

**None of these arms is submittable.** They are fed the benchmark's own
`explanation` field, which is the answer key. They exist for the same reason the
"fit on all 10,000, score on the same 10,000" run exists in this repo: to put a
number on what total leakage buys, so every honest number afterwards can be
discounted against it.

What they measure that arm 0.1 could not. 0.1 handed the model exactly one note,
its own, and scored 97.50 against a base of 75.20. That is perfect retrieval:
recall 1.0, precision 1.0. No retriever is like that. A real one returns the
right passage somewhere inside a stack of plausible neighbours, and the model
has to pick. So this sweeps K:

    K=1    the right note alone                  = arm 0.1, precision 1.0
    K=4    right note + 3 same-subject notes     precision 0.25
    K=16   right note + 15                       precision 0.06
    K=64   right note + 63                       precision 0.016
    miss   64 notes, the right one NOT among them

The last arm is the one people forget. Every retriever misses sometimes, and
what matters is whether a miss costs a point or costs several: a model that
believes a confidently-wrong note may score *below* its own closed-book
baseline. That number is the risk premium on the whole scaffold route, and it
is not knowable from precision alone.

Distractor notes are drawn from the same subject on purpose. Cross-subject
noise is trivially ignorable — a question about IEEE 802.11 next to a note
about federated learning is not a test of anything. A retriever's actual
failure mode is returning passages that are topically right and specifically
wrong, and that is what same-subject sampling reproduces.

The context is a flat, unlabelled list. Nothing marks which note is relevant,
nothing numbers them, and their order is shuffled per row: any structure would
leak position information the real system will not have.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CONTEXT_TEMPLATE = "Reference notes:\n{notes}\n\n{question}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleqna", type=Path, required=True,
                    help="decrypted TeleQnA.txt/json carrying `explanation`")
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default="dev1000")
    ap.add_argument("--k", type=int, action="append", default=None,
                    help="notes per row, repeatable; default 4 16 64")
    ap.add_argument("--miss", type=int, default=64,
                    help="also build an arm of this many notes with the row's "
                         "own note excluded; 0 to skip")
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()
    ks = args.k or [4, 16, 64]

    raw = json.loads(args.teleqna.read_text(encoding="utf-8"))
    by_index = {int(k.split()[-1]): v for k, v in raw.items()}
    test = [json.loads(l) for l in args.test.open(encoding="utf-8")]
    bad = [i for i, r in enumerate(test)
           if by_index.get(i, {}).get("question", "").strip() != r["question"].strip()]
    if bad:
        raise SystemExit(f"ABORT: {len(bad)} rows misaligned, first at {bad[0]}")

    expl, subject = {}, {}
    for i, r in enumerate(test):
        expl[r["sample_id"]] = (by_index[i].get("explanation") or "").strip()
        subject[r["sample_id"]] = r["subject"]
    # The distractor pool is the whole 10,000, not the split: a retriever
    # searches the corpus, not the evaluation slice.
    pool: dict[str, list[str]] = {}
    for sid, subj in subject.items():
        if expl[sid]:
            pool.setdefault(subj, []).append(sid)

    rows = [json.loads(l) for l in args.split.open(encoding="utf-8")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def build(k: int, include_own: bool, name: str) -> None:
        rng = random.Random(args.seed + k + (0 if include_own else 977))
        path = args.out_dir / f"{args.tag}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                sid = r["sample_id"]
                others = [s for s in pool[subject[sid]] if s != sid]
                want = k - 1 if include_own else k
                picked = rng.sample(others, min(want, len(others)))
                notes = [expl[s] for s in picked]
                if include_own:
                    notes.append(expl[sid])
                rng.shuffle(notes)
                out = dict(r)
                out["question"] = CONTEXT_TEMPLATE.format(
                    notes="\n".join(f"- {n}" for n in notes),
                    question=r["question"])
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        words = k * 20
        print(f"{path.name}: {len(rows)} rows, {k} notes each "
              f"(~{words} context words), own note {'in' if include_own else 'ABSENT'}")

    for k in ks:
        build(k, True, f"hay{k}")
    if args.miss:
        build(args.miss, False, f"miss{args.miss}")


if __name__ == "__main__":
    main()
