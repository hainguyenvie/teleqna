#!/usr/bin/env python3
"""Build the retrieval-ceiling arms: the gold `explanation` served as context.

Why this measurement comes before any corpus work. Every plan on the table —
teacher (distil retrieved knowledge into weights) or scaffold (serve retrieval
behind the endpoint) — is bounded by the same quantity: *if the perfect
supporting passage were in the context window, how many rows would the model
then get right?* The benchmark hands us that passage for free. `TeleQnA.zip`
from netop-team ships an `explanation` field for all 10,000 rows, AES-encrypted
with the published password `teleqnadataset`, and it aligns with our parquet
10,000/10,000 by question string.

So this is the ceiling of knowledge injection, measured rather than assumed.
Nothing above it is reachable by putting facts into the model, by any route.

Three arms, and the third is what makes the first one believable:

  expl      the row's own explanation as context.        <- the ceiling
  shuf      another row's explanation, same subject.     <- control
  base      no context at all (already measured: 75.2)

The control matters because "accuracy went up with context" has a boring
explanation available: any context at all changes the decoding, and a model
handed a wall of telecom prose may simply become more careful. If `shuf` also
rises, the ceiling number is measuring attention, not knowledge. If `shuf`
lands at or below base, the `expl` gain is retrieval and can be spent.

**Give-away stratification.** An explanation that restates the gold option
verbatim turns the task into string matching, and a ceiling built out of those
rows would promise a retrieval gain no real corpus can deliver: a real
retriever returns the paragraph a fact came from, not the answer key. So every
row is scored for how much of the gold option's vocabulary the explanation
already contains, *relative to the best distractor*, and the strata are written
alongside the data. The honest ceiling is read on the low-give-away stratum;
the high one is reported and discounted.

    giveaway = |content(gold) & content(expl)| / |content(gold)|
    margin   = giveaway(gold) - max over distractors of the same ratio

`margin` is the discriminating quantity. A high `giveaway` with `margin` near
zero means the explanation shares vocabulary with every option (it is on-topic,
which proves nothing); a high margin means it points at one of them.

The explanation is used for audit only. It must never select training documents
or tune anything — that is test-set-informed selection, and it leaks.

Output files are plain eval sets in the existing schema, so `eval_dev_vllm.py`
scores them unmodified: the context is prepended into the `question` field and
the harness template is untouched.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
STOP = {
    "the", "and", "for", "are", "that", "this", "with", "from", "not", "can",
    "used", "use", "which", "when", "what", "does", "his", "her", "its", "was",
    "were", "has", "have", "had", "will", "would", "should", "could", "may",
    "might", "must", "shall", "all", "any", "other", "than", "then", "there",
    "these", "those", "such", "only", "also", "more", "most", "some", "one",
    "two", "into", "over", "under", "between", "within", "during", "after",
    "before", "above", "below", "each", "both", "same", "different", "using",
    "based", "provide", "provides", "provided", "following", "above.",
}

# Wrapper for the context. Deliberately plain: no instruction to trust it, no
# chain-of-thought scaffold, nothing that would make this arm a prompt
# experiment as well as a knowledge experiment.
CONTEXT_TEMPLATE = (
    "Reference note:\n{context}\n\n{question}"
)


def content(text: str) -> set[str]:
    return {t for t in TOKEN.findall((text or "").lower()) if t not in STOP}


def giveaway(expl: str, gold: str, distractors: list[str]) -> tuple[float, float]:
    """Fraction of the gold option's content words the explanation carries, and
    how much that exceeds the best distractor's fraction."""
    e = content(expl)
    g = content(gold)
    if not g:
        return (0.0, 0.0)
    gv = len(g & e) / len(g)
    best = 0.0
    for d in distractors:
        dc = content(d)
        if dc:
            best = max(best, len(dc & e) / len(dc))
    return (round(gv, 4), round(gv - best, 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teleqna", type=Path, required=True,
                    help="decrypted TeleQnA.txt/json with the explanation field")
    ap.add_argument("--split", type=Path, required=True,
                    help="the eval split to build arms for (e.g. dev1000.jsonl)")
    ap.add_argument("--test", type=Path, required=True,
                    help="the harness test.jsonl, for index alignment")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default="dev1000")
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    raw = json.loads(args.teleqna.read_text(encoding="utf-8"))
    by_index = {int(k.split()[-1]): v for k, v in raw.items()}

    test = [json.loads(l) for l in args.test.open(encoding="utf-8")]
    # Alignment is asserted, not hoped for: the explanation is only usable if
    # the row it belongs to is the row we are scoring.
    mismatch = [i for i, r in enumerate(test)
                if by_index.get(i, {}).get("question", "").strip()
                != r["question"].strip()]
    if mismatch:
        raise SystemExit(f"ABORT: {len(mismatch)} question strings do not align, "
                         f"first at index {mismatch[0]}")
    expl_by_id = {r["sample_id"]: by_index[i].get("explanation", "")
                  for i, r in enumerate(test)}

    rows = [json.loads(l) for l in args.split.open(encoding="utf-8")]
    missing = [r["sample_id"] for r in rows if not expl_by_id.get(r["sample_id"])]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} rows have no explanation")

    # The control draws from the same subject, so it is matched on register and
    # vocabulary and differs only in being about another question.
    rng = random.Random(args.seed)
    by_subject: dict[str, list[str]] = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(r["sample_id"])
    shuffled: dict[str, str] = {}
    for subj, ids in by_subject.items():
        pool = ids[:]
        for _ in range(64):
            rng.shuffle(pool)
            if all(a != b for a, b in zip(ids, pool)):
                break
        else:
            raise SystemExit(f"could not derange subject {subj}")
        shuffled.update(dict(zip(ids, pool)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    strata = {}
    f_expl = (args.out_dir / f"{args.tag}_expl.jsonl").open("w", encoding="utf-8")
    f_shuf = (args.out_dir / f"{args.tag}_explshuf.jsonl").open("w", encoding="utf-8")
    for r in rows:
        sid = r["sample_id"]
        gold = r["choices"][int(r["answer"])]
        distractors = [c for i, c in enumerate(r["choices"]) if i != int(r["answer"])]
        expl = expl_by_id[sid]
        gv, margin = giveaway(expl, gold, distractors)
        strata[sid] = {"subject": r["subject"], "giveaway": gv, "margin": margin,
                       "expl_words": len(expl.split())}

        own = dict(r)
        own["question"] = CONTEXT_TEMPLATE.format(context=expl,
                                                  question=r["question"])
        f_expl.write(json.dumps(own, ensure_ascii=False) + "\n")

        other = dict(r)
        other["question"] = CONTEXT_TEMPLATE.format(
            context=expl_by_id[shuffled[sid]], question=r["question"])
        f_shuf.write(json.dumps(other, ensure_ascii=False) + "\n")
    f_expl.close()
    f_shuf.close()

    hi = sum(1 for v in strata.values() if v["margin"] >= 0.34)
    lo = sum(1 for v in strata.values() if v["margin"] < 0.34)
    (args.out_dir / f"{args.tag}_ceiling.strata.json").write_text(
        json.dumps({"rows": len(rows), "seed": args.seed,
                    "margin_hi_ge_0.34": hi, "margin_lo": lo,
                    "by_id": strata}, indent=1))
    print(f"{len(rows)} rows -> {args.tag}_expl.jsonl / {args.tag}_explshuf.jsonl")
    print(f"give-away margin >= 0.34: {hi}   below: {lo}")


if __name__ == "__main__":
    main()
