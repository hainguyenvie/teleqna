#!/usr/bin/env python3
"""Second stage after scan_contamination.py: separate real leakage from stem collision.

Why this exists
---------------
An 8-gram scan of `OTel-LLM-Data.jsonl` flagged 530/10,000 teleqna questions. Every
inspected example turned out to be a shared multiple-choice STEM -- "which of the
following is not a benefit", "what is the term used to refer to" -- attached to a
completely different question. Generic English boilerplate, not leakage.

The k=8 window is wide enough to catch a stem and narrow enough to stop before the
part that makes a question distinctive. So the first-stage count is an upper bound
on leakage in exactly the way it is a lower bound on coverage: it over-reports when
targets share phrasing, under-reports when targets are shorter than k.

This stage indexes the question's TAIL instead of every window. The last 8 words of
a question carry its specifics, not its formula. A tail hit is then confirmed by
checking the full normalised question is present in the record verbatim.

Reports three tiers, so a caller can tell which failure they are looking at:

    tail_hit  -- last 8 words of the question found in a record
    full_hit  -- whole question found verbatim (this is leakage)
    gold_hit  -- whole question AND its gold option text in the same record
                 (this is leakage with the answer key attached)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scan_contamination import iter_corpus, norm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--test", type=Path, default=Path("data/teleqna/test.jsonl"))
    ap.add_argument("--field", action="append", default=None)
    ap.add_argument("--corpus-pattern", default="*.md")
    ap.add_argument("--tail", type=int, default=8, help="words of question tail to index")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    fields = args.field or ["input", "output", "text"]

    rows = [json.loads(l) for l in args.test.open(encoding="utf-8") if l.strip()]
    questions = [norm(r["question"]) for r in rows]
    golds = []
    for r in rows:
        ans, choices = r.get("answer"), r.get("choices") or []
        golds.append(norm(choices[ans]) if isinstance(ans, int) and ans < len(choices) else "")

    # index the distinctive tail, not every window
    tails: dict[str, list[int]] = {}
    too_short = 0
    for i, q in enumerate(questions):
        words = q.split()
        if len(words) < args.tail:
            too_short += 1
            continue
        tails.setdefault(" ".join(words[-args.tail:]), []).append(i)
    print(f"indexed {len(tails):,} tails; {too_short} questions shorter than {args.tail} words",
          file=sys.stderr, flush=True)

    tail_hit: set[int] = set()
    full_hit: set[int] = set()
    gold_hit: set[int] = set()
    examples: list[dict] = []
    n = 0
    for blob in iter_corpus(args.corpus, fields, args.corpus_pattern):
        n += 1
        text = norm(blob)
        words = text.split()
        cands: set[int] = set()
        for j in range(len(words) - args.tail + 1):
            got = tails.get(" ".join(words[j:j + args.tail]))
            if got:
                cands.update(got)
        for i in cands:
            tail_hit.add(i)
            if questions[i] in text:
                full_hit.add(i)
                if golds[i] and golds[i] in text:
                    gold_hit.add(i)
                if len(examples) < 10:
                    examples.append({"item": i, "question": rows[i]["question"][:200],
                                     "record": re.sub(r"\s+", " ", blob)[:500]})
        if n % 50000 == 0:
            print(f"  ...{n:,} | tail={len(tail_hit)} full={len(full_hit)} gold={len(gold_hit)}",
                  file=sys.stderr, flush=True)

    result = {
        "corpus": str(args.corpus), "records": n, "tail_words": args.tail,
        "questions": len(rows), "questions_too_short": too_short,
        "tail_hit": len(tail_hit), "full_hit": len(full_hit), "gold_hit": len(gold_hit),
        "examples": examples,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
