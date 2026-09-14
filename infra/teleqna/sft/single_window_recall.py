#!/usr/bin/env python3
"""How many rows have the answer inside ONE retrieved window, not spread across 32.

The retrieval sweep reported `answer_in_ctx` over the concatenation of all k
windows. That is the right metric for serving - the model reads the whole block -
but it is the wrong one for generation, and the difference is not cosmetic.
Arm D scores 0.8321 over 32 concatenated windows while the corpus-wide
single-window containment ceiling measured 0.827; a pooled metric can clear a
per-window ceiling only by counting rows whose answer terms are split across
several windows, and no such row yields a groundable fact.

A generator is handed one window and asked for a fact with verbatim evidence.
If the answer is assembled from window 3 and window 17 there is no span to
quote, the evidence gate rejects the item, and the GPU hours are spent producing
nothing. So the number that predicts the size of the training set is: for how
many rows does at least one single window pass the same 80%-of-gold-terms bar?

Reported three ways, because they answer different questions:
  pooled     the sweep's number, reproduced here as a check that this script
             applies the identical criterion
  single     at least one window passes alone -> a fact can be grounded
  best_rank  where that window sits in the BM25 ordering, which says how deep
             the generator has to read before it can stop
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
# Copied verbatim from retrieve_ctx.py. A wider stop list drops more of the gold
# answer's terms and so makes the 80% bar easier to clear: a first draft with an
# invented list scored the pooled column at 89.81% where the sweep that produced
# this very file reported 83.21%. The whole point of the pooled column is to be
# comparable, so the list has to be the same list, not a similar one.
STOP = {
    "the", "and", "for", "that", "with", "this", "are", "was", "which", "from",
    "has", "have", "not", "can", "may", "shall", "will", "its", "their", "than",
    "when", "what", "which", "where", "how", "why", "who", "does", "did", "any",
    "all", "one", "two", "following", "above", "below", "used", "use", "using",
    "purpose", "main", "key", "type", "types", "based", "into", "such", "other",
}
# The block emitted by retrieve_ctx.py: "[1] words...\n\n[2] words...", then a
# "\n\n---\n\n" rule, then the original question. Without cutting at the rule the
# last window absorbs the question text and scores higher than it earned.
MARKER = re.compile(r"(?m)^\[(\d+)\]\s", re.MULTILINE)
RULE = "\n\n---\n\n"


def terms(text: str) -> set[str]:
    return {t for t in TOKEN.findall(text.lower()) if t not in STOP}


def split_windows(question: str) -> list[str]:
    """Recover the individual windows from the concatenated prompt.

    Cut the trailing question off first, then split on the [n] markers. The
    question is not neutral filler: it restates the stem, and TeleQnA stems
    share vocabulary with their own gold option, so leaving it attached hands
    free terms to whichever window happens to be last.
    """
    body = question.split(RULE)[0]
    marks = list(MARKER.finditer(body))
    if not marks:
        return []
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out.append(body[m.end():end])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thresh", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    n = pooled = single = 0
    by_sub = collections.defaultdict(lambda: [0, 0, 0])   # single, pooled, n
    rank_hist = collections.Counter()
    per_row = []

    with args.rag.open(encoding="utf-8") as fh:
        for line in fh:
            if args.limit and n >= args.limit:
                break
            row = json.loads(line)
            n += 1
            gold = row["choices"][int(row["answer"])]
            gt = terms(gold)
            sub = row.get("subject", "?")
            by_sub[sub][2] += 1
            if not gt:
                continue
            wins = split_windows(row["question"])
            need = args.thresh * len(gt)

            pooled_hit = len(gt & terms(" ".join(wins))) >= need
            pooled += pooled_hit
            by_sub[sub][1] += pooled_hit

            best = -1
            for i, w in enumerate(wins):
                if len(gt & terms(w)) >= need:
                    best = i
                    break
            if best >= 0:
                single += 1
                by_sub[sub][0] += 1
                rank_hist[best] += 1
            per_row.append({"sample_id": row["sample_id"], "subject": sub,
                            "single": best >= 0, "pooled": bool(pooled_hit),
                            "best_window": best, "n_ctx": len(wins)})
            if n % 2000 == 0:
                print(f"{n}  single {single/n*100:.1f}%  pooled {pooled/n*100:.1f}%",
                      flush=True)

    # Depth needed: how many windows must be read to cover x% of the groundable
    # rows. This is what sets the generation budget, not k.
    cum, depth = 0, {}
    for r in sorted(rank_hist):
        cum += rank_hist[r]
        depth[r + 1] = round(cum / max(single, 1), 4)

    report = {
        "rag": str(args.rag), "n": n, "thresh": args.thresh,
        "pooled_frac": round(pooled / n, 4),
        "single_window_frac": round(single / n, 4),
        "lost_to_splitting": round((pooled - single) / n, 4),
        "by_subject": {s: {"n": v[2], "single": v[0], "pooled": v[1],
                           "single_frac": round(v[0] / v[2], 4) if v[2] else 0,
                           "pooled_frac": round(v[1] / v[2], 4) if v[2] else 0}
                       for s, v in sorted(by_sub.items())},
        "cumulative_by_depth": {k: depth[k] for k in sorted(depth)
                                if k in (1, 2, 4, 8, 16, 24, 32)},
        "note": "single_window_frac is the ceiling on groundable training rows: "
                "a fact needs one quotable span, not terms scattered over k windows",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    args.out.with_suffix(".rows.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in per_row))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
