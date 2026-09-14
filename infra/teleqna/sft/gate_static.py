#!/usr/bin/env python3
"""The deterministic half of the gate, before any GPU is spent on the other half.

validate_grounded_qa.py loads a model to ask whether the fact is already known
closed-book. That check is worth running, but it is worth running on items that
have already survived the cheap checks: an item whose evidence is not actually
in the chunk is fabricated no matter what the closed-book probe says, and paying
for a 122B forward pass to discover that is backwards.

Each check is a separate column rather than one pass/fail, because the failure
mode tells you what to fix. Evidence not found verbatim means the teacher is
paraphrasing the source and the prompt needs tightening. Answer not inside its
own evidence means the span was chosen badly and the cloze view will be
unbuildable. Degenerate distractors mean the hard-negative half of the recipe
is not happening and the mcq views will teach nothing about discrimination -
which is the measured error mode of this benchmark, so it is the one that
matters most.

Normalisation for the verbatim test is whitespace-only. Lowercasing or stripping
punctuation would let a paraphrase pass, and the whole value of the evidence
field is that it can be quoted back to a reader who has the specification open.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep", type=Path, help="write surviving items here")
    ap.add_argument("--max-answer-words", type=int, default=12)
    args = ap.parse_args()

    chunk = {}
    for line in args.chunks.open(encoding="utf-8"):
        r = json.loads(line)
        chunk[r["chunk_id"]] = norm(r["text"])

    fail = collections.Counter()
    by_kind = collections.defaultdict(collections.Counter)
    seen_q: set[str] = set()
    kept, n = [], 0

    for line in args.items.open(encoding="utf-8"):
        r = json.loads(line)
        n += 1
        kind = r.get("kind", "?")
        by_kind[kind]["n"] += 1
        bad = []

        ev, ans = norm(r.get("evidence", "")), norm(r.get("answer", ""))
        src = chunk.get(r.get("chunk_id"), "")

        if not ev or not ans or not norm(r.get("question", "")):
            bad.append("empty_field")
        if ev and src and ev not in src:
            bad.append("evidence_not_verbatim")
        if ev and not src:
            bad.append("chunk_missing")
        if ans and ev and ans.lower() not in ev.lower():
            bad.append("answer_not_in_evidence")
        if len(ans.split()) > args.max_answer_words:
            bad.append("answer_too_long")

        d = [norm(x) for x in (r.get("distractors") or [])]
        if len(d) != 3:
            bad.append("distractor_count")
        elif len(set(x.lower() for x in d)) != 3:
            bad.append("distractors_duplicated")
        elif any(x.lower() == ans.lower() for x in d):
            bad.append("distractor_equals_answer")
        w = [norm(x) for x in (r.get("why_wrong") or [])]
        if len(w) != len(d):
            bad.append("why_wrong_count")

        key = norm(r.get("question", "")).lower()
        if key in seen_q:
            bad.append("duplicate_question")
        else:
            seen_q.add(key)

        for b in bad:
            fail[b] += 1
            by_kind[kind][b] += 1
        # Two severities, because they have different consequences. A missing
        # verbatim span means the teacher invented the quote and the item is
        # unusable in every view. An answer that does not appear inside its own
        # evidence only costs the cloze view - expand_views.py already skips
        # cloze in that case and builds mcq, qa, reverse and statement anyway -
        # so rejecting it outright would throw away four usable views to protect
        # one, and would understate the yield of the full run by a third.
        hard = [b for b in bad if b != "answer_not_in_evidence"]
        if hard:
            by_kind[kind]["rejected"] += 1
        else:
            kept.append(r)
            by_kind[kind]["kept"] += 1
            if bad:
                by_kind[kind]["kept_without_cloze"] += 1

    windows = len({r["chunk_id"] for r in kept})
    strict = sum(1 for r in kept
                 if norm(r["answer"]).lower() in norm(r["evidence"]).lower())
    report = {
        "items_in": n, "items_kept": len(kept),
        "keep_rate": round(len(kept) / n, 4) if n else 0,
        "keep_rate_strict_all_views": round(strict / n, 4) if n else 0,
        "kept_without_cloze": len(kept) - strict,
        "windows_with_a_surviving_item": windows,
        "items_per_surviving_window": round(len(kept) / windows, 2) if windows else 0,
        "failures": dict(fail.most_common()),
        "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        "note": "static checks only; the closed-book novelty probe is separate "
                "and should run on the survivors, not on everything",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    if args.keep:
        with args.keep.open("w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
