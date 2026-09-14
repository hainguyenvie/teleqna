#!/usr/bin/env python3
"""Deterministic gates between raw generations and the training file.

Order is cheapest-first; each rejection is counted so a collapsing keep-rate
points at the failing stage instead of just shrinking the output.

  schema        4-5 distinct choices, valid answer index, non-empty strings
  style         no "according to this document/passage" leakage; has the
                release tag; question long enough to mean something
  grounded      the `evidence` field is a verbatim (whitespace-normalised)
                substring of the source chunk — checked, not judged
  distinct      distractors are not the gold restated (exact/normalised)
  dedup         one question per normalised 6-word tail across the whole run
  contamination the hard gate: any generated question whose normalised text
                collides with a teleqna test row (6-word-tail match, or
                Jaccard >= 0.6 on content words) is dropped. We generate from
                the same specs the benchmark was built from, so collisions are
                expected and their count is reported, not hidden.

Output rows use the test.jsonl schema (sample_id/question/choices/answer/
subject) so run_baseline.py and the training builder consume them unchanged.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

WS = re.compile(r"\s+")
TAG = re.compile(r"\[[^\]]*\]")
STYLE_BAD = re.compile(
    r"(?i)\b(this|the above|the following|the given)\s+"
    r"(document|excerpt|passage|section|clause|text|table|figure)\b"
    r"|according to (this|the) (document|excerpt|passage|section|clause|text)")
WORD = re.compile(r"[a-z0-9][a-z0-9.\-]+")


def norm(s: str) -> str:
    return WS.sub(" ", TAG.sub(" ", s)).strip().lower()


def words(s: str) -> list[str]:
    return WORD.findall(norm(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--tail", type=int, default=6)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    chunk_text = {json.loads(l)["chunk_id"]: norm(json.loads(l)["text"])
                  for l in args.chunks.open(encoding="utf-8")}

    test_tails: set[str] = set()
    test_words: list[set[str]] = []
    for line in args.test.open(encoding="utf-8"):
        q = json.loads(line)["question"]
        w = words(q)
        if len(w) >= args.tail:
            test_tails.add(" ".join(w[-args.tail:]))
        test_words.append(set(w))

    drops = collections.Counter()
    seen_tails: set[str] = set()
    kept: list[dict] = []
    total = 0
    for line in args.raw.open(encoding="utf-8"):
        rec = json.loads(line)
        ctext = chunk_text.get(rec["chunk_id"], "")
        for item in rec["items"]:
            total += 1
            q = item.get("question")
            choices = item.get("choices")
            ans = item.get("answer")
            ev = item.get("evidence")
            if (not isinstance(q, str) or not isinstance(choices, list)
                    or not all(isinstance(c, str) and c.strip() for c in choices)
                    or not 4 <= len(choices) <= 5
                    or not isinstance(ans, int) or not 0 <= ans < len(choices)
                    or not isinstance(ev, str) or not ev.strip()):
                drops["schema"] += 1
                continue
            if len({norm(c) for c in choices}) != len(choices):
                drops["schema"] += 1
                continue
            literature = "doc_id" in rec
            # Literature-sourced questions carry no "[3GPP Release N]" tag —
            # the real test's Research-publications/overview rows are 99.9%
            # untagged (README.md provenance table), so requiring one here
            # would train a tag that this slice's real counterpart never has.
            needs_release = not literature and "Release" not in q
            if STYLE_BAD.search(q) or len(words(q)) < 5 or needs_release:
                drops["style"] += 1
                continue
            if norm(ev) not in ctext:
                drops["grounded"] += 1
                continue
            gold = norm(choices[ans])
            if any(norm(c) == gold for i, c in enumerate(choices) if i != ans):
                drops["distinct"] += 1
                continue
            w = words(q)
            tail = " ".join(w[-args.tail:])
            if tail in seen_tails:
                drops["dedup"] += 1
                continue
            wset = set(w)
            if tail in test_tails or any(
                    len(wset & t) / max(1, len(wset | t)) >= args.jaccard
                    for t in test_words):
                drops["contamination"] += 1
                continue
            seen_tails.add(tail)
            if literature:
                meta = {"chunk_id": rec["chunk_id"], "doc_id": rec["doc_id"],
                         "collection": rec["collection"], "tier": rec["tier"],
                         "mode": item.get("mode", "standard"),
                         "evidence": ev.strip()}
            else:
                meta = {"chunk_id": rec["chunk_id"], "spec": rec["spec"],
                         "release": rec["release"], "tier": rec["tier"],
                         "mode": item.get("mode", "standard"),
                         "evidence": ev.strip()}
            kept.append({
                "sample_id": f"synth-{len(kept):06d}",
                "sample_index": len(kept),
                "question": q.strip(),
                "choices": [c.strip() for c in choices],
                "answer": ans,
                "subject": "synthetic",
                "meta": meta,
            })

    with args.out.open("w", encoding="utf-8") as out:
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"total": total, "kept": len(kept),
              "keep_rate": round(len(kept) / max(1, total), 4),
              "drops": dict(drops),
              "by_mode": dict(collections.Counter(
                  r["meta"]["mode"] for r in kept)),
              "by_tier": dict(collections.Counter(
                  r["meta"]["tier"] for r in kept))}
    args.out.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
