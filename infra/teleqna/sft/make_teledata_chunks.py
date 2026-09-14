#!/usr/bin/env python3
"""Cut Tele-Data's `standard` shard into generation-ready chunks.

Why a second chunker instead of reusing make_spec_chunks.py. That one reads the
GSMA `marked/` tree: markdown, `#`-delimited headings, one file per spec, path
carrying `Rel-18/23_series/23501`. Tele-Data ships the same corpus as plain text
inside a jsonl, one record per spec, headings surviving only as bare numbered
clause lines, and provenance living in `metadata.file_name` ("29557-i20"). The
two formats have nothing in common below the word level.

Why the corpus is worth re-chunking at all — measured, in
`results/teledata/coverage_standard_w400.json`:

    Standards specifications rows with >=80% of their rare terms in one
    400-word window:   distill_v1 (ours)      105 / 1,999
                       Tele-Data standard   1,323 / 1,999

That is 12.6x the addressable rows, and it is the quantitative reason distill1
bought only +1.1: the corpus, not the method, was the binding constraint.

Selection is corpus-intrinsic on purpose. Chunks are not ranked by which test
errors their vocabulary explains — that is test-set-informed selection and it
leaks. Every spec in the shard contributes, capped, in a deterministic hash
order.

Two exclusions are carried over from the previous generator because they were
paid for once already:

  series 21   Release Description / Summary of Work Items. Catalogues of what
              exists in a release, not statements of how anything works. They
              rank first in errors_to_specs.json for the bad reason that a
              document name-dropping every technology in the release matches
              the rare terms of almost any question. 43% of what survived the
              first pilot's gates was 3GPP document trivia.
  front matter  foreword, scope, references, definitions, change history. States
              nothing testable.

Chunk size defaults to 400 words because that is the window the coverage number
above was measured at: it is the size at which "the fact is in this passage" is
true, rather than "the fact is somewhere in this 12,000-word specification".
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

# 3GPP clause headings survive the text conversion as a bare numbered line:
# "5.2.3.1  Nmfaf_3daDataManagement_Configure service operation". Annex
# headings keep their letter. Anything else is body text.
CLAUSE = re.compile(r"^(\d+(?:\.\d+){0,4})\s+(\S.*)$")
ANNEX = re.compile(r"^(Annex\s+[A-Z][\w.]*)\s*[:(]?\s*(.*)$")
SCOPE = re.compile(r"^\d+(?:\.\d+)*\s+Scope\b|^Scope$", re.IGNORECASE)
# Table-of-contents lines are dot leaders; change-history tables are date rows.
TOC = re.compile(r"\.{5,}")
FRONT_MATTER = re.compile(
    r"(?i)\b(foreword|scope|references|definitions,? symbols and abbreviations"
    r"|abbreviations|contents|copyright|change history|document history"
    r"|introduction|modal verbs terminology)\b")
EXCLUDE_SERIES = {"21"}


def spec_of(meta: dict) -> str:
    """{'series': '29', 'file_name': '29557-i20'} -> '29.557'."""
    name = str(meta.get("file_name", ""))
    digits = re.match(r"(\d+)", name)
    series = str(meta.get("series", "")).strip()
    if not digits:
        return series or "?"
    num = digits.group(1)
    if series and num.startswith(series):
        return f"{series}.{num[len(series):]}"
    return f"{num[:2]}.{num[2:]}" if len(num) > 2 else num


def sections(text: str):
    """Yield (heading, body) per clause, starting at the first Scope clause.

    Everything before Scope is cover page, foreword and the modal-verbs
    boilerplate that every 3GPP document repeats verbatim - keeping it would
    teach the model the shape of a specification rather than its content.
    """
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if SCOPE.match(line.strip()):
            start = i
            break
    head, body = "", []
    for raw in lines[start:]:
        line = raw.strip()
        if not line or TOC.search(line):
            continue
        m = CLAUSE.match(line) or ANNEX.match(line)
        # A clause heading is short. A body sentence that happens to start with
        # a number ("5.2 dB is the target...") is not, and misreading it as a
        # heading would cut a chunk in the middle of a fact.
        if m and len(line.split()) <= 12:
            if body:
                yield head, "\n".join(body)
            head = line
            body = []
        else:
            body.append(line)
    if body:
        yield head, "\n".join(body)


def pack(text: str, target: int, hard_max: int):
    """Greedy-pack sections to `target` words, never exceeding `hard_max`."""
    buf, words, heads = [], 0, []
    for head, body in sections(text):
        n = len(body.split())
        if n == 0:
            continue
        if words and words + n > hard_max:
            yield heads, "\n".join(buf)
            buf, words, heads = [], 0, []
        heads.append(head)
        buf.append((head + "\n" + body) if head else body)
        words += n
        if words >= target:
            yield heads, "\n".join(buf)
            buf, words, heads = [], 0, []
    if words >= 40:            # a 40-word tail carries at most one fact; keep it
        yield heads, "\n".join(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, required=True,
                    help="tele-data/standard/standard.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-words", type=int, default=400)
    ap.add_argument("--max-words", type=int, default=650)
    ap.add_argument("--per-spec-cap", type=int, default=120,
                    help="chunks kept per specification. A cap rather than a "
                         "ranking: 38.331 is 900 chunks of the pool and would "
                         "otherwise dominate a set meant to cover 2,000 test "
                         "rows spread over dozens of specs.")
    ap.add_argument("--out-report", type=Path, default=None)
    args = ap.parse_args()

    kept, stats = [], collections.Counter()
    per_spec: dict[str, list] = {}
    for line in args.shard.open(encoding="utf-8"):
        try:
            rec = json.loads(line)
        except Exception:
            stats["unparsable_record"] += 1
            continue
        meta = rec.get("metadata") or {}
        series = str(meta.get("series", "")).strip()
        if series in EXCLUDE_SERIES:
            stats["skipped_series_21"] += 1
            continue
        spec = spec_of(meta)
        release = meta.get("release")
        for i, (heads, text) in enumerate(pack(rec.get("content", ""),
                                               args.target_words,
                                               args.max_words)):
            if FRONT_MATTER.search(" ".join(heads)):
                stats["front_matter"] += 1
                continue
            if len(text.split()) < 40:
                stats["too_short"] += 1
                continue
            per_spec.setdefault(spec, []).append({
                "chunk_id": f"{meta.get('file_name', rec.get('id'))}#{i}",
                "spec": spec, "series": series, "release": release,
                "headings": heads, "text": text,
                "words": len(text.split()),
            })

    # Deterministic sample per spec, not a prefix: a prefix is the opening
    # clauses of every document, which is the most generic part of each.
    for spec, chunks in sorted(per_spec.items()):
        chunks.sort(key=lambda c: hashlib.sha256(
            ("teledata-std-v1|" + c["chunk_id"]).encode()).hexdigest())
        kept.extend(chunks[:args.per_spec_cap])
        stats["capped_away"] += max(0, len(chunks) - args.per_spec_cap)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for c in kept:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    words = sorted(c["words"] for c in kept)
    report = {
        "shard": str(args.shard), "chunks": len(kept),
        "specs": len(per_spec),
        "target_words": args.target_words, "max_words": args.max_words,
        "per_spec_cap": args.per_spec_cap,
        "words_median": words[len(words) // 2] if words else 0,
        "words_p90": words[int(0.9 * (len(words) - 1))] if words else 0,
        "dropped": dict(stats),
        "top_specs": collections.Counter(
            c["spec"] for c in kept).most_common(20),
    }
    print(json.dumps(report, indent=1))
    if args.out_report:
        args.out_report.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
