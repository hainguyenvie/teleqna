#!/usr/bin/env python3
"""Cut ranked 3GPP specs into generation-ready chunks.

The error analysis (map_errors_to_specs.py) ranked spec files by how many
hard-core baseline errors their vocabulary explains. Generation budget follows
that ranking: --top-files takes the N densest files, --extra-files adds a
deterministic sample of the rest of the corpus so the training mix does not
collapse onto the failure topics (the model is right on 72% of the test and
must stay right).

Chunks are heading-delimited and greedy-packed to a word budget. Front matter
(copyright, logos, postal addresses) is dropped by starting at the first
"Scope" heading; references/annex-only tails are kept — annexes carry real
facts. Each chunk records the spec identity (series, number, release) so the
generated questions can carry the same provenance tag the test questions have.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
SCOPE = re.compile(r"(?i)^#+\s*\**\s*(1\.?\s+)?scope\b")
IMG_LINE = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$")


def spec_identity(rel_path: str) -> dict:
    """marked/Rel-18/23_series/23501/raw.md -> TS 23.501, Release 18."""
    m = re.search(r"Rel-(\d+)/(\d+)_series/(\d+)", rel_path)
    if not m:
        return {"release": None, "spec": rel_path}
    rel, series, num = m.groups()
    spec = f"{num[:len(series)]}.{num[len(series):]}"
    return {"release": int(rel), "spec": spec, "series": series}


def sections(text: str):
    """Yield (heading_trail, body) per heading-delimited section, post-Scope."""
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if SCOPE.match(line):
            start = i
            break
    trail: dict[int, str] = {}
    head, body = "", []
    for line in lines[start:]:
        m = HEADING.match(line)
        if m:
            if body:
                yield head, "\n".join(body)
            level = len(m.group(1))
            trail[level] = re.sub(r"\**", "", m.group(2)).strip()
            trail = {k: v for k, v in trail.items() if k <= level}
            head = " > ".join(trail[k] for k in sorted(trail))
            body = []
        elif not IMG_LINE.match(line):
            body.append(line)
    if body:
        yield head, "\n".join(body)


def pack(file_path: Path, rel_path: str, min_words: int, max_words: int):
    ident = spec_identity(rel_path)
    buf, buf_heads, n = [], [], 0
    for head, body in sections(file_path.read_text(encoding="utf-8",
                                                   errors="replace")):
        words = body.split()
        if not words:
            continue
        buf.append(body)
        buf_heads.append(head)
        n += len(words)
        if n >= min_words:
            yield {**ident, "source": rel_path,
                   "headings": [h for h in dict.fromkeys(buf_heads) if h],
                   "text": "\n\n".join(buf)[: max_words * 8]}
            buf, buf_heads, n = [], [], 0
    if n >= min_words // 2:
        yield {**ident, "source": rel_path,
               "headings": [h for h in dict.fromkeys(buf_heads) if h],
               "text": "\n\n".join(buf)[: max_words * 8]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ranking", type=Path, required=True,
                    help="errors_to_specs.json from map_errors_to_specs.py")
    ap.add_argument("--top-files", type=int, default=60)
    ap.add_argument("--extra-files", type=int, default=25)
    ap.add_argument("--min-words", type=int, default=700)
    ap.add_argument("--max-words", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ranking = json.loads(args.ranking.read_text())
    ranked = [d["file"] for d in ranking["top_files"][: args.top_files]]
    rest = sorted(str(p.relative_to(args.corpus))
                  for p in args.corpus.rglob("*.md")
                  if str(p.relative_to(args.corpus)) not in set(ranked))
    rng = random.Random(args.seed)
    extra = rng.sample(rest, min(args.extra_files, len(rest)))

    n = 0
    with args.out.open("w", encoding="utf-8") as out:
        for tier, files in (("error_dense", ranked), ("broad", extra)):
            for rel_path in files:
                for chunk in pack(args.corpus / rel_path, rel_path,
                                  args.min_words, args.max_words):
                    chunk["chunk_id"] = f"c{n:06d}"
                    chunk["tier"] = tier
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    n += 1
    print(f"{n} chunks from {len(ranked)} error-dense + {len(extra)} broad files "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
