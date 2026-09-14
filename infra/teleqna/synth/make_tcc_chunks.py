#!/usr/bin/env python3
"""Cut GSMA/Telco-Common-Corpus research articles into generation-ready
chunks — the literature-side counterpart to make_spec_chunks.py.

Why this corpus at all: teleqna/README.md's provenance table shows 3GPP can
reach at most 1,810/10,000 test rows (18%); "Research publications" (4,500)
+ "Research overview" (2,000) is 65% of the benchmark and has no 3GPP
provenance. GSMA published the sources behind it too — IEEE-Access
(64k open-access journal articles) and OpenAlex — inside Telco-Common-Corpus,
filtered out already by fetch_tcc.py into tcc_research.jsonl.

Same heading-delimited greedy-pack strategy as make_spec_chunks.py, minus the
"Scope"-anchored front-matter skip (a journal article's front matter is the
title/authors/abstract, which is exactly what should stay — the abstract is
often the highest-density paragraph in the whole document).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HEADING = re.compile(r"^(#{1,4})\s+(.*)$")


def sections(text: str):
    lines = text.split("\n")
    trail: dict[int, str] = {}
    head, body = "", []
    for line in lines:
        m = HEADING.match(line)
        if m:
            if body:
                yield head, "\n".join(body)
            level = len(m.group(1))
            trail[level] = re.sub(r"\**", "", m.group(2)).strip()
            trail = {k: v for k, v in trail.items() if k <= level}
            head = " > ".join(trail[k] for k in sorted(trail))
            body = []
        else:
            body.append(line)
    if body:
        yield head, "\n".join(body)


def pack(doc: dict, min_words: int, max_words: int):
    buf, buf_heads, n = [], [], 0
    for head, body in sections(doc["text"]):
        words = body.split()
        if not words:
            continue
        buf.append(body)
        buf_heads.append(head)
        n += len(words)
        if n >= min_words:
            yield {"doc_id": doc["doc_id"], "collection": doc["collection"],
                   "title": doc["title"],
                   "headings": [h for h in dict.fromkeys(buf_heads) if h],
                   "text": "\n\n".join(buf)[: max_words * 8]}
            buf, buf_heads, n = [], [], 0
    if n >= min_words // 2:
        yield {"doc_id": doc["doc_id"], "collection": doc["collection"],
               "title": doc["title"],
               "headings": [h for h in dict.fromkeys(buf_heads) if h],
               "text": "\n\n".join(buf)[: max_words * 8]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, required=True,
                    help="tcc_research.jsonl from fetch_tcc.py")
    ap.add_argument("--min-words", type=int, default=700)
    ap.add_argument("--max-words", type=int, default=1800)
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="cap number of source docs, for a pilot batch")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    docs = [json.loads(l) for l in args.docs.open(encoding="utf-8")]
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    if args.limit_docs:
        docs = docs[: args.limit_docs]

    n = 0
    by_coll = {}
    with args.out.open("w", encoding="utf-8") as out:
        for doc in docs:
            for chunk in pack(doc, args.min_words, args.max_words):
                chunk["chunk_id"] = f"t{n:06d}"
                chunk["tier"] = "tcc_" + doc["collection"].lower()
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                n += 1
                by_coll[doc["collection"]] = by_coll.get(doc["collection"], 0) + 1
    print(f"{n} chunks from {len(docs)} docs -> {args.out}  {by_coll}")


if __name__ == "__main__":
    main()
