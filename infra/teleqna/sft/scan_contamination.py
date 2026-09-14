#!/usr/bin/env python3
"""Shingle-scan any corpus against our test sets before it is used for anything.

Why this exists
---------------
`AdaptKey/AdaptKey-Nemotron-30b` publishes a 5 GB training corpus that turned out
to contain ~9,986 of the 10,000 teleqna test rows verbatim — options, TeleQnA's
own category label, and the gold answer as the SFT target. It is a publicly
promoted model with a leaderboard claim. Nothing about it looked suspicious from
the outside.

So: no third-party corpus or checkpoint gets used on any track until it has been
through this. A non-trivial hit count is blocking, not a caveat to note.

The blind spot, stated up front
-------------------------------
An 8-gram shingle cannot exist for a question shorter than 8 words. On teleqna
that hides 1,312 of 10,000 rows, and the AdaptKey misses matched that blind spot
almost exactly (1,326 missed vs 1,312 short; Lexicon 305/305). **Every number
this prints is a lower bound.** `--k` lowers the window at the cost of false
positives; the `too short to index` line reports how many items are invisible at
the chosen k.

Usage
-----
    python3 scan_contamination.py CORPUS.jsonl \
        --field input --field output \
        --target teleqna=data/teleqna/test.jsonl:question \
        --target srsran=data/srsranbench/test.jsonl:question

Targets are `name=path[:field]`. `.jsonl` files are read line by line; `.json`
files may be a list or a dict of records. Streams the corpus, so a 5 GB file
costs one pass and no memory.

The corpus itself may be a `.jsonl`, a `.parquet`, a directory of parquet shards
(`GSMA/Telco-Common-Corpus`), or a directory of plain files (`GSMA/3GPP` ships
84k markdown specs; use `--corpus-pattern '*.md'`).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


def norm(s: str) -> str:
    """Lowercase, drop bracketed provenance tags, collapse whitespace.

    Tags are stripped because a corpus that reformats "[3GPP Release 18]" away
    is still carrying the same question.
    """
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", "", s)).strip().lower()


def shingles(text: str, k: int):
    words = norm(text).split()
    for i in range(len(words) - k + 1):
        yield " ".join(words[i:i + k])


def iter_corpus(path: Path, fields: list[str], pattern: str):
    """Yield one text blob per corpus record.

    Three shapes, because the corpora we have to clear arrive in all three:
    a `.jsonl` of records, a directory of `.parquet` shards (Telco-Common-Corpus),
    and a directory of markdown files (GSMA/3GPP ships 84k of them).
    """
    if path.is_dir():
        shards = sorted(path.rglob("*.parquet"))
        if shards:
            for shard in shards:
                yield from iter_corpus(shard, fields, pattern)
            return
        for f in sorted(path.rglob(pattern)):
            if f.is_file():
                yield f.read_text(encoding="utf-8", errors="replace")
        return

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.ParquetFile(path)
        names = [f for f in fields if f in table.schema_arrow.names] or ["text"]
        for batch in table.iter_batches(batch_size=512, columns=names):
            cols = [batch.column(nm).to_pylist() for nm in names]
            for row in zip(*cols):
                yield " \n ".join(str(v or "") for v in row)
        return

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            yield " \n ".join(str(rec.get(f, "") or "") for f in fields)


def load_target(spec: str) -> tuple[str, list[str]]:
    name, _, rest = spec.partition("=")
    path, _, field = rest.partition(":")
    field = field or "question"
    p = Path(path)
    if p.suffix == ".jsonl":
        records = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    else:
        loaded = json.loads(p.read_text(encoding="utf-8"))
        records = loaded if isinstance(loaded, list) else list(loaded.values())
    return name, [str(r.get(field, "")) for r in records]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--target", action="append", required=True,
                    help="name=path[:field]; repeatable")
    ap.add_argument("--field", action="append", default=None,
                    help="corpus record fields to search; default input+output+text")
    ap.add_argument("--corpus-pattern", default="*.md",
                    help="when corpus is a directory of plain files, which to read")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--strip-prefix", default=None,
                    help="regex; removed from every target item before indexing. "
                         "Use when all items share a boilerplate header that "
                         "would otherwise match every corpus record.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args()
    fields = args.field or ["input", "output", "text"]
    strip = re.compile(args.strip_prefix, re.S) if args.strip_prefix else None

    targets: dict[str, list[str]] = {}
    index: dict[str, tuple[str, int]] = {}
    for spec in args.target:
        name, items = load_target(spec)
        if strip:
            items = [strip.sub("", it) for it in items]
        targets[name] = items
        short = 0
        for i, q in enumerate(items):
            got = False
            for s in shingles(q, args.k):
                index.setdefault(s, (name, i))
                got = True
            short += not got
        print(f"{name}: {len(items)} items, {short} too short to index at k={args.k}",
              file=sys.stderr, flush=True)
    print(f"index: {len(index):,} shingles", file=sys.stderr, flush=True)

    hits: dict[str, collections.Counter] = {k: collections.Counter() for k in targets}
    examples: list[dict] = []
    n = 0
    for blob in iter_corpus(args.corpus, fields, args.corpus_pattern):
        n += 1
        seen: set[tuple[str, int]] = set()
        for s in shingles(blob, args.k):
            hit = index.get(s)
            if hit is not None and hit not in seen:
                seen.add(hit)
                hits[hit[0]][hit[1]] += 1
                if len(examples) < args.examples:
                    examples.append({
                        "target": hit[0], "item": hit[1],
                        "shingle": s,
                        "corpus_record": re.sub(r"\s+", " ", blob)[:400],
                    })
        if n % 20000 == 0:
            print("  ..." + f"{n:,} | " +
                  " | ".join(f"{k}={len(v)}" for k, v in hits.items()),
                  file=sys.stderr, flush=True)

    result = {
        "corpus": str(args.corpus),
        "records": n,
        "k": args.k,
        "matched": {k: {"hit": len(v), "total": len(targets[k]),
                        "frac": round(len(v) / max(1, len(targets[k])), 4)}
                    for k, v in hits.items()},
        "note": "lower bound; items shorter than k words cannot be detected",
        "examples": examples,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
