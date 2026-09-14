#!/usr/bin/env python3
"""Split the rows no window grounds into the causes that have different prices.

"Not groundable" was one number and it hid four situations that cost nothing,
nothing, a corpus download, and nothing-can-be-done respectively:

  split      the answer IS in the corpus, but its terms straddle two windows.
             Known to be a real population: pooled containment 83.21% against
             single-window 67.03%, so 16.18% of all rows are lost to the cut
             alone. Fix is structure-aware chunking. Free.

  ranked-out the document is in the corpus and BM25 put it below the cut.
             Fix is the dense rerank already running. Free.

  absent     no document in the corpus carries the answer at all. This is the
             only bucket a download can help, and it is the one worth paying
             for.

  broken     the gold label is wrong, so nothing grounds it because there is
             nothing true to ground. No amount of corpus moves these.

Splitting split/ranked-out from absent needs a retriever-free pass over the
whole corpus, which corpus_containment.py already does. This script does the
cheap half - the part decidable from what has already been computed - and
writes the subset that still needs that expensive pass, so the expensive pass
runs over hundreds of rows instead of ten thousand.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")


def official(completion: str) -> str:
    m = STRICT.findall(completion or "") or LOOSE.findall(completion or "")
    return m[-1].strip().rstrip(".").upper() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--field", default="completion")
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--audit", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True,
                    help="rows needing the retriever-free corpus pass")
    args = ap.parse_args()

    ground = {}
    for line in args.rows.open(encoding="utf-8"):
        r = json.loads(line)
        ground[r["sample_id"]] = r

    gold, raw = {}, {}
    for line in args.gold.open(encoding="utf-8"):
        r = json.loads(line)
        gold[r["sample_id"]] = (chr(65 + int(r["answer"])), len(r["choices"]))
        raw[r["sample_id"]] = r

    broken = set()
    if args.audit and args.audit.exists():
        a = json.load(args.audit.open(encoding="utf-8"))
        cand = a if isinstance(a, list) else (a.get("items") or a.get("rows") or [])
        for r in cand:
            sid = r.get("sample_id") if isinstance(r, dict) else r
            if sid:
                broken.add(sid)

    j = json.load(args.eval.open(encoding="utf-8"))
    res = j["results"] if isinstance(j, dict) else j

    cells = collections.Counter()
    by_sub = collections.defaultdict(collections.Counter)
    need_pass, ids = [], collections.defaultdict(list)

    for r in res:
        sid = r.get("sample_id")
        gr, g = ground.get(sid), gold.get(sid)
        if gr is None or g is None:
            continue
        letter, nch = g
        pred = official(r.get(args.field) or "")
        if pred in {chr(65 + i) for i in range(nch)} and pred == letter:
            continue                                   # right: not our problem
        if gr.get("single"):
            kind = "addressable_single_window"
        elif sid in broken:
            kind = "broken_label"
        elif gr.get("pooled"):
            kind = "split_across_windows"              # free fix: chunking
        else:
            kind = "needs_corpus_pass"                 # absent OR ranked out
            need_pass.append(raw[sid])
        cells[kind] += 1
        ids[kind].append(sid)
        by_sub[gr.get("subject", "?")][kind] += 1

    report = {
        "wrong_rows_total": sum(cells.values()),
        "cells": dict(cells.most_common()),
        "cells_pct": {k: round(v / max(sum(cells.values()), 1), 4)
                      for k, v in cells.most_common()},
        "by_subject": {s: dict(c) for s, c in sorted(
            by_sub.items(), key=lambda kv: -sum(kv[1].values()))},
        "note": "split_across_windows is recoverable without downloading "
                "anything: the terms are already in the corpus, the window "
                "boundary cut them apart. needs_corpus_pass still mixes "
                "ranked-out with genuinely absent; corpus_containment.py "
                "separates those two.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    for k, v in ids.items():
        args.out.with_suffix(f".{k}.ids.txt").write_text("\n".join(v) + "\n")
    with args.subset.open("w", encoding="utf-8") as fh:
        for r in need_pass:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "by_subject"}, indent=1))
    print("subset rows:", len(need_pass), "->", args.subset)


if __name__ == "__main__":
    main()
