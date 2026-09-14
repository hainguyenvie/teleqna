#!/usr/bin/env python3
"""Map baseline errors to the corpus documents that carry the missing facts.

The error analysis says where the baseline fails (Standards specifications,
3GPP-tagged rows, 30% hard-core error rate). Synthetic training data has to be
generated *from* somewhere, and generating uniformly over a 3 GB corpus wastes
most of the budget on facts the model already knows. This ranks corpus files by
how many hard-core errors their vocabulary can explain, so generation starts
from the densest documents.

Never emits test rows themselves — the output is a ranked list of corpus file
paths plus per-file counts, which seeds `make_synth_mcq` later. The test set
stays out of every generated artifact.

Same rare-term machinery as measure_otel_coverage.py: a hard-core row "hits" a
file when >= --min-frac of its rare terms (question + gold) occur in that file.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from measure_otel_coverage import terms_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True,
                    help="directory of .md spec files")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--errors", required=True,
                    help="comma-separated results.jsonl; hard-core = wrong in all")
    ap.add_argument("--max-df", type=int, default=100)
    ap.add_argument("--min-frac", type=float, default=0.6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.test.open(encoding="utf-8")]
    hardcore: set[str] | None = None
    for path in args.errors.split(","):
        wrong = {json.loads(l)["sample_id"]
                 for l in open(path) if not json.loads(l)["correct"]}
        hardcore = wrong if hardcore is None else hardcore & wrong

    df: collections.Counter = collections.Counter()
    all_terms = []
    for r in rows:
        t = terms_of(r["question"]) | terms_of(r["choices"][int(r["answer"])])
        all_terms.append(t)
        df.update(t)
    rare = {t for t, c in df.items() if c <= args.max_df}

    index: dict[str, list[int]] = collections.defaultdict(list)
    rare_of: dict[int, set[str]] = {}
    subjects: dict[int, str] = {}
    for i, r in enumerate(rows):
        if r["sample_id"] not in hardcore:
            continue
        rt = all_terms[i] & rare
        if not rt:
            continue
        rare_of[i] = rt
        subjects[i] = r["subject"]
        for term in rt:
            index[term].append(i)
    print(f"hard-core rows with rare terms: {len(rare_of)}", file=sys.stderr)

    files = sorted(args.corpus.rglob("*.md"))
    print(f"corpus files: {len(files)}", file=sys.stderr, flush=True)
    per_file: list[dict] = []
    best_frac: dict[int, float] = collections.defaultdict(float)
    for k, f in enumerate(files, 1):
        blob = terms_of(f.read_text(encoding="utf-8", errors="replace"))
        matched: collections.Counter = collections.Counter()
        for term in blob & index.keys():
            for i in index[term]:
                matched[i] += 1
        hits = []
        for i, c in matched.items():
            frac = c / len(rare_of[i])
            if frac > best_frac[i]:
                best_frac[i] = frac
            if frac >= args.min_frac:
                hits.append(i)
        if hits:
            per_file.append({
                "file": str(f.relative_to(args.corpus)),
                "n_errors_explained": len(hits),
                "subjects": dict(collections.Counter(subjects[i] for i in hits)),
            })
        if k % 2000 == 0:
            print(f"  ...{k}/{len(files)} files, "
                  f"{sum(1 for v in best_frac.values() if v >= args.min_frac)}"
                  f"/{len(rare_of)} errors matched", file=sys.stderr, flush=True)

    per_file.sort(key=lambda d: -d["n_errors_explained"])
    matched_rows = sum(1 for v in best_frac.values() if v >= args.min_frac)
    report = {
        "corpus": str(args.corpus),
        "hardcore_rows": len(rare_of),
        "rows_explained_by_some_file": matched_rows,
        "min_frac": args.min_frac,
        "top_files": per_file[:300],
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "top_files"},
                     indent=2))
    for d in per_file[:15]:
        print(f"  {d['n_errors_explained']:>4}  {d['file']}")


if __name__ == "__main__":
    main()
