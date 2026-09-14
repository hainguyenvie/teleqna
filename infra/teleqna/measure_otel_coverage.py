#!/usr/bin/env python3
"""Measure how much of teleqna the OTel-LLM 606k SFT set can actually teach.

Contamination scanning answered "is the test set inside the training data"
(no: 15/10,000 verbatim). This answers the opposite question: for each test
row, is there at least one SFT record whose text carries the rare technical
vocabulary of that question and its gold answer? A specialist trained on this
set can only learn facts the set contains, so a question whose rare terms
never co-occur in any single record is one the SFT stage cannot fix.

Method
------
- Terms: lowercase alnum tokens (dots/dashes kept, so "802.11ax" and
  "ng-eNB" survive), length >= 3, stopwords dropped, bracket provenance tags
  stripped. Each test row contributes the terms of its question plus its gold
  choice.
- Only terms appearing in <= --max-df test rows are indexed ("rare terms").
  Ubiquitous vocabulary ("network", "3gpp") says nothing about whether a
  specific fact is present, and it is what made the first-stage contamination
  scan over-report.
- One streaming pass over the SFT records (anchor + prompt + completion —
  the prompt embeds the retrieval chunks, which is where the knowledge is).
  For each record, count how many of each candidate row's rare terms appear;
  keep the per-row maximum fraction across all records.

coverage(q) = max over records of |rare(q) in record| / |rare(q)|

Rows with no rare terms at all are reported separately: they are generic
questions the metric cannot judge (and the model rarely misses).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
STOP = set("""the and for with that this from are was were has have been which what
when where how why can could should would may might must not all any each between
following true false none above about into over under more most less least other
than then them they their there here also does doing done being its it's used use
using two one three four five per only such same different means term terms refer
refers defined define definition option options answer question correct""".split())


def norm(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", " ", text).lower()


def terms_of(text: str) -> set[str]:
    return {t for t in TOKEN.findall(norm(text)) if t not in STOP}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--errors", type=Path, default=None,
                    help="results.jsonl files whose wrong rows define the "
                         "hard-core subset; comma-separated, a row counts "
                         "when every file got it wrong")
    ap.add_argument("--max-df", type=int, default=100)
    ap.add_argument("--field", action="append", default=None,
                    help="record fields to read, repeatable. Default is the "
                         "OTel SFT triple anchor/prompt/completion; pass "
                         "--field content for corpora like Tele-Data.")
    ap.add_argument("--window", type=int, default=0,
                    help="split each record into windows of this many words, "
                         "0 = whole record. A corpus of full papers and a "
                         "corpus of SFT records are not comparable at record "
                         "granularity: a 6,000-word paper carries the rare "
                         "terms of many unrelated questions and scores a "
                         "coverage no retrievable passage would. Pass ~400 to "
                         "ask the question that matters — is the fact in one "
                         "passage — rather than 'is it somewhere in the file'.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.test.open(encoding="utf-8")]
    row_terms: list[set[str]] = []
    df: collections.Counter = collections.Counter()
    for r in rows:
        t = terms_of(r["question"]) | terms_of(r["choices"][int(r["answer"])])
        row_terms.append(t)
        df.update(t)

    rare = {t for t, c in df.items() if c <= args.max_df}
    index: dict[str, list[int]] = collections.defaultdict(list)
    rare_of: list[set[str]] = []
    for i, t in enumerate(row_terms):
        rt = t & rare
        rare_of.append(rt)
        for term in rt:
            index[term].append(i)
    judgeable = [i for i, rt in enumerate(rare_of) if rt]
    print(f"test rows {len(rows)}, judgeable {len(judgeable)}, "
          f"indexed terms {len(index):,}", file=sys.stderr, flush=True)

    best = [0.0] * len(rows)
    n = 0
    with args.sft.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n += 1
            fields = args.field or ("anchor", "prompt", "completion")
            blob = " ".join(str(rec.get(k, "") or "") for k in fields)
            if args.window:
                words = blob.split()
                blobs = [" ".join(words[s:s + args.window])
                         for s in range(0, max(1, len(words)), args.window)]
            else:
                blobs = [blob]
            for blob in blobs:
                blob_terms = terms_of(blob)
                matched: collections.Counter = collections.Counter()
                for term in blob_terms & index.keys():
                    for i in index[term]:
                        matched[i] += 1
                for i, c in matched.items():
                    frac = c / len(rare_of[i])
                    if frac > best[i]:
                        best[i] = frac
            if n % 50000 == 0:
                covered = sum(best[i] >= 0.5 for i in judgeable)
                print(f"  ...{n:,} records | >=50% covered: "
                      f"{covered}/{len(judgeable)}", file=sys.stderr, flush=True)

    hardcore: set[str] | None = None
    if args.errors:
        for path in str(args.errors).split(","):
            wrong = {json.loads(l)["sample_id"]
                     for l in open(path) if not json.loads(l)["correct"]}
            hardcore = wrong if hardcore is None else hardcore & wrong

    def bucket(indices):
        vals = [best[i] for i in indices if rare_of[i]]
        if not vals:
            return {}
        return {
            "n": len(vals),
            "full": sum(v >= 0.999 for v in vals),
            "ge80": sum(v >= 0.8 for v in vals),
            "ge50": sum(v >= 0.5 for v in vals),
            "lt20": sum(v < 0.2 for v in vals),
            "mean": round(sum(vals) / len(vals), 4),
        }

    by_subject = {}
    for subj in sorted({r["subject"] for r in rows}):
        by_subject[subj] = bucket([i for i, r in enumerate(rows)
                                   if r["subject"] == subj])
    report = {
        "sft": str(args.sft), "records": n, "max_df": args.max_df,
        "no_rare_terms": len(rows) - len(judgeable),
        "overall": bucket(range(len(rows))),
        "by_subject": by_subject,
    }
    if hardcore is not None:
        ids = {r["sample_id"]: i for i, r in enumerate(rows)}
        hc = [ids[s] for s in hardcore if s in ids]
        rest = [i for i in range(len(rows)) if rows[i]["sample_id"] not in hardcore]
        report["hardcore_errors"] = bucket(hc)
        report["rest"] = bucket(rest)
    per_row = [{"sample_id": rows[i]["sample_id"], "subject": rows[i]["subject"],
                "coverage": round(best[i], 4), "n_rare_terms": len(rare_of[i])}
               for i in range(len(rows))]
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out.with_suffix(".rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_row), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
