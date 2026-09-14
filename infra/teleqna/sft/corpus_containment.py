#!/usr/bin/env python3
"""Does the corpus contain the answer at all, independently of any retriever?

The retrieved answer-in-context number (37.2% at k=8) confounds three different
failures and prescribes three different fixes:

  (a) the corpus does not carry the fact   -> go find more sources
  (b) BM25 carries it but did not rank it  -> better retriever (dense, hybrid)
  (c) it is there and was retrieved, but   -> the metric is wrong, not the corpus
      phrased differently

This measures the retriever-free upper bound: over EVERY document in the corpus
(not the top-8), is there one that carries the answer's rare terms? Anything
above the retrieved number is (b) and is bought with engineering; anything the
upper bound itself is missing is (a) and can only be bought with more corpus.

Document level rather than window level on purpose. It is an upper bound - if
no document contains the terms then no window inside one does - and computing
it needs a single pass instead of a pass per window size.

Two targets, because they answer different questions. The gold OPTION text is
what the model has to produce. The EXPLANATION is TeleQnA's own statement of
why that option is right, so it names the fact rather than the phrasing, and it
is the closer proxy for "the source document is in here somewhere".
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
STOP = {
    "the", "and", "for", "that", "with", "this", "are", "was", "which", "from",
    "has", "have", "not", "can", "may", "shall", "will", "its", "their", "than",
    "when", "what", "where", "how", "why", "who", "does", "did", "any", "all",
    "one", "two", "following", "above", "below", "used", "use", "using", "such",
    "purpose", "main", "key", "type", "types", "based", "into", "other", "correct",
    "answer", "option", "because", "refers", "statement", "true", "false",
}


def terms(text: str) -> set[str]:
    return {t for t in TOKEN.findall((text or "").lower()) if t not in STOP}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--teleqna", required=True, help="TeleQnA.json, for explanations")
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.8)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.test, encoding="utf-8")]
    src = json.load(open(a.teleqna, encoding="utf-8"))
    expl = {}
    for v in src.values():
        expl[v["question"].strip()] = v.get("explanation", "")
    matched = sum(1 for r in rows if r["question"].strip() in expl)
    print(f"explanations aligned by question string: {matched}/{len(rows)}")

    targets = []      # (answer_terms, explanation_terms)
    for r in rows:
        gold = r["choices"][int(r["answer"])]
        targets.append((terms(gold), terms(expl.get(r["question"].strip(), ""))))

    vocab = {}
    for at, et in targets:
        for t in at | et:
            vocab.setdefault(t, len(vocab))
    print(f"target vocabulary: {len(vocab)} terms", flush=True)

    postings = collections.defaultdict(list)
    ndocs = 0
    for path in a.corpus:
        for line in open(path, encoding="utf-8"):
            content = json.loads(line).get("content", "")
            seen = set()
            for t in TOKEN.findall(content.lower()):
                i = vocab.get(t)
                if i is not None:
                    seen.add(i)
            for i in seen:
                postings[i].append(ndocs)
            ndocs += 1
            if ndocs % 20000 == 0:
                print(f"  scanned {ndocs} docs", flush=True)
    print(f"scanned {ndocs} docs", flush=True)

    post = {i: np.array(v, dtype=np.int32) for i, v in postings.items()}

    def best_cov(tset):
        if not tset:
            return None
        acc = np.zeros(ndocs, dtype=np.int16)
        for t in tset:
            i = vocab.get(t)
            p = post.get(i)
            if p is not None:
                acc[p] += 1
        return float(acc.max()) / len(tset)

    stat = collections.defaultdict(lambda: collections.Counter())
    per_row = []
    for r, (at, et) in zip(rows, targets):
        ca, ce = best_cov(at), best_cov(et)
        s = r.get("subject", "?")
        stat[s]["n"] += 1
        stat[s]["ans"] += (ca is not None and ca >= a.thr)
        stat[s]["expl"] += (ce is not None and ce >= a.thr)
        stat[s]["no_expl"] += (ce is None)
        per_row.append({"sample_id": r["sample_id"], "subject": s,
                        "answer_cov": None if ca is None else round(ca, 3),
                        "expl_cov": None if ce is None else round(ce, 3)})
        if len(per_row) % 1000 == 0:
            print(f"  scored {len(per_row)}/{len(rows)}", flush=True)

    tot = collections.Counter()
    print(f"\n{'subject':28s} {'n':>5s} {'gold-answer':>12s} {'explanation':>12s}")
    for s in sorted(stat, key=lambda k: -stat[k]["n"]):
        c = stat[s]; tot.update(c)
        print(f"{s:28s} {c['n']:5d} {c['ans']/c['n']*100:11.1f}% "
              f"{c['expl']/c['n']*100:11.1f}%")
    print(f"{'TOTAL':28s} {tot['n']:5d} {tot['ans']/tot['n']*100:11.1f}% "
          f"{tot['expl']/tot['n']*100:11.1f}%")

    Path(a.out).write_text(json.dumps({
        "corpus": a.corpus, "docs": ndocs, "thr": a.thr, "n": len(rows),
        "answer_in_corpus": tot["ans"], "answer_frac": round(tot["ans"]/tot["n"], 4),
        "explanation_in_corpus": tot["expl"],
        "explanation_frac": round(tot["expl"]/tot["n"], 4),
        "by_subject": {s: {"n": c["n"], "answer": c["ans"], "explanation": c["expl"]}
                       for s, c in sorted(stat.items())},
        "rows": per_row,
    }, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
