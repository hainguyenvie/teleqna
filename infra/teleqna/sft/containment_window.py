#!/usr/bin/env python3
"""Containment at window level, to tighten the document-level 89.1% ceiling.

The document-level measurement asks whether some document in the corpus carries
80% of a row's gold-answer terms. arxiv documents are whole papers of ~6,000
words, so that test can be passed by terms scattered across pages that never
state the fact together - it is an upper bound, and a loose one.

A 400-word window is the unit a prompt actually carries and the unit a
generator would be grounded on, so window-level containment is the honest
ceiling: if no single window holds the answer's terms, no retriever can put the
answer in front of the model in one piece.

Sampled rather than exhaustive. The corpus has ~1.5M windows and the answer is
a rate, not a per-row fact, so 1,000 rows gives +/-3pp at 95% - far tighter than
the 20-point gaps this number is being used to arbitrate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
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
    "purpose", "main", "key", "type", "types", "based", "into", "other",
}


def terms(t: str) -> set[str]:
    return {x for x in TOKEN.findall((t or "").lower()) if x not in STOP}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--win", type=int, default=400)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--thr", type=float, default=0.8)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.test, encoding="utf-8")]
    rows.sort(key=lambda r: hashlib.sha256(
        ("win-cont|" + r["sample_id"]).encode()).hexdigest())
    rows = rows[:a.sample]

    tsets, vocab = [], {}
    for r in rows:
        t = terms(r["choices"][int(r["answer"])])
        tsets.append(t)
        for x in t:
            vocab.setdefault(x, len(vocab))
    q_len = np.array([max(1, len(t)) for t in tsets], dtype=np.float32)

    # term id -> the sampled queries that need it
    t2q = collections.defaultdict(list)
    for qi, t in enumerate(tsets):
        for x in t:
            t2q[vocab[x]].append(qi)
    t2q = {k: np.array(v, dtype=np.int32) for k, v in t2q.items()}
    print(f"rows={len(rows)} vocab={len(vocab)}", flush=True)

    best = np.zeros(len(rows), dtype=np.float32)
    acc = np.zeros(len(rows), dtype=np.float32)
    nwin = ndoc = 0
    for path in a.corpus:
        for line in open(path, encoding="utf-8"):
            words = json.loads(line).get("content", "").split()
            ndoc += 1
            n = len(words)
            starts = range(0, max(1, n - a.win + a.stride), a.stride)
            for s in starts:
                chunk = words[s:s + a.win]
                if len(chunk) < a.win // 3 and s:
                    break
                ids = set()
                for tok in TOKEN.findall(" ".join(chunk).lower()):
                    i = vocab.get(tok)
                    if i is not None:
                        ids.add(i)
                nwin += 1
                if not ids:
                    continue
                touched = []
                for i in ids:
                    q = t2q.get(i)
                    if q is not None:
                        acc[q] += 1
                        touched.append(q)
                if touched:
                    idx = np.unique(np.concatenate(touched))
                    best[idx] = np.maximum(best[idx], acc[idx] / q_len[idx])
                    acc[idx] = 0
            if ndoc % 10000 == 0:
                print(f"  {ndoc} docs, {nwin} windows, "
                      f"hit={float((best >= a.thr).mean())*100:.1f}%", flush=True)

    hit = best >= a.thr
    bysub = collections.defaultdict(lambda: [0, 0])
    for r, h in zip(rows, hit):
        s = r.get("subject", "?")
        bysub[s][0] += int(h)
        bysub[s][1] += 1
    rep = {"corpus": a.corpus, "docs": ndoc, "windows": nwin,
           "sample": len(rows), "win": a.win, "stride": a.stride, "thr": a.thr,
           "window_containment": round(float(hit.mean()), 4),
           "mean_best_coverage": round(float(best.mean()), 4),
           "by_subject": {s: {"n": v[1], "hit": v[0],
                              "frac": round(v[0] / v[1], 4)}
                          for s, v in sorted(bysub.items())}}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
