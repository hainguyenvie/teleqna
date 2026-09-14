#!/usr/bin/env python3
"""Retrieve over distilled QA pairs instead of over raw corpus windows.

Why this is a different problem from the corpus retriever, and an easier one.
BM25 over the shards has to match a question against a 400-word slab of a paper
- different register, different length, the fact diluted across a page. The QA
corpora are already the fact, at ~27 words, phrased as a question. Matching a
question to a question is a far better-posed lexical problem than matching a
question to a paragraph, and the answer sits in the retrieved unit rather than
somewhere inside it.

The index deliberately keeps the pairs that restate benchmark rows verbatim.
That is the user's call and it is what a deployed system would see, but it also
makes the headline hit rate uninformative on its own: a row whose own question
is in the index retrieves its own answer and proves nothing about the method.
So every number is reported twice - over all rows, and over only the rows whose
question does NOT appear verbatim in the index. The second column is the one
that says whether this generalises.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
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
norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())


def terms(text: str) -> list[str]:
    return [t for t in TOKEN.findall((text or "").lower()) if t not in STOP]


def load_pairs(paths):
    for p in paths:
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            t = pq.read_table(p)
            qs = t.column("question").to_pylist()
            ans = t.column("answer").to_pylist()
            for q, a in zip(qs, ans):
                if q and a:
                    yield str(q), str(a)
        else:
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                q = r.get("question") or r.get("input")
                a = r.get("answer") or r.get("output")
                if q and a:
                    yield str(q), str(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--pairs", action="append", required=True)
    ap.add_argument("--out")
    ap.add_argument("--report", required=True)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--explanations", help="TeleQnA.json, to expand the query")
    ap.add_argument("--max-df-frac", type=float, default=0.25)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.test, encoding="utf-8")]
    expl = {}
    if a.explanations:
        for v in json.load(open(a.explanations, encoding="utf-8")).values():
            expl[v["question"].strip()] = v.get("explanation", "")

    qterms = []
    for r in rows:
        t = terms(r["question"])
        if expl:
            t += terms(expl.get(r["question"].strip(), ""))
        qterms.append(t)
    vocab = {t: i for i, t in enumerate({t for ts in qterms for t in ts})}
    print(f"queries={len(rows)} query vocabulary={len(vocab)}", flush=True)

    # index
    texts, postings, dl = [], collections.defaultdict(list), []
    verbatim = set()
    tnorm = {norm(r["question"]): r["sample_id"] for r in rows}
    tnotag = {norm(re.sub(r"\[[^\]]+\]", "", r["question"])): r["sample_id"] for r in rows}
    for q, ans in load_pairs(a.pairs):
        sid = tnorm.get(norm(q)) or tnotag.get(norm(q))
        if sid:
            verbatim.add(sid)
        body = f"Q: {q}\nA: {ans}"
        d = len(texts)
        texts.append(body)
        tf = collections.Counter()
        n = 0
        for t in TOKEN.findall(body.lower()):
            n += 1
            i = vocab.get(t)
            if i is not None:
                tf[i] += 1
        dl.append(n)
        for i, c in tf.items():
            postings[i].append((d, c))
        if len(texts) % 100000 == 0:
            print(f"  indexed {len(texts)} pairs", flush=True)
    N = len(texts)
    avgdl = sum(dl) / N
    dl_arr = np.array(dl, dtype=np.float32)
    print(f"indexed {N} pairs, avgdl={avgdl:.0f}; "
          f"{len(verbatim)}/{len(rows)} benchmark rows appear verbatim", flush=True)

    k1, b = 1.5, 0.75
    idf, pdoc, ptf = {}, {}, {}
    for i, pl in postings.items():
        df = len(pl)
        if df == 0 or df > a.max_df_frac * N:
            continue
        idf[i] = math.log(1 + (N - df + 0.5) / (df + 0.5))
        pdoc[i] = np.fromiter((d for d, _ in pl), dtype=np.int32, count=df)
        ptf[i] = np.fromiter((c for _, c in pl), dtype=np.float32, count=df)
    print(f"postings kept: {len(idf)}", flush=True)

    out_f = open(a.out, "w", encoding="utf-8") if a.out else None
    stat = collections.defaultdict(lambda: collections.Counter())
    for qi, (r, qt) in enumerate(zip(rows, qterms)):
        score = np.zeros(N, dtype=np.float32)
        for t in set(qt):
            i = vocab.get(t)
            if i is None or i not in idf:
                continue
            d, tf = pdoc[i], ptf[i]
            score_add = idf[i] * tf * (k1 + 1) / (
                tf + k1 * (1 - b + b * dl_arr[d] / avgdl))
            np.add.at(score, d, score_add)
        top = np.argpartition(-score, min(a.k, N - 1))[:a.k]
        top = top[np.argsort(-score[top])]
        picked = [texts[d] for d in top if score[d] > 0]

        gold = r["choices"][int(r["answer"])]
        gt = set(terms(gold))
        ctx = set(TOKEN.findall(" ".join(picked).lower()))
        ok = bool(gt) and len(gt & ctx) >= 0.8 * len(gt)

        grp = "verbatim" if r["sample_id"] in verbatim else "held"
        for key in (grp, "all"):
            stat[key]["n"] += 1
            stat[key]["hit"] += ok
        stat[r.get("subject", "?") + ("" if grp == "held" else " *")]["n"] += 0

        if out_f:
            block = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(picked))
            new = dict(r)
            new["question"] = ("Reference question-answer pairs retrieved from "
                               "the telecom literature. They may or may not "
                               "cover this question.\n\n" + block +
                               "\n\n---\n\n" + r["question"]) if picked else r["question"]
            new["n_ctx"] = len(picked)
            out_f.write(json.dumps(new, ensure_ascii=False) + "\n")
        if (qi + 1) % 1000 == 0:
            print(f"  scored {qi+1}/{len(rows)} "
                  f"(all {stat['all']['hit']/stat['all']['n']*100:.1f}%)", flush=True)
    if out_f:
        out_f.close()

    # per-subject, held rows only - the generalising half
    bysub = collections.defaultdict(lambda: [0, 0])
    rep = {"pairs": a.pairs, "k": a.k, "explanations": bool(a.explanations),
           "indexed_pairs": N, "rows_verbatim_in_index": len(verbatim)}
    for key in ("all", "verbatim", "held"):
        c = stat[key]
        if c["n"]:
            rep[key] = {"n": c["n"], "answer_in_ctx": c["hit"],
                        "frac": round(c["hit"] / c["n"], 4)}
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
