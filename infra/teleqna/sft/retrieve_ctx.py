#!/usr/bin/env python3
"""BM25 retrieval over the Tele-Data shards, emitting RAG-context eval sets.

Why this exists. The rare-term coverage metric this track has been steering by
turned out to be flat against accuracy: from coverage 0.5 to 0.99 the 122B
scores 79-80% regardless, and on Standards specifications - the subject the
whole corpus campaign targets - it is flat even at coverage 1.0. Vocabulary
overlap says the words are somewhere in the corpus; it does not say the fact
is, nor that anything can find it. This measures the end-to-end thing instead:
retrieve, put in context, score.

Two stages, because the shards are documents rather than passages. arxiv ships
one record per paper (~6k words) and standard one per specification, so a
document-level hit says only that the paper is relevant. Stage 1 ranks the
93,111 documents, stage 2 slides windows inside the survivors and ranks those.
A window is what fits in a prompt and what a generator would later be grounded
on, so it is the unit that has to be right.

Only terms that occur in some test question are indexed. Terms appearing in no
query contribute nothing to any BM25 score, so carrying them would cost memory
and time for arithmetic that is provably zero.

Emits, alongside the eval set, the one diagnostic the accuracy number cannot
give on its own: whether the gold answer string is present in the retrieved
context at all. Accuracy conflates "the corpus does not have it", "retrieval
missed it" and "the model had it and still chose wrong"; answer-in-context
separates the first two from the third.
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
    "when", "what", "which", "where", "how", "why", "who", "does", "did", "any",
    "all", "one", "two", "following", "above", "below", "used", "use", "using",
    "purpose", "main", "key", "type", "types", "based", "into", "such", "other",
}


def terms(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


def window_spans(nwords: int, size: int, stride: int):
    """Word-index spans, so a document is split once and never re-tokenised."""
    if nwords <= size:
        return [(0, nwords)]
    out = []
    for i in range(0, nwords - size + stride, stride):
        end = min(i + size, nwords)
        if end - i < size // 3:
            break
        out.append((i, end))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--corpus", action="append", required=True,
                    help="path to a shard jsonl (field `content`)")
    ap.add_argument("--out", required=True, help="eval jsonl with context")
    ap.add_argument("--report", required=True)
    ap.add_argument("-k", type=int, default=8, help="windows kept per question")
    ap.add_argument("--top-docs", type=int, default=20)
    ap.add_argument("--win", type=int, default=400)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--max-df-frac", type=float, default=0.25,
                    help="drop terms in more than this fraction of documents; "
                         "BM25 idf is near zero there and the postings are huge")
    ap.add_argument("--explanations",
                    help="TeleQnA.json. Adds each row's explanation to its "
                         "query. TEST-DERIVED - the explanation is answer-key "
                         "material, so a retriever using it cannot be part of "
                         "a submitted system; it measures how much of the gap "
                         "is query formulation rather than corpus or ranking.")
    ap.add_argument("--measure-only", action="store_true",
                    help="skip writing the eval set; only report hit rates")
    ap.add_argument("--with-options", action="store_true",
                    help="add option text to the query. OFF by default: the "
                         "gold option is part of the answer key, so a retriever "
                         "that reads it is doing test-informed retrieval")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    print(f"queries: {len(rows)}", flush=True)

    expl = {}
    if args.explanations:
        src = json.load(open(args.explanations, encoding="utf-8"))
        for v in src.values():
            expl[v["question"].strip()] = v.get("explanation", "")
        hit = sum(1 for r in rows if r["question"].strip() in expl)
        print(f"explanations aligned: {hit}/{len(rows)}", flush=True)

    qterms = []
    for r in rows:
        t = terms(r["question"])
        if expl:
            t += terms(expl.get(r["question"].strip(), ""))
        if args.with_options:
            for c in r["choices"]:
                t += terms(c)
        qterms.append(t)
    vocab = {t: i for i, t in enumerate({t for ts in qterms for t in ts})}
    print(f"query vocabulary: {len(vocab)} terms", flush=True)

    # ---- pass 1: document-level postings, restricted to the query vocabulary
    docs: list[str] = []
    postings = collections.defaultdict(list)          # term id -> [(doc, tf)]
    dl = []
    for path in args.corpus:
        for line in open(path, encoding="utf-8"):
            content = json.loads(line).get("content", "")
            d = len(docs)
            docs.append(content)
            tf = collections.Counter()
            n = 0
            for t in TOKEN.findall(content.lower()):
                n += 1
                i = vocab.get(t)
                if i is not None:
                    tf[i] += 1
            dl.append(n)
            for i, c in tf.items():
                postings[i].append((d, c))
            if len(docs) % 20000 == 0:
                print(f"  indexed {len(docs)} docs", flush=True)
    N = len(docs)
    avgdl = sum(dl) / N
    dl_arr = np.array(dl, dtype=np.float32)
    print(f"indexed {N} docs, avgdl={avgdl:.0f}", flush=True)

    k1, b = 1.5, 0.75
    idf, post_doc, post_tf = {}, {}, {}
    kept = 0
    for i, pl in postings.items():
        df = len(pl)
        if df == 0 or df > args.max_df_frac * N:
            continue
        idf[i] = math.log(1 + (N - df + 0.5) / (df + 0.5))
        post_doc[i] = np.fromiter((d for d, _ in pl), dtype=np.int32, count=df)
        post_tf[i] = np.fromiter((c for _, c in pl), dtype=np.float32, count=df)
        kept += 1
    print(f"postings kept: {kept}/{len(postings)} terms", flush=True)

    # ---- per-document window cache
    # The top-`top_docs` lists of 10,000 queries overlap heavily, so without a
    # cache the same paper is split and tokenised hundreds of times. Only the
    # vocabulary ids are cached, never the window strings: ids are a few dozen
    # ints per window, the text of every window of every candidate document is
    # gigabytes.
    _words: dict[int, list[str]] = {}
    _wins: dict[int, tuple] = {}

    def doc_words(d: int) -> list[str]:
        w = _words.get(d)
        if w is None:
            w = docs[d].split()
            _words[d] = w
        return w

    def doc_windows(d: int):
        got = _wins.get(d)
        if got is None:
            w = doc_words(d)
            spans = window_spans(len(w), args.win, args.stride)
            wsets = []
            for a, bnd in spans:
                ids = set()
                for tok in TOKEN.findall(" ".join(w[a:bnd]).lower()):
                    i = vocab.get(tok)
                    if i is not None and i in idf:
                        ids.add(i)
                wsets.append(ids)
            got = (spans, wsets)
            _wins[d] = got
        return got

    # ---- stage 1: rank documents, stage 2: rank windows inside them
    out_f = None if args.measure_only else open(args.out, "w", encoding="utf-8")
    stat = collections.Counter()
    ans_in_ctx_by_sub = collections.defaultdict(lambda: [0, 0])
    for qi, (row, qt) in enumerate(zip(rows, qterms)):
        score = np.zeros(N, dtype=np.float32)
        for t in set(qt):
            i = vocab.get(t)
            if i is None or i not in idf:
                continue
            d, tf = post_doc[i], post_tf[i]
            denom = tf + k1 * (1 - b + b * dl_arr[d] / avgdl)
            np.add.at(score, d, idf[i] * tf * (k1 + 1) / denom)
        top = np.argpartition(-score, min(args.top_docs, N - 1))[:args.top_docs]
        top = top[np.argsort(-score[top])]

        qids = {vocab[t] for t in set(qt) if t in vocab and vocab[t] in idf}
        cand = []
        for d in top:
            if score[d] <= 0:
                continue
            spans, wsets = doc_windows(d)
            for wi, ws in enumerate(wsets):
                hit = qids & ws
                if hit:
                    cand.append((sum(idf[i] for i in hit), d, wi))
        cand.sort(key=lambda x: -x[0])
        picked, seen = [], set()
        for _, d, wi in cand:
            spans, _ = doc_windows(d)
            a, bnd = spans[wi]
            w = " ".join(doc_words(d)[a:bnd])
            key = w[:120]
            if key in seen:
                continue
            seen.add(key)
            picked.append(w)
            if len(picked) >= args.k:
                break

        gold = row["choices"][int(row["answer"])]
        ctx_l = " ".join(picked).lower()
        gold_terms = set(terms(gold))
        in_ctx = bool(gold_terms) and len(gold_terms & set(TOKEN.findall(ctx_l))) \
            >= 0.8 * len(gold_terms)
        stat["answer_in_ctx"] += in_ctx
        stat["no_ctx"] += not picked
        sub = row.get("subject", "?")
        ans_in_ctx_by_sub[sub][0] += in_ctx
        ans_in_ctx_by_sub[sub][1] += 1

        block = "\n\n".join(f"[{i+1}] {w}" for i, w in enumerate(picked))
        new = dict(row)
        new["question"] = (
            "Reference material retrieved from the telecom literature. It may "
            "or may not contain the answer.\n\n" + block +
            "\n\n---\n\n" + row["question"]) if picked else row["question"]
        new["n_ctx"] = len(picked)
        if out_f is not None:
            out_f.write(json.dumps(new, ensure_ascii=False) + "\n")
        if (qi + 1) % 500 == 0:
            print(f"  retrieved {qi+1}/{len(rows)} "
                  f"(answer-in-ctx {stat['answer_in_ctx']/(qi+1)*100:.1f}%)",
                  flush=True)
    if out_f is not None:
        out_f.close()

    rep = {
        "test": args.test, "corpus": args.corpus, "k": args.k,
        "top_docs": args.top_docs, "win": args.win, "stride": args.stride,
        "with_options": args.with_options,
        "explanations": bool(args.explanations), "n": len(rows),
        "answer_in_ctx": stat["answer_in_ctx"],
        "answer_in_ctx_frac": round(stat["answer_in_ctx"] / len(rows), 4),
        "rows_with_no_context": stat["no_ctx"],
        "by_subject": {s: {"n": v[1], "answer_in_ctx": v[0],
                           "frac": round(v[0] / v[1], 4)}
                       for s, v in sorted(ans_in_ctx_by_sub.items())},
    }
    Path(args.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
