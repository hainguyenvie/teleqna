#!/usr/bin/env python3
"""Dense re-ranking of the BM25 candidate pool, measured at a deployable k.

The job is re-ranking, not searching. BM25 with query expansion and top-50
documents already pulls the answer into a 32-window pool for 83% of rows, but
that pool is ~12,800 words - past the context window once a thinking budget is
added, so it cannot be served. Meanwhile the same retriever at the deployable
k=8 reaches 69.7%, against a measured single-window ceiling of 82.7%. That
13-point gap is entirely a ranking problem inside an already-good pool, which
is where a dense encoder is strongest and where its weakness - rare technical
identifiers it has never tokenised well - matters least, because BM25 has
already used them to build the pool.

Scored at k=8 against the k=8 BM25 arm, never against the k=32 number: a
32-window context trivially contains more terms, so comparing across k measures
context size rather than ranking quality.

Manual last-token pooling rather than sentence-transformers, which is not in
this venv and whose install would touch the transformers pin the vLLM nightly
depends on. Left padding puts the last real token at position -1 for every row
in the batch, so pooling is a slice instead of a gather.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

import torch

BLOCK = re.compile(r"^\[(\d+)\]\s(.*)$", re.MULTILINE)
SEP = "\n\n---\n\n"
QPROMPT = ("Instruct: Given a query, retrieve documents that answer the query "
           "\nQuery: ")
TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
STOP = {"the", "and", "for", "that", "with", "this", "are", "was", "which",
        "from", "has", "have", "not", "can", "may", "shall", "will", "its",
        "their", "than", "when", "what", "where", "how", "why", "who", "does",
        "did", "any", "all", "one", "two", "following", "above", "below",
        "used", "use", "using", "such", "purpose", "main", "key", "type",
        "types", "based", "into", "other"}


def terms(t):
    return {x for x in TOKEN.findall((t or "").lower()) if x not in STOP}


@torch.no_grad()
def embed(texts, tok, model, bs, maxlen, device, tag=""):
    out = torch.empty(len(texts), model.config.hidden_size,
                      dtype=torch.float32, device="cpu")
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=maxlen,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state[:, -1]      # left padding => last real token
        out[i:i + bs] = torch.nn.functional.normalize(h.float(), dim=-1).cpu()
        if (i // bs) % 50 == 0:
            print(f"  {tag} {i+len(batch)}/{len(texts)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", required=True, help="jsonl with the k=32 pool inlined")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--report", required=True)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=640)
    ap.add_argument("--explanations", help="TeleQnA.json; expands the dense query "
                                           "the same way the BM25 arm was expanded")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rag, encoding="utf-8")]
    if a.sample:
        rows.sort(key=lambda r: hashlib.sha256(
            ("dense|" + r["sample_id"]).encode()).hexdigest())
        rows = rows[:a.sample]

    expl = {}
    if a.explanations:
        for v in json.load(open(a.explanations, encoding="utf-8")).values():
            expl[v["question"].strip()] = v.get("explanation", "")

    # split the inlined pool back into candidates, dedup globally
    pool, cand_ids, queries = {}, [], []
    for r in rows:
        parts = r["question"].split(SEP)
        raw_q = parts[-1]
        blocks = [b.strip() for _, b in BLOCK.findall(parts[0])] if len(parts) > 1 else []
        ids = []
        for b in blocks:
            cid = hashlib.sha256(b.encode()).hexdigest()[:16]
            pool.setdefault(cid, b)
            ids.append(cid)
        cand_ids.append(ids)
        q = raw_q
        if expl:
            e = expl.get(raw_q.strip(), "")
            if e:
                q = raw_q + "\n" + e
        queries.append(QPROMPT + q)
    keys = list(pool)
    print(f"rows={len(rows)} unique candidates={len(keys)} "
          f"mean pool={sum(len(c) for c in cand_ids)/len(rows):.1f}", flush=True)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(a.model, local_files_only=True,
                                      dtype=torch.bfloat16).eval().cuda()
    dev = next(model.parameters()).device

    demb = embed([pool[k] for k in keys], tok, model, a.batch, a.maxlen, dev, "doc")
    qemb = embed(queries, tok, model, a.batch, a.maxlen, dev, "qry")
    pos = {k: i for i, k in enumerate(keys)}

    out_f = open(a.out, "w", encoding="utf-8") if a.out else None
    stat = collections.Counter()
    bysub = collections.defaultdict(lambda: [0, 0])
    for qi, (r, ids) in enumerate(zip(rows, cand_ids)):
        if not ids:
            picked = []
        else:
            idx = torch.tensor([pos[c] for c in ids])
            sim = demb[idx] @ qemb[qi]
            order = torch.argsort(sim, descending=True)[:a.k]
            picked = [pool[ids[j]] for j in order.tolist()]

        gold = r["choices"][int(r["answer"])]
        gt = terms(gold)
        ctx = set(TOKEN.findall(" ".join(picked).lower()))
        ok = bool(gt) and len(gt & ctx) >= 0.8 * len(gt)
        stat["hit"] += ok
        stat["n"] += 1
        s = r.get("subject", "?")
        bysub[s][0] += int(ok)
        bysub[s][1] += 1

        if out_f:
            block = "\n\n".join(f"[{i+1}] {w}" for i, w in enumerate(picked))
            new = {kk: vv for kk, vv in r.items() if kk != "n_ctx"}
            raw_q = r["question"].split(SEP)[-1]
            new["question"] = ("Reference material retrieved from the telecom "
                               "literature. It may or may not contain the "
                               "answer.\n\n" + block + SEP + raw_q) if picked else raw_q
            new["n_ctx"] = len(picked)
            out_f.write(json.dumps(new, ensure_ascii=False) + "\n")
    if out_f:
        out_f.close()

    rep = {"rag": a.rag, "model": a.model, "k": a.k, "sample": len(rows),
           "explanations": bool(a.explanations),
           "unique_candidates": len(keys),
           "answer_in_ctx": stat["hit"],
           "answer_in_ctx_frac": round(stat["hit"] / stat["n"], 4),
           "by_subject": {s: {"n": v[1], "hit": v[0], "frac": round(v[0]/v[1], 4)}
                          for s, v in sorted(bysub.items())}}
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
