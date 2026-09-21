#!/usr/bin/env python3
"""Deep retrieval for the functionally-uncovered questions: rebuild (or load) the BM25 index over the filtered
store, retrieve top-100 for those questions with two query forms (question+options; question only), save
candidates to data/kit/tb_deep_candidates.jsonl for dense reranking on a GPU."""
import json, collections
from pathlib import Path
import bm25s
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
unc = [json.loads(l)["sample_id"] for l in open(R / "data/kit/uncovered_functional.jsonl", encoding="utf-8")]
docs = []; meta = []
for l in open(R / "data/kit/tb_chunks.jsonl", encoding="utf-8"):
    r = json.loads(l); docs.append(r["text"]); meta.append((r["src"], r["doc"]))
idx = R / "data/kit/bm25_index"
if idx.exists():
    retr = bm25s.BM25.load(str(idx)); print("loaded index", flush=True)
else:
    print(f"indexing {len(docs):,}", flush=True); retr = bm25s.BM25(); retr.index(bm25s.tokenize(docs, stopwords="en")); retr.save(str(idx)); print("saved index", flush=True)
K = 100; out = {}
for form, qf in (("qo", lambda r: r["question"] + " " + " ".join(r["choices"])), ("q", lambda r: r["question"])):
    res, sc = retr.retrieve(bm25s.tokenize([qf(test[q]) for q in unc], stopwords="en"), k=K)
    for q, ids in zip(unc, res): out.setdefault(q, set()).update(int(i) for i in ids)
with open(R / "data/kit/tb_deep_candidates.jsonl", "w", encoding="utf-8") as fh:
    for q, ids in out.items():
        fh.write(json.dumps(dict(sample_id=q, cands=[dict(i=i, src=meta[i][0], text=docs[i]) for i in sorted(ids)]), ensure_ascii=False) + "\n")
print(f"candidates for {len(out)} questions, mean {sum(len(v) for v in out.values())/len(out):.0f} each"); print("DEEP_DONE")
