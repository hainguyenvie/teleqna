#!/usr/bin/env python3
"""Turn the Wikipedia pages fetched for the uncovered questions (data/kit/apisearch.jsonl) into windows: split each page
into ~250-word chunks, BM25 (bm25s) over that question's own chunks with question + options as query (never the
answer), keep top-8 -> data/kit/tb_api_windows.jsonl and an 8B eval set data/eval/tb_api8.jsonl (RAG format, 12k budget)."""
import json, re, collections
from pathlib import Path
import bm25s
from transformers import AutoTokenizer
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
pages = collections.defaultdict(list)
for l in open(R / "data/kit/apisearch.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["text"]: pages[r["sample_id"]].append(r)
qtok = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
def trunc(s, n):
    ids = qtok(s, add_special_tokens=False)["input_ids"]; return s if len(ids) <= n else qtok.decode(ids[:n])
def chunks(text, n=250):
    w = text.split(); return [" ".join(w[i:i + n]) for i in range(0, len(w), n) if len(w[i:i + n]) > 40]
nw = 0; nq = 0
with open(R / "data/kit/tb_api_windows.jsonl", "w", encoding="utf-8") as fw, open(R / "data/eval/tb_api8.jsonl", "w", encoding="utf-8") as fe:
    for sid, ps in pages.items():
        q = test[sid]; texts, srcs = [], []
        for p in ps:
            for c in chunks(p["text"]): texts.append(f"[{p['title']}] {c}"); srcs.append(p["title"])
        if not texts: continue
        retriever = bm25s.BM25(); retriever.index(bm25s.tokenize(texts, stopwords="en", show_progress=False), show_progress=False)
        query = bm25s.tokenize([q["question"] + " " + " ".join(q["choices"])], stopwords="en", show_progress=False)
        res, sc = retriever.retrieve(query, k=min(8, len(texts)), show_progress=False)
        top = [int(i) for i in res[0]]
        refs = [trunc(texts[i], 12000 // 8) for i in top]
        for i in top: fw.write(json.dumps(dict(win_id=f"ta{nw:06d}", text=texts[i], for_q=[sid], src="wikipedia:" + srcs[i]), ensure_ascii=False) + "\n"); nw += 1
        body = "\n\n".join(f"[{k+1}] {t}" for k, t in enumerate(refs))
        fe.write(json.dumps({**q, "question": f"{HEADER}\n\n{body}\n\n---\n\n{q['question']}", "n_ctx": len(refs)}, ensure_ascii=False) + "\n"); nq += 1
print(f"api windows: {nq} questions -> {nw} windows"); print("API_WINDOWS_DONE")
