#!/usr/bin/env python3
"""Tier-1 corpus: the strong-RAG windows that change the model's closed-book letter for at least one
question (label-free; win_single screen). Writes data/kit/windows_keep.jsonl with n_change per window."""
import json, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: (x.get("parsed") or "") for x in d if isinstance(x, dict) and "sample_id" in x}
base = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json")
single = L(R / "results/landscape/win_single_q3_8b_base_nothink512.json")
chg = collections.Counter(); seen = collections.Counter()
for k, p in single.items():
    q, w = k.split("|"); seen[w] += 1
    if p != base[q]: chg[w] += 1
n = kept = 0; toks = 0
with open(R / "data/kit/windows_keep.jsonl", "w", encoding="utf-8") as fh:
    for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
        w = json.loads(l); n += 1
        if chg[w["win_id"]] == 0: continue
        w["n_change"] = chg[w["win_id"]]; w["n_q"] = seen[w["win_id"]]; kept += 1; toks += int(len(w["text"].split()) * 1.35)
        fh.write(json.dumps(w, ensure_ascii=False) + "\n")
print(f"windows {n:,} kept {kept:,} ({kept/n*100:.1f}%)  ~{toks/1e6:.1f}M source tokens -> data/kit/windows_keep.jsonl")
