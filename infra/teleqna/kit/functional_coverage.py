#!/usr/bin/env python3
"""Functional coverage by tag/subject: RAG-8 (old windows), TB-8 (new windows), union; and for base-wrong rows."""
import json, re, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: x for x in d if isinstance(x, dict) and "sample_id" in x}
B = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json"); RG = L(R / "results/landscape/ragstrong_q3_8b_base_nothink512.json"); TB = L(R / "results/landscape/tb_rag8_q3_8b_base_nothink512.json")
def tag(q):
    m = re.search(r"\[([^\]]+)\]\s*$", test[q]["question"]); s = m.group(1) if m else "untagged"; return "3GPP" if "3GPP" in s else s
ids = list(test)
def row(n, I): return f"  {n:34s} n={len(I):5d}  RAG-8 {sum(RG[q]['correct'] for q in I)/len(I)*100:5.1f}%  TB-8 {sum(TB[q]['correct'] for q in I)/len(I)*100:5.1f}%  UNION {sum(RG[q]['correct'] or TB[q]['correct'] for q in I)/len(I)*100:5.1f}%"
print("functional coverage (some window set makes Qwen3-8B answer correctly):"); print(row("ALL 10k", ids))
print(row("base-right", [q for q in ids if B[q]["correct"]])); print(row("base-wrong", [q for q in ids if not B[q]["correct"]]))
for t in sorted(set(map(tag, ids))): print(row("tag " + t, [q for q in ids if tag(q) == t]))
for s in sorted(set(test[q]["subject"] for q in ids)): print(row("subj " + s, [q for q in ids if test[q]["subject"] == s]))
unc = [q for q in ids if not (RG[q]["correct"] or TB[q]["correct"])]
with open(R / "data/kit/uncovered_functional.jsonl", "w", encoding="utf-8") as fh:
    for q in unc: fh.write(json.dumps(dict(sample_id=q, tag=tag(q), subject=test[q]["subject"], base_correct=bool(B[q]["correct"]), question=test[q]["question"], gold=test[q]["choices"][test[q]["answer"]]), ensure_ascii=False) + "\n")
print(f"functionally uncovered: {len(unc)} ({len(unc)/100:.1f}%) -> data/kit/uncovered_functional.jsonl; by tag {collections.Counter(tag(q) for q in unc).most_common(8)}")
print("FUNCTIONAL_COVERAGE_DONE")
