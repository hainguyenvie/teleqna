#!/usr/bin/env python3
"""ANSWER-KEY-DERIVED (restricted): windows whose single-window context flips >=1 base-wrong question to gold.
Used only for the A/B experiment arm "tier1gf" against the label-free keep set; also builds the keep-only
in-context eval set (all 10k, context = the question's kept windows) to measure the label-free filter's T1."""
import json, collections
from pathlib import Path
from transformers import AutoTokenizer
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TOK = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
def trunc(s, n):
    ids = TOK(s, add_special_tokens=False)["input_ids"]; return s if len(ids) <= n else TOK.decode(ids[:n])
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: (x.get("parsed") or "") for x in d if isinstance(x, dict) and "sample_id" in x}
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
gold = {q: chr(65 + r["answer"]) for q, r in test.items()}
base = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json"); single = L(R / "results/landscape/win_single_q3_8b_base_nothink512.json")
gf = set()
for k, p in single.items():
    q, w = k.split("|")
    if base[q] != gold[q] and p == gold[q]: gf.add(w)
(R / "data/restricted").mkdir(exist_ok=True)
json.dump(sorted(gf), open(R / "data/restricted/windows_goldfix.json", "w"))
print(f"gold-fix windows {len(gf)} -> data/restricted/windows_goldfix.json (ANSWER-KEY-DERIVED)")
keep = {json.loads(l)["win_id"] for l in open(R / "data/kit/windows_keep.jsonl", encoding="utf-8")}
q2w = collections.defaultdict(list)
for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
    w = json.loads(l)
    if w["win_id"] in keep:
        for q in w["for_q"]: q2w[q].append(w["text"])
n0 = 0
with open(R / "data/eval/keep_only.jsonl", "w", encoding="utf-8") as fh:
    for q, r in test.items():
        refs = q2w.get(q, [])
        if not refs: n0 += 1; fh.write(json.dumps(r, ensure_ascii=False) + "\n"); continue
        refs = refs[:8]; per = max(400, 12000 // len(refs)); refs = [trunc(x, per) for x in refs]
        body = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(refs))
        fh.write(json.dumps({**r, "question": f"{HEADER}\n\n{body}\n\n---\n\n{r['question']}", "n_ctx": len(refs)}, ensure_ascii=False) + "\n")
print(f"keep_only eval set: 10000 rows, {n0} with no kept window (plain prompt)")
