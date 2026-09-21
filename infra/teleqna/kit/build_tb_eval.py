#!/usr/bin/env python3
"""Functional coverage eval set: for each question, its top-8 trace-back windows (new corpora) in the harness
RAG format, 12k-token budget. Scored by eval_dev_vllm; union with RAG-8 = functional coverage."""
import json, collections
from pathlib import Path
from transformers import AutoTokenizer
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TOK = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
def trunc(s, n):
    ids = TOK(s, add_special_tokens=False)["input_ids"]; return s if len(ids) <= n else TOK.decode(ids[:n])
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
q2w = collections.defaultdict(list)
for l in open(R / "data/kit/tb_windows.jsonl", encoding="utf-8"):
    w = json.loads(l)
    for q in w["for_q"]: q2w[q].append(w["text"])
n0 = 0
with open(R / "data/eval/tb_rag8.jsonl", "w", encoding="utf-8") as fh:
    for q, r in test.items():
        refs = q2w.get(q, [])[:8]
        if not refs: n0 += 1; fh.write(json.dumps(r, ensure_ascii=False) + "\n"); continue
        per = max(400, 12000 // len(refs)); refs = [trunc(x, per) for x in refs]
        body = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(refs))
        fh.write(json.dumps({**r, "question": f"{HEADER}\n\n{body}\n\n---\n\n{r['question']}", "n_ctx": len(refs)}, ensure_ascii=False) + "\n")
print(f"tb_rag8 eval set: 10000 rows, {n0} without windows"); print("TB_EVAL_BUILT")
