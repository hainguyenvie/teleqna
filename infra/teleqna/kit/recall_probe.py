#!/usr/bin/env python3
"""Recall probe on HELD-OUT self-contained QA about trained windows: closed-book short answer, scored by token-F1 >= 0.5
or containment against the gated span. Compares any checkpoints. Usage: recall_probe.py --ckpt PATH --tag TAG [--n 3000]"""
import json, re, random, argparse, unicodedata, glob, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def norm(s): return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower())).strip()
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--holdout", default="data/kit/recall/holdout_s*.jsonl"); a = ap.parse_args()
rows = [json.loads(l) for f in sorted(glob.glob(str(R / a.holdout))) for l in open(f, encoding="utf-8")]; random.Random(0).shuffle(rows); rows = rows[:a.n]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True); llm = LLM(model=a.ckpt, gpu_memory_utilization=a.gpu_mem, max_model_len=2048, dtype="bfloat16")
P = "Answer the telecom question with a short phrase only (no explanation).\n\nQuestion: {q}"
outs = llm.generate([tok.apply_chat_template([{"role": "user", "content": P.format(q=r["prompt"])}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows], SamplingParams(temperature=0.0, max_tokens=32))
def f1(a, b):
    A, B = norm(a).split(), norm(b).split()
    if not A or not B: return 0.0
    c = sum((collections.Counter(A) & collections.Counter(B)).values()); return 0.0 if c == 0 else 2 * c / (len(A) + len(B))
hit = 0; ex = []
for r, o in zip(rows, outs):
    ans = o.outputs[0].text.strip(); ok = norm(r["completion"]) in norm(ans) or f1(r["completion"], ans) >= 0.5; hit += ok
    if len(ex) < 6: ex.append((r["prompt"][:80], r["completion"][:30], ans[:40], ok))
print(f"{a.tag}: held-out self-contained QA recall {hit/len(rows)*100:.1f}% (n={len(rows)})")
for e in ex: print("   ", e)
print("RECALL_PROBE_DONE")
