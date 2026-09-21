#!/usr/bin/env python3
"""Hallucination audit of the study-kit (training data) against its source windows — no test labels involved.
Samples rows per view (facts, factview, qa, mcq, cmcq, register) and asks a judge model whether the excerpt supports
the statement (SUPPORTED / PARTIAL / NOT SUPPORTED). Reports the rate per view. Usage: kit_audit.py --judge PATH [--n 150]"""
import json, glob, random, argparse, collections, re
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
ap = argparse.ArgumentParser(); ap.add_argument("--judge", required=True); ap.add_argument("--n", type=int, default=150); ap.add_argument("--gpu-mem", type=float, default=0.28); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rng = random.Random(a.seed)
W = {}
for f in ("data/kit/windows_keep.jsonl", "data/kit/windows_rest.jsonl"):
    for l in open(R / f, encoding="utf-8"): w = json.loads(l); W[w["win_id"]] = w["text"]
pool = collections.defaultdict(list)
def add(view, win, stmt, src):
    if win in W and len(pool[view]) < 20000: pool[view].append((win, stmt, src))
for f in glob.glob(str(R / "data/kit/tier1/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier1c/views_s*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if rng.random() > 0.05: continue
        r = json.loads(l)
        if not r.get("ok", True): continue
        v = r["view"]; t = r["text"]
        try: m = json.loads(t) if v in ("facts", "qa", "mcq") else None
        except Exception: continue
        if v == "facts": add("facts", r["win_id"], m.get("fact", ""), f)
        elif v == "factview": add("factview", r["win_id"], t, f)
        elif v == "qa": add("qa", r["win_id"], f"Q: {m.get('q','')} A: {m.get('a','')}", f)
        elif v == "mcq": add("cmcq" if "tier1c" in f else "mcq", r["win_id"], f"Q: {m.get('q','')} Correct answer: {m['options'][m['answer']] if isinstance(m.get('options'), list) and isinstance(m.get('answer'), int) and m['answer'] < len(m['options']) else ''}", f)
        elif v == "register": add("register", r["win_id"], t[:1200], f)
print({k: len(v) for k, v in pool.items()}, flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.judge, local_files_only=True); llm = LLM(model=a.judge, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16")
P = ("EXCERPT:\n{doc}\n\nSTATEMENT:\n{stmt}\n\nIs every factual claim in the STATEMENT supported by the EXCERPT (telecom technical text)? "
     "Reply with exactly one word: SUPPORTED, PARTIAL (some claims supported, others not stated), or NOT_SUPPORTED (contradicted or invented).")
items, prompts = [], []
for v, rows in pool.items():
    for win, stmt, src in rng.sample(rows, min(a.n, len(rows))):
        items.append((v, win, stmt)); prompts.append(tok.apply_chat_template([{"role": "user", "content": P.format(doc=W[win][:6000], stmt=stmt)}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))
res = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
for (v, win, stmt), o in zip(items, outs):
    t = o.outputs[0].text.strip().upper(); lab = "SUPPORTED" if t.startswith("SUPPORTED") else ("PARTIAL" if t.startswith("PARTIAL") else ("NOT_SUPPORTED" if "NOT" in t else "?"))
    res[v][lab] += 1
    if lab == "NOT_SUPPORTED" and len(ex[v]) < 3: ex[v].append(stmt[:160])
for v, c in res.items():
    n = sum(c.values()); print(f"{v:9s} n={n:4d} supported {c['SUPPORTED']/n*100:5.1f}%  partial {c['PARTIAL']/n*100:5.1f}%  NOT {c['NOT_SUPPORTED']/n*100:5.1f}%  ?{c['?']}")
    for e in ex[v]: print("    NOT:", e.replace("\n", " "))
print("KIT_AUDIT_DONE")
