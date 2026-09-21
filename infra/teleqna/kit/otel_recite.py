#!/usr/bin/env python3
"""Synthetic source from a stronger teacher (OTel-2.0-31B-IT) for the functionally-uncovered questions: the teacher
writes a reference passage about the question's topic (no options shown), twice at different seeds; passages are
kept only if their checkable atoms agree (label-free gate). Then a Qwen3-8B eval set is built with the passage in
context — scored separately (gold used only to measure how much the teacher covers)."""
import json, re, unicodedata, argparse, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; GEN = str(Path.home() / "projects/_shared/models/OTel-2.0-31B-IT")
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
ATOM = re.compile(r"\b(?:TS\s?\d+\.\d+|TR\s?\d+\.\d+|Rel-?\d+|\d+(?:\.\d+)?\s?(?:ms|s|dB|dBm|GHz|MHz|kHz|Mbps|Gbps|bit|bits|bytes|%)|[A-Z]{3,7}|\d{2,4})\b")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
PROMPT = ("You are a senior telecom standards and research expert. Write a precise reference passage (6-10 sentences) that "
          "contains the technical facts needed to answer the question below: definitions, exact values, identifiers, "
          "procedures, conditions, and the relevant specification or paper context. State facts only; do not answer the "
          "question directly and do not speculate — if you are unsure of a value, omit it.\n\nQuestion: {q}")
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="data/kit/uncovered_functional.jsonl"); ap.add_argument("--gpu-mem", type=float, default=0.85); a = ap.parse_args()
rows = [json.loads(l) for l in open(R / a.inp, encoding="utf-8")]
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(GEN, local_files_only=True)
llm = LLM(model=GEN, gpu_memory_utilization=a.gpu_mem, max_model_len=4096, dtype="bfloat16")
rend = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
prompts = [rend(PROMPT.format(q=re.sub(r"\s*\[[^\]]*\]\s*$", "", r["question"]))) for r in rows]
g1 = llm.generate(prompts, SamplingParams(temperature=0.3, top_p=0.9, max_tokens=400, seed=1))
g2 = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=400, seed=2))
keep = 0
with open(R / "data/kit/otel_recite.jsonl", "w", encoding="utf-8") as fh, open(R / "data/eval/otel_recite_eval.jsonl", "w", encoding="utf-8") as ev:
    for r, o1, o2 in zip(rows, g1, g2):
        t1, t2 = o1.outputs[0].text.strip(), o2.outputs[0].text.strip()
        a1 = {x for x in ATOM.findall(t1) if len(x) >= 2}; a2 = {x for x in ATOM.findall(t2) if len(x) >= 2}
        agree = len(a1 & a2) / max(1, len(a1 | a2)); ok = agree >= 0.5 and len(t1) > 200
        keep += ok
        fh.write(json.dumps(dict(sample_id=r["sample_id"], text=t1, text2=t2, atom_agree=round(agree, 3), ok=ok), ensure_ascii=False) + "\n")
        q = test[r["sample_id"]]
        ev.write(json.dumps({**q, "question": f"{HEADER}\n\n[1] {t1}\n\n---\n\n{q['question']}", "n_ctx": 1, "gate_ok": ok}, ensure_ascii=False) + "\n")
print(f"recited {len(rows)}; gate-kept (atom agreement>=0.5) {keep}"); print("OTEL_RECITE_DONE")
