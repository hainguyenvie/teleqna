#!/usr/bin/env python3
"""Preference pairs for letter-DPO (label-free w.r.t. the test set): synthetic evidence-gated MCQs (kit mcq + cmcq),
label = generator's evidence-backed answer. The policy checkpoint scores the letter probabilities under 2 rotations;
a pair is kept when the model is wrong or unsure (p_gold < 0.9): chosen = gold letter, rejected = the model's most
probable WRONG letter. Output rows {prompt, chosen, rejected} for train_dpo_letter.py."""
import json, re, glob, random, argparse, math, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=120000); ap.add_argument("--out", default="data/kit/dpo/pairs.jsonl")
ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--seed", type=int, default=5); ap.add_argument("--max-pgold", type=float, default=0.9); a = ap.parse_args()
rng = random.Random(a.seed)
files = sorted(glob.glob(str(R / "data/kit/tier1/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier1c/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier25c/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier[2345]/views_s*.jsonl")))
mcqs = []
for f in files:
    for l in open(f, encoding="utf-8"):
        if '"view": "mcq"' not in l or '"ok": true' not in l: continue
        try: m = json.loads(json.loads(l)["text"])
        except Exception: continue
        if isinstance(m, dict) and isinstance(m.get("options"), list) and 4 <= len(m["options"]) <= 5 and isinstance(m.get("answer"), int) and 0 <= m["answer"] < len(m["options"]) and isinstance(m.get("q"), str): mcqs.append(m)
rng.shuffle(mcqs); mcqs = mcqs[:a.n]; print(f"gated mcqs {len(mcqs):,}", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True); llm = LLM(model=a.ckpt, gpu_memory_utilization=a.gpu_mem, max_model_len=2048, dtype="bfloat16")
LET = {c: tok.encode(" " + c, add_special_tokens=False)[-1] for c in "ABCDE"}
def user(m, s):
    n = len(m["options"]); ch = [m["options"][(i + s) % n] for i in range(n)]
    return TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=m["q"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(ch)))
prompts, meta = [], []
for i, m in enumerate(mcqs):
    n = len(m["options"])
    for s in rng.sample(range(n), 2): prompts.append(tok.apply_chat_template([{"role": "user", "content": user(m, s)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "ANSWER:"); meta.append((i, s))
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)); st = collections.Counter(); Path(R / a.out).parent.mkdir(parents=True, exist_ok=True)
with open(R / a.out, "w", encoding="utf-8") as f:
    for (i, s), o in zip(meta, outs):
        m = mcqs[i]; n = len(m["options"]); lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        pr = [math.exp(lp[LET[c]].logprob) if LET[c] in lp else 0.0 for c in "ABCDE"[:n]]; z = sum(pr) or 1; pr = [x / z for x in pr]
        g = (m["answer"] - s) % n; wrong = max((j for j in range(n) if j != g), key=lambda j: pr[j])
        if pr[g] >= a.max_pgold: st["confident-right"] += 1; continue
        st["pair"] += 1; f.write(json.dumps(dict(prompt=user(m, s), chosen=f" {chr(65+g)}", rejected=f" {chr(65+wrong)}", p_gold=round(pr[g], 3)), ensure_ascii=False) + "\n")
print("pairs:", dict(st)); print("DPO_PAIRS_DONE")
