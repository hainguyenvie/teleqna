#!/usr/bin/env python3
"""Label-free window signal #2: the model's letter distribution WITH the window in context.
For every (question, window) row of data/eval/win_single.jsonl, render the harness prompt, prefill the
assistant turn with "ANSWER:" and read the next-token log-probs over the letter tokens. Also the same for the
plain question (no window) once per question. Writes results/kit/win_conf.jsonl:
{sample_id, win_id, p: {A:..}, top, margin, plain_top, plain_margin}. No gold used."""
import json, math, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; BASE = str(Path.home() / "projects/_shared/models/Qwen3-8B")
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
llm = LLM(model=BASE, gpu_memory_utilization=0.70, max_model_len=16384, dtype="bfloat16", enable_prefix_caching=True)
LET = {c: tok.encode(" " + c, add_special_tokens=False)[-1] for c in "ABCDE"}
def build(row):
    n = len(row["choices"]); letters = ",".join(chr(65 + i) for i in range(n))
    ch = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(row["choices"]))
    p = TEMPLATE.format(letters=letters, question=row["question"], choices=ch)
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "ANSWER:", n
rows = [json.loads(l) for l in open(R / "data/eval/win_single.jsonl", encoding="utf-8")]
plain = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
items = [(r["sample_id"], *build(r)) for r in rows] + [(f"{q}|PLAIN", *build(r)) for q, r in plain.items()]
print(f"{len(items):,} prompts", flush=True)
sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
outs = llm.generate([p for _, p, _ in items], sp)
res = {}
for (sid, _, n), o in zip(items, outs):
    lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
    probs = {c: math.exp(lp[LET[c]].logprob) if LET[c] in lp else 0.0 for c in "ABCDE"[:n]}
    z = sum(probs.values()) or 1.0; probs = {c: v / z for c, v in probs.items()}
    top = sorted(probs.items(), key=lambda x: -x[1]); res[sid] = dict(p=probs, top=top[0][0], margin=top[0][1] - (top[1][1] if len(top) > 1 else 0.0))
(R / "results/kit").mkdir(parents=True, exist_ok=True)
with open(R / "results/kit/win_conf.jsonl", "w", encoding="utf-8") as fh:
    for r in rows:
        q, w = r["sample_id"].split("|"); x = res[r["sample_id"]]; pl = res[f"{q}|PLAIN"]
        fh.write(json.dumps(dict(sample_id=q, win_id=w, p=x["p"], top=x["top"], margin=x["margin"], plain_top=pl["top"], plain_margin=pl["margin"])) + "\n")
print("WIN_CONF_DONE", flush=True)
