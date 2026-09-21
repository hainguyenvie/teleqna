#!/usr/bin/env python3
"""Per-question option probabilities under the harness chat prompt (next token after 'ANSWER:'), saved for
partial-learning diagnostics: results/kit/letterprobs_<tag>.json = {sample_id: [p_A, p_B, ...]}."""
import json, math, argparse
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--gpu-mem", type=float, default=0.28); a = ap.parse_args()
rows = [json.loads(l) for l in open(R / "data/eval/otfull10000.jsonl", encoding="utf-8")]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
llm = LLM(model=a.ckpt, gpu_memory_utilization=a.gpu_mem, max_model_len=2048, dtype="bfloat16")
LET = {c: tok.encode(" " + c, add_special_tokens=False)[-1] for c in "ABCDE"}
def chat(r):
    n = len(r["choices"]); ch = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
    p = ("Answer the following multiple choice question. The entire content of your response should be of the following format: "
         f"'ANSWER: $LETTER' (without quotes) where LETTER is one of {','.join(chr(65+i) for i in range(n))}.\n\n{r['question']}\n\n{ch}")
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "ANSWER:"
outs = llm.generate([chat(r) for r in rows], SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)); res = {}; c = 0
for r, o in zip(rows, outs):
    lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}; n = len(r["choices"])
    probs = [math.exp(lp[LET[ch]].logprob) if LET[ch] in lp else 0.0 for ch in "ABCDE"[:n]]; res[r["sample_id"]] = probs
    c += max(range(n), key=lambda i: probs[i]) == r["answer"]
(R / "results/kit").mkdir(exist_ok=True); json.dump(res, open(R / f"results/kit/letterprobs_{a.tag}.json", "w"))
print(f"{a.tag}: chat logprob acc {c/len(rows)*100:.2f}"); print("LETTERPROBS_DONE")
