#!/usr/bin/env python3
"""Mode-mismatch diagnostic: score the same MCQs (a) harness chat prompt, (b) raw-text prompt with no chat template
("Question: ...\nA) ..\nAnswer:" -> next token letter via log-probs). If (b) gains more over base than (a), the kit's
knowledge is stored in raw-text mode and the eval format is the barrier."""
import json, math, argparse
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--gpu-mem", type=float, default=0.45); a = ap.parse_args()
rows = [json.loads(l) for l in open(R / "data/eval/otfull10000.jsonl", encoding="utf-8")]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
llm = LLM(model=a.ckpt, gpu_memory_utilization=a.gpu_mem, max_model_len=2048, dtype="bfloat16")
LET = {c: tok.encode(" " + c, add_special_tokens=False)[-1] for c in "ABCDE"}
def raw(r):
    ch = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
    return f"Question: {r['question']}\n{ch}\nAnswer:"
def chat(r):
    n = len(r["choices"]); ch = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
    p = ("Answer the following multiple choice question. The entire content of your response should be of the following format: "
         f"'ANSWER: $LETTER' (without quotes) where LETTER is one of {','.join(chr(65+i) for i in range(n))}.\n\n{r['question']}\n\n{ch}")
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + "ANSWER:"
sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
res = {}
for mode, f in (("raw", raw), ("chat", chat)):
    outs = llm.generate([f(r) for r in rows], sp); c = 0
    for r, o in zip(rows, outs):
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}; n = len(r["choices"])
        probs = [math.exp(lp[LET[ch]].logprob) if LET[ch] in lp else 0.0 for ch in "ABCDE"[:n]]
        c += max(range(n), key=lambda i: probs[i]) == r["answer"]
    res[mode] = c / len(rows) * 100
print(f"{a.tag}: raw-text prompt {res['raw']:.2f} | chat harness prompt (logprob) {res['chat']:.2f}"); print("RAWMODE_DONE")
