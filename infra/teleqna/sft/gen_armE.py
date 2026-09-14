#!/usr/bin/env python3
"""Arm E stage 1: turn gated evidence into one-sentence grounded facts.

Runs the 31B WITH the retrieved windows in the prompt. This is teacher-time only;
the resulting sentence goes into a training target whose prompt is the plain
closed-book harness prompt, so nothing retrieved survives to serving.

enable_thinking=False for the same reason the eval harness uses it: on this model
the pre-closed thought channel is what keeps output disciplined.
"""
import argparse, json, pathlib, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--data", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--max-new", type=int, default=120)
ap.add_argument("--max-model-len", type=int, default=8192)
ap.add_argument("--gpu-mem", type=float, default=0.90)
a = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

rows = [json.loads(l) for l in a.data.open(encoding="utf-8")]
print(f"{len(rows)} prompts", flush=True)
tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for r in rows]
lens = sorted(len(tok(p).input_ids) for p in prompts)
print(f"prompt tokens p50={lens[len(lens)//2]} p99={lens[int(len(lens)*.99)]} max={lens[-1]}",
      flush=True)
if lens[-1] > a.max_model_len - a.max_new:
    print(f"!! {sum(1 for x in lens if x > a.max_model_len - a.max_new)} prompts will "
          f"not fit in {a.max_model_len}", flush=True)

llm = LLM(model=a.base, gpu_memory_utilization=a.gpu_mem,
          max_model_len=a.max_model_len, tensor_parallel_size=1, enforce_eager=False)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_new))

# Strip any channel scaffold the template leaves behind, keep the first sentence
# block, and refuse anything that talks about "the document" -- the instruction
# forbade it and a target that references a passage the student cannot see is
# exactly the failure mode that poisoned the earlier prose SFT.
BAD = re.compile(r"\b(the (reference|document|passage|text|material|question|option)s?"
                 r"|according to|as (stated|mentioned|described)|the correct answer)\b", re.I)
n_bad = n_empty = 0
with a.out.open("w") as fh:
    for r, o in zip(rows, outs):
        t = o.outputs[0].text
        t = re.split(r"<\|?channel", t)[0]
        t = re.sub(r"<[^>]{0,40}>", " ", t)
        t = " ".join(t.split()).strip().strip('"')
        flag = ""
        if not t:
            n_empty += 1
            flag = "empty"
        elif BAD.search(t):
            n_bad += 1
            flag = "meta"
        fh.write(json.dumps({**r, "fact": t, "flag": flag}) + "\n")
print(f"wrote {a.out}  (empty {n_empty}, meta-referring {n_bad})", flush=True)
