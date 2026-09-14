#!/usr/bin/env python3
"""Arm H stage 1b: run the MCQ generator and parse what comes back."""
import argparse, json, pathlib, re

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--data", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--max-new", type=int, default=320)
ap.add_argument("--max-model-len", type=int, default=4096)
ap.add_argument("--gpu-mem", type=float, default=0.90)
a = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

rows = [json.loads(l) for l in a.data.open()]
print(f"{len(rows)} generation prompts", flush=True)
tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) for r in rows]
lens = sorted(len(tok(p).input_ids) for p in prompts)
print(f"prompt tokens p50={lens[len(lens)//2]} p99={lens[int(len(lens)*.99)]} max={lens[-1]}",
      flush=True)

# Temperature 0.8: 16,000 items generated greedily from overlapping windows come
# back near-duplicated. Diversity is the point here, and every item is filtered
# afterwards anyway.
llm = LLM(model=a.base, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
          tensor_parallel_size=1, enforce_eager=False)
outs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, seed=17,
                                            max_tokens=a.max_new))

Q = re.compile(r"^Q:\s*(.+?)\s*$", re.M)
OPT = re.compile(r"^\s*([A-E])\)\s*(.+?)\s*$", re.M)
COR = re.compile(r"^CORRECT:\s*\**\s*([A-E])\b", re.M)

kept = bad = 0
with a.out.open("w") as fh:
    for r, o in zip(rows, outs):
        t = o.outputs[0].text
        t = re.split(r"<\|?channel", t)[0]
        mq, mc = Q.search(t), COR.search(t)
        opts = OPT.findall(t)
        if not (mq and mc and len(opts) >= 3):
            bad += 1
            continue
        seen, choices = set(), []
        for L, txt in opts:
            if L in seen:
                continue
            seen.add(L)
            choices.append((L, txt))
        choices.sort(key=lambda x: x[0])
        letters = [L for L, _ in choices]
        if letters != [chr(65 + i) for i in range(len(letters))]:
            bad += 1
            continue
        gi = ord(mc.group(1)) - 65
        if not (0 <= gi < len(choices)):
            bad += 1
            continue
        fh.write(json.dumps({
            "src_id": r["src_id"], "win": r["win"], "window": r["window"],
            "question": mq.group(1), "choices": [c for _, c in choices],
            "answer": gi,
        }, ensure_ascii=False) + "\n")
        kept += 1
print(f"parsed {kept} items, discarded {bad} malformed -> {a.out}", flush=True)
