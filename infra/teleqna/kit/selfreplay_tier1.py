#!/usr/bin/env python3
"""Anti-drift anchor for tier 1: Qwen3-8B's own greedy answers on the kit's synthetic MCQs (mcq view, gated),
harness format byte-for-byte, parseable rows only. No benchmark question anywhere (measured self-replay recipe)."""
import json, re, glob, random
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; BASE = str(Path.home() / "projects/_shared/models/Qwen3-8B")
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-E])\s*$", re.M)
rows = []
import sys
VIEWS = sys.argv[1] if len(sys.argv) > 1 else "data/kit/tier1/views_s*.jsonl"; OUT = sys.argv[2] if len(sys.argv) > 2 else "data/kit/tier1/anchor_selfreplay.jsonl"
for f in sorted(glob.glob(str(R / VIEWS))):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if r["view"] == "mcq" and r["ok"]:
            m = json.loads(r["text"])
            if re.search(r"(?i)excerpt|passage|the document|text above", m["q"] + " ".join(m["options"])): continue
            m["q"] = m["q"].replace("~", " "); m["options"] = [o.replace("~", " ") for o in m["options"]]; rows.append((r["win_id"], m))
random.Random(20260914).shuffle(rows); rows = rows[:80000]
print(f"{len(rows)} synthetic MCQ", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
import os
llm = LLM(model=BASE, gpu_memory_utilization=float(os.environ.get("GPUMEM", "0.90")), max_model_len=2048, dtype="bfloat16")
prompts = []
for wid, m in rows:
    opts = list(m["options"]); n = len(opts)
    letters = ", ".join(chr(65 + i) for i in range(n)); ch = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(opts))
    prompts.append(TEMPLATE.format(letters=letters, question=m["q"], choices=ch))
outs = llm.generate([tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts],
                    SamplingParams(temperature=0.0, max_tokens=16))
out = R / OUT; n = 0; agree = 0
with open(out, "w", encoding="utf-8") as fh:
    for (wid, m), p, o in zip(rows, prompts, outs):
        t = o.outputs[0].text.strip(); mm = STRICT.match(t) or STRICT.search(t)
        if not mm: continue
        ag = (ord(mm.group(1)) - 65 == m["answer"]); n += 1; agree += ag
        fh.write(json.dumps(dict(prompt=p, completion=f"ANSWER: {mm.group(1)}", win_id=wid, gen_answer=chr(65 + m["answer"]), agree=bool(ag)), ensure_ascii=False) + "\n")
print(f"anchor rows kept {n}/{len(rows)}; base agrees with generator gold on {agree/max(n,1)*100:.1f}% -> {out}", flush=True)
print("SELFREPLAY_DONE", flush=True)
