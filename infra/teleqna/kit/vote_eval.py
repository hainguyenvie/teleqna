#!/usr/bin/env python3
"""Self-consistency at inference (same model, no retrieval): for each question, sample K answers under each of
R choice rotations (letters mapped back to original indices), majority-vote. Reports greedy, vote-all, and
"vote only when greedy margin < tau" (label-free gating). Usage: vote_eval.py --ckpt PATH --tag TAG [--k 8 --rot 4]"""
import json, re, argparse, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-E])\s*(?:$|\n|\.)", re.M); LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-E])")
def parse(t, n):
    m = STRICT.findall(t or "") or LOOSE.findall(t or "")
    if not m: return -1
    g = ord(m[-1].upper()) - 65; return g if g < n else -1
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--k", type=int, default=8); ap.add_argument("--rot", type=int, default=4); ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--gpu-mem", type=float, default=0.40)
a = ap.parse_args()
rows = [json.loads(l) for l in open(R / "data/eval/otfull10000.jsonl", encoding="utf-8")]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
llm = LLM(model=a.ckpt, gpu_memory_utilization=a.gpu_mem, max_model_len=2048, dtype="bfloat16")
def prompt(r, shift):
    n = len(r["choices"]); ch = [r["choices"][(i + shift) % n] for i in range(n)]   # displayed position i shows original index (i+shift)%n
    p = TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=r["question"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(ch)))
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
prompts, meta = [], []
for r in rows:
    for s in range(min(a.rot, len(r["choices"]))): prompts.append(prompt(r, s)); meta.append((r["sample_id"], s))
greedy = llm.generate([p for p, m in zip(prompts, meta) if m[1] == 0], SamplingParams(temperature=0.0, max_tokens=16))
samp = llm.generate(prompts, SamplingParams(temperature=a.temp, top_p=0.95, max_tokens=16, n=a.k))
byq = {r["sample_id"]: r for r in rows}
votes = collections.defaultdict(collections.Counter); g = {}
for (sid, s), o in zip([m for m in meta if m[1] == 0], greedy):
    g[sid] = parse(o.outputs[0].text, len(byq[sid]["choices"]))
for (sid, s), o in zip(meta, samp):
    n = len(byq[sid]["choices"])
    for out in o.outputs:
        i = parse(out.text, n)
        if i >= 0: votes[sid][(i + s) % n] += 1
res = {}
for sid, r in byq.items():
    v = votes[sid]; tot = sum(v.values()); top = v.most_common(2)
    vt = top[0][0] if top else -1; margin = (top[0][1] - (top[1][1] if len(top) > 1 else 0)) / tot if tot else 0.0
    res[sid] = dict(gold=r["answer"], greedy=g[sid], vote=vt, margin=margin, share=(top[0][1] / tot if tot else 0.0))
acc = lambda f: sum(f(x) for x in res.values()) / len(res) * 100
print(f"{a.tag}: greedy {acc(lambda x: x['greedy']==x['gold']):.2f} | vote({a.rot}rot x {a.k}) {acc(lambda x: x['vote']==x['gold']):.2f}")
for tau in (1.0, 0.9, 0.75, 0.5):
    print(f"  vote only when vote-margin<{tau} else greedy: {acc(lambda x, t=tau: (x['vote'] if x['margin']<t else x['greedy'])==x['gold']):.2f}")
(R / "results/kit").mkdir(exist_ok=True); json.dump(res, open(R / f"results/kit/vote_{a.tag}.json", "w"))
print("VOTE_DONE")
