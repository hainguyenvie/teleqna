#!/usr/bin/env python3
"""Who is wrong when the trained model consistently disagrees with the kit gold on a synthetic MCQ — the kit or the model?
OTel-31B reads the source window and answers the MCQ; compare with kit gold and with the model's vote. Sample of
cons-wrong (and cons-right as control) from a retention run's questions.jsonl. Measurement only (synthetic data, no test key)."""
import json, re, random, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; JUDGE = str(Path.home() / "projects/_shared/models/OTel-2.0-31B-IT")
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="data/kit/ret_merge_l/questions.jsonl"); ap.add_argument("--n", type=int, default=500); ap.add_argument("--gpu-mem", type=float, default=0.85); a = ap.parse_args()
rng = random.Random(0); qs = [json.loads(l) for l in open(R / a.run, encoding="utf-8")]
groups = {"cons-wrong": [q for q in qs if q["cls"] == "cons-wrong"], "cons-right": [q for q in qs if q["cls"] == "cons-right"], "uncertain": [q for q in qs if q["cls"] == "uncertain"]}
sample = {g: rng.sample(v, min(a.n if g == "cons-wrong" else a.n // 2, len(v))) for g, v in groups.items()}
need = {q["win"] for v in sample.values() for q in v}; wtext = {}
for l in open(R / "data/kit/big/windows_all.jsonl", encoding="utf-8"):
    j = l.find('"win_id": "'); w = l[j + 11:l.find('"', j + 11)]
    if w in need: wtext[w] = json.loads(l)["text"]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(JUDGE, local_files_only=True); llm = LLM(model=JUDGE, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16")
P = ("You are grading a multiple-choice question written from the PASSAGE below. Using ONLY the passage, pick the correct option. "
     "If the passage does not support any option, or supports more than one, answer 'ANSWER: X'.\n\nPASSAGE:\n{doc}\n\nQUESTION: {q}\n\n{ch}\n\nReply with exactly one line: ANSWER: <letter or X>")
items = [(g, q) for g, v in sample.items() for q in v]
prompts = [tok.apply_chat_template([{"role": "user", "content": P.format(doc=wtext.get(q["win"], "")[:6000], q=q["q"], ch="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(q["choices"])))}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for g, q in items]
outs = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))]
res = collections.defaultdict(collections.Counter); ex = []
with open(R / "results/kit/conswrong_audit_raw.jsonl", "w", encoding="utf-8") as fr:
    for (g, q), o in zip(items, outs): fr.write(json.dumps(dict(group=g, q=q["q"], gold=q["gold"], vote=q.get("vote"), out=o[:600]), ensure_ascii=False) + "\n")
for (g, q), o in zip(items, outs):
    mm = re.findall(r"ANSWER\s*:\s*\**\s*([A-EX])\b", o.upper()); j = mm[-1] if mm else "?"
    ji = ord(j) - 65 if j in "ABCDE" else -1
    if ji == q["gold"]: v = "judge=kit-gold (model wrong)"
    elif ji == q.get("vote"): v = "judge=model (kit gold wrong)"
    elif j == "X": v = "unsupported/ambiguous"
    elif j == "?": v = "unparsed"
    else: v = "judge=third option"
    res[g][v] += 1
    if g == "cons-wrong" and len(ex) < 12: ex.append((v, q["q"][:90], "GOLD:", q["choices"][q["gold"]][:40], "MODEL:", q["choices"][q["vote"]][:40], "JUDGE:", j))
for g, c in res.items():
    n = sum(c.values()); print(f"{g:11s} n={n}: " + " | ".join(f"{k} {100*v/n:.1f}%" for k, v in c.most_common()))
for e in ex: print("  ", e)
print("CONSWRONG_AUDIT_DONE")
