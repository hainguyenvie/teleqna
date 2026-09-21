#!/usr/bin/env python3
"""On-policy variant of hard_judge: from a vote run (--run dir with questions.jsonl incl. pgold), take questions with
0 < pgold < 1 (the model sometimes samples gold: "partially known"), judge with OTel-31B reading the window, keep judge == gold,
and emit rows with the on-policy weight (pgold) so the trainer mix can duplicate rows proportional to it
(GRPO with binary reward on a one-token answer reduces to CE on gold weighted by p(gold))."""
import json, re, argparse, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; JUDGE = str(Path.home() / "projects/_shared/models/OTel-2.0-31B-IT")
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="data/kit/onp1"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--pmin", type=float, default=0.01); ap.add_argument("--pmax", type=float, default=0.99); a = ap.parse_args()
qs = [json.loads(l) for l in open(R / a.run / "questions.jsonl", encoding="utf-8")]
hard = [q for q in qs if q.get("pgold") is not None and a.pmin <= q["pgold"] <= a.pmax and q.get("vote") != q["gold"] or (q.get("pgold") is not None and a.pmin <= q["pgold"] < 1.0 and q.get("vote") == q["gold"] and q["pgold"] <= a.pmax)]
hard = [q for i, q in enumerate(hard) if i % a.nshards == a.shard]
need = {q["win"] for q in hard}; wtext = {}
for l in open(R / "data/kit/big/windows_all.jsonl", encoding="utf-8"):
    j = l.find('"win_id": "'); w = l[j + 11:l.find('"', j + 11)]
    if w in need: wtext[w] = json.loads(l)["text"]
print(f"shard {a.shard}/{a.nshards}: {len(hard):,} hard questions, {len(wtext):,} windows", flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(JUDGE, local_files_only=True); llm = LLM(model=JUDGE, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, dtype="bfloat16")
P = ("You are grading a multiple-choice question written from the PASSAGE below. Using ONLY the passage, pick the correct option. "
     "If the passage does not support any option, or supports more than one, answer 'ANSWER: X'.\n\nPASSAGE:\n{doc}\n\nQUESTION: {q}\n\n{ch}\n\nReply with exactly one line: ANSWER: <letter or X>")
prompts = [tok.apply_chat_template([{"role": "user", "content": P.format(doc=wtext.get(q["win"], "")[:6000], q=q["q"], ch="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(q["choices"])))}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in hard]
outs = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=384))]
st = collections.Counter(); kept = 0
with open(R / a.run / f"rows_partial_verified_s{a.shard}.jsonl", "w", encoding="utf-8") as f:
    for q, o in zip(hard, outs):
        mm = re.findall(r"ANSWER\s*:\s*\**\s*([A-EX])\b", o.upper()); j = mm[-1] if mm else "?"
        ji = ord(j) - 65 if j in "ABCDE" else -1
        if ji != q["gold"]: st["kit-wrong" if ji == q.get("vote") else ("ambiguous" if j == "X" else "other")] += 1; continue
        st["verified"] += 1; kept += 1; n = len(q["choices"])
        for s in range(n):
            ch = [q["choices"][(i + s) % n] for i in range(n)]
            f.write(json.dumps(dict(prompt=TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=q["q"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(ch))), completion=f"ANSWER: {chr(65 + (q['gold'] - s) % n)}", cls=q["cls"], src=q["src"], win=q["win"], pgold=q["pgold"]), ensure_ascii=False) + "\n")
print(f"shard {a.shard}: verified {kept:,}/{len(hard):,} {dict(st)}"); print("ONP_JUDGE_DONE")
