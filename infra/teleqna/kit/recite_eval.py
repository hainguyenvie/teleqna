#!/usr/bin/env python3
"""Recite-then-answer, closed-book (same model, no corpus): pass 1 writes what it knows about the question's
topic; pass 2 answers the harness prompt with that recitation as "reference material". Reports plain vs recite."""
import json, re, argparse
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
RECITE = ("You are a telecom standards expert. Before answering, recall from memory the relevant technical facts for the "
          "question below: definitions, exact values, identifiers, procedures and conditions from the 3GPP/IEEE specifications "
          "or research literature it refers to. Write 4-8 precise sentences of facts only. Do not answer the question, do not "
          "mention the options.\n\nQuestion: {question}")
STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-E])\s*(?:$|\n|\.)", re.M); LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-E])")
def parse(t, n):
    m = STRICT.findall(t or "") or LOOSE.findall(t or "")
    if not m: return -1
    g = ord(m[-1].upper()) - 65; return g if g < n else -1
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--gpu-mem", type=float, default=0.45)
ap.add_argument("--with-options", action="store_true", help="show the options in the recitation prompt too"); a = ap.parse_args()
rows = [json.loads(l) for l in open(R / "data/eval/otfull10000.jsonl", encoding="utf-8")]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True)
llm = LLM(model=a.ckpt, gpu_memory_utilization=a.gpu_mem, max_model_len=4096, dtype="bfloat16")
chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def qtext(r):
    if not a.with_options: return r["question"]
    return r["question"] + "\n" + "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
rec = llm.generate([chat(RECITE.format(question=qtext(r))) for r in rows], SamplingParams(temperature=0.0, max_tokens=320))
def harness(r, ctx=None):
    n = len(r["choices"]); q = r["question"] if ctx is None else f"{HEADER}\n\n[1] {ctx}\n\n---\n\n{r['question']}"
    return chat(TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=q, choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))))
recs = [o.outputs[0].text.strip() for o in rec]
plain = llm.generate([harness(r) for r in rows], SamplingParams(temperature=0.0, max_tokens=16))
withr = llm.generate([harness(r, c) for r, c in zip(rows, recs)], SamplingParams(temperature=0.0, max_tokens=16))
out = []; ap_ = ar = 0; un = 0
for r, p, w, c in zip(rows, plain, withr, recs):
    n = len(r["choices"]); gp = parse(p.outputs[0].text, n); gr = parse(w.outputs[0].text, n)
    ap_ += gp == r["answer"]; ar += gr == r["answer"]; un += gr < 0
    out.append(dict(sample_id=r["sample_id"], subject=r["subject"], gold=r["answer"], plain=gp, recite=gr, recitation=c))
N = len(rows); print(f"{a.tag}: plain {ap_/N*100:.2f} | recite-then-answer {ar/N*100:.2f} (unparsed {un})  mean recitation chars {sum(len(c) for c in recs)/N:.0f}")
import collections
bys = collections.defaultdict(lambda: [0, 0, 0])
for o in out: s = bys[o["subject"]]; s[0] += 1; s[1] += o["plain"] == o["gold"]; s[2] += o["recite"] == o["gold"]
for s, (n, p, q) in sorted(bys.items()): print(f"  {s:26s} plain {p/n*100:6.2f}  recite {q/n*100:6.2f}")
(R / "results/kit").mkdir(exist_ok=True)
with open(R / f"results/kit/recite_{a.tag}.jsonl", "w", encoding="utf-8") as fh:
    for o in out: fh.write(json.dumps(o, ensure_ascii=False) + "\n")
print("RECITE_DONE")
