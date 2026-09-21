#!/usr/bin/env python3
"""Label audit of ot-full rows (measurement only; never feeds training): a judge model reads the top windows retrieved for
the question and names the option the excerpts support (or NONE). Agreement with the gold key estimates label noise +
retrieval misses; disagreements are printed for reading. Usage: label_audit.py --judge PATH [--n 300] [--extra FILE]"""
import json, random, argparse, collections, re
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
ap = argparse.ArgumentParser(); ap.add_argument("--judge", required=True); ap.add_argument("--n", type=int, default=300); ap.add_argument("--gpu-mem", type=float, default=0.28); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="q3_8b"); ap.add_argument("--max-tokens", type=int, default=256); ap.add_argument("--debug", type=int, default=0); a = ap.parse_args()
rng = random.Random(a.seed)
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
q2w = collections.defaultdict(list); W = {}
for f in ("data/kit/windows_keep.jsonl", "data/kit/windows_rest.jsonl", "data/kit/tb_windows.jsonl"):
    try:
        for l in open(R / f, encoding="utf-8"):
            w = json.loads(l); W[w["win_id"]] = w["text"]
            for q in w.get("for_q", []): q2w[q].append(w["win_id"])
    except FileNotFoundError: pass
groups = {"random": rng.sample(sorted(test), a.n)}
try:
    S = [json.loads(l) for l in open(R / "data/kit/broke_judge_sample.jsonl", encoding="utf-8")]
    groups["broke(tier2)"] = [x["sample_id"] for x in S if x["group"] == "broke"]; groups["fixed(tier2)"] = [x["sample_id"] for x in S if x["group"] == "fixed"]
except FileNotFoundError: pass
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.judge, local_files_only=True); llm = LLM(model=a.judge, gpu_memory_utilization=a.gpu_mem, max_model_len=16384, dtype="bfloat16")
P = ("You are checking an exam key against source material.\n\nSOURCE EXCERPTS:\n{docs}\n\nQUESTION: {q}\n{opts}\n\n"
     "Decide which option the SOURCE EXCERPTS support. Briefly quote the decisive sentence if there is one (one line), then finish with a final line "
     "exactly of the form 'VERDICT: <letter>' if exactly one option is supported, or 'VERDICT: NONE' if the excerpts do not settle the question or support several options.")
items, prompts = [], []
for g, qs in groups.items():
    for q in qs:
        r = test[q]; docs = "\n\n".join(f"[{i+1}] {W[w][:2500]}" for i, w in enumerate(q2w.get(q, [])[:5])) or "(no excerpts retrieved)"
        opts = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
        items.append((g, q)); prompts.append(tok.apply_chat_template([{"role": "user", "content": P.format(docs=docs, q=r["question"], opts=opts)}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens))
for o in outs[:a.debug]: print("RAW:", repr(o.outputs[0].text[:200]))
res = collections.defaultdict(collections.Counter); dis = collections.defaultdict(list); out_rows = []
for (g, q), o in zip(items, outs):
    t = o.outputs[0].text.strip(); m = re.search(r"VERDICT\s*:\s*\**\s*([A-E]|NONE)", t, re.I) or re.search(r"(?:correct answer is|ANSWER\s*:)\s*\**\s*([A-E])\b", t, re.I); r = test[q]
    lab = m.group(1).upper() if m else "?"
    if lab == "NONE" or lab == "?": res[g]["none/unclear"] += 1
    elif ord(lab) - 65 == r["answer"]: res[g]["agrees with gold"] += 1
    else:
        res[g]["contradicts gold"] += 1
        if len(dis[g]) < 6: dis[g].append(f"[{q}] {r['question'][:100]} | gold: {r['choices'][r['answer']][:50]} | judge: {r['choices'][ord(lab)-65][:50]}")
    out_rows.append(dict(sample_id=q, group=g, judge=lab, gold=chr(65 + r["answer"]), n_ctx=len(q2w.get(q, [])[:5]), question=r["question"], choices=r["choices"], raw=t[:1200]))
for g, c in res.items():
    n = sum(c.values()); print(f"{g:14s} n={n:4d} agrees {c['agrees with gold']/n*100:5.1f}%  contradicts {c['contradicts gold']/n*100:5.1f}%  none/unclear {c['none/unclear']/n*100:5.1f}%")
    for d in dis[g]: print("    ", d)
json.dump(out_rows, open(R / f"results/kit/label_audit_{a.tag}.json", "w"), indent=0); print("LABEL_AUDIT_DONE")
