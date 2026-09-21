#!/usr/bin/env python3
"""Is the missing knowledge absent from the corpus, or only missed by retrieval?
Take the questions whose CURRENT RAG windows could not decide the answer (31B said X / no windows, from
wrong_taxonomy_<tag>.json), run a much deeper BM25 pass over the whole filtered store (two query forms, k=40 each,
question + options, never the answer), give OTel-31B the top-N chunks and ask again. Gold is used only to score."""
import json, re, argparse, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; JUDGE = str(Path.home() / "projects/_shared/models/OTel-2.0-31B-IT")
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="wise_u3"); ap.add_argument("--k", type=int, default=40); ap.add_argument("--top", type=int, default=10)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--gpu-mem", type=float, default=0.85)
ap.add_argument("--cache", default="data/kit/retr_ceiling_passages.jsonl"); ap.add_argument("--retrieve-only", action="store_true"); ap.add_argument("--judge-only", action="store_true"); a = ap.parse_args()
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
tax = json.load(open(R / f"results/kit/wrong_taxonomy_{a.tag}.json"))
sel = [r["q"] for r in tax if r["src"] in ("unsupported (X)", "no windows")]
if a.n: sel = sel[:a.n]
print(f"questions with no decisive current window: {len(sel):,}", flush=True)
cache = R / a.cache
if not a.judge_only:
    import bm25s
    retr = bm25s.BM25.load(str(R / "data/kit/bm25_index")); print("index loaded", flush=True)
if not a.judge_only:
    need = {}
    for form, qf in (("qo", lambda r: r["question"] + " " + " ".join(r["choices"])), ("q", lambda r: r["question"])):
        res, sc = retr.retrieve(bm25s.tokenize([qf(test[q]) for q in sel], stopwords="en"), k=a.k)
        for q, ids, ss in zip(sel, res, sc):
            d = need.setdefault(q, {})
            for i, s in zip(ids, ss): d[int(i)] = max(d.get(int(i), 0.0), float(s))
    allids = {i for d in need.values() for i in d}
    print(f"chunks to fetch: {len(allids):,}", flush=True)
    texts = {}
    with open(R / "data/kit/tb_chunks.jsonl", encoding="utf-8") as fh:
        for n, l in enumerate(fh):
            if n in allids: texts[n] = json.loads(l)["text"]
    print(f"fetched {len(texts):,}", flush=True)
    with open(cache, "w", encoding="utf-8") as fh:
        for q in sel:
            top = sorted(need[q].items(), key=lambda x: -x[1])[:a.top]
            fh.write(json.dumps(dict(q=q, passages=[texts.get(i, "")[:2200] for i, s in top]), ensure_ascii=False) + "\n")
    print(f"cached passages -> {cache}", flush=True)
    if a.retrieve_only: print("RETRIEVE_ONLY_DONE"); raise SystemExit
PASS = {json.loads(l)["q"]: json.loads(l)["passages"] for l in open(cache, encoding="utf-8")}
sel = [q for q in sel if q in PASS]
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(JUDGE, local_files_only=True); llm = LLM(model=JUDGE, gpu_memory_utilization=a.gpu_mem, max_model_len=24576, dtype="bfloat16")
P = ("Using ONLY the reference passages, answer the multiple-choice question. If the passages do not support any option, or support more than one, answer 'ANSWER: X'.\n\nPASSAGES:\n{doc}\n\n{mc}\n\nReply with exactly one line: ANSWER: <letter or X>")
def mc(q): r = test[q]; return r["question"] + "\n\n" + "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
prompts = []
for q in sel:
    doc = "\n\n".join(f"[{j+1}] {t}" for j, t in enumerate(PASS[q][:a.top]))
    prompts.append(tok.apply_chat_template([{"role": "user", "content": P.format(doc=doc, mc=mc(q))}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
outs = [o.outputs[0].text for o in llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=384))]
c = collections.Counter(); rows = []
for q, o in zip(sel, outs):
    mm = re.findall(r"ANSWER\s*:\s*\**\s*([A-EX])\b", o.upper()); j = mm[-1] if mm else "?"
    g = chr(65 + test[q]["answer"]); k = "deep=gold (retrieval was the gap)" if j == g else "still unsupported (X)" if j == "X" else "deep=other option" if j in "ABCDE" else "unparsed"
    c[k] += 1; rows.append(dict(q=q, subject=test[q]["subject"], verdict=k, judge=j, gold=g))
n = len(sel); print(f"\ndeep retrieval (k={a.k} x2 forms, top-{a.top} passages) on {n} questions:")
for k, v in c.most_common(): print(f"  {k:40s} {v:5d} ({100*v/n:.1f}%)")
sub = collections.defaultdict(collections.Counter)
for r in rows: sub[r["subject"]][r["verdict"]] += 1
print("by subject:"); [print(f"  {s:26s} " + " | ".join(f"{k[:26]} {v}" for k, v in cc.most_common(3))) for s, cc in sub.items()]
json.dump(rows, open(R / f"results/kit/retrieval_ceiling_{a.tag}.json", "w"), ensure_ascii=False)
print("RETRIEVAL_CEILING_DONE")
