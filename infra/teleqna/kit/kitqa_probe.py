#!/usr/bin/env python3
"""Is the knowledge in the weights? Closed-book answer to the kit's own QA pairs (view qa, evidence-gated) drawn from the
windows of (a) test rows the checkpoint still gets wrong although a window fixes them in context, (b) rows it fixed,
(c) rows kept right. Score = containment of the gated answer string in the model's short answer (no test labels used
for scoring; test labels only define the buckets). Compares base vs checkpoint."""
import json, re, random, argparse, collections, unicodedata, glob
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=600); ap.add_argument("--gpu-mem", type=float, default=0.28); a = ap.parse_args()
rng = random.Random(0)
T = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: bool(x["correct"]) for x in d}
B = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json"); V = L(R / f"results/landscape/{a.tag}_otfull10000_base_nothink512.json")
cov1 = set(json.load(open(R / "results/kit/coverage_sets.json"))["cov1"])
buckets = {"unfixed-with-source": [q for q in T if not B[q] and not V[q] and q in cov1], "fixed": [q for q in T if not B[q] and V[q]], "kept-right": [q for q in T if B[q] and V[q]]}
q2w = collections.defaultdict(set)
for f in ("data/kit/windows_keep.jsonl", "data/kit/windows_rest.jsonl"):
    for l in open(R / f, encoding="utf-8"):
        w = json.loads(l)
        for q in w.get("for_q", []): q2w[q].add(w["win_id"])
need = {}
for b, qs in buckets.items():
    for q in rng.sample(qs, min(a.n, len(qs))):
        for w in q2w.get(q, ()): need.setdefault(w, []).append((b, q))
qa = collections.defaultdict(list)
for f in glob.glob(str(R / "data/kit/tier1/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier2/views_s*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if '"view": "qa"' not in l or '"ok": true' not in l: continue
        i = l.find('"win_id": "'); w = l[i + 11:l.find('"', i + 11)]
        if w in need:
            try: m = json.loads(json.loads(l)["text"])
            except Exception: continue
            if isinstance(m, dict) and isinstance(m.get("q"), str) and isinstance(m.get("a"), str) and 2 <= len(m["a"].split()) <= 12: qa[w].append((m["q"], m["a"]))
items = []
for w, lst in need.items():
    for (b, q) in lst[:1]:
        for (qq, aa) in rng.sample(qa[w], min(2, len(qa[w]))): items.append((b, q, w, qq, aa))
print({b: sum(1 for x in items if x[0] == b) for b in buckets}, flush=True)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True); llm = LLM(model=a.ckpt, gpu_memory_utilization=a.gpu_mem, max_model_len=2048, dtype="bfloat16")
P = "Answer the telecom question with a short phrase only (no explanation).\n\nQuestion: {q}"
outs = llm.generate([tok.apply_chat_template([{"role": "user", "content": P.format(q=x[3])}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for x in items], SamplingParams(temperature=0.0, max_tokens=40))
res = collections.defaultdict(lambda: [0, 0])
for x, o in zip(items, outs):
    ans = norm(o.outputs[0].text); gold = norm(x[4]); hit = gold in ans or (len(gold.split()) >= 3 and sum(t in ans for t in gold.split()) / len(gold.split()) >= 0.8)
    res[x[0]][0] += 1; res[x[0]][1] += hit
for b, (n, k) in res.items(): print(f"{a.tag:10s} {b:22s} kit-QA closed-book recall {k/n*100:5.1f}% (n={n})")
import itertools
for x, o in itertools.islice(zip(items, outs), 0, 14): print("  Q:", x[3][:90], "| GOLD:", x[4][:40], "| MODEL:", o.outputs[0].text.strip()[:60].replace("\n", " "))
json.dump([dict(bucket=x[0], q=x[3], gold=x[4], model=o.outputs[0].text.strip()) for x, o in zip(items, outs)], open(R / f"results/kit/kitqa_{a.tag}_{Path(a.ckpt).name}.json", "w"), indent=0)
print("KITQA_DONE")
