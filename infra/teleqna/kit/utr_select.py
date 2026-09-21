#!/usr/bin/env python3
"""Uncertainty-targeted re-study (UTR), label-free w.r.t. the test set. Let the checkpoint answer gated synthetic MCQs
(kit mcq tiers 1-5 + sibling views) under R rotations x K samples; classify each question by vote share and by agreement
with the generator's gold (known by construction, never the test key): consistent-right / consistent-wrong / uncertain.
Reports retention (agreement with kit gold) per source, then emits gold-labelled rows (all rotations) for the HARD
questions (uncertain + consistent-wrong) that pass an evidence gate (gold option text verbatim in the window), the hard
window ids, and a subsample of consistent-right vote rows for stability."""
import json, re, glob, random, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-E])")
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=300000); ap.add_argument("--rot", type=int, default=4)
ap.add_argument("--k", type=int, default=4); ap.add_argument("--out", default="data/kit/utr"); ap.add_argument("--gpu-mem", type=float, default=0.28); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--easy-keep", type=int, default=60000, help="consistent-right questions kept as vote rows for stability")
a = ap.parse_args(); rng = random.Random(a.seed); out = R / a.out; out.mkdir(parents=True, exist_ok=True)
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
files = sorted(glob.glob(str(R / "data/kit/tier1/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier1c/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier[2345]/views_s*.jsonl")))
sibf = sorted(glob.glob(str(R / "data/kit/sib/views_s?.jsonl")) + glob.glob(str(R / "data/kit/sib_rest/views_s?.jsonl")))
mcqs = []
for f in files + sibf:
    src = ("sib" if "/sib" in f else f.split("/")[-2])
    for l in open(f, encoding="utf-8"):
        if '"view": "mcq"' not in l or '"ok": true' not in l: continue
        r = json.loads(l)
        try: m = json.loads(r["text"])
        except Exception: continue
        if isinstance(m, dict) and isinstance(m.get("options"), list) and 4 <= len(m["options"]) <= 5 and isinstance(m.get("q"), str) and isinstance(m.get("answer"), int) and 0 <= m["answer"] < len(m["options"]):
            mcqs.append(dict(q=m["q"], choices=m["options"], gold=m["answer"], win=r["win_id"], src=src, ev=m.get("evidence", "")))
print(f"gated mcq {len(mcqs):,} by src {collections.Counter(x['src'] for x in mcqs)}", flush=True)
rng.shuffle(mcqs); mcqs = mcqs[:a.n]
def user(r, shift):
    n = len(r["choices"]); ch = [r["choices"][(i + shift) % n] for i in range(n)]
    return TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=r["q"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(ch)))
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.ckpt, local_files_only=True); llm = LLM(model=a.ckpt, gpu_memory_utilization=a.gpu_mem, max_model_len=2048, dtype="bfloat16")
prompts, meta = [], []
for i, r in enumerate(mcqs):
    for s in range(min(a.rot, len(r["choices"]))):
        prompts.append(tok.apply_chat_template([{"role": "user", "content": user(r, s)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)); meta.append((i, s))
outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.95, max_tokens=8, n=a.k))
votes = [collections.Counter() for _ in mcqs]
for (i, s), o in zip(meta, outs):
    n = len(mcqs[i]["choices"])
    for x in o.outputs:
        m = LOOSE.findall(x.text)
        if m:
            g = ord(m[-1].upper()) - 65
            if g < n: votes[i][(g + s) % n] += 1
cls = []
for i, r in enumerate(mcqs):
    v = votes[i]; tot = sum(v.values())
    if not tot: cls.append("novote"); continue
    ans, c = v.most_common(1)[0]; share = c / tot; r["share"] = share; r["vote"] = ans; r["pgold"] = v[r["gold"]] / tot
    cls.append("uncertain" if share < 0.6 else ("cons-right" if ans == r["gold"] else "cons-wrong") if share >= 0.75 else ("mid-right" if ans == r["gold"] else "mid-wrong"))
for src in sorted(set(x["src"] for x in mcqs)):
    idx = [i for i, x in enumerate(mcqs) if x["src"] == src]; cc = collections.Counter(cls[i] for i in idx)
    agree = sum(1 for i in idx if mcqs[i].get("vote") == mcqs[i]["gold"]) / max(1, len(idx))
    print(f"{src:8s} n={len(idx):7,d} vote==gold {100*agree:5.1f}%  " + " ".join(f"{k}={100*cc[k]/len(idx):.1f}%" for k in ("cons-right", "cons-wrong", "mid-right", "mid-wrong", "uncertain", "novote")), flush=True)
cc = collections.Counter(cls); print("ALL", {k: f"{100*v/len(mcqs):.1f}%" for k, v in cc.items()}, f"vote==gold {100*sum(1 for x in mcqs if x.get('vote') == x['gold'])/len(mcqs):.1f}%")
# evidence gate needs window text
hard = [i for i, c in enumerate(cls) if c in ("uncertain", "cons-wrong", "mid-wrong")]
need = {mcqs[i]["win"] for i in hard}; wtext = {}
for l in open(R / "data/kit/big/windows_all.jsonl", encoding="utf-8"):
    j = l.find('"win_id": "'); w = l[j + 11:l.find('"', j + 11)]
    if w in need: wtext[w] = norm(json.loads(l)["text"])
gated = [i for i in hard if len(norm(mcqs[i]["choices"][mcqs[i]["gold"]])) >= 3 and norm(mcqs[i]["choices"][mcqs[i]["gold"]]) in wtext.get(mcqs[i]["win"], "")]
print(f"hard {len(hard):,} (uncertain {cc['uncertain']:,}, cons-wrong {cc['cons-wrong']:,}, mid-wrong {cc['mid-wrong']:,}); evidence-gated {len(gated):,}; windows {len({mcqs[i]['win'] for i in gated}):,}", flush=True)
# what do hard questions look like? gold longest share, numeric, option count
def longest(r): return r["gold"] == max(range(len(r["choices"])), key=lambda j: len(r["choices"][j]))
for name, idx in (("cons-right", [i for i, c in enumerate(cls) if c == "cons-right"]), ("hard-gated", gated)):
    if idx: print(f"  {name}: gold-longest {100*sum(longest(mcqs[i]) for i in idx)/len(idx):.1f}%  numeric-gold {100*sum(bool(re.fullmatch(r'[\d.,%\s\-]+', mcqs[i]['choices'][mcqs[i]['gold']].strip())) for i in idx)/len(idx):.1f}%  5-opt {100*sum(len(mcqs[i]['choices'])==5 for i in idx)/len(idx):.1f}%  by src {collections.Counter(mcqs[i]['src'] for i in idx).most_common(6)}")
with open(out / "rows_hard.jsonl", "w", encoding="utf-8") as f:
    for i in gated:
        r = mcqs[i]; n = len(r["choices"])
        for s in range(n):
            f.write(json.dumps(dict(prompt=user(r, s), completion=f"ANSWER: {chr(65 + (r['gold'] - s) % n)}", cls=cls[i], share=r["share"], src=r["src"], win=r["win"]), ensure_ascii=False) + "\n")
easy = [i for i, c in enumerate(cls) if c == "cons-right"]; rng.shuffle(easy); easy = easy[:a.easy_keep]
with open(out / "rows_easy.jsonl", "w", encoding="utf-8") as f:
    for i in easy:
        r = mcqs[i]; n = len(r["choices"])
        for s in range(min(a.rot, n)):
            f.write(json.dumps(dict(prompt=user(r, s), completion=f"ANSWER: {chr(65 + (r['vote'] - s) % n)}", share=r["share"], src=r["src"]), ensure_ascii=False) + "\n")
with open(out / "hard_wins.txt", "w") as f: f.write("\n".join(sorted({mcqs[i]["win"] for i in gated})))
with open(out / "questions.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(mcqs): f.write(json.dumps(dict(q=r["q"], choices=r["choices"], gold=r["gold"], win=r["win"], src=r["src"], cls=cls[i], share=r.get("share"), vote=r.get("vote"), pgold=r.get("pgold")), ensure_ascii=False) + "\n")
print(f"wrote rows_hard {len(gated)*4:,}+ rows, rows_easy {len(easy)*a.rot:,} rows"); print("UTR_DONE")
