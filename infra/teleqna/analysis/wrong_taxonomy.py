#!/usr/bin/env python3
"""Taxonomy of the remaining wrong answers of a checkpoint (measurement only, gold used to select the wrong set):
(1) label-free question-form buckets (negation, true/false statement, numeric, definition, acronym, release-tag, compare,
'which of the following'), accuracy per bucket; (2) OTel-31B reads the question's RAG windows -> learnable-from-source
(judge=gold) / label doubtful (judge=model) / unsupported (X); (3) OTel-31B closed-book on the same wrong set.
Usage: wrong_taxonomy.py TAG [--n 1200]"""
import json, re, sys, random, argparse, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; LS = R / "results/landscape"; JUDGE = str(Path.home() / "projects/_shared/models/OTel-2.0-31B-IT")
ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--n", type=int, default=1200); ap.add_argument("--gpu-mem", type=float, default=0.85); a = ap.parse_args()
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: x for x in d if isinstance(x, dict) and "sample_id" in x}
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
res = L(LS / f"{a.tag}_otfull10000_base_nothink512.json"); Q = list(test)
forms = [("negation (NOT/EXCEPT/FALSE)", re.compile(r"\b(NOT|EXCEPT|FALSE|incorrect|does not|is not)\b", re.I)),
         ("which statement is true/correct", re.compile(r"which (of the following )?(statement|option)s? (is|are) (true|correct)", re.I)),
         ("which of the following", re.compile(r"which of the following", re.I)), ("how many / number", re.compile(r"how many|number of|how much", re.I)),
         ("max/min/maximum", re.compile(r"\b(maximum|minimum|max|min|highest|lowest|largest|smallest)\b", re.I)),
         ("what is X (definition)", re.compile(r"^what (is|are|does) ", re.I)), ("acronym (stand for)", re.compile(r"stand for|abbreviat|acronym", re.I)),
         ("purpose/why", re.compile(r"purpose|why|what is the (main )?(goal|objective|role|function)", re.I)), ("3GPP release tag", re.compile(r"\[3GPP Release", re.I)),
         ("difference/compare", re.compile(r"differ|compar|versus|\bvs\b|between", re.I)), ("when/what year", re.compile(r"\bwhen\b|what year|which year", re.I)),
         ("who/which organization", re.compile(r"\bwho\b|which (organization|body|group|company|entity)", re.I))]
print("question form              n     acc(%)   wrong   share-of-wrong")
W = [q for q in Q if not res[q]["correct"]]
for name, rx in forms:
    qs = [q for q in Q if rx.search(test[q]["question"])]; w = [q for q in qs if not res[q]["correct"]]
    if qs: print(f"{name:26s}{len(qs):6d}  {100*(len(qs)-len(w))/len(qs):6.1f}  {len(w):6d}  {100*len(w)/len(W):5.1f}%")
print(f"total wrong {len(W)}")
# option-shape of wrong: gold vs picked
def shape(q):
    r = test[q]; ch = r["choices"]; p = res[q].get("parsed") or ""; pi = ord(p) - 65 if p else -1
    g = r["answer"]; out = []
    if pi < 0 or pi >= len(ch): return ["unparsed"]
    if re.search(r"(?i)all of the above|both", ch[g]): out.append("gold=All, missed")
    if re.search(r"(?i)all of the above|both", ch[pi]): out.append("picked All, gold not")
    if re.search(r"(?i)none of the above", ch[pi]) or re.search(r"(?i)none of the above", ch[g]): out.append("None-of-above involved")
    if re.fullmatch(r"[\d.,%\s\-]+", ch[g].strip()): out.append("numeric gold")
    a1, a2 = set(re.findall(r"[a-z0-9]+", ch[g].lower())), set(re.findall(r"[a-z0-9]+", ch[pi].lower()))
    if a1 and a2 and len(a1 & a2) / len(a1 | a2) >= 0.5: out.append("gold~picked near-duplicate (jaccard>=0.5)")
    if re.search(r"\b(increase|decrease|higher|lower|more|less|greater|smaller|faster|slower|before|after|up|down)\b", (ch[g] + " " + ch[pi]).lower()): out.append("directional pair")
    if len(ch[pi]) > len(ch[g]) * 1.5: out.append("picked much longer")
    if len(ch[g]) > len(ch[pi]) * 1.5: out.append("picked much shorter")
    return out or ["other"]
c = collections.Counter(s for q in W for s in shape(q)); print("\nwrong-answer shape:"); [print(f"  {k:44s} {v:5d} ({100*v/len(W):.1f}%)") for k, v in c.most_common()]
# 31B judge with the question's windows, and closed-book
q2w = collections.defaultdict(list)
for f in ("data/kit/windows_keep.jsonl", "data/kit/windows_rest.jsonl"):
    for l in open(R / f, encoding="utf-8"):
        w = json.loads(l)
        for q in w.get("for_q", []): q2w[q].append(w["text"])
rng = random.Random(0); S = rng.sample(W, min(a.n, len(W)))
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(JUDGE, local_files_only=True); llm = LLM(model=JUDGE, gpu_memory_utilization=a.gpu_mem, max_model_len=16384, dtype="bfloat16")
def mc(q): r = test[q]; return r["question"] + "\n\n" + "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"]))
PS = ("Using ONLY the reference passages, answer the multiple-choice question. If the passages do not support any option, or support more than one, answer 'ANSWER: X'.\n\nPASSAGES:\n{doc}\n\n{mc}\n\nReply with exactly one line: ANSWER: <letter or X>")
PC = ("Answer the multiple-choice question from your own knowledge. Reply with exactly one line: ANSWER: <letter>.\n\n{mc}")
ps = [tok.apply_chat_template([{"role": "user", "content": PS.format(doc="\n\n".join(f"[{i+1}] {t[:2500]}" for i, t in enumerate(q2w.get(q, [])[:5])) or "(no passages)", mc=mc(q))}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in S]
pc = [tok.apply_chat_template([{"role": "user", "content": PC.format(mc=mc(q))}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in S]
os_ = [o.outputs[0].text for o in llm.generate(ps, SamplingParams(temperature=0.0, max_tokens=384))]; oc = [o.outputs[0].text for o in llm.generate(pc, SamplingParams(temperature=0.0, max_tokens=384))]
def parse(o): mm = re.findall(r"ANSWER\s*:\s*\**\s*([A-EX])\b", o.upper()); return mm[-1] if mm else "?"
cs = collections.Counter(); cc = collections.Counter(); both = collections.Counter(); rows = []
for q, a1, a2 in zip(S, os_, oc):
    g = chr(65 + test[q]["answer"]); m = res[q].get("parsed") or ""; j1, j2 = parse(a1), parse(a2); has = bool(q2w.get(q))
    k1 = ("no windows" if not has else "judge+source=gold (learnable)" if j1 == g else "judge+source=model (label doubtful)" if j1 == m else "unsupported (X)" if j1 == "X" else "judge+source=third")
    k2 = ("31B closed-book=gold" if j2 == g else "31B closed-book=model" if j2 == m else "31B closed-book other")
    cs[k1] += 1; cc[k2] += 1; both[(k1, k2)] += 1; rows.append(dict(q=q, subject=test[q]["subject"], gold=g, model=m, src=k1, cb=k2))
n = len(S); print(f"\n31B with the question's windows (n={n}):"); [print(f"  {k:40s} {v:5d} ({100*v/n:.1f}%)") for k, v in cs.most_common()]
print("31B closed-book on the same wrong set:"); [print(f"  {k:40s} {v:5d} ({100*v/n:.1f}%)") for k, v in cc.most_common()]
print("cross (source-class x closed-book):"); [print(f"  {k[0]:40s} x {k[1]:24s} {v:4d}") for k, v in both.most_common(8)]
sub = collections.defaultdict(collections.Counter)
for r in rows: sub[r["subject"]][r["src"]] += 1
print("by subject (source-class):"); [print(f"  {s:26s} " + " | ".join(f"{k[:22]} {v}" for k, v in c.most_common(3))) for s, c in sub.items()]
json.dump(rows, open(R / f"results/kit/wrong_taxonomy_{a.tag}.json", "w"), ensure_ascii=False); print("TAXONOMY_DONE")
