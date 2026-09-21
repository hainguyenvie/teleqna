#!/usr/bin/env python3
"""Ceiling of ensembling/consolidation: majority vote across checkpoints x orderings on the test set (gold for measurement only).
Also: agreement structure (all agree / split) and accuracy in each; the 'consolidation ceiling' = vote accuracy."""
import json, sys, collections, itertools
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; LS = R / "results/landscape"
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: x for x in d if isinstance(x, dict) and "sample_id" in x}
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
rot = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull_rot1.jsonl", encoding="utf-8"))}
Q = list(test); tags = sys.argv[1:]
votes = collections.defaultdict(collections.Counter); per = {}
for t in tags:
    a = L(LS / f"{t}_otfull10000_base_nothink512.json"); b = L(LS / f"{t}_otfull_rot1_base_nothink512.json")
    per[t + ":o"] = {}; per[t + ":r"] = {}
    for q in Q:
        p = a[q].get("parsed") or ""
        if p and ord(p) - 65 < len(test[q]["choices"]): txt = test[q]["choices"][ord(p) - 65]; votes[q][txt] += 1; per[t + ":o"][q] = txt
        p = b[q].get("parsed") or ""
        if p and ord(p) - 65 < len(rot[q]["choices"]): txt = rot[q]["choices"][ord(p) - 65]; votes[q][txt] += 1; per[t + ":r"][q] = txt
gold = {q: test[q]["choices"][test[q]["answer"]] for q in Q}
def vote_acc(keys):
    ok = 0
    for q in Q:
        c = collections.Counter(per[k].get(q) for k in keys if per[k].get(q) is not None)
        if c and c.most_common(1)[0][0] == gold[q]: ok += 1
    return 100 * ok / len(Q)
print("single (orig / rot1):"); [print(f"  {t:16s} {vote_acc([t+':o']):.2f} / {vote_acc([t+':r']):.2f}   orig+rot1 vote {vote_acc([t+':o', t+':r']):.2f}") for t in tags]
allk = [k for t in tags for k in (t + ":o", t + ":r")]
print(f"\nmajority vote over all {len(allk)} (ckpt x ordering): {vote_acc(allk):.2f}")
print(f"majority vote over orig orderings only ({len(tags)}): {vote_acc([t+':o' for t in tags]):.2f}")
for n in (2, 3):
    best = max(((vote_acc([k for t in c for k in (t+':o', t+':r')]), c) for c in itertools.combinations(tags, n)), key=lambda x: x[0]); print(f"best {n}-ckpt vote: {best[0]:.2f} {best[1]}")
# agreement structure
oracle = 100 * sum(1 for q in Q if gold[q] in votes[q]) / len(Q)
unan = [q for q in Q if len(votes[q]) == 1]; print(f"\noracle-union (any vote right): {oracle:.2f}; unanimous {len(unan)} q -> acc {100*sum(1 for q in unan if votes[q].most_common(1)[0][0]==gold[q])/len(unan):.1f}")
split = [q for q in Q if len(votes[q]) > 1]
top_ok = sum(1 for q in split if votes[q].most_common(1)[0][0] == gold[q]); any_ok = sum(1 for q in split if gold[q] in votes[q])
print(f"split {len(split)} q: majority right {top_ok} ({100*top_ok/len(split):.1f}%), some vote right {any_ok}, none right {len(split)-any_ok}")
# by vote share of the top answer
print("top-answer share bucket: n, majority acc")
for lo, hi in ((0.3, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.99), (0.99, 1.01)):
    qs = [q for q in Q if lo <= votes[q].most_common(1)[0][1] / max(1, sum(votes[q].values())) < hi]
    if qs: print(f"  [{lo:.2f},{hi:.2f}) n={len(qs):5d} acc={100*sum(1 for q in qs if votes[q].most_common(1)[0][0]==gold[q])/len(qs):5.1f}")
print("ENS_DONE")
