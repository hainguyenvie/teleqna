#!/usr/bin/env python3
"""Did training move probability toward the gold option on rows it did not fix? Compare letter probabilities of two
checkpoints on the buckets: unfixed-with-source, fixed, broke, base-right-kept. Gold used for measurement only."""
import json, sys, statistics as st
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; A, B = sys.argv[1], sys.argv[2]
PA = json.load(open(R / f"results/kit/letterprobs_{A}.json")); PB = json.load(open(R / f"results/kit/letterprobs_{B}.json"))
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: bool(x["correct"]) for x in d}
Bc = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json"); Tc = L(R / f"results/landscape/{sys.argv[3]}_otfull10000_base_nothink512.json")
cov1 = set(json.load(open(R / "results/kit/coverage_sets.json"))["cov1"]); ids = list(Bc)
buckets = {"base-wrong, unfixed, source in windows": [q for q in ids if not Bc[q] and not Tc[q] and q in cov1],
           "base-wrong, unfixed, no source": [q for q in ids if not Bc[q] and not Tc[q] and q not in cov1],
           "fixed": [q for q in ids if not Bc[q] and Tc[q]], "broke": [q for q in ids if Bc[q] and not Tc[q]], "kept right": [q for q in ids if Bc[q] and Tc[q]]}
def rank(p, g): return sorted(range(len(p)), key=lambda i: -p[i]).index(g) + 1
for name, qs in buckets.items():
    ga = [PA[q][test[q]["answer"]] for q in qs]; gb = [PB[q][test[q]["answer"]] for q in qs]
    up = sum(b > a + 0.05 for a, b in zip(ga, gb)); down = sum(b < a - 0.05 for a, b in zip(ga, gb))
    ra = [rank(PA[q], test[q]["answer"]) for q in qs]; rb = [rank(PB[q], test[q]["answer"]) for q in qs]
    print(f"{name:42s} n={len(qs):5d} | gold prob mean {st.mean(ga):.3f} -> {st.mean(gb):.3f} | up(>+.05) {up/len(qs)*100:4.1f}% down {down/len(qs)*100:4.1f}% | gold rank 2nd: {sum(r==2 for r in ra)/len(qs)*100:4.1f}% -> {sum(r==2 for r in rb)/len(qs)*100:4.1f}%  rank>=3: {sum(r>=3 for r in ra)/len(qs)*100:4.1f}% -> {sum(r>=3 for r in rb)/len(qs)*100:4.1f}%")
print("PARTIAL_DONE")
