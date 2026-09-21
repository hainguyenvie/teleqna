#!/usr/bin/env python3
"""Read the t1gate results and decide. Prints GATE_PASS or GATE_FAIL."""
import json
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: bool(x["correct"]) for x in d if isinstance(x, dict) and "sample_id" in x}
qs = json.load(open(R / "data/kit/tier1/gate_questions.json"))
B = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json")
acc = lambda D: sum(D[q] for q in qs) / len(qs) * 100
b = acc(B); res = {n: acc(L(R / f"results/landscape/t1gate_{n}_q3_8b_base_nothink512.json")) for n in ("verbatim", "rewrite", "factview", "all")}
v = res["verbatim"] - b
print(f"n={len(qs)} base {b:.2f} | verbatim {res['verbatim']:.2f} (+{v:.2f}) | rewrite {res['rewrite']:.2f} ({(res['rewrite']-b)/max(v,1e-9)*100:.0f}% of verbatim lift) | "
      f"factview {res['factview']:.2f} ({(res['factview']-b)/max(v,1e-9)*100:.0f}%) | all {res['all']:.2f}")
ok = (res["rewrite"] - b) >= 0.6 * v and res["all"] >= res["verbatim"] - 2.0 and v >= 3.0
print("GATE_PASS" if ok else "GATE_FAIL")
