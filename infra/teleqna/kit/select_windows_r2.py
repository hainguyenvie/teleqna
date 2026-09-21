#!/usr/bin/env python3
"""Label-free refinement R2: keep a window only if the letter it induces (single-window context) differs from the
closed-book letter AND equals the majority letter over all single-window predictions for that question.
No gold used. Writes data/kit/windows_r2.json."""
import json, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: (x.get("parsed") or "") for x in d if isinstance(x, dict) and "sample_id" in x}
base = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json"); single = L(R / "results/landscape/win_single_q3_8b_base_nothink512.json")
perq = collections.defaultdict(list)
for k, p in single.items():
    q, w = k.split("|"); perq[q].append((w, p))
keep = set()
for q, lst in perq.items():
    letters = collections.Counter(p for w, p in lst if p)
    if not letters: continue
    maj = letters.most_common(1)[0][0]
    for w, p in lst:
        if p and p != base[q] and p == maj: keep.add(w)
json.dump(sorted(keep), open(R / "data/kit/windows_r2.json", "w"))
print(f"R2 windows {len(keep)} -> data/kit/windows_r2.json (label-free)")
