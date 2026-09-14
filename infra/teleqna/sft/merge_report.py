#!/usr/bin/env python3
"""Did weight-space composition rescue any of arm H's 203 unique rows?

Arm J answered the data-mixing version of this question with 17 of 203. The
merges answer the weight-space version, where there are no competing gradients
at all -- and the s sweep is monotonically decreasing (87.24 / 87.15 / 86.52 /
84.09), which already says no sweet spot exists. What is left to establish is
whether the small s=0.3 loss buys anything at all, or whether arm H's delta is
pure cost from the first increment.
"""
import json, os

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R, L = f"{ROOT}/results", f"{ROOT}/results/landscape"
bj = json.load(open(f"{R}/error_buckets.json"))
buckets = {s: b for b, v in bj.items() for s in v}

def load(p):
    d = json.load(open(p))
    return ({r["sample_id"]: bool(r["correct"]) for r in d["results"]},
            {r["sample_id"]: bool(r["parsed"]) for r in d["results"]})

base, basep = load(f"{L}/otfull_armH_base_nothink512.json")
armH, _ = load(f"{L}/otfull_armH_armH_nothink512.json")
armG, _ = load(f"{L}/otfull_armGfull_armGfull_nothink512.json")
armJ, _ = load(f"{L}/otfull_armJ_armJ_nothink512.json")
mg = {s: load(f"{L}/otfull_mergeGH_{s}_nothink512.json")[0] for s in ["m03", "m06", "m10"]}

sids = [s for s in base if basep[s] and s in armH and s in armG and s in armJ
        and all(s in m for m in mg.values())]
n = len(sids)
wrong = [s for s in sids if not base[s]]
right = [s for s in sids if base[s]]
fh = {s for s in wrong if armH[s]}
fg = {s for s in wrong if armG[s]}
onlyH = fh - fg

print(f"{n} rows scorable in all seven systems; arm H holds {len(onlyH)} rows arm G-full misses\n")
hdr = f"  {'system':10s} {'ot-full':>8s}  {'only-H won':>12s}  {'G fixes kept':>14s}  {'broken':>7s}"
print(hdr)
rows = [("armG_full", armG), ("m03", mg["m03"]), ("m06", mg["m06"]),
        ("m10", mg["m10"]), ("armJ", armJ), ("armH", armH)]
for nm, d in rows:
    fj = {s for s in wrong if d[s]}
    print(f"  {nm:10s} {sum(d[s] for s in sids)/n*100:8.2f}  "
          f"{len(onlyH & fj):5d} /{len(onlyH):<5d}  "
          f"{len(fg & fj):6d} /{len(fg):<5d}  "
          f"{sum(1 for s in right if not d[s]):7d}")

ch = [s for s in sids if buckets.get(s) == "C_hard"]
ohc = onlyH & set(ch)
print(f"\n  C_hard (n={len(ch)}), arm H holds {len(ohc)} of them alone:")
for nm, d in rows[:-1]:
    won = len(ohc & {s for s in ch if d[s]})
    print(f"    {nm:10s} {sum(d[s] for s in ch)/len(ch)*100:6.2f}   only-H won {won}/{len(ohc)}")
