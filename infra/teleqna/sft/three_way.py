#!/usr/bin/env python3
"""Arm J against the two arms it was built from.

Arm J exists to answer one question: can a single adapter hold arm G-full's
1,021 label-driven fixes AND the 203 rows only arm H's synthetic facts reached?
The headline will not say. This will:

  * only-H rows recovered  -- how much of arm H's unique knowledge survived
  * rows broken            -- arm H broke 926, arm G-full 357; where did J land
  * bucket C_hard          -- the two arms overlapped on only 25 of 471 there,
                              so it is where additivity shows up first
"""
import json, os, collections

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
armJ, armJp = load(f"{L}/otfull_armJ_armJ_nothink512.json")

sids = [s for s in base if basep[s] and s in armH and s in armG and s in armJ]
n = len(sids)
print(f"{n} rows scorable in all four (base-unparsed stripped)\n")

for nm, d in [("base", base), ("armH", armH), ("armG_full", armG), ("armJ", armJ)]:
    print(f"  {nm:10s} {sum(d[s] for s in sids)/n*100:6.2f}")
print(f"  armJ unparsed on the full file: {sum(1 for s in armJp if not armJp[s])}")
print()

wrong = [s for s in sids if not base[s]]
right = [s for s in sids if base[s]]
fh = {s for s in wrong if armH[s]}
fg = {s for s in wrong if armG[s]}
fj = {s for s in wrong if armJ[s]}
onlyH = fh - fg

print(f"  arm H's unique knowledge: {len(onlyH)} rows arm G-full misses and arm H gets")
print(f"    of those, arm J gets  : {len(onlyH & fj)}  ({len(onlyH & fj)/max(1,len(onlyH))*100:.1f}%)"
      f"  = +{len(onlyH & fj)/n*100:.2f} pts over arm G-full's reach")
print(f"  arm G-full's fixes      : {len(fg)}, of those arm J keeps {len(fg & fj)}"
      f"  ({len(fg & fj)/max(1,len(fg))*100:.1f}%)")
print(f"  arm J total fixes       : {len(fj)}   (G {len(fg)} | H {len(fh)} | union {len(fg|fh)})")
print()
for nm, d in [("armH", armH), ("armG_full", armG), ("armJ", armJ)]:
    print(f"  rows the base had right that {nm:10s} broke: {sum(1 for s in right if not d[s]):4d} / {len(right)}")
print()

print(f"  {'bucket':8s} {'n':>5s} {'base':>7s} {'armH':>7s} {'armG':>7s} {'armJ':>7s}   only-H  J gets")
for b in ["A", "B", "C", "C_hard"]:
    sel = [s for s in sids if buckets.get(s) == b]
    if not sel: continue
    m = len(sel)
    w = [s for s in sel if not base[s]]
    oh = {s for s in w if armH[s]} - {s for s in w if armG[s]}
    print(f"  {b:8s} {m:5d} " + " ".join(f"{sum(d[s] for s in sel)/m*100:7.2f}"
          for d in (base, armH, armG, armJ)) +
          f"   {len(oh):5d}  {len(oh & fj):5d}")
