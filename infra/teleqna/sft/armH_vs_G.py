#!/usr/bin/env python3
"""Does arm H know anything arm G-full does not?

Arm H is net -2.54, but its bucket C_hard went 3.48 -> 17.21 with 77 wins
against 10 losses, on questions it has never seen. That is real transfer, the
first clean one in this campaign. The question that decides whether it earns
another run: are the rows it fixes the SAME rows arm G-full already fixes from
the labels, or different ones? Additive means a combined arm is worth building;
overlapping means arm H is a slower road to where we already stand.
"""
import json, os, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
L = f"{R}/landscape"
buckets = json.load(open(f"{R}/error_buckets.json"))

def load(p):
    d = json.load(open(p))
    return ({r["sample_id"]: bool(r["correct"]) for r in d["results"]},
            {r["sample_id"]: bool(r["parsed"]) for r in d["results"]})

base, basep = load(f"{L}/otfull_armH_base_nothink512.json")
armH, _ = load(f"{L}/otfull_armH_armH_nothink512.json")
armG, _ = load(f"{L}/otfull_armGfull_armGfull_nothink512.json")

# strip base-unparsed rows, exactly as every other table in this campaign does
sids = [s for s in base if basep[s] and s in armH and s in armG]
print(f"{len(sids)} rows scorable in all three (base-unparsed stripped)\n")

def report(name, sel):
    n = len(sel)
    if not n: return
    hb, hh, hg = (sum(d[s] for s in sel) for d in (base, armH, armG))
    wrong = [s for s in sel if not base[s]]
    fh = {s for s in wrong if armH[s]}
    fg = {s for s in wrong if armG[s]}
    onlyH, onlyG, both = fh - fg, fg - fh, fh & fg
    union = fg | fh
    right = [s for s in sel if base[s]]
    bh = sum(1 for s in right if not armH[s])
    bg = sum(1 for s in right if not armG[s])
    print(f"  {name:12s} n={n:5d}   base {hb/n*100:5.2f}   armH {hh/n*100:5.2f}   armG {hg/n*100:5.2f}")
    print(f"    base-wrong {len(wrong):5d}: fixed by G {len(fg):4d} | by H {len(fh):4d} | "
          f"both {len(both):4d} | only-H {len(onlyH):4d} | only-G {len(onlyG):4d}")
    if wrong:
        print(f"    union of fixes {len(union):4d}  ->  +{len(onlyH)/n*100:5.2f} pts available to arm G "
              f"from rows only H gets")
    print(f"    base-right {len(right):5d}: broken by G {bg:4d} | by H {bh:4d}")
    print()

report("ALL", sids)
for b in ["A", "B", "C", "C_hard"]:
    report(f"bucket {b}", [s for s in sids if buckets.get(s) == b])

# where does the damage go? if arm H collapsed onto one letter it is a style
# artefact, not forgetting.
import sys
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ref = {}
for l in open(CANON):
    r = json.loads(l)
    ref[r["sample_id"]] = r
raw = {r["sample_id"]: r for r in json.load(open(f"{L}/otfull_armH_armH_nothink512.json"))["results"]}
broke = [s for s in sids if base[s] and not armH[s]]
got = collections.Counter(raw[s].get("answer") or raw[s].get("parsed_answer") for s in broke)
gold = collections.Counter(chr(65 + int(ref[s]["answer"])) for s in broke)
print(f"the {len(broke)} rows arm H broke:")
print(f"  arm H answered : {dict(sorted(got.items(), key=lambda kv: -kv[1]))}")
print(f"  gold was       : {dict(sorted(gold.items(), key=lambda kv: -kv[1]))}")
nopt = collections.Counter(len(ref[s]["choices"]) for s in broke)
print(f"  option counts  : {dict(sorted(nopt.items()))}")
