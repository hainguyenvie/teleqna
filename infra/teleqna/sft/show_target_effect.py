#!/usr/bin/env python3
"""Same row, same label, two target shapes -- and what the model actually emits.

Arm E and arm F trained on identical rows with identical labels. The only
difference is the string after the prompt. This prints, for real rows: the two
training targets, and the two completions the trained models produced at eval.
"""
import json, os, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r
gold = {s: chr(65 + int(r["answer"])) for s, r in ref.items()}

E = {json.loads(l)["sample_id"]: json.loads(l)
     for l in open(f"{ROOT}/data/train/eligible/armE_wide.jsonl")}
F = {json.loads(l)["sample_id"]: json.loads(l)
     for l in open(f"{ROOT}/data/train/eligible/armF_wide.jsonl")}

def res(p):
    return {r["sample_id"]: r for r in json.load(open(p))["results"]}
eres = res(f"{R}/landscape/otfull_armEwide_armEwide_nothink512.json")
fres = res(f"{R}/landscape/otfull_armFwide_armFwide_nothink512.json")

flip = [s for s in E if E[s]["tier"] == "flip"]
taught_right = [s for s in flip if E[s]["completion"].rstrip()[-1] == gold[s]]
# the interesting cases: label was RIGHT, arm F obeyed it, arm E did not
cases = [s for s in taught_right
         if fres[s]["correct"] and not eres[s]["correct"]]
print(f"flip rows {len(flip)}; label correct on {len(taught_right)}; "
      f"of those, arm F right & arm E wrong on {len(cases)}\n")

for sid in cases[:3]:
    r = ref[sid]
    print("=" * 100)
    print(f"{sid}   gold = {gold[sid]}   taught label = {E[sid]['completion'].rstrip()[-1]}")
    print(f"Q: {r['question'][:150]}")
    for i, c in enumerate(r["choices"]):
        print(f"   {chr(65+i)}) {c[:95]}" + ("   <-- gold" if i == int(r["answer"]) else ""))
    print(f"\n  --- what arm E was trained to say ---")
    print(f"  {E[sid]['completion']!r}")
    print(f"  --- what arm E actually said at eval ---")
    print(f"  {eres[sid]['completion'].strip()[:400]!r}")
    print(f"  parsed={eres[sid]['parsed']!r}  correct={eres[sid]['correct']}")
    print(f"\n  --- what arm F was trained to say ---")
    print(f"  {F[sid]['completion']!r}")
    print(f"  --- what arm F actually said at eval ---")
    print(f"  {fres[sid]['completion'].strip()[:200]!r}")
    print(f"  parsed={fres[sid]['parsed']!r}  correct={fres[sid]['correct']}")
    print()

# and the aggregate: how often each arm emitted exactly the taught letter
def obey(resd, traind, tier):
    sel = [s for s in traind if traind[s]["tier"] == tier and s in resd]
    ok = sum(1 for s in sel if resd[s]["parsed"] == traind[s]["completion"].rstrip()[-1])
    return ok, len(sel)
print("=" * 100)
print("how often the trained model emits EXACTLY the letter it was taught:")
for tier in ["flip", "retain", "anchor"]:
    oe, ne = obey(eres, E, tier)
    of, nf = obey(fres, F, tier)
    print(f"  {tier:7s}  arm E (prose target on flip only) {oe}/{ne} = {oe/ne*100:5.1f}%"
          f"   |   arm F (bare letter) {of}/{nf} = {of/nf*100:5.1f}%")
