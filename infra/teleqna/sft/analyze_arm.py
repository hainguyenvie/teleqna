#!/usr/bin/env python3
"""Read an adapter's ot-full result the way arm V had to be read.

Two confounds sank the first reading of arm V and both are handled here:

  * rows the base leaves UNPARSED score zero by construction, so an adapter that
    only fixes formatting looks like it learned something. They are reported
    separately and stripped from every knowledge slice.
  * rows the adapter TRAINED on cannot show transfer. The trained/untrained split
    is the result; the headline is not.

Transfer is read on dev1000 + clean_holdout500, which were excluded at the source
when the training file was built.
"""
import argparse, json, os, collections, math

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ap = argparse.ArgumentParser()
ap.add_argument("--result", required=True, help="the *_<arm>_nothink512.json file")
ap.add_argument("--base", required=True, help="the *_base_nothink512.json file")
ap.add_argument("--train", required=True, help="the training jsonl, for tier slices")
ap.add_argument("--name", default="arm")
a = ap.parse_args()

ref = {json.loads(l)["sample_id"]: json.loads(l) for l in open(CANON)}
buckets = json.load(open(f"{R}/error_buckets.json"))

def load(p):
    d = json.load(open(p))
    return ({r["sample_id"]: bool(r["correct"]) for r in d["results"]},
            {r["sample_id"]: bool(r["parsed"]) for r in d["results"]},
            d["summary"])

arm, armp, arms = load(a.result)
base, basep, bases = load(a.base)

# arm V's file predates the tier field; it is all one tier there.
tier = {}
for l in open(a.train):
    r = json.loads(l)
    tier[r["sample_id"]] = r.get("tier", "trained")

held = set(json.load(open(f"{ROOT}/data/clean_holdout500.json")))
for l in open(f"{ROOT}/data/dev1000.jsonl"):
    held.add(json.loads(l)["sample_id"])

ids = sorted(set(arm) & set(base))

def mcnemar(sel):
    w = sum(1 for s in sel if arm[s] and not base[s])
    l = sum(1 for s in sel if base[s] and not arm[s])
    if w + l == 0:
        return w, l, 1.0
    # two-sided exact binomial against p=0.5
    n, k = w + l, min(w, l)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return w, l, p

def line(name, sel):
    sel = [s for s in sel if s in arm]
    if not sel:
        return
    b = sum(base[s] for s in sel) / len(sel) * 100
    v = sum(arm[s] for s in sel) / len(sel) * 100
    w, l, p = mcnemar(sel)
    print("  %-38s %5d  %6.2f -> %6.2f  %+6.2f   %4d/%-4d  p=%.4g"
          % (name, len(sel), b, v, v - b, w, l, p))

print(f"=== {a.name} ===")
print(f"base    {bases['accuracy']*100:.2f}%  unparsed {bases['unparsed']}")
print(f"{a.name:7s} {arms['accuracy']*100:.2f}%  unparsed {arms['unparsed']}")
print(f"headline delta {(arms['accuracy']-bases['accuracy'])*100:+.2f} pts\n")

unp = [s for s in ids if not basep[s]]
fixed = sum(1 for s in unp if arm[s])
print(f"format repair: base left {len(unp)} rows unparsed, {a.name} answers "
      f"{fixed} of them correctly = {fixed/len(ids)*100:+.2f} pts of the headline")
print("everything below EXCLUDES those rows.\n")

ok = [s for s in ids if basep[s]]
print("  %-38s %5s  %-20s %-9s %s" % ("slice", "n", "base -> arm", "win/lose", "p"))
line("all base-parsable rows", ok)
print()
for t in sorted(set(tier.values())):
    line(f"trained: {t}", [s for s in ok if tier.get(s) == t])
line("never trained", [s for s in ok if s not in tier])
print()
line("HOLDOUT dev1000 + clean500", [s for s in ok if s in held])
print()
for b in ["A", "B", "C", "C_hard"]:
    line(f"bucket {b}", [s for s in buckets[b] if s in ok])
print()
for b in ["C", "C_hard"]:
    line(f"bucket {b}, never trained", [s for s in buckets[b] if s in ok and s not in tier])
