#!/usr/bin/env python3
"""Arm F: the arm E tiers with the flip target stripped back to a bare letter.

The diagnosis this tests. In arm E the retain and anchor tiers carry a bare
`ANSWER: X` and the flip tier carries `<fact>. This is B) ..., not C) ...
ANSWER: B`. Measured afterwards:

    retain rows   arm E obeys the taught letter on ~100%  (97.5% of its errors
                  there are simply rows where the taught letter was wrong)
    flip rows     arm E obeys the taught letter on  33.6%

Same trainer, same epochs, same rows -- the only thing that differs is the
target's shape. So the fact-prefixed target is not being learned, and the flip
tier's potential is untapped: at 69.7% label accuracy and full obedience those
855 rows would go from 29.4% to ~69.7%, worth ~+3.4 points instead of the +0.77
arm E actually got.

If arm F beats arm E, the grounded-fact form was a net negative for this
objective and the recipe is "evidence-labelled low-margin rows, bare letters".
If it loses, the fact prefix is buying something the letter cannot, and the fix
is to train it harder rather than drop it. Either answer is worth having; right
now I am guessing.
"""
import argparse, json, os, sys, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=f"{ROOT}/data/train/eligible/armE_wide.jsonl")
ap.add_argument("--out", default=f"{ROOT}/data/train/eligible/armF_wide.jsonl")
a = ap.parse_args()

CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ref = {json.loads(l)["sample_id"]: json.loads(l) for l in open(CANON)}

rows = [json.loads(l) for l in open(a.src)]
n_changed = 0
for r in rows:
    letter = r["completion"].rstrip()[-1]
    bare = f"ANSWER: {letter}"
    if r["completion"] != bare:
        n_changed += 1
    r["completion"] = bare

with open(a.out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

c = collections.Counter(r["tier"] for r in rows)
print(f"wrote {len(rows)} rows -> {a.out}")
print(f"  tiers: {dict(c)}   ({n_changed} targets stripped to a bare letter)")

def gold(sid):
    return chr(65 + int(ref[sid]["answer"]))
for t in ["flip", "retain", "anchor"]:
    sel = [r for r in rows if r["tier"] == t]
    ok = sum(1 for r in sel if r["completion"][-1] == gold(r["sample_id"]))
    print(f"  AUDIT {t:7s} n={len(sel):5d}  label accuracy {ok/len(sel)*100:5.2f}%")
ok = sum(1 for r in rows if r["completion"][-1] == gold(r["sample_id"]))
print(f"  AUDIT overall n={len(rows)}  label accuracy {ok/len(rows)*100:.2f}%")
print("  (identical to arm E by construction -- only the target STRING changed)")
