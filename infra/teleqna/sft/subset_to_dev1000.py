#!/usr/bin/env python3
"""Restrict an ot-full-10000 result to the dev-1000 rows, for a same-question read.

dev-1000 is a subset of ot-full-10000, so a run over the full 10,000 already
contains an answer for every dev-1000 question. Pulling those 1,000 out costs no
GPU and gives a figure on exactly the questions the twelve 8B arms were scored
on, which a headline over 10,000 rows does not: two models can differ by several
points on 10,000 rows purely because the extra 9,000 have a different subject mix
from dev-1000.

What this fixes and what it does not:

  fixed        the question set. Same 1,000 items, same golds, same scorer.
  measured     the serving stack. The 8B arms ran on vLLM 0.11.0 and the 122B
               can only run on the nightly, so this was going to be a forbidden
               cross-stack read. It was settled by moving the 8B up instead:
               base thinking scores 75.50 on the nightly against 75.20 on
               0.11.0, a +0.30pp stack delta, which is inside the 0.50pp A/A
               floor measured from scoring a byte-identical adapter twice.

So the stack is not a confound worth carrying here, and both references below
are quoted as measured. The remaining honest caveat is smaller and different:
0.30pp is not zero, so gaps of that order mean nothing either way.
"""
import json
import sys

RES = sys.argv[1]
DEV = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/tensara/projects/telelogs/runs/teleqna-sft/data/dev1000.jsonl"

want, subj = set(), {}
for line in open(DEV):
    if line.strip():
        r = json.loads(line)
        want.add(r["sample_id"])
        subj[r["sample_id"]] = r.get("subject", "?")

d = json.load(open(RES))
rows = [r for r in d["results"] if r["sample_id"] in want]
missing = len(want) - len(rows)

print(f"=== {RES.split('/')[-1]} restricted to dev-1000")
print(f"full-set accuracy : {d['summary']['accuracy']:.4f} over {d['summary']['total']}")
if missing:
    print(f"WARN: {missing} dev-1000 rows absent from this result — "
          f"the subset figure below is over {len(rows)}, not 1000")

n = len(rows)
if n == 0:
    sys.exit(3)
ok = sum(1 for r in rows if r["correct"])
unp = sum(1 for r in rows if not r["parsed"])
print(f"dev-1000 subset   : {ok}/{n} = {ok/n:.4f}   unparsed={unp}")

per = {}
for r in rows:
    s = subj[r["sample_id"]]
    v = per.setdefault(s, [0, 0])
    v[0] += 1
    v[1] += int(bool(r["correct"]))
print(f"\n  {'subject':<26s} {'n':>5s} {'acc':>7s}")
for s, (m, k) in sorted(per.items(), key=lambda kv: -kv[1][0]):
    print(f"  {s:<26s} {m:5d} {k/m:7.4f}")

print("\n8B references on dev-1000 (stack delta measured at +0.30pp, inside noise):")
print("  base thinking, nightly   0.7550")
print("  base thinking, 0.11.0    0.7520")
print("  dpo3@200 thinking        0.7650   (best 8B arm, 0.11.0)")
print(f"\ngap vs 8B base (nightly) : {(ok/n - 0.7550)*100:+.2f}pp")
print(f"gap vs best 8B arm       : {(ok/n - 0.7650)*100:+.2f}pp")
print("noise floor for reading these: 0.50pp (A/A, identical weights scored twice)")
