#!/usr/bin/env python3
"""Arm G: arm F's recipe over the whole benchmark, with bucket A protected.

Arm F established that the model obeys a bare-letter target essentially
perfectly, so the score becomes the training set's label accuracy and the job is
to maximise that. Scored schemes (`infra/label_schemes.py`,
`infra/scheme_bybucket.py`):

    scheme                              labels   acc     bucket A if obeyed
    vote@32 only                          8932  80.23%
    evidence only                         8917  86.83%     -0.80 pts   <-- trap
    evidence, vote when margin==1.0       8920  87.00%     -0.36 pts
    evidence, vote when margin>=0.90      8921  87.03%     +0.27 pts   <-- chosen

The trap is worth naming: "evidence only" has the highest bucket-C accuracy
(62.07%) and looks best on the headline, but the base model answers bucket A
correctly 98.98% of the time while the evidence labels it at 97.39% -- so full
obedience would DRAG 5,000 already-correct rows down. Deferring to the model's
own high-margin vote where it is confident keeps that half intact and costs
almost nothing on C.

Holdout is excluded so transfer stays readable. A submission model would rerun
this with `--include-holdout`, which is a separate decision.
"""
import argparse, json, os, sys, collections, random

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
sys.path.insert(0, f"{ROOT}/infra")
from eval_dev_vllm import build_prompt

R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ap = argparse.ArgumentParser()
ap.add_argument("--margin", type=float, default=0.90)
ap.add_argument("--include-holdout", action="store_true")
ap.add_argument("--out", default=f"{ROOT}/data/train/eligible/armG.jsonl")
a = ap.parse_args()

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r
gold = {s: chr(65 + int(r["answer"])) for s, r in ref.items()}

ev = {}
# The holdout file only exists once branch A has judged those 1,068 rows; it is
# optional so the same script still builds the holdout-excluded arm G.
for f in ["esc2730_ragstrong8_reparsed.json", "blind6202_ragstrong8_reparsed.json",
          "holdout1068_ragstrong8_reparsed.json"]:
    if not os.path.exists(f"{R}/landscape/{f}"):
        print(f"  (no {f} — skipping)")
        continue
    for r in json.load(open(f"{R}/landscape/{f}")):
        ev[r["sample_id"]] = r["letter"]

SHIFT = {"base": 0, "cot": 0, "p1": 1, "p2": 2}
votes = collections.defaultdict(collections.Counter)
for arm, s in SHIFT.items():
    for l in open(f"{R}/passk_otfull_{arm}/all.jsonl"):
        r = json.loads(l)
        nc = r["n_choices"]
        for got in r["letters"]:
            if got:
                votes[r["sample_id"]][(ord(got) - 65 - s) % nc] += 1
pvote, margin = {}, {}
for sid, c in votes.items():
    if c:
        tot = sum(c.values())
        top, n1 = c.most_common(1)[0]
        pvote[sid] = chr(65 + top)
        margin[sid] = n1 / tot

held = set(json.load(open(f"{ROOT}/data/clean_holdout500.json")))
for l in open(f"{ROOT}/data/dev1000.jsonl"):
    held.add(json.loads(l)["sample_id"])
assert len(held) > 1400

rows = []
for sid in sorted(ref):
    if not a.include_holdout and sid in held:
        continue
    if sid not in pvote or sid not in ev:
        continue
    confident = margin.get(sid, 0) >= a.margin
    letter = pvote[sid] if confident else ev[sid]
    if not letter:
        continue
    rows.append({"sample_id": sid, "prompt": build_prompt(ref[sid]),
                 "completion": f"ANSWER: {letter}",
                 "tier": "keep" if confident else
                         ("retain" if letter == pvote[sid] else "flip")})

random.seed(3)
random.shuffle(rows)
with open(a.out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

c = collections.Counter(r["tier"] for r in rows)
print(f"wrote {len(rows)} rows -> {a.out}")
print(f"  tiers: {dict(c)}")
print(f"  holdout rows present: {sum(1 for r in rows if r['sample_id'] in held)}"
      + ("  (intentional)" if a.include_holdout else "  (must be 0)"))
for t in sorted(c):
    sel = [r for r in rows if r["tier"] == t]
    ok = sum(1 for r in sel if r["completion"][-1] == gold[r["sample_id"]])
    print(f"  AUDIT {t:7s} n={len(sel):5d}  label accuracy {ok/len(sel)*100:5.2f}%")
ok = sum(1 for r in rows if r["completion"][-1] == gold[r["sample_id"]])
print(f"  AUDIT overall n={len(rows)}  label accuracy {ok/len(rows)*100:.2f}%")
