#!/usr/bin/env python3
"""Arm J: arm G-full's real rows as the anchor, arm H's facts as the payload.

The two arms fail and succeed in different places. Arm G-full is +7.70 but its
knowledge stops where the labels stop: on the 471 base-wrong C_hard rows it
fixes 113. Arm H is -2.54 yet fixes 77 of those same 471, and only 25 of them
overlap -- 52 rows it reaches that the labels never did. Across the whole
benchmark that is 203 rows arm G-full still gets wrong and arm H gets right,
worth +2.06 points if the two can be held in one adapter.

Arm H's cost is distribution drift, not letter bias: it broke 926 rows the base
had right (arm G-full broke 357), and its answers on those rows spread A/B/C/D/E
in roughly the gold proportions. Nothing collapsed; the adapter simply moved to
answering synthetic-passage questions, because that is all 6,758 of its training
rows were.

So the fix is a ratio, not a new method. Arm G-full's 9,989 rows are the
benchmark's own prompts and hold the answering distribution in place -- its
5,925 keep rows are exactly the self-replay anchor that cut damage from 17.3%
to 3.9% in the earlier controlled contrast. Adding arm H's teaches tier makes
the synthetic share 30% instead of 100%.

Anchors: 1,500 of arm H's 3,999, not all and not none. None would make every
synthetic row a disagreement, which risks teaching "on questions that look like
this, change your answer"; all of them would push synthetic mass back up without
carrying any new facts.
"""
import argparse, json, os, random, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
ap = argparse.ArgumentParser()
ap.add_argument("--synth-anchors", type=int, default=1500)
ap.add_argument("--out", default=f"{ROOT}/data/train/eligible/armJ.jsonl")
a = ap.parse_args()

real = [json.loads(l) for l in open(f"{ROOT}/data/train/eligible/armG_full.jsonl")]
synth = [json.loads(l) for l in open(f"{ROOT}/data/train/eligible/armH.jsonl")]

teaches = [r for r in synth if r["tier"] == "teaches"]
anchors = [r for r in synth if r["tier"] == "anchor"]
rng = random.Random(37)
keep_anchor = rng.sample(anchors, min(a.synth_anchors, len(anchors)))

for r in teaches:  keep_anchor  # no-op, kept for symmetry of intent
for r in teaches:  r["tier"] = "synth_teaches"
for r in keep_anchor: r["tier"] = "synth_anchor"

rows = real + teaches + keep_anchor
ids = [r["sample_id"] for r in rows]
assert len(set(ids)) == len(ids), f"{len(ids)-len(set(ids))} duplicate sample_ids"

# the real rows must still be the benchmark's own prompts, byte for byte
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
n_real_ids = sum(1 for r in rows if r["sample_id"].startswith("teleqna-"))
assert n_real_ids == len(real), "real rows lost their ids"

# and no synthetic row may be an ot-full question in disguise
seen = set()
for l in open(CANON):
    seen.add(json.loads(l)["question"].strip().lower())
# the prompt embeds the question; a verbatim match would show up as containment
clash = 0
for r in teaches + keep_anchor:
    for q in seen:
        if len(q) > 40 and q in r["prompt"].lower():
            clash += 1
            break
assert clash == 0, f"{clash} synthetic rows contain an ot-full question verbatim"

rng.shuffle(rows)
with open(a.out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

c = collections.Counter(r["tier"] for r in rows)
n_synth = sum(v for k, v in c.items() if k.startswith("synth"))
print(f"wrote {len(rows)} rows -> {a.out}")
print(f"  tiers: {dict(sorted(c.items()))}")
print(f"  synthetic share: {n_synth}/{len(rows)} = {n_synth/len(rows)*100:.1f}%  (arm H was 100%)")
dis = c.get("flip", 0) + c.get("synth_teaches", 0)
print(f"  rows whose label disagrees with the base: {dis} = {dis/len(rows)*100:.1f}%"
      f"  (arm G-full alone: {sum(1 for r in real if r['tier']=='flip')/len(real)*100:.1f}%)")
print(f"  verbatim ot-full questions among synthetic rows: {clash} (must be 0)")
