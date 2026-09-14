#!/usr/bin/env python3
"""Arm H stage 3: the training set of synthetic questions.

Filter outcome on 15,451 cleaned items:
    teaches  2,759   resolvable from the passage AND the base gets it wrong closed-book
    anchor  11,278   resolvable AND the base already gets it right
    discard  1,414   not resolvable even with the passage — a broken item, not a hard one

Only the `teaches` rows carry knowledge the model does not have. Anchors are
included at a ratio close to arm G's, purely to hold the format and stop the
model learning "when in doubt, change your answer"; they cannot pin ot-full
answers the way arm G's anchors did, because these are different questions.

The whole point: NOT ONE of these questions is in ot-full. So the entire 10,000-row
benchmark is a clean test set for this arm, with no trained/untrained split to
misread — which is the measurement the holdout could never quite give.
"""
import argparse, json, os, random, sys, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
sys.path.insert(0, f"{ROOT}/infra")
from eval_dev_vllm import build_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--anchors", type=int, default=4000)
ap.add_argument("--out", default=f"{ROOT}/data/train/eligible/armH.jsonl")
a = ap.parse_args()

rows = [json.loads(l) for l in open(f"{ROOT}/data/synth_mcq_scored.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    by[r["tier"]].append(r)
print({k: len(v) for k, v in by.items()})

rng = random.Random(31)
keep = list(by["teaches"]) + rng.sample(by["anchor"], min(a.anchors, len(by["anchor"])))
rng.shuffle(keep)

# guard: none of these may be an ot-full question
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
real = set()
for l in open(CANON):
    r = json.loads(l)
    real.add(r["question"].strip().lower())
# The generator was shown windows retrieved FOR the ot-full questions, so it can
# occasionally reproduce one verbatim. Those rows would silently turn arm H into
# arm G, so they are dropped rather than tolerated.
before = len(keep)
keep = [r for r in keep if r["question"].strip().lower() not in real]
clash = before - len(keep)

out = []
for r in keep:
    out.append({"sample_id": f"synth-{r['src_id']}-{r['win']}",
                "prompt": build_prompt({"question": r["question"],
                                        "choices": r["choices"],
                                        "answer": r["answer"]}),
                "completion": f"ANSWER: {chr(65 + r['answer'])}",
                "tier": r["tier"]})
with open(a.out, "w") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

c = collections.Counter(r["tier"] for r in out)
print(f"wrote {len(out)} rows -> {a.out}")
print(f"  tiers: {dict(c)}")
print(f"  verbatim clashes with ot-full: {clash} (must be 0)")
print("  no label accuracy audit is possible here: these questions have no gold key.")
print("  their labels were validated behaviourally — the model answers them")
print("  correctly when it can see the passage they came from.")
