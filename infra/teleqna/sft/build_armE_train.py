#!/usr/bin/env python3
"""Arm E stage 2: assemble the training file.

Three tiers, each doing a different job:

  flip      the gated rows, where an independent judge reading evidence
            overrules the model's own 32-sample vote. Target is the grounded
            fact plus an explicit contrast with the distractor the model used
            to choose -- this is the hard negative, and it is what arm V was
            missing when it learned only a letter.
  retain    rows where the evidence CONFIRMS the vote. Without these the flip
            tier teaches "when unsure, switch", which is not the lesson.
  anchor    the model's own confident, unchanged answers, sampled from outside
            the escalation set. [[teleqna-self-replay-anchor]] measured this as
            worth +9.7 against catastrophic drift; it costs nothing to include.

The prompt of every row is the plain closed-book harness prompt. No retrieved
text, no explanation, no answer key reaches the student.
"""
import argparse, json, os, random, sys, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
sys.path.insert(0, f"{ROOT}/infra")
from eval_dev_vllm import build_prompt          # the exact harness wording

R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ap = argparse.ArgumentParser()
ap.add_argument("--tier", choices=["wide", "high"], default="wide")
ap.add_argument("--anchor", type=int, default=2000)
ap.add_argument("--no-roundtrip", action="store_true",
                help="keep facts that fail the round-trip check (for the ablation)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

held = set(json.load(open(f"{ROOT}/data/clean_holdout500.json")))
for l in open(f"{ROOT}/data/dev1000.jsonl"):
    held.add(json.loads(l)["sample_id"])
assert len(held) > 1400

facts = {}
for l in open(f"{ROOT}/data/armE_facts_rt.jsonl"):
    r = json.loads(l)
    facts[r["sample_id"]] = r

# vote + margin, for the retain and anchor tiers
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
    if not c:
        continue
    tot = sum(c.values())
    top, n1 = c.most_common(1)[0]
    pvote[sid] = chr(65 + top)
    margin[sid] = n1 / tot

p31ev = {r["sample_id"]: r["letter"] for r in
         json.load(open(f"{R}/landscape/esc2730_ragstrong8_reparsed.json"))}

rows, seen = [], set()

def add(sid, completion, tier):
    if sid in seen or sid in held:
        return
    seen.add(sid)
    rows.append({"sample_id": sid, "prompt": build_prompt(ref[sid]),
                 "completion": completion, "tier": tier})

# ---- flip -----------------------------------------------------------------
n_flip_drop = n_rt_drop = 0
for sid, f in facts.items():
    if a.tier == "high" and f["tier"] != "high":
        continue
    if f["flag"]:                       # empty or meta-referring: unusable target
        n_flip_drop += 1
        continue
    if not a.no_roundtrip and not f["rt_agree"]:
        # the sentence does not carry the answer even when handed straight back,
        # so training on it teaches an assertion the model cannot derive
        n_rt_drop += 1
        continue
    fact = f["fact"].rstrip(". ") + "."
    comp = (f"{fact} This is {f['good']}) {f['good_text']}, "
            f"not {f['bad']}) {f['bad_text']}.\nANSWER: {f['good']}")
    add(sid, comp, "flip")

# ---- retain ---------------------------------------------------------------
for sid, letter in p31ev.items():
    if letter and pvote.get(sid) == letter:
        add(sid, f"ANSWER: {letter}", "retain")

# ---- anchor ---------------------------------------------------------------
cand = [sid for sid, m in margin.items()
        if m >= 0.95 and sid not in seen and sid not in held and sid not in p31ev]
random.seed(11)
for sid in random.sample(cand, min(a.anchor, len(cand))):
    add(sid, f"ANSWER: {pvote[sid]}", "anchor")

random.seed(3)
random.shuffle(rows)
with open(a.out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

c = collections.Counter(r["tier"] for r in rows)
print(f"wrote {len(rows)} rows -> {a.out}")
print(f"  tiers: {dict(c)}   (dropped {n_flip_drop} unusable facts, "
      f"{n_rt_drop} that failed the round-trip)")
print(f"  holdout rows present: {sum(1 for r in rows if r['sample_id'] in held)} (must be 0)")

# purity audit -- after the fact, never used to select anything
def gold(sid):
    return chr(65 + int(ref[sid]["answer"]))
for t in ["flip", "retain", "anchor"]:
    sel = [r for r in rows if r["tier"] == t]
    if not sel:
        continue
    ok = sum(1 for r in sel if r["completion"].rstrip()[-1] == gold(r["sample_id"]))
    print(f"  AUDIT {t:7s} n={len(sel):5d}  label accuracy {ok/len(sel)*100:5.2f}%")
ok = sum(1 for r in rows if r["completion"].rstrip()[-1] == gold(r["sample_id"]))
print(f"  AUDIT overall n={len(rows)}  label accuracy {ok/len(rows)*100:.2f}%")
