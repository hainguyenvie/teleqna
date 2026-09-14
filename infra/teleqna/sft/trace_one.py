#!/usr/bin/env python3
"""Trace real rows end to end through the label pipeline, one tier at a time."""
import json, os, re, collections, sys

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

# stage 1a: the 32 rollouts, letters mapped back to the ORIGINAL option order
SHIFT = {"base": 0, "cot": 0, "p1": 1, "p2": 2}
raw = collections.defaultdict(dict)
votes = collections.defaultdict(collections.Counter)
for arm, s in SHIFT.items():
    for l in open(f"{R}/passk_otfull_{arm}/all.jsonl"):
        r = json.loads(l)
        nc = r["n_choices"]
        raw[r["sample_id"]][arm] = r["letters"]
        for got in r["letters"]:
            if got:
                votes[r["sample_id"]][(ord(got) - 65 - s) % nc] += 1

# stage 1b: what the 31B answered while READING the retrieved evidence
ev = {}
for f in ["esc2730_ragstrong8_reparsed.json", "blind6202_ragstrong8_reparsed.json"]:
    for r in json.load(open(f"{R}/landscape/{f}")):
        ev[r["sample_id"]] = r["letter"]

# the evidence text itself
evtext = {}
for f in ["escalate2730_rag8_strong.jsonl", "blind6202_rag8_strong.jsonl"]:
    for l in open(f"{ROOT}/data/{f}"):
        r = json.loads(l)
        evtext[r["sample_id"]] = r["question"].split("\n\n---\n\n")[0]

train = {}
for l in open(f"{ROOT}/data/train/eligible/armG.jsonl"):
    r = json.loads(l)
    train[r["sample_id"]] = r

want = sys.argv[1:] if len(sys.argv) > 1 else None
if not want:
    # one clean example per tier: flip where the gate rescued a wrong vote,
    # retain, and keep
    want = []
    for tier in ["flip", "retain", "keep"]:
        for sid, t in train.items():
            if t["tier"] != tier or sid not in evtext:
                continue
            gold = chr(65 + int(ref[sid]["answer"]))
            lab = t["completion"][-1]
            vote = chr(65 + votes[sid].most_common(1)[0][0])
            if tier == "flip" and not (lab == gold and vote != gold):
                continue
            if tier == "retain" and not (lab == gold):
                continue
            if tier == "keep" and not (lab == gold and len(ref[sid]["choices"]) >= 4):
                continue
            want.append(sid)
            break

for sid in want:
    r = ref[sid]
    gold = chr(65 + int(r["answer"]))
    c = votes[sid]
    tot = sum(c.values())
    top, n1 = c.most_common(1)[0]
    vote, margin = chr(65 + top), n1 / tot
    t = train.get(sid)
    print("=" * 100)
    print(f"{sid}   [{r['subject']}]   tier = {t['tier'] if t else 'not trained'}")
    print(f"Q: {r['question']}")
    for i, ch in enumerate(r["choices"]):
        print(f"   {chr(65+i)}) {ch}" + ("      <-- GOLD (never used to build the label)"
                                          if i == int(r["answer"]) else ""))
    print(f"\n  STEP 1  32 rollouts, letters mapped back to original option order")
    for arm in ["base", "cot", "p1", "p2"]:
        print(f"     {arm:5s} k=8 -> {''.join(x or '.' for x in raw[sid].get(arm, []))}")
    print(f"     tally after mapping: "
          + "  ".join(f"{chr(65+k)}={v}" for k, v in sorted(c.items())))
    print(f"     vote = {vote}   margin = {n1}/{tot} = {margin:.3f}")
    if sid in ev:
        e = evtext[sid]
        print(f"\n  STEP 2  BM25 retrieval, 8 windows, {len(e)} chars. First window:")
        w = re.split(r"\n\n(?=\[\d+\] )", e)
        first = (w[1] if len(w) > 1 else w[0])[:600]
        print("     " + first.replace("\n", " "))
        print(f"\n  STEP 3  the SAME 31B re-answers with those windows in the prompt")
        print(f"     evidence answer = {ev[sid]}")
    else:
        print("\n  STEP 2-3  skipped: margin was high enough that no evidence was needed")
    print(f"\n  STEP 4  label rule: margin>=0.90 -> keep the vote, else take the evidence")
    if t:
        print(f"     -> tier {t['tier']}, label {t['completion'][-1]}   "
              f"(gold is {gold} — checked only afterwards, for the audit)")
        print(f"\n  STEP 5  the training row that goes into armG.jsonl:")
        print(f"     prompt     = <plain closed-book harness prompt, {len(t['prompt'])} chars, "
              f"NO evidence>")
        print(f"     completion = {t['completion']!r}")
    print()
