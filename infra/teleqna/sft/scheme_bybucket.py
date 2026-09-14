#!/usr/bin/env python3
"""Label accuracy per bucket, and the damage a scheme would do where the model
is already better than its own labels.

The trap this exists to catch: bucket A rows are answered correctly 99.62% of the
time by the base model. If a scheme labels them at 93%, then full obedience --
which arm F demonstrably has -- would DRAG THEM DOWN 6 points, costing far more
than the flip tier gains. A headline label accuracy of 87% can hide that
completely.

So for every scheme, per bucket: label accuracy, the base model's own accuracy,
and the resulting delta if the model simply obeys.
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

def letters(p):
    d = json.load(open(p))
    return {r["sample_id"]: (r["parsed"] or None) for r in d["results"]}

p122 = letters(f"{R}/vllm122b/otfull_base_think_merged.json")
base = {r["sample_id"]: bool(r["correct"]) for r in
        json.load(open(f"{R}/otel31b/otfull_nothink_base_nothink512.json"))["results"]}

ev = {}
for f in ["esc2730_ragstrong8_reparsed.json", "blind6202_ragstrong8_reparsed.json"]:
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

buckets = json.load(open(f"{R}/error_buckets.json"))
bof = {}
for b in ["A", "B", "C"]:
    for i in buckets[b]:
        bof[i] = b

ids = [s for s in ref if s in pvote and s in ev and s in base]

SCHEMES = {
    "evidence only": lambda s: ev.get(s),
    "evidence, vote when margin==1.0":
        lambda s: pvote.get(s) if margin.get(s, 0) >= 0.999 else ev.get(s),
    "evidence, vote when margin>=0.75":
        lambda s: pvote.get(s) if margin.get(s, 0) >= 0.75 else ev.get(s),
    "evidence, vote when base already agrees with vote AND margin>=0.9":
        lambda s: pvote.get(s) if margin.get(s, 0) >= 0.9 else ev.get(s),
}

for name, fn in SCHEMES.items():
    lab = {s: fn(s) for s in ids}
    have = [s for s in ids if lab[s]]
    print(f"\n=== {name} ===")
    print("  %-10s %6s %9s %9s %9s" % ("bucket", "n", "label acc", "base acc", "if obeyed"))
    tot_delta = 0
    for b in ["A", "B", "C"]:
        sel = [s for s in have if bof.get(s) == b]
        if not sel:
            continue
        la = sum(1 for s in sel if lab[s] == gold[s]) / len(sel) * 100
        ba = sum(base[s] for s in sel) / len(sel) * 100
        d = (la - ba) * len(sel) / 10000
        tot_delta += d
        print("  %-10s %6d %8.2f%% %8.2f%% %+8.2f pts" % (b, len(sel), la, ba, d))
    la = sum(1 for s in have if lab[s] == gold[s]) / len(have) * 100
    ba = sum(base[s] for s in have) / len(have) * 100
    print("  %-10s %6d %8.2f%% %8.2f%% %+8.2f pts total" % ("ALL", len(have), la, ba, tot_delta))
