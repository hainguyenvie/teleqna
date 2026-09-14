#!/usr/bin/env python3
"""Recover the chosen letter from a run whose format collapsed under long context.

The official strict rule is `ANSWER: X` and that stays the submitted score. But
when a long evidence block pushes this model into prose ("The correct answer is
**C) ..."), the strict number measures formatting, not knowledge, and using it to
compare arms is simply wrong. This recovers the judgement for ANALYSIS.

Every run is validated the same way: on the rows the strict parser did read, the
extractor must reproduce its verdict exactly. Anything less and the output is
not used.
"""
import argparse, json, os, pathlib, re, collections

ap = argparse.ArgumentParser()
ap.add_argument("--result", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path)
a = ap.parse_args()

CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

PATS = [
    re.compile(r"\bANSWER:\s*\**\s*([A-E])\b"),
    re.compile(r"correct answer is\s*\**\s*\(?([A-E])\)?[\)\.\*: ]", re.I),
    re.compile(r"answer is\s*\**\s*\(?([A-E])\)?[\)\.\*: ]", re.I),
    re.compile(r"^\s*\**\s*\(?([A-E])\)[\s\*]", re.M),
    re.compile(r"option\s*\**\s*\(?([A-E])\)?\b[^.]{0,40}\bis (?:the )?correct", re.I),
]

def extract(c):
    for p in PATS:
        m = p.search(c)
        if m:
            return m.group(1).upper()
    return None

d = json.load(open(a.result))
rows = d["results"]
rec, agree, checked, unrec = [], 0, 0, 0
bysub = collections.defaultdict(lambda: [0, 0])
for r in rows:
    sid = r["sample_id"]
    m = ref[sid]
    gold = chr(65 + int(m["answer"]))
    letter = extract(r["completion"])
    if letter is None:
        unrec += 1
    ok = letter == gold
    if r["parsed"]:
        checked += 1
        agree += int(bool(r["correct"]) == ok)
    bysub[m["subject"]][0] += 1
    bysub[m["subject"]][1] += int(ok)
    rec.append({"sample_id": sid, "letter": letter, "gold": gold, "correct": ok,
                "subject": m["subject"], "harness_parsed": bool(r["parsed"])})

n = len(rows)
acc = sum(x["correct"] for x in rec)
print(f"{a.result.name}")
print(f"  VALIDATION: matches the strict harness on {agree}/{checked} rows it read"
      + ("" if agree == checked else "   !! DO NOT USE"))
print(f"  strict     : {d['summary']['correct']}/{n} = {d['summary']['accuracy']:.4f}"
      f"   (unparsed {d['summary']['unparsed']})")
print(f"  re-extracted: {acc}/{n} = {acc/n:.4f}   (unrecoverable {unrec})")
print("  %-28s %5s %7s" % ("subject", "n", "acc"))
for k, (t, c) in sorted(bysub.items(), key=lambda kv: -kv[1][0]):
    print("  %-28s %5d %7.4f" % (k, t, c / t))
if a.out:
    json.dump(rec, open(a.out, "w"))
    print(f"  wrote {a.out}")
