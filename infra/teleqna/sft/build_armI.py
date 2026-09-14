#!/usr/bin/env python3
"""Arm I: arm E's exact content with the letter moved to the front.

The three-way comparison this completes, on identical rows and identical labels:

    arm E   "<fact>. This is B) ..., not C) ...\\nANSWER: B"   83.03
    arm F   "ANSWER: B"                                        85.25
    arm I   "ANSWER: B\\n<fact>. This is B) ..., not C) ..."    ?

Arm E's defect was diagnosed as ORDERING, not the presence of prose: the letter
was trained conditioned on 58 tokens the model never generates at eval, so
P(letter | prompt) — the distribution the benchmark actually queries — was never
updated. Putting the letter first trains that distribution directly while keeping
the fact in the loss.

If arm I lands on arm F, rationales contribute nothing to this objective and the
branch closes with a clean answer instead of an inference. If arm I beats it,
every one of the fifteen catalogued failures deserves a rerun, because they all
used the losing order.

Parser safety checked before running: the harness STRICT pattern is
`(?i)^ANSWER\\s*:\\s*([A-Za-z\\d ,]+)\\s*(?:$|\\n|\\.)` under re.MULTILINE, so a
first line of "ANSWER: B" matches and trailing prose on later lines does not
interfere. Arm I is scorable by the official parser.
"""
import json, os, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
SRC = f"{ROOT}/data/train/eligible/armE_wide.jsonl"
OUT = f"{ROOT}/data/train/eligible/armI_wide.jsonl"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ref = {json.loads(l)["sample_id"]: json.loads(l) for l in open(CANON)}
gold = {s: chr(65 + int(r["answer"])) for s, r in ref.items()}

rows, n_reordered = [], 0
for l in open(SRC):
    r = json.loads(l)
    c = r["completion"]
    if "\nANSWER: " in c:
        prose, _, letter = c.rpartition("\nANSWER: ")
        letter = letter.strip()
        assert len(letter) == 1 and letter.isalpha(), f"bad letter {letter!r}"
        r["completion"] = f"ANSWER: {letter}\n{prose.strip()}"
        n_reordered += 1
    rows.append(r)

with open(OUT, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

c = collections.Counter(r["tier"] for r in rows)
print(f"wrote {len(rows)} rows -> {OUT}")
print(f"  tiers: {dict(c)}   ({n_reordered} completions reordered)")
for t in sorted(c):
    sel = [r for r in rows if r["tier"] == t]
    ok = sum(1 for r in sel if r["completion"].split("\n")[0][-1] == gold[r["sample_id"]])
    print(f"  AUDIT {t:7s} n={len(sel):5d}  label accuracy {ok/len(sel)*100:5.2f}%")
print("  (labels identical to arm E and arm F by construction)")
ex = next(r for r in rows if r["tier"] == "flip")
print(f"\nexample flip completion:\n{ex['completion'][:300]!r}")
