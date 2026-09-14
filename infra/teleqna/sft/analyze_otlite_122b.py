#!/usr/bin/env python3
"""Error analysis of the 122B's otlite run: where the 185 misses actually go.

The headline (81.50) says how many, not which kind, and the kinds want different
fixes. Three are separated here because each has a different owner:

  unparsed      the reply never produced a parsable ANSWER line, so the official
                scorer takes it as wrong regardless of content. This is a
                harness/format problem, and it is free to fix if the content was
                right. Split further into "ran out of budget mid-thought" and
                "finished but answered in prose".
  wrong letter  the model committed to a distractor. This is the real error rate
                and the only part that needs knowledge or discrimination.
  by subject    tells us whether the misses concentrate where our 8B also fails
                (Standards specifications) or somewhere else entirely, which
                decides whether the 122B is a teacher worth distilling from.

Position bias is checked too: if wrong picks cluster on one letter, some of the
loss is a decoding artefact rather than a knowledge gap. The gold-text signal
for unparsed rows is deliberately weak evidence — a reply that walks through all
five options contains every option's text, so a match sizes the opportunity but
does not claim the point.
"""
import json
import sys
from collections import Counter, defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/tensara/projects/telelogs/runs/teleqna-sft/results/vllm122b/otlite_base_think.json"
DATA = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/tensara/projects/telelogs/runs/teleqna-sft/data/otlite1000.jsonl"

rows = {}
for line in open(DATA):
    if line.strip():
        r = json.loads(line)
        rows[r["sample_id"]] = r

d = json.load(open(RES))
res = d["results"]
summ = d["summary"]
print(f"=== {RES.split('/')[-1]}")
print(f"total={summ['total']}  acc={summ['accuracy']}  "
      f"lenient={summ['accuracy_lenient']}  unparsed={summ['unparsed']}")

wrong = [r for r in res if not r["correct"]]
unparsed = [r for r in wrong if not r["parsed"]]
wrong_letter = [r for r in wrong if r["parsed"]]
print(f"\nmisses={len(wrong)}  = unparsed {len(unparsed)} + wrong-letter {len(wrong_letter)}")

# ---- unparsed: budget vs drift -------------------------------------------
trunc = [r for r in unparsed if "</think>" not in (r["completion"] or "")]
drift = [r for r in unparsed if "</think>" in (r["completion"] or "")]
gold_in_tail = 0
for r in drift:
    row = rows.get(r["sample_id"])
    if not row:
        continue
    gold = row["choices"][int(row["answer"])].strip().lower()
    if gold and gold in (r["completion"] or "")[-1500:].lower():
        gold_in_tail += 1
print(f"\n-- unparsed breakdown")
print(f"   hit token budget mid-thought (no </think>) : {len(trunc)}")
print(f"   finished thinking, answered in prose       : {len(drift)}")
print(f"     of those, gold option text in the tail   : {gold_in_tail}"
      f"   (weak signal, see docstring)")

# ---- by subject -----------------------------------------------------------
per = defaultdict(lambda: {"n": 0, "wrong": 0, "unparsed": 0})
for r in res:
    row = rows.get(r["sample_id"])
    s = row["subject"] if row else "?"
    per[s]["n"] += 1
    if not r["correct"]:
        per[s]["wrong"] += 1
        if not r["parsed"]:
            per[s]["unparsed"] += 1
print(f"\n-- by subject")
print(f"   {'subject':<26s} {'n':>5s} {'acc':>7s} {'wrong':>6s} {'unpars':>7s}")
for s, v in sorted(per.items(), key=lambda kv: -kv[1]["wrong"]):
    acc = (v["n"] - v["wrong"]) / v["n"]
    print(f"   {s:<26s} {v['n']:5d} {acc:7.4f} {v['wrong']:6d} {v['unparsed']:7d}")

# ---- position bias among committed answers --------------------------------
gold_c, pick_c, conf = Counter(), Counter(), Counter()
for r in wrong_letter:
    row = rows.get(r["sample_id"])
    if not row:
        continue
    g = chr(65 + int(row["answer"]))
    gold_c[g] += 1
    pick_c[r["parsed"]] += 1
    conf[(g, r["parsed"])] += 1
print(f"\n-- among the {len(wrong_letter)} committed-but-wrong")
print(f"   gold letter was : {dict(sorted(gold_c.items()))}")
print(f"   model picked    : {dict(sorted(pick_c.items()))}")
print(f"   top confusions  : {conf.most_common(6)}")

# ---- does option count matter? --------------------------------------------
byk = defaultdict(lambda: [0, 0])
for r in res:
    row = rows.get(r["sample_id"])
    if not row:
        continue
    k = len(row["choices"])
    byk[k][0] += 1
    byk[k][1] += int(bool(r["correct"]))
print(f"\n-- accuracy by number of options")
for k, (n, ok) in sorted(byk.items()):
    print(f"   {k} options: n={n:4d}  acc={ok/n:.4f}")

# ---- verbosity of hits vs misses ------------------------------------------
def avg(xs):
    return sum(xs) / len(xs) if xs else 0
lc = [len(r["completion"] or "") for r in res if r["correct"]]
lw = [len(r["completion"] or "") for r in wrong_letter]
lu = [len(r["completion"] or "") for r in unparsed]
print(f"\n-- mean completion length (chars)")
print(f"   correct        : {avg(lc):8.0f}  (n={len(lc)})")
print(f"   wrong letter   : {avg(lw):8.0f}  (n={len(lw)})")
print(f"   unparsed       : {avg(lu):8.0f}  (n={len(lu)})")

# ---- concrete misses ------------------------------------------------------
print(f"\n=== 6 committed-but-wrong, verbatim ===")
for r in wrong_letter[:6]:
    row = rows.get(r["sample_id"])
    if not row:
        continue
    g = int(row["answer"])
    print(f"\n--- {r['sample_id']}  [{row['subject']}]  gold={chr(65+g)} picked={r['parsed']}")
    print(f"    Q    : {row['question'][:190]}")
    print(f"    gold : {row['choices'][g][:120]}")
    pi = ord(r["parsed"]) - 65
    if 0 <= pi < len(row["choices"]):
        print(f"    pick : {row['choices'][pi][:120]}")
    print(f"    tail : {repr((r['completion'] or '')[-200:])}")
