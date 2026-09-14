#!/usr/bin/env python3
"""Build otlite1000.jsonl from GSMA/ot-lite's teleqna split.

ot-lite is a separate HF repo from ot-full, not a config inside it, and it ships
raw JSON rather than parquet — so it needs its own conversion into the
{question, choices, answer, subject, sample_id, sample_index} shape the scorer
reads.

The join back to ot-full is keyed on question+choices, not question alone. The
corpus contains repeated stems: three ot-lite rows have the literal question
text "Research Papers", and "What is the definition of operating band?" appears
more than once with different option sets. A question-only key silently maps all
of those onto whichever row came first, which manufactures label mismatches that
are really join errors. With the wider key, all 1000 rows match exactly one
ot-full row and the two releases agree on every label.

sample_id is inherited from ot-full so results on this file stay joinable with
dev-1000 and ot-full-10000 results already in the tree; answer/choices are taken
from ot-lite itself, which is the authority on what "otlite" means.
"""
import hashlib
import json
import sys

D = "/home/tensara/projects/telelogs/runs/teleqna-sft/data/"
SRC = "/home/tensara/otlite/test_teleqna.json"
OUT = D + "otlite1000.jsonl"


def key(r):
    q = " ".join(r["question"].split()).lower()
    ch = "|".join(" ".join(c.split()).lower() for c in r["choices"])
    return hashlib.md5((q + "||" + ch).encode()).hexdigest()


full = {}
for line in open(D + "otfull10000.jsonl"):
    if line.strip():
        r = json.loads(line)
        full.setdefault(key(r), r)

lite = json.load(open(SRC))
rows, disagree = [], 0
for i, r in enumerate(lite):
    f = full.get(key(r))
    if f is None:
        print(f"ABORT: otlite row {i} has no ot-full match")
        sys.exit(3)
    if f["answer"] != r["answer"]:
        disagree += 1
    rows.append({
        "question": r["question"], "choices": r["choices"],
        "answer": r["answer"], "subject": r["subject"],
        "sample_index": i, "sample_id": f["sample_id"],
    })

if disagree:
    print(f"ABORT: {disagree} label disagreements between ot-lite and ot-full")
    sys.exit(4)

with open(OUT, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

ids = set(r["sample_id"] for r in rows)
print(f"wrote {OUT}  n={len(rows)}  unique_sample_id={len(ids)}  label_disagreements=0")
print("schema:", sorted(rows[0].keys()))
from collections import Counter
print("subjects:", dict(Counter(r["subject"] for r in rows)))
