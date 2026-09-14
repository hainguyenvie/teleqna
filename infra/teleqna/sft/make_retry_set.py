#!/usr/bin/env python3
"""Extract the rows a scored run left unparsed, for a re-score with more budget.

On otlite the 122B left 31 rows unparsed and every one of them was a truncation:
no </think> in the completion, i.e. the reply hit the token ceiling mid-thought
and never reached an ANSWER line. None had finished thinking and answered in
prose. So this is a budget problem, not a format problem, and re-running exactly
those rows with a larger ceiling is the whole fix.

Selection is on `parsed` being empty rather than on `correct` being false. A row
the model got wrong by committing to a distractor is a real miss and re-running
it with more tokens would only be fishing; a row with no parsable answer was
never actually scored on its content.

Writes the same schema as the source set, so the scorer reads it unchanged, and
keeps sample_id so the re-scored rows can be merged back by key.
"""
import json
import sys

RES = sys.argv[1]
SRC = sys.argv[2]
OUT = sys.argv[3]

rows = {}
for line in open(SRC):
    if line.strip():
        r = json.loads(line)
        rows[r["sample_id"]] = r

d = json.load(open(RES))
bad = [r for r in d["results"] if not r["parsed"]]

trunc = sum(1 for r in bad if "</think>" not in (r["completion"] or ""))
print(f"unparsed={len(bad)}  of which truncated={trunc}  drifted={len(bad) - trunc}")

n = 0
with open(OUT, "w") as fh:
    for r in bad:
        row = rows.get(r["sample_id"])
        if row is None:
            print(f"WARN: {r['sample_id']} not in source set, skipped")
            continue
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {OUT}  n={n}")
if n == 0:
    sys.exit(3)
