#!/usr/bin/env python3
"""Fold a retry run back into its parent run and report the corrected number.

The retry only re-scored rows the parent left unparsed, so the merge is a
straight overwrite keyed on sample_id: every retried row replaces its parent
row, every other row is untouched. Rows the parent scored are never re-rolled,
which is what keeps the merged figure comparable with the parent's own.

Reports how many retried rows are still truncated. That number is the honest
caveat on the result: if the bigger budget did not finish them either, the
remaining zeros are not a knowledge verdict, and raising the ceiling again is
the next move rather than concluding the model does not know.
"""
import json
import sys

parent_p, retry_p = sys.argv[1], sys.argv[2]
parent = json.load(open(parent_p))
retry = json.load(open(retry_p))

by_id = {r["sample_id"]: r for r in parent["results"]}
n_before = sum(1 for r in parent["results"] if r["correct"])
total = len(parent["results"])

still_trunc = fixed = newly_right = 0
for r in retry["results"]:
    old = by_id.get(r["sample_id"])
    if old is None:
        print(f"WARN: {r['sample_id']} not in parent, ignored")
        continue
    if not r["parsed"]:
        if "</think>" not in (r["completion"] or ""):
            still_trunc += 1
    else:
        fixed += 1
        if r["correct"]:
            newly_right += 1
    by_id[r["sample_id"]] = r

merged = list(by_id.values())
n_after = sum(1 for r in merged if r["correct"])
unparsed_after = sum(1 for r in merged if not r["parsed"])

print(f"\n=== otlite, corrected for truncation")
print(f"  retried rows            : {len(retry['results'])}")
print(f"  now produce an answer   : {fixed}")
print(f"  still truncated         : {still_trunc}")
print(f"  of the newly answered, correct: {newly_right}")
print(f"\n  accuracy before : {n_before}/{total} = {n_before/total:.4f}")
print(f"  accuracy after  : {n_after}/{total} = {n_after/total:.4f}"
      f"   ({(n_after-n_before)/total*100:+.2f}pp)")
print(f"  unparsed        : {parent['summary']['unparsed']} -> {unparsed_after}")

out = parent_p.replace(".json", "_merged.json")
summ = dict(parent["summary"])
summ.update(arm="base+retry", total=total, correct=n_after,
            accuracy=round(n_after / total, 4), unparsed=unparsed_after)
json.dump({"summary": summ, "results": merged}, open(out, "w"),
          ensure_ascii=False, indent=1)
print(f"\n  wrote {out}")
