"""Merge the three blind-judge shards into one result file, then reparse it.

Shards were split by index modulo 3, so the union must be exactly the 6,202 rows
and every sample_id must appear once. Both are asserted rather than assumed --
a silently short merge would quietly shrink the next training set.
"""
import json, os, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
L = f"{ROOT}/results/landscape"
OUT = f"{L}/blind6202_ragstrong8_base_nothink512.json"

rows, summ = [], None
for s in range(3):
    d = json.load(open(f"{L}/blind_sh{s}_ragstrong8_base_nothink512.json"))
    rows.extend(d["results"])
    print(f"shard {s}: {len(d['results'])} rows, strict acc {d['summary']['accuracy']:.4f}, "
          f"unparsed {d['summary']['unparsed']}")
    summ = d["summary"]

ids = [r["sample_id"] for r in rows]
dup = [k for k, v in collections.Counter(ids).items() if v > 1]
assert not dup, f"{len(dup)} duplicate ids across shards, e.g. {dup[:3]}"
want = {json.loads(l)["sample_id"] for l in open(f"{ROOT}/data/blind6202.jsonl")}
missing = want - set(ids)
assert not missing, f"{len(missing)} rows missing from the merge"
assert len(rows) == 6202, f"expected 6202 rows, merged {len(rows)}"

merged = {"summary": {"stack": "vllm", "arm": "base", "thinking": False,
                      "max_new": 512, "total": len(rows),
                      "correct": sum(r["correct"] for r in rows),
                      "accuracy": sum(r["correct"] for r in rows) / len(rows),
                      "unparsed": sum(1 for r in rows if not r["parsed"]),
                      "merged_from": 3},
          "results": rows}
json.dump(merged, open(OUT, "w"))
print(f"\nmerged {len(rows)} rows -> {OUT}")
print(f"  strict {merged['summary']['accuracy']:.4f}, unparsed {merged['summary']['unparsed']}")
