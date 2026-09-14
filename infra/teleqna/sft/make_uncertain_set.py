"""The rows whose label is still weak, with their retrieved evidence attached.

Arm G's label accuracy breaks down as keep 95.80% / retain 84.17% / flip 63.88%.
The keep tier is the model's own confident vote and needs no second opinion; the
other 3,451 rows carry essentially all of the remaining label error, and they are
the only rows worth spending a 122B pass on.

Evidence text is copied verbatim from whichever strong-retrieval set covers the
row, so the 122B reads exactly what the 31B read and any disagreement is the
model, not the context.
"""
import json, os, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
OUT = f"{ROOT}/data/uncertain_rag8_strong.jsonl"

want = {}
for l in open(f"{ROOT}/data/train/eligible/armG.jsonl"):
    r = json.loads(l)
    if r["tier"] in ("flip", "retain"):
        want[r["sample_id"]] = r["tier"]
print(f"uncertain rows in arm G: {len(want)}  {collections.Counter(want.values())}")

n = 0
seen = set()
with open(OUT, "w") as fh:
    for f in ["escalate2730_rag8_strong.jsonl", "blind6202_rag8_strong.jsonl"]:
        for l in open(f"{ROOT}/data/{f}"):
            r = json.loads(l)
            if r["sample_id"] in want and r["sample_id"] not in seen:
                seen.add(r["sample_id"])
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
print(f"wrote {n} rows -> {OUT}")
missing = set(want) - seen
assert not missing, f"{len(missing)} uncertain rows have no evidence, e.g. {list(missing)[:3]}"
