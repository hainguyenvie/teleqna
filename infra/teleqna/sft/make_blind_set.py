"""Emit the rows the evidence gate has never seen, in the escalate2730 schema."""
import json, os

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ids = set(json.load(open(f"{ROOT}/results/landscape/gate_blind_ids.json")))
n = 0
with open(f"{ROOT}/data/blind6202.jsonl", "w") as fh:
    for l in open(CANON):
        r = json.loads(l)
        if r["sample_id"] in ids:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
print(f"{n} rows -> {ROOT}/data/blind6202.jsonl (wanted {len(ids)})")
assert n == len(ids), "id list and canonical set disagree"
