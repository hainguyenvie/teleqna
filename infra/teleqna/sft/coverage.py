"""How much of the benchmark is the evidence gate currently blind to?

The gate only ever saw the 2,730 rows the vote margin flagged. If a large part of
bucket C sits outside that set, arm E is training on a fraction of the questions
it could be training on, and extending retrieval is worth a card.
"""
import json, os, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ref = {json.loads(l)["sample_id"]: json.loads(l) for l in open(CANON)}
buckets = json.load(open(f"{R}/error_buckets.json"))
esc = {json.loads(l)["sample_id"] for l in open(f"{ROOT}/data/escalate2730.jsonl")}
held = set(json.load(open(f"{ROOT}/data/clean_holdout500.json")))
for l in open(f"{ROOT}/data/dev1000.jsonl"):
    held.add(json.loads(l)["sample_id"])

SHIFT = {"base": 0, "cot": 0, "p1": 1, "p2": 2}
votes = collections.defaultdict(collections.Counter)
for arm, s in SHIFT.items():
    for l in open(f"{R}/passk_otfull_{arm}/all.jsonl"):
        r = json.loads(l)
        nc = r["n_choices"]
        for got in r["letters"]:
            if got:
                votes[r["sample_id"]][(ord(got) - 65 - s) % nc] += 1
margin, pvote = {}, {}
for sid, c in votes.items():
    if c:
        tot = sum(c.values())
        top, n1 = c.most_common(1)[0]
        margin[sid] = n1 / tot
        pvote[sid] = chr(65 + top)

def gold(sid):
    return chr(65 + int(ref[sid]["answer"]))

print(f"escalated set: {len(esc)}")
for b in ["A", "B", "C", "C_hard"]:
    ids = set(buckets[b])
    inside = ids & esc
    print(f"  bucket {b:6s} n={len(ids):5d}  inside gate {len(inside):5d} "
          f"({len(inside)/len(ids)*100:5.1f}%)  outside {len(ids)-len(inside):5d}")

out = [s for s in ref if s not in esc and s not in held]
wrong = [s for s in out if pvote.get(s) and pvote[s] != gold(s)]
print(f"\noutside the gate, outside holdout: {len(out)} rows, "
      f"vote wrong on {len(wrong)} ({len(wrong)/len(out)*100:.1f}%)")
print("  of those wrong rows, margin distribution:")
h = collections.Counter()
for s in wrong:
    h[round(margin[s] * 4) / 4] += 1
for k in sorted(h):
    print(f"    margin ~{k:.2f}: {h[k]}")
outC = [s for s in buckets["C"] if s not in esc and s not in held]
print(f"\nbucket C rows the gate cannot see and could still train on: {len(outC)}")
json.dump(sorted(set(out)), open(f"{R}/landscape/gate_blind_ids.json", "w"))
print(f"wrote {R}/landscape/gate_blind_ids.json")
