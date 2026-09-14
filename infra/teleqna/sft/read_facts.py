import json, os, random, collections
ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ref = {}
for l in open(CANON):
    r = json.loads(l)
    ref[r["sample_id"]] = r
rows = [json.loads(l) for l in open(f"{ROOT}/data/armE_facts.jsonl")]
def gold(sid): return chr(65 + int(ref[sid]["answer"]))
ok = [r for r in rows if r["good"] == gold(r["sample_id"])]
bad = [r for r in rows if r["good"] != gold(r["sample_id"])]
print(f"{len(rows)} facts   gate right {len(ok)} ({len(ok)/len(rows)*100:.1f}%)  gate wrong {len(bad)}")
lens = sorted(len(r["fact"].split()) for r in rows)
print(f"fact length words: p10={lens[len(lens)//10]} p50={lens[len(lens)//2]} p90={lens[int(len(lens)*.9)]} max={lens[-1]}")
print(f"flagged: {collections.Counter(r['flag'] for r in rows)}")
random.seed(5)
for tag, pool in [("GATE RIGHT", ok), ("GATE WRONG", bad)]:
    print("\n" + "#"*95 + f"\n{tag}\n" + "#"*95)
    for r in random.sample(pool, 5):
        print(f"\n[{r['tier']}] {r['sample_id']}  {ref[r['sample_id']]['question'][:120]}")
        print(f"   gate says {r['good']}) {r['good_text'][:90]}")
        print(f"   model had {r['bad']}) {r['bad_text'][:90]}")
        print(f"   gold is   {gold(r['sample_id'])}")
        print(f"   FACT: {r['fact'][:300]}")
