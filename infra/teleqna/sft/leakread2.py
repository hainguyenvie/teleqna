"""Read the two flagged questions I had not yet inspected, from the saved scan."""
import json, os, random, re

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
d = json.load(open(f"{ROOT}/results/contamination_scan.json"))
ref = [json.loads(l) for l in open(os.path.expanduser(
        "~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl"))]
random.seed(7)
sample = random.sample(ref, 400)

for qi in [212, 340]:
    q = sample[qi]
    ch = q["choices"]
    ch = eval(ch) if isinstance(ch, str) else ch
    print("=" * 100)
    print(f"q#{qi}  {q['question']}")
    for i, c in enumerate(ch):
        print(f"   {chr(65+i)}) {c}" + ("   <-- GOLD" if i == int(q["answer"]) else ""))
    hits = [h for h in d["hits"] if h["qi"] == qi and h["leaked"]]
    hits.sort(key=lambda h: h["span_chars"])
    if hits:
        h = hits[0]
        print(f"\n  tightest: shard={h['shard']} opts={h['n_opts']} span={h['span_chars']} "
              f"stem_cover={h['stem_cover']}")
        print("  " + h["excerpt"][:1200])
    print()

# and how many flagged events carry the question's own wording verbatim, which is
# what an actual copy of the benchmark would look like
print("=" * 100)
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
verbatim = 0
for h in d["hits"]:
    stem = norm(sample[h["qi"]]["question"])[:60]
    if stem and stem in h["excerpt"]:
        verbatim += 1
print(f"co-occurrence events: {len(d['hits'])}")
print(f"events whose excerpt contains the question stem verbatim: {verbatim}")
