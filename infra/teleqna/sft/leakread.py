"""Print the actual text behind each flagged co-occurrence so I can judge it.

A mechanical rule cannot tell a leaked benchmark item from a spec's own
definitions list, and the difference decides whether the whole evidence route is
legitimate. So: read them.
"""
import json, os, re, glob, random

C = os.path.expanduser("~/projects/telelogs/shared/corpora/tele-data")
ref = [json.loads(l) for l in open(os.path.expanduser(
        "~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl"))]
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
random.seed(7)
sample = random.sample(ref, 400)

WANT = [264, 276, 155, 270, 366, 318, 261, 127]
opt_index = {}
for qi in WANT:
    ch = sample[qi]["choices"]
    ch = eval(ch) if isinstance(ch, str) else ch
    for c in ch:
        n = norm(c)
        if len(n.split()) >= 4:
            opt_index.setdefault(n, set()).add(qi)

best = {}
for shard in ["standard", "wiki"]:
    for p in sorted(glob.glob(f"{C}/{shard}/*.jsonl")):
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                t = norm(line)
                per = {}
                for s, qs in opt_index.items():
                    j = t.find(s)
                    if j >= 0:
                        for qi in qs:
                            per.setdefault(qi, []).append((j, j + len(s)))
                for qi, sp in per.items():
                    if len(sp) < 3:
                        continue
                    lo, hi = min(a for a, _ in sp), max(b for _, b in sp)
                    if qi not in best or (hi - lo) < best[qi][0]:
                        best[qi] = (hi - lo, len(sp), shard, t[max(0, lo - 300): hi + 300])

for qi in WANT:
    q = sample[qi]
    ch = q["choices"]
    ch = eval(ch) if isinstance(ch, str) else ch
    print("=" * 100)
    print(f"q#{qi}  {q['question']}")
    for i, c in enumerate(ch):
        print(f"   {chr(65+i)}) {c}" + ("   <-- GOLD" if i == int(q["answer"]) else ""))
    if qi in best:
        span, n, shard, ex = best[qi]
        print(f"\n  tightest doc: shard={shard} opts={n} span={span} chars")
        print("  " + ex[:1100].replace("\n", " "))
    else:
        print("  (no doc found in standard/wiki)")
    print()
