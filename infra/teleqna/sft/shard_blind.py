"""Split the blind evidence set across the three free cards.

One card was going to take ~1.5h for 6,202 rows of ~20k-token prompts. Two cards
are idle now that both arm E evals have landed, so the same work finishes in a
third of the time. Sharding is by index modulo, matching the pass@k runs, so a
shard's membership is reproducible from its number alone.
"""
import json, os

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
SRC = f"{ROOT}/data/blind6202_rag8_strong.jsonl"
N = 3
rows = [json.loads(l) for l in open(SRC)]
for s in range(N):
    part = [r for i, r in enumerate(rows) if i % N == s]
    p = f"{ROOT}/data/blind6202_rag8_strong_sh{s}.jsonl"
    with open(p, "w") as fh:
        for r in part:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"shard {s}: {len(part)} rows -> {p}")
print(f"total {len(rows)}")
