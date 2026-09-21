#!/usr/bin/env python3
"""Chat rows from evidence-gated synthetic MCQs (kit mcq view incl. contrastive): harness prompt under --rot random choice
rotations, completion 'ANSWER: <letter>' (generator's evidence-backed answer; no test data). For pack_chat.py."""
import json, glob, random, argparse
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
ap = argparse.ArgumentParser(); ap.add_argument("--globs", required=True, help="comma-separated globs of view files"); ap.add_argument("--out", required=True)
ap.add_argument("--rot", type=int, default=2); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max", type=int, default=0); a = ap.parse_args()
rng = random.Random(a.seed); rows = []
for g in a.globs.split(","):
    for f in sorted(glob.glob(str(R / g))):
        for l in open(f, encoding="utf-8"):
            if '"view": "mcq"' not in l or '"ok": true' not in l: continue
            try: m = json.loads(json.loads(l)["text"])
            except Exception: continue
            if not (isinstance(m, dict) and isinstance(m.get("options"), list) and 4 <= len(m["options"]) <= 5 and isinstance(m.get("answer"), int) and 0 <= m["answer"] < len(m["options"]) and isinstance(m.get("q"), str)): continue
            rows.append(m)
rng.shuffle(rows); rows = rows[:a.max] if a.max else rows; n_out = 0
Path(R / a.out).parent.mkdir(parents=True, exist_ok=True)
with open(R / a.out, "w", encoding="utf-8") as f:
    for m in rows:
        n = len(m["options"]); shifts = rng.sample(range(n), min(a.rot, n))
        for s in shifts:
            ch = [m["options"][(i + s) % n] for i in range(n)]; letter = chr(65 + (m["answer"] - s) % n)
            u = TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n)), question=m["q"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(ch)))
            f.write(json.dumps(dict(prompt=u, completion=f"ANSWER: {letter}"), ensure_ascii=False) + "\n"); n_out += 1
print(f"mcq questions {len(rows):,} -> rows {n_out:,} -> {a.out}"); print("ROWS_DONE")
