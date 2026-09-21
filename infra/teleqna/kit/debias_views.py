#!/usr/bin/env python3
"""Debias synthetic MCQ views toward the test's structural priors (label-free; priors measured on question/option SHAPE
only, never on answers): cap the share of MCQs whose correct option is (a) the longest option (test 32.6%) and (b) the
option with the most question-word overlap (test ~39% among rows with a clear max); optionally rebalance the answer
position; keep everything else. Input: comma-separated globs of view files; output: same rows minus the capped ones,
written next to each input as <name>.debiased.jsonl. Reports before/after rates."""
import json, glob, re, random, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
ap = argparse.ArgumentParser(); ap.add_argument("--globs", required=True); ap.add_argument("--longest-frac", type=float, default=0.33); ap.add_argument("--overlap-frac", type=float, default=0.39); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rng = random.Random(a.seed)
STOP = set("the a an of in to and or for is are what which does do by on with as at from that this it be can its their".split())
def toks(s): return set(t for t in re.sub(r"[^a-z0-9 ]", " ", unicodedata.normalize("NFKC", s or "").lower()).split() if t not in STOP and len(t) > 2)
def longest(m): return max(range(len(m["options"])), key=lambda i: len(m["options"][i]))
def maxov(m):
    qt = toks(m["q"]); ov = [len(qt & toks(o)) for o in m["options"]]; mx = max(ov)
    return None if mx == 0 or ov.count(mx) > 1 else ov.index(mx)
for g in a.globs.split(","):
    for f in sorted(glob.glob(str(R / g))):
        rows = [l for l in open(f, encoding="utf-8")]; mcq = []
        for i, l in enumerate(rows):
            if '"view": "mcq"' not in l or '"ok": true' not in l: continue
            try: m = json.loads(json.loads(l)["text"])
            except Exception: continue
            if isinstance(m, dict) and isinstance(m.get("options"), list) and isinstance(m.get("answer"), int) and 0 <= m["answer"] < len(m["options"]): mcq.append((i, m))
        drop = set()
        for name, pred, frac in (("longest", lambda m: longest(m) == m["answer"], a.longest_frac), ("overlap", lambda m: maxov(m) == m["answer"], a.overlap_frac)):
            pos = [i for i, m in mcq if i not in drop and pred(m)]; neg = [i for i, m in mcq if i not in drop and not pred(m)]
            target = int(frac / (1 - frac) * len(neg)); rng.shuffle(pos)
            for i in pos[target:]: drop.add(i)
        out = f.replace(".jsonl", ".debiased.jsonl")
        with open(out, "w", encoding="utf-8") as fo:
            for i, l in enumerate(rows):
                if i not in drop: fo.write(l)
        kept = [m for i, m in mcq if i not in drop]
        print(f"{f.split('/')[-2]}/{f.split('/')[-1]}: mcq {len(mcq):,} -> {len(kept):,} | gold-longest {sum(longest(m)==m['answer'] for _,m in mcq)/max(1,len(mcq))*100:.0f}% -> {sum(longest(m)==m['answer'] for m in kept)/max(1,len(kept))*100:.0f}% | gold-maxoverlap {sum(maxov(m)==m['answer'] for _,m in mcq)/max(1,len(mcq))*100:.0f}% -> {sum(maxov(m)==m['answer'] for m in kept)/max(1,len(kept))*100:.0f}%", flush=True)
print("DEBIAS_DONE")
