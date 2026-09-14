#!/usr/bin/env python3
"""Independent re-score of stored completions: official parser vs repo cascade.

Trusts nothing in the summary block - recomputes from `completion` and the gold
jsonl, and separately counts the failure modes that a summary hides: rows with
no ANSWER line at all, and rows that ran to the token ceiling (a truncated
rollout is indistinguishable from a wrong answer in the accuracy column).
"""
import glob, json, os, re, sys

STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE  = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
BARE   = re.compile(r"^\s*([A-Ea-e])(?:[).:,\s]|$)")

def pick(ms, n):
    if not ms: return ""
    g = ms[-1].strip().rstrip(".").upper()
    return g if g in {chr(65+i) for i in range(n)} else ""

def official(c, n): return pick(STRICT.findall(c or "") or LOOSE.findall(c or ""), n)
def lenient(c, n):
    return pick(STRICT.findall(c or "") or LOOSE.findall(c or "") or BARE.findall(c or ""), n)

# Keyed by split name, and chosen per result file by which split actually
# contains its sample_ids. Matching on row count instead silently scores one
# 1,000-row split against another's answer key.
golds = {}
for p in sys.argv[1].split(","):
    rows = [json.loads(l) for l in open(p)]
    golds[os.path.basename(p)] = {r["sample_id"]: (chr(65+int(r["answer"])), len(r["choices"]))
                                  for r in rows}


def pick_gold(res):
    ids = {r["sample_id"] for r in res}
    best, hit = None, 0
    for name, g in golds.items():
        h = len(ids & set(g))
        if h > hit:
            best, hit = name, h
    return (best, golds[best]) if hit >= 0.99 * len(ids) else (None, None)

print(f"{'file':40s} {'split':16s} {'n':>5s} {'official':>9s} {'cascade':>8s} {'bare+':>6s} "
      f"{'unpars':>7s} {'trunc?':>7s} {'meanlen':>8s}")
for d in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try: j = json.load(open(f))
        except Exception: continue
        res = j.get("results") if isinstance(j, dict) else None
        if not res or "completion" not in res[0]: continue
        split, gold = pick_gold(res)
        if not gold: continue
        off = len_ = bare = unp = 0
        lens = []
        for r in res:
            g = gold.get(r["sample_id"])
            if not g: continue
            ans, n = g
            c = r.get("completion", "")
            lens.append(len(c))
            o, l = official(c, n), lenient(c, n)
            off += (o == ans); len_ += (l == ans); bare += (l == ans and o != ans)
            unp += (not o)
        # a rollout that never emitted ANSWER and is in the top length decile
        # is a token-ceiling casualty, not a wrong answer
        cut = sorted(lens)[int(len(lens) * 0.90)] if lens else 0
        trunc = sum(1 for r in res
                    if not official(r.get("completion", ""), 5)
                    and len(r.get("completion", "")) >= cut)
        nm = os.path.basename(f)[:-5]
        print(f"{nm[:40]:40s} {split[:16]:16s} {len(res):5d} {off/len(res)*100:8.2f}% "
              f"{len_/len(res)*100:7.2f}% {bare:6d} {unp:7d} {trunc:7d} "
              f"{sum(lens)/len(lens):8.0f}")
