#!/usr/bin/env python3
"""Union coverage over shards: what fraction of the test set ANY source reaches.

Per-shard numbers understate the corpus we will actually build, because a row
covered by arxiv and a row covered by standard are both addressable - the
generator reads whichever shard carries it. What matters for planning is the
union, and just as much the residual: the rows no shard reaches at all, broken
down by subject and by provenance tag, because that is the shopping list.

`web` is excluded by default: its verify pass surfaced CCNA exam-dump pages
carrying the question and a starred answer, so including it would count
leakage as coverage.
"""
import argparse, collections, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--shards", default="arxiv,standard,wiki")
ap.add_argument("--test", required=True)
ap.add_argument("--thr", type=float, default=0.8)
ap.add_argument("--dump-residual")
a = ap.parse_args()

test = {json.loads(l)["sample_id"]: json.loads(l)
        for l in open(a.test, encoding="utf-8")}

cov = collections.defaultdict(dict)          # sample_id -> shard -> coverage
subj = {}
for sh in a.shards.split(","):
    p = Path(a.dir) / f"coverage_{sh}_w400.rows.jsonl"
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        cov[r["sample_id"]][sh] = r["coverage"]
        subj[r["sample_id"]] = r["subject"]

def tag(sid):
    m = re.findall(r"\[([^\]]+)\]", test[sid]["question"]) if sid in test else []
    if not m: return "(no tag)"
    t = m[-1]
    if "Release" in t or "3GPP" in t: return "3GPP"
    if "IEEE" in t: return "IEEE " + t.split()[-1]
    return t[:24]

shards = a.shards.split(",")
per = collections.defaultdict(lambda: collections.Counter())
best_src = collections.Counter()
residual = []
for sid, d in cov.items():
    s = subj[sid]
    per[s]["n"] += 1
    top = max(d.items(), key=lambda kv: kv[1])
    per[s]["union_ge"] += top[1] >= a.thr
    per[s]["union_full"] += top[1] >= 0.999
    for sh in shards:
        per[s][sh] += d.get(sh, 0) >= a.thr
    if top[1] >= a.thr:
        best_src[top[0]] += 1
    else:
        residual.append((sid, s, tag(sid), round(top[1], 3)))

W = max(len(s) for s in per) + 1
hdr = f"{'subject':{W}s} {'n':>5s} {'∪≥thr':>7s} {'%':>6s} {'∪full':>6s} " + \
      " ".join(f"{sh:>9s}" for sh in shards)
print(hdr); print("-" * len(hdr))
tot = collections.Counter()
for s in sorted(per, key=lambda k: -per[k]["n"]):
    c = per[s]; tot.update(c)
    print(f"{s:{W}s} {c['n']:5d} {c['union_ge']:7d} {c['union_ge']/c['n']*100:5.1f}% "
          f"{c['union_full']:6d} " + " ".join(f"{c[sh]:9d}" for sh in shards))
print("-" * len(hdr))
print(f"{'TOTAL':{W}s} {tot['n']:5d} {tot['union_ge']:7d} {tot['union_ge']/tot['n']*100:5.1f}% "
      f"{tot['union_full']:6d} " + " ".join(f"{tot[sh]:9d}" for sh in shards))

print(f"\nnguồn phủ tốt nhất cho mỗi hàng đã phủ: {dict(best_src)}")
print(f"\n=== RESIDUAL: {len(residual)} hàng không nguồn nào đạt ≥{a.thr} ===")
by = collections.Counter((s, t) for _, s, t, _ in residual)
for (s, t), n in by.most_common(18):
    print(f"  {n:5d}  {s:26s} {t}")
if a.dump_residual:
    with open(a.dump_residual, "w", encoding="utf-8") as fh:
        for sid, s, t, c in residual:
            fh.write(json.dumps({"sample_id": sid, "subject": s, "tag": t,
                                 "best_coverage": c}, ensure_ascii=False) + "\n")
    print(f"\n-> {a.dump_residual}")
