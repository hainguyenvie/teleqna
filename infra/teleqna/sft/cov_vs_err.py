#!/usr/bin/env python3
"""Cross the corpus's union coverage against what the model actually gets wrong.

Coverage on its own is not a plan. A corpus can reach 90% of the benchmark and
still be worthless if the rows it reaches are the ones already answered
correctly - the gain would have to come from rows nobody can teach. So this
joins per-row coverage with per-row correctness from a stored eval and reports
the only two cells that decide whether to build the corpus:

  wrong & covered    the addressable error mass - what a corpus can buy
  wrong & residual   errors no source reaches - the shopping list, or the wall
"""
import argparse, collections, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--shards", default="arxiv,standard,wiki")
ap.add_argument("--result", required=True, help="eval json with per-row correct")
ap.add_argument("--thr", type=float, default=0.8)
a = ap.parse_args()

cov, subj = collections.defaultdict(dict), {}
for sh in a.shards.split(","):
    for line in (Path(a.dir) / f"coverage_{sh}_w400.rows.jsonl").open():
        r = json.loads(line)
        cov[r["sample_id"]][sh] = r["coverage"]
        subj[r["sample_id"]] = r["subject"]

res = json.load(open(a.result))["results"]
ok = {r["sample_id"]: bool(r["correct"]) for r in res}

cell = collections.Counter()
bysub = collections.defaultdict(collections.Counter)
for sid, correct in ok.items():
    if sid not in cov:
        continue
    covered = max(cov[sid].values()) >= a.thr
    k = ("right" if correct else "WRONG", "covered" if covered else "residual")
    cell[k] += 1
    bysub[subj[sid]][k] += 1

n = sum(cell.values())
wrong = cell[("WRONG", "covered")] + cell[("WRONG", "residual")]
print(f"n={n}   đúng={n-wrong} ({(n-wrong)/n*100:.2f}%)   SAI={wrong}\n")
print(f"{'':10s} {'covered':>10s} {'residual':>10s}")
for st in ("right", "WRONG"):
    print(f"{st:10s} {cell[(st,'covered')]:10d} {cell[(st,'residual')]:10d}")

wc, wr = cell[("WRONG", "covered")], cell[("WRONG", "residual")]
print(f"\ntrong {wrong} câu SAI: {wc} ({wc/wrong*100:.1f}%) có nguồn phủ, "
      f"{wr} ({wr/wrong*100:.1f}%) không nguồn nào chạm")
cc = cell[("right", "covered")] + wc
print(f"tỉ lệ đúng trên phần ĐÃ phủ  : {cell[('right','covered')]/cc*100:.2f}%")
rr = cell[("right", "residual")] + wr
print(f"tỉ lệ đúng trên phần CHƯA phủ: {cell[('right','residual')]/rr*100:.2f}%")

print(f"\n{'subject':28s} {'sai&phủ':>9s} {'sai&thiếu':>10s} {'acc phủ':>9s} {'acc thiếu':>10s}")
for s in sorted(bysub, key=lambda k: -(bysub[k][("WRONG","covered")])):
    c = bysub[s]
    cv = c[("right","covered")] + c[("WRONG","covered")]
    rs = c[("right","residual")] + c[("WRONG","residual")]
    print(f"{s:28s} {c[('WRONG','covered')]:9d} {c[('WRONG','residual')]:10d} "
          f"{c[('right','covered')]/cv*100 if cv else 0:8.1f}% "
          f"{c[('right','residual')]/rs*100 if rs else 0:9.1f}%")
