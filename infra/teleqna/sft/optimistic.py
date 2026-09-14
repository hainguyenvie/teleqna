#!/usr/bin/env python3
"""The optimistic cases, assembled and checked rather than quoted.

Two upper bounds worth having on the board:

  1. training on the exact test questions WITH the published answer key -- the
     transductive ceiling, and not submittable;
  2. the same model reading strong retrieved evidence over all 10,000 rows --
     which has never been run as one job, but exists as three disjoint runs of
     the identical config, so it can be composed if the parts really do partition
     the benchmark. That is asserted here, not assumed.
"""
import json, os

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
allids = {json.loads(l)["sample_id"] for l in open(CANON)}

print("=== 1. strong retrieval, composed over the full 10,000 ===")
parts = [("escalated (hardest, margin<0.75)", "esc2730_ragstrong8_reparsed.json"),
         ("blind (margin>=0.75, incl. all C_hard)", "blind6202_ragstrong8_reparsed.json"),
         ("holdout", "holdout1068_ragstrong8_reparsed.json")]
seen, tot, cor = set(), 0, 0
for name, f in parts:
    rows = json.load(open(f"{R}/landscape/{f}"))
    ids = {r["sample_id"] for r in rows}
    assert not (ids & seen), f"{name} overlaps an earlier part"
    seen |= ids
    c = sum(1 for r in rows if r["correct"])
    tot += len(rows); cor += c
    print(f"  {name:42s} n={len(rows):5d}  {c/len(rows)*100:6.2f}%")
assert seen == allids, f"parts cover {len(seen)} of {len(allids)} rows"
print(f"  {'ALL — partition verified':42s} n={tot:5d}  {cor/tot*100:6.2f}%")
print("  (re-extracted; the strict harness scores this far lower because long")
print("   context breaks the ANSWER: format — see reparse.py)")

print("\n=== 2. trained on the exact test questions WITH the answer key ===")
for tag in ["otfull_M1", "otfull_akv2", "otfull_answerkey"]:
    for suf in ["_M1_nothink512.json", "_akv2_nothink512.json", "_base_nothink512.json"]:
        p = f"{R}/otfull_arms/{tag}{suf}"
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"  {tag}{suf}: {d['summary']['accuracy']*100:.2f}%  "
                  f"(n={d['summary']['total']}, unparsed={d['summary']['unparsed']})")
import glob
for p in sorted(glob.glob(f"{R}/**/*answerkey*.json", recursive=True) +
                glob.glob(f"{R}/**/*M1*.json", recursive=True) +
                glob.glob(f"{R}/**/*akv2*.json", recursive=True)):
    try:
        d = json.load(open(p))
        s = d.get("summary", {})
        if s.get("total", 0) >= 500:
            print(f"  {p.replace(R+'/', ''):58s} {s['accuracy']*100:6.2f}%  n={s['total']}")
    except Exception:
        pass

print("\n=== 3. reference ceilings already on the board ===")
p = f"{R}/landscape/otfull_expl_base_nothink512.json"
if os.path.exists(p):
    d = json.load(open(p))
    print(f"  gold explanation of the same question in context: "
          f"{d['summary']['accuracy']*100:.2f}%  (n={d['summary']['total']})")
