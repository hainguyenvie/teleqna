#!/usr/bin/env python3
"""Per-subject breakdown of where the baseline actually fails."""
import collections
import json

R = "/home/tensara/projects/telelogs/runs/bench4/teleqna"
rows = {}
for line in open(R + "/data/test.jsonl"):
    r = json.loads(line)
    rows[r["sample_id"]] = r


def res(p):
    return {r["sample_id"]: r for r in (json.loads(l) for l in open(p))}


nt = res(R + "/results/b0_nothink/results.jsonl")
th = res(R + "/results/b0_think/results.jsonl")
perm = res(R + "/results/b0_nothink_perm/results.jsonl")

S = collections.defaultdict(lambda: dict(n=0, nt=0, th=0, hard=0, flip=0,
                                          nc=collections.Counter()))
for sid, r in rows.items():
    d = S[r.get("subject", "?")]
    d["n"] += 1
    a, b = nt[sid]["correct"], th[sid]["correct"]
    d["nt"] += a
    d["th"] += b
    if not a and not b:
        d["hard"] += 1
        d["nc"][len(r["choices"])] += 1
    if sid in perm and perm[sid]["correct"] != a:
        d["flip"] += 1

tot_hard = sum(d["hard"] for d in S.values())
tot_err = sum(d["n"] - d["nt"] for d in S.values())
hdr = ("subject", "n", "no-think", "think", "delta", "err_nt", "%allerr",
       "hard", "hard%row", "%allhard", "permflip")
print("{:26s} {:>5s} {:>8s} {:>7s} {:>6s} {:>7s} {:>8s} {:>6s} {:>9s} {:>9s} {:>9s}".format(*hdr))
for s, d in sorted(S.items(), key=lambda kv: -kv[1]["hard"]):
    err = d["n"] - d["nt"]
    print("{:26s} {:5d} {:8.2f} {:7.2f} {:+6.2f} {:7d} {:7.1f}% {:6d} {:8.1f}% {:8.1f}% {:9d}".format(
        s, d["n"], d["nt"] / d["n"] * 100, d["th"] / d["n"] * 100,
        (d["th"] - d["nt"]) / d["n"] * 100, err, err / tot_err * 100,
        d["hard"], d["hard"] / d["n"] * 100, d["hard"] / tot_hard * 100, d["flip"]))
print("TOTAL  errors_nt=%d  hard-core=%d" % (tot_err, tot_hard))
print("\nhard-core by choice count, per subject:")
for s, d in sorted(S.items(), key=lambda kv: -kv[1]["hard"]):
    print("  %-26s %s" % (s, dict(sorted(d["nc"].items()))))
