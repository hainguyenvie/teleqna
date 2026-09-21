#!/usr/bin/env python3
"""No-RAG cascade ceiling: Qwen3-8B answers when its label-free confidence (32-sample vote margin over 4
presentations, results/sweep/q3_8b_signal.jsonl) is >= tau, otherwise the second model answers.
Reports accuracy vs fraction deferred for several taus, the union oracle, and the vote-based 8B alone.
Usage: cascade_eval.py --primary otfull_q3_8b --secondary otfull_otel31b_cb [--signal results/sweep/q3_8b_signal.jsonl]"""
import json, argparse
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: x for x in d if isinstance(x, dict) and "sample_id" in x}
ap = argparse.ArgumentParser(); ap.add_argument("--primary", required=True); ap.add_argument("--secondary", required=True)
ap.add_argument("--signal", default="results/sweep/q3_8b_signal.jsonl"); a = ap.parse_args()
P = L(R / f"results/landscape/{a.primary}_base_nothink512.json"); S = L(R / f"results/landscape/{a.secondary}_base_nothink512.json")
sig = {json.loads(l)["sample_id"]: json.loads(l) for l in open(R / a.signal)}
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
ids = [q for q in P if q in S and q in sig]; n = len(ids)
acc = lambda f: sum(f(q) for q in ids) / n * 100
print(f"n={n}  primary {acc(lambda q: P[q]['correct']):.2f}  secondary {acc(lambda q: S[q]['correct']):.2f}  "
      f"union oracle {acc(lambda q: P[q]['correct'] or S[q]['correct']):.2f}  vote-top 8B alone {acc(lambda q: sig[q]['vote_top'] == test[q]['answer']):.2f}")
print(f"{'rule':34s} {'deferred%':>9s} {'acc':>7s}")
for name, cond in [("unanimous_views (defer if not)", lambda s: s["unanimous_views"])] + \
                  [(f"margin >= {t}", (lambda t: lambda s: s["margin"] >= t)(t)) for t in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0)]:
    keep = [q for q in ids if cond(sig[q])]; defer = n - len(keep)
    a1 = (sum(P[q]["correct"] for q in keep) + sum(S[q]["correct"] for q in ids if not cond(sig[q]))) / n * 100
    a2 = (sum(sig[q]["vote_top"] == test[q]["answer"] for q in keep) + sum(S[q]["correct"] for q in ids if not cond(sig[q]))) / n * 100
    print(f"{name:34s} {defer/n*100:8.1f}% {a1:7.2f}  (8B vote-top instead of greedy: {a2:.2f})")
