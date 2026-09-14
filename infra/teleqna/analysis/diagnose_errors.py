#!/usr/bin/env python3
"""Diagnose WHY tuning sits ~76 when the eval-time probe ceiling was ~90.

Splits the gap into (a) what DPO actually fixed/broke on dev-1000, (b) the
structure of persistent errors, (c) whether the DPO pair pool even covers the
content of the persistent errors (data gap) or covers them and still fails to
transfer (method gap). Pure JSON + arithmetic, runs on the host, no GPU.
"""
import json, math, re, collections

ROOT = "/home/tensara/projects/telelogs"
SFT = ROOT + "/runs/teleqna-sft"

def load(p):
    return json.load(open(p))

def loadl(p):
    return [json.loads(l) for l in open(p)]

dev = {r["sample_id"]: r for r in loadl(SFT + "/data/dev1000.jsonl")}

def resmap(path):
    d = load(path)
    return {r["sample_id"]: bool(r["correct"]) for r in d["results"]}

base_nt = resmap(SFT + "/results/dev1000_base_aa_nothink.json")
dpo2_nt = resmap(SFT + "/results/dev1000_dpo2_step100.json")
dpo3_nt = resmap(SFT + "/results/dev1000_dpo3_step200.json")
dpo3_th = resmap(SFT + "/results/dev1000think_dpo3_step200.json")
# base think per-row from the official vLLM run (cross-stack but per-row)
b0_think = {r["sample_id"]: bool(r["correct"])
            for r in loadl(ROOT + "/runs/bench4/teleqna/results/b0_think/results.jsonl")
            if r["sample_id"] in dev}

def flips(base, tuned, name):
    fixed = [s for s in dev if not base[s] and tuned[s]]
    broke = [s for s in dev if base[s] and not tuned[s]]
    print(f"{name}: base_wrong={sum(not v for v in base.values())} "
          f"fixed={len(fixed)} broke={len(broke)} net={len(fixed)-len(broke)}")
    return set(fixed), set(broke)

print("== FLIPS (no-think, same stack) ==")
f2, b2 = flips(base_nt, dpo2_nt, "dpo2@100")
f3, b3 = flips(base_nt, dpo3_nt, "dpo3@200")
print("\n== FLIPS (think; base=vLLM b0_think per-row, cross-stack) ==")
f3t, b3t = flips(b0_think, dpo3_th, "dpo3@200-think")

# persistent wrong: wrong at base AND wrong after best tuning, both modes
persist = [s for s in dev
           if not base_nt[s] and not dpo3_nt[s] and not b0_think[s] and not dpo3_th[s]]
wrong_any_base = [s for s in dev if not base_nt[s] or not b0_think[s]]
print(f"\npersistent-wrong (all 4 arms wrong): {len(persist)}")

def dist(ids, key):
    c = collections.Counter(key(dev[s]) for s in ids)
    return dict(c.most_common())

ALL_ABOVE = re.compile(r"(?i)all of the above")
BOTH = re.compile(r"(?i)^both ")
def shape(r):
    ch = r["choices"]
    if any(ALL_ABOVE.search(c) for c in ch): return "has_all_above"
    if any(BOTH.match(c) for c in ch): return "has_both"
    return "plain"

print("\n== persistent-wrong structure ==")
print("subject:", dist(persist, lambda r: r.get("subject", "?")))
print("n_choices:", dist(persist, lambda r: len(r["choices"])))
print("shape:", dist(persist, shape))
print("subject of FIXED (dpo3 think):", dist(f3t, lambda r: r.get("subject", "?")))
print("subject of BROKEN (dpo3 think):", dist(b3t, lambda r: r.get("subject", "?")))

# ---- coverage: max IDF-cosine of each dev row vs the DPO pair pool ----
W = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
def toks(t):
    return set(W.findall(t.lower()))

pairs = loadl(SFT + "/data/dpo_pairs_v2.jsonl")
# one doc per unique prompt (fix+anchor share prompts rarely; dedup by prompt)
seen, pdocs = set(), []
for p in pairs:
    if p["prompt"] in seen: continue
    seen.add(p["prompt"]); pdocs.append((p["sample_id"], toks(p["prompt"]), p["kind"]))
print(f"\npair-pool unique prompts: {len(pdocs)}")

df = collections.Counter()
for _, tk, _ in pdocs:
    for t in tk: df[t] += 1
N = len(pdocs)
idf = {t: math.log(N / c) for t, c in df.items()}

inv = collections.defaultdict(list)
for i, (_, tk, _) in enumerate(pdocs):
    for t in tk:
        if df[t] < N * 0.2:      # skip near-stopwords in the index
            inv[t].append(i)

norm = [math.sqrt(sum(idf.get(t, 0) ** 2 for t in tk)) or 1 for _, tk, _ in pdocs]

def best_match(r):
    qtk = toks(r["question"] + " " + " ".join(r["choices"]))
    qn = math.sqrt(sum(idf.get(t, 0) ** 2 for t in qtk)) or 1
    scores = collections.defaultdict(float)
    for t in qtk:
        w = idf.get(t)
        if w is None or df[t] >= N * 0.2: continue
        for i in inv[t]:
            scores[i] += w * w
    if not scores: return 0.0, None
    i, s = max(scores.items(), key=lambda kv: kv[1])
    return s / (qn * norm[i]), pdocs[i][0]

classes = {
    "fixed_think": list(f3t), "broken_think": list(b3t),
    "persistent": persist,
    "always_right": [s for s in dev if base_nt[s] and dpo3_nt[s] and b0_think[s] and dpo3_th[s]],
}
out_rows = {}
print("\n== coverage: max IDF-cosine vs dpo pair pool, by outcome class ==")
for name, ids in classes.items():
    sims = []
    for s in ids:
        sim, pid = best_match(dev[s])
        sims.append(sim)
        out_rows[s] = {"class": name, "sim": round(sim, 3), "top_pair": pid}
    sims.sort()
    if sims:
        q = lambda f: round(sims[int(f * (len(sims) - 1))], 3)
        print(f"{name:13s} n={len(sims):4d} p25={q(.25)} med={q(.5)} p75={q(.75)} "
              f"frac>0.35={round(sum(x > 0.35 for x in sims)/len(sims), 3)}")

json.dump(out_rows, open(SFT + "/results/diag_coverage.json", "w"))

# a few persistent examples with their best pair, for eyeballing
print("\n== 5 persistent-wrong examples (worst-covered) ==")
ps = sorted(persist, key=lambda s: out_rows[s]["sim"])
for s in ps[:5]:
    r = dev[s]
    print(f"--- {s} sim={out_rows[s]['sim']} subj={r.get('subject')}")
    print("Q:", r["question"][:220].replace("\n", " "))
print("\n== 3 persistent-wrong examples (BEST-covered, sim>0.5) ==")
for s in [x for x in ps if out_rows[x]["sim"] > 0.5][-3:]:
    r = dev[s]
    print(f"--- {s} sim={out_rows[s]['sim']} top_pair={out_rows[s]['top_pair']} subj={r.get('subject')}")
    print("Q:", r["question"][:220].replace("\n", " "))
