import json, collections, sys
S = sys.argv[1]
cov, subj = collections.defaultdict(dict), {}
for sh in ("arxiv", "standard", "wiki"):
    for l in open(f"{S}/results/teledata/coverage_{sh}_w400.rows.jsonl"):
        r = json.loads(l)
        cov[r["sample_id"]][sh] = r["coverage"]
        subj[r["sample_id"]] = r["subject"]
res = json.load(open(f"{S}/results/vllm122b/otfull_base_think_merged.json"))["results"]
ok = {r["sample_id"]: bool(r["correct"]) for r in res}

# Fixed bands, not deciles: 3,791 rows tie at coverage 1.0, and any rank-based
# split cuts through that tie -- sorting the tuple then orders those rows by
# correctness, which manufactures a perfect correlation out of nothing.
BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.999), (0.999, 1.01)]
print("=== accuracy theo band coverage (122B ot-full, 80.90 official) ===")
for lo, hi in BANDS:
    ch = [ok[s] for s in ok if s in cov and lo <= max(cov[s].values()) < hi]
    if ch:
        print(f"  cov [{lo:.2f},{hi:.2f})  n={len(ch):5d}  acc={sum(ch)/len(ch)*100:5.2f}%")

print("\n=== per subject ===")
for s in sorted(set(subj.values())):
    print(f"  {s}")
    for lo, hi in BANDS:
        ch = [ok[x] for x in ok if x in cov and subj[x] == s
              and lo <= max(cov[x].values()) < hi]
        if len(ch) >= 20:
            print(f"     [{lo:.2f},{hi:.2f})  n={len(ch):5d}  acc={sum(ch)/len(ch)*100:5.2f}%")
