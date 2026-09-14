#!/usr/bin/env python3
"""Error-structure analysis for the Qwen3-8B pivot.

Joins every per-row artefact we own on sample_id and answers:
  - where the 8B's 2,805 errors live and how stable they are
  - which of them a stronger model / evidence / labels already solve
  - what label accuracy a distillation set would need for the 8B to reach 85
Read-only. Writes a text report to stdout.
"""
import json, os, re, collections, math

SFT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
B4 = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna")
LS = f"{SFT}/results/landscape"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_b4(name):
    """bench4 runner: results.jsonl with sample_id/correct/parsed_answer."""
    rows = load_jsonl(f"{B4}/results/{name}/results.jsonl")
    return {r["sample_id"]: r for r in rows}


def load_ls(name):
    """landscape runner: {'summary','results'} or a bare list."""
    d = json.load(open(f"{LS}/{name}.json"))
    rows = d["results"] if isinstance(d, dict) else d
    return {r["sample_id"]: r for r in rows}


def acc(d, key="correct"):
    n = len(d)
    return 100.0 * sum(1 for r in d.values() if r.get(key)) / max(n, 1)


def mcnemar(a, b, ids):
    """b vs a: returns (fixed, broken, p) two-sided exact binomial."""
    fixed = sum(1 for i in ids if not a[i]["correct"] and b[i]["correct"])
    broken = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    n, k = fixed + broken, min(fixed, broken)
    if n == 0:
        return fixed, broken, 1.0
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return fixed, broken, p


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


test = {f"teleqna-{i:05d}": r for i, r in enumerate(load_jsonl(f"{B4}/data/test.jsonl"))}
IDS = sorted(test)
print(f"test rows {len(IDS)}  fields {list(next(iter(test.values())))}")

runs = {}
for tag, name in [("8b_nothink", "b0_nothink"), ("8b_think", "b0_think"),
                  ("8b_nothink_rot", "b0_nothink_perm"), ("8b_think_rot", "b0_think_perm"),
                  ("otel8b_it", "o1_otel8b")]:
    try:
        runs[tag] = load_b4(name)
    except Exception as e:
        print(f"!! {tag}: {e}")

for tag, name in [("31b_base", "otfull_armGfull_base_nothink512"),
                  ("31b_armGfull", "otfull_armGfull_armGfull_nothink512"),
                  ("31b_armG", "otfull_armG_armG_nothink512"),
                  ("31b_expl", "otfull_expl_base_nothink512"),
                  ("31b_armH", "otfull_armH_armH_nothink512")]:
    try:
        runs[tag] = load_ls(name)
    except Exception as e:
        print(f"!! {tag}: {e}")

hr("1. headline — every run we can join, same 10,000 rows")
for tag, d in runs.items():
    cov = len(set(d) & set(IDS))
    print(f"  {tag:16s} n={cov:6d}  acc={acc(d):6.2f}")

# ---------------------------------------------------------------- 2. stability
hr("2. Qwen3-8B stability: which errors are knife-edge and which are hard-core")
arms = ["8b_nothink", "8b_think", "8b_nothink_rot", "8b_think_rot"]
have = [a for a in arms if a in runs]
cnt = collections.Counter()
nright = {}
for i in IDS:
    k = sum(1 for a in have if runs[a].get(i, {}).get("correct"))
    nright[i] = k
    cnt[k] += 1
print(f"  arms used: {have}")
for k in sorted(cnt, reverse=True):
    print(f"   right in {k}/{len(have)} arms : {cnt[k]:5d}  ({100*cnt[k]/len(IDS):5.2f}%)")
hardcore = [i for i in IDS if nright[i] == 0]
unstable = [i for i in IDS if 0 < nright[i] < len(have)]
solid = [i for i in IDS if nright[i] == len(have)]
print(f"\n  solid-right {len(solid)}   unstable {len(unstable)}   hard-core wrong {len(hardcore)}")
print(f"  -> {100*len(unstable)/len(IDS):.2f}% of the benchmark is decided by presentation, not knowledge")
print(f"  -> best achievable by fixing only consistency = {100*(len(solid)+len(unstable))/len(IDS):.2f}%")

# ---------------------------------------------------------------- 3. taxonomy
hr("3. where the no-think errors live")
base = runs["8b_nothink"]
wrong = [i for i in IDS if not base.get(i, {}).get("correct")]
print(f"  errors {len(wrong)} of {len(IDS)}")


def bucketise(keyfn, title):
    tot, err, hard = collections.Counter(), collections.Counter(), collections.Counter()
    for i in IDS:
        k = keyfn(i)
        tot[k] += 1
        if not base.get(i, {}).get("correct"):
            err[k] += 1
        if nright[i] == 0:
            hard[k] += 1
    print(f"\n  by {title}")
    print(f"    {'bucket':34s} {'rows':>6s} {'acc':>7s} {'errors':>7s} {'%err':>6s} {'hardcore':>9s}")
    for k, n in sorted(tot.items(), key=lambda kv: -err[kv[0]]):
        print(f"    {str(k):34s} {n:6d} {100*(n-err[k])/n:6.2f}% {err[k]:7d} "
              f"{100*err[k]/max(len(wrong),1):5.1f}% {hard[k]:9d}")


bucketise(lambda i: test[i]["subject"], "subject")
bucketise(lambda i: f"{len(test[i]['choices'])}-choice", "choice count")

SHAPE = re.compile(r"all of the above|both .+ and |none of the above", re.I)
bucketise(lambda i: ("shape-cued" if any(SHAPE.search(c) for c in test[i]["choices"])
                     else "plain"), "distractor shape cue")


def goldlen_bucket(i):
    ch = test[i]["choices"]
    g = ch[test[i]["answer"]]
    if len(g) == max(len(c) for c in ch):
        return "gold is longest option"
    if len(g) == min(len(c) for c in ch):
        return "gold is shortest option"
    return "gold in the middle"


bucketise(goldlen_bucket, "gold option length rank")


def qlen_bucket(i):
    w = len(test[i]["question"].split())
    return "q <= 10 words" if w <= 10 else ("q 11-20 words" if w <= 20 else "q > 20 words")


bucketise(qlen_bucket, "question length")


def tag_bucket(i):
    q = test[i]["question"]
    m = re.search(r"\[([^\]]+)\]\s*$", q)
    if not m:
        return "no provenance tag"
    t = m.group(1)
    if "3GPP" in t:
        return "3GPP " + (re.search(r"Release \d+", t).group(0) if re.search(r"Release \d+", t) else "other")
    if "IEEE" in t:
        return "IEEE"
    return "other tag"


bucketise(tag_bucket, "provenance tag")

# predicted letter distribution on errors
pl = collections.Counter(base[i].get("parsed_answer") for i in wrong)
gl = collections.Counter(chr(65 + test[i]["answer"]) for i in wrong)
print("\n  letter distribution on the 8B's errors")
print("    predicted", dict(sorted(pl.items(), key=lambda kv: str(kv[0]))))
print("    gold     ", dict(sorted(gl.items(), key=lambda kv: str(kv[0]))))

# ---------------------------------------------------------------- 4. crossmodel
hr("4. cross-model: what already solves the 8B's errors")
pairs = [("31b_base", "OTel-2.0-31B-IT closed book"),
         ("31b_armGfull", "31B + arm G-full adapter"),
         ("31b_expl", "31B with the gold explanation in context (ceiling)"),
         ("otel8b_it", "OTel-LLM-8B-IT (released AT&T 8B)"),
         ("8b_think", "the same 8B with thinking on")]
print(f"  {'other model':44s} {'8Bwrong->right':>15s} {'8Bright->wrong':>15s} {'union':>8s}")
for tag, label in pairs:
    if tag not in runs:
        continue
    o = runs[tag]
    ids = [i for i in IDS if i in o]
    f_, b_, _ = mcnemar(base, o, ids)
    union = sum(1 for i in ids if base[i]["correct"] or o[i]["correct"])
    print(f"  {label:44s} {f_:15d} {b_:15d} {100*union/len(ids):7.2f}%")

if "31b_expl" in runs:
    e = runs["31b_expl"]
    dead = [i for i in IDS if i in e and not e[i]["correct"]]
    print(f"\n  rows even the gold explanation cannot fix: {len(dead)} ({100*len(dead)/len(IDS):.2f}%)")
    dead_and_8bwrong = [i for i in dead if not base[i]["correct"]]
    print(f"   of which the 8B also gets wrong: {len(dead_and_8bwrong)}")
    print(f"  -> practical maximum for any 8B on this benchmark ~ {100*(len(IDS)-len(dead))/len(IDS):.2f}%")

# joint frontier: rows nobody gets
avail = [t for t in ["31b_base", "31b_armGfull", "8b_think", "otel8b_it"] if t in runs]
nobody = [i for i in IDS if not base.get(i, {}).get("correct")
          and not any(runs[t].get(i, {}).get("correct") for t in avail)]
print(f"\n  8B-wrong AND wrong in every other model we have ({avail}): {len(nobody)}")
sub = collections.Counter(test[i]["subject"] for i in nobody)
for k, v in sub.most_common():
    print(f"     {k:30s} {v:5d}")

# ---------------------------------------------------------------- 5. labels
hr("5. label sources: what accuracy a distillation set would have on the 10,000")
lab = {}
try:
    for r in load_jsonl(f"{SFT}/data/train/eligible/armG_full.jsonl"):
        m = re.search(r"ANSWER:\s*([A-E])", r["completion"])
        if m:
            lab[r["sample_id"]] = (m.group(1), r.get("tier"))
except Exception as ex:
    print("armG_full labels unreadable:", ex)

if lab:
    tot = collections.Counter()
    ok = collections.Counter()
    for i, (letter, tier) in lab.items():
        if i not in test:
            continue
        good = (letter == chr(65 + test[i]["answer"]))
        tot[tier] += 1
        ok[tier] += good
        tot["ALL"] += 1
        ok["ALL"] += good
    print("  arm G-full label set (no answer key used to build it)")
    for t in sorted(tot, key=lambda t: -tot[t]):
        print(f"    {str(t):10s} n={tot[t]:6d}  label accuracy {100*ok[t]/tot[t]:6.2f}%")
    # what the 8B currently answers on each tier
    print("\n  the 8B's own accuracy on those tiers, and how far a fitted model would move")
    for t in sorted(tot, key=lambda t: -tot[t]):
        if t == "ALL":
            continue
        ids = [i for i in lab if lab[i][1] == t and i in base]
        cur = 100 * sum(1 for i in ids if base[i]["correct"]) / len(ids)
        agree = 100 * sum(1 for i in ids if base[i].get("parsed_answer") == lab[i][0]) / len(ids)
        print(f"    {t:10s} 8B now {cur:6.2f}%   label acc {100*ok[t]/tot[t]:6.2f}%   "
              f"8B already agrees with label {agree:6.2f}%")

for tag, label in [("31b_base", "31B closed-book predictions as labels"),
                   ("31b_armGfull", "31B+armG-full predictions as labels")]:
    if tag in runs:
        o = runs[tag]
        ids = [i for i in IDS if i in o]
        a = 100 * sum(1 for i in ids if o[i]["correct"]) / len(ids)
        agree = 100 * sum(1 for i in ids if o[i].get("parsed") == base.get(i, {}).get("parsed_answer")) / len(ids)
        print(f"\n  {label}: accuracy {a:.2f}%  (8B already agrees on {agree:.2f}% of rows)")

# oracle of small ensembles
combos = [("8b_nothink", "31b_base"), ("8b_nothink", "8b_think"),
          ("31b_base", "31b_armGfull"), ("8b_think", "31b_armGfull")]
print("\n  oracle of pairs (upper bound of any router over the two):")
for a, b in combos:
    if a in runs and b in runs:
        ids = [i for i in IDS if i in runs[a] and i in runs[b]]
        o = 100 * sum(1 for i in ids if runs[a][i]["correct"] or runs[b][i]["correct"]) / len(ids)
        print(f"    {a:14s} + {b:14s} -> {o:6.2f}%")

# ---------------------------------------------------------------- 6. to 85
hr("6. arithmetic to 85% from the Qwen3-8B no-think baseline")
cur = acc(base)
print(f"  now {cur:.2f}%  target 85.00%  -> need {int(round((85 - cur) * 100))} more correct rows")
print(f"  of the {len(wrong)} errors that is {100*(85-cur)*100/len(wrong):.1f}% of them\n")
print(f"  {'subject':30s} {'rows':>6s} {'acc':>7s} {'errors':>7s} {'unstable':>9s} {'31B fixes':>10s}")
for s in sorted({test[i]["subject"] for i in IDS}):
    ids = [i for i in IDS if test[i]["subject"] == s]
    e = [i for i in ids if not base[i]["correct"]]
    uns = [i for i in e if nright[i] > 0]
    f31 = [i for i in e if runs.get("31b_base", {}).get(i, {}).get("correct")]
    print(f"  {s:30s} {len(ids):6d} {100*(len(ids)-len(e))/len(ids):6.2f}% {len(e):7d} "
          f"{len(uns):9d} {len(f31):10d}")
