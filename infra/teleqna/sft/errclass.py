#!/usr/bin/env python3
"""Classify what is still wrong after arm E, so the next arm aims at something.

Three cuts, in order of what would change the plan:

 1. regression vs persistent -- a row arm E BROKE is a different problem from a
    row it never solved, and mixing them has hidden both before.
 2. does the evidence exist? a row whose source passage was never retrieved
    cannot be fixed by any amount of training on retrieved evidence; a row whose
    passage WAS retrieved and is still wrong is a comprehension/discrimination
    failure and is addressable.
 3. shape of the item -- near-synonym options, "all of the above", numeric
    answers, negated stems. These need different treatments, which is the
    divide-and-conquer the plan has been assuming without ever measuring.

Every feature here is computable without the label except the ones used only for
reporting accuracy inside a cluster.
"""
import json, os, re, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
R = f"{ROOT}/results"
CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")

ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

def load(p):
    d = json.load(open(p))
    return {r["sample_id"]: r for r in d["results"]}

arm = load(f"{R}/landscape/otfull_armEwide_armEwide_nothink512.json")
base = load(f"{R}/landscape/otfull_armEwide_base_nothink512.json")
buckets = json.load(open(f"{R}/error_buckets.json"))
bof = {}
for b, ids in buckets.items():
    for i in ids:
        bof.setdefault(i, set()).add(b)
trained = {}
for l in open(f"{ROOT}/data/train/eligible/armE_wide.jsonl"):
    r = json.loads(l)
    trained[r["sample_id"]] = r["tier"]

# was the answer ever retrievable? use whichever strong-evidence set covers it
TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")
eviden = {}
for f in ["escalate2730_rag8_strong.jsonl", "blind6202_rag8_strong.jsonl"]:
    p = f"{ROOT}/data/{f}"
    if not os.path.exists(p):
        continue
    for l in open(p):
        r = json.loads(l)
        ch = r["choices"]
        ch = eval(ch) if isinstance(ch, str) else ch
        body = r["question"].split("\n\n---\n\n")[0]
        ctx = set(TOKEN.findall(body.lower()))
        g = set(TOKEN.findall(re.sub(r"\W+", " ", ch[int(r["answer"])].lower())))
        n_opt = sum(1 for c in ch
                    if (lambda t: bool(t) and len(t & ctx) >= 0.8 * len(t))(
                        set(TOKEN.findall(re.sub(r"\W+", " ", c.lower())))))
        eviden[r["sample_id"]] = (bool(g) and len(g & ctx) >= 0.8 * len(g), n_opt)

STOP = set("the a an of to in for and or is are be by on with as at from that this it its".split())
def toks(s):
    return {w for w in TOKEN.findall(s.lower()) if w not in STOP}

CATCHALL = re.compile(r"\b(all|none|both) of the (above|these)\b|^(all|none) of them", re.I)
NUMERIC = re.compile(r"\d")
NEG = re.compile(r"\b(not|never|cannot|except|incorrect|false|un\w+ed)\b", re.I)

def shape(sid, chosen):
    r = ref[sid]
    ch = r["choices"]
    gi = int(r["answer"])
    g = ch[gi]
    feats = []
    if any(CATCHALL.search(c) for c in ch):
        feats.append("catch-all option present")
    if sum(1 for c in ch if NUMERIC.search(c)) >= max(2, len(ch) - 1):
        feats.append("numeric/quantity")
    if NEG.search(r["question"]):
        feats.append("negated stem")
    if chosen and chosen != "?" and 0 <= ord(chosen) - 65 < len(ch):
        c = ch[ord(chosen) - 65]
        tg, tc = toks(g), toks(c)
        if tg and tc:
            j = len(tg & tc) / len(tg | tc)
            if j >= 0.4:
                feats.append("near-synonym of gold")
            elif j >= 0.15:
                feats.append("partial overlap with gold")
        if len(c) > 2.0 * len(g):
            feats.append("chose much longer option")
        elif len(g) > 2.0 * len(c):
            feats.append("chose much shorter option")
    if not feats:
        feats.append("no obvious shape")
    return feats

ids = sorted(set(arm) & set(base))
wrong = [s for s in ids if not arm[s]["correct"]]
regress = [s for s in wrong if base[s]["correct"]]
persist = [s for s in wrong if not base[s]["correct"]]
print(f"arm E wrong on {len(wrong)}/{len(ids)} rows "
      f"({len(regress)} regressions, {len(persist)} never solved)\n")

print("=== 1. by training exposure ===")
c = collections.Counter(trained.get(s, "never trained") for s in wrong)
for k, v in c.most_common():
    tot = sum(1 for s in ids if trained.get(s, "never trained") == k)
    print(f"  {k:15s} wrong {v:5d} / {tot:5d} = {v/tot*100:5.1f}%")

print("\n=== 2. was the answer retrievable? ===")
cov = [s for s in wrong if s in eviden]
print(f"  evidence set covers {len(cov)}/{len(wrong)} of the wrong rows")
if cov:
    hit = [s for s in cov if eviden[s][0]]
    print(f"    passage HAS the answer  : {len(hit):5d} ({len(hit)/len(cov)*100:.1f}%) "
          f"-> discrimination failure, addressable")
    print(f"    passage does NOT        : {len(cov)-len(hit):5d} "
          f"-> retrieval failure, training cannot fix")
    h = collections.Counter(eviden[s][1] for s in hit)
    print("    of the hits, options co-present in the window: "
          + ", ".join(f"{k}:{v}" for k, v in sorted(h.items())))

print("\n=== 3. item shape (wrong rows) ===")
sc = collections.Counter()
for s in wrong:
    for f in shape(s, arm[s]["parsed"]):
        sc[f] += 1
scall = collections.Counter()
for s in ids:
    for f in shape(s, arm[s]["parsed"]):
        scall[f] += 1
print("  %-32s %7s %7s %8s" % ("feature", "wrong", "all", "err rate"))
for k, v in sc.most_common():
    print("  %-32s %7d %7d %7.1f%%" % (k, v, scall[k], v / scall[k] * 100))

print("\n=== 4. by subject ===")
sub = collections.Counter(ref[s]["subject"] for s in wrong)
suball = collections.Counter(ref[s]["subject"] for s in ids)
for k, v in sub.most_common():
    print("  %-28s %5d / %5d = %5.1f%%" % (k, v, suball[k], v / suball[k] * 100))

json.dump({"wrong": wrong, "regress": regress, "persist": persist},
          open(f"{R}/landscape/armE_errors.json", "w"))
print(f"\nwrote {R}/landscape/armE_errors.json")
