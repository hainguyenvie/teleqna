#!/usr/bin/env python3
"""Have the diagnosed problems been fixed? Bucket accuracy per checkpoint (CPU, gold used for measurement only).
Buckets: length bias (gold longest / not / shortest), All-of-the-above prior, lexical-overlap shortcut, gold position,
numeric gold, option count, 'how many/max/min', near-duplicate options; never-core (wrong in every OLD sane ckpt) fixed;
lineage unions. Usage: status_buckets.py TAG [TAG ...]"""
import json, re, sys, collections, unicodedata, itertools
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; LS = R / "results/landscape"
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
STOP = set("the a an of in on for to and or is are be by with as at from that this which what how does do it its".split())
def words(s): return {w for w in re.findall(r"[a-z0-9][a-z0-9.\-_]+", s.lower()) if w not in STOP and len(w) > 2}
def L(tag):
    p = LS / f"{tag}_otfull10000_base_nothink512.json"
    if not p.exists(): p = LS / f"{tag}_otfull10000_base_nothink32.json"
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: x for x in d if isinstance(x, dict) and "sample_id" in x}
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
Q = list(test)
tags = sys.argv[1:] or ["base", "vd6_ep1", "vd8_ep1", "ens_ep1", "merge_f", "big4_step2500"]
res = {}
for t in tags:
    try: res[t] = L("otfull_q3_8b" if t == "base" else t) if t != "base" else L.__globals__["json"] and None
    except FileNotFoundError: print("missing", t)
base = json.load(open(LS / "otfull_q3_8b_base_nothink512.json")); base = base["results"] if isinstance(base, dict) and "results" in base else base
if isinstance(base, dict): base = list(base.values())
res["base"] = {x["sample_id"]: x for x in base if isinstance(x, dict) and "sample_id" in x}
for t in tags:
    if t != "base": res[t] = L(t)
tags = ["base"] + [t for t in tags if t != "base"]
# per-question features
F = {}
for q, r in test.items():
    ch = r["choices"]; g = r["answer"]; lens = [len(c) for c in ch]; qw = words(r["question"])
    ov = [len(qw & words(c)) for c in ch]
    jac = max((len(words(a) & words(b)) / max(1, len(words(a) | words(b))) for a, b in itertools.combinations(ch, 2)), default=0)
    F[q] = dict(gold_longest=lens[g] == max(lens), gold_shortest=lens[g] == min(lens), gold_all=bool(re.search(r"(?i)all of the above|both", ch[g])),
                has_all=any(re.search(r"(?i)all of the above", c) for c in ch), gold_maxov=ov[g] == max(ov) and max(ov) > 0,
                pos=chr(65 + g), numeric=bool(re.fullmatch(r"[\d.,%\s\-]+", ch[g].strip())), nopt=len(ch),
                howmany=bool(re.search(r"(?i)how many|maximum|minimum|max\.|min\.|number of", r["question"])), neardup=jac >= 0.7,
                std=bool(re.search(r"\[3GPP Release|\bTS \d|\bTR \d|IEEE|ETSI|RFC", r["question"])), subject=r.get("subject", ""))
def acc(t, qs): return 100 * sum(1 for q in qs if res[t][q]["correct"]) / max(1, len(qs))
def pred(t, q): return (res[t][q].get("parsed") or "")
buckets = [("ALL", lambda f: True), ("gold=longest", lambda f: f["gold_longest"]), ("gold!=longest", lambda f: not f["gold_longest"]),
           ("gold=shortest", lambda f: f["gold_shortest"]), ("All-of-above gold", lambda f: f["gold_all"]), ("has All, gold not All", lambda f: f["has_all"] and not f["gold_all"]),
           ("gold=max-overlap", lambda f: f["gold_maxov"]), ("gold!=max-overlap", lambda f: not f["gold_maxov"]), ("numeric gold", lambda f: f["numeric"]),
           ("5-option", lambda f: f["nopt"] == 5), ("4-option", lambda f: f["nopt"] == 4), ("how many/max/min", lambda f: f["howmany"]), ("near-dup options", lambda f: f["neardup"]),
           ("std-spec question", lambda f: f["std"])] + [(f"gold={p}", (lambda p: lambda f: f["pos"] == p)(p)) for p in "ABCDE"]
print("bucket".ljust(24) + "n".rjust(6) + "".join(t[:14].rjust(15) for t in tags))
for name, fn in buckets:
    qs = [q for q in Q if fn(F[q])]
    print(name.ljust(24) + str(len(qs)).rjust(6) + "".join(f"{acc(t, qs):15.1f}" for t in tags))
print("\npredicted-letter share (%)  (gold share: " + " ".join(f"{p}={100*sum(1 for q in Q if F[q]['pos']==p)/len(Q):.1f}" for p in "ABCDE") + ")")
for t in tags:
    c = collections.Counter(pred(t, q) for q in Q); print(f"  {t:16s} " + " ".join(f"{p}={100*c[p]/len(Q):.1f}" for p in "ABCDE") + f"  unparsed={c['']}")
print("\nwrong-answer shape (count among wrong): chose longest / chose All / chose max-overlap / chose shortest")
for t in tags:
    W = [q for q in Q if not res[t][q]["correct"]]
    def chose(q, key):
        p = pred(t, q); ch = test[q]["choices"]
        if not p or ord(p) - 65 >= len(ch): return False
        i = ord(p) - 65; lens = [len(c) for c in ch]
        if key == "longest": return lens[i] == max(lens)
        if key == "shortest": return lens[i] == min(lens)
        if key == "all": return bool(re.search(r"(?i)all of the above", ch[i]))
        qw = words(test[q]["question"]); ov = [len(qw & words(c)) for c in ch]; return ov[i] == max(ov) and max(ov) > 0
    print(f"  {t:16s} wrong={len(W):5d}  longest={sum(chose(q,'longest') for q in W):4d}  All={sum(chose(q,'all') for q in W):4d}  maxov={sum(chose(q,'maxov') for q in W):4d}  shortest={sum(chose(q,'shortest') for q in W):4d}")
# never core over OLD sane checkpoints (everything evaluated before big4/ens/merge)
old = [f[:-len("_otfull10000_base_nothink512.json")] for f in __import__("os").listdir(LS) if f.endswith("_otfull10000_base_nothink512.json")]
old = [t for t in old if not re.match(r"big4_|ens|merge_", t) and t != "otfull_q3_8b"]
oldres = {}
for t in old:
    try:
        r = L(t)
        if len(r) == len(Q) and acc.__globals__ and 100 * sum(1 for q in Q if r[q]["correct"]) / len(Q) >= 74: oldres[t] = r
    except Exception: pass
never = [q for q in Q if all(not oldres[t][q]["correct"] for t in oldres)]
flick = [q for q in Q if not all(oldres[t][q]["correct"] for t in oldres) and q not in never]
print(f"\nold sane ckpts {len(oldres)}: never-core {len(never)}, flicker {len(flick)}, oracle-union {100*(len(Q)-len(never))/len(Q):.2f}")
for t in tags: print(f"  {t:16s} never-core fixed {sum(1 for q in never if res[t][q]['correct']):4d}/{len(never)}   flicker correct {acc(t, flick):5.1f}%")
new = [t for t in tags if t != "base"]
print("\nlineage unions:")
for a, b in itertools.combinations(new, 2):
    u = sum(1 for q in Q if res[a][q]["correct"] or res[b][q]["correct"]); both = sum(1 for q in Q if res[a][q]["correct"] and res[b][q]["correct"])
    print(f"  {a:14s} ∪ {b:14s} = {100*u/len(Q):.2f}  (both {100*both/len(Q):.2f}; a-only {sum(1 for q in Q if res[a][q]['correct'] and not res[b][q]['correct'])}, b-only {sum(1 for q in Q if res[b][q]['correct'] and not res[a][q]['correct'])})")
u = sum(1 for q in Q if any(res[t][q]["correct"] for t in new)); print(f"  union of all {len(new)} new = {100*u/len(Q):.2f}; still-never (incl. new) {sum(1 for q in never if not any(res[t][q]['correct'] for t in new))}")
# subject split
subs = collections.Counter(F[q]["subject"] for q in Q)
print("\nby subject (top 8):"); print("subject".ljust(30) + "n".rjust(6) + "".join(t[:14].rjust(15) for t in tags))
for s, n in subs.most_common(8):
    qs = [q for q in Q if F[q]["subject"] == s]; print(str(s)[:30].ljust(30) + str(n).rjust(6) + "".join(f"{acc(t, qs):15.1f}" for t in tags))
print("STATUS_DONE")
