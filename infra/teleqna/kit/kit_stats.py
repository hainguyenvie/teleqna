#!/usr/bin/env python3
"""Kit quality report (data-auditor metrics, arXiv:2601.17717): per-view counts, gate pass rates, tokens,
amplification; lexical diversity (distinct-2/3, mean pairwise content-word Jaccard) among the views of the same
fact and among the registers of the same window; excerpt-style phrasing rate; latex-artifact rate."""
import json, glob, re, itertools, collections, random
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
STOP = set("the a an of in on for to and or is are be by with as at from that this which it its can may shall".split())
STYLE = re.compile(r"(?i)excerpt|passage|the document|text above|according to the")
def cw(s): return set(w for w in re.findall(r"[a-z0-9][a-z0-9.\-_]+", s.lower()) if w not in STOP and len(w) > 2)
def jac(a, b): return len(a & b) / max(1, len(a | b))
def distinct(texts, n):
    grams = collections.Counter()
    for t in texts:
        w = t.lower().split(); grams.update(tuple(w[i:i+n]) for i in range(len(w) - n + 1))
    return len(grams) / max(1, sum(grams.values()))
cnt = collections.Counter(); tok = collections.Counter(); style = collections.Counter(); tilde = collections.Counter()
fv = collections.defaultdict(list); regs = collections.defaultdict(list); src = 0
for f in glob.glob(str(R / "data/kit/tier1/views_s*.jsonl")):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); v = r["view"]; cnt[v, r["ok"]] += 1
        if not r["ok"]: continue
        t = r["text"] if v not in ("facts", "qa", "mcq") else " ".join(str(x) for x in json.loads(r["text"]).values())
        tok[v] += int(len(t.split()) * 1.35)
        if STYLE.search(t): style[v] += 1
        if "~" in t: tilde[v] += 1
        if v == "verbatim": src += int(len(t.split()) * 1.35)
        if v == "factview": fv[r["win_id"], r["fact"]].append(t)
        if v == "register": regs[r["win_id"]].append(t)
print("counts (view, ok):", dict(sorted(cnt.items())))
print("tokens kept per view:", {k: f"{v/1e6:.1f}M" for k, v in sorted(tok.items())}, f" source {src/1e6:.1f}M  amplification {sum(tok.values())/max(src,1):.1f}x")
print("excerpt-style phrasing rate:", {v: f"{style[v]/max(1,cnt[v,True])*100:.1f}%" for v in tok}, " tilde artifacts:", {v: f"{tilde[v]/max(1,cnt[v,True])*100:.1f}%" for v in tok})
rng = random.Random(1)
keys = rng.sample(list(fv), min(3000, len(fv)))
pj = [jac(cw(a), cw(b)) for k in keys for a, b in itertools.combinations(fv[k], 2)]
per = [len(fv[k]) for k in keys]
print(f"factview: views/fact mean {sum(per)/len(per):.1f}; pairwise content-word Jaccard mean {sum(pj)/len(pj):.3f} (share <0.5: {sum(x<0.5 for x in pj)/len(pj)*100:.0f}%); distinct-2 {distinct([t for k in keys for t in fv[k]],2):.3f} distinct-3 {distinct([t for k in keys for t in fv[k]],3):.3f}")
wk = rng.sample(list(regs), min(500, len(regs)))
rj = [jac(cw(a), cw(b)) for k in wk for a, b in itertools.combinations(regs[k], 2)]
print(f"registers: per window mean {sum(len(regs[k]) for k in wk)/len(wk):.1f}; pairwise Jaccard mean {sum(rj)/len(rj):.3f}; distinct-3 {distinct([t for k in wk for t in regs[k]],3):.3f}")
print("KIT_STATS_DONE")
