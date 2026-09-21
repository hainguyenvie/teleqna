#!/usr/bin/env python3
"""Why did training push 588 base-right rows to a specific wrong option? For each broke row, count kit lines (facts /
factview / qa / mcq / register of the windows retrieved for that question) that contain the PICKED option text vs the
GOLD option text; same for fixed rows as control. Also check anchors (self-replay MCQs) whose answer text matches the
picked option. Gold used for measurement only. CPU."""
import json, re, unicodedata, collections, glob, sys
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; TAG = sys.argv[1] if len(sys.argv) > 1 else "kit2_ep1"
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: x for x in d}
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
B = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json"); T = L(R / f"results/landscape/{TAG}_otfull10000_base_nothink512.json")
def picked(x):
    m = re.search(r"ANSWER\s*:\s*([A-E])", x.get("completion", "")); return ord(m.group(1)) - 65 if m else None
broke = [q for q in B if B[q]["correct"] and not T[q]["correct"] and picked(T[q]) is not None]
fixed = [q for q in B if not B[q]["correct"] and T[q]["correct"]]
q2w = collections.defaultdict(set)
for f in ("data/kit/windows_keep.jsonl", "data/kit/windows_rest.jsonl"):
    for l in open(R / f, encoding="utf-8"):
        w = json.loads(l)
        for q in w.get("for_q", []): q2w[q].add(w["win_id"])
need = {}  # win_id -> list of (q, text_to_find, label)
for q in broke:
    g = norm(test[q]["choices"][test[q]["answer"]]); p = norm(test[q]["choices"][picked(T[q])])
    for w in q2w.get(q, ()): need.setdefault(w, []).append((q, g, "gold")); need[w].append((q, p, "picked"))
for q in fixed:
    g = norm(test[q]["choices"][test[q]["answer"]]); p = norm(test[q]["choices"][picked(B[q])]) if picked(B[q]) is not None else ""
    for w in q2w.get(q, ()): need.setdefault(w, []).append((q, g, "gold")); need[w].append((q, p, "picked"))
cnt = collections.defaultdict(collections.Counter)  # q -> {gold: n, picked: n}
files = glob.glob(str(R / "data/kit/tier1_fv30/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier1/views_s*.jsonl")) + glob.glob(str(R / "data/kit/tier2/views_s*.jsonl"))
for f in files:
    for l in open(f, encoding="utf-8"):
        i = l.find('"win_id": "'); w = l[i + 11:l.find('"', i + 11)] if i >= 0 else None
        if w not in need or '"view": "verbatim"' in l or ('"view": "factview"' in l and "tier1/views" in f): continue
        t = norm(l)
        for q, s, lab in need[w]:
            if len(s) >= 4 and s in t: cnt[q][lab] += 1
def summarize(qs, name):
    both = [(cnt[q]["gold"], cnt[q]["picked"]) for q in qs]
    more_p = sum(p > g for g, p in both); more_g = sum(g > p for g, p in both); zero = sum(g == 0 and p == 0 for g, p in both)
    print(f"{name}: n={len(qs)} | kit mentions picked>gold {more_p} ({more_p/len(qs)*100:.0f}%) gold>picked {more_g} ({more_g/len(qs)*100:.0f}%) neither in kit {zero} ({zero/len(qs)*100:.0f}%) | mean lines gold {sum(g for g,_ in both)/len(qs):.1f} picked {sum(p for _,p in both)/len(qs):.1f}")
summarize(broke, "BROKE (picked = new wrong option)"); summarize(fixed, "FIXED (picked = base's old wrong option)")
# anchors: self-replay MCQ rows whose answer text equals the picked option
anc = collections.Counter()
for f in ("data/kit/tier1/anchor_selfreplay.jsonl", "data/kit/tier2/anchor_selfreplay.jsonl"):
    for l in open(R / f, encoding="utf-8"):
        r = json.loads(l)
        if not r.get("agree", True): continue
        anc[norm(r.get("gen_answer_text") or r.get("answer_text") or "")] += 1
hits_p = sum(anc[norm(test[q]["choices"][picked(T[q])])] > 0 for q in broke); hits_g = sum(anc[norm(test[q]["choices"][test[q]["answer"]])] > 0 for q in broke)
print(f"anchors whose answer text equals: picked option {hits_p}/{len(broke)}  gold option {hits_g}/{len(broke)}  (anchor keys sample: {list(json.loads(open(R/'data/kit/tier1/anchor_selfreplay.jsonl').readline()).keys())})")
# samples
for q in broke[:6]:
    print(f"[{q}] {test[q]['question'][:110]} | GOLD: {test[q]['choices'][test[q]['answer']][:60]} | PICKED: {test[q]['choices'][picked(T[q])][:60]} | kit lines gold {cnt[q]['gold']} picked {cnt[q]['picked']}")
print("BROKE_AUDIT_DONE")
