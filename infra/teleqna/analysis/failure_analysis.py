#!/usr/bin/env python3
"""Failure analysis of base-wrong ot-full rows: where the knowledge is (corpus coverage classes), what separates
rows the trained model fixed from rows it did not, how the model is wrong (option-shape patterns), whether sources
conflict / release-mismatch, and a hand-readable sample. Gold is used for measurement only. CPU, server-side."""
import json, re, collections, math, random, unicodedata, sys
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: x for x in d if isinstance(x, dict) and "sample_id" in x}
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
B = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json"); TAG = sys.argv[1] if len(sys.argv) > 1 else "kit1lr2e5_ep1"; T = L(R / f"results/landscape/{TAG}_otfull10000_base_nothink512.json")
RG = L(R / "results/landscape/ragstrong_q3_8b_base_nothink512.json"); W = L(R / "results/landscape/win_single_q3_8b_base_nothink512.json")
sig = {json.loads(l)["sample_id"]: json.loads(l) for l in open(R / "results/sweep/q3_8b_signal.jsonl")}
keep = {json.loads(l)["win_id"] for l in open(R / "data/kit/windows_keep.jsonl", encoding="utf-8")}
gold = {q: chr(65 + r["answer"]) for q, r in test.items()}; bl = {q: (x.get("parsed") or "") for q, x in B.items()}
perq = collections.defaultdict(list)
for k, x in W.items():
    q, w = k.split("|"); perq[q].append((w, x.get("parsed") or ""))
wrong = [q for q in test if not B[q]["correct"]]
wtext = {}; need = {w for q in wrong for w, _ in perq[q]}
for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
    w = json.loads(l)
    if w["win_id"] in need: wtext[w["win_id"]] = w["text"]
# kit text (facts/factview/qa) for those windows, from tier1 + tier2 views (streaming)
kit_txt = collections.defaultdict(list)
import glob
for f in sorted(glob.glob(str(R / "data/kit/tier1/views_s*.jsonl"))) + sorted(glob.glob(str(R / "data/kit/tier2/views_s*.jsonl"))):
    for l in open(f, encoding="utf-8"):
        if '"view": "facts"' not in l and '"view": "factview"' not in l and '"view": "qa"' not in l: continue
        r = json.loads(l)
        if r["win_id"] in need and r["ok"]: kit_txt[r["win_id"]].append(norm(r["text"]))
print(f"base-wrong {len(wrong)}; windows loaded {len(wtext)}; kit windows with text {len(kit_txt)}", flush=True)

def gold_in(q, texts):
    g = norm(test[q]["choices"][test[q]["answer"]])
    return len(g) >= 4 and any(g in t for t in texts)
rec = []
for q in wrong:
    r = test[q]; ws = perq[q]; letters = [p for _, p in ws if p]
    fixw = [w for w, p in ws if p == gold[q]]
    cls = ("single-window fixes" if fixw else "RAG-8 combination fixes" if RG[q]["correct"] else
           "gold text in some window, not used" if gold_in(q, [wtext.get(w, "") for w, _ in ws]) else "gold text absent from windows")
    rel = re.search(r"\[3GPP Release (\d+)\]", r["question"]); relq = int(rel.group(1)) if rel else None
    relmis = None
    if relq and fixw:
        rels = set(int(x) for w in fixw for x in re.findall(r"(?:Release|Rel-?)\s?(\d{1,2})\b", wtext.get(w, "")))
        relmis = bool(rels) and relq not in rels
    pred = bl[q]; ch = r["choices"]; pi = ord(pred) - 65 if pred else -1
    lens = [len(c) for c in ch]
    rec.append(dict(q=q, subject=r["subject"], cls=cls, n_fix=len(fixw), n_kept=sum(1 for w, _ in ws if w in keep),
                    n_letters=len(set(letters)), margin=sig.get(q, {}).get("margin", 1.0), fixed=bool(T[q]["correct"]),
                    gold_in_kit=gold_in(q, [t for w in fixw for t in kit_txt.get(w, [])]) if fixw else False,
                    n_fv_gold=sum(1 for w in fixw for t in kit_txt.get(w, []) if norm(ch[r["answer"]]) in t) if fixw else 0,
                    pred_longest=(pi >= 0 and lens[pi] == max(lens)), gold_longest=(lens[r["answer"]] == max(lens)),
                    pred_all=(pi >= 0 and bool(re.search(r"(?i)all of the above|both", ch[pi]))), gold_all=bool(re.search(r"(?i)all of the above|both", ch[r["answer"]])),
                    pred_none=(pi >= 0 and bool(re.search(r"(?i)none of the above", ch[pi]))), numeric_gold=bool(re.fullmatch(r"[\d.,%\s\-]+", ch[r["answer"]].strip())),
                    relq=relq, relmis=relmis, unparsed=(pi < 0)))
def pct(n, d): return f"{n/d*100:5.1f}%" if d else "  n/a"
print("\n=== 1. WHERE IS THE KNOWLEDGE (base-wrong rows) ===")
byc = collections.Counter(x["cls"] for x in rec)
for c, n in byc.most_common(): print(f"  {c:38s} {n:5d} ({pct(n,len(rec))})  fixed by {TAG}: {sum(x['fixed'] for x in rec if x['cls']==c)} ({pct(sum(x['fixed'] for x in rec if x['cls']==c), n)})")
print("  by subject (class shares):")
for s in sorted(set(x["subject"] for x in rec)):
    xs = [x for x in rec if x["subject"] == s]; cc = collections.Counter(x["cls"] for x in xs)
    print(f"    {s:26s} n={len(xs):4d}  " + "  ".join(f"{k.split()[0]}:{pct(v,len(xs))}" for k, v in cc.most_common()))
print("\n=== 2. AMONG ROWS WITH A FIXING WINDOW: what separates fixed vs not (lr 2e-5) ===")
hf = [x for x in rec if x["n_fix"] > 0]
def split(name, key):
    groups = collections.defaultdict(list)
    for x in hf: groups[key(x)].append(x)
    print(f"  {name}:"); 
    for k, xs in sorted(groups.items(), key=lambda kv: str(kv[0])): print(f"    {str(k):22s} n={len(xs):4d} fixed {pct(sum(x['fixed'] for x in xs), len(xs))}")
split("n fixing windows", lambda x: min(x["n_fix"], 4))
split("base margin bin", lambda x: "<0.5" if x["margin"] < 0.5 else "<0.9" if x["margin"] < 0.9 else ">=0.9")
split("gold option text verbatim in kit facts/factview/qa of fixing windows", lambda x: x["gold_in_kit"])
split("kit lines containing gold text", lambda x: "0" if x["n_fv_gold"] == 0 else "1-9" if x["n_fv_gold"] < 10 else "10-49" if x["n_fv_gold"] < 50 else "50+")
split("distinct letters across the 8 windows (source conflict)", lambda x: min(x["n_letters"], 4))
split("subject", lambda x: x["subject"])
split("gold is 'All of the above/Both'", lambda x: x["gold_all"])
split("gold numeric", lambda x: x["numeric_gold"])
split("release tag mismatch (question tag vs fixing window)", lambda x: x["relmis"])
print("\n=== 3. HOW THE BASE IS WRONG (option-shape patterns on wrong rows) ===")
print(f"  picked longest option {pct(sum(x['pred_longest'] for x in rec), len(rec))} (gold is longest {pct(sum(x['gold_longest'] for x in rec), len(rec))})")
print(f"  picked 'All of the above/Both' {pct(sum(x['pred_all'] for x in rec), len(rec))} (gold is {pct(sum(x['gold_all'] for x in rec), len(rec))}); picked 'None of the above' {pct(sum(x['pred_none'] for x in rec), len(rec))}")
print(f"  unparsed {sum(x['unparsed'] for x in rec)}; base low-margin (<0.5) among wrong {pct(sum(x['margin']<0.5 for x in rec), len(rec))} vs among right {pct(sum(sig.get(q,{}).get('margin',1)<0.5 for q in test if B[q]['correct']), sum(B[q]['correct'] for q in test))}")
print(f"  windows disagree (>=3 distinct letters) among wrong {pct(sum(x['n_letters']>=3 for x in rec), len(rec))}")
print("\n=== 4. SAMPLE FAILURES (still wrong after lr 2e-5, has fixing window) ===")
random.seed(3)
for x in random.sample([x for x in hf if not x["fixed"]], 8):
    r = test[x["q"]]; g = r["choices"][r["answer"]]; p = bl[x["q"]]; pc = r["choices"][ord(p)-65] if p else "?"
    fw = [w for w, pp in perq[x["q"]] if pp == gold[x["q"]]][0]; txt = wtext.get(fw, ""); i = txt.lower().find(g.lower()[:40])
    snip = txt[max(0, i-200):i+200].replace("\n", " ") if i >= 0 else "(gold text not verbatim in window) " + txt[:300].replace("\n", " ")
    print(f"\n[{x['q']}] {r['subject']} | margin {x['margin']:.2f} | fix windows {x['n_fix']} | gold-in-kit {x['gold_in_kit']} ({x['n_fv_gold']} lines)\nQ: {r['question']}\nGOLD: {g}\nBASE/TRAINED PICKED: {pc}\nWINDOW: …{snip}…")
print("\n=== 5. SAMPLE 'gold text absent from windows' rows ===")
for x in random.sample([x for x in rec if x["cls"] == "gold text absent from windows"], 5):
    r = test[x["q"]]; print(f"[{x['q']}] {r['subject']} Q: {r['question']} | GOLD: {r['choices'][r['answer']]} | picked: {r['choices'][ord(bl[x['q']])-65] if bl[x['q']] else '?'}")
json.dump(rec, open(R / "results/kit/failure_records.json", "w"))
print("\nFAILURE_ANALYSIS_DONE")
