#!/usr/bin/env python3
"""Per-question source coverage after trace-back: gold option text verbatim in (a) strong-RAG windows, (b) new
trace-back top-k windows, (c) either. Reports by tag / subject and separately for base-wrong rows; writes the
uncovered list (data/kit/uncovered.jsonl) for the next acquisition round. Gold used for measurement only."""
import json, re, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
def L(p):
    d = json.load(open(p)); d = d["results"] if isinstance(d, dict) and "results" in d else d
    if isinstance(d, dict): d = list(d.values())
    return {x["sample_id"]: x for x in d if isinstance(x, dict) and "sample_id" in x}
B = L(R / "results/landscape/otfull_q3_8b_base_nothink512.json")
def tag(q):
    m = re.search(r"\[([^\]]+)\]\s*$", test[q]["question"])
    if not m: return "untagged"
    s = m.group(1); return "3GPP" if "3GPP" in s else s
gold = {q: norm(r["choices"][r["answer"]]) for q, r in test.items()}
cov_old = collections.defaultdict(bool); cov_new = collections.defaultdict(bool); src_new = collections.defaultdict(set)
for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
    w = json.loads(l); t = norm(w["text"])
    for q in w["for_q"]:
        if len(gold[q]) >= 4 and gold[q] in t: cov_old[q] = True
for l in open(R / "data/kit/tb_windows.jsonl", encoding="utf-8"):
    w = json.loads(l); t = norm(w["text"])
    for q in w["for_q"]:
        if len(gold[q]) >= 4 and gold[q] in t: cov_new[q] = True; src_new[q].add(w["src"])
ids = list(test); either = {q: cov_old[q] or cov_new[q] for q in ids}
def row(name, I):
    n = len(I); return f"  {name:28s} n={n:5d}  strongRAG {sum(cov_old[q] for q in I)/n*100:5.1f}%  traceback {sum(cov_new[q] for q in I)/n*100:5.1f}%  either {sum(either[q] for q in I)/n*100:5.1f}%"
print("gold option text verbatim in retrieved windows (lower bound of coverage: paraphrased answers do not count)")
print(row("ALL", ids))
for t in sorted(set(map(tag, ids))): print(row("tag " + t, [q for q in ids if tag(q) == t]))
for s in sorted(set(test[q]["subject"] for q in ids)): print(row("subj " + s, [q for q in ids if test[q]["subject"] == s]))
bw = [q for q in ids if not B[q]["correct"]]; print(row("BASE-WRONG", bw))
for t in sorted(set(map(tag, bw))): print(row("  base-wrong tag " + t, [q for q in bw if tag(q) == t]))
print("new-coverage sources (base-wrong rows newly covered):", collections.Counter(s for q in bw if cov_new[q] and not cov_old[q] for s in src_new[q]).most_common(10))
unc = [q for q in ids if not either[q]]
with open(R / "data/kit/uncovered.jsonl", "w", encoding="utf-8") as fh:
    for q in unc: fh.write(json.dumps(dict(sample_id=q, tag=tag(q), subject=test[q]["subject"], base_correct=bool(B[q]["correct"]), question=test[q]["question"], gold=test[q]["choices"][test[q]["answer"]]), ensure_ascii=False) + "\n")
print(f"uncovered (either): {len(unc)} -> data/kit/uncovered.jsonl; of which base-wrong {sum(not B[q]['correct'] for q in unc)}")
print("COVERAGE_REPORT_DONE")
