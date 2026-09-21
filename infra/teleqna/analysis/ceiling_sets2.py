#!/usr/bin/env python3
"""Second ceiling ladder on run-3 TARGET, separating "document selection" from "generator loss".

t1b_full   the single most specific serving section, untruncated (cap 8k tokens)
t1c_wins   only the question's strong-RAG windows that were LOCATED inside a run-3 section
t2b_kit12k kit passages of the serving sections, overlap-ranked, 12k-token budget (= RAG budget)
"""
import json, re, math, collections, unicodedata
from pathlib import Path
from transformers import AutoTokenizer

R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TOK = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
STOP = set("the a an of in on for to and or is are be by with as at from that this which what how does do it its".split())
WORD = re.compile(r"[a-z0-9][a-z0-9.\-_]+")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
def words(s): return [w for w in WORD.findall(s.lower()) if w not in STOP and len(w) > 2]
def ntok(s): return len(TOK(s, add_special_tokens=False)["input_ids"])
def trunc(s, n):
    ids = TOK(s, add_special_tokens=False)["input_ids"]
    return s if len(ids) <= n else TOK.decode(ids[:n])
def wrap(refs, q): return f"{HEADER}\n\n" + "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(refs)) + f"\n\n---\n\n{q}"

test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
target = json.load(open(R / "data/eg3/groups.json"))["target"]
secs = [json.loads(l) for l in open(R / "data/eg3/sections_sel.jsonl", encoding="utf-8")]
q2sec = collections.defaultdict(list)
for s in secs:
    s["norm"] = norm(s["text"])
    for q in s["qs"]: q2sec[q].append(s)
kit = collections.defaultdict(list)
for l in open(R / "data/eg3/synth_clean.jsonl", encoding="utf-8"):
    r = json.loads(l); kit[r["sec_id"]].append(r["text"])
q2win = collections.defaultdict(list)
for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
    w = json.loads(l)
    for q in w["for_q"]:
        if q in q2sec: q2win[q].append(w["text"])
print(f"target {len(target)}  windows for target qs {sum(map(len, q2win.values())):,}", flush=True)

out = {"t1b_full": [], "t1c_wins": [], "t2b_kit12k": []}; st = collections.Counter()
for q in target:
    row = test[q]; ss = sorted(q2sec[q], key=lambda s: len(s["qs"]))
    if not ss: st["no_section"] += 1; continue
    # t1b: most specific section, untruncated
    out["t1b_full"].append({**row, "question": wrap([trunc(ss[0]["text"], 8000)], row["question"]), "n_ctx": 1})
    # t1c: located windows = the question's strong-RAG windows whose 12-word prefix sits in ANY selected section
    allnorm = [s["norm"] for s in secs]
    located = []
    for wt in q2win[q]:
        t = norm(wt).split()
        if len(t) < 12: continue
        keys = [" ".join(t[0:12]), " ".join(t[max(0, len(t)//2-6):max(0, len(t)//2-6)+12])]
        if any(k in sn for k in keys for sn in (s["norm"] for s in q2sec[q])) or any(k in sn for k in keys for sn in allnorm):
            located.append(wt)
    st["located_wins"] += len(located)
    if located:
        out["t1c_wins"].append({**row, "question": wrap(located[:8], row["question"]), "n_ctx": len(located[:8])})
    else: st["no_located_win"] += 1
    # t2b: kit, 12k budget
    pool = [p for s in ss[:2] for p in kit[s["sec_id"]]]
    df = collections.Counter(w for p in pool for w in set(words(p))); qw = set(words(row["question"]))
    def score(p):
        pw = set(words(p)); return sum(math.log(1 + len(pool) / (1 + df[w])) for w in qw & pw) / math.sqrt(len(pw) + 1)
    refs, used = [], 0
    for p in sorted(pool, key=score, reverse=True):
        n = ntok(p)
        if used + n > 12000: continue
        refs.append(p); used += n
        if used > 11900: break
    out["t2b_kit12k"].append({**row, "question": wrap(refs, row["question"]), "n_ctx": len(refs)})
for name, rows in out.items():
    with open(R / f"data/eval/ceil_{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> ceil_{name}.jsonl rows {len(rows)} mean ctx refs {sum(r['n_ctx'] for r in rows)/max(len(rows),1):.1f}", flush=True)
print("stats", dict(st), flush=True)
