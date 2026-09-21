#!/usr/bin/env python3
"""Build the T1/T2 ceiling sets for run-3's TARGET questions (PLAN_CLOSED_BOOK_8B §5.2).

T1  raw serving section(s) in context      -> "does the document contain the knowledge"
T2  top kit passages of those sections     -> "did the generator keep it"

Both are wrapped exactly like data/eval/otfull_rag8_strong.jsonl (header, [k] refs,
'---', question) so the numbers sit on the same scale as the strong-RAG ceiling.
Ranking of kit passages uses the question text only (never options or answer).
"""
import json, re, math, collections
from pathlib import Path
from transformers import AutoTokenizer

R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TOK = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
BUDGET = 6000          # tokens of reference material per arm
MAXSEC = 2             # serving sections per question (T1)
STOP = set("the a an of in on for to and or is are be by with as at from that this which what how does do it its".split())
WORD = re.compile(r"[a-z0-9][a-z0-9.\-_]+")

def words(s): return [w for w in WORD.findall(s.lower()) if w not in STOP and len(w) > 2]
def ntok(s): return len(TOK(s, add_special_tokens=False)["input_ids"])
def trunc(s, n):
    ids = TOK(s, add_special_tokens=False)["input_ids"]
    return s if len(ids) <= n else TOK.decode(ids[:n])
def wrap(refs, q):
    body = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(refs))
    return f"{HEADER}\n\n{body}\n\n---\n\n{q}"

test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
target = json.load(open(R / "data/eg3/groups.json"))["target"]
secs = [json.loads(l) for l in open(R / "data/eg3/sections_sel.jsonl", encoding="utf-8")]
q2sec = collections.defaultdict(list)
for s in secs:
    for q in s["qs"]: q2sec[q].append(s)
kit = collections.defaultdict(list)
for l in open(R / "data/eg3/synth_clean.jsonl", encoding="utf-8"):
    r = json.loads(l); kit[r["sec_id"]].append(r["text"])
print(f"target {len(target)}  sections {len(secs)}  kit passages {sum(map(len, kit.values())):,}", flush=True)

t1, t2, stats = [], [], collections.Counter()
for q in target:
    row = test[q]; ss = q2sec[q]
    if not ss: stats["no_section"] += 1; continue
    # T1: prefer the sections that serve the fewest questions (most specific), up to MAXSEC
    ss = sorted(ss, key=lambda s: len(s["qs"]))[:MAXSEC]
    per = BUDGET // len(ss)
    refs1 = [trunc(s["text"], per) for s in ss]
    # T2: rank kit passages of the same sections by idf-weighted overlap with the question
    pool = [p for s in ss for p in kit[s["sec_id"]]]
    df = collections.Counter(w for p in pool for w in set(words(p)))
    qw = set(words(row["question"]))
    def score(p):
        pw = set(words(p))
        return sum(math.log(1 + len(pool) / (1 + df[w])) for w in qw & pw) / math.sqrt(len(pw) + 1)
    ranked = sorted(pool, key=score, reverse=True)
    refs2, used = [], 0
    for p in ranked:
        n = ntok(p)
        if used + n > BUDGET: continue
        refs2.append(p); used += n
        if used > BUDGET - 80: break
    stats["t2_passages"] += len(refs2)
    for arm, refs in (("t1", refs1), ("t2", refs2)):
        (t1 if arm == "t1" else t2).append({**row, "question": wrap(refs, row["question"]), "n_ctx": len(refs)})
for name, rows in (("ceil_t1_raw", t1), ("ceil_t2_kit", t2)):
    with open(R / f"data/eval/{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> data/eval/{name}.jsonl  rows {len(rows)}  mean prompt tokens "
          f"{sum(ntok(r['question']) for r in rows[:200]) / min(200, len(rows)):.0f} (first 200)", flush=True)
print("stats", dict(stats), "mean T2 passages/q", stats["t2_passages"] / max(len(t2), 1))
