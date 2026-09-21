#!/usr/bin/env python3
"""Data gate before training: T2 of the tier-1 kit on the 300 pilot questions, restricted to their windows
that are in the keep set. Reference = the same kept windows verbatim (what a perfect kit would give).
Writes data/eval/t1gate_{verbatim,rewrite,factview,all}.jsonl; scoring is done by the job."""
import json, glob, collections
from pathlib import Path
from transformers import AutoTokenizer
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TOK = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
def ntok(s): return len(TOK(s, add_special_tokens=False)["input_ids"])
def trunc(s, n):
    ids = TOK(s, add_special_tokens=False)["input_ids"]; return s if len(ids) <= n else TOK.decode(ids[:n])
def wrap(refs, q): return f"{HEADER}\n\n" + "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(refs)) + f"\n\n---\n\n{q}"
qs = json.load(open(R / "data/kit/pilot_sample.json"))["questions"]
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
keep = {}
for l in open(R / "data/kit/windows_keep.jsonl", encoding="utf-8"):
    w = json.loads(l); keep[w["win_id"]] = w
q2w = collections.defaultdict(list)
for w in keep.values():
    for q in w["for_q"]:
        if q in set(qs): q2w[q].append(w["win_id"])
views = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(str(R / "data/kit/tier1/views_s*.jsonl")):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if not r["ok"] or r["win_id"] not in keep: continue
        v = r["view"]
        if v == "register": views["rewrite"][r["win_id"]].append(r["text"])
        elif v == "factview": views["factview"][r["win_id"]].append(r["text"])
        elif v == "facts": views["factview"][r["win_id"]].append(json.loads(r["text"])["fact"])
        elif v == "verbatim": views["verbatim"][r["win_id"]].append(r["text"])
qs_ok = [q for q in qs if q2w[q] and all(views["verbatim"].get(w) for w in q2w[q])]
print(f"pilot questions with >=1 kept window and tier-1 views present: {len(qs_ok)}/{len(qs)}", flush=True)
sets = {"verbatim": ["verbatim"], "rewrite": ["rewrite"], "factview": ["factview"], "all": ["verbatim", "rewrite", "factview"]}
for name, vs in sets.items():
    rows = []
    for q in qs_ok:
        refs = []
        for wid in q2w[q]:
            parts = [p for v in vs for p in views[v].get(wid, [])]
            if parts: refs.append("\n".join(parts))
        per = max(400, 12000 // max(1, len(refs))); refs = [trunc(x, per) for x in refs]
        rows.append({**test[q], "question": wrap(refs, test[q]["question"]), "n_ctx": len(refs)})
    with open(R / f"data/eval/t1gate_{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> t1gate_{name}.jsonl rows {len(rows)} mean tokens {sum(ntok(r['question']) for r in rows[:60])/60:.0f}", flush=True)
json.dump(qs_ok, open(R / "data/kit/tier1/gate_questions.json", "w"))
print("T1GATE_BUILT", flush=True)
