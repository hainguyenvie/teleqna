#!/usr/bin/env python3
"""Turn pilot views into T2 eval sets: for each sampled question, context = the gated views of its 8
windows (per view type, and one combo), wrapped like otfull_rag8_strong. Reference = RAG-8 on the same
questions (already scored) and base."""
import json, collections
from pathlib import Path
from transformers import AutoTokenizer
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TOK = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
BUDGET = 12000
def ntok(s): return len(TOK(s, add_special_tokens=False)["input_ids"])
def trunc(s, n):
    ids = TOK(s, add_special_tokens=False)["input_ids"]; return s if len(ids) <= n else TOK.decode(ids[:n])
def wrap(refs, q): return f"{HEADER}\n\n" + "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(refs)) + f"\n\n---\n\n{q}"
def render(view, text):
    if view in ("textbook", "notes", "relations", "dense"): return text
    r = json.loads(text)
    if view == "facts": return r["fact"]
    if view == "qa": return f"Q: {r['q']}\nA: {r['a']}. {r['why']}"
    if view == "mcq":
        opts = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(r["options"]))
        return f"Q: {r['q']}\n{opts}\nCorrect: {chr(65+r['answer'])}) {r['options'][r['answer']]}"
sample = json.load(open(R / "data/kit/pilot_sample.json")); qs = sample["questions"]; wset = set(sample["windows"])
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
q2w = collections.defaultdict(list)
for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
    w = json.loads(l)
    if w["win_id"] in wset:
        for q in w["for_q"]: q2w[q].append(w["win_id"])
views = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(R / "data/kit/pilot_views.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["ok"]: views[r["view"]][r["win_id"]].append(render(r["view"], r["text"]))
combos = {v: [v] for v in views}; combos["combo_fqn"] = ["facts", "qa", "notes"]; combos["combo_all"] = list(views)
for name, vs in combos.items():
    rows = []
    for q in qs:
        refs = []
        for wid in q2w[q]:
            parts = [p for v in vs for p in views[v].get(wid, [])]
            if parts: refs.append("\n".join(parts))
        per = max(400, BUDGET // max(1, len(refs)))
        refs = [trunc(x, per) for x in refs]
        rows.append({**test[q], "question": wrap(refs, test[q]["question"]), "n_ctx": len(refs)})
    with open(R / f"data/eval/kit_{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> kit_{name}.jsonl rows {len(rows)} mean tokens {sum(ntok(r['question']) for r in rows[:100])/100:.0f}", flush=True)
print("KIT_EVAL_BUILT", flush=True)
