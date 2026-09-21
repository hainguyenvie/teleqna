#!/usr/bin/env python3
"""Second pass over the pilot views: faithfulness-gate the prose views (atoms >= 70% present in the
source window, as run 3 did), and re-measure with a 24k budget so truncation is separated from loss."""
import json, re, collections, unicodedata
from pathlib import Path
from transformers import AutoTokenizer
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
TOK = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B"), local_files_only=True)
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
ATOM = re.compile(r"\b(?:TS\s?\d+\.\d+|TR\s?\d+\.\d+|Rel-?\d+|\d+(?:\.\d+)?\s?(?:ms|s|dB|dBm|GHz|MHz|kHz|Mbps|Gbps|bit|bits|bytes|%)|[A-Z]{3,7}|\d{2,4})\b")
STOP = {"the","and","for","this","that","with","not","are","was","its","can","may","shall","from"}
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
def ntok(s): return len(TOK(s, add_special_tokens=False)["input_ids"])
def trunc(s, n):
    ids = TOK(s, add_special_tokens=False)["input_ids"]; return s if len(ids) <= n else TOK.decode(ids[:n])
def wrap(refs, q): return f"{HEADER}\n\n" + "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(refs)) + f"\n\n---\n\n{q}"
def faithful(text, srcn):
    atoms = {x for x in ATOM.findall(text) if x.lower() not in STOP and len(x) >= 2}
    if not atoms: return True, 1.0
    hit = sum(1 for x in atoms if norm(x) in srcn) / len(atoms); return hit >= 0.70, hit
def render(view, text):
    if view in ("textbook", "notes", "relations"): return text
    r = json.loads(text)
    if view == "facts": return r["fact"]
    if view == "qa": return f"Q: {r['q']}\nA: {r['a']}. {r['why']}"
    if view == "mcq":
        opts = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(r["options"]))
        return f"Q: {r['q']}\n{opts}\nCorrect: {chr(65+r['answer'])}) {r['options'][r['answer']]}"
sample = json.load(open(R / "data/kit/pilot_sample.json")); qs = sample["questions"]; wset = set(sample["windows"])
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
q2w = collections.defaultdict(list); wtext = {}
for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
    w = json.loads(l)
    if w["win_id"] in wset:
        wtext[w["win_id"]] = norm(w["text"])
        for q in w["for_q"]: q2w[q].append(w["win_id"])
views = collections.defaultdict(lambda: collections.defaultdict(list)); gate = collections.Counter(); hits = collections.defaultdict(list)
for l in open(R / "data/kit/pilot_views.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not r["ok"]: continue
    txt = render(r["view"], r["text"])
    if r["view"] in ("textbook", "notes", "relations", "facts"):
        ok, h = faithful(txt, wtext[r["win_id"]]); hits[r["view"]].append(h)
        if not ok: gate[r["view"], "drop"] += 1; continue
        gate[r["view"], "keep"] += 1
    views[r["view"]][r["win_id"]].append(txt)
for v in hits:
    h = sorted(hits[v]); print(f"faithfulness {v:10s} p10/p50 atom-hit {h[len(h)//10]:.2f}/{h[len(h)//2]:.2f}  keep {gate[v,'keep']} drop {gate[v,'drop']}")
sets = {"g_notes": (["notes"], 12000), "g_textbook": (["textbook"], 12000), "g_notes_tb": (["notes", "textbook"], 24000),
        "g_fqn24k": (["facts", "qa", "notes"], 24000), "g_all24k": (list(views), 24000), "g_notes24k": (["notes"], 24000)}
for name, (vs, budget) in sets.items():
    rows = []
    for q in qs:
        refs = []
        for wid in q2w[q]:
            parts = [p for v in vs for p in views[v].get(wid, [])]
            if parts: refs.append("\n".join(parts))
        per = max(400, budget // max(1, len(refs))); refs = [trunc(x, per) for x in refs]
        rows.append({**test[q], "question": wrap(refs, test[q]["question"]), "n_ctx": len(refs)})
    with open(R / f"data/eval/kit2_{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> kit2_{name}.jsonl mean tokens {sum(ntok(r['question']) for r in rows[:100])/100:.0f}", flush=True)
print("KIT2_BUILT")
