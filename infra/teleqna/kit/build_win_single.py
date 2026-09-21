#!/usr/bin/env python3
"""One eval row per (question, window): the window alone in context. Label-free usefulness of a
window = the predicted letter differs from the closed-book letter. Gold is only used afterwards, in
reporting, never in selection."""
import json, collections
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
HEADER = "Reference material retrieved from the telecom literature. It may or may not contain the answer."
test = {r["sample_id"]: r for r in map(json.loads, open(R / "data/eval/otfull10000.jsonl", encoding="utf-8"))}
n = 0
with open(R / "data/eval/win_single.jsonl", "w", encoding="utf-8") as fh:
    for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
        w = json.loads(l)
        for q in w["for_q"]:
            t = test[q]
            fh.write(json.dumps({**t, "sample_id": f"{q}|{w['win_id']}",
                                 "question": f"{HEADER}\n\n[1] {w['text']}\n\n---\n\n{t['question']}", "n_ctx": 1},
                                ensure_ascii=False) + "\n"); n += 1
print("rows", n, flush=True)
