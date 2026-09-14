#!/usr/bin/env python3
"""Turn the retrieved context back into a chunk file the generator can read.

retrieve_ctx.py emits an eval set with the windows inlined in the prompt; this
recovers them as first-class chunks, deduplicated, each carrying the list of
test rows that retrieved it. That list is the targeting signal: a window pulled
by eleven questions is eleven times more worth generating from than one pulled
by a single question, and windows pulled only by rows the model already answers
correctly can be dropped entirely.

Output schema matches make_teledata_chunks.py so gen_grounded_qa.py,
validate_grounded_qa.py and expand_views.py all read it unchanged.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

BLOCK = re.compile(r"^\[(\d+)\]\s(.*)$", re.MULTILINE)
SEP = "\n\n---\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", required=True, help="otfull_ragK.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--wrong-only", help="eval json; keep only windows pulled "
                                         "by at least one row it gets wrong")
    a = ap.parse_args()

    wrong = None
    if a.wrong_only:
        res = json.load(open(a.wrong_only))["results"]
        wrong = {r["sample_id"] for r in res if not r["correct"]}
        print(f"targeting {len(wrong)} wrong rows")

    chunks: dict[str, dict] = {}
    for line in open(a.rag, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("n_ctx"):
            continue
        head = r["question"].split(SEP)[0]
        for _, body in BLOCK.findall(head):
            body = body.strip()
            if len(body.split()) < 60:
                continue
            cid = "w-" + hashlib.sha256(body.encode()).hexdigest()[:16]
            c = chunks.get(cid)
            if c is None:
                c = chunks[cid] = {"chunk_id": cid, "text": body,
                                   "spec": "?", "series": None, "release": None,
                                   "headings": [], "source": "retrieved",
                                   "pulled_by": []}
            c["pulled_by"].append(r["sample_id"])

    # Provenance, recovered from the text itself: 3GPP windows name their spec,
    # arxiv windows do not. The generator branches its prompt on this, so a
    # paper must not be handed the "specification excerpt" wording.
    TS = re.compile(r"\bTS\s*(\d{2}\.\d{3})|\b3GPP\b")
    for c in chunks.values():
        m = TS.search(c["text"][:600])
        if m and m.group(1):
            c["spec"] = m.group(1)
            c["series"] = m.group(1).split(".")[0]
            c["kind"] = "spec"
        else:
            c["kind"] = "spec" if m else "paper"

    rows = list(chunks.values())
    if wrong is not None:
        rows = [c for c in rows if any(s in wrong for s in c["pulled_by"])]

    rows.sort(key=lambda c: (-len(set(c["pulled_by"])), c["chunk_id"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for c in rows:
            c["pulled_by"] = sorted(set(c["pulled_by"]))
            c["n_pulls"] = len(c["pulled_by"])
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    pulls = collections.Counter(c["n_pulls"] for c in rows)
    rep = {
        "rag": a.rag, "wrong_only": a.wrong_only,
        "unique_windows": len(chunks), "written": len(rows),
        "kind": dict(collections.Counter(c["kind"] for c in rows)),
        "words_total": sum(len(c["text"].split()) for c in rows),
        "pull_histogram": {str(k): pulls[k] for k in sorted(pulls)[:10]},
        "windows_pulled_once": pulls[1],
        "max_pulls": max((c["n_pulls"] for c in rows), default=0),
    }
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
