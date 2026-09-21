#!/usr/bin/env python3
"""Rewrite the trace-back chunk store: drop noisy sources (patents, IETF mailing list), append extra chunks (IEEE)."""
import json, glob
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
DROP = ("tcc_uspto", "tcc_epo", "tcc_ietf-mail-daily")
src = R / "data/kit/tb_chunks.jsonl"; tmp = R / "data/kit/tb_chunks.v2.jsonl"; n = kept = 0
with open(tmp, "w", encoding="utf-8") as fh:
    for l in open(src, encoding="utf-8"):
        n += 1
        if any(f'"src": "{d}' in l for d in DROP): continue
        fh.write(l); kept += 1
    have = {json.loads(l)["doc"] for l in open(tmp, encoding="utf-8") if '"src": "ieee' in l} if False else set()
    for f in sorted(glob.glob(str(R / "data/kit/extra_chunks/*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); fh.write(json.dumps(dict(cid=f"c{kept:08d}", src=r["src"], doc=r["doc"], text=r["text"]), ensure_ascii=False) + "\n"); kept += 1
tmp.replace(src); print(f"store: {n:,} -> {kept:,} chunks (dropped patents/mail, added extra)"); print("STORE_FILTER_DONE")
