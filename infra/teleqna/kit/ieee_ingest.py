#!/usr/bin/env python3
"""IEEE/Bluetooth PDFs -> text -> ~800-word chunks for the trace-back index (data/kit/extra_chunks/ieee.jsonl).
Drops running headers/footers and page numbers heuristically; keeps clause headings as chunk boundaries."""
import re, json, glob, os, collections
from pathlib import Path
import pymupdf
C = Path.home() / "projects/_shared/corpora/ieee"; OUT = Path.home() / "projects/teleqna/runs/teleqna-8b/data/kit/extra_chunks"; OUT.mkdir(parents=True, exist_ok=True)
CLAUSE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\s+[A-Z][^\n]{3,80}$", re.M)
def pages(pdf):
    doc = pymupdf.open(pdf); lines_all = []
    freq = collections.Counter()
    texts = [p.get_text("text") for p in doc]
    for t in texts:
        for ln in t.splitlines()[:3] + t.splitlines()[-3:]: freq[ln.strip()] += 1
    hdr = {k for k, v in freq.items() if v > len(texts) * 0.2 and k}
    for t in texts:
        ls = [ln for ln in t.splitlines() if ln.strip() and ln.strip() not in hdr and not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
        lines_all.append("\n".join(ls))
    return "\n".join(lines_all)
n = 0; stats = collections.Counter()
with open(OUT / "ieee.jsonl", "w", encoding="utf-8") as fh:
    for pdf in sorted(glob.glob(str(C / "*.pdf"))):
        name = os.path.basename(pdf); txt = pages(pdf)
        txt = re.sub(r"-\n(?=[a-z])", "", txt); txt = re.sub(r"[ \t]+", " ", txt)
        pos = [m.start() for m in CLAUSE.finditer(txt)] + [len(txt)]
        if len(pos) < 10: pos = list(range(0, len(txt), 4000)) + [len(txt)]
        buf, bw = [], 0
        for i in range(len(pos) - 1):
            seg = txt[pos[i]:pos[i+1]].strip(); w = len(seg.split())
            if not w: continue
            buf.append(seg); bw += w
            if bw >= 600:
                body = "\n".join(buf)
                for j in range(0, len(body.split()), 900):
                    piece = " ".join(body.split()[j:j+900])
                    if len(piece.split()) >= 60:
                        fh.write(json.dumps(dict(cid=f"ieee{n:07d}", src="ieee_" + re.sub(r"[^0-9a-z.]", "", name.lower())[:16], doc=name, text=piece), ensure_ascii=False) + "\n"); n += 1; stats[name] += 1
                buf, bw = [], 0
        print(name, "chars", len(txt), "chunks", stats[name], flush=True)
print("total chunks", n); print("IEEE_INGEST_DONE")
