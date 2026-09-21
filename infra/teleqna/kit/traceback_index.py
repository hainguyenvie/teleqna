#!/usr/bin/env python3
"""Trace test questions back to public sources: chunk GSMA/3GPP (all releases), Telco-Common-Corpus
(IEEE-Access / OpenAlex / RFC collections), and tele-data; BM25 (bm25s) over ~800-word chunks; retrieve
top-k per question (question + options, never the answer index); write new windows and a coverage report
(gold used only to report: is the gold option text verbatim in some retrieved chunk).
Outputs: data/kit/tb_chunks.jsonl (chunk store), data/kit/tb_windows.jsonl (win_id, text, for_q, src),
results/kit/traceback_coverage.json"""
import json, re, glob, os, argparse, collections, unicodedata
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; C = Path.home() / "projects/_shared/corpora"
ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=16); ap.add_argument("--max-words", type=int, default=800)
ap.add_argument("--skip-build", action="store_true"); a = ap.parse_args()
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()
HEAD = re.compile(r"^#{1,4}\s+\S", re.M)
def chunks(text, mw):
    pos = [m.start() for m in HEAD.finditer(text)] + [len(text)]
    if len(pos) < 3: pos = [0, len(text)]
    buf, bw = [], 0
    for i in range(len(pos) - 1):
        seg = text[pos[i]:pos[i+1]].strip(); 
        if not seg: continue
        w = seg.split()
        for j in range(0, len(w), mw):            # long sections: hard split
            piece = " ".join(w[j:j+mw]); buf.append(piece); bw += len(w[j:j+mw])
            if bw >= mw * 0.7: yield "\n".join(buf); buf, bw = [], 0
    if buf: yield "\n".join(buf)
store = R / "data/kit/tb_chunks.jsonl"
if not a.skip_build or not store.exists():
    n = 0; cnt = collections.Counter()
    with open(store, "w", encoding="utf-8") as fh:
        def emit(src, doc, text):
            global n
            for c in chunks(text, a.max_words):
                if len(c.split()) < 60: continue
                fh.write(json.dumps(dict(cid=f"c{n:08d}", src=src, doc=doc, text=c), ensure_ascii=False) + "\n"); n += 1; cnt[src] += 1
        # 3GPP: one raw.md per spec per release (prefer 'marked' tree; skip 'original' duplicates)
        for f in sorted(glob.glob(str(C / "gsma-3gpp/marked/*/*/*/raw.md"))):
            try: emit("3gpp", os.path.relpath(f, C / "gsma-3gpp"), open(f, encoding="utf-8", errors="ignore").read())
            except Exception: pass
        print("3gpp chunks", cnt["3gpp"], flush=True)
        # TCC parquet shards
        try:
            import pyarrow.parquet as pq
            for f in sorted(glob.glob(str(C / "telco-common-corpus/**/*.parquet"), recursive=True)):
                t = pq.read_table(f); cols = t.column_names
                tcol = next((c for c in ("text", "content", "body") if c in cols), None); coll = next((c for c in ("collection", "source", "subset") if c in cols), None)
                if not tcol: continue
                rows = t.to_pylist()
                for r in rows:
                    src = "tcc_" + str(r.get(coll, "na")).lower()[:20] if coll else "tcc"
                    if "patent" in src: continue
                    emit(src, str(r.get("doc_id", r.get("id", os.path.basename(f)))), r[tcol] or "")
                print(os.path.basename(f), dict(cnt), flush=True)
        except Exception as e: print("TCC skipped:", e, flush=True)
        for f in sorted(glob.glob(str(R / "data/kit/extra_chunks/*.jsonl"))):      # IEEE / Bluetooth PDFs etc.
            for l in open(f, encoding="utf-8"):
                r = json.loads(l); fh.write(json.dumps(dict(cid=f"c{n:08d}", src=r["src"], doc=r["doc"], text=r["text"]), ensure_ascii=False) + "\n"); n += 1; cnt[r["src"]] += 1
        for sub in ("standard", "arxiv", "wiki"):
            for l in open(C / f"tele-data/{sub}/{sub}.jsonl", encoding="utf-8"):
                r = json.loads(l); emit("teledata_" + sub, r["id"], r["content"])
        print("chunks total", n, dict(cnt), flush=True)
import bm25s
docs = []; meta = []
for l in open(store, encoding="utf-8"):
    r = json.loads(l); docs.append(r["text"]); meta.append((r["cid"], r["src"], r["doc"]))
print(f"indexing {len(docs):,} chunks", flush=True)
tokens = bm25s.tokenize(docs, stopwords="en"); retr = bm25s.BM25(); retr.index(tokens)
test = [json.loads(l) for l in open(R / "data/eval/otfull10000.jsonl", encoding="utf-8")]
qs = [r["question"] + " " + " ".join(r["choices"]) for r in test]
res, scores = retr.retrieve(bm25s.tokenize(qs, stopwords="en"), k=a.k)
cov = collections.Counter(); bysrc = collections.Counter(); win_for = collections.defaultdict(list)
json.dump({r["sample_id"]: [int(i) for i in ids] for r, ids in zip(test, res)}, open(R / "data/kit/tb_retrieval.json", "w"))   # persist positions
with open(R / "data/kit/tb_windows.jsonl", "w", encoding="utf-8") as fh:
    for r, ids in zip(test, res):
        g = norm(r["choices"][r["answer"]]); hit = False
        for i in ids:
            i = int(i); cid, src, doc = meta[i]
            if norm(docs[i]).find(g) >= 0 and len(g) >= 4: hit = True; bysrc[src] += 1
            win_for[i].append(r["sample_id"])
    for i, qlist in win_for.items():
        fh.write(json.dumps(dict(win_id="t" + meta[i][0], text=docs[i], for_q=sorted(set(qlist)), src=meta[i][1], doc=meta[i][2]), ensure_ascii=False) + "\n")
    for r, ids in zip(test, res): pass
print(f"gold text verbatim in top-{a.k} chunks: {cov['gold_in_topk']}/{len(test)}  by source of hits: {bysrc.most_common(8)}")
json.dump(dict(k=a.k, gold_in_topk=cov["gold_in_topk"], by_source=bysrc, n_windows=len(win_for)), open(R / "results/kit/traceback_coverage.json", "w"), indent=1)
print("TRACEBACK_DONE", flush=True)
