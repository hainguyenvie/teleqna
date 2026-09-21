#!/usr/bin/env python3
"""Public-API source acquisition for the functionally uncovered test questions (question text only, never the answer):
Wikipedia search (top 3) -> plain-text extracts. OpenAlex was dropped (paid budget since 2026), arXiv API rate-limits.
Resumable: skips sample_ids already in the output. Output rows: {sample_id, src, title, text}."""
import json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path
H = Path(__file__).resolve().parent.parent / "data/api_sources"; INP = H / "uncovered_functional.jsonl"; OUT = H / "apisearch.jsonl"
UA = {"User-Agent": "teleqna-source-trace/1.0 (research; contact honghainguyen2003@gmail.com)"}
def get(url):
    for t in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r: return r.read().decode("utf-8", "replace")
        except Exception as e:
            time.sleep(3 * (t + 1))
    return ""
def wiki(q):
    r = get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(dict(action="query", list="search", srsearch=q, srlimit=3, format="json")))
    if not r: return []
    res = []
    for h in json.loads(r).get("query", {}).get("search", []):
        e = get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(dict(action="query", prop="extracts", explaintext=1, titles=h["title"], format="json", exlimit=1)))
        if not e: continue
        for p in json.loads(e).get("query", {}).get("pages", {}).values():
            t = p.get("extract") or ""
            if len(t) > 300: res.append(("wikipedia", h["title"], t[:20000]))
    return res
rows = [json.loads(l) for l in open(INP, encoding="utf-8")]
done = {json.loads(l)["sample_id"] for l in open(OUT, encoding="utf-8")} if OUT.exists() else set()
print(f"{len(rows)} questions, {len(done)} already done", flush=True)
n = 0
with open(OUT, "a", encoding="utf-8") as out:
    for r in rows:
        if r["sample_id"] in done: continue
        q = re.sub(r"\s*\[[^\]]*\]\s*$", "", r["question"])
        found = wiki(q)
        if not found: out.write(json.dumps(dict(sample_id=r["sample_id"], src="api_none", title="", text=""), ensure_ascii=False) + "\n")
        for src, title, text in found:
            out.write(json.dumps(dict(sample_id=r["sample_id"], src="api_" + src, title=title, text=text), ensure_ascii=False) + "\n")
        out.flush(); n += 1
        if n % 25 == 0: print(n, "questions;", sum(1 for _ in open(OUT, encoding="utf-8")), "rows", flush=True)
        time.sleep(0.3)
print("API_SOURCES_DONE", n)
