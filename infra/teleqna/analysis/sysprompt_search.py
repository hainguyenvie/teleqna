#!/usr/bin/env python3
"""Search over SYSTEM prompts (the only prompt surface we can ship: the chat template's default system message,
since the GSMA harness owns the user message). Scores each candidate against a served checkpoint through its
OpenAI-compatible API, with the exact Inspect SINGLE_ANSWER_TEMPLATE and the exact choice() parsing.
Dev set = synthetic kit MCQs whose gold was verified by OTel-31B reading the source window (never the test key).
Test set = TeleQnA, used only for a final confirmation run of the chosen prompt."""
import json, re, argparse, random, sys, collections
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
T = ("Answer the following multiple choice question. The entire content of your response should be of the following "
     "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
ap = argparse.ArgumentParser(); ap.add_argument("--prompts", required=True, help="json file: {name: system_prompt}")
ap.add_argument("--set", default="dev", choices=["dev", "test"]); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--url", default="http://127.0.0.1:8020/v1/chat/completions"); ap.add_argument("--model", default="wise-o3")
ap.add_argument("--workers", type=int, default=64); ap.add_argument("--max-tokens", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=""); a = ap.parse_args()
rng = random.Random(a.seed)
if a.set == "test":
    rows = [json.loads(l) for l in open(R / "data/eval/otfull10000.jsonl", encoding="utf-8")]
    items = [dict(q=r["question"], choices=r["choices"], gold=r["answer"], id=r["sample_id"], subject=r.get("subject", "")) for r in rows]
else:
    import unicodedata
    def _n(x): return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", x or "").lower()).strip()
    tq = {_n(json.loads(l)["question"]) for l in open(R / "data/eval/otfull10000.jsonl", encoding="utf-8")}
    seen = set(); items = []; dropped = 0; fewshot_qs = set()
    for l in open(R / "gsma_serve/fewshot_questions.txt", encoding="utf-8") if (R / "gsma_serve/fewshot_questions.txt").exists() else []:
        fewshot_qs.add(l.rstrip("\n"))
    for l in open(R / "data/kit/onp1/rows_partial_verified.jsonl", encoding="utf-8"):
        r = json.loads(l); q = r["prompt"].split("\n\n")[1]
        if q in seen: continue
        seen.add(q)
        if _n(q) in tq or q in fewshot_qs: dropped += 1; continue
        opts = [x[3:] for x in r["prompt"].split("\n\n")[2].split("\n")]
        items.append(dict(q=q, choices=opts, gold=ord(r["completion"].strip()[-1]) - 65, id=f"dev{len(items)}", subject="synthetic-verified"))
    rng.shuffle(items); print(f"dev pool {len(items):,} (dropped {dropped} items that collide with the test set or are used as few-shot)", flush=True)
if a.n: items = items[:a.n]
def ask(sysmsg, it):
    letters = ",".join(chr(65 + i) for i in range(len(it["choices"])))
    user = T.format(letters=letters, question=it["q"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(it["choices"])))
    msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": user}]
    body = json.dumps(dict(model=a.model, messages=msgs, max_tokens=a.max_tokens, temperature=0)).encode()
    req = urllib.request.Request(a.url, data=body, headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp: return json.load(resp)["choices"][0]["message"]["content"]
        except Exception as e: err = e
    return ""
def parse(txt, n):
    m = STRICT.search(txt or "") or LOOSE.search(txt or "")
    if not m: return -1
    c = m.group(1).strip().upper()[:1]
    i = ord(c) - 65 if c.isalpha() else -1
    return i if 0 <= i < n else -1
prompts = json.load(open(a.prompts))
print(f"{a.set} set: {len(items):,} items, {len(prompts)} prompts", flush=True)
res = {}
for name, sysmsg in prompts.items():
    with ThreadPoolExecutor(max_workers=a.workers) as ex: outs = list(ex.map(lambda it: ask(sysmsg, it), items))
    ok = unp = 0; wrong = []
    for it, o in zip(items, outs):
        p = parse(o, len(it["choices"]))
        if p < 0: unp += 1
        if p == it["gold"]: ok += 1
        else: wrong.append(it["id"])
    res[name] = dict(acc=100 * ok / len(items), unparsed=unp, wrong=wrong)
    print(f"  {name:28s} acc {100*ok/len(items):6.2f}  unparsed {unp:4d}  (n={len(items)})", flush=True)
base = max(res.values(), key=lambda r: -len(r["wrong"]))
if a.out: json.dump({k: dict(acc=v["acc"], unparsed=v["unparsed"]) for k, v in res.items()}, open(R / a.out, "w"), indent=1)
best = max(res.items(), key=lambda kv: kv[1]["acc"]); print(f"best: {best[0]} {best[1]['acc']:.2f}")
print("SYSPROMPT_DONE")
