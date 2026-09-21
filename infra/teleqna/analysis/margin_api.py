#!/usr/bin/env python3
"""Decision-boundary vs knowledge, measured through the served model's logprobs (no extra GPU):
for every test question ask for the answer with logprobs, read the probability mass on each option letter,
and split the wrong answers into near-tie (boundary) vs confident-wrong (knowledge). Gold only scores."""
import json, re, argparse, statistics as st, collections
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
T = ("Answer the following multiple choice question. The entire content of your response should be of the following "
     "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8020/v1/chat/completions")
ap.add_argument("--model", default="wise-o3"); ap.add_argument("--n", type=int, default=10000); ap.add_argument("--workers", type=int, default=64)
ap.add_argument("--out", default="results/kit/margin_wise_o3.json"); a = ap.parse_args()
rows = [json.loads(l) for l in open(R / "data/eval/otfull10000.jsonl", encoding="utf-8")][:a.n]
def ask(r):
    letters = ",".join(chr(65 + i) for i in range(len(r["choices"])))
    user = T.format(letters=letters, question=r["question"], choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(r["choices"])))
    body = json.dumps(dict(model=a.model, messages=[{"role": "user", "content": user}], max_tokens=6, temperature=0, logprobs=True, top_logprobs=20)).encode()
    req = urllib.request.Request(a.url, data=body, headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp: d = json.load(resp)
            ch = d["choices"][0]; content = ch["message"]["content"]; lp = ch.get("logprobs", {}).get("content", [])
            n = len(r["choices"]); probs = [0.0] * n
            import math
            for tokinfo in lp:
                tk = (tokinfo.get("token") or "").strip().upper()
                if len(tk) == 1 and "A" <= tk <= chr(64 + n):      # this is the answer-letter position
                    for x in tokinfo.get("top_logprobs", []):
                        c = (x.get("token") or "").strip().upper()
                        if len(c) == 1 and "A" <= c <= chr(64 + n): probs[ord(c) - 65] = max(probs[ord(c) - 65], math.exp(x["logprob"]))
                    break
            s = sum(probs) or 1.0; probs = [p / s for p in probs]
            return content, probs
        except Exception: pass
    return "", [0.0] * len(r["choices"])
with ThreadPoolExecutor(max_workers=a.workers) as ex: outs = list(ex.map(ask, rows))
STRICT = re.compile(r"(?i)ANSWER\s*:\s*([A-E])")
res = []; 
for r, (txt, probs) in zip(rows, outs):
    m = STRICT.search(txt or ""); pred = ord(m.group(1).upper()) - 65 if m else -1
    g = r["answer"]; order = sorted(range(len(probs)), key=lambda i: -probs[i])
    margin = probs[order[0]] - probs[order[1]] if len(order) > 1 else 1.0
    res.append(dict(id=r["sample_id"], subject=r.get("subject", ""), correct=pred == g, pred=pred, gold=g, pgold=probs[g], margin=margin, rank_gold=order.index(g) + 1))
W = [r for r in res if not r["correct"]]; C = [r for r in res if r["correct"]]
print(f"acc {100*len(C)/len(res):.2f}  wrong {len(W)}")
for name, S in (("wrong", W), ("right", C)):
    m = [r["margin"] for r in S]; pg = [r["pgold"] for r in S]
    print(f"  {name:6s} n={len(S):5d} margin median {st.median(m):.2f} | near-tie(<0.2) {100*sum(x<0.2 for x in m)/len(S):5.1f}% | confident(>0.8) {100*sum(x>0.8 for x in m)/len(S):5.1f}% | P(gold) median {st.median(pg):.2f} | gold 2nd {100*sum(r['rank_gold']==2 for r in S)/len(S):5.1f}%")
print("  potential of a perfect tie-breaker among wrong:")
for t in (0.05, 0.1, 0.2, 0.3):
    k = [r for r in W if r["margin"] < t]; g2 = sum(1 for r in k if r["rank_gold"] == 2)
    print(f"    margin<{t}: {len(k):4d} wrong ({100*len(k)/len(res):.2f} pts), gold is 2nd in {g2} of them -> max +{100*g2/len(res):.2f} pts")
json.dump(res, open(R / a.out, "w")); print("MARGIN_DONE")
