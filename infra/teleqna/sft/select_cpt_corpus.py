#!/usr/bin/env python3
"""Select the CPT (continued-pretraining) corpus, error-targeted.

The DPO diagnosis showed letter-preference training cannot inject the missing
knowledge: fixes were uncorrelated with training-pair relevance (fix-rate flat
at ~23% across similarity bins) and 164 dev rows stay wrong in every arm —
66 Research publications + 50 Standards specifications. CPT feeds the model
the SOURCE TEXT itself with a plain LM objective; this script picks which
text, following the standing policy: spend the budget where the model is
wrong, keep a breadth floor so it does not overfit the failure topics.

Selection (dev-1000 is burnable for this; held-out 9,000 untouched):
  - wrong set = union of base no-think (A/A) and base think (b0) dev errors
  - every wrong question contributes its top-5 most-similar TCC docs
  - remaining budget filled by global max-similarity ranking to ~TCC_TARGET
  - plus a deterministic breadth sample of the unselected docs
  - all 11,296 spec chunks ride along (already error-density-ranked upstream)

Pure stdlib; runs on the storage host, no GPU.
"""
import collections
import hashlib
import json
import math
import re

ROOT = "/home/tensara/projects/telelogs"
SFT = ROOT + "/runs/teleqna-sft"
SYN = ROOT + "/runs/bench4/synth/data"
TCC_TARGET = 100_000_000   # tokens, error-targeted fill
BREADTH_TARGET = 20_000_000
TOK_PER_WORD = 1.35
W = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")


def loadl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def toks(t):
    return set(W.findall(t.lower()))


dev = {r["sample_id"]: r for r in loadl(SFT + "/data/dev1000.jsonl")}
wrong = set()
d = json.load(open(SFT + "/results/dev1000_base_aa_nothink.json"))
wrong |= {r["sample_id"] for r in d["results"] if not r["correct"]}
wrong |= {r["sample_id"] for r in loadl(ROOT + "/runs/bench4/teleqna/results/b0_think/results.jsonl")
          if r["sample_id"] in dev and not r["correct"]}
qsets = {s: toks(dev[s]["question"] + " " + " ".join(dev[s]["choices"])) for s in wrong}
print(f"wrong questions (union no-think/think): {len(wrong)}")

# pass 1: doc scoring sets (title + first 20k chars is enough for relevance)
docs = []
for r in loadl(SYN + "/tcc_telecom.jsonl"):
    docs.append((r["doc_id"], toks(r.get("title", "") + " " + r["text"][:20000]),
                 int(r["word_count"])))
N = len(docs)
df = collections.Counter(t for _, tk, _ in docs for t in tk)
idf = {t: math.log(N / c) for t, c in df.items()}
inv = collections.defaultdict(list)
for i, (_, tk, _) in enumerate(docs):
    for t in tk:
        if df[t] < N * 0.15:
            inv[t].append(i)
dnorm = [math.sqrt(sum(idf[t] ** 2 for t in tk)) or 1 for _, tk, _ in docs]

best_by_doc = collections.defaultdict(float)
top5 = set()
for s, qtk in qsets.items():
    qn = math.sqrt(sum(idf.get(t, 0) ** 2 for t in qtk)) or 1
    scores = collections.defaultdict(float)
    for t in qtk:
        if t in idf and df[t] < N * 0.15:
            for i in inv[t]:
                scores[i] += idf[t] ** 2
    ranked = sorted(((v / (qn * dnorm[i]), i) for i, v in scores.items()), reverse=True)
    for sim, i in ranked[:5]:
        top5.add(i)
    for sim, i in ranked[:50]:
        best_by_doc[i] = max(best_by_doc[i], sim)

picked = set(top5)
budget = TCC_TARGET - sum(docs[i][2] * TOK_PER_WORD for i in picked)
for i, _sim in sorted(((i, s) for i, s in best_by_doc.items()), key=lambda kv: -kv[1]):
    if budget <= 0:
        break
    if i in picked:
        continue
    picked.add(i)
    budget -= docs[i][2] * TOK_PER_WORD

rest = [i for i in range(N) if i not in picked]
rest.sort(key=lambda i: hashlib.sha256(("breadth|" + docs[i][0]).encode()).hexdigest())
bbudget = BREADTH_TARGET
breadth = set()
for i in rest:
    if bbudget <= 0:
        break
    breadth.add(i)
    bbudget -= docs[i][2] * TOK_PER_WORD

sel_ids = {docs[i][0]: ("targeted" if i in picked else "breadth")
           for i in picked | breadth}
print(f"tcc docs: targeted={len(picked)} breadth={len(breadth)} "
      f"tokens~{sum(docs[i][2]*TOK_PER_WORD for i in picked|breadth)/1e6:.0f}M")

out = open(SFT + "/data/cpt_corpus.jsonl", "w", encoding="utf-8")
n_tok = 0
for r in loadl(SYN + "/tcc_telecom.jsonl"):
    tier = sel_ids.get(r["doc_id"])
    if not tier:
        continue
    text = (r.get("title", "") + "\n\n" + r["text"]).strip()
    out.write(json.dumps({"text": text, "src": "tcc", "tier": tier,
                          "doc_id": r["doc_id"]}, ensure_ascii=False) + "\n")
    n_tok += int(r["word_count"] * TOK_PER_WORD)

n_spec = 0
for r in loadl(SYN + "/chunks.jsonl"):
    head = f"3GPP TS {r['spec']} (Release {r['release']})"
    text = head + "\n\n" + r["text"].strip()
    out.write(json.dumps({"text": text, "src": "spec", "tier": r.get("tier", "?"),
                          "doc_id": r["chunk_id"]}, ensure_ascii=False) + "\n")
    n_spec += 1
    n_tok += int(len(r["text"].split()) * TOK_PER_WORD)
out.close()
print(f"cpt_corpus.jsonl written: spec_chunks={n_spec} total~{n_tok/1e6:.0f}M tokens")
