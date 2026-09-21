#!/usr/bin/env python3
"""Tokenize and pack tier 1 into (ids, mask) blocks of 1024.
  kit     verbatim / register / facts / factview / qa views, shuffled as documents so the ~10 views of a
          fact land in different contexts (Allen-Zhu: exposures in varied contexts), loss on every token
  replay  tele-data standard/wiki/arxiv prose, ~10% of kit tokens (forgetting guard, SMT/Active Reading)
  anchor  self-replay MCQ rows, loss only on the completion, upsampled to ~33% of sequences (eg_pack recipe)"""
import json, glob, random, argparse, collections, fnmatch
glob.fnmatch = fnmatch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; BASE = str(Path.home() / "projects/_shared/models/Qwen3-8B")
TD = Path.home() / "projects/_shared/corpora/tele-data"
ap = argparse.ArgumentParser(); ap.add_argument("--blk", type=int, default=1024); ap.add_argument("--replay", type=float, default=0.10)
ap.add_argument("--anchor-frac", type=float, default=0.33); ap.add_argument("--out", default="data/kit/tier1/pack")
ap.add_argument("--windows", default="", help="optional JSON list of win_ids to restrict the kit to (A/B arms)")
ap.add_argument("--views", default="data/kit/tier1/views_s*.jsonl", help="comma-separated globs of view files")
ap.add_argument("--factview-override", default="", help="glob of factview files that REPLACE the factview rows of --views (e.g. K=30)")
ap.add_argument("--anchor", default="data/kit/tier1/anchor_selfreplay.jsonl")
ap.add_argument("--chat-qa", type=float, default=0.0, help="fraction of qa docs rendered as chat turns (user: q -> assistant: answer + why), loss on assistant only")
ap.add_argument("--mcq-gold", action="store_true", help="add gated synthetic MCQs in harness chat format with the evidence-backed generator answer as target (loss on completion)")
a = ap.parse_args()
WSET = set(json.load(open(R / a.windows))) if a.windows else None
tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
rng = random.Random(20260914)
import re
STYLE_BAD = re.compile(r"(?i)\bexcerpt\b|shall be rewritten|no preamble|markdown heading|\btext above\b|according to (this|the) (excerpt|passage)|(the|this) (document|passage|text) (does not|doesn't|is silent|lacks|omits|provides no|has no)")
SENT = re.compile(r"(?<=[.!?])\s+|\n+")
scrub_stat = collections.Counter()
def scrub(s, key):
    """drop sentences that talk about 'the excerpt' or echo the generation instructions; keep the rest"""
    if not STYLE_BAD.search(s): return s
    keep = [x for x in SENT.split(s) if x.strip() and not STYLE_BAD.search(x)]
    scrub_stat[key] += 1
    return " ".join(keep).strip()
def clean(s):
    s = s.replace("~", " ").replace("\\textless", "<").replace("\\textgreater", ">").replace("\\{", "{").replace("\\}", "}")
    return re.sub(r"[ \t]{2,}", " ", s)
docs = []; st = collections.Counter(); dropped = collections.Counter(); chat_rows = []; mcq_rows = []
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be of the following "
            "format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}")
files = [f for g in a.views.split(",") for f in sorted(glob.glob(str(R / g)))]
OVR = set()
if a.factview_override:
    ofiles = sorted(glob.glob(str(R / a.factview_override))); files += ofiles
    for f in ofiles:
        for l in open(f, encoding="utf-8"):
            if '"view": "factview"' in l: OVR.add(json.loads(l)["win_id"])
    print(f"factview override covers {len(OVR):,} windows", flush=True)
print("view files:", len(files), flush=True)
for f in files:
    is_override = bool(a.factview_override) and glob.fnmatch.fnmatch(f, str(R / a.factview_override))
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if not r["ok"]: continue
        if WSET is not None and r["win_id"] not in WSET: continue
        if r["view"] == "mcq":
            if a.mcq_gold:
                m = json.loads(r["text"])
                if not re.search(r"(?i)excerpt|passage|the document|text above", m["q"] + " ".join(m["options"])):
                    n_ = len(m["options"]); ch = "\n".join(f"{chr(65+i)}) {clean(c)}" for i, c in enumerate(m["options"]))
                    prompt = TEMPLATE.format(letters=",".join(chr(65 + i) for i in range(n_)), question=clean(m["q"]), choices=ch)
                    mcq_rows.append((prompt, f"ANSWER: {chr(65 + m['answer'])}")); st["mcq_gold"] += 1
            continue
        v = r["view"]
        if v == "factview" and not is_override and r["win_id"] in OVR: continue   # replaced by the K=30 files for this window
        if v in ("verbatim", "register", "factview"): t = r["text"]
        elif v == "facts": t = json.loads(r["text"])["fact"]
        elif v == "qa":
            q = json.loads(r["text"])
            if a.chat_qa > 0 and rng.random() < a.chat_qa and not STYLE_BAD.search(q["q"] + q["why"]):
                chat_rows.append((clean(q["q"]), clean(f"{q['a']}. {q['why']}"))); st["qa_chat"] += 1; continue
            t = f"Q: {q['q']}\nA: {q['a']}. {q['why']}"
        else: continue
        if v != "verbatim":
            t = scrub(t, (v, r.get("register", "")))
            if len(t) < (200 if v == "register" else 20): dropped[v] += 1; continue
        docs.append(clean(t)); st[v] += 1
print(f"docs scrubbed (sentences removed) by view/register: {dict(scrub_stat)}", flush=True)
print(f"docs dropped after scrub (too short): {dict(dropped)}", flush=True)
rng.shuffle(docs)
ids, mask = [], []
for t in docs:
    e = tok(t + "\n\n", add_special_tokens=False)["input_ids"]; ids.extend(e); mask.extend([1] * len(e))
n_kit = len(ids); print(f"kit docs {dict(st)}  tokens {n_kit:,}", flush=True)
# replay
want = int(n_kit * a.replay); got = 0; files = [TD / "standard/standard.jsonl", TD / "wiki/wiki.jsonl", TD / "arxiv/arxiv.jsonl"]
pools = []
for f in files:
    with open(f, encoding="utf-8") as fh:
        for i, l in enumerate(fh):
            if rng.random() < (1.0 if "standard" in str(f) else 0.25): pools.append(json.loads(l)["content"])
            if len(pools) > 400000: break
rng.shuffle(pools)
for t in pools:
    if got >= want: break
    e = tok(t[:12000] + "\n\n", add_special_tokens=False)["input_ids"][:3000]; ids.extend(e); mask.extend([1] * len(e)); got += len(e)
print(f"replay tokens {got:,} ({got/max(n_kit,1)*100:.1f}% of kit)", flush=True)
n_docs = len(docs)
# anchor
anchor_all = [json.loads(l) for g in a.anchor.split(",") for f in sorted(glob.glob(str(R / g))) for l in open(f, encoding="utf-8")]
# Keep only anchors where the base's answer agrees with the evidence-backed generator answer: the anchor must
# pin the ANSWER format, not re-teach the base's wrong beliefs about the very windows the kit is teaching.
anchor = [r for r in anchor_all if r.get("agree", True)]
print(f"anchors: {len(anchor_all):,} rows, kept agree-only {len(anchor):,} ({len(anchor)/max(1,len(anchor_all))*100:.1f}%)", flush=True)
a_ids, a_mask = [], []
for r in anchor:
    p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; ci = tok(r["completion"], add_special_tokens=False)["input_ids"]
    a_ids.extend(pi + ci); a_mask.extend([0] * len(pi) + [1] * len(ci))
c_ids, c_mask = [], []
for u, asst in chat_rows + mcq_rows:
    p = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; ci = tok(asst, add_special_tokens=False)["input_ids"]
    c_ids.extend(pi + ci); c_mask.extend([0] * len(pi) + [1] * len(ci))
print(f"chat-format rows: qa {len(chat_rows):,} + mcq-gold {len(mcq_rows):,} = {len(c_ids):,} tokens, loss-bearing {sum(c_mask):,}", flush=True)
ids.extend(c_ids); mask.extend(c_mask)
reps = int(min(12, max(1, round(a.anchor_frac * n_docs / max(len(anchor), 1)))))
print(f"anchor {len(anchor):,} rows x{reps} = {len(anchor)*reps:,} rows ({len(anchor)*reps/(n_docs+len(anchor)*reps)*100:.1f}% of sequences); "
      f"loss-bearing tokens {sum(a_mask)*reps:,}", flush=True)
ids.extend(a_ids * reps); mask.extend(a_mask * reps)
nblk = len(ids) // a.blk
I = np.array(ids[:nblk * a.blk], dtype=np.int32).reshape(nblk, a.blk); M = np.array(mask[:nblk * a.blk], dtype=np.int8).reshape(nblk, a.blk)
perm = np.random.default_rng(20260914).permutation(nblk)
np.save(R / f"{a.out}_ids.npy", I[perm]); np.save(R / f"{a.out}_mask.npy", M[perm])
print(f"{nblk:,} blocks of {a.blk} ({nblk*a.blk/1e6:.0f}M tokens) -> {a.out}_ids.npy / _mask.npy", flush=True)
print("PACK_DONE", flush=True)
