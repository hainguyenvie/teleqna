#!/usr/bin/env python3
"""Early-signal probe (after Chang et al. 2024, arXiv:2406.11813): log-probability of a fact's answer span under
three probe depths, on TRAIN facts (windows in the kit) vs HELD-OUT facts (windows not in the kit), for a checkpoint
vs the base. The facts come from the 31B pilot generation (independent of the 8B training generator).
  mem   memorization: source-window sentence containing the span, cut right before the span -> P(span | prefix)
  sem   semantic: the fact sentence (a paraphrase of the source), cut before the span
  qa    compositional/extractive: "Question: q\\nAnswer:" -> P(span | question)
Reports mean log-prob per depth and DiD = (train_ckpt - train_base) - (heldout_ckpt - heldout_base), plus greedy
exact-match on the qa probe. Also a format probe: 300 synthetic MCQs, unparsed rate and agreement with base.
"""
import json, re, argparse, unicodedata, collections, random
from pathlib import Path
import torch
R = Path.home() / "projects/teleqna/runs/teleqna-8b"; BASE = str(Path.home() / "projects/_shared/models/Qwen3-8B")
def norm(s): return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").lower()).strip()

def build():
    keep = {json.loads(l)["win_id"] for l in open(R / "data/kit/windows_keep.jsonl", encoding="utf-8")}
    wtext = {}
    sample = json.load(open(R / "data/kit/pilot_sample.json")); wset = set(sample["windows"])
    for l in open(R / "data/eg2/windows.jsonl", encoding="utf-8"):
        w = json.loads(l)
        if w["win_id"] in wset: wtext[w["win_id"]] = w["text"]
    probes = []
    for l in open(R / "data/kit/pilot_views.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["view"] != "facts" or not r["ok"]: continue
        f = json.loads(r["text"]); span = f["answer"]; src = wtext[r["win_id"]]
        i = src.lower().find(span.lower())
        if i < 0: continue
        st = max(src.rfind(". ", 0, i), src.rfind("\n", 0, i)) + 1; mem_prefix = src[st:i].strip()
        j = f["fact"].lower().find(span.lower())
        if len(mem_prefix) < 15 or j < 10: continue
        probes.append(dict(win_id=r["win_id"], group="train" if r["win_id"] in keep else "heldout", span=span,
                           mem=mem_prefix, sem=f["fact"][:j].strip(), qa=f"Question: {f['q']}\nAnswer:"))
    random.Random(7).shuffle(probes)
    byg = collections.defaultdict(list)
    for p in probes: byg[p["group"]].append(p)
    n = min(1500, len(byg["train"]), len(byg["heldout"]))
    out = byg["train"][:n] + byg["heldout"][:n]
    json.dump(out, open(R / "data/kit/tier1/probes.json", "w"), ensure_ascii=False)
    print(f"probes: train {n} heldout {n} (available {len(byg['train'])}/{len(byg['heldout'])})")

@torch.no_grad()
def score(model, tok, prefixes, spans, bs=16):
    out = []
    for k in range(0, len(prefixes), bs):
        P, Sp = prefixes[k:k+bs], spans[k:k+bs]
        full = [p + " " + s for p, s in zip(P, Sp)]
        enc = tok(full, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = model(**enc).logits.float(); lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = enc.input_ids[:, 1:]; tl = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for b, (p, s) in enumerate(zip(P, Sp)):
            npre = len(tok(p, add_special_tokens=False)["input_ids"]); L = int(enc.attention_mask[b].sum())
            pad = int((enc.attention_mask[b] == 0).sum()) if tok.padding_side == "left" else 0
            seg = tl[b, pad + npre - 1: pad + L - 1]; out.append(float(seg.mean()) if len(seg) else float("nan"))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--build", action="store_true"); ap.add_argument("--probes", default="data/kit/tier1/probes.json"); ap.add_argument("--out", default="results/kit"); a = ap.parse_args()
    if a.build: build()
    probes = json.load(open(R / a.probes))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    res = {}
    for depth in ("mem", "sem", "qa"):
        sc = score(model, tok, [p[depth] for p in probes], [p["span"] for p in probes])
        for p, s in zip(probes, sc): p[f"lp_{depth}"] = s
        for g in ("train", "heldout"):
            v = [p[f"lp_{depth}"] for p in probes if p["group"] == g and p[f"lp_{depth}"] == p[f"lp_{depth}"]]
            res[f"{depth}_{g}"] = sum(v) / len(v)
    # greedy exact match on qa probe (train/heldout)
    em = collections.Counter(); cnt = collections.Counter()
    for k in range(0, len(probes), 32):
        P = probes[k:k+32]
        enc = tok([p["qa"] for p in P], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        for p, g in zip(P, gen):
            t = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True); cnt[p["group"]] += 1; em[p["group"]] += norm(p["span"]) in norm(t)
    for g in ("train", "heldout"): res[f"em_{g}"] = em[g] / cnt[g] * 100
    (R / a.out).mkdir(parents=True, exist_ok=True); json.dump(res, open(R / f"{a.out}/probe_{a.tag}.json", "w"), indent=1)
    print(a.tag, {k: round(v, 3) for k, v in res.items()}, flush=True)
    print("PROBE_DONE", flush=True)
if __name__ == "__main__":
    (R / "results/kit").mkdir(parents=True, exist_ok=True); main()
