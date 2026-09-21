#!/usr/bin/env python3
"""TIES-merging (Yadav et al. 2023) of N fine-tunes of the same base: task vectors = ckpt - base; TRIM each to its top-k%
magnitudes; ELECT the sign per parameter by summed magnitude; DISJOINT-MEAN the agreeing entries; add lambda * merged
vector to the base. Usage: ties_merge.py --base B --ckpts A,C --out DIR [--k 0.2 --lam 1.0]. Also --plain for the
task-arithmetic mean (no trim/elect)."""
import argparse, json, shutil, torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--ckpts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=float, default=0.2); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--plain", action="store_true"); ap.add_argument("--weights", default="", help="comma weights for the plain merge (default uniform)"); a = ap.parse_args()
def shards(d):
    idx = d / "model.safetensors.index.json"
    return sorted({d / v for v in json.load(open(idx))["weight_map"].values()}) if idx.exists() else [d / "model.safetensors"]
def load(d):
    t = {}
    for s in shards(d):
        with safe_open(str(s), "pt") as f:
            for k in f.keys(): t[k] = f.get_tensor(k)
    return t
base = Path(a.base); dirs = [Path(p) for p in a.ckpts.split(",")]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
B = load(base); tv = [load(d) for d in dirs]; M = {}
for name, b in B.items():
    bf = b.float(); deltas = [t[name].float() - bf for t in tv]
    if a.plain:
        w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0 / len(deltas)] * len(deltas)
        merged = sum(wi * d for wi, d in zip(w, deltas))
    else:
        trimmed = []
        for d in deltas:
            flat = d.abs().flatten(); kth = max(1, int(flat.numel() * a.k))
            thr = torch.topk(flat, kth, largest=True).values.min() if flat.numel() > 1 else 0
            trimmed.append(torch.where(d.abs() >= thr, d, torch.zeros_like(d)))
        S = torch.stack(trimmed); sign = torch.sign(S.sum(0)); agree = (torch.sign(S) == sign) & (S != 0)
        merged = (S * agree).sum(0) / agree.sum(0).clamp(min=1)
    M[name] = (bf + a.lam * merged).to(b.dtype).contiguous()
save_file(M, str(out / "model.safetensors"), metadata={"format": "pt"})
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json"):
    src = dirs[0] / f if (dirs[0] / f).exists() else base / f
    if src.exists(): shutil.copy(src, out / f)
print(f"{'plain' if a.plain else 'TIES k=%.2f' % a.k} lam={a.lam}: merged {len(dirs)} ckpts -> {out}")
