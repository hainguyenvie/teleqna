#!/usr/bin/env python3
"""Uniform weight average of N checkpoints from the same lineage (SWA-style denoising of fragile decisions).
Usage: soup_n.py --out DIR --ckpts A,B,C[,...]"""
import argparse, json, shutil, torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
ap = argparse.ArgumentParser(); ap.add_argument("--ckpts", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
dirs = [Path(p) for p in a.ckpts.split(",")]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
def shards(d):
    idx = d / "model.safetensors.index.json"
    return sorted({d / v for v in json.load(open(idx))["weight_map"].values()}) if idx.exists() else [d / "model.safetensors"]
def load(d):
    t = {}
    for s in shards(d):
        with safe_open(str(s), "pt") as f:
            for k in f.keys(): t[k] = f.get_tensor(k)
    return t
acc = None; n = 0
for d in dirs:
    t = load(d); n += 1
    if acc is None: acc = {k: v.float() for k, v in t.items()}; dt = {k: v.dtype for k, v in t.items()}
    else:
        for k in acc: acc[k] += t[k].float()
    del t
M = {k: (acc[k] / n).to(dt[k]).contiguous() for k in acc}; save_file(M, str(out / "model.safetensors"), metadata={"format": "pt"})
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "special_tokens_map.json"):
    if (dirs[0] / f).exists(): shutil.copy(dirs[0] / f, out / f)
print(f"averaged {n} checkpoints -> {out}")
