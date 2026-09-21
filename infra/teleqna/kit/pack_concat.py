#!/usr/bin/env python3
"""Concatenate packs (same block size) and shuffle blocks: pack_concat.py --out OUT --packs A,B[,C]"""
import argparse, numpy as np
from pathlib import Path
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
ap = argparse.ArgumentParser(); ap.add_argument("--packs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
I = np.concatenate([np.load(R / f"{p}_ids.npy") for p in a.packs.split(",")]); M = np.concatenate([np.load(R / f"{p}_mask.npy") for p in a.packs.split(",")])
perm = np.random.RandomState(a.seed).permutation(I.shape[0]); np.save(R / f"{a.out}_ids.npy", I[perm]); np.save(R / f"{a.out}_mask.npy", M[perm])
print(f"{I.shape[0]:,} blocks ({I.shape[0]*I.shape[1]/1e6:.0f}M tokens) -> {a.out}"); print("PACK_DONE")
