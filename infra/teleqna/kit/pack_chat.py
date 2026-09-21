#!/usr/bin/env python3
"""Pack chat rows {prompt, completion} into (ids, mask) blocks of 1024: loss on the completion tokens AND on the
<|im_end|> that follows (teaches the model to stop after 'ANSWER: X'). Rows are concatenated and cut into blocks."""
import json, argparse, numpy as np
from pathlib import Path
from transformers import AutoTokenizer
R = Path.home() / "projects/teleqna/runs/teleqna-8b"
ap = argparse.ArgumentParser(); ap.add_argument("--rows", required=True); ap.add_argument("--out", required=True); ap.add_argument("--blk", type=int, default=1024); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(str(Path.home() / "projects/_shared/models/Qwen3-8B")); END = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
ids, mask, n, npad = [], [], 0, 0; PAD = tok.pad_token_id
for l in open(R / a.rows, encoding="utf-8"):
    r = json.loads(l)
    p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pi = tok(p, add_special_tokens=False)["input_ids"]; ci = tok(r["completion"], add_special_tokens=False)["input_ids"] + END
    row = pi + ci
    if len(row) > a.blk: continue
    room = a.blk - (len(ids) % a.blk)
    if len(row) > room:   # never let a row straddle a block: its completion would train without its prompt (was 7% of blocks)
        ids.extend([PAD] * room); mask.extend([0] * room); npad += room
    ids.extend(row); mask.extend([0] * len(pi) + [1] * len(ci)); n += 1
nblk = len(ids) // a.blk; I = np.array(ids[:nblk * a.blk], dtype=np.int32).reshape(nblk, a.blk); M = np.array(mask[:nblk * a.blk], dtype=np.int8).reshape(nblk, a.blk)
perm = np.random.RandomState(a.seed).permutation(nblk); np.save(R / f"{a.out}_ids.npy", I[perm]); np.save(R / f"{a.out}_mask.npy", M[perm])
print(f"{n:,} rows -> {nblk:,} blocks ({nblk*a.blk/1e6:.0f}M tokens), loss-bearing {int(M.sum()):,}, pad {npad:,} -> {a.out}_ids.npy"); print("PACK_DONE")
