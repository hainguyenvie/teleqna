#!/usr/bin/env python3
"""Merge a LoRA adapter into the base weights so vLLM can serve the tuned
model as a plain checkpoint. The held-out 9,000 must be scored on the SAME
stack as the official baselines (vLLM harness, temp 0.6, seed 42) — the
transformers eval_dev.py stack reads ~+0.2pp higher (A/A no-think: 74.00 vs
73.80) and its greedy decoding differs from the baseline arm's sampling."""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cpu", local_files_only=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    tok.save_pretrained(args.out)
    print(f"merged -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
