#!/usr/bin/env python3
"""Measure each GRPO prompt's sampled pass-rate, then keep only the band that
can actually produce a gradient.

Two independent problems make unfiltered MCQ prompts a bad RL diet:

  1. Zero variance = zero gradient. GRPO's advantage is computed inside the
     group of k rollouts for one prompt. If all k agree — always right or
     always wrong — the advantage is 0 for every one of them and the prompt
     contributes nothing but generation cost.

  2. Guessing. With 4-5 choices a policy that reasons badly still lands on
     the gold letter 20-25% of the time, and outcome-only reward will happily
     reinforce whatever reasoning produced that lucky hit. Filtering out the
     prompts that are trivially guessable (and the ones that are hopeless) is
     the standard mitigation; DAPO does the same thing online as "dynamic
     sampling", we do it once offline because it doubles as a difficulty map.

So: sample k completions per prompt with thinking on, score with the
benchmark's own parser, and write the count. The companion filter keeps
prompts with 1 <= correct <= k-1, tightened by --lo/--hi.

Runs on the vllm venv; needs no peft (profile the merged model you will
actually start GRPO from — pass rates shift when the policy shifts).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
BARE = re.compile(r"^\s*([A-Ea-e])(?:[).:,\s]|$)")


def parse(text: str, n: int) -> str:
    m = STRICT.findall(text or "") or LOOSE.findall(text or "") or BARE.findall(text or "")
    if not m:
        return ""
    got = m[-1].strip().rstrip(".").upper()
    return got if got in {chr(65 + i) for i in range(n)} else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size. 1 for the 8B; the 122B's BF16 "
                         "weights are 250GB and one H200 holds 143GB.")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="4096 is right for the 8B at a 1,024-token budget. A "
                         "pass@k ceiling is destroyed by truncation — a rollout "
                         "that runs out of tokens scores zero and is "
                         "indistinguishable from a wrong answer — so a verbose "
                         "model needs both this and --max-tokens raised.")
    ap.add_argument("--engine-kwargs", default="",
                    help="JSON merged into LLM(). The 122B needs "
                         "language_model_only, disable_custom_all_reduce and a "
                         "max_num_seqs its Mamba cache can hold; each of those "
                         "was found by a 250GB crash.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"profiling {len(rows)} prompts x k={args.k}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=True) for r in rows]

    engine = dict(model=args.model, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len,
                  tensor_parallel_size=args.tp, enforce_eager=False)
    if args.engine_kwargs:
        engine.update(json.loads(args.engine_kwargs))
    print("engine:", {k: v for k, v in engine.items() if k != "model"}, flush=True)
    llm = LLM(**engine)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        top_k=20, max_tokens=args.max_tokens, seed=42)
    outs = llm.generate(prompts, sp)

    hist = [0] * (args.k + 1)
    with args.out.open("w", encoding="utf-8") as f:
        for row, out in zip(rows, outs):
            n = row["n_choices"]
            got = [parse(o.text, n) for o in out.outputs]
            correct = sum(g == row["answer"] for g in got)
            unparsed = sum(1 for g in got if not g)
            hist[correct] += 1
            rec = dict(row)
            rec["k"] = args.k
            rec["correct"] = correct
            rec["unparsed"] = unparsed
            # keep the letters, not just the count: the count answers "is this
            # prompt trainable", the letters additionally answer "would voting
            # have got it" — the sampled-branch counterpart of the 4-permutation
            # vote, at no extra generation cost.
            rec["letters"] = got
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(rows)
    print("\npass-rate histogram (correct out of k):")
    for c, n in enumerate(hist):
        bar = "#" * int(60 * n / max(1, total))
        print(f"  {c}/{args.k}  {n:6d}  {n/total*100:5.1f}%  {bar}")
    usable = sum(hist[1:args.k])
    print(f"\nnon-degenerate (1..{args.k-1}): {usable} = {usable/total*100:.1f}% "
          f"| always-wrong {hist[0]} | always-right {hist[args.k]}")
    print(f"mean pass-rate: {sum(c*n for c, n in enumerate(hist))/(total*args.k):.3f}")


if __name__ == "__main__":
    main()
