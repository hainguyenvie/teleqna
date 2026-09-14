#!/usr/bin/env python3
"""One shard of the ot-full pass@k measurement, on one card.

Why this exists next to profile_pass_rate.py rather than as a flag on it. That
script hardcodes `enable_thinking=True`, and this base cannot be scored that
way: its greedy think arm parses 52/1000 on ot-lite against 732 for no-think,
941 rows unparsed. A thinking pass@k here would measure the channel scaffold
falling over, not the ceiling. It also has no shard split, and the whole point
of this run is to spread 10,000 prompts x k over every idle card at once.

The parser is imported from profile_pass_rate rather than copied, so this
measurement and every earlier pass-rate number grade identically.

max_tokens is deliberately generous. A rollout that runs out of budget scores
zero and is indistinguishable from a wrong answer, which biases a ceiling
downward -- and the no-think eval that produced the reference number ran at
max_new 8000 and still left 7.8% of rows unparsed, so this model does ramble.
vLLM stops at EOS regardless, so a high cap costs KV reservation and nothing
else.

Sharding is by index modulo nshards, not by contiguous block: the benchmark is
ordered by subject, so contiguous shards would give each card a different
subject mix and make a half-finished run unreadable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_pass_rate import parse  # noqa: E402  -- same grader, on purpose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--thinking", action="store_true",
                    help="off by default; this base's think channel is broken")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    rows = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(rows)} prompts x k={args.k} "
          f"thinking={args.thinking}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=args.thinking)
               for r in rows]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, tensor_parallel_size=1,
              enforce_eager=False)
    # seed varies with the shard so the shards are not eight copies of the same
    # eight samples; within a shard it is fixed so the run is reproducible.
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        top_k=20, max_tokens=args.max_tokens,
                        seed=42 + args.shard)
    outs = llm.generate(prompts, sp)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hist = [0] * (args.k + 1)
    ntrunc = 0
    with args.out.open("w", encoding="utf-8") as f:
        for row, out in zip(rows, outs):
            n = row["n_choices"]
            got = [parse(o.text, n) for o in out.outputs]
            ntrunc += sum(1 for o in out.outputs if o.finish_reason == "length")
            rec = dict(row)
            rec["k"] = args.k
            rec["correct"] = sum(g == row["answer"] for g in got)
            rec["unparsed"] = sum(1 for g in got if not g)
            rec["letters"] = got
            hist[rec["correct"]] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tot = len(rows)
    print(f"\nshard {args.shard}: pass@1 "
          f"{sum(c * n for c, n in enumerate(hist)) / (tot * args.k):.4f}  "
          f"pass@{args.k} {sum(hist[1:]) / tot:.4f}  "
          f"band {sum(hist[1:args.k]) / tot:.4f}  "
          f"truncated {ntrunc}/{tot * args.k}", flush=True)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
