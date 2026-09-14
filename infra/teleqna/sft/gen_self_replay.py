#!/usr/bin/env python3
"""Build replay anchors out of the base model's own correct answers.

The failed run used `anchor_mcq.jsonl` as its anchor. An audit of that file
says it is 94.0% the generated refute schema and carries aggregate options
("all of the above" / "none of the above") on 1.24% of items. The base model
uses that schema on 0.0% of its own answers, and the benchmark carries
aggregate options on 11.85%. So the anchor was not a sample of the behaviour
being protected -- it was more of the treatment, in the treatment's voice, with
the treatment's option distribution. It could not have anchored anything.

The behaviour we want to keep is the base model's, so the anchor should be the
base model's own output: greedy, no-think, the harness template byte-for-byte,
and only the rows it got right. Wrong rows are dropped rather than corrected --
an anchor exists to preserve what already works, and replaying an error would
cement it on exactly the questions the knowledge rows are trying to fix.

Drawn from heldout9000, never dev1000. That keeps dev1000 a clean read on
retention: any recovery there has to come from the anchor generalising, not
from the model having been shown the answer. A production run can widen the
pool afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dev_vllm import build_prompt, parse_official  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(r)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for r in rows]

    llm = LLM(model=args.base, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, tensor_parallel_size=1, enforce_eager=False)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                max_tokens=args.max_new))

    kept, dropped = [], {"wrong": 0, "unparsed": 0, "truncated": 0}
    for r, p, o in zip(rows, prompts, outs):
        comp = o.outputs[0].text
        got = parse_official(comp, len(r["choices"]))
        if not got:
            dropped["unparsed"] += 1
            continue
        # A completion that ran into the token budget has no closing answer
        # line of its own; training on the stump teaches the model to stop
        # mid-sentence.
        if o.outputs[0].finish_reason != "stop":
            dropped["truncated"] += 1
            continue
        if got != chr(65 + int(r["answer"])):
            dropped["wrong"] += 1
            continue
        kept.append({"prompt": build_prompt(r), "completion": comp,
                     "sample_id": f"{r['sample_id']}::replay",
                     "view": "replay", "role": "anchor",
                     "subject": r.get("subject", "?")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for k in kept:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    report = {"in": len(rows), "kept": len(kept), "dropped": dropped,
              "keep_rate": round(len(kept) / len(rows), 4)}
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
