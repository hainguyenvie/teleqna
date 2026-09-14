#!/usr/bin/env python3
"""Score the view-transfer probe for base and one adapter off a single load.

Two metrics, because the cells are not the same shape. The mcq cells are graded
on the official letter parse, exactly as the benchmark is. The reframed cell is
free text, so it is graded on whether the gold answer string survives in the
generation -- a loose test, deliberately, since the question is whether the fact
is present at all and not whether it is phrased well.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STRICT = re.compile(r"ANSWER:\s*([A-J])\b", re.I)
LOOSE = re.compile(r"\b([A-J])\b")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def parse_letter(text: str):
    m = STRICT.search(text)
    if m:
        return m.group(1).upper()
    tail = text.strip()[-40:]
    m = LOOSE.findall(tail)
    return m[-1].upper() if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--adapter", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for r in rows]

    llm = LLM(model=args.base, gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, tensor_parallel_size=1,
              enable_lora=bool(args.adapter), max_lora_rank=64, max_loras=1)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)

    arms = [("base", None)]
    for i, spec in enumerate(args.adapter, start=1):
        name, path = spec.split("=", 1)
        arms.append((name, LoRARequest(name, i, path)))

    report = {}
    for name, lora in arms:
        outs = llm.generate(prompts, sp, lora_request=lora) if lora else \
            llm.generate(prompts, sp)
        cells = defaultdict(lambda: [0, 0])
        for r, o in zip(rows, outs):
            txt = o.outputs[0].text
            if r["view"].startswith("mcq"):
                ok = parse_letter(txt) == r["gold_letter"]
            else:
                ok = norm(r["answer_text"]) in norm(txt)
            c = cells[(r["pool"], r["cell"])]
            c[0] += int(ok); c[1] += 1
        report[name] = {f"{p}/{c}": round(v[0] / v[1] * 100, 2)
                        for (p, c), v in sorted(cells.items())}
        print(f"\n{name}")
        for k, v in report[name].items():
            print(f"  {k:20s} {v:6.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\n-> {args.out}")

    if len(arms) == 2:
        b, a = report["base"], report[arms[1][0]]
        print(f"\n{'cell':22s} {'base':>7s} {arms[1][0]:>7s} {'delta':>7s}")
        for k in b:
            print(f"{k:22s} {b[k]:7.2f} {a[k]:7.2f} {a[k]-b[k]:+7.2f}")


if __name__ == "__main__":
    main()
