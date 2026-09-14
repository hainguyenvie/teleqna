#!/usr/bin/env python3
"""dev-1000 scorer on vLLM, with LoRA served rather than merged.

Why this replaces eval_dev.py as the measurement stack. The transformers path
takes ~3h25m for one 1,000-row thinking eval — batch 32, up to 6,000 new tokens,
and every batch waits for its slowest sequence. Two cards therefore buy about
six measurements a day, which made *measuring* the bottleneck rather than
training. It also punished exactly the models this project is now producing: one
trained to justify before answering is more verbose, so it was both slower to
score and, at the old 8-token no-think budget, scored near zero for reasons that
had nothing to do with what it knew.

vLLM is also what serving uses, so this measures closer to deployment.

The contract is otherwise held byte-for-byte:

  prompt   the Inspect harness template, identical to eval_dev.py and
           run_baseline.py, chat-templated with the same enable_thinking flag
  decode   greedy (temperature 0), as before
  parse    the same STRICT -> LOOSE -> BARE cascade, last match wins
  output   the same {"summary", "results"} schema, so every analysis script
           already written keeps working

**This is a new eval stack and its numbers are not comparable with the
transformers ones.** Nothing in this project may be compared across the two;
the A/A references (base thinking, base no-think, dpo3@200) have to be
re-measured here, which is why --adapter takes a list: one model load scores
every arm, and all of them land in the same stack on the same day.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
BARE = re.compile(r"^\s*([A-Ea-e])(?:[).:,\s]|$)")

TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)


def parse(completion: str, n: int) -> str:
    """This repo's lenient cascade, kept for continuity with earlier arms."""
    m = STRICT.findall(completion or "") or LOOSE.findall(completion or "") \
        or BARE.findall(completion or "")
    if not m:
        return ""
    got = m[-1].strip().rstrip(".").upper()
    return got if got in {chr(65 + i) for i in range(n)} else ""


def parse_official(completion: str, n: int) -> str:
    """Inspect's `parse_answers`, verbatim: STRICT then LOOSE, then give up.

    The harness comment is explicit that an unparsable reply is simply wrong —
    there is no bare-letter fallback. Audited over every stored arm, the extra
    fallback credited 0 rows everywhere except one format-drifted checkpoint,
    where it added 6. Since this track now trains models that justify before
    answering, and those are precisely the ones that drift, the official number
    is the one to report.
    """
    m = STRICT.findall(completion or "") or LOOSE.findall(completion or "")
    if not m:
        return ""
    got = m[-1].strip().rstrip(".").upper()
    return got if got in {chr(65 + i) for i in range(n)} else ""


def build_prompt(row: dict) -> str:
    ch = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(row["choices"]))
    letters = ",".join(chr(65 + i) for i in range(len(row["choices"])))
    return TEMPLATE.format(letters=letters, question=row["question"], choices=ch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default="dev1000",
                    help="prefix for the output file names")
    ap.add_argument("--adapter", action="append", default=[],
                    help="name=/path/to/adapter, repeatable. Every adapter is "
                         "scored in the same process off one model load.")
    ap.add_argument("--with-base", action="store_true",
                    help="also score the bare base model")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--open-thought", action="store_true",
                    help="end the prompt with the thought channel opened but "
                         "not closed; see the block in main()")
    ap.add_argument("--max-new", type=int, default=0,
                    help="0 = 6000 thinking / 512 no-think. The no-think "
                         "default is 512 rather than eval_dev.py's 8: a model "
                         "trained to justify first needs room to reach its own "
                         "answer line, and a base model stops early anyway, so "
                         "the wider budget costs it nothing and makes the two "
                         "comparable.")
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size. 1 for the 8B arms; 2 is what "
                         "the 122B needs — its BF16 weights are 250GB and one "
                         "H200 holds 143GB.")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="Deliberately not the model's native window. Qwen3.5 "
                         "advertises 262144, but every TeleQnA prompt is a few "
                         "hundred tokens and KV cache reserved for a window we "
                         "never use is memory the weights need.")
    ap.add_argument("--engine-kwargs", default="",
                    help="JSON dict merged into the LLM() call, for engine "
                         "flags that differ between vLLM versions (the 122B is "
                         "served from a nightly tree, the 8B arms from 0.11.0). "
                         "Keeps one scorer for both instead of forking it.")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": build_prompt(r)}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=args.thinking)
        for r in rows]

    # --open-thought: leave the thought channel OPEN instead of pre-closed.
    #
    # Gemma-4-derived templates offer only two endings, and on OTel-2.0-31B they
    # behave nothing alike. enable_thinking=False ends the prompt with
    #     <|turn>model\n<|channel>thought\n<channel|>\n
    # -- an empty thought opened and immediately closed -- and the model then
    # answers in the required format on 987 of 1000 rows. enable_thinking=True
    # ends with a bare <|turn>model\n, and the model stops opening a thought
    # channel at all, answers "C) 150 ms" conversationally, and emits ANSWER: on
    # 59 rows. So what enforces the format is not the instruction in the prompt,
    # which is identical either way -- it is the channel scaffold.
    #
    # That leaves an untried third ending: opened but not closed. It gives the
    # model the scaffold it clearly recognises while still leaving room to think
    # before answering, which is exactly the combination neither flag produces.
    if args.open_thought:
        CLOSE = "<channel|>"
        out = []
        for p in prompts:
            i = p.rfind(CLOSE)
            if i < 0 or "<|channel>thought" not in p:
                raise SystemExit(
                    "--open-thought: this template has no pre-closed thought "
                    "channel to open, so the flag would silently do nothing")
            out.append(p[:i])
        prompts = out
        print("open-thought: prompts end with",
              repr(prompts[0][-40:]), flush=True)

    max_new = args.max_new or (6000 if args.thinking else 512)
    arms = []
    if args.with_base:
        arms.append(("base", None))
    for spec in args.adapter:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--adapter wants name=path, got {spec!r}")
        arms.append((name, path))
    if not arms:
        raise SystemExit("nothing to score: pass --with-base and/or --adapter")

    engine = dict(model=args.base, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len,
                  tensor_parallel_size=args.tp,
                  enable_lora=bool(args.adapter),
                  max_lora_rank=args.max_lora_rank, enforce_eager=False)
    if args.engine_kwargs:
        engine.update(json.loads(args.engine_kwargs))
    print("engine:", {k: v for k, v in engine.items() if k != "model"}, flush=True)
    llm = LLM(**engine)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, (name, path) in enumerate(arms, 1):
        lreq = LoRARequest(name, i, path) if path else None
        outs = llm.generate(prompts, sp, lora_request=lreq)

        results, correct, correct_off, by_subject = [], 0, 0, {}
        for row, out in zip(rows, outs):
            completion = out.outputs[0].text
            n = len(row["choices"])
            gold = chr(65 + int(row["answer"]))
            got = parse(completion, n)
            got_off = parse_official(completion, n)
            ok, ok_off = got == gold, got_off == gold
            correct += ok
            correct_off += ok_off
            # by_subject follows the official scorer — it is what gets reported
            by_subject.setdefault(row.get("subject", "?"), []).append(int(ok_off))
            results.append({"sample_id": row["sample_id"], "parsed": got_off,
                            "parsed_lenient": got, "correct": bool(ok_off),
                            "correct_lenient": bool(ok), "completion": completion})

        summary = {
            "stack": "vllm", "adapter": path, "arm": name,
            "thinking": args.thinking, "max_new": max_new,
            "total": len(rows), "correct": correct_off,
            "accuracy": round(correct_off / len(rows), 4),
            "accuracy_lenient": round(correct / len(rows), 4),
            "bare_only_rows": correct - correct_off,
            "unparsed": sum(1 for r in results if not r["parsed"]),
            "by_subject": {s: {"n": len(v), "acc": round(sum(v) / len(v), 4)}
                            for s, v in sorted(by_subject.items())},
        }
        mode = "think" if args.thinking else f"nothink{max_new}"
        out_path = args.out_dir / f"{args.tag}_{name}_{mode}.json"
        out_path.write_text(json.dumps({"summary": summary, "results": results},
                                        ensure_ascii=False, indent=1))
        std = summary["by_subject"].get("Standards specifications", {})
        print(f"  -> {out_path.name:<52s} acc={summary['accuracy']:.4f} "
              f"StdSpec={std.get('acc', -1):.4f} unparsed={summary['unparsed']}",
              flush=True)


if __name__ == "__main__":
    main()
