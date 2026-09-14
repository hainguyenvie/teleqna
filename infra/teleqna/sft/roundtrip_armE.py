#!/usr/bin/env python3
"""Arm E stage 1b: does the generated fact, alone, carry the answer?

A training target is only worth writing into the weights if the sentence itself
is what settles the question. So each fact is fed back as the ONLY context --
no retrieval, no options in the query, nothing else -- and the model answers the
original MCQ. Three outcomes:

  agree   the fact moves the model to the gated letter -> keep
  inert   the model still answers something else -> the sentence does not carry
          the knowledge, so training on it teaches an assertion the model cannot
          derive; drop it
  self-refuting facts ("the material does not contain...") are dropped before
          this even runs

This is not a label check -- gold is never consulted here -- so it stays a
legitimate filter. The purity audit against gold happens afterwards, separately.
"""
import argparse, json, os, pathlib, re, sys

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
sys.path.insert(0, f"{ROOT}/infra")
from eval_dev_vllm import build_prompt, parse

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--facts", type=pathlib.Path, required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
a = ap.parse_args()

CANON = os.path.expanduser("~/projects/telelogs/runs/bench4/teleqna/data/test.jsonl")
ref = {}
for l in open(CANON):
    r = json.loads(l)
    ch = r["choices"]
    r["choices"] = eval(ch) if isinstance(ch, str) else ch
    ref[r["sample_id"]] = r

NOEV = re.compile(r"\b(does not (contain|provide|specify|mention)|cannot (support|determine|"
                  r"be determined)|no information|not (enough|sufficient) information|"
                  r"is not (present|available|specified))\b", re.I)

rows = [json.loads(l) for l in a.facts.open()]
keep = [r for r in rows if not r["flag"] and not NOEV.search(r["fact"])]
print(f"{len(rows)} facts -> {len(keep)} after dropping flagged and self-refuting "
      f"({len(rows)-len(keep)} out)", flush=True)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)

prompts = []
for r in keep:
    q = dict(ref[r["sample_id"]])
    q["question"] = f"{r['fact']}\n\n{q['question']}"
    prompts.append(tok.apply_chat_template(
        [{"role": "user", "content": build_prompt(q)}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))

llm = LLM(model=a.base, gpu_memory_utilization=0.90, max_model_len=4096,
          tensor_parallel_size=1, enforce_eager=False)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))

n_agree = 0
with a.out.open("w") as fh:
    for r, o in zip(keep, outs):
        got = parse(o.outputs[0].text, len(ref[r["sample_id"]]["choices"]))
        agree = got == r["good"]
        n_agree += agree
        fh.write(json.dumps({**r, "rt_letter": got, "rt_agree": bool(agree)}) + "\n")
print(f"fact carries the gated answer on {n_agree}/{len(keep)} = "
      f"{n_agree/len(keep)*100:.1f}%", flush=True)
print(f"wrote {a.out}", flush=True)
