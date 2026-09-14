#!/usr/bin/env python3
"""Arm H stage 2: clean the generated items, then keep only the ones that teach.

Three defects in the raw generation, all fixed here:

  * gold-letter bias — A 5533 / B 5428 / C 3978 / D 1053. Training on that would
    install a letter prior, which is exactly the selection bias the whole project
    has been trying to avoid. Options are rotated per item so the gold letter is
    uniform.
  * 3.28% of items refer to "the passage" / "according to the text", which the
    instruction forbade and which cannot be answered closed-book.
  * 19 items have duplicate options.

Then the part that decides whether arm H can work at all. An item the base model
already answers correctly teaches nothing; an item it cannot answer even WITH the
passage in front of it is a broken item, not a hard one. So both passes are run
in one model load and items are sorted:

    resolvable (right with the passage) AND base wrong closed-book -> teaches
    resolvable AND base right                                      -> anchor
    not resolvable                                                 -> discard

That is arm G's flip/keep structure applied to synthetic questions, and the same
reason it works: the labels are only trusted where evidence backs them.
"""
import argparse, json, os, pathlib, random, re, sys, collections

ROOT = os.path.expanduser("~/projects/telelogs/runs/teleqna-sft")
sys.path.insert(0, f"{ROOT}/infra")
from eval_dev_vllm import build_prompt, parse

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--data", type=pathlib.Path,
                default=pathlib.Path(f"{ROOT}/data/synth_mcq.jsonl"))
ap.add_argument("--out", type=pathlib.Path,
                default=pathlib.Path(f"{ROOT}/data/synth_mcq_scored.jsonl"))
ap.add_argument("--gpu-mem", type=float, default=0.90)
a = ap.parse_args()

META = re.compile(r"\b(the (passage|text|document|article|excerpt|paper)|according to)\b", re.I)
rows = [json.loads(l) for l in a.data.open()]
print(f"{len(rows)} raw items", flush=True)

clean = []
drop = collections.Counter()
rng = random.Random(23)
for r in rows:
    ch = [c.strip() for c in r["choices"]]
    if len(set(c.lower() for c in ch)) < len(ch):
        drop["duplicate options"] += 1
        continue
    if META.search(r["question"]) or any(META.search(c) for c in ch):
        drop["refers to the passage"] += 1
        continue
    if not (3 <= len(ch) <= 5):
        drop["bad option count"] += 1
        continue
    # rotate so the gold letter is uniform across the set
    k = len(ch)
    want = rng.randrange(k)
    s = (want - r["answer"]) % k
    rot = [None] * k
    for i, c in enumerate(ch):
        rot[(i + s) % k] = c
    assert rot[want] == ch[r["answer"]]
    clean.append({**r, "choices": rot, "answer": want})
print(f"{len(clean)} after cleaning; dropped {dict(drop)}", flush=True)
g = collections.Counter(chr(65 + r["answer"]) for r in clean)
print(f"  gold letter after rotation: {dict(sorted(g.items()))}", flush=True)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)

def render(row, with_ctx):
    q = dict(row)
    if with_ctx:
        q = dict(q, question=("Reference material retrieved from the telecom literature.\n\n"
                              + row["window"] + "\n\n---\n\n" + row["question"]))
    return tok.apply_chat_template([{"role": "user", "content": build_prompt(q)}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)

closed = [render(r, False) for r in clean]
opened = [render(r, True) for r in clean]
lens = sorted(len(tok(p).input_ids) for p in opened)
print(f"with-context prompt tokens p50={lens[len(lens)//2]} max={lens[-1]}", flush=True)

llm = LLM(model=a.base, gpu_memory_utilization=a.gpu_mem, max_model_len=4096,
          tensor_parallel_size=1, enforce_eager=False)
sp = SamplingParams(temperature=0.0, max_tokens=512)
oc = llm.generate(closed, sp)
oo = llm.generate(opened, sp)

n = collections.Counter()
with a.out.open("w") as fh:
    for r, c, o in zip(clean, oc, oo):
        k = len(r["choices"])
        goldL = chr(65 + r["answer"])
        cb = parse(c.outputs[0].text, k)
        wc = parse(o.outputs[0].text, k)
        resolvable = wc == goldL
        base_right = cb == goldL
        tier = ("discard" if not resolvable else
                ("teaches" if not base_right else "anchor"))
        n[tier] += 1
        fh.write(json.dumps({**r, "closed_book": cb, "with_ctx": wc,
                             "resolvable": resolvable, "base_right": base_right,
                             "tier": tier}, ensure_ascii=False) + "\n")
print(f"tiers: {dict(n)}", flush=True)
tot = sum(n.values())
print(f"  resolvable with the passage: {(tot-n['discard'])/tot*100:.1f}%", flush=True)
print(f"  of those, base already right: {n['anchor']/(tot-n['discard'])*100:.1f}%", flush=True)
print(f"wrote {a.out}", flush=True)
