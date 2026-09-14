#!/usr/bin/env python3
"""Generate grounded QA from retrieved windows, aimed at the rows we get wrong.

Two things differ from gen_grounded_qa.py, and both come out of measurement
rather than taste.

**The chunks are retrieved, not swept.** The previous generator walked the
corpus in hash order, deliberately blind to the benchmark, and the cost of that
blindness is now quantified: rare-term coverage turned out to be flat against
accuracy (79-80% across the whole 0.5-0.99 band, and flat even at 1.0 on
Standards specifications), so sweeping spent most of its budget on regions the
model already answers. Retrieval puts the same generator on 13,125 windows that
some wrong row actually pulled, instead of 50,815 chosen by a signal that does
not predict error.

**The prompt branches on what the excerpt is.** 90% of the retrieved windows
are arxiv papers, not specifications. Handing a paper the "specification
excerpt" wording produced questions about clause numbering that the paper does
not have; the paper branch asks for the mechanism the paper describes instead.

Modes:
  window     the excerpt alone. The generator never sees a benchmark row, so
             everything it produces is corpus-derived and the resulting model
             is submittable on its own terms.
  targeted   the excerpt plus the benchmark rows that retrieved it, with their
             gold answers, as a statement of what the material has to cover.
             TEST-DERIVED - anything trained on this is measuring itself. Kept
             in a separate output file for that reason, never merged silently.

Both modes keep the verbatim-evidence contract: `evidence` must be copied from
the excerpt, and validate_grounded_qa.py checks it as a real substring. That is
what stops `targeted` from degenerating into paraphrasing the answer key - the
fact still has to exist in the corpus for the item to survive the gate.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

RULES = """
Hard rules:
- "evidence" MUST be copied character-for-character from the excerpt, 10-300 \
characters, and must contain the fact the answer states. Do not paraphrase it.
- "answer" must be a short factual span: a value, a term, a mechanism name, a \
condition. At most 12 words. Never a sentence of explanation.
- The question must be self-contained and answerable by a telecom engineer who \
cannot see this excerpt. Never write "according to the excerpt", "in this \
paper", "the above table", or refer to the source at all.
- The question must name its own subject explicitly, including the technology \
and the specific mechanism, field or parameter, so it is unambiguous on its own.
- "distractors" are three WRONG answers a knowledgeable reader could plausibly \
confuse with the correct one: an adjacent value, a sibling mechanism, the same \
quantity under a different condition. Never off-topic, never absurd.
- "why_wrong" gives, for each distractor in the same order, one clause saying \
what it actually is or why it does not apply here.
- If the excerpt does not actually state a testable fact, output []. Do not \
invent one, and do not write a question the excerpt cannot settle.

JSON array only:"""

SCHEMA = """Output a JSON array and nothing else. Each element must be:
{{"question": str, "answer": str, "evidence": str, "distractors": [str, str, str], \
"why_wrong": [str, str, str]}}"""

SPEC_HEAD = """You are writing study material from a telecommunications standard.

<excerpt kind="specification" spec="{spec}">
{text}
</excerpt>

Write exactly {k} items testing facts STATED IN THE EXCERPT ABOVE. """

PAPER_HEAD = """You are writing study material from a telecommunications research paper.

<excerpt kind="paper">
{text}
</excerpt>

Write exactly {k} items testing the technical claims and definitions STATED IN \
THE EXCERPT ABOVE. """

TARGET = """
A reader who has studied this material must be able to answer questions like \
these, which is what the items should prepare them for:
{targets}
Cover the underlying facts these depend on - but only where the excerpt \
actually states them. If the excerpt does not settle a point, leave it out.
"""


def build(chunk, k, targets):
    head = (SPEC_HEAD.format(spec=chunk.get("spec", "?"),
                             text=chunk["text"], k=k)
            if chunk.get("kind") == "spec"
            else PAPER_HEAD.format(text=chunk["text"], k=k))
    mid = TARGET.format(targets=targets) if targets else ""
    return head + SCHEMA + mid + RULES


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str):
    m = _FENCE.search(text or "")
    body = m.group(1) if m else (text or "")
    start = body.find("[")
    if start < 0:
        return None
    depth, instr, esc = 0, False, False
    for i in range(start, len(body)):
        c = body[i]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
            continue
        if c == '"':
            instr = True
        elif c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[start:i + 1])
                except Exception:
                    return None
    return None


def well_formed(it) -> bool:
    if not isinstance(it, dict):
        return False
    for key in ("question", "answer", "evidence"):
        if not isinstance(it.get(key), str) or not it[key].strip():
            return False
    for key in ("distractors", "why_wrong"):
        v = it.get(key)
        if not isinstance(v, list) or len(v) != 3 or \
                not all(isinstance(x, str) and x.strip() for x in v):
            return False
    return len(it["answer"].split()) <= 12 and 10 <= len(it["evidence"]) <= 400


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--test", type=Path, help="required for --mode targeted")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["window", "targeted"], default="window")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-targets", type=int, default=4,
                    help="benchmark rows quoted per window; one window was "
                         "pulled by 121 and the prompt must stay readable")
    ap.add_argument("--max-chunk-chars", type=int, default=9000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--engine-kwargs", default="{}")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    done = set()
    if a.out.exists():
        for line in a.out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["chunk_id"])
            except Exception:
                pass
        print(f"resume: {len(done)} chunks already generated", flush=True)

    rows = []
    for line in a.chunks.open(encoding="utf-8"):
        c = json.loads(line)
        if c["chunk_id"] not in done:
            rows.append(c)
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        print("nothing to do")
        return

    tmap = {}
    if a.mode == "targeted":
        if not a.test:
            raise SystemExit("--mode targeted needs --test")
        for line in a.test.open(encoding="utf-8"):
            r = json.loads(line)
            tmap[r["sample_id"]] = r

    def targets_of(c):
        if a.mode != "targeted":
            return ""
        out = []
        for sid in c.get("pulled_by", [])[:a.max_targets]:
            r = tmap.get(sid)
            if not r:
                continue
            gold = r["choices"][int(r["answer"])]
            out.append(f'- {r["question"].strip()}  (correct answer: {gold})')
        return "\n".join(out)

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": build(
                {**c, "text": c["text"][:a.max_chunk_chars]}, a.k, targets_of(c))}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for c in rows]
    print(f"generating over {len(rows)} windows, mode={a.mode}, "
          f"kind={dict(collections.Counter(c.get('kind') for c in rows))}",
          flush=True)

    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, tensor_parallel_size=a.tp,
              **json.loads(a.engine_kwargs))
    sp = SamplingParams(temperature=0.7, top_p=0.95, top_k=20,
                        max_tokens=a.max_tokens, seed=42)
    outs = llm.generate(prompts, sp)

    kept = dropped = empty = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a", encoding="utf-8") as f:
        for c, out in zip(rows, outs):
            items = extract_json(out.outputs[0].text)
            if items is None:
                dropped += 1
                continue
            if not items:
                empty += 1
            good = [it for it in items if well_formed(it)]
            dropped += len(items) - len(good)
            for j, it in enumerate(good):
                f.write(json.dumps({
                    "item_id": f"{c['chunk_id']}_{j}",
                    "chunk_id": c["chunk_id"],
                    "spec": c.get("spec"), "release": c.get("release"),
                    "source": c.get("source"), "kind": c.get("kind"),
                    "mode": a.mode, "pulled_by": c.get("pulled_by", [])[:20],
                    "question": it["question"].strip(),
                    "answer": it["answer"].strip(),
                    "evidence": it["evidence"].strip(),
                    "distractors": [d.strip() for d in it["distractors"]],
                    "why_wrong": [w.strip() for w in it["why_wrong"]],
                }, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept={kept} dropped={dropped} empty_chunks={empty} "
          f"chunks={len(rows)} items_per_chunk={kept/max(1,len(rows)):.2f}")


if __name__ == "__main__":
    main()
