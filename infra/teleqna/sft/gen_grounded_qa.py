#!/usr/bin/env python3
"""Generate spec-grounded QA locally, with the spec chunk in the context window.

Why local, and why the teacher's own knowledge does not matter much here. The
DeepSeek balance is exhausted and the Viettel gateway key is dead, so there is
no external teacher. That turns out to be tolerable for this particular job:
with the excerpt in front of it the task is reading comprehension, not recall,
so the source of truth is the document rather than the teacher's weights. This
is context distillation — the student is later trained to produce, closed-book,
what it can only produce today with the document in view.

Two shapes are emitted per chunk, because this track has two measured failure
modes and they need different supervision:

  short_qa  question -> a short factual answer. Targets the 16% of dev rows
            where no sampled trace ever reaches the gold letter, i.e. the
            knowledge hole. Answer is kept under ~12 words so that automatic
            equivalence checking downstream is honest rather than a judge's
            opinion.
  mcq_why   question -> one gold and three near-miss options, each with a
            written reason it is wrong. The distractor probe put +15.83pp on
            option discrimination, and every previous attempt supervised only
            the letter — one bit, and a channel that four DPO recipes and an
            SFT run have already exhausted. Supervising the *reason* is the
            part that was never tried.

Nothing here is trusted on the generator's say-so. Every item carries a
verbatim `evidence` span that validate_grounded_qa.py checks is a real
substring of the chunk, and the double gate there drops anything the student
can already answer closed-book.

Resumable: chunk_ids already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

# 45.912 and 45.050 are 25% of the chunk pool between them and are known noise
# for this benchmark — the same two specs the DPO/GRPO builders downweight.
DOWNWEIGHT_SPECS = {"45.912", "45.050"}

# The 21 series is Release Description / Summary of Work Items: catalogues of
# what exists in a release, not statements of how anything works. They rank
# first in errors_to_specs.json (21.917 "explains" 1,276 error rows, more than
# 23.501) for a bad reason — a document that name-drops every technology in the
# release matches the rare terms of almost any question. The first pilot took
# chunks in file order, which is that ranking, and 43% of what survived the
# gates was 3GPP document trivia: portal usage, change-request procedure, what
# the second digit of a version number means. Term-dense, knowledge-empty.
EXCLUDE_SERIES = {"21"}

# Front matter states nothing testable. Matched against the chunk's headings.
FRONT_MATTER = re.compile(
    r"(?i)\b(foreword|scope|references|definitions,? symbols and abbreviations"
    r"|abbreviations|contents|copyright|change history|document history"
    r"|introduction)\b")

PROMPT = """You are writing study material from a 3GPP specification excerpt.

<spec_excerpt spec="TS {spec}" release="{release}">
{text}
</spec_excerpt>

Write exactly {k} items testing facts STATED IN THE EXCERPT ABOVE. Output a JSON \
array and nothing else.

Each element must be:
{{"question": str, "answer": str, "evidence": str, "distractors": [str, str, str], \
"why_wrong": [str, str, str]}}

Hard rules:
- "evidence" MUST be copied character-for-character from the excerpt, 10-300 \
characters, and must contain the fact the answer states. Do not paraphrase it.
- "answer" must be a short factual span: a value, a term, a procedure name, a \
condition. At most 12 words. Never a sentence of explanation.
- The question must be self-contained and answerable by a telecom engineer who \
cannot see this excerpt. Never write "according to the excerpt", "in this \
section", "the above table", or refer to the document at all.
- The question must name its own subject explicitly, including the spec's \
technology and the specific procedure, field or parameter, so it is unambiguous \
on its own.
- "distractors" are three WRONG answers that a knowledgeable reader could \
plausibly confuse with the correct one: an adjacent value, a sibling procedure, \
the same field in a different state. Never off-topic, never obviously absurd.
- "why_wrong" gives, for each distractor in the same order, one clause saying \
what it actually is or why it does not apply here.
- Skip boilerplate: copyright pages, change histories, document scope, contact \
addresses, reference lists. If the excerpt is entirely boilerplate, output [].

JSON array only:"""

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


def well_formed(item) -> bool:
    if not isinstance(item, dict):
        return False
    for key in ("question", "answer", "evidence"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            return False
    for key in ("distractors", "why_wrong"):
        v = item.get(key)
        if not isinstance(v, list) or len(v) != 3:
            return False
        if not all(isinstance(x, str) and x.strip() for x in v):
            return False
    return len(item["answer"].split()) <= 12 and 10 <= len(item["evidence"]) <= 400


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("-k", type=int, default=4, help="items per chunk")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-chunk-chars", type=int, default=9000)
    ap.add_argument("--downweight-keep", type=int, default=60,
                    help="max chunks kept from each noisy spec")
    ap.add_argument("--per-spec-cap", type=int, default=400)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=1400)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    done: set[str] = set()
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["chunk_id"])
            except Exception:
                pass
        print(f"resume: {len(done)} chunks already generated", flush=True)

    import hashlib

    pool, skipped = [], collections.Counter()
    for line in args.chunks.open(encoding="utf-8"):
        r = json.loads(line)
        if r["chunk_id"] in done:
            continue
        if str(r.get("series")) in EXCLUDE_SERIES:
            skipped["release_description_series"] += 1
            continue
        if FRONT_MATTER.search(" ".join(r.get("headings") or [])):
            skipped["front_matter"] += 1
            continue
        if r["text"].count(".....") > 3:
            skipped["table_of_contents"] += 1
            continue
        pool.append(r)

    # Shuffle before the per-spec cap. The chunk file is ordered by the
    # errors_to_specs ranking, so taking a prefix samples one document's opening
    # pages rather than the corpus; a deterministic hash order fixes that
    # without losing reproducibility.
    pool.sort(key=lambda r: hashlib.sha256(
        ("grounded-v1|" + r["chunk_id"]).encode()).hexdigest())

    rows, per_spec = [], {}
    for r in pool:
        spec = r.get("spec", "?")
        cap = args.downweight_keep if spec in DOWNWEIGHT_SPECS else args.per_spec_cap
        if per_spec.get(spec, 0) >= cap:
            continue
        per_spec[spec] = per_spec.get(spec, 0) + 1
        rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"generating over {len(rows)} chunks from {len(per_spec)} specs "
          f"(skipped {dict(skipped)})", flush=True)
    if not rows:
        return

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(
                spec=r.get("spec", "?"), release=r.get("release", "?"),
                text=r["text"][:args.max_chunk_chars], k=args.k)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for r in rows]

    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
              max_model_len=8192, enforce_eager=False)
    sp = SamplingParams(temperature=0.7, top_p=0.95, top_k=20,
                        max_tokens=args.max_tokens, seed=42)
    outs = llm.generate(prompts, sp)

    kept = dropped = empty = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        for r, out in zip(rows, outs):
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
                    "item_id": f"{r['chunk_id']}_{j}",
                    "chunk_id": r["chunk_id"],
                    "spec": r.get("spec"), "release": r.get("release"),
                    "source": r.get("source"),
                    "question": it["question"].strip(),
                    "answer": it["answer"].strip(),
                    "evidence": it["evidence"].strip(),
                    "distractors": [d.strip() for d in it["distractors"]],
                    "why_wrong": [w.strip() for w in it["why_wrong"]],
                }, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept={kept}  malformed_or_unparsable={dropped}  "
          f"chunks_returning_empty={empty}", flush=True)


if __name__ == "__main__":
    main()
