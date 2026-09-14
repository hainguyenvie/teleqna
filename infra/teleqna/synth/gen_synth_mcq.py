#!/usr/bin/env python3
"""Generate teleqna-style MCQs from spec chunks via the DeepSeek API.

Privacy boundary, stated once and enforced by construction: every prompt this
script sends contains ONLY public 3GPP spec text and our own instructions.
No teleqna test row, in whole or in part, ever reaches a third-party API —
the contamination direction is handled downstream by gate_synth.py, which
compares generated questions against the test set locally.

Half the questions per chunk are "hard_negative" mode: the distractor probe
showed +15.8pp when the original distractors are replaced with foreign ones,
i.e. the baseline fails on fine option discrimination, not on missing
knowledge. Distractors that differ from the gold by one release number, one
adjacent value, or one sibling procedure name train exactly that.

Each question must carry a verbatim `evidence` quote from the chunk; the
downstream gate rejects the question if the quote is not actually a substring
of the chunk. That turns groundedness from a judged property into a checked
one.

Resumable: chunk_ids already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

API = os.environ.get("SYNTH_API_URL",
                     "https://api.deepseek.com/chat/completions")

PROMPT = """You write exam questions for a telecom-standards benchmark. Below is an \
excerpt of 3GPP TS {spec} (Release {release}), sections: {headings}.

<spec_excerpt>
{text}
</spec_excerpt>

Write exactly {k} multiple-choice questions that test knowledge stated in this \
excerpt. Output a JSON array only, no prose. Each element:
{{"question": str, "choices": [str, ...], "answer": int, "evidence": str, "mode": str}}

Rules for every question:
- The question must be answerable by a telecom expert who has NOT seen this \
excerpt: never write "according to this document/section/passage", never rely \
on the excerpt's layout. End the question text with " [3GPP Release {release}]".
- 4 or 5 choices, exactly one correct (index `answer`, 0-based).
- `evidence` is a VERBATIM quote of 5-30 consecutive words copied from the \
excerpt that proves the correct choice. Do not paraphrase it.
- Every distractor must be false per the excerpt, but plausible in-domain and \
of the same kind as the correct choice (a value vs values, a procedure vs \
procedures).

{mode_blocks}
Do not ask about section numbering, figure numbers, or editorial matters."""

# Literature-side counterpart (make_tcc_chunks.py chunks: doc_id/collection/
# title instead of spec/release). No "[3GPP Release N]" tag — the real test's
# Research-publications/overview rows are 99.9% untagged (README.md provenance
# table), so a literature-sourced question should look like them: bare claim,
# no citation of "this paper" or the excerpt's structure.
PROMPT_LIT = """You write exam questions for a telecom-research benchmark. Below is an \
excerpt of a peer-reviewed telecom research article ("{title}"), sections: {headings}.

<article_excerpt>
{text}
</article_excerpt>

Write exactly {k} multiple-choice questions that test knowledge/claims stated in \
this excerpt. Output a JSON array only, no prose. Each element:
{{"question": str, "choices": [str, ...], "answer": int, "evidence": str, "mode": str}}

Rules for every question:
- The question must be answerable by a telecom expert who has NOT seen this \
paper: never write "according to this paper/article/study/excerpt", never say \
"the authors", never rely on the excerpt's layout. State the claim or finding \
directly as a general technical question.
- 4 or 5 choices, exactly one correct (index `answer`, 0-based).
- `evidence` is a VERBATIM quote of 5-30 consecutive words copied from the \
excerpt that proves the correct choice. Do not paraphrase it.
- Every distractor must be false per the excerpt, but plausible in-domain and \
of the same kind as the correct choice (a mechanism vs mechanisms, a result vs \
results, a technique vs techniques).

{mode_blocks}
Do not ask about figure numbers, table numbers, section numbering, author \
names, affiliations, funding, or other editorial/bibliographic matters — only \
the technical content."""

HARD_BLOCK = """
{n} of the questions use mode "hard_negative": each distractor is a MINIMAL \
corruption of the truth — an adjacent release number, a neighbouring numeric \
value or unit, a sibling message/procedure/IE name that also exists in 5G, or a \
near-synonym term with a different technical meaning. Someone with shallow \
knowledge should find all options equally credible."""

STD_BLOCK = """
{n} of the questions use mode "standard": ordinary exam distractors."""

SHAPE_BLOCK = """
{n} of the questions use mode "shape_prior". Each must have one choice that is \
the literal string "All of the above." or "None of the above." (mix both across \
these {n}; if only one, pick either). Rules:
- "All of the above.": write it as the correct answer in about 90% of these \
questions — construct the other choices as several independently true facts from \
the excerpt (e.g. several correct parameter values, several correct conditions), \
so that "All of the above." is genuinely correct. In the rest, make exactly one \
of the other choices false so "All of the above." is a wrong choice.
- "None of the above.": write it as a WRONG choice in virtually every case — the \
excerpt states a clear correct answer, put it among the other choices, and \
"None of the above." is there as a plausible-looking trap.
- `evidence` must still be a verbatim 5-30 word quote proving the correct choice \
(for a correct "All of the above.", quote the sentence enumerating the facts; for \
a wrong "None of the above.", quote the sentence stating the real correct fact)."""


def call(payload: dict, key: str, retries: int = 5) -> str:
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(8 * (attempt + 1))
            sys.stderr.write(f"  retry {attempt + 1}: {exc}\n")
    raise RuntimeError("unreachable")


def parse_json_array(text: str):
    """Last valid JSON array in the text. Robust to thinking preambles and
    markdown fences: tries every '[' from the end with raw_decode."""
    dec = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r"\[", text)]
    for i in reversed(starts):
        try:
            val, _ = dec.raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(val, list) and val and all(isinstance(x, dict) for x in val):
            return val
    # Bare object stream: at temperature the model sometimes forgets the
    # enclosing brackets entirely (finish_reason=stop, well-formed objects,
    # no array). Walk the text collecting every top-level dict.
    items = []
    i = text.find("{")
    while i != -1:
        try:
            val, end = dec.raw_decode(text[i:])
        except Exception:
            break
        if not isinstance(val, dict):
            break
        items.append(val)
        i = text.find("{", i + end)
    return items or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--per-chunk", type=int, default=6)
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--shape-frac", type=float, default=0.0,
                    help="fraction of per-chunk questions forced into "
                         "'All/None of the above' shape (mode=shape_prior); "
                         "see SHAPE_BLOCK")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    # Self-hosted vLLM (SYNTH_API_URL points at the cluster service) needs no
    # key; the DeepSeek path does.
    key = os.environ.get("DEEPSEEK_API_KEY", "local")
    if "deepseek.com" in API and key == "local":
        sys.exit("DEEPSEEK_API_KEY is not set")

    done = set()
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["chunk_id"])
            except Exception:
                pass

    chunks = [json.loads(l) for l in args.chunks.open(encoding="utf-8")]
    todo = [c for c in chunks if c["chunk_id"] not in done][: args.limit]
    print(f"{len(todo)} chunks to generate ({len(done)} already done)",
          file=sys.stderr, flush=True)

    k_hard = round(args.per_chunk * args.hard_frac)
    k_shape = round(args.per_chunk * args.shape_frac)
    k_std = args.per_chunk - k_hard - k_shape
    mode_blocks = "\n".join(
        blk.format(n=n) for blk, n in
        ((HARD_BLOCK, k_hard), (STD_BLOCK, k_std), (SHAPE_BLOCK, k_shape))
        if n > 0)
    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0, "questions": 0}
    out = args.out.open("a", encoding="utf-8")

    def work(chunk: dict) -> None:
        literature = "doc_id" in chunk
        if literature:
            prompt = PROMPT_LIT.format(
                title=chunk["title"],
                headings="; ".join(chunk["headings"][:6]) or "(untitled)",
                text=chunk["text"], k=args.per_chunk, mode_blocks=mode_blocks)
        else:
            prompt = PROMPT.format(
                spec=chunk["spec"], release=chunk["release"],
                headings="; ".join(chunk["headings"][:6]) or "(untitled)",
                text=chunk["text"], k=args.per_chunk, mode_blocks=mode_blocks)
        try:
            payload = {"model": args.model, "temperature": args.temperature,
                       # 6 questions with 5 long choices + evidence regularly
                       # blow past 4k tokens; truncation was 38% of pilot fails
                       "max_tokens": 8000,
                       "messages": [{"role": "user", "content": prompt}]}
            if os.environ.get("SYNTH_NO_THINK", "1") == "1":
                # Qwen3-family soft switch, honoured by the vLLM chat template;
                # unknown fields are ignored by other providers.
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            content = call(payload, key)
            items = parse_json_array(content)
            assert isinstance(items, list) and items, "no JSON array"
        except Exception as exc:
            with lock:
                counters["fail"] += 1
                sys.stderr.write(f"  {chunk['chunk_id']} failed: {exc}\n")
                with args.out.with_suffix(".failures.jsonl").open("a") as fh:
                    fh.write(json.dumps({"chunk_id": chunk["chunk_id"],
                                         "error": str(exc),
                                         "content": locals().get("content", "")[-1500:]})
                             + "\n")
            return
        if literature:
            rec = {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                   "collection": chunk["collection"], "title": chunk["title"],
                   "tier": chunk["tier"], "model": args.model, "items": items}
        else:
            rec = {"chunk_id": chunk["chunk_id"], "spec": chunk["spec"],
                   "release": chunk["release"], "source": chunk["source"],
                   "tier": chunk["tier"], "model": args.model, "items": items}
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            counters["ok"] += 1
            counters["questions"] += len(items)
            if counters["ok"] % 20 == 0:
                print(f"  ...{counters['ok']}/{len(todo)} chunks, "
                      f"{counters['questions']} questions",
                      file=sys.stderr, flush=True)

    threads: list[threading.Thread] = []
    sem = threading.Semaphore(args.workers)

    def guarded(c):
        with sem:
            work(c)

    for c in todo:
        t = threading.Thread(target=guarded, args=(c,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print(json.dumps(counters))


if __name__ == "__main__":
    main()
