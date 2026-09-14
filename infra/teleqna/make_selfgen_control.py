#!/usr/bin/env python3
"""A valid guessability control for the TS-Guessing probe.

The first control attempt was wrong and is kept in the record: masking a
distractor transplanted from a different question scored 0/599, but that
removed guessability along with memorisation — a transplanted distractor is not
a plausible answer to the question it was pasted into, so nothing about the
context points at it. It measured "can the model reproduce unrelated text",
which is trivially no.

A valid control needs text that is simultaneously:
  (a) a plausible wrong answer to *this* question, so context genuinely
      constrains it, and
  (b) certainly absent from every training corpus, so recall is impossible.

Only one source satisfies both: distractors written fresh, now, by the model
itself. This script generates one per row, matched to the length of the
distractor it replaces, and writes a probe file where the mask target is the
generated option.

Read the result as an UPPER BOUND on guessability, not an estimate. A model
reproducing its own recent output is easier than reproducing a third party's,
and the generator saw the gold answer while TeleQnA's did too — both push this
control's score up. So:

    real probe rate  <=  this control rate   =>  the observed exact matches
                                                 need no memorisation to explain
    real probe rate  >   this control rate   =>  the excess is the part that
                                                 recall has to account for

Compared per length band, because the whole reason this control exists is that
the raw rate is dominated by one- and two-word options.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

MODEL = os.environ.get("TELEQNA_MODEL", "Qwen/Qwen3-8B")
ENDPOINT = os.environ.get(
    "VLLM_CHAT_URL", "http://telelogs-bench4-vllm:8000/v1/chat/completions"
)
TOKEN = re.compile(r"[a-z0-9]+")

PROMPT = """Write one additional answer option for this telecommunications exam question.

The option must be:
- plausible enough that a non-expert might pick it
- factually WRONG
- about {words} word(s) long
- different from every option already listed

Question: {question}

Existing options:
{options}

Output only the new option text. No label, no quotes, no explanation."""


def normalize(text: str) -> str:
    return " ".join(TOKEN.findall((text or "").lower()))


def target_index(row: dict[str, Any]) -> int:
    """Same choice of victim distractor as run_tsguess, so lengths line up."""
    gold = int(row["answer"])
    others = [i for i in range(len(row["choices"])) if i != gold]
    digest = hashlib.sha256(("tsguess:" + row["sample_id"]).encode()).digest()
    return others[digest[0] % len(others)]


def call_model(prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "top_p": 0.95,
        "top_k": 20,
    }
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = json.load(response)
            return (raw["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            if attempt == 3:
                return ""
            time.sleep(2 * attempt)
    return ""


def build_one(row: dict[str, Any], length_tolerance: float) -> dict[str, Any] | None:
    victim = target_index(row)
    reference = row["choices"][victim]
    want = max(1, len(reference.split()))
    others = "\n".join(
        f"{chr(65 + i)}) {c}" for i, c in enumerate(row["choices"]) if i != victim
    )
    existing = {normalize(c) for c in row["choices"]}
    for attempt in range(3):
        candidate = call_model(
            PROMPT.format(words=want, question=row["question"], options=others),
            temperature=0.7 + 0.1 * attempt,
            max_tokens=120,
        )
        candidate = candidate.strip().strip('"').strip().split("\n")[0].strip()
        if not candidate or normalize(candidate) in existing:
            continue
        ratio = len(candidate.split()) / want
        if not (1 / (1 + length_tolerance)) <= ratio <= (1 + length_tolerance):
            continue
        choices = list(row["choices"])
        choices[victim] = candidate
        return {
            **row,
            "choices": choices,
            "variant": "selfgen",
            "selfgen_index": victim,
            "replaced_reference": reference,
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="probe orig.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="output jsonl")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--length-tolerance", type=float, default=0.6)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    rows = [r for r in rows if len(r["choices"]) > 2]
    print(json.dumps({"stage": "start", "rows": len(rows), "model": MODEL}), flush=True)

    built: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(build_one, r, args.length_tolerance) for r in rows]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            if record:
                built.append(record)
            if index % 100 == 0:
                print(json.dumps({"stage": "progress", "done": index, "kept": len(built)}),
                      flush=True)

    built.sort(key=lambda r: r["sample_index"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in built:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # The mask target must be the generated option, so run_tsguess's own
    # deterministic victim choice has to land on it. It does: build_one used the
    # identical formula. Assert it rather than trust it.
    mismatched = sum(1 for r in built if target_index(r) != r["selfgen_index"])
    print(json.dumps({"stage": "done", "kept": len(built), "dropped": len(rows) - len(built),
                      "mask_target_mismatches": mismatched, "path": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
