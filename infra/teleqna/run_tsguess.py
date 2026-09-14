#!/usr/bin/env python3
"""TS-Guessing: the one teleqna probe that is evidence rather than inference.

Every accuracy-based contamination test has the same weakness — a model that
truly knows telecom and a model that memorised a public file both score high,
and both lose points when you perturb the question. The comparison is
suggestive, never conclusive.

This test removes the ambiguity by asking the model to reproduce text it could
not possibly derive:

  mask=distractor   blank out one WRONG option and ask the model to fill it in.
                    A distractor is arbitrary text invented by TeleQnA's
                    generator. It is not a fact about telecommunications. No
                    amount of domain knowledge lets a model reconstruct it.
                    An exact match is only explainable one way: the model has
                    seen this row.

  mask=gold         blank out the CORRECT option instead. Reported for
                    comparison only. A knowledgeable model can partly
                    reconstruct a gold answer from the question, so a high score
                    here is ambiguous — which is precisely why the distractor
                    arm is the one that decides.

A decision rule was fixed before the first run — "<2% clean, 2-10% partial
exposure, >10% compromised" — on the assumption that reproducing a wrong option
is only explainable by recall. The first run scored 4.84% and the length
breakdown falsified that assumption: 15.87% at one word, 1.55% at 7-12 words.
Monotone decay with length is the signature of guessing; recall would be flat or
rising, because longer strings are more distinctive.

A bare rate therefore decides nothing, and this function no longer pretends
otherwise. The rate is only interpretable against `make_selfgen_control.py`,
which measures what guessing alone scores on options that are plausible for the
question and certainly absent from any corpus, at a matched length distribution.
Compare per length band; the verdict belongs to that comparison, not to a
threshold on this number alone.

Scored on normalised text: exact match, token F1, and the length-controlled
random-overlap floor (what F1 you would get by emitting the question's own
tokens), so a high F1 that is really just topical vocabulary cannot be mistaken
for recall.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL = os.environ.get("TELEQNA_MODEL", "Qwen/Qwen3-8B")
ENDPOINT = os.environ.get(
    "VLLM_CHAT_URL", "http://telelogs-bench4-vllm:8000/v1/chat/completions"
)
TOKEN = re.compile(r"[a-z0-9]+")
PRINT_LOCK = threading.Lock()

PROMPT = """You are given a multiple-choice question from a benchmark dataset. One of the options has been replaced by [MASK].

Reproduce the exact text that was removed. Output only that text, with no label, no quotes, and no explanation.

Question: {question}

{options}"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokens(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


def normalize(text: str) -> str:
    return " ".join(tokens(text))


def token_f1(predicted: str, reference: str) -> float:
    p, r = Counter(tokens(predicted)), Counter(tokens(reference))
    if not p or not r:
        return 0.0
    common = sum((p & r).values())
    if not common:
        return 0.0
    precision, recall = common / sum(p.values()), common / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def mask_index(row: dict[str, Any], mask: str) -> int:
    gold = int(row["answer"])
    if mask == "gold":
        return gold
    others = [i for i in range(len(row["choices"])) if i != gold]
    digest = hashlib.sha256(("tsguess:" + row["sample_id"]).encode()).digest()
    return others[digest[0] % len(others)]


def build_prompt(row: dict[str, Any], target: int) -> str:
    options = "\n".join(
        f"{chr(65 + i)}) {'[MASK]' if i == target else c}"
        for i, c in enumerate(row["choices"])
    )
    return PROMPT.format(question=row["question"], options=options)


def request_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    target = mask_index(row, args.mask)
    reference = row["choices"][target]
    prompt = build_prompt(row, target)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.monotonic()
    last_error = ""
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
            predicted = (raw["choices"][0]["message"].get("content") or "").strip()
            predicted = predicted.strip('"').strip()
            return {
                "sample_id": row["sample_id"],
                "subject": row.get("subject"),
                "mask": args.mask,
                "masked_index": target,
                "reference": reference,
                "predicted": predicted,
                "exact_match": normalize(predicted) == normalize(reference),
                "token_f1": round(token_f1(predicted, reference), 4),
                "question_f1": round(token_f1(row["question"], reference), 4),
                "usage": raw.get("usage", {}),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "completed_at": now_iso(),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - retried, then recorded
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                time.sleep(2 * attempt)
    return {
        "sample_id": row["sample_id"],
        "subject": row.get("subject"),
        "mask": args.mask,
        "masked_index": target,
        "reference": reference,
        "predicted": "",
        "exact_match": False,
        "token_f1": 0.0,
        "question_f1": 0.0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "completed_at": now_iso(),
        "error": last_error,
    }


def verdict(exact_rate: float) -> str:
    """Deliberately does not return a contamination call from a bare rate.

    See the module docstring: the original threshold was invalidated by the
    length breakdown. Anything above the guessing floor measured by
    make_selfgen_control.py is what recall has to account for, and nothing here
    knows that number.
    """
    if exact_rate == 0.0:
        return "0% — no reconstruction at all; compare against the self-generated control anyway"
    return (
        f"{exact_rate:.2%} raw — NOT a verdict. Compare per length band against "
        "results/tsguess_selfgen (the guessing upper bound). Only the excess "
        "over that control is attributable to memorisation."
    )


def summarize(records: list[dict[str, Any]], mask: str) -> dict[str, Any]:
    total = len(records)
    exact = sum(1 for r in records if r["exact_match"])
    by_subject: dict[str, list[int]] = {}
    for record in records:
        bucket = by_subject.setdefault(record.get("subject") or "<none>", [0, 0])
        bucket[1] += 1
        bucket[0] += 1 if record["exact_match"] else 0
    rate = exact / total if total else 0.0
    return {
        "mask": mask,
        "total": total,
        "exact_match": exact,
        "exact_match_rate": round(rate, 4),
        "mean_token_f1": round(sum(r["token_f1"] for r in records) / max(1, total), 4),
        "mean_question_overlap_floor": round(
            sum(r["question_f1"] for r in records) / max(1, total), 4
        ),
        "high_f1_rate_ge_0.8": round(
            sum(1 for r in records if r["token_f1"] >= 0.8) / max(1, total), 4
        ),
        "request_errors": sum(1 for r in records if r.get("error")),
        "exact_match_by_subject": {
            k: {"exact": v[0], "total": v[1], "rate": round(v[0] / v[1], 4)}
            for k, v in sorted(by_subject.items())
        },
        "verdict": verdict(rate) if mask == "distractor" else
        "comparison arm only — knowledge can reconstruct a gold answer",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mask", choices=["distractor", "gold"], default="distractor")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    rows = [r for r in rows if len(r["choices"]) > 2]
    if args.limit:
        rows = rows[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.open(encoding="utf-8"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record.get("error"):
                done[record["sample_id"]] = record
    pending = [r for r in rows if r["sample_id"] not in done]

    print(json.dumps({"stage": "start", "total": len(rows), "pending": len(pending),
                      "mask": args.mask, "model": MODEL}), flush=True)
    print(json.dumps({"stage": "prompt_example",
                      "prompt": build_prompt(rows[0], mask_index(rows[0], args.mask))}),
          flush=True)

    completed = len(done)
    with results_path.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(request_one, r, args) for r in pending]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                done[record["sample_id"]] = record
                completed += 1
                if completed % 50 == 0 or completed == len(rows):
                    running = summarize(list(done.values()), args.mask)
                    with PRINT_LOCK:
                        print(json.dumps({"stage": "progress", "completed": completed,
                                          "exact_match_rate": running["exact_match_rate"]}),
                              flush=True)

    records = [done[r["sample_id"]] for r in rows if r["sample_id"] in done]
    summary = summarize(records, args.mask)
    top = sorted(records, key=lambda r: -r["token_f1"])[:10]
    summary["closest_reconstructions"] = [
        {"sample_id": r["sample_id"], "f1": r["token_f1"],
         "reference": r["reference"][:160], "predicted": r["predicted"][:160]}
        for r in top
    ]
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
