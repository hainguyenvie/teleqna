#!/usr/bin/env python3
"""Build the memorisation probe set for teleqna.

TeleQnA has been public on GitHub since October 2023. Every model on the GSMA
leaderboard was pretrained after that. So before any number from this track can
be trusted, one question has to be answered: is the score telecom knowledge, or
is it recall of a public file?

Accuracy alone cannot answer it — a memorising model and a knowing model both
score high. What separates them is *which surface changes hurt*. This script
builds four variants of the same stratified sample so the arms are paired
row-for-row, which makes McNemar's test applicable and roughly triples the
statistical power over comparing two independent runs.

  orig        untouched rows. The control. Re-run so it shares the sample,
              the endpoint and the decoding settings with everything else.

  perm        choices deterministically rotated. Catches recall of the answer
              *letter* or of the option ordering. Weak on its own: a model that
              memorised the gold *text* survives this untouched.

  distract    gold text kept verbatim, distractors replaced by distractors
              sampled from topically-near rows in the same subject. Read this
              one carefully — swapping in foreign distractors usually makes a
              question *easier*, so accuracy is expected to rise. It is a
              control, not evidence: it shows the model can still answer when
              the option set is one it has never seen.

  paraphrase  stem rewritten by the locally-served model, options untouched,
              gold unchanged. The paraphraser never sees the options or the
              answer, so it cannot leak. Catches dependence on the exact
              question string. A meaning-preserving rewrite costs a knowing
              model nothing and costs a memorising model a lot.

The fifth arm — the one that is actual evidence rather than inference — lives in
`run_tsguess.py`, because it needs a different prompt and a different scorer.

Everything here is deterministic given --seed, except the paraphrase calls,
which are temperature-sampled and validated; rows whose paraphrase fails
validation after --paraphrase-retries attempts are dropped from that arm only
and recorded in dropped_paraphrase.jsonl. No arm ever silently shrinks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

MODEL = os.environ.get("TELEQNA_MODEL", "Qwen/Qwen3-8B")
ENDPOINT = os.environ.get(
    "VLLM_CHAT_URL", "http://telelogs-bench4-vllm:8000/v1/chat/completions"
)

BRACKET_TAG = re.compile(r"\[[^\]]+\]")
ACRONYM = re.compile(r"\b[A-Z][A-Z0-9\-]{2,}\b")
TOKEN = re.compile(r"[a-z0-9]+")

PARAPHRASE_PROMPT = """Rewrite the following telecommunications exam question so that it asks for exactly the same information using different wording.

Rules:
- Keep every technical term, acronym, number, and bracketed tag exactly as written.
- Do not answer the question. Do not add or remove any requirement.
- Do not add explanations, quotes, or labels. Output only the rewritten question.

Question: {question}"""


def normalize(text: str) -> str:
    return " ".join(TOKEN.findall((text or "").lower()))


def rng_for(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def stratified_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    """Proportional allocation by subject, deterministic given seed."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row.get("subject") or "<none>"].append(row)
    total = len(rows)
    picked: list[dict] = []
    for subject in sorted(by_subject):
        pool = by_subject[subject]
        take = max(1, round(size * len(pool) / total))
        picked.extend(rng_for("sample", subject, str(seed)).sample(pool, min(take, len(pool))))
    picked.sort(key=lambda r: r["sample_index"])
    return picked


def rotate(row: dict) -> dict:
    choices = list(row["choices"])
    count = len(choices)
    shift = hashlib.sha256(row["sample_id"].encode()).digest()[0] % count
    order = [(i + shift) % count for i in range(count)]
    return {
        **row,
        "choices": [choices[j] for j in order],
        "answer": order.index(int(row["answer"])),
        "variant": "perm",
    }


def swap_distractors(row: dict, pool_by_subject: dict[str, list[str]]) -> dict | None:
    """Keep the gold verbatim, draw fresh distractors from the same subject.

    Foreign distractors are rejected when they overlap the gold too much, which
    would create a second defensible answer, and when they duplicate one another.
    """
    gold_text = row["choices"][int(row["answer"])]
    gold_tokens = set(normalize(gold_text).split())
    pool = pool_by_subject.get(row.get("subject") or "<none>", [])
    if len(pool) < 50:
        return None
    rng = rng_for("distract", row["sample_id"])
    need = len(row["choices"]) - 1
    chosen: list[str] = []
    seen = {normalize(gold_text)}
    for candidate in rng.sample(pool, min(len(pool), 400)):
        key = normalize(candidate)
        if not key or key in seen:
            continue
        candidate_tokens = set(key.split())
        overlap = len(candidate_tokens & gold_tokens) / max(1, len(candidate_tokens | gold_tokens))
        if overlap > 0.5:
            continue
        chosen.append(candidate)
        seen.add(key)
        if len(chosen) == need:
            break
    if len(chosen) < need:
        return None
    position = rng.randrange(len(row["choices"]))
    choices = chosen[:position] + [gold_text] + chosen[position:]
    return {**row, "choices": choices, "answer": position, "variant": "distract"}


def call_model(prompt: str, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if temperature > 0:
        payload.update(top_p=0.95, top_k=20)
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
                raise
            time.sleep(2 * attempt)
    return ""


def paraphrase_ok(original: str, rewritten: str) -> tuple[bool, str]:
    """A paraphrase that drops technical content is a different question."""
    if not rewritten:
        return False, "empty"
    if "\n" in rewritten.strip():
        return False, "multiline"
    if normalize(rewritten) == normalize(original):
        return False, "identical"
    ratio = len(rewritten) / max(1, len(original))
    if not 0.5 <= ratio <= 2.0:
        return False, f"length_ratio={ratio:.2f}"
    for tag in BRACKET_TAG.findall(original):
        if tag not in rewritten:
            return False, f"lost_tag={tag}"
    lost = {a for a in ACRONYM.findall(original)} - {a for a in ACRONYM.findall(rewritten)}
    if lost:
        return False, f"lost_acronym={sorted(lost)[:3]}"
    if re.search(r"(?i)\b(answer|option [A-E1-5]|correct)\b", rewritten):
        return False, "leaks_answer_language"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--paraphrase-retries", type=int, default=3)
    parser.add_argument("--no-paraphrase", action="store_true",
                        help="build the offline arms only; no endpoint needed")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    sample = stratified_sample(rows, args.size, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, records: list[dict]) -> None:
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"arm": name, "rows": len(records), "path": str(path)}), flush=True)

    dump("orig", [{**r, "variant": "orig"} for r in sample])
    dump("perm", [rotate(r) for r in sample])

    pool_by_subject: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        gold = int(row["answer"])
        for index, choice in enumerate(row["choices"]):
            if index != gold:
                pool_by_subject[row.get("subject") or "<none>"].append(choice)
    distract = [swap_distractors(r, pool_by_subject) for r in sample]
    dropped_distract = [r["sample_id"] for r, d in zip(sample, distract) if d is None]
    dump("distract", [d for d in distract if d])
    if dropped_distract:
        print(json.dumps({"arm": "distract", "dropped": len(dropped_distract),
                          "reason": "no clean foreign distractors"}), flush=True)

    if args.no_paraphrase:
        return

    paraphrased: list[dict] = []
    dropped: list[dict] = []
    for index, row in enumerate(sample, 1):
        best, reason = "", "not_attempted"
        for attempt in range(args.paraphrase_retries):
            candidate = call_model(
                PARAPHRASE_PROMPT.format(question=row["question"]),
                temperature=0.3 + 0.2 * attempt,
                max_tokens=256,
            )
            candidate = candidate.strip().strip('"').strip()
            ok, reason = paraphrase_ok(row["question"], candidate)
            if ok:
                best = candidate
                break
        if best:
            paraphrased.append({**row, "question": best, "original_question": row["question"],
                                "variant": "paraphrase"})
        else:
            dropped.append({"sample_id": row["sample_id"], "reason": reason,
                            "question": row["question"]})
        if index % 50 == 0:
            print(json.dumps({"stage": "paraphrase", "done": index, "kept": len(paraphrased)}),
                  flush=True)
    dump("paraphrase", paraphrased)
    with (args.out / "dropped_paraphrase.jsonl").open("w", encoding="utf-8") as handle:
        for record in dropped:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"arm": "paraphrase", "kept": len(paraphrased), "dropped": len(dropped)}),
          flush=True)


if __name__ == "__main__":
    main()
