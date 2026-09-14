#!/usr/bin/env python3
"""Profile the GSMA ot-full teleqna split before any modelling decision.

Reads the parquet exactly as the GSMA harness does (pyarrow, row order preserved)
and reports the things that decide how a specialist is built: answer-index
balance overall and per subject, choice-count distribution, question and choice
lengths, duplicates, and the two priors that burned the srsranbench track
(position and choice length).

teleqna differs from srsranbench in three ways that matter:
  * 5 subjects with very different sizes and difficulty
  * variable choice count (the original TeleQnA has 3-5 options per question)
  * a "[3GPP Release NN]" suffix on standards questions, which is a free feature

Also writes a readable JSONL mirror next to the parquet so the rest of the
tooling never has to open a parquet file again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = ROOT / "data/teleqna/test-00000-of-00001.parquet"
DEFAULT_JSONL = ROOT / "data/teleqna/test.jsonl"

RELEASE_TAG = re.compile(r"\[3GPP\s+Release\s+(\d+)\]", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

# Words that a distractor-generation process tends to over-produce. Checking
# these is how "none of the above"-style shortcuts get caught before training.
TELL_PATTERNS = {
    "all of the above": re.compile(r"(?i)^all of the above"),
    "none of the above": re.compile(r"(?i)^none of the above"),
    "both a and b": re.compile(r"(?i)^both "),
    "not mentioned": re.compile(r"(?i)not (mentioned|specified|provided|discussed)"),
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def percentiles(values: list[int], points=(0, 25, 50, 75, 90, 99, 100)) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    out = {}
    for p in points:
        index = min(len(ordered) - 1, max(0, round((p / 100) * (len(ordered) - 1))))
        out[f"p{p}"] = ordered[index]
    return out


def index_distribution(rows: list[dict]) -> dict:
    """Answer-index counts plus the two constants any score has to beat."""
    answers = Counter(int(r["answer"]) for r in rows)
    counts = Counter(len(r["choices"]) for r in rows)
    n = len(rows)
    # score of a fixed-letter policy, for every letter that ever exists
    fixed = {}
    for letter_index in range(max(counts) if counts else 0):
        fixed[chr(65 + letter_index)] = round(answers.get(letter_index, 0) / n, 4)
    return {
        "n": n,
        "answer_index_counts": {str(k): answers[k] for k in sorted(answers)},
        "answer_index_share": {str(k): round(answers[k] / n, 4) for k in sorted(answers)},
        "best_fixed_letter": max(fixed.items(), key=lambda kv: kv[1]) if fixed else None,
        "fixed_letter_scores": fixed,
        "random_baseline": round(sum(1 / len(r["choices"]) for r in rows) / n, 4),
        "choice_count_distribution": {str(k): counts[k] for k in sorted(counts)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--samples", type=int, default=0, help="print N full examples")
    args = parser.parse_args()

    digest = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    rows = pq.read_table(args.parquet).to_pylist()

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            record = {
                "sample_id": f"teleqna-{index:05d}",
                "sample_index": index,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),
                "subject": row.get("subject"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row.get("subject") or "<none>"].append(row)

    q_lengths = [len(r["question"]) for r in rows]
    c_lengths = [len(c) for r in rows for c in r["choices"]]

    gold_len, distractor_len = [], []
    gold_longest, gold_shortest = 0, 0
    for row in rows:
        lengths = [len(c) for c in row["choices"]]
        gold = int(row["answer"])
        for position, length in enumerate(lengths):
            (gold_len if position == gold else distractor_len).append(length)
        if lengths[gold] == max(lengths):
            gold_longest += 1
        if lengths[gold] == min(lengths):
            gold_shortest += 1

    seen: dict[str, int] = {}
    duplicate_questions = 0
    for row in rows:
        key = normalize(row["question"])
        if key in seen:
            duplicate_questions += 1
        else:
            seen[key] = 1

    exact_rows = Counter(
        normalize(r["question"]) + "||" + "||".join(normalize(c) for c in r["choices"])
        for r in rows
    )
    exact_duplicate_rows = sum(count - 1 for count in exact_rows.values() if count > 1)

    # contradictory gold on an identical stem+choices key
    gold_by_key: dict[str, set] = defaultdict(set)
    for row in rows:
        key = normalize(row["question"]) + "||" + "||".join(normalize(c) for c in row["choices"])
        gold_by_key[key].add(int(row["answer"]))
    contradictory = sum(1 for golds in gold_by_key.values() if len(golds) > 1)

    releases = Counter()
    tagged = 0
    for row in rows:
        match = RELEASE_TAG.search(row["question"])
        if match:
            tagged += 1
            releases[match.group(1)] += 1

    tells = {}
    for name, pattern in TELL_PATTERNS.items():
        present = 0
        is_gold = 0
        for row in rows:
            hits = [position for position, c in enumerate(row["choices"]) if pattern.search(c)]
            if hits:
                present += 1
                if int(row["answer"]) in hits:
                    is_gold += 1
        tells[name] = {
            "rows_with_it": present,
            "rows_where_it_is_gold": is_gold,
            "gold_rate": round(is_gold / present, 4) if present else None,
        }

    vocab = Counter()
    for row in rows:
        vocab.update(set(w.lower() for w in WORD.findall(row["question"])))

    report = {
        "parquet": str(args.parquet),
        "sha256": digest,
        "rows": len(rows),
        "jsonl": str(args.jsonl),
        "overall": index_distribution(rows),
        "by_subject": {
            subject: index_distribution(subject_rows)
            for subject, subject_rows in sorted(
                by_subject.items(), key=lambda kv: -len(kv[1])
            )
        },
        "question_chars": percentiles(q_lengths),
        "choice_chars": percentiles(c_lengths),
        "gold_choice_mean_chars": round(sum(gold_len) / len(gold_len), 1),
        "distractor_choice_mean_chars": round(sum(distractor_len) / len(distractor_len), 1),
        "gold_is_longest_choice": gold_longest,
        "gold_is_shortest_choice": gold_shortest,
        "duplicate_questions": duplicate_questions,
        "unique_questions": len(seen),
        "exact_duplicate_rows": exact_duplicate_rows,
        "keys_with_contradictory_gold": contradictory,
        "release_tagged_questions": tagged,
        "release_tag_distribution": {k: releases[k] for k in sorted(releases, key=int)},
        "distractor_tells": tells,
        "top_question_words": vocab.most_common(40),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    for row in rows[: args.samples]:
        print("\n" + "=" * 70)
        print(f"[{row.get('subject')}] {row['question']}")
        for position, choice in enumerate(row["choices"]):
            mark = " <== gold" if position == int(row["answer"]) else ""
            print(f"  [{position}] {choice}{mark}")


if __name__ == "__main__":
    main()
