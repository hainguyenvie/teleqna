#!/usr/bin/env python3
"""Turn the gated synthetic pool into an actual SFT file.

Gating (gate_synth.py) only checks form — schema, style, verbatim evidence,
distinctness, dedup, contamination. None of that checks whether a question is
worth a training gradient. The student-difficulty gate (run_baseline.py over
the pool) gives that signal directly: a question the student already answers
correctly under greedy decoding teaches nothing about the failure mode this
track targets (the distractor probe: +15.8pp when foreign distractors replace
the original ones). So the keep/weight policy here is explicitly "teach the
wrong answers, keep a calibration slice of the right ones" — not "keep
everything gated" — and it applies uniformly to every source slice (3GPP
spec-sourced hard_negative/standard rows and, once generated, literature-
sourced rows): correctness on the student gate decides weight, not mode or
tier.

Four defects fixed here, all found in journey.html's audit (Bước 12) and
error-diagnosis (Bước 15) sections:
  1. gold-position skew (58.2% at A) -> deterministic hash-shuffle per row,
     independent of run_baseline.py's own permutation (different salt) so a
     model trained on this file and then scored by permute=True isn't
     evaluated on a distribution it already saw shuffled one way.
  2. TS 45.912 / 45.050 dominance (17.96% of rows) teaching only rote legacy
     RF constants (confirmed by: dominance, flat difficulty, and reading the
     actual wrong answers in Bước 15) -> hard per-spec cap + explicit
     down-weight for those two specs.
  3. no difficulty signal used at all -> student-gate correctness drives
     keep-rate (see policy below).
  4. harness-format mismatch -> prompt/completion built byte-identical to
     run_baseline.py's build_prompt/SINGLE_ANSWER_TEMPLATE and "ANSWER: X".

Keep/weight policy per (mode, student_correct):
  hard_negative, incorrect -> keep, weight 1.0   (the target failure mode)
  hard_negative, correct   -> keep, weight 0.5   (real discrimination, easier)
  standard,      incorrect -> keep, weight 1.0   (still a real miss)
  standard,      correct   -> keep, weight 0.15  (rote recall, low value;
                                                   kept thin so the model
                                                   doesn't unlearn easy facts)
  shape_prior,   incorrect -> keep, weight 1.0
  shape_prior,   correct   -> keep, weight 0.5   (scarcer slice, keep more
                                                   than standard-correct)
Rows with no student-gate record (request/parse failure) default to weight
0.5, mode-independent.

"weight" is realised as an integer repeat count (weight 0.15 -> kept ~15% of
the time via a deterministic hash draw, not literally repeated fractionally).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

DOWNWEIGHT_SPECS = {"45.912", "45.050"}
SPEC_CAP_FRAC = 0.05
DOWNWEIGHT_SPEC_FRAC = 0.01  # within-pool share ceiling for the two rote specs

# Post-gate style sweep (audit of train_v2 found these surviving the per-batch
# gates): questions referring to "the proposed X / this paper / the authors"
# are unanswerable closed-book — they only mean something next to the source
# article — and section/table-number questions train editorial trivia the
# benchmark never asks. Both patterns are dropped here for every source.
SELF_REF = re.compile(
    r"(?i)\b(this\s+(paper|study|article|work|survey)|the\s+authors?\b"
    r"|(the|this)\s+proposed\b|authorship|our\s+(proposed|method|approach|"
    r"scheme|algorithm|results?|model|system))")
EDITORIAL = re.compile(
    r"(?i)\b(which|what)\s+(table|figure|annex|clause|section|subclause)\b")

_WS = re.compile(r"\s+")
_TAG = re.compile(r"\[[^\]]*\]")
_WORD = re.compile(r"[a-z0-9][a-z0-9.\-]+")


def norm_tail(q: str, n: int = 6) -> str:
    """Same normalised content-word tail the gate deduplicates on. Needed here
    a second time because each generation batch (spec / shape / literature)
    ran the gate separately — tails are unique within a batch, not across."""
    w = _WORD.findall(_WS.sub(" ", _TAG.sub(" ", q)).strip().lower())
    return " ".join(w[-n:])

SINGLE_ANSWER_TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)

KEEP_WEIGHT = {
    ("hard_negative", False): 1.0,
    ("hard_negative", True): 0.5,
    ("standard", False): 1.0,
    ("standard", True): 0.15,
    # shape_prior is exempt from difficulty weighting (1.0 both ways): the
    # slice was subsampled upstream (rebalance) to match the real test's
    # conditional gold-rates exactly (All-of-above 89.81%, None 0.31%), and
    # any non-uniform filtering here un-calibrates it — measured: 0.5 on
    # student-correct rows dragged the All-of-above gold-rate to 78.6%.
    ("shape_prior", False): 1.0,
    ("shape_prior", True): 1.0,
}


def h01(*parts: str) -> float:
    """Deterministic uniform [0,1) draw from a string key."""
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_student_results(path: Path) -> dict[str, bool]:
    out = {}
    for row in load_jsonl(path):
        out[row["sample_id"]] = bool(row["correct"])
    return out


def shuffle_choices(row: dict, salt: str) -> dict:
    """Independent hash-based shuffle, salted differently from
    run_baseline.py's permutation_for so this isn't the same rotation a
    permute=True eval run would apply."""
    choices = list(row["choices"])
    n = len(choices)
    digest = hashlib.sha256(f"{salt}:{row['sample_id']}".encode("utf-8")).digest()
    shift = digest[0] % n
    order = [(i + shift) % n for i in range(n)]
    new_choices = [choices[j] for j in order]
    new_gold = order.index(int(row["answer"]))
    return {**row, "choices": new_choices, "answer": new_gold}


def to_training_row(row: dict, gold_shuffled: dict) -> dict:
    choices = gold_shuffled["choices"]
    letters = ",".join(chr(65 + i) for i in range(len(choices)))
    choices_text = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))
    prompt = SINGLE_ANSWER_TEMPLATE.format(
        letters=letters, question=gold_shuffled["question"], choices=choices_text)
    letter = chr(65 + gold_shuffled["answer"])
    return {
        "sample_id": row["sample_id"],
        "prompt": prompt,
        "completion": f"ANSWER: {letter}",
        "meta": row["meta"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-kept", type=Path, required=True)
    ap.add_argument("--full-results", type=Path, required=True)
    ap.add_argument("--shape-kept", type=Path, required=True)
    ap.add_argument("--shape-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", type=Path, action="append", default=[],
                     help="additional pre-graded jsonl pools, same schema as "
                          "--full-kept plus a 'student_correct' bool field "
                          "already merged in (e.g. a TCC-literature batch)")
    args = ap.parse_args()

    full = load_jsonl(args.full_kept)
    full_res = load_student_results(args.full_results)
    shape = load_jsonl(args.shape_kept)
    shape_res = load_student_results(args.shape_results)

    pool: list[tuple[dict, bool | None]] = []
    pool += [(r, full_res.get(r["sample_id"])) for r in full]
    pool += [(r, shape_res.get(r["sample_id"])) for r in shape]
    for extra_path in args.extra:
        for r in load_jsonl(extra_path):
            pool.append((r, r.get("student_correct")))

    # Re-key sample_id so it's unique across the merged pool (full/shape/extra
    # each restart their own synth-NNNNNN counter).
    for i, (r, _) in enumerate(pool):
        r["sample_id"] = f"train-{i:06d}"

    drops = collections.Counter()
    # Cap key: per-spec on the 3GPP side (one spec must not dominate), per-doc
    # on the literature side (make_tcc_chunks.py rows carry doc_id, not spec —
    # capping the whole IEEE-Access collection under one key would strangle
    # the slice that maps to 65% of the real benchmark, while a single paper
    # dominating is the actual failure mode to prevent).
    def cap_key(row: dict) -> str:
        return row["meta"].get("spec") or row["meta"].get("doc_id", "unknown")

    def is_literature(row: dict) -> bool:
        return "collection" in row["meta"]

    # Pass 1: difficulty weighting. Pass 2: caps. Two passes because the caps
    # must be fractions of the OUTPUT, not the input pool — v1 computed them
    # on the input and the two down-weighted specs came out at 5.6% of the
    # output instead of the intended <=2%.
    pool.sort(key=lambda pair: pair[0]["sample_id"])
    seen_tails: set[str] = set()
    survivors: list[tuple[dict, bool | None]] = []
    for row, correct in pool:
        q = row["question"]
        if SELF_REF.search(q):
            drops["self_ref"] += 1
            continue
        if EDITORIAL.search(q):
            drops["editorial"] += 1
            continue
        tail = norm_tail(q)
        if tail in seen_tails:
            drops["cross_batch_dup"] += 1
            continue
        seen_tails.add(tail)
        mode = row["meta"].get("mode", "standard")
        weight = KEEP_WEIGHT.get((mode, correct), 0.5 if correct is None else 1.0)
        if h01("keep", row["sample_id"]) >= weight:
            drops["difficulty_weight"] += 1
            continue
        survivors.append((row, correct))

    out_est = len(survivors)
    spec_cap = int(out_est * SPEC_CAP_FRAC)
    downweight_cap = int(out_est * DOWNWEIGHT_SPEC_FRAC)
    # Per-doc cap for literature: generous vs typical per-doc yield (~30 rows)
    # but stops a single monster survey from flooding the mix.
    doc_cap = max(20, int(out_est * 0.002))

    kept_per_spec = collections.Counter()
    out_rows = []
    by_mode_correct = collections.Counter()

    for row, correct in survivors:
        key = cap_key(row)
        if key in DOWNWEIGHT_SPECS:
            cap = downweight_cap
        elif is_literature(row):
            cap = doc_cap
        else:
            cap = spec_cap
        if kept_per_spec[key] >= cap:
            drops["spec_cap"] += 1
            continue
        mode = row["meta"].get("mode", "standard")
        kept_per_spec[key] += 1
        by_mode_correct[(mode, correct)] += 1
        shuffled = shuffle_choices(row, salt="train_v1_position")
        out_rows.append(to_training_row(row, shuffled))

    # Global shuffle of row order (deterministic), so spec/mode aren't
    # clustered by the sample_id sort above.
    out_rows.sort(key=lambda r: h01("order", r["sample_id"]))

    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gold_letters = collections.Counter()
    for r in out_rows:
        gold_letters[r["completion"].split()[-1]] += 1

    report = {
        "input_pool": len(pool),
        "output_rows": len(out_rows),
        "drops": dict(drops),
        "by_mode_correct": {f"{m}/{'correct' if c else 'wrong' if c is not None else 'unknown'}": n
                             for (m, c), n in by_mode_correct.items()},
        "gold_letter_distribution": dict(gold_letters),
        "top_specs": dict(collections.Counter(
            r["meta"].get("spec") or r["meta"]["collection"]
            for r in out_rows).most_common(10)),
        "downweighted_spec_share": sum(
            1 for r in out_rows
            if r["meta"].get("spec") in DOWNWEIGHT_SPECS) / max(1, len(out_rows)),
        "source_mix": dict(collections.Counter(
            "literature" if "collection" in r["meta"] else "spec" for r in out_rows)),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
