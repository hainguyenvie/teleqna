#!/usr/bin/env python3
"""Build DPO preference pairs from the synthetic pools + student-gate picks.

Why DPO after two failed SFT runs: the dose-response sweep showed SFT's
damage is immediate (70.2% at 0.24 epoch vs 73.8% baseline) while its gains
saturate at ~85 questions — maximum-likelihood on the gold letter shifts
shared circuits and erases more baseline knowledge than the data teaches.
DPO's implicit KL anchor to the reference model targets exactly that failure:
it moves the *relative* preference between two letters and is penalised for
drifting from the base elsewhere.

Pairs: rows where the student answered wrong, chosen = "ANSWER: <gold>",
rejected = "ANSWER: <the letter the student actually picked>" — the model's
own confusions, not synthetic ones. Choice order is shuffled with the same
salt/permutation as build_train_set.py, and the student's pick (recorded on
the unshuffled row) is mapped through the same permutation.

Filters reused from build_train_set.py: self-ref/editorial style sweep,
cross-batch dedup, per-spec caps with the 45.912/45.050 down-weight, per-doc
caps for literature.
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
DOWNWEIGHT_SPEC_FRAC = 0.01

SINGLE_ANSWER_TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)

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
    w = _WORD.findall(_WS.sub(" ", _TAG.sub(" ", q)).strip().lower())
    return " ".join(w[-n:])


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def load_picks(path: Path) -> dict[str, str]:
    """sample_id -> parsed letter, only for wrong answers with a valid parse."""
    out = {}
    for r in load_jsonl(path):
        if not r["correct"] and r.get("parsed_answer"):
            out[r["sample_id"]] = r["parsed_answer"]
    return out


def load_correct_ids(path: Path) -> set[str]:
    return {r["sample_id"] for r in load_jsonl(path) if r["correct"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, action="append", required=True,
                     help="kept-pool jsonl; repeatable")
    ap.add_argument("--results", type=Path, action="append", required=True,
                     help="matching student-gate results jsonl, same order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multi-neg", action="store_true",
                     help="fix rows emit one pair per WRONG choice (gold vs "
                          "each distractor), not just gold vs the student's "
                          "pick. Runs 1-3 plateaued at net +4..6 rows with "
                          "single-negative pairs — pushing gold above ONE "
                          "alternative lets probability mass slosh to the "
                          "others; above ALL of them is a different geometry.")
    ap.add_argument("--anchor-frac", type=float, default=0.0,
                     help="also emit this many anchor pairs per fix pair: "
                          "rows the student answered CORRECTLY, chosen=gold, "
                          "rejected=a deterministic distractor. Anchors give "
                          "~zero gradient while the model stays right and "
                          "pull back the moment it starts to drift — aimed "
                          "at the lost-rows problem (3 lost per 2 gained at "
                          "every dose of the anchor-less run).")
    args = ap.parse_args()
    assert len(args.pool) == len(args.results)

    def h01(*parts: str) -> float:
        d = hashlib.sha256("|".join(parts).encode()).digest()
        return int.from_bytes(d[:8], "big") / 2**64

    cand: list[dict] = []
    anchors_raw: list[dict] = []
    for pool_path, res_path in zip(args.pool, args.results):
        picks = load_picks(res_path)
        correct_ids = load_correct_ids(res_path) if args.anchor_frac > 0 else set()
        for r in load_jsonl(pool_path):
            pick = picks.get(r["sample_id"])
            if pick is not None:
                r["_pick_index"] = ord(pick) - 65
                if r["_pick_index"] >= len(r["choices"]):
                    continue
                r["_kind"] = "fix"
                cand.append(r)
            elif r["sample_id"] in correct_ids:
                # rejected = deterministic distractor (never the gold)
                d = hashlib.sha256(("anchor|" + r["question"]).encode()).digest()
                others = [i for i in range(len(r["choices"])) if i != int(r["answer"])]
                r["_pick_index"] = others[d[0] % len(others)]
                r["_kind"] = "anchor"
                anchors_raw.append(r)

    if args.anchor_frac > 0:
        want = int(len(cand) * args.anchor_frac)
        anchors_raw.sort(key=lambda r: hashlib.sha256(
            ("asel|" + r["question"]).encode()).hexdigest())
        cand += anchors_raw[:want]

    for i, r in enumerate(cand):
        r["sample_id"] = f"dpo-{i:06d}"

    def cap_key(row):
        return row["meta"].get("spec") or row["meta"].get("doc_id", "?")

    def is_lit(row):
        return "collection" in row["meta"]

    drops = collections.Counter()
    seen: set[str] = set()
    survivors = []
    for r in sorted(cand, key=lambda x: x["sample_id"]):
        q = r["question"]
        if SELF_REF.search(q):
            drops["self_ref"] += 1
            continue
        if EDITORIAL.search(q):
            drops["editorial"] += 1
            continue
        t = norm_tail(q)
        if t in seen:
            drops["dup"] += 1
            continue
        seen.add(t)
        survivors.append(r)

    n = len(survivors)
    spec_cap = int(n * SPEC_CAP_FRAC)
    down_cap = int(n * DOWNWEIGHT_SPEC_FRAC)
    doc_cap = max(20, int(n * 0.002))
    per_key = collections.Counter()
    out_rows = []
    for r in survivors:
        key = cap_key(r)
        cap = down_cap if key in DOWNWEIGHT_SPECS else (doc_cap if is_lit(r) else spec_cap)
        if per_key[key] >= cap:
            drops["cap"] += 1
            continue
        per_key[key] += 1

        choices = list(r["choices"])
        nc = len(choices)
        digest = hashlib.sha256(f"train_v1_position:{r['sample_id']}".encode()).digest()
        shift = digest[0] % nc
        order = [(i + shift) % nc for i in range(nc)]
        shuffled = [choices[j] for j in order]
        gold_new = order.index(int(r["answer"]))
        pick_new = order.index(r["_pick_index"])
        if gold_new == pick_new:
            drops["degenerate"] += 1
            continue
        letters = ",".join(chr(65 + i) for i in range(nc))
        ctext = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(shuffled))
        prompt = SINGLE_ANSWER_TEMPLATE.format(
            letters=letters, question=r["question"], choices=ctext)
        kind = r.get("_kind", "fix")
        if args.multi_neg and kind == "fix":
            rejected_list = [i for i in range(nc) if i != gold_new]
        else:
            rejected_list = [pick_new]
        for j, rej in enumerate(rejected_list):
            out_rows.append({
                "sample_id": f"{r['sample_id']}-{j}",
                "kind": kind,
                "prompt": prompt,
                "chosen": f"ANSWER: {chr(65 + gold_new)}",
                "rejected": f"ANSWER: {chr(65 + rej)}",
                "meta": r["meta"],
            })

    out_rows.sort(key=lambda r: h01("order", r["sample_id"]))
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "candidates": len(cand), "after_style": len(survivors),
        "pairs": len(out_rows), "drops": dict(drops),
        "by_kind": dict(collections.Counter(r["kind"] for r in out_rows)),
        "source_mix": dict(collections.Counter(
            "literature" if "collection" in r["meta"] else "spec" for r in out_rows)),
        "chosen_letters": dict(collections.Counter(
            r["chosen"][-1] for r in out_rows)),
        "rejected_letters": dict(collections.Counter(
            r["rejected"][-1] for r in out_rows)),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
