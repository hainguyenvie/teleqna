#!/usr/bin/env python3
"""Build the GRPO prompt set for the teleqna specialist.

Why GRPO, and why on thinking traces: the coverage diagnosis showed letter
supervision cannot inject knowledge (fix-rate flat against training-pair
similarity). GRPO does not supervise the letter — it rewards a whole sampled
reasoning path that lands on the right letter, which is the channel that
trains *selection*. The distractor probe (+15.83pp) and the any-branch oracle
(83.85%) both say the information is already inside the model on most missed
rows; RL is how a policy learns to reach it reliably (pass@k -> pass@1).

Prompt selection follows the gradient, not just the error list. GRPO's
advantage is zero when all k rollouts agree, so rows the student answers
*consistently* — right or wrong — contribute nothing. The useful band is where
it is unstable. Greedy student-gate results are the only difficulty signal
available, so:

  * every row the student got WRONG          (the errors we want fixed)
  * a deterministic slice of rows it got RIGHT (keeps consistency pressure,
    and a greedy-right row still splits under sampling)

Prompts carry no label — the gold letter travels in a separate column that
only the reward function reads. Choice order is shuffled with the same salt
as every other builder here, and the same style/dedup/cap filters apply.
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


def loadl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, action="append", required=True)
    ap.add_argument("--results", type=Path, action="append", required=True)
    ap.add_argument("--test", type=Path, required=True,
                    help="the real 10,000 — used only as a contamination guard")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--correct-frac", type=float, default=0.35,
                    help="student-correct rows to keep, as a fraction of the "
                         "wrong ones")
    ap.add_argument("--limit", type=int, default=6000,
                    help="cap the prompt set; GRPO cost is k rollouts per "
                         "prompt per epoch, so this is the real budget knob")
    args = ap.parse_args()
    assert len(args.pool) == len(args.results)

    test_tails = {norm_tail(r["question"]) for r in loadl(args.test)}
    print(f"contamination guard: {len(test_tails)} test tails indexed")

    wrong, right = [], []
    for pool_path, res_path in zip(args.pool, args.results):
        verdict = {r["sample_id"]: r["correct"] for r in loadl(res_path)}
        for r in loadl(pool_path):
            v = verdict.get(r["sample_id"])
            if v is None:
                continue
            r["_band"] = "right" if v else "wrong"
            (right if v else wrong).append(r)
    print(f"student-gate pools: wrong={len(wrong)} right={len(right)}")

    def hkey(r, salt):
        return hashlib.sha256((salt + "|" + r["question"]).encode()).hexdigest()

    right.sort(key=lambda r: hkey(r, "grpo-right"))
    keep_right = right[:int(len(wrong) * args.correct_frac)]
    cand = wrong + keep_right
    cand.sort(key=lambda r: hkey(r, "grpo-order"))

    drops = collections.Counter()
    seen: set[str] = set()
    survivors = []
    for r in cand:
        q = r["question"]
        if SELF_REF.search(q):
            drops["self_ref"] += 1
            continue
        if EDITORIAL.search(q):
            drops["editorial"] += 1
            continue
        t = norm_tail(q)
        if t in test_tails:
            drops["TEST_OVERLAP"] += 1
            continue
        if t in seen:
            drops["dup"] += 1
            continue
        seen.add(t)
        survivors.append(r)

    n = len(survivors)
    spec_cap, down_cap = int(n * SPEC_CAP_FRAC), int(n * DOWNWEIGHT_SPEC_FRAC)
    doc_cap = max(20, int(n * 0.002))
    per_key: collections.Counter = collections.Counter()
    rows = []
    for r in survivors:
        meta = r["meta"]
        is_lit = "collection" in meta
        key = meta.get("spec") or meta.get("doc_id", "?")
        cap = down_cap if key in DOWNWEIGHT_SPECS else (doc_cap if is_lit else spec_cap)
        if per_key[key] >= cap:
            drops["cap"] += 1
            continue
        per_key[key] += 1

        choices = list(r["choices"])
        nc = len(choices)
        digest = hashlib.sha256(f"train_v1_position:{r['sample_id']}".encode()).digest()
        order = [(i + digest[0] % nc) % nc for i in range(nc)]
        shuffled = [choices[j] for j in order]
        gold = order.index(int(r["answer"]))
        prompt = SINGLE_ANSWER_TEMPLATE.format(
            letters=",".join(chr(65 + i) for i in range(nc)),
            question=r["question"],
            choices="\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(shuffled)))
        rows.append({
            "prompt": [{"role": "user", "content": prompt}],
            "answer": chr(65 + gold),
            "n_choices": nc,
            "src": "literature" if is_lit else "spec",
            "band": r["_band"],
        })
        if len(rows) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "prompts": len(rows), "drops": dict(drops),
        "by_src": dict(collections.Counter(r["src"] for r in rows)),
        "by_n_choices": dict(collections.Counter(r["n_choices"] for r in rows)),
        "by_band": dict(collections.Counter(r["band"] for r in rows)),
        "gold_letters": dict(collections.Counter(r["answer"] for r in rows)),
    }
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if drops["TEST_OVERLAP"]:
        print(f"NOTE: {drops['TEST_OVERLAP']} generated rows collided with a real "
              f"test question tail and were dropped — expected to be small; a "
              f"large number means the generator is echoing the benchmark.")


if __name__ == "__main__":
    main()
