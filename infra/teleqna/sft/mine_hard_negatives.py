#!/usr/bin/env python3
"""Mine distractors from the student's own confusions, and prove they are harder.

The measured residual says 63% of what survives context is discrimination, not
missing knowledge, so the recipe needs negatives that are hard for THIS model.
The 122B already writes distractors, but a 122B's sense of "confusable" is its
own; the student is OTel-2.0-31B-IT, and what fools a 122B need not fool it, nor
the reverse.

The criterion here is not "does a model call this distractor plausible" - that
is a judgement, and judgements about difficulty are exactly what models are bad
at. It is behavioural, two forward passes on the same MCQ:

    closed book   the model must pick the DISTRACTOR. If it already picks gold
                  without help, the negative teaches nothing.
    with evidence the model must pick GOLD. If the evidence does not flip it,
                  the item is ambiguous or the gold is wrong, and training on it
                  teaches the model to ignore evidence.

An item is kept only if it flips. That is the same shape as the training
objective - teacher with the passage, student without - so the kept set is by
construction the set where that objective has a gradient to give.

The 122B's own distractors go through the identical two passes as a control.
Without it "the mined negatives are hard" is unfalsifiable: some fraction of any
distractor set flips, and only the difference between the arms says whether
paying for this pass bought anything.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

ANS = re.compile(r"(?i)ANSWER\s*:\s*([A-Z])")
TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]{2,}")

GEN_SYS = (
    "You write examination distractors for telecom certification exams."
)
GEN_USER = """Below is a passage from a telecom specification or paper, a question about it, and the correct answer.

PASSAGE:
{evidence}

QUESTION: {question}
CORRECT ANSWER: {answer}

Write the 3 wrong answers that a competent telecom engineer would be most likely to confuse with the correct one. Each must be:
- definitely WRONG according to the passage
- the same kind of thing as the correct answer (if it names an entity, name a different entity; if it is a value, give a different value)
- similar in length and phrasing to the correct answer
- not a negation or paraphrase of the correct answer

Reply with only a JSON array of 3 strings."""

# Verbatim from eval_dev_vllm.py. A different wording here would make the
# confusability measured in this script incomparable with every accuracy number
# the campaign is quoted against, and the whole point of the two passes is that
# they read the same way the scorer does.
TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: \'ANSWER: $LETTER\' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)
CTX_TEMPLATE = (
    "Reference material.\n\n{context}\n\n---\n\n" + TEMPLATE
)


def terms(s: str) -> set[str]:
    return set(TOKEN.findall((s or "").lower()))


def jac(a: str, b: str) -> float:
    ta, tb = terms(a), terms(b)
    return len(ta & tb) / len(ta | tb) if (ta | tb) else 1.0


def parse_json_list(text: str) -> list[str]:
    m = re.search(r"\[.*?\]", text or "", re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in v if isinstance(x, (str, int, float))]


def build_mcq(gold: str, negs: list[str], seed: int) -> tuple[str, str]:
    """Shuffle so gold's letter is not predictable from the construction."""
    opts = [gold] + negs
    rng = random.Random(seed)
    rng.shuffle(opts)
    letter = chr(65 + opts.index(gold))
    body = "\n".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(opts))
    return body, letter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--dup", type=float, default=0.75,
                    help="reject a proposal this close to the gold answer")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = []
    for line in args.items.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("question") and r.get("answer") and r.get("evidence"):
            rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"items: {len(rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def chat(user: str, system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else [])
        msgs.append({"role": "user", "content": user})
        # enable_thinking=False leaves the thought channel pre-closed. This
        # model's format is held up by that scaffold, not by the instruction -
        # opening the channel makes it answer conversationally and drop the
        # ANSWER: line, which would look like a reasoning failure and is not.
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)

    llm = LLM(model=args.model, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, trust_remote_code=True,
              enforce_eager=False)

    # ---- stage 1: propose ---------------------------------------------------
    prompts = [chat(GEN_USER.format(evidence=r["evidence"][:3000],
                                    question=r["question"],
                                    answer=r["answer"]), GEN_SYS) for r in rows]
    gen = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9,
                                               max_tokens=400, seed=0))
    proposed = [parse_json_list(o.outputs[0].text) for o in gen]

    # ---- stage 2: static filter --------------------------------------------
    fail = collections.Counter()
    kept_negs: list[list[str]] = []
    for r, cand in zip(rows, proposed):
        gold = r["answer"]
        out, seen = [], set()
        for c in cand:
            cl = c.lower().strip()
            if not cl:
                fail["empty"] += 1
            elif cl == gold.lower().strip():
                fail["equals_answer"] += 1
            elif jac(c, gold) >= args.dup:
                fail["paraphrase_of_answer"] += 1
            elif cl in seen:
                fail["duplicate"] += 1
            elif len(c.split()) > 3 * max(len(gold.split()), 4):
                fail["too_long"] += 1
            else:
                seen.add(cl)
                out.append(c)
        kept_negs.append(out[:3])
    fail["no_usable_proposal"] = sum(1 for x in kept_negs if len(x) < 3)

    # ---- stage 3: the two behavioural passes, mined arm and control arm -----
    def two_pass(negs_per_row: list[list[str]], tag: str):
        idx, mcqs = [], []
        for i, (r, negs) in enumerate(zip(rows, negs_per_row)):
            if len(negs) < 3:
                continue
            body, letter = build_mcq(r["answer"], negs, seed=i)
            idx.append((i, letter))
            mcqs.append((body, r))
        letters = ",".join(chr(65 + i) for i in range(4))
        closed = [chat(TEMPLATE.format(letters=letters, question=r["question"],
                                       choices=body)) for body, r in mcqs]
        opened = [chat(CTX_TEMPLATE.format(
            context=r["evidence"][:3000], letters=letters,
            question=r["question"], choices=body)) for body, r in mcqs]
        # 512, not 16. This model justifies before it answers, so a short budget
        # truncates the response before the ANSWER: line ever appears - which
        # scores as unparsable and looks exactly like a model that cannot do the
        # task. The first run of this script measured 99% unparsable for that
        # reason alone and its numbers meant nothing.
        sp = SamplingParams(temperature=0.0, max_tokens=512)
        oc = llm.generate(closed, sp)
        oo = llm.generate(opened, sp)

        def pick(o):
            m = ANS.search(o.outputs[0].text or "")
            return m.group(1).upper() if m else ""

        res = []
        pos = collections.Counter()
        for (i, letter), a, b in zip(idx, oc, oo):
            pa, pb = pick(a), pick(b)
            pos[letter] += 1
            res.append({"i": i, "gold_letter": letter,
                        "closed": pa, "opened": pb,
                        "confusable": pa != letter and pa != "",
                        "resolvable": pb == letter})
        n = max(len(res), 1)
        stats = {
            "arm": tag, "n_validated": len(res),
            "confusable_closed_book": round(
                sum(r["confusable"] for r in res) / n, 4),
            "resolvable_with_evidence": round(
                sum(r["resolvable"] for r in res) / n, 4),
            "flips_both": round(sum(r["confusable"] and r["resolvable"]
                                    for r in res) / n, 4),
            "unparsable_closed": round(
                sum(1 for r in res if not r["closed"]) / n, 4),
            "gold_letter_hist": dict(sorted(pos.items())),
        }
        return res, stats

    mined, s_mined = two_pass(kept_negs, "mined_31b")
    ctrl_negs = [[str(x) for x in (r.get("distractors") or [])][:3] for r in rows]
    ctrl, s_ctrl = two_pass(ctrl_negs, "control_122b")

    # The two arms do not cover the same items: the mined arm drops whatever
    # failed to yield three clean proposals. Comparing 1,263 items against 1,500
    # lets a selection effect masquerade as a difference in difficulty, since
    # the items the miner could not handle may be systematically easier. The
    # honest headline is the paired one, over items both arms actually scored.
    m_by, c_by = {r["i"]: r for r in mined}, {r["i"]: r for r in ctrl}
    both = sorted(set(m_by) & set(c_by))
    nb = max(len(both), 1)
    paired = {
        "n_paired": len(both),
        "mined_confusable": round(sum(m_by[i]["confusable"] for i in both) / nb, 4),
        "control_confusable": round(sum(c_by[i]["confusable"] for i in both) / nb, 4),
        "mined_flips": round(sum(m_by[i]["confusable"] and m_by[i]["resolvable"]
                                 for i in both) / nb, 4),
        "control_flips": round(sum(c_by[i]["confusable"] and c_by[i]["resolvable"]
                                   for i in both) / nb, 4),
        "mined_only": sum(1 for i in both
                          if m_by[i]["confusable"] and not c_by[i]["confusable"]),
        "control_only": sum(1 for i in both
                            if c_by[i]["confusable"] and not m_by[i]["confusable"]),
    }
    # McNemar counts: the discordant pairs are the whole evidence. If mined_only
    # and control_only are close, the arms differ by shuffling which items are
    # hard, not by making more of them hard.
    paired["discordant_ratio"] = round(
        paired["mined_only"] / max(paired["control_only"], 1), 3)

    keep_i = {r["i"] for r in mined if r["confusable"] and r["resolvable"]}
    with args.out.open("w", encoding="utf-8") as fh:
        for r in mined:
            if r["i"] in keep_i:
                src = rows[r["i"]]
                fh.write(json.dumps({
                    "chunk_id": src.get("chunk_id"), "question": src["question"],
                    "answer": src["answer"], "evidence": src["evidence"],
                    "distractors": kept_negs[r["i"]],
                    "origin": "mined_31b",
                    "closed_book_pick": r["closed"], "gold_letter": r["gold_letter"],
                }, ensure_ascii=False) + "\n")

    report = {
        "items_in": len(rows),
        "proposals_parsed": round(
            sum(1 for c in proposed if c) / max(len(proposed), 1), 4),
        "static_failures": dict(fail.most_common()),
        "arms": [s_mined, s_ctrl],
        "delta_confusable_mined_minus_control": round(
            s_mined["confusable_closed_book"] - s_ctrl["confusable_closed_book"], 4),
        "delta_flips_mined_minus_control": round(
            s_mined["flips_both"] - s_ctrl["flips_both"], 4),
        "paired": paired,
        "kept": len(keep_i),
        "keep_rate": round(len(keep_i) / max(len(rows), 1), 4),
        "note": "confusable and resolvable are behavioural, not judged. A high "
                "confusable rate with a low resolvable rate means the negatives "
                "are ambiguous rather than hard, and that set would teach the "
                "model to distrust its evidence.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
