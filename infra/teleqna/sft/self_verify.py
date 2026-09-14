#!/usr/bin/env python3
"""Try to capture the branch-selection headroom without training anything.

The 122B's pass@k profile on 500 dev rows splits the benchmark cleanly:

    406 rows (81.2%)  all 8 samples agree      86.9% correct, nothing to win
     94 rows (18.8%)  the samples disagree     plurality 46.8%, oracle 81.9%

So the entire 6.6-point gap between vote@8 (81.60) and pass@8 (88.20) lives in
those 94 rows, and they are identifiable at serving time with no labels - you
just check whether the samples agreed. This script asks how much of that 35-point
gap can be taken by asking the model again, in a better-posed way, and nothing
else. No verifier is trained, no retrieval is used.

Three arms, because "self-verification" names several different questions and
they are not equivalent:

  greedy       re-answer the original question once at temperature 0. The
               baseline that must be beaten: if a single careful pass already
               matches plurality, sampling-then-picking was never the point.
  adjudicate   re-ask with the option set narrowed to the letters the samples
               actually produced. This is the arm with a mechanism: the gold is
               among the candidates 81.9% of the time, so cutting the menu from
               five to two or three is a strictly easier question.
  verify       ask, per candidate, whether that specific option is correct, and
               keep the one affirmed. Slower (one call per candidate) but it is
               the shape a trained verifier would take, so its score estimates
               what training could buy before any training happens.

`adjudicate` has a confound worth stating rather than hiding: a narrower menu
raises the score of blind guessing too (1/2 or 1/3 rather than 1/5). The report
prints the guess floor for each row's candidate count so the gain can be read
against it instead of against zero.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")

TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)
VERIFY = (
    "Consider the following multiple choice question.\n\n{question}\n\n{choices}"
    "\n\nIs option {letter} the correct answer? The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of Y,N."
)


def parse(completion: str, allowed: set[str]) -> str:
    m = STRICT.findall(completion or "") or LOOSE.findall(completion or "")
    if not m:
        return ""
    got = m[-1].strip().rstrip(".").upper()
    return got if got in allowed else ""


def choices_block(row: dict, keep: list[str] | None = None) -> str:
    out = []
    for i, c in enumerate(row["choices"]):
        letter = chr(65 + i)
        if keep is None or letter in keep:
            out.append(f"{letter}) {c}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passk", type=Path, required=True,
                    help="profile_pass_rate.py rollout file for the model")
    ap.add_argument("--split", type=Path, required=True,
                    help="the eval split, for question and choice text")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="greedy,adjudicate,verify")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.95)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--engine-kwargs", default="")
    args = ap.parse_args()

    split = {json.loads(l)["sample_id"]: json.loads(l)
             for l in args.split.open(encoding="utf-8")}
    rollouts = [json.loads(l) for l in args.passk.open(encoding="utf-8")]

    split_rows, agree_rows = [], []
    for r in rollouts:
        letters = [x for x in r["letters"] if x]
        cands = sorted(set(letters))
        (split_rows if len(cands) > 1 else agree_rows).append((r, cands))
    print(f"disagreeing rows: {len(split_rows)} / {len(rollouts)}   "
          f"agreeing: {len(agree_rows)}", flush=True)

    def score(rows, pred):
        return sum(1 for (r, _), p in zip(rows, pred) if p == r["answer"])

    plurality = []
    for r, cands in split_rows:
        cnt = collections.Counter(x for x in r["letters"] if x)
        plurality.append(cnt.most_common(1)[0][0])
    oracle = sum(1 for (r, c) in split_rows if r["answer"] in c)
    agree_correct = sum(1 for (r, c) in agree_rows if c and c[0] == r["answer"])

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    engine = dict(model=args.model, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len, tensor_parallel_size=args.tp,
                  enforce_eager=False)
    if args.engine_kwargs:
        engine.update(json.loads(args.engine_kwargs))
    print("engine:", {k: v for k, v in engine.items() if k != "model"}, flush=True)
    llm = LLM(**engine)

    def ask(prompts: list[str], temperature: float):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True) for p in prompts]
        sp = SamplingParams(temperature=temperature, top_p=0.95, top_k=20,
                            max_tokens=args.max_tokens, seed=42)
        return [o.outputs[0].text for o in llm.generate(chats, sp)]

    results = {"plurality": {"correct": score(split_rows, plurality),
                             "n": len(split_rows)},
               "oracle_among_candidates": {"correct": oracle,
                                           "n": len(split_rows)}}
    want = [a for a in args.arms.split(",") if a.strip()]

    if "greedy" in want:
        prompts, allowed = [], []
        for r, _ in split_rows:
            row = split[r["sample_id"]]
            n = len(row["choices"])
            allowed.append({chr(65 + i) for i in range(n)})
            prompts.append(TEMPLATE.format(
                letters=",".join(chr(65 + i) for i in range(n)),
                question=row["question"], choices=choices_block(row)))
        pred = [parse(t, a) for t, a in zip(ask(prompts, 0.0), allowed)]
        results["greedy"] = {"correct": score(split_rows, pred),
                             "n": len(split_rows),
                             "unparsed": sum(1 for p in pred if not p)}

    if "adjudicate" in want:
        prompts, allowed = [], []
        for r, cands in split_rows:
            row = split[r["sample_id"]]
            allowed.append(set(cands))
            prompts.append(TEMPLATE.format(
                letters=",".join(cands), question=row["question"],
                choices=choices_block(row, keep=cands)))
        pred = [parse(t, a) for t, a in zip(ask(prompts, 0.0), allowed)]
        results["adjudicate"] = {"correct": score(split_rows, pred),
                                 "n": len(split_rows),
                                 "unparsed": sum(1 for p in pred if not p)}

    if "verify" in want:
        prompts, index = [], []
        for i, (r, cands) in enumerate(split_rows):
            row = split[r["sample_id"]]
            for letter in cands:
                index.append((i, letter))
                prompts.append(VERIFY.format(
                    question=row["question"], choices=choices_block(row),
                    letter=letter))
        outs = ask(prompts, 0.0)
        votes: dict[int, list[str]] = collections.defaultdict(list)
        for (i, letter), text in zip(index, outs):
            if parse(text, {"Y", "N"}) == "Y":
                votes[i].append(letter)
        pred = []
        for i, (r, cands) in enumerate(split_rows):
            yes = votes.get(i, [])
            # No affirmation, or several: fall back to plurality rather than
            # guessing. A verifier that abstains must not lose to the baseline.
            pred.append(yes[0] if len(yes) == 1 else plurality[i])
        results["verify"] = {"correct": score(split_rows, pred),
                             "n": len(split_rows),
                             "affirmed_exactly_one": sum(
                                 1 for i in range(len(split_rows))
                                 if len(votes.get(i, [])) == 1)}

    n_all = len(rollouts)
    guess_floor = sum(1 / len(c) for _, c in split_rows) / len(split_rows)
    print()
    print(f"{'arm':<26}{'on 94':>9}{'':>3}{'overall 500':>13}")
    for name, v in results.items():
        acc = 100 * v["correct"] / v["n"]
        overall = 100 * (agree_correct + v["correct"]) / n_all
        print(f"{name:<26}{acc:8.2f}%{'':>3}{overall:12.2f}%")
    print(f"\nblind-guess floor on the narrowed menus: {100*guess_floor:.1f}%")
    print(f"agreeing rows contribute {agree_correct}/{len(agree_rows)} "
          f"({100*agree_correct/len(agree_rows):.1f}%) to every arm")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"disagreeing": len(split_rows), "agreeing": len(agree_rows),
         "agree_correct": agree_correct, "guess_floor": round(guess_floor, 4),
         "arms": results,
         "note": "no training, no retrieval; arms differ only in how the model "
                 "is re-asked about rows its own samples disagreed on"},
        indent=1))


if __name__ == "__main__":
    main()
