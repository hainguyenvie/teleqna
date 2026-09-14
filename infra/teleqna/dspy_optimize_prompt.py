#!/usr/bin/env python3
"""Optimise the serving-side system prompt for teleqna with DSPy.

What is and is not fixed
------------------------
Inspect's `multiple_choice(cot=False)` replaces `state.user_prompt.text` with
SINGLE_ANSWER_TEMPLATE and calls `generate()`. What reaches the endpoint is a
single user message and no system message.

That is what the harness *sends*. It is not a limit on what the model *sees* —
the endpoint is ours, so the request can be rewritten, augmented with retrieved
text, fanned out into several calls, or post-processed, and the harness cannot
tell. Only three things are genuinely fixed:

  * the information available — question text and option texts, nothing else.
    Note `subject` is Sample metadata and never reaches the prompt, so any
    subject-conditional behaviour has to classify the subject itself.
  * the output contract — the reply must contain `ANSWER: X`, last match wins,
    letter in range.
  * what gets declared on the leaderboard row.

This stage optimises a **system prompt** only. That is a choice, not a
constraint: it is the cheapest thing that deploys, it adds zero extra calls per
question, and it is a clean control for the structural changes (retrieval,
self-consistency) that come after. The user message is passed through verbatim
so that the optimised prompt is measured against exactly the harness's wording.

What this can and cannot buy
----------------------------
The baseline already says where prompting has no room. Enabling thinking gained
+2.78 on Research publications (p < 0.0001) but +0.35 on Standards
specifications (p = 0.75) while churning 17% of its answers there — deliberation
without the underlying fact. A system prompt is strictly less powerful than
reasoning tokens, so Standards specifications should not move, and if it does,
that is a result worth being suspicious of rather than pleased about.

Splits
------
The optimiser only ever sees train + val from `make_splits.py`. Every reported
number comes from the held-out 9,000 rows via `run_baseline.py --system`, i.e.
through the identical scorer as the baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.base import Adapter as BaseAdapter
from dspy.teleprompt import COPRO, GEPA

TASK_URL = os.environ.get("VLLM_BASE", "http://telelogs-bench4-vllm:8000/v1")
# Reflection runs on the second service so proposal generation does not queue
# behind the thousands of task calls an optimiser makes.
REFLECT_URL = os.environ.get("VLLM_REFLECT_BASE", "http://telelogs-rl-vllm:8000/v1")
MODEL = os.environ.get("TELEQNA_MODEL", "Qwen/Qwen3-8B")

STRICT_ANSWER = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
LOOSE_ANSWER = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")

SINGLE_ANSWER_TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)

SEED_INSTRUCTIONS = (
    "You are answering a multiple-choice question about telecommunications: "
    "3GPP and IEEE standards, network architecture, and the wireless research "
    "literature. Identify the single best option and reply in the exact format "
    "the question asks for."
)


def parse_answer(completion: str, choice_count: int) -> str:
    """Verbatim port of Inspect's parse_answers() for multiple_correct=False."""
    matches = STRICT_ANSWER.findall(completion or "")
    if not matches:
        matches = LOOSE_ANSWER.findall(completion or "")
    if not matches:
        return ""
    matched = matches[-1].strip().rstrip(".").upper()
    allowed = {chr(65 + index) for index in range(choice_count)}
    return matched if matched in allowed else ""


def harness_prompt(row: dict[str, Any]) -> str:
    choices = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(row["choices"]))
    letters = ",".join(chr(65 + i) for i in range(len(row["choices"])))
    return SINGLE_ANSWER_TEMPLATE.format(
        letters=letters, question=row["question"], choices=choices
    )


class AnswerMCQ(dspy.Signature):
    """Placeholder. The optimiser rewrites this docstring; it becomes the system prompt."""

    harness_prompt: str = dspy.InputField(desc="the message the benchmark harness sent, verbatim")
    response: str = dspy.OutputField(desc="a reply whose last line is 'ANSWER: $LETTER'")


class ServingAdapter(BaseAdapter):
    """Emit exactly what a self-hosted endpoint would put in front of the model.

    DSPy's default adapters wrap fields in their own scaffolding, which would
    make the optimised artefact undeployable — the harness's message would no
    longer arrive intact. This one puts the optimised instructions in the system
    slot and passes the harness message through untouched, so whatever wins here
    is exactly what serving will do.

    Demos are ignored deliberately. Any few-shot example would have to come from
    the 10,000 rows, which are the test set; synthetic demos are a later stage
    with their own provenance to establish.
    """

    def format(self, signature, demos, inputs) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": signature.instructions},
            {"role": "user", "content": inputs["harness_prompt"]},
        ]

    def parse(self, signature, completion: str) -> dict[str, Any]:
        return {"response": completion}


class Program(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.answer = dspy.Predict(AnswerMCQ.with_instructions(SEED_INSTRUCTIONS))

    def forward(self, harness_prompt: str, **_: Any):
        return self.answer(harness_prompt=harness_prompt)


def build_examples(path: Path) -> list[dspy.Example]:
    out = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        out.append(
            dspy.Example(
                harness_prompt=harness_prompt(row),
                target=chr(65 + int(row["answer"])),
                n_choices=len(row["choices"]),
                subject=row.get("subject"),
                gold_text=row["choices"][int(row["answer"])],
                sample_id=row["sample_id"],
            ).with_inputs("harness_prompt")
        )
    return out


def score_only(gold, pred, trace=None) -> float:
    parsed = parse_answer(getattr(pred, "response", ""), gold.n_choices)
    return 1.0 if parsed == gold.target else 0.0


def feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA metric. Returns a score, plus text for the reflection model to read.

    The feedback names the failure concretely — what was chosen, what was
    correct, and which subject it came from — because a reflection model given
    only 'wrong' has nothing to reason from. This sends dev-split questions and
    gold answers to the reflection LM, which is the self-hosted Qwen3-8B; no
    benchmark content leaves the cluster.
    """
    raw = getattr(pred, "response", "") or ""
    parsed = parse_answer(raw, gold.n_choices)
    ok = parsed == gold.target
    if pred_name is None:
        return 1.0 if ok else 0.0

    if not parsed:
        detail = (
            "The reply contained no parsable 'ANSWER: X' line, so it scored wrong "
            f"regardless of content. The reply began: {raw[:200]!r}"
        )
    elif ok:
        detail = f"Correct — chose {parsed}."
    else:
        detail = (
            f"Chose {parsed}; the correct answer was {gold.target}, whose text is "
            f"{gold.gold_text!r}. Subject: {gold.subject}."
        )
    return dspy.Prediction(score=1.0 if ok else 0.0, feedback=detail)


def instructions_of(program: dspy.Module) -> str:
    return program.answer.signature.instructions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--optimizer", choices=["gepa", "copro"], default="gepa")
    parser.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-metric-calls", type=int, default=0,
                        help="overrides --auto when > 0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--thinking", action="store_true",
                        help="run the task model with native thinking (expensive)")
    args = parser.parse_args()

    task_lm = dspy.LM(
        f"openai/{MODEL}", api_base=TASK_URL, api_key="EMPTY", model_type="chat",
        temperature=0.6 if args.thinking else 0.0,
        max_tokens=args.max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": bool(args.thinking)}},
    )
    reflect_lm = dspy.LM(
        f"openai/{MODEL}", api_base=REFLECT_URL, api_key="EMPTY", model_type="chat",
        temperature=1.0, max_tokens=8000,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    dspy.configure(lm=task_lm, adapter=ServingAdapter())

    train = build_examples(args.splits / "train.jsonl")
    val = build_examples(args.splits / "val.jsonl")
    args.out.mkdir(parents=True, exist_ok=True)

    program = Program()
    evaluate = dspy.Evaluate(
        devset=val, metric=score_only, num_threads=args.threads,
        display_progress=True, display_table=0,
    )
    before = evaluate(program)
    before_score = getattr(before, "score", before)
    print(json.dumps({"stage": "seed", "val_accuracy": round(float(before_score) / 100, 4),
                      "instructions": instructions_of(program)}, ensure_ascii=False), flush=True)

    if args.optimizer == "gepa":
        kwargs = dict(metric=feedback_metric, reflection_lm=reflect_lm,
                      num_threads=args.threads, track_stats=True,
                      candidate_selection_strategy="pareto",
                      reflection_minibatch_size=6, seed=0)
        if args.max_metric_calls:
            kwargs["max_metric_calls"] = args.max_metric_calls
        else:
            kwargs["auto"] = args.auto
        optimizer = GEPA(**kwargs)
        optimized = optimizer.compile(program, trainset=train, valset=val)
    else:
        optimizer = COPRO(
            prompt_model=reflect_lm, metric=score_only, breadth=8, depth=3,
            init_temperature=1.0, track_stats=True,
        )
        optimized = optimizer.compile(
            program, trainset=train,
            eval_kwargs={"num_threads": args.threads, "display_progress": True},
        )

    after = evaluate(optimized)
    after_score = getattr(after, "score", after)
    final = instructions_of(optimized)

    result = {
        "stage": "done",
        "optimizer": args.optimizer,
        "thinking": args.thinking,
        "val_accuracy_seed": round(float(before_score) / 100, 4),
        "val_accuracy_optimized": round(float(after_score) / 100, 4),
        "val_delta_pp": round(float(after_score) - float(before_score), 2),
        "train_rows": len(train), "val_rows": len(val),
        "seed_instructions": SEED_INSTRUCTIONS,
        "optimized_instructions": final,
        "optimized_chars": len(final),
    }
    (args.out / "system_prompt.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "system_prompt.txt").write_text(final, encoding="utf-8")
    optimized.save(str(args.out / "program.json"))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
