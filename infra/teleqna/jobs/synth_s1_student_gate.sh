#!/bin/bash
# Student difficulty gate over the full synthetic set.
#
# 39,827 questions survived the six deterministic gates. Those gates check
# form (schema, style, verbatim groundedness, distinctness, dedup,
# contamination); none of them checks whether a question is worth training on.
# A question the student already answers correctly carries no gradient toward
# the failure mode we measured — the distractor probe (+15.83pp) says the
# baseline loses on fine option discrimination, so the questions worth their
# tokens are the ones where discrimination actually bites.
#
# This runs the *student* (Qwen3-8B, byte-identical prompt/decoding to the
# baseline arm) over every kept question and records per-question correctness.
# The training builder then down-weights the easy ones. It also gives a second,
# cheaper read on the degenerate-hard-negative defect: distractors invented out
# of thin air ("N2s", "N2c") are easy, so they surface here as a cluster the
# student aces.
#
# Reference: the pilot (1,196 questions) scored 58.53% — 13.4 points below the
# same model's 71.95% on the real benchmark, i.e. the generated set is harder
# than the target. Harder is not automatically better, which is exactly why the
# per-question record matters more than the aggregate.
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
SYNTH=/workspace/telelogs-bench4/synth
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

"$PY" run_baseline.py \
  --data "$SYNTH/data/full_kept.jsonl" \
  --out "$SYNTH/results/s1_student_gate" \
  --workers 48 --max-tokens 32 --temperature 0.0
