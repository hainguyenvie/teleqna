#!/bin/bash
# Calibration for the TS-Guessing probe.
#
# The pre-registered threshold (>2% distractor exact-match => contamination)
# assumed that reproducing a wrong option is only explainable by having seen the
# row. The length breakdown falsifies that assumption for short options: the
# distractor exact-match rate is 15.87% at one word and 1.55% at 7-12 words, and
# a monotone decay with length is the signature of guessing, not of recall.
#
# Rather than move the threshold by argument, measure the guessing rate directly.
#
# probe_distract's rows carry gold text from their own question and distractors
# transplanted from OTHER questions in the same subject. That option SET never
# existed in any corpus — my script built it. So masking a transplanted
# distractor and asking the model to fill it in cannot be answered from recall
# of this row; whatever it scores is what pure context-guessing scores, at the
# same length distribution as the real probe.
#
# Read the result as: real_probe_rate - calibration_rate, per length band.
# A difference at or below zero means the 4.84% headline was entirely guessing.
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

"$PY" run_tsguess.py \
  --data "$ROOT/data/probe/distract.jsonl" \
  --out "$ROOT/results/tsguess_calibration" \
  --mask distractor --workers 16
