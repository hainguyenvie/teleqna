#!/bin/bash
# Closed-book teleqna baseline, no native thinking — the arm the harness's own
# prompt asks for ("the entire content of your response should be 'ANSWER: $LETTER'").
#
# Both arms run all 10,000 rows. The permuted arm is the control: on this
# benchmark there is no position prior to destroy (always-A = 22.1%), so the gap
# measures order-robustness — whether the model keys on presentation, most
# importantly on "All of the above" sitting in the last slot.
#
# Reference points to compare against (computed offline, no model):
#   random baseline                     21.89%
#   always A                            22.10%
#   distractor-shape heuristic          39.72%   <- the real floor
#   active telecom professionals        64.86%   (TeleQnA paper)
#   GPT-4 (2023)                        74.91%   (TeleQnA paper)
#   AT&T OTel-LLM-8.3B-QnA              91.20%   (leaderboard, n≈10237)
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

"$PY" run_baseline.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/results/b0_nothink" \
  --workers 32 --max-tokens 32 --temperature 0.0

"$PY" run_baseline.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/results/b0_nothink_perm" \
  --permute \
  --workers 32 --max-tokens 32 --temperature 0.0
