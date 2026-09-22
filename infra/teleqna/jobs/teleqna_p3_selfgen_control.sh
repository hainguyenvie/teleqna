#!/bin/bash
# Valid guessability control for TS-Guessing. See make_selfgen_control.py for
# why the p2 transplant control was not one (it scored 0/599 by removing
# guessability, not by isolating memorisation).
#
# Generate one fresh distractor per row, length-matched, written by the served
# model itself — plausible for the question, impossible to have been memorised.
# Then mask exactly that option and ask the model to reconstruct it.
#
# The resulting rate is an UPPER BOUND on guessability. Compare per length band
# against results/tsguess_distractor.
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

"$PY" make_selfgen_control.py \
  --data "$ROOT/data/probe/orig.jsonl" \
  --out "$ROOT/data/probe/selfgen.jsonl" \
  --workers 16

"$PY" run_tsguess.py \
  --data "$ROOT/data/probe/selfgen.jsonl" \
  --out "$ROOT/results/tsguess_selfgen" \
  --mask distractor --workers 16
