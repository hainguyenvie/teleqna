#!/bin/bash
# Stage 1 of prompt optimisation: evolve a serving-side system prompt with GEPA,
# no-think task model.
#
# no-think first because it costs 5 tokens a question against 968, so the
# optimiser can afford hundreds of candidate evaluations. The winning prompt is
# then re-tested under thinking separately — a prompt that only helps the cheap
# arm is still worth knowing about, and one that helps both is the one to ship.
#
# The optimiser only ever sees train (400) + val (600). The number that gets
# reported comes from heldout (9,000), scored by run_baseline.py — the same code
# path, template and parser as the baseline, so the comparison is like for like.
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

"$PY" make_splits.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/data/splits" \
  --train 400 --val 600 --seed 20260804

"$PY" dspy_optimize_prompt.py \
  --splits "$ROOT/data/splits" \
  --out "$ROOT/results/gepa_nothink" \
  --optimizer gepa --auto light \
  --threads 16 --max-tokens 40
