#!/bin/bash
# Memorisation probe. Runs before any training decision, because TeleQnA has
# been public since October 2023 and Qwen3-8B was pretrained after that.
#
# Five arms on one stratified 600-row sample, paired row-for-row:
#   probe_orig         control
#   probe_perm         choices rotated          -> letter/order recall
#   probe_distract     foreign distractors      -> control, expect accuracy UP
#   probe_paraphrase   stem rewritten locally   -> surface-form dependence
#   tsguess_distractor blank a WRONG option     -> DIRECT evidence
#   tsguess_gold       blank the RIGHT option   -> comparison only
#
# Decoding matches the b0_nothink baseline exactly (greedy, 32 tokens, no
# thinking) so the probe arms and the headline number are on the same footing.
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

PROBE="$ROOT/data/probe"

# Build. The paraphrase arm calls the served model; everything else is offline
# and deterministic given --seed.
"$PY" make_probe_set.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$PROBE" \
  --size 600 --seed 20260803

for arm in orig perm distract paraphrase; do
  "$PY" run_baseline.py \
    --data "$PROBE/$arm.jsonl" \
    --out "$ROOT/results/probe_$arm" \
    --workers 16 --max-tokens 32 --temperature 0.0
done

"$PY" run_tsguess.py \
  --data "$PROBE/orig.jsonl" \
  --out "$ROOT/results/tsguess_distractor" \
  --mask distractor --workers 16

"$PY" run_tsguess.py \
  --data "$PROBE/orig.jsonl" \
  --out "$ROOT/results/tsguess_gold" \
  --mask gold --workers 16

"$PY" compare_probe.py \
  --results-root "$ROOT/results" \
  --out "$ROOT/results/probe_report.json"
