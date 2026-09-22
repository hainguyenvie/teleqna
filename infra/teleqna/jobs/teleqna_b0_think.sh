#!/bin/bash
# Same two arms with Qwen3 native thinking enabled.
#
# On srsranbench thinking *cost* 4.8 points (77.70% vs 82.49% no-think) at 690
# tokens/case against 5. teleqna is knowledge recall in the same way, so the
# expectation is the same — but Standards specifications questions are longer
# and more clause-like than srsRAN identifier lookups, so this is the subject
# where thinking could plausibly pay. The per-subject breakdown in summary.json
# is the thing to read, not the headline number.
#
# max_tokens is set high enough that truncation, if it appears in the summary's
# `truncated` count, is the model's doing and not the cap's. That failure mode
# already bit this repo (commit 09fa2ed, 299/320 collection failures) and it is
# the most likely explanation for the sub-random leaderboard rows.
#
# temperature 0.6 / top_p 0.95 / top_k 20 is Qwen3's recommended thinking
# sampling and is what full4_eval.py used for the published 4-benchmark base.
set -euo pipefail
ROOT=/workspace/telelogs-bench4/teleqna
PY=/workspace/telelogs-bench4/dspy/.venv/bin/python
cd "$ROOT/code"

"$PY" run_baseline.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/results/b0_think" \
  --thinking \
  --workers 24 --max-tokens 6000 --temperature 0.6

"$PY" run_baseline.py \
  --data "$ROOT/data/test.jsonl" \
  --out "$ROOT/results/b0_think_perm" \
  --thinking --permute \
  --workers 24 --max-tokens 6000 --temperature 0.6
