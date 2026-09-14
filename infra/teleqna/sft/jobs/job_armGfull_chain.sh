#!/usr/bin/env bash
# Branch A end to end: wait for the last evidence, build the full-coverage label
# set, train, score.
#
# Chained rather than launched by hand so the card is claimed the moment it is
# free. Each step checks its input before running -- an earlier version of this
# pattern happily started a run over a half-written file.
set -uo pipefail
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
L="$SFT/logs/landscape"
CARDS="${CARDS:?set CARDS}"
export CARDS

for i in $(seq 1 240); do
  grep -aq "HOLDOUT REPARSE DONE" "$L/holdout_judge.log" 2>/dev/null && break
  [ "$i" = 1 ] && echo "waiting for the holdout evidence judge $(date -Iseconds)"
  sleep 30
done
grep -aq "HOLDOUT REPARSE DONE" "$L/holdout_judge.log" \
  || { echo "ABORT: holdout judge never finished"; exit 15; }
echo "holdout evidence ready $(date -Iseconds)"

cd "$SFT"
"$PY" -u infra/build_armG.py --include-holdout \
  --out "$SFT/data/train/eligible/armG_full.jsonl"
n=$(wc -l < "$SFT/data/train/eligible/armG_full.jsonl")
echo "armG_full rows = $n"
[ "$n" -ge 9800 ] || { echo "ABORT: expected ~10000 rows, got $n"; exit 16; }

bash /tmp/job_train_armGfull.sh
echo "#### armG_full TRAIN CHAIN DONE $(date -Iseconds)"

TAG=otfull_armGfull RANK=64 \
  ADAPTERS="armGfull=$SFT/models/ctxdistill-armG_full-adapter" \
  bash /tmp/job_eval_arms.sh
echo "#### armG_full EVAL DONE $(date -Iseconds)"
