#!/usr/bin/env bash
# Everything left, strictly serialised on card 4.
#
# Written because the two jobs were about to race: arm H's wait loop would have
# claimed the card the instant arm I's training ended, pushing arm I's eval
# behind 25 minutes of arm H training. Order matters here — arm I answers whether
# rationales contribute anything, and that answer shapes how arm H's result
# should be read.
set -uo pipefail
export CARDS=4
SFT=/home/tensara/projects/telelogs/runs/teleqna-sft
L="$SFT/logs/landscape"

for i in $(seq 1 240); do
  grep -aq "TRAIN armI_wide DONE" "$L/train_armI_wide.log" 2>/dev/null && break
  [ "$i" = 1 ] && echo "waiting for arm I training $(date -Iseconds)"
  sleep 30
done
grep -aq "TRAIN armI_wide DONE" "$L/train_armI_wide.log" \
  || { echo "ABORT: arm I training never finished"; exit 15; }
echo "#### arm I trained $(date -Iseconds)"

TAG=otfull_armI RANK=64 \
  ADAPTERS="armI=$SFT/models/ctxdistill-armI_wide-adapter" \
  bash /tmp/job_eval_arms.sh
echo "#### arm I EVAL DONE $(date -Iseconds)"

bash /tmp/job_train_armH.sh
echo "#### arm H TRAIN DONE $(date -Iseconds)"

TAG=otfull_armH RANK=64 \
  ADAPTERS="armH=$SFT/models/ctxdistill-armH-adapter" \
  bash /tmp/job_eval_arms.sh
echo "#### arm H EVAL DONE $(date -Iseconds)"
