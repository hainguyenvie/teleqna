#!/usr/bin/env bash
# Arm J on card 4: train, then score on the full 10,000.
#
# The number that decides this arm is not the headline. It is whether the 203
# rows only arm H reached survive alongside arm G-full's 1,021 -- and whether
# arm H's 926 broken rows drop back toward arm G-full's 357 now that synthetic
# questions are 30% of the training set instead of all of it.
set -uo pipefail
export CARDS=4
SFT=/home/tensara/projects/telelogs/runs/teleqna-sft

bash /tmp/job_train_armJ.sh
echo "#### arm J TRAIN DONE $(date -Iseconds)"

TAG=otfull_armJ RANK=64 \
  ADAPTERS="armJ=$SFT/models/ctxdistill-armJ-adapter" \
  bash /tmp/job_eval_arms.sh
echo "#### arm J EVAL DONE $(date -Iseconds)"
