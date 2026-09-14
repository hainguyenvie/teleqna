#!/usr/bin/env bash
# Score the three weight-space compositions of arm G-full and arm H in one load.
#
# s=0 is arm G-full reproduced exactly, so it is not re-run; 87.24 is the line
# these three have to beat. What is being tested is whether arm H's 203 unique
# rows can be dialled in without its drift coming back with them.
set -uo pipefail
export CARDS=4
SFT=/home/tensara/projects/telelogs/runs/teleqna-sft

TAG=otfull_mergeGH RANK=128 \
  ADAPTERS="m03=$SFT/models/merge-GH-0.3,m06=$SFT/models/merge-GH-0.6,m10=$SFT/models/merge-GH-1.0" \
  bash /tmp/job_eval_arms.sh
echo "#### merge EVAL DONE $(date -Iseconds)"
