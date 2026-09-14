#!/usr/bin/env bash
# Wait for the three GPU jobs to land, printing a status line every 15 minutes so
# the run is legible afterwards, and exit as soon as all three are done.
#
# grep -q, not grep -c: `c=$(grep -ac X f || echo 0)` yields the literal string
# "0\n0" when the file is missing, and `[ "$c" != 0 ]` is then TRUE, so the first
# version of this exited immediately claiming everything was finished.
L=~/projects/telelogs/runs/teleqna-sft/logs/landscape
done_if() { grep -aq "$2" "$1" 2>/dev/null; }
for i in $(seq 1 120); do
  done_if $L/eval_armV2.log      "EVAL otfull_armV2 DONE" && v2=1 || v2=0
  done_if $L/train_armE_high.log "TRAIN armE_high DONE"   && eh=1 || eh=0
  done_if $L/train_armE_wide.log "TRAIN armE_wide DONE"   && ew=1 || ew=0
  done_if /tmp/blind_retr.log    "TRIM DONE"              && br=1 || br=0
  done_if /tmp/corpscan3.log     "leak rule"              && cs=1 || cs=0
  if [ $((i % 15)) -eq 1 ]; then
    echo "[$(date -Iseconds)] armV2eval=$v2 armE_high=$eh armE_wide=$ew blind=$br corpscan=$cs"
    grep -haoE "[0-9]+/(214|252|10000) \[[0-9:]+<[0-9:]+" \
      $L/train_armE_high.log $L/train_armE_wide.log $L/eval_armV2.log 2>/dev/null | tail -3
  fi
  if [ "$v2" = 1 ] && [ "$eh" = 1 ] && [ "$ew" = 1 ]; then
    echo "ALL THREE GPU JOBS DONE $(date -Iseconds)  blind=$br corpscan=$cs"
    exit 0
  fi
  sleep 60
done
echo "watch timed out after 2h"
