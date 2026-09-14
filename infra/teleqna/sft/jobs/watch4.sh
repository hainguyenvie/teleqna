#!/usr/bin/env bash
L=~/projects/telelogs/runs/teleqna-sft/logs/landscape
for i in $(seq 1 180); do
  grep -aq "TRAIN armG DONE" $L/train_armG.log 2>/dev/null && g=1 || g=0
  grep -aq "EVAL otfull_armG DONE" $L/eval_armG.log 2>/dev/null && ge=1 || ge=0
  grep -aq "122B REPARSE DONE" $L/judge_122b.log 2>/dev/null && j=1 || j=0
  [ $((i % 10)) -eq 1 ] && {
    echo "[$(date -Iseconds)] armG_train=$g armG_eval=$ge judge122b=$j"
    grep -haoE "[0-9]+/(524|10000|3451) \[[0-9:]+<[0-9:]+" $L/train_armG.log $L/eval_armG.log $L/judge_122b.log 2>/dev/null | tail -3
  }
  [ "$g" = 1 ] && [ "$ge" = 0 ] && { echo "ARM G TRAINED $(date -Iseconds)"; exit 0; }
  sleep 60
done
echo "watch4 timed out"
