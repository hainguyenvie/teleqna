#!/usr/bin/env bash
L=~/projects/telelogs/runs/teleqna-sft/logs/landscape
for i in $(seq 1 150); do
  grep -aq "arm I EVAL DONE" $L/card4_chain.log 2>/dev/null && a=1 || a=0
  grep -aq "arm H TRAIN DONE" $L/card4_chain.log 2>/dev/null && b=1 || b=0
  grep -aq "arm H EVAL DONE"  $L/card4_chain.log 2>/dev/null && c=1 || c=0
  [ $((i % 10)) -eq 1 ] && {
    echo "[$(date -Iseconds)] armI_eval=$a armH_train=$b armH_eval=$c"
    grep -haoE "[0-9]+/(252|10000) \[[0-9:]+<[0-9:]+" $L/card4_chain.log 2>/dev/null | tail -1
  }
  [ "$a" = 1 ] && { echo "ARM I EVAL DONE $(date -Iseconds)"; exit 0; }
  sleep 45
done
echo "watch5 timed out"
