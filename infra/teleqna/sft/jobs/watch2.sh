#!/usr/bin/env bash
# Wait for the arm E evals and the blind judge.
L=~/projects/telelogs/runs/teleqna-sft/logs/landscape
for i in $(seq 1 180); do
  grep -aq "EVAL otfull_armEhigh DONE" $L/eval_armEhigh.log 2>/dev/null && a=1 || a=0
  grep -aq "TRAIN armE_wide DONE"      $L/train_armE_wide.log 2>/dev/null && b=1 || b=0
  grep -aq "EVAL otfull_armEwide DONE" $L/eval_armEwide.log 2>/dev/null && c=1 || c=0
  grep -aq "BLIND REPARSE DONE"        $L/blind_judge.log 2>/dev/null && d=1 || d=0
  grep -aq "leak rule"                 /tmp/corpscan3.log 2>/dev/null && e=1 || e=0
  [ $((i % 15)) -eq 1 ] && echo "[$(date -Iseconds)] eval_high=$a train_wide=$b eval_wide=$c blind=$d corpscan=$e"
  [ "$a" = 1 ] && [ "$b" = 1 ] && { echo "ARM E HIGH EVAL + WIDE TRAIN DONE $(date -Iseconds) blind=$d corpscan=$e"; exit 0; }
  sleep 60
done
echo "watch2 timed out"
