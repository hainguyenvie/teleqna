#!/usr/bin/env bash
# Wait for the three blind-judge shards.
L=~/projects/telelogs/runs/teleqna-sft/logs/landscape
for i in $(seq 1 120); do
  n=0
  for s in 0 1 2; do grep -aq "BLIND SHARD $s DONE" $L/blind_sh$s.log 2>/dev/null && n=$((n+1)); done
  grep -aq "leak rule" /tmp/corpscan3.log 2>/dev/null && cs=1 || cs=0
  if [ $((i % 10)) -eq 1 ]; then
    echo "[$(date -Iseconds)] shards done=$n/3 corpscan=$cs"
    grep -haoE "[0-9]+/206[0-9] \[[0-9:]+<[0-9:]+" $L/blind_sh0.log $L/blind_sh1.log $L/blind_sh2.log 2>/dev/null | tail -3
  fi
  [ "$n" = 3 ] && { echo "ALL BLIND SHARDS DONE $(date -Iseconds) corpscan=$cs"; exit 0; }
  sleep 60
done
echo "watch3 timed out"
