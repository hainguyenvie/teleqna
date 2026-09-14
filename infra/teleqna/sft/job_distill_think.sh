#!/usr/bin/env bash
# Track B, thinking arm — the measurement the whole track turns on.
#
# Fixes two faults in the previous sweep, both mine:
#   * timeout was 10800s. This model is verbose, so a large share of its
#     rollouts run to the full 6000-token budget and a 1,000-row thinking eval
#     takes ~3.4h at ~5 rows/min. step25_think was killed at the 3h mark and
#     reported only as "FAILED", because `timeout ... || echo FAILED` throws the
#     exit code away. 124 and a real crash looked identical.
#   * checkpoints were swept oldest-first. The no-think ladder says later is
#     better (step25 63.20 with 134 unparsed, step100 69.00 with 5), so the
#     final checkpoint is the one worth 3.4h, not the first.
#
# Order is deliberate: step236 first, then step100. If step236 does not clear
# base thinking 75.10 there is no reason to spend another 3.4h on step100.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
BASE=/workspace/telelogs-base
ROOT=/workspace/teleqna-spare4
INFRA="$BASE/runs/teleqna-sft/infra"
PYT="$BASE/venvs/venv-train/bin/python"
MODEL="$BASE/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DEV="$BASE/runs/teleqna-sft/data/dev1000.jsonl"
RUN="${RUN:-distill1}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Wait rather than abort. The previous eval was killed mid-run and CUDA takes a
# while to hand the memory back; aborting here just means re-dropping the job by
# hand later. Capped so this can never sit on the queue forever, and it still
# refuses to start on a card someone else has taken for good.
for i in $(seq 1 120); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free_mib" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "node card 0 busy (${free_mib}MiB free) — waiting $(date -Iseconds)"
  sleep 30
done
echo "node card 0 free=${free_mib}MiB  $(date -Iseconds)"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: node card 0 still busy after 60min"; exit 14; }

for step in 236 100; do
  CK="$ROOT/models/teleqna-$RUN-checkpoints/checkpoint-$step"
  [ -d "$CK" ] || { echo "missing $CK"; continue; }
  OUT="$ROOT/results/dev1000_${RUN}_step${step}_think.json"
  if [ -e "$OUT" ]; then echo "skip $OUT"; continue; fi
  echo "#### $RUN step $step thinking START $(date -Iseconds)"
  set +e
  timeout 28800 "$PYT" "$INFRA/eval_dev.py" \
    --base "$MODEL" --adapter "$CK" --thinking --batch 32 \
    --data "$DEV" --out "$OUT"
  rc=$?
  set -e
  # Report the code. 124 is the timeout, 137 an OOM kill, anything else a real
  # crash — the previous sweep collapsed all three into the word FAILED.
  if [ $rc -ne 0 ]; then echo "#### step $step FAILED rc=$rc $(date -Iseconds)"; continue; fi
  "$PYT" - "$OUT" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
by = s.get("by_subject", {})
print(f"  -> {sys.argv[1].split('/')[-1]}  acc={s['accuracy']:.4f} "
      f"unparsed={s['unparsed']}", flush=True)
for k, v in sorted(by.items()):
    print(f"       {k:<28s} n={v['n']:<4d} acc={v['acc']:.4f}", flush=True)
PY
done
echo "#### THINK SWEEP DONE $(date -Iseconds)"
