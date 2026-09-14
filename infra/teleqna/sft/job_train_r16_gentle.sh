#!/bin/bash
# Dose-response run after the first r16 run REGRESSED on real dev-1000
# (68.4% vs 73.8% baseline; flips 154 lost / 100 gained) while its synthetic
# eval-loss bottomed near epoch 0.8 and rose through epoch 2 — the kept
# checkpoints all sat in the overfit zone. This run lowers the dose (1 epoch,
# LR 2e-5, the setting the proven v13 run used) and keeps EVERY 300-step
# checkpoint so dev-1000 can be swept across training time: if no point on
# the curve beats baseline, the problem is the data, not the dose.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PY="$BASE_ROOT/venvs/venv-train/bin/python"
export BASE_MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export PROJ_ROOT="$ROOT" TRAIN_DATA="$ROOT/data/train_v3.jsonl"
export LORA_R=16 RUN_NAME=r16g EPOCHS=1 LR=2e-5 BATCH=16 ACCUM=2 MAXLEN=1024
export SAVE_STEPS=300 SAVE_LIMIT=100
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free=${free_mib}MiB"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: GPU not free"; exit 14; }

echo "#### teleqna-sft r16-gentle START $(date -Iseconds)"
timeout 21600 "$PY" "$ROOT/infra/train_teleqna.py"
echo "#### teleqna-sft r16-gentle TRAIN DONE $(date -Iseconds)"

# Sweep dev-1000 over every checkpoint, then the final adapter.
for ckpt in "$ROOT"/models/teleqna-r16g-checkpoints/checkpoint-*; do
  step=$(basename "$ckpt" | cut -d- -f2)
  out="$ROOT/results/dev1000_r16g_step${step}.json"
  [ -e "$out" ] && continue
  echo "#### eval dev-1000 @step $step"
  timeout 3600 "$PY" "$ROOT/infra/eval_dev.py" \
    --base "$BASE_MODEL" --adapter "$ckpt" \
    --data "$ROOT/data/dev1000.jsonl" --out "$out"
done
echo "#### teleqna-sft r16-gentle ALL DONE $(date -Iseconds)"
