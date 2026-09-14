#!/bin/bash
# DPO run 1: 24,063 pairs (student's real wrong picks), LoRA r16, beta 0.2.
# Sweeps dev-1000 across every checkpoint afterwards, same as the SFT runs.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PY="$BASE_ROOT/venvs/venv-train/bin/python"
export BASE_MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export PROJ_ROOT="$ROOT" TRAIN_DATA="$ROOT/data/dpo_pairs.jsonl"
export LORA_R=16 RUN_NAME=dpo1 BETA=0.2 EPOCHS=1 LR=1e-5 BATCH=16 ACCUM=2
export SAVE_STEPS=150
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free=${free_mib}MiB"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: GPU not free"; exit 14; }

echo "#### teleqna-dpo1 TRAIN START $(date -Iseconds)"
timeout 21600 "$PY" "$ROOT/infra/train_dpo.py"
echo "#### teleqna-dpo1 TRAIN DONE $(date -Iseconds)"

for ckpt in "$ROOT"/models/teleqna-dpo1-checkpoints/checkpoint-* "$ROOT"/models/teleqna-dpo1-adapter; do
  [ -d "$ckpt" ] || continue
  name=$(basename "$ckpt" | sed 's/checkpoint-/step/; s/teleqna-dpo1-adapter/final/')
  out="$ROOT/results/dev1000_dpo1_${name}.json"
  [ -e "$out" ] && continue
  echo "#### eval dev-1000 @$name"
  timeout 3600 "$PY" "$ROOT/infra/eval_dev.py" \
    --base "$BASE_MODEL" --adapter "$ckpt" \
    --data "$ROOT/data/dev1000.jsonl" --out "$out"
done
echo "#### teleqna-dpo1 ALL DONE $(date -Iseconds)"
