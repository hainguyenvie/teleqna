#!/bin/bash
# Twin probe run 2/2: LoRA r=64 on train_v3.jsonl. Compared against r=64 on
# dev-1000; if r=64 shows no advantage the task is not capacity-bound.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PY="$BASE_ROOT/venvs/venv-train/bin/python"
export BASE_MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export PROJ_ROOT="$ROOT" TRAIN_DATA="$ROOT/data/train_v3.jsonl"
export LORA_R=64 RUN_NAME=r64 EPOCHS=2 LR=1e-4 BATCH=16 ACCUM=2 MAXLEN=1024
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free=${free_mib}MiB"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: GPU not free"; exit 14; }

echo "#### teleqna-sft r64 START $(date -Iseconds)"
timeout 21600 "$PY" "$ROOT/infra/train_teleqna.py"
echo "#### teleqna-sft r64 DONE $(date -Iseconds)"

echo "#### eval dev-1000 r64 START $(date -Iseconds)"
timeout 7200 "$PY" "$ROOT/infra/eval_dev.py" \
  --base "$BASE_MODEL" \
  --adapter "$ROOT/models/teleqna-r64-adapter" \
  --data "$ROOT/data/dev1000.jsonl" \
  --out "$ROOT/results/dev1000_r64.json"
echo "#### eval dev-1000 r64 DONE $(date -Iseconds)"
