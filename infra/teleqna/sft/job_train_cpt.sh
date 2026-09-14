#!/bin/bash
# CPT campaign 1: ~145M tokens (100M error-targeted TCC + 20M breadth + 25M
# spec chunks), LoRA r=64, lr 1e-4, seq 4096, ~3% MCQ format anchors.
# ETA ~6-8h train on one H200, then dev-1000 no-think eval per checkpoint
# (~4 min each). Think eval of the winner is a separate decision.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PY="$BASE_ROOT/venvs/venv-train/bin/python"
export BASE_MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
export PROJ_ROOT="$ROOT" CPT_CORPUS="$ROOT/data/cpt_corpus.jsonl"
export MCQ_ANCHORS="$ROOT/data/cpt_mcq_anchors.jsonl" ANCHOR_EVERY=30
export LORA_R=64 RUN_NAME=cpt1 SEQ=4096 BATCH=8 ACCUM=4 EPOCHS=1 LR=1e-4
export SAVE_STEPS=200
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free=${free_mib}MiB"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: GPU not free"; exit 14; }

echo "#### teleqna-cpt1 TRAIN START $(date -Iseconds)"
timeout 57600 "$PY" "$ROOT/infra/train_cpt.py"
echo "#### teleqna-cpt1 TRAIN DONE $(date -Iseconds)"

for ckpt in "$ROOT"/models/teleqna-cpt1-checkpoints/checkpoint-* "$ROOT"/models/teleqna-cpt1-adapter; do
  [ -d "$ckpt" ] || continue
  name=$(basename "$ckpt" | sed 's/checkpoint-/step/; s/teleqna-cpt1-adapter/final/')
  out="$ROOT/results/dev1000_cpt1_${name}.json"
  [ -e "$out" ] && continue
  echo "#### eval dev-1000 @$name"
  timeout 3600 "$PY" "$ROOT/infra/eval_dev.py" \
    --base "$BASE_MODEL" --adapter "$ckpt" \
    --data "$ROOT/data/dev1000.jsonl" --out "$out"
done
echo "#### teleqna-cpt1 ALL DONE $(date -Iseconds)"
