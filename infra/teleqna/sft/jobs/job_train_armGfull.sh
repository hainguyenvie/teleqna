#!/usr/bin/env bash
# Arm E: train on grounded facts with an explicit hard negative, not on letters.
#
# Same trainer, same LoRA rank, same LR and epochs as arm V, so the difference
# between the two is the TARGET and nothing else -- arm V learned "the letter is
# B", arm E learns "<fact>. This is B) ..., not C) ...". Arm V fit its labels to
# 99.91% and transferred nothing; if the fact form also transfers nothing, the
# weights route is finished and I want that answer cleanly rather than confounded
# by a hyperparameter change.
#
# MAXCOMP goes 64 -> 256 because the target is now a sentence, and silently
# truncating it would cut off the ANSWER: line that the whole thing exists to
# produce.
set -uo pipefail
HOME_ROOT=/home/tensara
R="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROJ_ROOT="$R"
export BASE_MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
export CUDA_VISIBLE_DEVICES="${CARDS:?set CARDS}"

cd "$R"
for i in $(seq 1 180); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  [ "$free" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "card $CUDA_VISIBLE_DEVICES busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 20
done
[ "$free" -gt 100000 ] || { echo "ABORT: card stuck at ${free}MiB"; exit 14; }
echo "card $CUDA_VISIBLE_DEVICES free=${free}MiB $(date -Iseconds)"

env MAXLEN=1280 MAXCOMP=256 BATCH=16 ACCUM=2 EPOCHS=2 LR=1e-4 LORA_R=64 \
    ALPHA=0.0 SAVE_STEPS=50 SAVE_LIMIT=8 ALLOW_OVERWRITE=1 \
    TRAIN_DATA="$R/data/train/eligible/armG_full.jsonl" RUN_NAME="armG_full" \
    "$PY" -u infra/train_ctxdistill.py
echo "#### TRAIN armG DONE rc=$? $(date -Iseconds)"
