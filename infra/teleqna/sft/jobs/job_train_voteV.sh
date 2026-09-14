#!/usr/bin/env bash
# Arm V: SFT the 32-sample cross-mode vote into one greedy pass. ALPHA=0 on
# purpose -- there is no evidence window here, so there is no teacher to distil
# from; this is the CE control that every previous arm was missing, run on data
# that carries no answer key.
#
# Checkpoints every 100 steps because every campaign on this project has been
# dose-dependent in the wrong direction: CPT1 was below base at all 8 probes,
# and the S-arms peaked early. A dose curve is the only way to find out which
# side of that this one falls on.
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
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
echo "card $CUDA_VISIBLE_DEVICES free=${free}MiB $(date -Iseconds)"
[ "$free" -gt 100000 ] || { echo "ABORT: card busy"; exit 14; }
env MAXLEN=1024 MAXCOMP=64 BATCH=16 ACCUM=2 EPOCHS=2 LR=1e-4 LORA_R=64 \
    ALPHA=0.0 SAVE_STEPS=100 SAVE_LIMIT=8 ALLOW_OVERWRITE=1 \
    TRAIN_DATA="$R/data/train/eligible/vote_distill.jsonl" RUN_NAME=armV \
    "$PY" -u infra/train_ctxdistill.py
echo "#### TRAIN armV DONE rc=$? $(date -Iseconds)"
