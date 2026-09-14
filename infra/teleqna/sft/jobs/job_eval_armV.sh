#!/usr/bin/env bash
# Wait for arm V to finish training, then score base + adapter on all 10,000 in
# one model load. One run covers every read we need: the 6,427 rows it trained
# on, the 1,000 of dev1000 and the 500 of clean_holdout500 it never saw. The gap
# between those is the whole question -- knowledge or memorised questions.
set -uo pipefail
HOME_ROOT=/home/tensara
R="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
LOG="$R/logs/landscape/train_armV.log"
ADP="$R/models/ctxdistill-armV-adapter"
t=0
until grep -q "TRAIN armV DONE" "$LOG" 2>/dev/null; do
  t=$((t+1)); [ $t -gt 180 ] && { echo "ABORT: training still running after 3h"; exit 1; }
  sleep 60
done
grep "TRAIN armV DONE" "$LOG"
[ -d "$ADP" ] || { echo "ABORT: no adapter at $ADP"; exit 2; }
# The training card is the one that just freed; wait for its memory to come back.
export CUDA_VISIBLE_DEVICES="${CARDS:-6}"
t=0
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")" -gt 100000 ]; do
  t=$((t+1)); [ $t -gt 30 ] && { echo "ABORT: card never freed"; exit 14; }
  sleep 20
done
echo "scoring armV on ot-full, card $CUDA_VISIBLE_DEVICES $(date -Iseconds)"
"$PY" "$R/infra/eval_dev_vllm.py" \
  --base "$MODEL" --data "$R/data/otfull10000.jsonl" \
  --out-dir "$R/results/landscape" --tag otfull_armV \
  --adapter "armV=$ADP" --with-base --max-new 512 \
  --tp 1 --gpu-mem 0.90 --max-model-len 4096 --max-lora-rank 64
echo "#### EVAL armV DONE rc=$? $(date -Iseconds)"
