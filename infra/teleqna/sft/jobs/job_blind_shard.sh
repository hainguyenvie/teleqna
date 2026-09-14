#!/usr/bin/env bash
# One shard of the blind-set evidence judging.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:?set CARDS}"
SH="${SH:?set SH}"
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for i in $(seq 1 360); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  [ "$free" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "card $CUDA_VISIBLE_DEVICES busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 15
done
[ "$free" -gt 100000 ] || { echo "ABORT: card stuck at ${free}MiB"; exit 14; }
echo "card $CUDA_VISIBLE_DEVICES free=${free}MiB shard=$SH $(date -Iseconds)"

echo "#### BLIND SHARD $SH START $(date -Iseconds)"
"$PY" -u "$SFT/infra/eval_dev_vllm.py" \
  --base "$MODEL" --data "$SFT/data/blind6202_rag8_strong_sh$SH.jsonl" \
  --out-dir "$SFT/results/landscape" --tag "blind_sh${SH}_ragstrong8" --with-base \
  --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 32768
echo "#### BLIND SHARD $SH DONE rc=$? $(date -Iseconds)"
