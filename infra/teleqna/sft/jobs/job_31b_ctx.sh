#!/usr/bin/env bash
# Score OTel-2.0-31B-IT on a context-carrying set, one card, no-think 512.
# Same binary, same scorer, same token budget as run_otel31b.sh's nothink arm --
# only --max-model-len differs, because a retrieved prompt is ~6k tokens against
# the closed-book arm's ~200 and 4096 would silently truncate the context.
set -uo pipefail
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
SET="${SET:?set SET}"; TAG="${TAG:-$SET}"; OUT="${OUT:-$SFT/results/landscape}"
MAXLEN="${MAXLEN:-12288}"
export CUDA_VISIBLE_DEVICES="${CARDS:?set CARDS}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
echo "card $CUDA_VISIBLE_DEVICES free=${free}MiB  set=$SET maxlen=$MAXLEN $(date -Iseconds)"
[ "$free" -gt 100000 ] || { echo "ABORT: card busy"; exit 14; }
"$PY" "$SFT/infra/eval_dev_vllm.py" \
  --base "$MODEL" --data "$SFT/data/$SET.jsonl" --out-dir "$OUT" \
  --tag "$TAG" --with-base --max-new 512 \
  --tp 1 --gpu-mem 0.90 --max-model-len "$MAXLEN"
echo "#### DONE rc=$? $(date -Iseconds)"
