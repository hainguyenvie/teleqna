#!/usr/bin/env bash
# Hard-negative mining on card 0, alongside the 122B generation on 4+6 and the
# rerank on 1.
#
# PATH must carry the venv bin. vLLM shells out to ninja while compiling the
# CUDA graphs, and a bare `python /path/to/venv/bin/python` does not put the
# venv's bin on PATH - the engine gets through weight loading and graph capture
# and only then dies with FileNotFoundError: 'ninja', which reads like a model
# problem and is not.
set -euo pipefail
HOME_ROOT=/home/tensara
ROOT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CARD="${CARD:-0}"
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CARD")
echo "card $CARD free=${free}MiB"
[ "$free" -gt 100000 ] || { echo "ABORT: card $CARD has only ${free}MiB"; exit 14; }
export CUDA_VISIBLE_DEVICES="$CARD"

LIMIT="${LIMIT:-1500}"
ITEMS="${ITEMS:-$ROOT/data/synth/pilot2k_kept.jsonl}"
TAG="${TAG:-pilot}"

echo "#### mine-neg START $(date -Iseconds) items=$ITEMS limit=$LIMIT"
"$PY" "$ROOT/infra/mine_hard_negatives.py" \
  --items "$ITEMS" \
  --model "$MODEL" \
  --out "$ROOT/data/synth/mined_neg_${TAG}.jsonl" \
  --report "$ROOT/results/rag/mined_neg_${TAG}.json" \
  --limit "$LIMIT" \
  --gpu-mem 0.90 --max-model-len 4096
echo "#### mine-neg DONE rc=$? $(date -Iseconds)"
