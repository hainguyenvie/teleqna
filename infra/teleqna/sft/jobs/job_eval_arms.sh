#!/usr/bin/env bash
# Score one or more adapters against the base on ot-full, in a single model load.
#
# Both arms in one load on purpose: the A/A noise floor on this stack is 0.50pp,
# so a comparison across two separate loads spends most of its resolution on
# engine variance. Base is included every time for the same reason -- the delta
# is the number, not the absolute.
set -uo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:?set CARDS}"
HOME_ROOT=/home/tensara
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
export PATH="$(dirname "$PY"):$PATH"
SFT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
MODEL="$HOME_ROOT/projects/telelogs/shared/models/OTel-2.0-31B-IT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SET="${SET:-otfull10000}"
TAG="${TAG:?set TAG}"
ADAPTERS="${ADAPTERS:?set ADAPTERS as name=path[,name=path]}"
RANK="${RANK:-64}"
OUT="${OUT:-$SFT/results/landscape}"

for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
  [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] || continue
  echo "reaping orphaned EngineCore pid=$pid"; kill -9 "$pid" 2>/dev/null || true; sleep 8
done

for i in $(seq 1 360); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  [ "$free" -gt 100000 ] && break
  [ "$i" = 1 ] && echo "card $CUDA_VISIBLE_DEVICES busy (${free}MiB) — waiting $(date -Iseconds)"
  sleep 20
done
[ "$free" -gt 100000 ] || { echo "ABORT: card stuck at ${free}MiB"; exit 14; }
echo "card $CUDA_VISIBLE_DEVICES free=${free}MiB $(date -Iseconds)"

AD=()
IFS=',' read -ra SPECS <<< "$ADAPTERS"
for s in "${SPECS[@]}"; do AD+=(--adapter "$s"); done

echo "#### EVAL $TAG START $(date -Iseconds)"
"$PY" -u "$SFT/infra/eval_dev_vllm.py" \
  --base "$MODEL" --data "$SFT/data/$SET.jsonl" --out-dir "$OUT" \
  --tag "$TAG" --with-base "${AD[@]}" \
  --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 4096 --max-lora-rank "$RANK"
echo "#### EVAL $TAG DONE rc=$? $(date -Iseconds)"
