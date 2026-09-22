#!/usr/bin/env bash
# Pilot: generate views with OTel-31B on GEN card, then score every view set with Qwen3-8B on EVAL card.
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b
PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GENCARD="${GENCARD:-5}"; EVALCARD="${EVALCARD:-6}"
cd "$ROOT"; mkdir -p code/kit data/kit
if ! grep -q KIT_PILOT_DONE logs/kit_pilot_gen.log 2>/dev/null; then
  echo "#### GEN START card=$GENCARD $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$GENCARD timeout --foreground 21600 "$PY" -u code/kit/gen_kit_pilot.py --n 300 > logs/kit_pilot_gen.log 2>&1
  grep -q KIT_PILOT_DONE logs/kit_pilot_gen.log || { echo "ABORT: generation failed"; tail -5 logs/kit_pilot_gen.log; exit 1; }
  echo "#### GEN DONE $(date -Iseconds)"
fi
"$PY" -u code/kit/build_kit_eval.py || { echo "ABORT: build failed"; exit 1; }
for f in data/eval/kit_*.jsonl; do
  ARM=$(basename "$f" .jsonl); OUT=results/landscape/${ARM}_q3_8b_base_nothink512.json
  [ -f "$OUT" ] && { echo "skip $ARM"; continue; }
  echo "#### $ARM START $(date -Iseconds)"
  CUDA_VISIBLE_DEVICES=$EVALCARD timeout --foreground 3600 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" \
    --data "$f" --out-dir results/landscape --tag "${ARM}_q3_8b" --with-base --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 16384 > logs/${ARM}_eval.log 2>&1
  echo "#### $ARM DONE rc=$? $(date -Iseconds)"
done
echo "#### KIT_PILOT_ALL_DONE $(date -Iseconds)"
