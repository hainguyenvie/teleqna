#!/usr/bin/env bash
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CARDS:-6}"
cd "$ROOT"
echo "#### GENDENSE START $(date -Iseconds)"
timeout --foreground 10800 "$PY" -u code/kit/gen_kit_pilot.py --n 300 --views dense --tag pilotdense > logs/kit_pilotdense_gen.log 2>&1
grep -q KIT_PILOT_DONE logs/kit_pilotdense_gen.log || { echo "ABORT gen"; tail -3 logs/kit_pilotdense_gen.log; exit 1; }
echo "#### GENDENSE DONE $(date -Iseconds)"
sed -e 's#data/kit/pilot_sample.json#data/kit/pilotdense_sample.json#; s#data/kit/pilot_views.jsonl#data/kit/pilotdense_views.jsonl#; s#data/eval/kit_{name}#data/eval/kitdense_{name}#; s#-> kit_{name}#-> kitdense_{name}#' code/kit/build_kit_eval.py > code/kit/build_kit_eval_dense.py
"$PY" -u code/kit/build_kit_eval_dense.py || { echo "ABORT build"; exit 1; }
ARM=kitdense_dense; f=data/eval/$ARM.jsonl
echo "#### $ARM START $(date -Iseconds)"
timeout --foreground 3600 "$PY" -u code/eval_dev_vllm.py --base "$HOME/projects/_shared/models/Qwen3-8B" --data "$f" \
  --out-dir results/landscape --tag "${ARM}_q3_8b" --with-base --max-new 512 --tp 1 --gpu-mem 0.90 --max-model-len 16384 > logs/${ARM}_eval.log 2>&1
echo "#### $ARM DONE rc=$? $(date -Iseconds)"
echo "#### KITDENSE_ALL_DONE"
