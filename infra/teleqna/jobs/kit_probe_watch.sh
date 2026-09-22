#!/usr/bin/env bash
# Probe the base once, then every new checkpoint (step*/ep*) as it appears. Usage: CARDS=5 bash jobs/kit_probe_watch.sh
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CARDS:?}"
cd "$ROOT"; mkdir -p results/kit
[ -f results/kit/probe_base.json ] || { echo "#### PROBE base START $(date -Iseconds)"; "$PY" -u code/kit/probe_logprob.py --build --ckpt "$HOME/projects/_shared/models/Qwen3-8B" --tag base > logs/probe_base.log 2>&1; tail -2 logs/probe_base.log; }
while true; do
  for d in $(ls -d models/kit/tier1/step* models/kit/tier1/ep* 2>/dev/null); do
    tag=$(basename "$d"); [ -f "results/kit/probe_$tag.json" ] && continue
    [ -f "$d/config.json" ] && [ -f "$d/model.safetensors.index.json" -o -f "$d/model.safetensors" ] || continue
    sleep 60   # let the save finish
    echo "#### PROBE $tag START $(date -Iseconds)"; "$PY" -u code/kit/probe_logprob.py --ckpt "$d" --tag "$tag" > logs/probe_$tag.log 2>&1; tail -2 logs/probe_$tag.log
    "$PY" code/kit/probe_report.py
  done
  grep -q "TIER1_CHAIN_DONE\|ABORT" logs/tier1_chain.log 2>/dev/null && [ -f results/kit/probe_ep2.json ] && break
  sleep 300
done
echo "#### PROBE WATCH DONE"
