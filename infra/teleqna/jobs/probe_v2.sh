#!/usr/bin/env bash
# Build the clean held-out (v2) and re-probe base + existing checkpoints into results/kit/v2 (card 7, alongside the watcher).
set -uo pipefail
ROOT=$HOME/projects/teleqna/runs/teleqna-8b; PY=$HOME/venv-vllm-nightly/bin/python
export PATH="$(dirname "$PY"):$PATH" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CARDS:-7}"
cd "$ROOT"
[ -f data/kit/tier1/probes_v2.json ] || { "$PY" -u code/kit/probe_heldout_v2.py > logs/probe_heldout_v2.log 2>&1; grep -E "questions with|probes_v2" logs/probe_heldout_v2.log; }
[ -f data/kit/tier1/probes_v2.json ] || { echo "ABORT heldout v2"; exit 1; }
for d in "$HOME/projects/_shared/models/Qwen3-8B:base" $(for s in models/kit/tier1/step* models/kit/tier1/ep*; do [ -d "$s" ] && echo "$s:$(basename $s)"; done); do
  ck="${d%%:*}"; tag="${d##*:}"; [ -f "results/kit/v2/probe_$tag.json" ] && continue
  "$PY" -u code/kit/probe_logprob.py --ckpt "$ck" --tag "$tag" --probes data/kit/tier1/probes_v2.json --out results/kit/v2 > logs/probe_v2_$tag.log 2>&1; tail -2 logs/probe_v2_$tag.log | head -1
done
"$PY" code/kit/probe_report.py results/kit/v2
echo "PROBE_V2_DONE"
