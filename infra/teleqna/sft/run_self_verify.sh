#!/usr/bin/env bash
# Capture the 122B's branch-selection headroom without training anything.
#
# This one IS submittable in principle: it uses only the question and the
# model's own samples. No answer key, no test-derived training data. If it
# works it is a serving-side change, and the leaderboard row still names a
# model that reproduces.
#
# Scope is 94 rows of 500 — the ones where the 122B's eight samples disagreed.
# The other 406 are untouched (all-agree, 86.9% correct), so the arms differ
# only where there is something to win.
#
# Cards 1+5, TP=2, same three engine flags the 122B needs.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-1,5}"
HOME_ROOT=/home/tensara
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B}"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
ROOT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
OUT="$ROOT/results/selfverify"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
ENGINE_KWARGS="${ENGINE_KWARGS:-{\"language_model_only\": true, \"disable_custom_all_reduce\": true, \"max_num_seqs\": 256\}}"

IFS=',' read -ra IDX <<< "$CUDA_VISIBLE_DEVICES"
for c in "${IDX[@]}"; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
  echo "card $c free=${free}MiB"
  [ "$free" -gt 135000 ] || { echo "ABORT: card $c has only ${free}MiB free"; exit 14; }
done

echo "#### self-verify START $(date -Iseconds)"
timeout 14400 "$PY" "$ROOT/infra/self_verify.py" \
  --passk "$ROOT/results/passk122b/passk_dev500_122b_k8.jsonl" \
  --split "$ROOT/data/dev1000.jsonl" \
  --model "$MODEL" --out "$OUT/selfverify_122b.json" \
  --arms greedy,adjudicate,verify \
  --max-tokens 6000 --max-model-len 8192 --gpu-mem 0.95 --tp 2 \
  --engine-kwargs "$ENGINE_KWARGS"
echo "#### self-verify DONE $(date -Iseconds)"
