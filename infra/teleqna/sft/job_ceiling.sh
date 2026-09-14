#!/usr/bin/env bash
# Phase 0.1 — measure the ceiling of knowledge injection, before building a corpus.
#
# Two arms of the same base model on the same 1,000 rows, differing only in what
# sits above the question:
#
#   ceilexpl   the row's own gold `explanation` as a reference note
#   ceilshuf   another row's explanation, drawn from the same subject
#
# The second is not padding. "Accuracy rose when I added context" has a boring
# reading — any block of telecom prose changes the decode and may just make the
# model more careful — and the deranged arm is what separates that from
# retrieval actually paying. Read the pair, never `ceilexpl` alone.
#
# Base thinking on this stack is 75.20 (dev1000_base_think.json, same card, same
# vLLM 0.11.0), so no base arm is re-run here.
#
# Card 4, this pod's own device-plugin allocation, sized to what is actually
# free — two killed evals left ~51GB of CUDA context behind that this service
# account cannot reap, and waiting for a clean card would mean waiting forever.
# 8B weights plus a KV cache for 8k of context fit in the remainder.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PYV="$BASE_ROOT/venvs/venv/bin/python"     # vllm 0.11.0 — the stack every arm is on
BASE="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DATA="$ROOT/data"
OUT="$ROOT/results/ceiling"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed -n '1p')
for i in $(seq 1 60); do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '1p')
  [ "$free_mib" -gt 45000 ] && break
  [ "$i" = 1 ] && echo "card has only ${free_mib}MiB free — waiting $(date -Iseconds)"
  sleep 30
done
[ "$free_mib" -gt 45000 ] || { echo "ABORT: under 45GB free on the card"; exit 14; }
GPU_MEM=$(python3 -c "print(round(max(0.25, min(0.85, ($free_mib - 8000) / $TOTAL_MIB)), 2))")
echo "card free=${free_mib}MiB of ${TOTAL_MIB}MiB -> gpu_memory_utilization=$GPU_MEM  $(date -Iseconds)"

for arm in expl explshuf; do
  case "$arm" in
    expl)     tag=ceilexpl ;;
    explshuf) tag=ceilshuf ;;
  esac
  f="$DATA/dev1000_${arm}.jsonl"
  [ -s "$f" ] || { echo "ABORT: missing $f"; exit 15; }
  echo "#### $tag START $(date -Iseconds)"
  timeout 10800 "$PYV" "$ROOT/infra/eval_dev_vllm.py" \
    --base "$BASE" --data "$f" --out-dir "$OUT" --tag "$tag" \
    --with-base --thinking --gpu-mem "$GPU_MEM"
  echo "#### $tag DONE $(date -Iseconds)"
done
echo "#### ALL DONE $(date -Iseconds)"
