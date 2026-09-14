#!/usr/bin/env bash
# Measure the sampled-trace ceiling on the real dev-1000, before spending a day
# on GRPO. The staged GRPO plan is justified by an "oracle 83.85 vs vote 74.71"
# gap, but that oracle is over four CHOICE-ORDER PERMUTATIONS, not over sampled
# reasoning traces at fixed order. This job measures the one that GRPO actually
# converts: pass@1 -> pass@8 on the benchmark's own rows, thinking on, T=1.
#
# max-tokens is 2560, not the 1024 the profile job uses: the thinking arm
# averages 968 tokens per case at greedy and sampling at T=1 runs longer, so a
# 1024 cap would truncate roughly half the rollouts and every truncation scores
# zero. That would measure the token budget, not the ceiling. 2560 + a ~600-token
# prompt still fits the profiler's 4096 max_model_len.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PYV="$BASE_ROOT/venvs/venv/bin/python"     # vllm 0.11.0
BASE_MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_DIR="${MODEL_DIR:-$BASE_MODEL}"
K="${K:-8}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free=${free_mib}MiB  model=$MODEL_DIR  k=$K"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: GPU not free"; exit 14; }

"$PYV" "$ROOT/infra/make_passk_set.py" \
  --data "$ROOT/data/dev1000.jsonl" \
  --out "$ROOT/data/dev1000_passk.jsonl"

echo "#### passk START $(date -Iseconds)"
timeout 21600 "$PYV" "$ROOT/infra/profile_pass_rate.py" \
  --model "$MODEL_DIR" \
  --data "$ROOT/data/dev1000_passk.jsonl" \
  --out "$ROOT/results/passk_dev1000_base_k${K}.jsonl" \
  -k "$K" --max-tokens 2560 --temperature 1.0
echo "#### passk DONE $(date -Iseconds)"

"$PYV" "$ROOT/infra/analyze_passk.py" \
  --profile "$ROOT/results/passk_dev1000_base_k${K}.jsonl" \
  --out "$ROOT/results/passk_dev1000_base_k${K}.summary.json" \
  --greedy 0.751
echo "#### analyze DONE $(date -Iseconds)"
