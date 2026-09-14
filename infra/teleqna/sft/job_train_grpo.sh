#!/bin/bash
# GRPO on thinking traces. MODEL_DIR is the bare base, and that is now a
# measured choice rather than a fallback:
#   * CPT campaign 1 was negative at every checkpoint and every dose, so there
#     is nothing to merge.
#   * dpo3@200 is +1.4 on dev-1000 but p=0.17 — not distinguishable from base —
#     and preference training sharpens the letter distribution, which shrinks
#     exactly the rollout diversity GRPO feeds on. Starting from base preserves
#     the 23.3% trainable band measured by the dev-1000 pass@8 run.
# That run also replaced the justification for this job. The old one cited an
# oracle of 83.85% that is taken over four CHOICE-ORDER PERMUTATIONS; at fixed
# order, sampling 8 thinking traces gives pass@8 = 84.0% against greedy 75.1%,
# so the headroom is real and is not a permutation artefact.
# Smoke first: 200 prompts, 1 epoch. If reward/mean climbs and completions
# stay well-formed, rerun with the full set.
set -euo pipefail
ROOT=/workspace/teleqna-sft
BASE_ROOT=/workspace/telelogs-base
PY="$BASE_ROOT/venvs/venv-grpo/bin/python"
BASE_MODEL="$BASE_ROOT/shared/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
MODEL_DIR="${MODEL_DIR:-$BASE_MODEL}"
DATA="${DATA:-$ROOT/data/grpo_smoke.jsonl}"
RUN="${RUN:-grpo1smoke}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# venv-grpo has trl 0.23.1 / peft / transformers 4.57.1 but NO vllm; venv has
# vllm 0.11.0 built against the same container torch. train_grpo.py appends
# this to sys.path (never prepends) so only vllm crosses over.
export EXTRA_SITE="$BASE_ROOT/venvs/venv/lib/python3.11/site-packages"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free=${free_mib}MiB  model=$MODEL_DIR  data=$DATA"
[ "$free_mib" -gt 100000 ] || { echo "ABORT: GPU not free"; exit 14; }

echo "#### $RUN GRPO START $(date -Iseconds)"
timeout 43200 "$PY" "$ROOT/infra/train_grpo.py" \
  --dataset "$DATA" \
  --model-path "$MODEL_DIR" \
  --output-dir "$ROOT/models/teleqna-$RUN" \
  --epochs 1 --learning-rate 1e-5 --beta 0.01 \
  --num-generations 8 --generation-batch-size 8 \
  --per-device-batch 8 --grad-accum 4 \
  --max-completion-length 2560 --temperature 1.0 \
  --lora-r 16 --vllm-gpu-mem 0.3 --save-steps 20
echo "#### $RUN GRPO DONE $(date -Iseconds)"

for ckpt in "$ROOT"/models/teleqna-$RUN/checkpoint-* "$ROOT"/models/teleqna-$RUN/final; do
  [ -d "$ckpt" ] || continue
  name=$(basename "$ckpt" | sed 's/checkpoint-/step/')
  out="$ROOT/results/dev1000think_${RUN}_${name}.json"
  [ -e "$out" ] && continue
  echo "#### eval dev-1000 (thinking) @$name"
  timeout 21600 "$BASE_ROOT/venvs/venv-train/bin/python" "$ROOT/infra/eval_dev.py" \
    --base "$MODEL_DIR" --adapter "$ckpt" --thinking --batch 32 \
    --data "$ROOT/data/dev1000.jsonl" --out "$out"
done
echo "#### $RUN ALL DONE $(date -Iseconds)"
