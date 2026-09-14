#!/usr/bin/env bash
# pass@k on Qwen3.5-122B — how much headroom is left for test-time compute.
#
# Why this and not more training. The 122B already scores 80.90 on the full
# 10,000 closed-book, +6.93 over the 8B we have been tuning all day. The
# question is what sits above it *without* retrieval, and the 8B's own profile
# says where to look:
#
#     greedy   75.1
#     vote@8   76.6      majority voting buys +1.5 and no more
#     pass@8   84.0      the gold letter IS produced, in 84% of rows
#                        ────
#                        7.4 points of "picked the wrong branch", not of
#                        missing knowledge
#
# That gap is a discrimination problem, and the distractor probe put +15.83pp
# there. Nobody has measured whether the same gap exists on the 122B. If its
# pass@8 is ~90 then a verifier is the cheapest 8 points left on this project;
# if it is ~83 then test-time compute is exhausted too and the honest move is
# to ship 80.90 and stop.
#
# 500 rows, not 1,000, and k=8 rather than k=4. Those cost the same and this
# way the metric is directly comparable with the stored 8B profile (restrict
# that one to the same sample_ids). Wilson CI on pass@8 at n=500 is about
# +/-3pp, which is far tighter than the 6-point decision boundary above.
#
# max_tokens 6000, not the profiler's 1024 default. A truncated rollout scores
# zero and is indistinguishable from a wrong answer, so a cheap budget would
# report a pass@k ceiling that is really a token ceiling — the 122B's otfull
# run needed a 243-row retry pass for exactly this reason, worth +1.11 points.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-1,5}"
HOME_ROOT=/home/tensara
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B}"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
ROOT="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
OUT="$ROOT/results/passk122b"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# max_num_seqs 256, not the default: the hybrid attention stack sizes its Mamba
# cache from this and a larger value fails at startup with
# "max_num_seqs exceeds available Mamba cache blocks".
ENGINE_KWARGS="${ENGINE_KWARGS:-{\"language_model_only\": true, \"disable_custom_all_reduce\": true, \"max_num_seqs\": 256\}}"

# Both cards, checked before the load. TP=2 dies on whichever card is short and
# surfaces it as a confusing NCCL failure rather than as an OOM.
IFS=',' read -ra IDX <<< "$CUDA_VISIBLE_DEVICES"
for c in "${IDX[@]}"; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
  echo "card $c free=${free}MiB"
  [ "$free" -gt 135000 ] || { echo "ABORT: card $c has only ${free}MiB free"; exit 14; }
done

echo "#### passk 122B START $(date -Iseconds)"
timeout 36000 "$PY" "$ROOT/infra/profile_pass_rate.py" \
  --model "$MODEL" --data "$ROOT/data/dev1000_passk.jsonl" \
  --out "$OUT/passk_dev500_122b_k8.jsonl" \
  -k 8 --limit "${LIMIT:-500}" --max-tokens 6000 --max-model-len 8192 \
  --temperature 1.0 --gpu-mem 0.95 --tp 2 --engine-kwargs "$ENGINE_KWARGS"
echo "#### passk 122B DONE $(date -Iseconds)"
