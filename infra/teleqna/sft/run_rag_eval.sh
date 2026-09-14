#!/usr/bin/env bash
# Ladder rung 1: the 122B with BM25-retrieved context, scored on all 10,000.
#
# The base arm it is read against is otfull_base_think_merged (80.90 official,
# same nightly venv, same scorer, same official parser), so the difference is
# the context and nothing else. Do not compare it against any number produced
# on vllm 0.11.0.
#
# TP=4 rather than the usual 2, and the reason is the context. Retrieved
# prompts run ~4,500 tokens against the base arm's ~200, and a thinking budget
# on top needs a 24k window. At TP=2 the 250GB of weights leave ~11.5GB a card
# for KV, which run_122b_eval.sh warns is the one knob that turns a comfortable
# fit into an OOM. At TP=4 the weights cost 62.5GB a card and leave ~75GB.
#
# num_key_value_heads is 2, fewer than the 4 ranks. vLLM replicates KV heads in
# that case rather than failing, so the split is legal - KV is duplicated across
# pairs of ranks, which the budget above already absorbs.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-0,1,4,6}"
HOME_ROOT=/home/tensara
S="$HOME_ROOT/projects/telelogs/runs/teleqna-sft"
PY="$HOME_ROOT/venv-vllm-nightly/bin/python"
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B}"
OUT="$S/results/rag"
SET="${SET:-otfull_rag8}"
export PATH="$HOME_ROOT/venv-vllm-nightly/bin:$PATH"   # worker JIT looks up ninja on PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# An EngineCore orphaned by a killed parent keeps every byte it reserved and
# stays invisible to a plain free-memory check until the load is already dead.
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do
  [ "$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')" = "1" ] || continue
  echo "reaping orphaned EngineCore pid=$pid"; kill -9 "$pid" 2>/dev/null; sleep 8
done

IFS=',' read -ra IDX <<< "$CUDA_VISIBLE_DEVICES"
for c in "${IDX[@]}"; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
  echo "card $c free=${free}MiB"
  [ "$free" -gt 100000 ] || { echo "ABORT: card $c only ${free}MiB free"; exit 14; }
done

echo "#### RAG EVAL $SET START $(date -Iseconds)"
"$PY" "$S/infra/eval_dev_vllm.py" \
  --base "$MODEL" --data "$S/data/$SET.jsonl" --out-dir "$OUT" \
  --tag "$SET" --with-base --thinking \
  --max-new 16000 --tp 4 --gpu-mem 0.92 --max-model-len 24576 \
  --engine-kwargs '{"language_model_only": true, "disable_custom_all_reduce": true, "max_num_seqs": 128}'
echo "#### RAG EVAL $SET DONE rc=$? $(date -Iseconds)"
