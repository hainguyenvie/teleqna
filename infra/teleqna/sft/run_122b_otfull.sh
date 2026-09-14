#!/usr/bin/env bash
# Score the 122B on the full ot-full teleqna 10,000, on its own.
#
# Split out of run_122b_eval.sh rather than reordering it, because that script
# was mid-flight on otlite when ot-full was asked for and bash reads a running
# script incrementally from a byte offset — editing it in place would have made
# the interpreter resume at a shifted position in changed text. A separate file
# costs nothing: every arm reloads the engine anyway, so chaining inside one
# script was never saving a model load.
#
# dev-1000 is deliberately absent. It is the split with twelve arms already
# scored, so it stays worth running, but it is a comparability measurement
# rather than a headline and it can follow later. Run run_122b_eval.sh with
# SKIP_OTLITE/SKIP_OTFULL, or just the scorer directly, when that is wanted.
#
# All three engine settings carry over, each one bought by a crash:
#   language_model_only       skip the vision tower this text benchmark never calls
#   disable_custom_all_reduce custom all-reduce shares buffers by CUDA IPC, which
#                             this container refuses despite NV18 and P2P OK
#   max_num_seqs 256          linear-attention layers hold one unpageable state
#                             block per in-flight sequence; only ~466 fit
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CARDS:-1,5}"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
ENGINE_KWARGS="${ENGINE_KWARGS:-{\"language_model_only\": true, \"disable_custom_all_reduce\": true, \"max_num_seqs\": 256\}}"
HOME_ROOT=/home/tensara
MODEL="${MODEL:-$HOME_ROOT/projects/telelogs/shared/models/Qwen3.5-122B-A10B}"
PY="${PY:-$HOME_ROOT/venv-vllm-nightly/bin/python}"
export PATH="$(dirname "$PY"):$PATH"   # worker JIT looks ninja up on PATH
INFRA=$HOME_ROOT/projects/telelogs/runs/teleqna-sft/infra
DATA=$HOME_ROOT/projects/telelogs/runs/teleqna-sft/data
OUT="${OUT:-$HOME_ROOT/projects/telelogs/runs/teleqna-sft/results/vllm122b}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

# Wait for the cards rather than aborting on them. This launches immediately
# after the otlite engine is torn down, and a teardown that has not finished
# releasing 250GB looks identical to a card someone else took.
IFS=',' read -ra IDX <<< "$CUDA_VISIBLE_DEVICES"
for c in "${IDX[@]}"; do
  for i in $(seq 1 60); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$c")
    [ "$free" -gt 135000 ] && break
    [ "$i" = 1 ] && echo "card $c busy (${free}MiB) — waiting $(date -Iseconds)"
    sleep 10
  done
  echo "card $c free=${free}MiB"
  [ "$free" -gt 135000 ] || { echo "ABORT: card $c stuck at ${free}MiB"; exit 14; }
done

echo "#### ot-full 10000 START $(date -Iseconds)"
"$PY" "$INFRA/eval_dev_vllm.py" \
  --base "$MODEL" --data "$DATA/otfull10000.jsonl" --out-dir "$OUT" \
  --tag otfull --with-base --thinking --tp 2 --gpu-mem 0.95 \
  --max-model-len 8192 ${ENGINE_KWARGS:+--engine-kwargs "$ENGINE_KWARGS"}
echo "#### ot-full 10000 DONE $(date -Iseconds)"
